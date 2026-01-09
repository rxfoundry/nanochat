"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import requests
import pandas as pd
# import pyarrow.fs as fs
# import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

import logging
logging.getLogger("httpx").setLevel(logging.ERROR)

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URI on the internet where the data is hosted and downloaded from on demand
BASE_URI = "hf://datasets/karpathy/fineweb-edu-100b-shuffle"
MAX_SHARD = 1822 # the last datashard is shard_01822.parquet
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(data_dir=None):
    """ Looks into a data dir and returns full paths to all parquet files. """
    data_dir = DATA_DIR if data_dir is None else data_dir
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        try:
            # Read entire parquet file with pandas
            pf = pd.read_parquet(filepath, engine='fastparquet')

            # Calculate chunk size to approximate row_groups behavior
            chunk_size = 1024  # approximate row_group size
            total_rows = len(df)

            # Iterate through chunks with DDP-style distribution
            for chunk_start in range(start * chunk_size, total_rows, step * chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_rows)
                if chunk_start >= total_rows:
                    break

                chunk_df = df.iloc[chunk_start:chunk_end]
                texts = chunk_df['text'].tolist()
                yield texts
        except Exception as e:
            print(f"Error reading {filepath}: {e}")


# -----------------------------------------------------------------------------
def download_single_file(index):
    """ Downloads a single file index, with some backoff """

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    uri = f"{BASE_URI}/{filename}"
    print(f"Downloading {filename}...")

    try:
        # Use requests with retries instead of pyarrow.fs
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(uri, timeout=300)
                response.raise_for_status()

                with open(filepath, 'wb') as f:
                    f.write(response.content)

                print(f"Successfully downloaded {filename}")
                return True
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Download attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"Failed to download {filename} after {max_retries} attempts: {e}")
                    return False
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu 100BT dataset shards")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards to download (default: -1), -1 = disable")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    args = parser.parse_args()

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers...")
    print(f"Target directory: {DATA_DIR}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {DATA_DIR}")
