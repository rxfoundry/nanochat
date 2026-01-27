"""
Train a tokenizer using our own BPE Tokenizer library.
In the style of GPT-4 tokenizer.
"""
import os
import time
import argparse
import torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset_rxf import parquets_iter_batched

# -----------------------------------------------------------------------------
# Parse command line arguments

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=10_000_000_000, help='Maximum characters to train on (default: 10B)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768, help='Vocabulary size (default: 32768 = 2^15)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Text iterator

def text_iterator():
    """
    1) Flatten the batches into a single iterator
    2) Crop every document to args.doc_cap characters
    3) Break when we've seen args.max_chars characters
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            # Ensure valid UTF-8 and handle encoding errors
            if isinstance(doc, bytes):
                doc_text = doc.decode('utf-8', errors='ignore')
            else:
                doc_text = str(doc)

            # Additional validation: ensure the string is properly formed
            try:
                # This will raise an exception if there are encoding issues
                doc_text.encode('utf-8').decode('utf-8')
            except (UnicodeDecodeError, UnicodeEncodeError):
                print(f"Skipping malformed document: {doc_text[:100]}...")
                continue

            if len(doc_text) > args.doc_cap:
                # Be extra careful when slicing - ensure we don't break UTF-8
                doc_text = doc_text[:args.doc_cap]
                # Verify the slice didn't break UTF-8 encoding
                try:
                    doc_text.encode('utf-8')
                except UnicodeEncodeError:
                    # If slicing broke UTF-8, find a safe boundary
                    safe_cap = args.doc_cap
                    while safe_cap > 0:
                        try:
                            doc_text[:safe_cap].encode('utf-8')
                            doc_text = doc_text[:safe_cap]
                            break
                        except UnicodeEncodeError:
                            safe_cap -= 1
            if len(doc_text) > 0:
                nchars += len(doc_text)
                yield doc_text
                if nchars > args.max_chars:
                    return

text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# The bits per byte on the validation set is then one of the primary metrics we care about.
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # the Python string representation of this token
    if token_str in special_set:
        token_bytes.append(0) # special characters are not counted
    else:
        id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# Log to report
# from nanochat.report import get_report
# token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
# get_report().log(section="Tokenizer training", data=[
#     vars(args), # argparse command line arguments
#     {"train_time": train_time},
#     {"num_special_tokens": len(special_set)},
#     {
#         "token_bytes_min": int(token_bytes_nonzero.min().item()),
#         "token_bytes_max": int(token_bytes_nonzero.max().item()),
#         "token_bytes_mean": token_bytes_nonzero.mean().item(),
#         "token_bytes_std": token_bytes_nonzero.std().item(),
#     }
# ])
