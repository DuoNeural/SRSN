#!/usr/bin/env python3
"""
DuoNeural Golden Eutectic Corpus Builder (SRSN Ryzen 5 3550H / Node-0)
Blends:
  - 65% TinyStories (roneneldan/TinyStories)
  - 20% FineWeb-Edu (HuggingFaceFW/fineweb-edu sample-10BT)
  - 15% DuoNeural CoT Reasoning (DuoNeural/cot-reasoning-2k)

Outputs:
  - Target: /dev/shm/train_golden.bin (and persistent backup to out_dir/train_golden.bin)
  - Target: /dev/shm/val_golden.bin (and persistent backup to out_dir/val_golden.bin)
"""

import os
import sys
import time
import shutil
import random
import argparse
import numpy as np
from tokenizers import Tokenizer
from datasets import load_dataset

def format_cot_item(item):
    messages = item.get("messages", [])
    parts = []
    for m in messages:
        role = m.get("role", "system").capitalize()
        content = m.get("content", "").strip()
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)

def build_golden_corpus(args):
    print("=" * 78)
    print("[Corpus] Building DuoNeural Golden Eutectic Corpus (10M Tokens)")
    print(f"         Weights: 65% TinyStories | 20% FineWeb-Edu | 15% DuoNeural CoT")
    print(f"         Tokenizer: {args.tokenizer_path}")
    print(f"         RAM Disk Destination: {args.ram_dir}")
    print(f"         Persistent Backup:     {args.out_dir}")
    print("=" * 78)

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.ram_dir, exist_ok=True)

    if not os.path.exists(args.tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {args.tokenizer_path}")
    
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    eos_id = tokenizer.token_to_id("</s>") or 3
    print(f"[Tokenizer] Vocab size: {tokenizer.get_vocab_size()} | EOS ID: {eos_id}")

    # 1. Stream Iterators
    print("[Data] Initializing data streams...")
    ds_tiny = iter(load_dataset("roneneldan/TinyStories", split="train", streaming=True))
    ds_edu = iter(load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True))
    
    print("[Data] Loading DuoNeural CoT in memory...")
    ds_cot = load_dataset("DuoNeural/cot-reasoning-2k", split="train")
    cot_texts = [format_cot_item(ex) for ex in ds_cot if ex.get("messages")]
    print(f"[Data] Loaded {len(cot_texts):,} formatted CoT traces.")

    cot_idx = 0
    random.seed(42)
    random.shuffle(cot_texts)

    def get_next_sample(domain_idx):
        nonlocal cot_idx, ds_tiny, ds_edu
        if domain_idx == 0:  # TinyStories (65%)
            try:
                item = next(ds_tiny)
                return item.get("text", "")
            except StopIteration:
                ds_tiny = iter(load_dataset("roneneldan/TinyStories", split="train", streaming=True))
                return next(ds_tiny).get("text", "")
        elif domain_idx == 1:  # FineWeb-Edu (20%)
            try:
                item = next(ds_edu)
                return item.get("text", "")
            except StopIteration:
                ds_edu = iter(load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True))
                return next(ds_edu).get("text", "")
        else:  # DuoNeural CoT (15%)
            txt = cot_texts[cot_idx % len(cot_texts)]
            cot_idx += 1
            return txt

    # 2. Build Validation Set
    print(f"\n[Val] Generating {args.num_val_tokens:,} validation tokens...")
    val_tokens = []
    domain_weights = [0.65, 0.20, 0.15]

    while len(val_tokens) < args.num_val_tokens:
        dom = random.choices([0, 1, 2], weights=domain_weights)[0]
        text = get_next_sample(dom).strip()
        if len(text) < 15:
            continue
        encoded = tokenizer.encode(text)
        val_tokens.extend(encoded.ids)
        val_tokens.append(eos_id)

    val_arr = np.array(val_tokens[:args.num_val_tokens], dtype=np.uint16)
    
    val_ram_path = os.path.join(args.ram_dir, "val_golden.bin")
    val_disk_path = os.path.join(args.out_dir, "val_golden.bin")
    
    with open(val_ram_path, "wb") as f:
        f.write(val_arr.tobytes())
    shutil.copyfile(val_ram_path, val_disk_path)
    print(f"[Val] Wrote {len(val_arr):,} tokens to {val_ram_path} and mirrored to {val_disk_path}")

    # 3. Build Training Set (10M tokens)
    print(f"\n[Train] Streaming and tokenizing {args.num_train_tokens:,} tokens directly to RAM disk...")
    train_ram_path = os.path.join(args.ram_dir, "train_golden.bin")
    train_disk_path = os.path.join(args.out_dir, "train_golden.bin")

    train_tokens = []
    total_written = 0
    t_start = time.time()

    with open(train_ram_path, "wb") as f_ram:
        while total_written < args.num_train_tokens:
            dom = random.choices([0, 1, 2], weights=domain_weights)[0]
            text = get_next_sample(dom).strip()
            if len(text) < 15:
                continue
            encoded = tokenizer.encode(text)
            train_tokens.extend(encoded.ids)
            train_tokens.append(eos_id)

            if len(train_tokens) >= 500000:
                chunk = np.array(train_tokens[:500000], dtype=np.uint16)
                f_ram.write(chunk.tobytes())
                total_written += len(chunk)
                train_tokens = train_tokens[500000:]
                elapsed = time.time() - t_start
                rate = total_written / elapsed
                pct = (total_written / args.num_train_tokens) * 100
                print(f"  [{pct:5.1f}%] Written {total_written:,} / {args.num_train_tokens:,} tokens ({rate:,.0f} tok/s)...", flush=True)

        if train_tokens and total_written < args.num_train_tokens:
            needed = args.num_train_tokens - total_written
            chunk = np.array(train_tokens[:needed], dtype=np.uint16)
            f_ram.write(chunk.tobytes())
            total_written += len(chunk)

    print(f"\n[Train] Finished writing {total_written:,} tokens to {train_ram_path} ({os.path.getsize(train_ram_path) / (1024*1024):.2f} MB)")
    print("[Disk] Mirroring /dev/shm dataset to persistent storage...")
    shutil.copyfile(train_ram_path, train_disk_path)
    print(f"[Disk] Mirrored to {train_disk_path} in {time.time() - t0:.2f}s total.")
    print("=" * 78)
    print("DuoNeural Golden Eutectic Corpus is ready for zero-latency pretraining!")
    print("=" * 78)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer-path", default="/home/duoneural/srsn_ryzen/tokenizer_16k.json")
    p.add_argument("--num-train-tokens", type=int, default=10000000)
    p.add_argument("--num-val-tokens", type=int, default=200000)
    p.add_argument("--ram-dir", default="/dev/shm")
    p.add_argument("--out-dir", default="/home/duoneural/srsn_ryzen")
    args = p.parse_args()
    build_golden_corpus(args)
