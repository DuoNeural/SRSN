import os
import sys
import time
import argparse
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

def build_and_train_tokenizer(texts_iter, vocab_size=16384, save_path="tokenizer_16k.json"):
    print(f"[Tokenizer] Training Byte-Level BPE tokenizer (vocab_size={vocab_size})...", flush=True)
    t0 = time.time()
    
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    tokenizer.train_from_iterator(texts_iter, trainer=trainer)
    
    # Post-processor to handle bos/eos if needed
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    tokenizer.save(save_path)
    print(f"[Tokenizer] Training complete in {time.time() - t0:.2f}s! Saved to {save_path}", flush=True)
    return tokenizer

def prepare_corpus(args):
    os.makedirs(args.out_dir, exist_ok=True)
    tok_path = os.path.join(args.out_dir, "tokenizer_16k.json")
    train_bin_path = os.path.join(args.out_dir, "train.bin")
    val_bin_path = os.path.join(args.out_dir, "val.bin")

    from datasets import load_dataset

    print(f"[Data] Loading streaming dataset '{args.dataset}'...", flush=True)
    ds = load_dataset(args.dataset, split="train", streaming=True)

    # 1. Train or load tokenizer
    if not os.path.exists(tok_path) or args.retrain_tokenizer:
        print("[Tokenizer] Gathering 25,000 text samples for tokenizer training...", flush=True)
        train_texts = []
        for i, item in enumerate(ds):
            if i >= 25000:
                break
            txt = item.get("text") or item.get("content") or ""
            if len(txt.strip()) > 20:
                train_texts.append(txt)
        tokenizer = build_and_train_tokenizer(train_texts, vocab_size=args.vocab_size, save_path=tok_path)
    else:
        print(f"[Tokenizer] Loading existing tokenizer from {tok_path}...", flush=True)
        tokenizer = Tokenizer.from_file(tok_path)

    eos_id = tokenizer.token_to_id("</s>") or 3
    print(f"[Tokenizer] Vocab size: {tokenizer.get_vocab_size()} | EOS ID: {eos_id}", flush=True)

    # 2. Tokenize and write uint16 binary files
    print(f"[Data] Streaming and tokenizing {args.num_train_tokens:,} train tokens and {args.num_val_tokens:,} val tokens...", flush=True)
    
    # Reload stream
    ds_iter = iter(load_dataset(args.dataset, split="train", streaming=True))
    
    # Validation tokens
    val_tokens = []
    while len(val_tokens) < args.num_val_tokens:
        try:
            item = next(ds_iter)
        except StopIteration:
            break
        txt = item.get("text") or item.get("content") or ""
        if len(txt.strip()) < 10:
            continue
        encoded = tokenizer.encode(txt)
        val_tokens.extend(encoded.ids)
        val_tokens.append(eos_id)
    
    val_arr = np.array(val_tokens[:args.num_val_tokens], dtype=np.uint16)
    with open(val_bin_path, "wb") as f:
        f.write(val_arr.tobytes())
    print(f"[Data] Wrote {len(val_arr):,} tokens to {val_bin_path} ({os.path.getsize(val_bin_path) / 1024:.1f} KB)", flush=True)

    # Train tokens
    train_tokens = []
    chunk_count = 0
    with open(train_bin_path, "wb") as f:
        while chunk_count < args.num_train_tokens:
            try:
                item = next(ds_iter)
            except StopIteration:
                break
            txt = item.get("text") or item.get("content") or ""
            if len(txt.strip()) < 10:
                continue
            encoded = tokenizer.encode(txt)
            train_tokens.extend(encoded.ids)
            train_tokens.append(eos_id)

            if len(train_tokens) >= 500000:
                chunk = np.array(train_tokens[:500000], dtype=np.uint16)
                f.write(chunk.tobytes())
                chunk_count += len(chunk)
                train_tokens = train_tokens[500000:]
                print(f"  Tokenized {chunk_count:,} / {args.num_train_tokens:,} tokens...", flush=True)

        if train_tokens and chunk_count < args.num_train_tokens:
            needed = args.num_train_tokens - chunk_count
            chunk = np.array(train_tokens[:needed], dtype=np.uint16)
            f.write(chunk.tobytes())
            chunk_count += len(chunk)

    print(f"[Data] Wrote {chunk_count:,} tokens to {train_bin_path} ({os.path.getsize(train_bin_path) / (1024 * 1024):.2f} MB)", flush=True)
    print("[Data] Dataset preparation complete! Zero-copy memmap ready.", flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--num-train-tokens", type=int, default=5000000)
    p.add_argument("--num-val-tokens", type=int, default=200000)
    p.add_argument("--out-dir", default="/home/duoneural/srsn_ryzen")
    p.add_argument("--retrain-tokenizer", action="store_true")
    args = p.parse_args()
    prepare_corpus(args)
