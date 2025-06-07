#!/usr/bin/env python3
import torch
from transformers import BertTokenizer
import os

def main():
    os.makedirs("exported", exist_ok=True)
    # 1) load your tokenizer
    tok = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
    # 2) read your lines
    with open("real_texts.txt") as f:
        lines = [l.strip() for l in f if l.strip()]
    # 3) tokenize all at once (batch)
    enc = tok(lines, padding=True, truncation=True, max_length=128, return_tensors="pt")
    # 4) save two tensors
    torch.save(enc["input_ids"],      "exported/input_ids.pt")
    torch.save(enc["attention_mask"], "exported/attention_mask.pt")
    print(f"✅ Saved {len(lines)} lines → exported/input_ids.pt & attention_mask.pt")

if __name__=="__main__":
    main()
