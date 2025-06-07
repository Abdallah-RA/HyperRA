# pretokenize_and_pack.py
import torch
from transformers import BertTokenizer
from torch import nn

class TextData(nn.Module):
    def __init__(self):
        super().__init__()
        tok = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
        lines = [l.strip() for l in open("real_texts.txt") if l.strip()]
        enc = tok(lines, padding=True, truncation=True, max_length=128, return_tensors="pt")
        self.register_buffer("ids",      enc["input_ids"])
        self.register_buffer("mask",     enc["attention_mask"])

    def forward(self):
        return self.ids, self.mask

if __name__=="__main__":
    m = TextData().eval()
    sm = torch.jit.script(m)
    sm.save("exported/text_data.pt")
    print("Saved text_data.pt")
