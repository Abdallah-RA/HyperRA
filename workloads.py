#!/usr/bin/env python3
import os
import requests
from torchvision.datasets import CIFAR10
from torchvision import transforms
from PIL import Image
from datasets import load_dataset

def prepare_images(out_dir="real_images", num=500):
    """
    Download CIFAR-10 test images (32×32) and upsample to 224×224.
    """
    os.makedirs(out_dir, exist_ok=True)
    # Download CIFAR10 test split
    ds = CIFAR10(root=".", train=False, download=True)
    transform = transforms.Resize((224,224))
    for idx, (img, label) in enumerate(ds):
        if idx >= num: break
        img = transform(img)
        path = os.path.join(out_dir, f"img_{idx:04d}.png")
        img.save(path)
    print(f"Saved {min(num,len(ds))} images to {out_dir}/")

def prepare_texts(out_file="real_texts.txt", num=500):
    """
    Download WikiText-2 test sentences (raw) and write one per line.
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    count = 0
    with open(out_file, "w") as f:
        for line in ds["text"]:
            line = line.strip()
            if not line: continue
            f.write(line.replace("\n"," ") + "\n")
            count += 1
            if count >= num:
                break
    print(f"Saved {count} text lines to {out_file}")

if __name__ == "__main__":
    # You may need to `pip install torchvision datasets pillow`
    prepare_images(out_dir="real_images", num=500)
    prepare_texts(out_file="real_texts.txt", num=500)
    print("Done!")
