import torch
from torchvision import models as vision_models
from transformers import AutoModel, AutoTokenizer

def load_vision(names, device):
    out = {}
    for n in names:
        print(f"[Vision] {n}")
        fn = getattr(vision_models, n)
        m = fn(pretrained=True, progress=False).to(device).eval()
        out[n] = m
    return out

def load_nlp(names, device):
    ms, ts = {}, {}
    for n in names:
        print(f"[NLP] {n}")
        tok = AutoTokenizer.from_pretrained(n)
        m = AutoModel.from_pretrained(n, torch_dtype=torch.float16).to(device).eval()
        ts[n], ms[n] = tok, m
    return ms, ts

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("→ Using device:", device)

    # Ten small/​tiny vision nets
    vision_list = [
        "mobilenet_v2",
        "mobilenet_v3_small",
        "efficientnet_b0",
        "shufflenet_v2_x1_0",
        "squeezenet1_0",
        "resnet18",
        "densenet121",
        "regnet_y_400mf",
        "vgg11",
        "alexnet"
    ]

    # Six small NLP models
    nlp_list = [
        "prajjwal1/bert-tiny",
        "sshleifer/tiny-gpt2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "distilbert-base-uncased",
        "google/mobilebert-uncased",
        "facebook/opt-125m"
    ]

    vision_models = load_vision(vision_list, device)
    nlp_models, nlp_tok   = load_nlp(nlp_list, device)

    print("\n✅ Loaded vision models:", list(vision_models.keys()))
    print("✅ Loaded NLP   models:", list(nlp_models.keys()))

    # Dummy inference
    img = torch.randn(1, 3, 224, 224).to(device)
    txt = "Hello world"

    with torch.no_grad():
        for name, m in vision_models.items():
            out = m(img)
            print(f"{name:20s} → vision output shape: {tuple(out.shape)}")
        for name, m in nlp_models.items():
            inp = nlp_tok[name](txt, return_tensors="pt").to(device)
            h   = m(**inp).last_hidden_state
            print(f"{name:20s} → NLP   output shape: {tuple(h.shape)}")

if __name__ == "__main__":
    main()
