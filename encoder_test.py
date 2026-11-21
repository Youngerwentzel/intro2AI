import os
from pathlib import Path

import torch
import timm
from PIL import Image
from torchvision import transforms
from timm.data import resolve_data_config

# =====================================================================
# ⚠️ IMPORTANT: Change this path if your dataset is somewhere else.
# Based on your setup, it's likely on your Desktop.
# =====================================================================
DATA_ROOT = Path("/Users/youngerwentzel/Desktop/AIAA2205Project")

print("Looking for data in:", DATA_ROOT)

# Check if the path is correct before running
if not (DATA_ROOT / "Validation").exists():
    print(f"\n❌ ERROR: Could not find the 'Validation' folder inside {DATA_ROOT}")
    print("Please check that the AIAA2205Project folder is exactly there, or change the path on line 11.")
    exit()

# ---------- config ----------
ENCODER_NAME = "tf_efficientnet_b0.in1k"
N_IMAGES = 300

# Mac M-series chips use "mps" for GPU acceleration. 
# If you have an Intel Mac, it will fall back to "cpu".
if torch.backends.mps.is_available():
    DEVICE = "mps" 
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# ---------- load encoder ----------
print(f"Downloading/Loading {ENCODER_NAME}...")
model = timm.create_model(ENCODER_NAME, pretrained=True, features_only=True).to(DEVICE).eval()

cfg = resolve_data_config({}, model=model)
tf = transforms.Compose([
    transforms.Resize(cfg["input_size"][1:]),
    transforms.CenterCrop(cfg["input_size"][1:]),
    transforms.ToTensor(),
    transforms.Normalize(cfg["mean"], cfg["std"]),
])

# ---------- smoke test ----------
paths = sorted((DATA_ROOT / "Validation").glob("*.jpg"))[:N_IMAGES]
print(f"\nFound {len(paths)} validation images")

img = Image.open(paths[0]).convert("RGB")
x = tf(img).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    feats = model(x)
print("\n[smoke] feature map shapes:")
for f in feats:
    print("  ", tuple(f.shape), f.dtype, f.device)

# ---------- size check ----------
sd = model.state_dict()
n_bytes = sum(v.numel() * v.element_size() for v in sd.values())
print(f"\n[size] {sum(v.numel() for v in sd.values())/1e6:.1f}M params, {n_bytes/1e6:.1f} MB (fp32)")

# ---------- embed all ----------
@torch.no_grad()
def embed(paths, batch=32):
    out = []
    for i in range(0, len(paths), batch):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i+batch]]).to(DEVICE)
        out.append(model(x)[-1].mean(dim=(2, 3)).cpu())
    return torch.cat(out)

F = embed(paths)
print(f"\n[embed] features: {tuple(F.shape)}")

# ---------- sanity ----------
Fn = torch.nn.functional.normalize(F, dim=1)
print(f"[sanity] per-channel std (mean over channels): {Fn.std(dim=0).mean().item():.4f}")
print(f"[sanity] mean random-pair cosine sim:          {(Fn[:20] @ Fn[20:40].T).mean().item():.4f}")

# ---------- retrieval ----------
sims = Fn @ Fn[0]
top = sims.topk(6).indices.tolist()[1:]
print("\n[retrieval] query:", paths[0].name)
for i in top:
    print(f"  {sims[i]:.3f}  {paths[i].name}")
