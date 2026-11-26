import re
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import timm
from timm.data import resolve_data_config
from torchvision import transforms

# =====================================================================
# 1. CONFIG
# =====================================================================
DATA_ROOT = Path("/Users/youngerwentzel/Desktop/AIAA2205Project")
ENCODER_NAME = "tf_efficientnet_b0.in1k"

# Prefer Apple Metal (MPS) on Apple Silicon, then CUDA, then CPU.
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print("Using device:", DEVICE)

# =====================================================================
# 2. VOCABULARY (built strictly from Training_Labels.csv)
# =====================================================================
def clean_and_tokenize(text: str) -> list[str]:
    text = str(text).lower().strip()
    return re.findall(r"[a-z0-9']+|[.,!?;:]", text)


class CaptionVocabulary:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.special_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        self.word2idx = {}
        self.idx2word = {}

    def build_vocab(self, descriptions):
        counter = Counter()
        for desc in descriptions:
            counter.update(clean_and_tokenize(desc))
        vocab_list = list(self.special_tokens) + [
            w for w, c in counter.items()
            if c >= self.min_freq and w not in self.special_tokens
        ]
        self.word2idx = {w: i for i, w in enumerate(vocab_list)}
        self.idx2word = {i: w for i, w in enumerate(vocab_list)}

    def encode(self, text: str) -> list[int]:
        bos = self.word2idx["<BOS>"]
        eos = self.word2idx["<EOS>"]
        unk = self.word2idx["<UNK>"]
        return [bos] + [self.word2idx.get(t, unk) for t in clean_and_tokenize(text)] + [eos]

    def decode(self, token_ids: list[int]) -> str:
        pad = self.word2idx["<PAD>"]
        bos = self.word2idx["<BOS>"]
        eos = self.word2idx["<EOS>"]
        words = []
        for idx in token_ids:
            if idx == eos:
                break
            if idx not in (pad, bos) and idx in self.idx2word:
                words.append(self.idx2word[idx])
        return re.sub(r"\s+([.,!?;:])", r"\1", " ".join(words)).strip()


def collate_captions(encoded_list: list[list[int]], pad_idx: int = 0) -> torch.Tensor:
    max_len = max(len(s) for s in encoded_list)
    return torch.tensor(
        [s + [pad_idx] * (max_len - len(s)) for s in encoded_list],
        dtype=torch.long,
    )


# =====================================================================
# 3. DECODER
# =====================================================================
class ImageCaptionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder_dim: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 40,
        max_img_tokens: int = 196,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_seq_len = max_seq_len

        self.img_proj = nn.Sequential(
            nn.Linear(encoder_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )
        self.img_pos_emb = nn.Embedding(max_img_tokens, d_model)
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_drop = nn.Dropout(dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def _prepare_memory(self, img_feats: torch.Tensor) -> torch.Tensor:
        if img_feats.dim() == 2:
            img_feats = img_feats.unsqueeze(1)  # (B, D) -> (B, 1, D)
        B, S, _ = img_feats.shape
        img_pos = torch.arange(S, device=img_feats.device).unsqueeze(0).expand(B, S)
        return self.img_proj(img_feats) + self.img_pos_emb(img_pos)

    def forward(self, img_feats: torch.Tensor, decoder_in: torch.Tensor) -> torch.Tensor:
        memory = self._prepare_memory(img_feats)
        B, T = decoder_in.shape
        tok_pos = torch.arange(T, device=decoder_in.device).unsqueeze(0).expand(B, T)
        tgt = self.emb_drop(self.tok_emb(decoder_in) + self.pos_emb(tok_pos))

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=decoder_in.device
        )
        pad_mask = torch.zeros(
            (B, T), dtype=tgt.dtype, device=decoder_in.device
        ).masked_fill(decoder_in == self.pad_idx, float("-inf"))

        out = self.decoder(
            tgt=tgt, memory=memory, tgt_mask=causal_mask, tgt_key_padding_mask=pad_mask
        )
        return self.fc_out(out)

    @torch.no_grad()
    def generate_greedy(
        self,
        img_feats: torch.Tensor,
        bos_idx: int = 1,
        eos_idx: int = 2,
        max_new_tokens: int = 25,
    ):
        self.eval()
        memory = self._prepare_memory(img_feats)
        B = memory.shape[0]
        seq = torch.full((B, 1), bos_idx, dtype=torch.long, device=memory.device)
        finished = torch.zeros(B, dtype=torch.bool, device=memory.device)

        for _ in range(min(max_new_tokens, self.max_seq_len - 1)):
            T = seq.shape[1]
            tok_pos = torch.arange(T, device=memory.device).unsqueeze(0).expand(B, T)
            tgt = self.tok_emb(seq) + self.pos_emb(tok_pos)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                T, device=memory.device
            )
            out = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal_mask)
            next_token = self.fc_out(out[:, -1, :]).argmax(dim=-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, eos_idx), next_token
            )
            seq = torch.cat([seq, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_idx)
            if finished.all():
                break
        return seq.tolist()


# =====================================================================
# 4. LOAD DATA + VOCAB + ENCODER
# =====================================================================
train_df = pd.read_csv(DATA_ROOT / "Training_Labels.csv")
vocab = CaptionVocabulary(min_freq=2)
vocab.build_vocab(train_df["description"])
print("Vocab size:", len(vocab.word2idx))

# Teammate's pretrained image encoder (frozen for this quick test).
# NOTE: features_only=True returns the last feature stage (320 channels for
# tf_efficientnet_b0). This matches the encoder_test.py / embed() setup.
encoder = (
    timm.create_model(ENCODER_NAME, pretrained=True, features_only=True)
    .to(DEVICE)
    .eval()
)
for p in encoder.parameters():
    p.requires_grad = False

cfg = resolve_data_config({}, model=encoder)
tf = transforms.Compose([
    transforms.Resize(cfg["input_size"][1:]),
    transforms.CenterCrop(cfg["input_size"][1:]),
    transforms.ToTensor(),
    transforms.Normalize(cfg["mean"], cfg["std"]),
])

# =====================================================================
# 5. SMALL BATCH: 32 REAL IMAGES + CAPTIONS
# =====================================================================
sample_df = train_df.iloc[:32].reset_index(drop=True)
imgs = torch.stack([
    tf(Image.open(DATA_ROOT / "Training" / fname).convert("RGB"))
    for fname in sample_df["image_filename"]
]).to(DEVICE)

with torch.no_grad():
    feats = encoder(imgs)
    print("\n[encoder] feature map shapes:")
    for f in feats:
        print("  ", tuple(f.shape), f.dtype, f.device)
    # (32, C, H', W') -> (32, C). C will be 320 for tf_efficientnet_b0.
    img_feats = feats[-1].mean(dim=(2, 3))

ENCODER_DIM = img_feats.shape[-1]
print(f"[encoder] pooled feature dim: {ENCODER_DIM}")
print(f"[encoder] img_feats shape: {tuple(img_feats.shape)}")

tokens = collate_captions(
    [vocab.encode(c) for c in sample_df["description"]], pad_idx=0
).to(DEVICE)
decoder_in = tokens[:, :-1]
targets = tokens[:, 1:]

# =====================================================================
# 6. TRAIN FOR 60 STEPS ON THESE 32 IMAGES
# =====================================================================
decoder = ImageCaptionDecoder(
    vocab_size=len(vocab.word2idx),
    encoder_dim=ENCODER_DIM,   # derived, not hard-coded
).to(DEVICE)
optimizer = torch.optim.AdamW(decoder.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

for step in range(1, 61):
    decoder.train()
    optimizer.zero_grad()
    logits = decoder(img_feats, decoder_in)
    loss = loss_fn(logits.reshape(-1, len(vocab.word2idx)), targets.reshape(-1))
    loss.backward()
    optimizer.step()

    if step % 15 == 0 or step == 1:
        print(f"Step {step:02d} | Loss: {loss.item():.4f}")

# =====================================================================
# 7. GENERATE + PRINT
# =====================================================================
preds = decoder.generate_greedy(
    img_feats[:3],
    bos_idx=vocab.word2idx["<BOS>"],
    eos_idx=vocab.word2idx["<EOS>"],
)
print("\n--- Predictions after 60 steps ---")
for i in range(3):
    print(f"Image {i} Target:    {sample_df.loc[i, 'description']}")
    print(f"Image {i} Predicted: {vocab.decode(preds[i])}\n")