# -*- coding: utf-8 -*-
"""
GLUQ inference script.

Loads a released GLUQ checkpoint and predicts the perceptual quality score of a
UHD image with the aspect-ratio-aligned patch-graph pipeline:

  UHD image --(18x12 grid of 256x256 patches)--> ResNet-50 features
            --(patch kNN graph, k=24)--> 3 residual GCN layers
            --(gated attention readout + affine calibration)--> quality score

Only the inference path is provided (no training code). The model/graph code is
copied verbatim from the training implementation so predictions match exactly.

Usage:
    python gluq_inference.py --ckpt gluq_uhd_iqa.pt --image path/to/image.jpg
    python gluq_inference.py --ckpt gluq_uhd_iqa.pt --image img.jpg --cuda

Patch grid, k, tau, alpha, etc. are read from the checkpoint, so the .pt is
self-describing.
"""
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from torchvision import transforms


# -----------------------------------------------------------------------------
# Model definition (verbatim from the training implementation)
# -----------------------------------------------------------------------------
class ResidualGCNLayer(nn.Module):
    def __init__(self, in_ch, out_ch, alpha=0.5, dropout=0.1):
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch)
        self.bn = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.alpha = alpha
        if in_ch != out_ch:
            self.res = nn.Linear(in_ch, out_ch)
        else:
            self.res = nn.Identity()

    def forward(self, h, Ahat):
        B, N, C = h.shape
        h2 = torch.bmm(Ahat, h)
        h2 = self.lin(h2).view(B * N, -1)
        h2 = self.bn(h2)
        h2 = F.relu(h2, inplace=True).view(B, N, -1)
        h2 = self.dropout(h2)
        return self.alpha * h2 + (1 - self.alpha) * self.res(h)


class GatedAttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU(inplace=True), nn.Linear(dim // 2, 1))

    def forward(self, h):
        w = self.gate(h)
        att = torch.softmax(w, dim=1)
        return (att * h).sum(dim=1)


class AffineCalib(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Parameter(torch.ones(1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.s * x + self.b


class R50_GCN_IQA(nn.Module):
    def __init__(self, hidden_dim=512, gcn_layers=2, dropout=0.1, k_graph=10, alpha=0.5, tau=0.35,
                 graph_mode="distance", tau_dist=None, backbone="resnet50"):
        super().__init__()
        self.k_graph = int(k_graph)
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.graph_mode = (graph_mode or "distance").lower()
        self.tau_dist = self.tau if tau_dist is None else float(tau_dist)
        backbone = (backbone or "resnet50").lower()
        if "34" in backbone:
            r = tvm.resnet34(weights=None); feat_dim = 512
        else:
            r = tvm.resnet50(weights=None); feat_dim = 2048
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.feat_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(feat_dim, hidden_dim)
        self.gcn = nn.ModuleList([ResidualGCNLayer(hidden_dim, hidden_dim, alpha=self.alpha, dropout=dropout)
                                  for _ in range(int(gcn_layers))])
        self.pool = GatedAttnPool(hidden_dim)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True),
                                  nn.Linear(hidden_dim // 2, 1))
        self.calib = AffineCalib()

    def _knn_graph_distance(self, coords, k):
        B, N, _ = coords.shape
        k_eff = min(int(k), max(N - 1, 0))
        device = coords.device
        if N <= 1 or k_eff == 0:
            return torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)
        d = torch.cdist(coords, coords, p=2)
        idx = torch.topk(d, k=k_eff + 1, largest=False).indices[:, :, 1:]
        A = torch.zeros(B, N, N, device=device)
        batch_idx = torch.arange(B, device=device)[:, None, None].expand_as(idx)
        row_idx = torch.arange(N, device=device)[None, :, None].expand_as(idx)
        A[batch_idx, row_idx, idx] = 1.0
        A = A + torch.eye(N, device=device).unsqueeze(0)
        return A / (A.sum(dim=-1, keepdim=True) + 1e-8)

    def _knn_graph_feature(self, h, k):
        B, N, _ = h.shape
        k_eff = min(int(k), max(N - 1, 0))
        device = h.device
        if N <= 1 or k_eff == 0:
            return torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)
        h_norm = F.normalize(h, p=2, dim=-1)
        sim = torch.bmm(h_norm, h_norm.transpose(1, 2))
        eye = torch.eye(N, device=device).unsqueeze(0)
        sim = sim.masked_fill(eye.bool(), float("-inf"))
        idx = torch.topk(sim, k=k_eff, largest=True).indices
        s = torch.gather(sim, dim=2, index=idx)
        w = torch.softmax(s / max(self.tau, 1e-6), dim=-1)
        A = torch.zeros(B, N, N, device=device)
        batch_idx = torch.arange(B, device=device)[:, None, None].expand_as(idx)
        row_idx = torch.arange(N, device=device)[None, :, None].expand_as(idx)
        A[batch_idx, row_idx, idx] = w
        A = A + torch.eye(N, device=device).unsqueeze(0)
        return A / (A.sum(dim=-1, keepdim=True) + 1e-8)

    def _knn_graph_hybrid(self, coords, h, k):
        B, N, _ = coords.shape
        k_eff = min(int(k), max(N - 1, 0))
        device = coords.device
        if N <= 1 or k_eff == 0:
            return torch.eye(N, device=device).unsqueeze(0).repeat(B, 1, 1)
        d = torch.cdist(coords, coords, p=2)
        closeness = torch.exp(-d / max(self.tau_dist, 1e-6))
        h_norm = F.normalize(h, p=2, dim=-1)
        cos = torch.bmm(h_norm, h_norm.transpose(1, 2))
        sim01 = (cos + 1.0) * 0.5
        score = closeness * sim01
        eye = torch.eye(N, device=device).unsqueeze(0)
        score = score.masked_fill(eye.bool(), float("-inf"))
        idx = torch.topk(score, k=k_eff, largest=True).indices
        s = torch.gather(score, dim=2, index=idx)
        w = torch.softmax(s / max(self.tau, 1e-6), dim=-1)
        A = torch.zeros(B, N, N, device=device)
        batch_idx = torch.arange(B, device=device)[:, None, None].expand_as(idx)
        row_idx = torch.arange(N, device=device)[None, :, None].expand_as(idx)
        A[batch_idx, row_idx, idx] = w
        A = A + torch.eye(N, device=device).unsqueeze(0)
        return A / (A.sum(dim=-1, keepdim=True) + 1e-8)

    def backbone_forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        return self.feat_pool(x).flatten(1)

    def forward(self, xs, coords):
        B, N, C, H, W = xs.shape
        feats = self.backbone_forward(xs.view(B * N, C, H, W))
        h = self.proj(feats).view(B, N, -1)
        gm = self.graph_mode
        if gm == "distance":
            Ahat = self._knn_graph_distance(coords, k=self.k_graph)
        elif gm == "feature":
            Ahat = self._knn_graph_feature(h, k=self.k_graph)
        elif gm == "hybrid":
            Ahat = self._knn_graph_hybrid(coords, h, k=self.k_graph)
        else:
            raise ValueError(f"Unknown graph_mode: {gm}. Use distance|feature|hybrid")
        for layer in self.gcn:
            h = layer(h, Ahat)
        g = self.pool(h)
        return self.calib(self.head(g)).squeeze(-1)


# -----------------------------------------------------------------------------
# Deterministic patch grid (jitter=0; identical to the evaluation protocol)
# -----------------------------------------------------------------------------
def grid_patches(img, grid_cols, grid_rows, patch_size=256):
    W, H = img.size
    gx, gy, ps = grid_cols, grid_rows, patch_size
    sx = max(1, (W - ps) // max(1, gx - 1))
    sy = max(1, (H - ps) // max(1, gy - 1))
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                  std=(0.229, 0.224, 0.225))])
    patches, coords = [], []
    for iy in range(gy):
        for ix in range(gx):
            x = int(np.clip(ix * sx, 0, max(0, W - ps)))
            y = int(np.clip(iy * sy, 0, max(0, H - ps)))
            patches.append(tf(img.crop((x, y, x + ps, y + ps))))
            coords.append([(x + ps / 2) / W, (y + ps / 2) / H])
    return torch.stack(patches, 0), torch.tensor(coords, dtype=torch.float32)


def load_gluq(ckpt_path, device="cpu"):
    st = torch.load(ckpt_path, map_location=device)
    cfg = st["cfg"]; tms = st["train_mean_std"]
    model = R50_GCN_IQA(
        hidden_dim=cfg["hidden_dim"], gcn_layers=cfg["gcn_layers"], dropout=cfg["dropout"],
        k_graph=cfg["k_graph"], alpha=cfg["alpha"], tau=cfg["tau"],
        graph_mode=cfg["graph_mode"], tau_dist=cfg.get("tau_dist"), backbone=cfg["backbone"],
    ).to(device)
    model.load_state_dict(st["model"], strict=True)
    model.eval()
    return model, cfg, tms


@torch.no_grad()
def predict_quality(ckpt_path, img_path, device="cpu"):
    model, cfg, tms = load_gluq(ckpt_path, device)
    xs, coords = grid_patches(Image.open(img_path).convert("RGB"),
                              cfg["grid_cols"], cfg["grid_rows"], cfg.get("patch_size", 256))
    y = model(xs.unsqueeze(0).to(device), coords.unsqueeze(0).to(device)).item()
    mu, sig = tms
    return y * sig + mu          # de-normalized MOS in the benchmark's scale


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="GLUQ inference: predict UHD image quality.")
    p.add_argument("--ckpt", required=True, help="path to the released .pt checkpoint")
    p.add_argument("--image", required=True, help="path to a UHD image")
    p.add_argument("--cuda", action="store_true", help="use CUDA if available")
    args = p.parse_args()
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"
    score = predict_quality(args.ckpt, args.image, device)
    print(f"Image: {args.image}")
    print(f"Predicted quality (MOS scale): {score:.4f}")
