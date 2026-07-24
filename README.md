# GLUQ — Pretrained Weights & Inference

This repository releases the **trained checkpoint** and a **standalone inference
script** for:

> **GLUQ: Global-Local Representation Learning for UHD Image Quality Assessment**

It contains *only* the model weights and the inference code needed to reproduce
the prediction pipeline. The full training repository is not released.

The released checkpoint is the **best-calibrated GLUQ model** from our study: trained with the EMA-normalized variance–correlation objective, it achieves the **lowest test-set RMSE** among the model variants we trained, consistent with GLUQ's emphasis on absolute-score calibration highlighted in the paper.

## Contents

| File | Description |
|---|---|
| `gluq_inference.py` | Self-contained inference script (model definition + patch-grid + load + predict). No training code. |
| `gluq_uhd_iqa.pt` | Trained GLUQ checkpoint (download from **Releases** — see below). |

## Setup

```bash
pip install -r requirements.txt
```

## Get the checkpoint

The `.pt` is distributed as a **GitHub Release asset** (not committed to the git
tree, to keep the repository lightweight):

1. Go to the **Releases** page of this repository.
2. Download `gluq_uhd_iqa.pt` (~98 MB).
3. Place it next to `gluq_inference.py` (or pass its path with `--ckpt`).

## Run inference

```bash
# CPU
python gluq_inference.py --ckpt gluq_uhd_iqa.pt --image path/to/your_uhd.jpg

# GPU
python gluq_inference.py --ckpt gluq_uhd_iqa.pt --image path/to/your_uhd.jpg --cuda
```

Prints the predicted perceptual quality score on the MOS scale of the
UHD-IQA benchmark.

## Configuration

All hyper-parameters are stored inside the checkpoint and are read automatically:

| Component | Value |
|---|---|
| Patch grid | (G_w, G_h) = (18, 12), patch 256×256, N = 216 |
| Backbone | ResNet-50 (ImageNet pre-trained) |
| Graph | patch kNN graph, k = 24, τ = 0.35 (exact graph construction stored in the checkpoint) |
| GCN | 3 residual layers, α = 0.55, hidden dim 512 |
| Readout | gated attention pooling + MLP + affine calibration |
| Loss (training) | EMA-normalized multi-objective, λ_corr = 0.8, λ_rank = 0.2 |


All hyper-parameters (graph construction, k, τ, α, dropout, etc.) are stored **inside** the checkpoint and are read automatically by `gluq_inference.py`, so the released `.pt` is self-describing and no flags are needed at inference time.

Patches use the deterministic aspect-ratio-aligned grid (no jitter at inference).

## Dataset

Tested on the [UHD-IQA benchmark database](https://database.mmsp-kn.de/uhd-iqa-benchmark-database.html).
The image data is **not** included here; download it from the benchmark page.

## Citation

If you use these weights, please cite the paper:

```bibtex
@article{zhu2026gluq,
  title   = {GLUQ: Global-Local Representation Learning for UHD Image Quality Assessment},
  author  = {Zhu, Bing and Chen, Enqi and Huang, Ming and Ren, Xuemin and Yu, Shaode and Sun, Qiurui},
  journal = {Entropy},
  year    = {2026},
  note    = {Paper ID: entropy-4427081}
}
```

## Authors

- **Bing Zhu** — Zhubing1218@cuc.edu.cn
- **Enqi Chen** — 202311043037@mails.cuc.edu.cn
- **Ming Huang** — 202311043023@cuc.edu.cn
- **Xuemin Ren** — 202520085410002@mails.cuc.edu.cn
- **Shaode Yu** *(corresponding)* — yushaodecuc@cuc.edu.cn
- **Qiurui Sun** *(corresponding)* — qiuruisun@bnu.edu.cn

## License

The code in this repository is released under the MIT License. The model weights
are released for academic research use.
