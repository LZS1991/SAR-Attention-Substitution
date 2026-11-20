# Channel attention and depthwise-separable convolution behave as substitutes in SAR water body segmentation

**A budget-matched U-Net ablation**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MMSegmentation](https://img.shields.io/badge/MMSegmentation-1.2.2-00b4d8)](https://github.com/open-mmlab/mmsegmentation)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

This repository contains the code for the study above. It is built on
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation) `v1.2.2` and provides the U-Net
variant used in the paper (referred to there as **SAR-ECA-U-Net**), its training configuration, and
the $2^{3}$ ablation grid behind the analysis.

---

## 📌 What the study shows

The paper does not propose a new attention mechanism. It asks a narrower question: in a U-Net for
SAR water body segmentation, which of three inexpensive design choices actually contributes, and how
do they interact at a fixed parameter budget?

- **A complete $2^{3}$ grid** over encoder-stage Efficient Channel Attention (ECA), depthwise-separable
  convolution (DSConv) and a composite CE–Dice objective, trained on DaliWS and GF-FloodNet.
- **Main effects** (mIoU points) are **1.17 / 1.36 / 0.93** on DaliWS and **0.58 / 0.34 / 0.54** on
  GF-FloodNet for ECA / DSConv / CE–Dice.
- **Attention and compression behave as substitutes, not complements.** The ECA × DSConv interaction
  is **−2.49** on DaliWS and **−0.84** on GF-FloodNet, the largest term in the grid. In the
  cross-entropy cells ECA is worth **+3.23 / +1.11** mIoU while the blocks are full-rank and
  **−0.23 / −0.08** once the first convolution of every block is depthwise separable.
- **Placement matters only before compression.** At 29.06M the encoder placement leads on both
  benchmarks (90.34 / 95.40); at 17.95M it leads on neither, and the spread across four placements
  falls to 0.40 and 0.09 mIoU against a run-to-run difference of 0.43 and 0.15 for the identical
  configuration.
- **The resulting network** reaches **90.51%** and **95.38%** mIoU with **17.95M** parameters, 38.2%
  fewer than the standard U-Net.

> Every cell is a single run and all numbers are best-validation values on a tile-level random split
> shared by all compared configurations. The margins are reported as descriptive differences.

---

## 🗂️ Repository layout

```text
sar-attention-substitution/
├── configs/
│   ├── _base_/
│   │   ├── models/
│   │   │   ├── fcn_unet_eca_s5-d16.py      # U-Net + ECA + DSConv + CE-Dice (lambda = 2.0)
│   │   │   └── fcn_unet_s5-d16.py          # plain U-Net baseline
│   │   ├── datasets/
│   │   │   ├── _custom_DaliWS.py
│   │   │   └── _custom_GF_Flood.py
│   │   ├── schedules/schedule_40k.py       # 40k iterations, polynomial LR
│   │   └── default_runtime.py
│   └── _custom/
│       ├── unet_eca-s5-d16_fcn_DaliWS-256x256.py
│       └── unet_eca-s5-d16_fcn_GFFlood-256x256.py
├── mmseg/
│   ├── models/backbones/unet_eca.py        # UNetECA: encoder-stage ECA + optional DSConv
│   ├── models/utils/eca_block.py           # ECA module
│   └── ...                                 # MMSegmentation v1.2.2 source
├── tools/
│   ├── train.py
│   └── test.py
├── requirements.txt
├── LICENSE                                  # Apache-2.0
└── README.md
```

---

## 🛠️ Installation

```bash
conda create -n sar-attn python=3.9 -y
conda activate sar-attn

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install mmcv==2.1.0 mmengine==0.10.0
pip install -r requirements.txt
```

The repository root contains the `mmseg` package itself (a fork of MMSegmentation 1.2.2 with the
`UNetECA` backbone and the ECA block added). Run every command from the repository root so that this
local copy is imported instead of a pip-installed one; no separate installation step is required.

Environment used in the paper: Python 3.9.25, PyTorch 2.1.0, CUDA 11.8.0, MMSegmentation 1.2.2,
one NVIDIA RTX A6000 (48 GB).

---

## 📊 Data preparation

### DaliWS

- 2,033 non-overlapping $256\times256$ tiles, GF-3, 5 m, pixel-level binary labels (0 background,
  1 water).
- Reference: <https://doi.org/10.3390/rs16040720>

### GF-FloodNet

- 13,388 tiles of $256\times256$ pixels covering 8 flood events in 7 countries; 4,178 contain flood
  area and 9,210 do not; ground sampling distance 1.5 m, 2.5 m and 4 m.
- The dataset is released as **GF-3 SAR and GF-2 multispectral imagery**; the subset used here is
  described in the paper.
- Reference: <https://doi.org/10.1080/17538947.2023.2230978>

Expected layout (paths are set in `configs/_base_/datasets/_custom_*.py`):

```text
data/DaliWS/
├── images/train/        *.png
├── images/val/
├── annotations/train/
└── annotations/val/
```

---

## 🚀 Training and evaluation

```bash
# DaliWS
python tools/train.py configs/_custom/unet_eca-s5-d16_fcn_DaliWS-256x256.py \
    --work-dir work_dirs/daliws

# GF-FloodNet
python tools/train.py configs/_custom/unet_eca-s5-d16_fcn_GFFlood-256x256.py \
    --work-dir work_dirs/gfflood
```

```bash
python tools/test.py configs/_custom/unet_eca-s5-d16_fcn_DaliWS-256x256.py \
    work_dirs/daliws/best_mIoU_iter_40000.pth --work-dir work_dirs/daliws/eval
```

### Key hyperparameters

| Setting | Value |
|---|---|
| Input size | $256\times256$ |
| Batch size | 24 |
| Iterations | 40,000, validation every 500 |
| Optimizer | SGD, momentum 0.9, weight decay $5\times10^{-4}$ |
| Learning rate | 0.01, polynomial decay, 500-iteration warm-up |
| Loss | $L_{CE} + \lambda L_{Dice}$ with $\lambda = 2.0$ (`DiceLoss` weight = 2.0) |
| Deep supervision | auxiliary FCN head on the second-to-last decoder stage, $0.4\,L_{CE} + 0.8\,L_{Dice}$, training only |
| Augmentation | horizontal flip (p = 0.5), random scale 0.5–2.0, random crop 256 with `cat_max_ratio = 0.75`, photometric distortion |
| Normalization | ImageNet mean/std, as provided by MMSegmentation |

---

## 📈 Results

### DaliWS

| Method | Backbone | Params (M) | mIoU | mACC | F1 |
|---|---|---:|---:|---:|---:|
| U-Net | S5-D16 | 29.06 | 86.57 | 92.01 | 92.53 |
| FCN | R50 | 49.48 | 87.92 | 93.85 | 93.36 |
| DeepLabV3 | R50 | 68.10 | 89.60 | 94.12 | 94.36 |
| DANet | R50 | 49.81 | 89.23 | 93.36 | 94.14 |
| GCNet | R50 | 49.62 | 89.02 | 93.01 | 94.01 |
| MaskFormer | Swin-T | 41.72 | 87.58 | 92.03 | 93.15 |
| Mask2Former | Swin-T | 47.40 | 88.56 | 93.18 | 93.74 |
| SegNeXt | MSCAN-B | 27.56 | 85.50 | 92.05 | 91.87 |
| **Ours** | — | **17.95** | **90.51** | **94.24** | **94.89** |

### GF-FloodNet

| Method | Backbone | Params (M) | mIoU | mACC | F1 |
|---|---|---:|---:|---:|---:|
| U-Net | S5-D16 | 29.06 | 93.74 | 95.96 | 96.72 |
| FCN | R50 | 49.48 | 93.17 | 95.93 | 96.41 |
| DeepLabV3 | R50 | 68.10 | 94.21 | 96.52 | 96.98 |
| DANet | R50 | 49.81 | 94.59 | 96.84 | 97.19 |
| GCNet | R50 | 49.62 | 94.13 | 96.26 | 96.93 |
| MaskFormer | Swin-T | 41.72 | 87.21 | 93.45 | 92.98 |
| Mask2Former | Swin-T | 47.40 | 84.19 | 88.06 | 91.08 |
| SegNeXt | MSCAN-B | 27.56 | 83.00 | 90.64 | 90.36 |
| **Ours** | — | **17.95** | **95.38** | **97.25** | **97.61** |

The Transformer-based baselines were trained with the same per-pixel objective rather than their
native set-prediction recipes, so the gap reflects both architecture and training recipe; the paper
treats this as a limitation of the comparison.

---

## 🧪 Component ablation and effect decomposition

Eight cells of the $2^{3}$ grid. ECA is switched at the encoder, DSConv switches the first
$3\times3$ convolution of every block, and CE-Dice switches the objective.

| ECA | DSConv | CE-Dice | Params (M) | DaliWS mIoU | GF-FloodNet mIoU |
|---|:---:|:---:|---:|---:|---:|
| — | — | — | 29.06 | 86.57 | 93.74 |
| ✓ | — | — | 29.06 | 89.80 | 94.85 |
| — | ✓ | — | 17.95 | 90.08 | 94.80 |
| — | — | ✓ | 29.06 | 88.74 | 94.51 |
| ✓ | ✓ | — | 17.95 | 89.85 | 94.72 |
| ✓ | — | ✓ | 29.06 | 90.34 | 95.40 |
| — | ✓ | ✓ | 17.95 | 90.44 | 94.98 |
| ✓ | ✓ | ✓ | 17.95 | **90.51** | **95.38** |

Effects estimated from the grid, in mIoU points. A main effect is the difference between the
four-cell averages at each factor level; an interaction is the difference of those differences,
averaged over the remaining factor.

| Term | DaliWS | GF-FloodNet |
|---|---:|---:|
| Main effect, ECA | 1.17 | 0.58 |
| Main effect, DSConv | 1.36 | 0.34 |
| Main effect, CE-Dice | 0.93 | 0.54 |
| Interaction, ECA × DSConv | **−2.49** | **−0.84** |
| Interaction, ECA × CE-Dice | −0.66 | 0.13 |
| Interaction, DSConv × CE-Dice | −0.84 | −0.24 |
| Interaction, three-way | 1.93 | 0.70 |

### Reproducing the grid

The three factors are controlled by two backbone flags and one loss weight, so the eight cells can
be produced from the two shipped configs without extra files:

```bash
CFG=configs/_custom/unet_eca-s5-d16_fcn_DaliWS-256x256.py

# ECA on/off          : model.backbone.enc_use_eca      (True | False)
# DSConv on/off       : model.backbone.use_dw           (True | False)
# CE-Dice on/off      : model.decode_head.loss_decode.1.loss_weight
#                       model.auxiliary_head.loss_decode.1.loss_weight   (2.0 / 0.8 | 0 / 0)

python tools/train.py $CFG \
    --cfg-options model.backbone.enc_use_eca=False model.backbone.use_dw=True \
                  model.decode_head.loss_decode.1.loss_weight=0 \
                  model.auxiliary_head.loss_decode.1.loss_weight=0 \
    --work-dir work_dirs/grid_eca0_dw1_loss0
```

Setting the Dice weight to 0 reproduces the pure cross-entropy cells used in the paper's
one-factor-at-a-time rows; the shipped default (`2.0` on the main head, `0.8` on the auxiliary head)
is the all-on cell.

### Attention placement
| DSConv | ECA placement | Params (M) | DaliWS mIoU | GF-FloodNet mIoU |
|---|---|:---:|---:|---:|
| off | decoder | 29.06 | 90.04 | 94.92 |
| off | skip | 29.06 | 89.89 | 94.77 |
| off | encoder | 29.06 | 90.34 | 95.40 |
| off | encoder + decoder | 29.06 | 89.85 | 94.63 |
| on | decoder | 17.95 | 89.98 | 95.32 |
| on | skip | 17.95 | 90.32 | 95.31 |
| on | encoder | 17.95 | 90.08 | 95.23 |
| on | encoder + decoder | 17.95 | 89.92 | 95.25 |

---

## ⚠️ Scope and known gaps

- **Single run per cell.** No multiple seeds. The identical configuration differs by 0.43 mIoU on
  DaliWS and 0.15 on GF-FloodNet between two runs; orderings smaller than that are not claimed.
- **All reported numbers are best-validation values.** No held-out test evaluation was performed.
- **Placement switches.** The released backbone (`UNetECA`) exposes `enc_use_eca` and `use_dw`. The
  decoder-only and skip-only attention variants used in the placement study are **not** part of this
  release, so the placement table above cannot be reproduced from this code as it stands.
- **Splits are drawn at tile level**, so adjacent tiles from the same event or river reach can appear
  on both sides of a split. Absolute values should not be transferred to operational settings.

---

## 📝 Citation

```bibtex
@article{chen2026substitutes,
  title   = {Channel attention and depthwise-separable convolution behave as substitutes
             in SAR water body segmentation: a budget-matched U-Net ablation},
  author  = {Chen, Jianhong and Liu, Wei and Ye, Zhengmao and Zhu, Qin and
             Huang, Yong and Wu, Xinying and Lu, Wanjie},
  journal = {Manuscript under review},
  year    = {2026},
  note    = {Code: https://github.com/LZS1991/sar-attention-substitution}
}
```

---

## 📄 License

Released under the [Apache License 2.0](LICENSE). MMSegmentation components retain their original
licenses; the datasets are governed by their own open-access terms (DaliWS
[doi:10.3390/rs16040720](https://doi.org/10.3390/rs16040720), GF-FloodNet
[doi:10.1080/17538947.2023.2230978](https://doi.org/10.1080/17538947.2023.2230978)).

## 🙏 Acknowledgements

Built upon [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and the ECA-Net attention
design ([Wang et al., CVPR 2020](https://doi.org/10.1109/CVPR42600.2020.01155)). Datasets: DaliWS
(Zhang et al., Remote Sensing 2024) and GF-FloodNet (Zhang et al., IJDE 2023).
