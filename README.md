# STELLAR: Learning Sparse Visual Representations via Spatial–Semantic Factorization

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2602.01905)
[![ICML Paper](https://img.shields.io/badge/ICML-2026-blue.svg)](https://openreview.net/pdf?id=ysOOfySED6)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![BibTeX](https://img.shields.io/badge/Cite-BibTeX-orange.svg)](#citation)

> **How many tokens are needed for one image?**
> We show that with a **spatial–semantic factorized representation**, **16 semantic tokens**
> paired with explicit spatial maps are enough for strong **image recognition and reconstruction**.
> STELLAR-H achieves **2.60 reconstruction FID** and **79.10% ImageNet linear-probing accuracy**,
> using approximately **90% fewer latent values** than a dense grid when counting both factors.

<p align="center">
  <img src="images/factorization.svg" alt="Spatial–semantic factorization" width="760">
</p>

By factorizing **"what"** (semantics) from **"where"** (spatial layout), STELLAR represents each image as the low-rank product of a **localization** matrix and a **semantics** matrix — modeling the multiple concepts in an image together with where they appear, for an efficient and holistic vision representation.

<p align="center">
  <img src="images/Method_compare.png" alt="STELLAR vs DINO, MAE and TiTok" width="640">
</p>

---

## Start here

| Your goal | Start with | What is available |
| :--- | :--- | :--- |
| Extract compact visual features | [Quick start](#quick-start-extract-features) | Five public pretrained models; encoder-only loading |
| Train or evaluate on your data | [Training and evaluation guide](docs/usage.md) | Image folders, seeded runs, checkpoint resume, DDP |

[Hugging Face models](https://huggingface.co/microsoft/STELLAR)

## Highlights

- **Sparse & unified** — one set of 16 tokens serves both high-level semantics and pixel-level reconstruction.
- **Factorized latents** — disentangles semantic content from spatial location, so each token captures a concept and *where* it appears.
- **Strong on both axes** — competitive FID/LPIPS for reconstruction *and* DINO-level linear probing / kNN for semantics.
- **Reusable representation** — use semantic tokens for compact downstream interfaces and retain spatial maps when layout matters.
- **Local or distributed training** — a MAE-initialized main recipe, optional random initialization, and label-free custom image folders.

The small output representation does **not** imply a 90% encoder speedup: the ViT
still processes dense image patches. Tokens are learned concepts, not guaranteed
objects or language-aligned embeddings.

### Unified representation

A lightweight 6-layer ViT decoder trained on **frozen** STELLAR features handles reconstruction, while linear probing / kNN measure semantic quality (`*` TiTok uses its own larger decoder).

| | | Reconstruction | | Semantics | |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model** | **# tokens** | **FID ↓** | **LPIPS ↓** | **Lin.** | **kNN** |
| DINO | 1 | - | - | 76.46 | 74.69 |
| DINO | 196 | 3.27 | 0.2121 | 70.31 | 54.41 |
| MAE | 196 | 3.02 | **0.2071** | 66.32 | 25.82 |
| TiTok* | 32 | 2.75 | 0.3281 | 33.42 | 7.30 |
| TiTok* | 64 | 1.99 | 0.2571 | 32.87 | 7.29 |
| **STELLAR** | **16** | **3.06** | **0.2077** | **73.26** | **67.25** |
| **STELLAR** | **196** | **2.85** | **0.2085** | **72.21** | **64.71** |
| **STELLAR (H)** | **16** | **2.60** | **0.1729** | **79.10** | **77.31** |

<p align="center">
  <img src="images/recon_semantic_bars.svg" alt="Reconstruction and semantic benchmarks" width="900">
</p>

---

## Method

STELLAR encodes an image into a small set of **latent queries** that become
**sparse tokens**. The representation is factorized into a **localization** matrix `L`
(*where* a concept appears, `n × r`) and a **semantics** matrix `S` (*what* the
concept is, `r × d`). Their low-rank product reassembles a dense feature map that
a lightweight decoder reconstructs, while the sparse tokens themselves carry the
semantics.

<p align="center">
  <img src="images/stellar.svg" alt="STELLAR architecture" width="960">
</p>

Training combines a **reconstruction** loss with self-supervised **clustering**
and **set-alignment** losses: online clustering assigns tokens to prototypes,
and optimal-transport matching aligns the tokens of a global view with those of
partial/augmented views. As a result the factorized tokens localize cleanly to
the semantic regions of an image.

<p align="center">
  <img src="images/match-viz.svg" alt="Clustering, matching and token localization" width="960">
</p>

---

## Installation

```sh
git clone https://github.com/microsoft/STELLAR.git
cd STELLAR

conda create -n stellar python=3.10.14
conda activate stellar

python -m pip install -r requirements-inference.txt
```

This is enough for feature extraction and reconstruction model loading. Run from
the repo root so `load_stellar` and `src` are importable. For training, follow the
[training dependency instructions](docs/usage.md#install-only-what-you-need).
No Azure account is required for local runs.

### External directory

Create one external directory to hold datasets and pretrained weights. We refer to it as `EXTERNAL` throughout:

```text
<EXTERNAL>/
├── ImageData/
│   └── ImageNet/                # ImageNet-1K (see Data preparation)
└── PretrainedModels/
    ├── vqgan/maskgit/           # MaskGIT-VQGAN tokenizer (for reconstruction)
    └── stellar/                 # optional checkpoints from your own training
```

You can point the code at it on the command line with `mounts.external=<EXTERNAL>`.

---

## Pretrained models

| Model | Backbone | # tokens | Type | Download |
| :--- | :--- | :---: | :--- | :--- |
| STELLAR-B16 | ViT-B/16 | 16 | main | [🤗 stellar-b16](https://huggingface.co/microsoft/STELLAR/blob/main/stellar-b16.safetensors) |
| STELLAR-L16 | ViT-L/16 | 16 | main | [🤗 stellar-l16](https://huggingface.co/microsoft/STELLAR/blob/main/stellar-l16.safetensors) |
| STELLAR-H16 | ViT-H/14 | 16 | main | [🤗 stellar-h16](https://huggingface.co/microsoft/STELLAR/blob/main/stellar-h16.safetensors) |
| STELLAR-B8 | ViT-B/16 | 8 | ablation | [🤗 stellar-b8](https://huggingface.co/microsoft/STELLAR/blob/main/stellar-b8.safetensors) |
| STELLAR-B24 | ViT-B/16 | 24 | ablation | [🤗 stellar-b24](https://huggingface.co/microsoft/STELLAR/blob/main/stellar-b24.safetensors) |

All weights are pretrained self-supervised on ImageNet-1K and hosted on the
[🤗 Hugging Face repo](https://huggingface.co/microsoft/STELLAR) as `safetensors`.

We also use the MaskGIT-VQGAN tokenizer from [TiTok](https://github.com/bytedance/1d-tokenizer/blob/main/README_TiTok.md) for image reconstruction. Download [`maskgit-vqgan-imagenet-f16-256.bin`](https://huggingface.co/fun-research/TiTok/blob/main/maskgit-vqgan-imagenet-f16-256.bin) and place it under `<EXTERNAL>/PretrainedModels/vqgan/maskgit/`.

---

## Quick start: extract features

STELLAR produces a small set of sparse tokens per image. For feature extraction you only need the encoder — no decoder or VQGAN tokenizer required.

```python
import torch
from load_stellar import load_stellar
from examples.common import read_image

model = load_stellar(
  "stellar-b16",
  revision="6794be9a20fb9d3944c5bcc6003512a347fa518f",
  device="cpu",                        # or "cuda"
)

image = read_image("your_image.jpg").unsqueeze(0)  # RGB in [0, 1]
image = image.to(next(model.parameters()).device)
with torch.no_grad():
    out = model.encode(image)

out["sparse"]    # (1, 16, 768)  factorized concept tokens  ("what")
out["spatial"]   # (1, 196, 16)  per-token spatial maps     ("where")
out["cls"]       # (1, 1, 768)   global image token
out["dense"]     # (1, 196, 768) dense patch features
```

| Key | Shape | Use |
| :--- | :--- | :--- |
| `sparse` | `(B, K, D)` | semantic concept tokens (classification, retrieval) |
| `spatial` | `(B, P, K)` | spatial distribution of each token (segmentation, visualization) |
| `cls` | `(B, 1, D)` | global representation |
| `dense` | `(B, P, D)` | dense per-patch features |
| `lowrank` | `(B, P, D)` | reassembled dense map (reconstruction input) |

`K` is the sparse-token count. At 224x224 input, B/L have `P=196`; H/14 has
`P=256`. Embedding widths are 768/1024/1280 for B/L/H. ImageNet normalization is
applied **inside** the model. The loader checks every required weight, supports
`local_files_only=True`, and does not download a second MAE backbone.

> The released checkpoints include the decoder and clustering heads too. For image
> reconstruction, `model.reconstruct(features)` is the decoder half of STELLAR — pass the
> factorized features from `model.encode(image)` (or `model.reconstruct(sparse, spatial)`)
> and it runs low-rank dense map → ViT decoder → VQGAN decoder, returning
> `out["reconstruction"]` as `(B, 3, H, W)` RGB pixels in `[0, 1]`. Load with
> `load_stellar(purpose="reconstruct", vq_model=...)`; use `purpose="pretrain"`
> for the full training model. H/14 reconstruction is 256x256; B/L reconstruction
> is 224x224. See [the complete guide](docs/usage.md#reconstruction).

---

## Data preparation

### ImageNet-1K

Download [ImageNet-1K](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data) and place it under `<EXTERNAL>/ImageData/ImageNet` so that it contains:

```text
ImageNet/
├── ILSVRC/Data/CLS-LOC/train/...
├── ILSVRC/Data/CLS-LOC/val/...
├── imagenet_class_index.json
└── ILSVRC2012_val_labels.json
```

### Custom dataset

STELLAR is fully self-supervised: it needs only images. The provided
[ImageFolderSSL](src/datasets/image_folder.py) reads flat or nested image folders
without labels, class subdirectories or ImageNet metadata:

```bash
python run.py --config-name stellar datamodule=image_folder \
  scratch.data_root=/path/to/train_images \
  scratch.test_data_root=/path/to/held_out_images \
  mounts.external=<EXTERNAL> datamodule.dataloaders.train.batch_size=16
```

For another data source, implement a `Dataset` in [src/datasets/](src/datasets/)
that returns the same dictionary keys:

```python
{
    "image":        global_views[0],   # (3, H, W)    one global crop
    "global_views": global_views,      # (V, 3, H, W) several global crops
    "local_views":  local_views,       # (V, 3, h, w) several local crops
    "labels":       global_views[0],   # unused by SSL; any tensor works
}
```

See [src/datasets/imagenet_dataset.py](src/datasets/imagenet_dataset.py) for a reference implementation of the multi-crop augmentation.

---

## Pretraining

Training is driven by [Hydra](https://hydra.cc/) configs under [configs/](configs/).
The local entry point seeds data splitting and wraps the Olympus/Lightning trainer:

```bash
python run.py --config-name stellar mounts.external=<EXTERNAL> scratch.seed=42
```

### Configuration guide

Start with [configs/stellar.yaml](configs/stellar.yaml) for pretraining. Edit the
recipe YAML or override individual values on the command line:

| File | What to change |
| :--- | :--- |
| [configs/stellar.yaml](configs/stellar.yaml) | Data/output roots, backbone, token count, batch size, learning rate, epochs and resume |
| [configs/datamodule/image_folder.yaml](configs/datamodule/image_folder.yaml) | Custom-image dataset setup and validation split; select with `datamodule=image_folder` |
| [configs/trainer/stellar_trainer.yaml](configs/trainer/stellar_trainer.yaml) | Lightning defaults: devices, precision, distributed strategy and gradient clipping |
| [configs/eval_cls.yaml](configs/eval_cls.yaml) | Classification probe: pretrained model, features, classes and training settings |
| [configs/eval_seg.yaml](configs/eval_seg.yaml) | Segmentation probe: pretrained model, features, classes and training settings |
| [configs/eval_recon.yaml](configs/eval_recon.yaml) | Reconstruction probe: pretrained model, decoder, tokenizer path and training settings |

Values in the selected recipe override its imported defaults. Common overrides:

| Config | Meaning |
| :--- | :--- |
| `mounts.external` / `mounts.output` | Dataset/weight root and output root |
| `job_name` | Run name under the output root |
| `scratch.data_root` / `scratch.test_data_root` | Train/test image folders when using `datamodule=image_folder` |
| `model.vit_pretrained` | MAE backbone for pretraining (base, large or huge) |
| `model.num_sparse_tokens` | number of sparse tokens (e.g. 16) |
| `model.do_recon` | enable VQGAN reconstruction branch |
| `model.do_clustering` | enable online clustering / self-distillation |
| `model.vq_model` | Local MaskGIT-VQGAN weights for reconstruction |
| `datamodule.dataloaders.train.batch_size` | Images per device per training step |
| `datamodule.dataloaders.train.num_workers` | Data-loading workers per device |
| `trainer.devices` / `trainer.num_nodes` | Devices per node and node count |
| `trainer.max_epochs` | training length |
| `optimizer.lr` | learning rate |
| `scratch.seed` / `scratch.resume` | Random seed and full Lightning checkpoint to resume |

```bash
python run.py --config-name stellar mounts.external=/path/to/external \
  trainer.devices=1 datamodule.dataloaders.train.batch_size=16 \
  optimizer.lr=0.00015 job_name=my_run

# Inspect the fully resolved config without loading data or starting training.
python run.py --config-name stellar --cfg job --resolve
```

The default is the **MAE-initialized B16 recipe**, with 150 epochs and learning
rate 0.00015. The paper used a global batch of 2048 on 16 A100-80GB GPUs; the local
default of one GPU does not reproduce that batch. See the
[recipe and reproduction notes](docs/usage.md#pretraining-on-imagenet).

### Optional training extensions

Two optional extensions are exposed as config flags. They are not benchmarked
reproductions of the released models:

**1. Second global-view alignment** (`model.do_global_align`)

Encodes a second global crop and aligns the two views' cluster assignments
(DINO-style cross-view self-distillation, with Sinkhorn-based set matching between
the sparse token slots).

```yaml
model:
  do_global_align: True
```

**2. Momentum-teacher scheduling** (`model.teacher_momentum_*`)

When a momentum teacher is enabled, the EMA momentum can follow a cosine
schedule from `teacher_momentum` to `teacher_momentum_final` over
`teacher_momentum_schedule_steps` optimizer steps (≈ `epochs × steps_per_epoch`).
Leaving the schedule fields as `null` keeps the momentum constant.

```yaml
model:
  momentum_teacher: True
  teacher_momentum: 0.996            # start
  teacher_momentum_final: 1.0        # end
  teacher_momentum_schedule_steps: 90000
```

Both extensions are off in the main recipe. Random initialization uses
`model.vit_pretrained=null`; the paper's two-stage EMA warm-up protocol is
described separately in [the guide](docs/usage.md#random-initialization-and-optional-extensions).

### Multi-GPU and resuming

Training uses Lightning's DDP strategy out of the box — just request more
devices (the Sinkhorn assignment and the EMA teacher are already
distributed-aware and synchronized across ranks):

```bash
python run.py --config-name stellar mounts.external=<EXTERNAL> \
  trainer.devices=8 trainer.num_nodes=1
```

Resume a complete training run with `scratch.resume=/path/to/last.ckpt`.
See [multi-node launch and batch accounting](docs/usage.md#multi-gpu-and-multi-node)
and [resume caveats](docs/usage.md#resume-and-repeatability). Gradient accumulation
does not enlarge per-forward Sinkhorn assignment batches.

When resuming, the model weights, optimizer state, and EMA teacher weights are
restored from the checkpoint as usual. The momentum **schedule position** is
restored too: [MomentumScheduleCallback](src/callbacks/momentum_schedule.py)
re-syncs the schedule from the restored `global_step` and (if left unset)
auto-computes `teacher_momentum_schedule_steps` from the total optimizer steps
for your device / accumulation setup. It is a no-op unless `momentum_teacher`
is enabled.

---

## Downstream evaluation

Frozen-feature evaluation configs are provided for the three reported tasks. Each loads a pretrained checkpoint and trains only a lightweight head.

```bash
# Linear probing (classification)
python run.py --config-name eval_cls mounts.external=<EXTERNAL>

# Reconstruction (frozen-feature ViT decoder)
python run.py --config-name eval_recon mounts.external=<EXTERNAL>

# Semantic segmentation (ADE20K)
python run.py --config-name eval_seg mounts.external=<EXTERNAL>
```

Each eval config selects which feature to probe via `model.feature_key`
(`sparse` for classification, `dense` for segmentation, `lowrank` for
reconstruction). They load the public Hub encoder by default and train a **new**
head. To test a trained head, use `mode=test +scratch.resume=/path/to/probe.ckpt`.
The [evaluation guide](docs/usage.md#frozen-feature-evaluation) covers dataset
paths, model sizes, held-out splits and hyperparameter selection. A standalone
[rFID/LPIPS command](docs/usage.md#reconstruction-metrics) evaluates the released
reconstruction decoder; it does not silently claim Table 1 reproduction.

---

## Pretrained weights on Hugging Face

The released weights are hosted at
[huggingface.co/microsoft/STELLAR](https://huggingface.co/microsoft/STELLAR) as
`safetensors`, including the trained encoder, reconstruction and clustering
modules (the external VQGAN is separate). The repo-local
[load_stellar.py](load_stellar.py) selects which modules to build:

```python
from load_stellar import load_stellar, list_models

print(list_models())                 # ['stellar-b16', 'stellar-l16', ...]
model = load_stellar("stellar-b16")   # downloads weights from the Hub
```

Download via `huggingface_hub` / the helper above (not `git clone`) so that downloads
are registered on the Hub.

---

## Research connections

STELLAR is a **self-supervised visual encoder and sparse visual token extractor**
for research on compact, spatially grounded representations. Its
**spatial-semantic factorization** retains semantic concept tokens together with
the spatial maps needed for image reconstruction.

- **Self-supervised learning (SSL):** frozen features support classification,
  semantic segmentation and reconstruction, as evaluated in the paper.
- **Vision-language models (VLMs) and visual token compression:** sparse tokens
  provide a compact interface for learned multimodal adapters. Appendix B.3
  evaluates alignment to a CLIP text tower. Researchers can feed `out["sparse"]`
  into a learned adapter for their language model; no pretrained VLM adapter or
  VLM speedup result is provided.
- **Representation autoencoders (RAEs) and latent diffusion:** reconstructable
  semantic features suggest a possible connection to
  [RAE-style generative modeling](https://arxiv.org/abs/2510.11690).
  **We have not conducted RAE-style image-generation experiments and make no
  generation-quality claims.** Reconstruction needs both semantic and spatial
  factors; reconstruction FID is not generation FID.
- **World models and latent dynamics:** separating content from layout may be
  useful for future video prediction research. Temporal consistency, action
  conditioning and planning have not been evaluated in this work.

## Citation

If you find STELLAR useful for your research, please cite:

```bibtex
@inproceedings{zhao2026stellar,
  title     = {Learning Sparse Visual Representations via Spatial-Semantic Factorization},
  author    = {Zhao, Theodore Zhengde and Kiblawi, Sid and Yang, Jianwei and Usuyama, Naoto and Tan, Reuben and Codella, Noel C and Naumann, Tristan and Poon, Hoifung and Wei, Mu},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.01905},
}
```

## License

This project is released under the [MIT License](LICENSE).