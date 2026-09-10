# Running STELLAR

Run commands from the repository root. Python 3.10 is the tested version.
The encoder needs no Azure account, Olympus trainer, VQGAN download, or MAE
weight download when loaded through `load_stellar`.

## Install Only What You Need

```bash
conda create -n stellar python=3.10
conda activate stellar
python -m pip install -r requirements-inference.txt
```

For reconstruction metrics, also install `requirements-eval.txt`. For the
Olympus training/probing recipes, install the existing training dependencies:

```bash
python -m pip install 'pip<24.1'
python -m pip install -r requirements.txt
python -m pip install -r requirements-inference.txt
```

The pip constraint accommodates legacy training-package metadata. The inference
requirements pin the tested PyTorch/Transformers API versions; match the PyTorch
CUDA wheel to your driver. A fresh-machine installation of the legacy training
stack and physical multi-GPU runs have not been validated.

## Choose Your Config

See the [configuration guide](../README.md#configuration-guide) for the main
files and common overrides. Use `--config-name stellar` for pretraining,
`eval_cls` for classification, `eval_seg` for segmentation, and `eval_recon`
for reconstruction probing. The evaluation recipes train new heads on frozen
pretrained features; they do not run the pretraining recipe.

Edit the selected YAML or pass `key=value` overrides. Inspect the effective
configuration before a run with `python run.py --config-name stellar --cfg job --resolve`.
Hydra requires a `+` prefix when adding a key absent from that recipe, such as
`+scratch.seed=42` for the evaluation configs.

## Weights And Inputs

```python
import torch
from load_stellar import load_stellar
from examples.common import read_image

revision = "6794be9a20fb9d3944c5bcc6003512a347fa518f"
model = load_stellar("stellar-b16", revision=revision, device="cuda")
image = read_image("your_image.jpg").unsqueeze(0).cuda()
with torch.no_grad():
    features = model.encode(image)
torch.save({key: features[key].cpu() for key in ("sparse", "spatial")}, "features.pt")
```

Input is RGB `(B,3,224,224)`, floating point in `[0,1]`. Do not apply ImageNet
normalization: the model does it internally. The example transform resizes the
short side to 256, then center-crops 224. Use task-appropriate crops for training.
B/L output 196 patch locations; H/14 outputs 256. The H decoder produces 256x256
pixels from its 16x16 VQ grid, even though its encoder input is 224x224.

`load_stellar` builds only the requested modules and rejects missing essential
weights. `purpose="encode"` and `purpose="reconstruct"` return evaluation-mode
models; `purpose="pretrain"` returns the complete train-mode model. The latter
loads weights, not optimizer state. Freeze parameters explicitly for downstream
training with `.requires_grad_(False)`.

Pin `revision` for repeatability. Downloads go through `huggingface_hub` and are
cached. After an initial download, `local_files_only=True` supports offline use.
On unreliable network filesystems set `HF_HUB_CACHE` to a local directory before
starting Python. Allow roughly 0.5 GB each for B8/B16/B24, 1.4 GB for L16, and
2.7 GB for H16, plus tokenizer and metric weights when needed.

## Reconstruction

```bash
export EXTERNAL=/path/to/external
hf download fun-research/TiTok maskgit-vqgan-imagenet-f16-256.bin \
  --revision ab646ed225080a3acb7c78440a574d7f67f16fa7 \
  --local-dir "$EXTERNAL/PretrainedModels/vqgan/maskgit"
export VQ_MODEL="$EXTERNAL/PretrainedModels/vqgan/maskgit/maskgit-vqgan-imagenet-f16-256.bin"
```

The decoder takes **features**, not an image:

```python
import os

model = load_stellar(purpose="reconstruct", vq_model=os.environ["VQ_MODEL"], revision=revision)
with torch.no_grad():
    features = model.encode(image.cpu())
    pixels = model.reconstruct(features)["reconstruction"]
    pixels_from_factors = model.reconstruct(features["sparse"], features["spatial"])["reconstruction"]
```

Both calls return the same RGB pixels in `[0,1]`. Sparse semantic tokens alone
do not specify the layout. The released VQ reconstruction path uses argmax and
a frozen tokenizer; it is not a differentiable RGB decoder for end-to-end losses.

## Pretraining On ImageNet

Prepare the ImageNet Kaggle layout described in the README. The default recipe
uses a **MAE-initialized ViT-B**, not random initialization. Paper Appendix A.1:

| Backbone | MAE initialization | Epochs | Learning rate | Global batch |
| :--- | :--- | ---: | ---: | ---: |
| B/16 | facebook/vit-mae-base | 150 | 0.00015 | 2048 |
| L/16 | facebook/vit-mae-large | 100 | 0.00005 | 2048 |
| H/14 | facebook/vit-mae-huge | 50 | 0.00005 | 2048 |

```bash
python run.py --config-name stellar mounts.external="$EXTERNAL" \
  trainer.devices=1 datamodule.dataloaders.train.batch_size=16 \
  scratch.seed=42 job_name=b16_local
```

This smaller batch is a local starting point, not a claim to reproduce the
reported numbers. The paper used 16 A100-80GB GPUs with 128 images per GPU.
Specify all changed hyperparameters and preserve the resolved config and logs.

For L/H, override `model.vit_pretrained`, `trainer.max_epochs`, and `optimizer.lr`
according to the table. An architecture's sparse-token count is independent of
its patch size: H16 means 16 sparse tokens on a ViT-H/14 backbone.

### Your Own Unlabeled Images

No class subdirectories, annotation files or ImageNet JSON metadata are needed:

```text
my_images/train/image01.jpg
my_images/train/nested/image02.png
my_images/test/image03.jpg
```

```bash
python run.py --config-name stellar datamodule=image_folder \
  scratch.data_root=/path/to/my_images/train \
  scratch.test_data_root=/path/to/my_images/test \
  mounts.external="$EXTERNAL" \
  datamodule.dataloaders.train.batch_size=16 job_name=custom
```

The dataset recursively finds images in sorted order and returns two global
crops and eight local crops. Labels are not consumed by SSL. Files must be valid
RGB-convertible images; corrupt inputs fail visibly. Olympus holds out 10% of
the training images using the seeded split. Use at least 20 images so validation
is nonempty; the training split still needs full batches. Evaluate learned
features on held-out data to assess custom-domain quality.

`scratch.test_data_root` is used only for explicit test runs or
`scratch.test_after_fit=true`; keep it disjoint from training. For custom SSL,
validation uses stochastic multi-crops and measures the SSL objective, not a
deterministic reconstruction benchmark. Use the metric CLI below for that.

Implement another `Dataset` returning the same dictionary as
[image_folder.py](../src/datasets/image_folder.py) to add archives, streaming, or
domain-specific augmentations. Keep tensors unnormalized and add new Hydra
dataset targets without changing the trainer.

### Random Initialization And Optional Extensions

`model.vit_pretrained=null` builds a random ViT-B. The paper's random-prior
ablation used 150 epochs of EMA warm-up (momentum 0.996, masking 0.6), then 75
epochs of standard training (masking 0.8). It is **not** the default main recipe.
`model.momentum_teacher=true` enables the teacher; a two-stage switch needs a
weights-only transfer with teacher keys removed, not an optimizer resume into a
different architecture. That full two-stage reproduction has not been validated.

`model.do_global_align=true` and cosine `teacher_momentum_final` are optional
extensions, not verified reproductions of the released checkpoints. Constant
teacher momentum is obtained by leaving `teacher_momentum_final=null`.

### Multi-GPU And Multi-Node

```bash
# One machine: Lightning starts one process per GPU.
python run.py --config-name stellar mounts.external="$EXTERNAL" \
  trainer.devices=8 trainer.num_nodes=1 \
  datamodule.dataloaders.train.batch_size=128

# Two machines: run on EACH node, with NODE_RANK=0 and NODE_RANK=1.
torchrun --nnodes=2 --nproc_per_node=8 --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port=29500 \
  run.py --config-name stellar mounts.external="$EXTERNAL" \
  trainer.devices=8 trainer.num_nodes=2 \
  datamodule.dataloaders.train.batch_size=128
```

Use shared dataset/output paths and matching code/environments on every node.
Keep `ddp_find_unused_parameters_true`: the model contains unused pooler and
optional-branch parameters. Global optimizer batch is per-device batch times
devices times nodes times `trainer.accumulate_grad_batches` (add this flag with
`+` if unset). Accumulation does **not** reproduce the larger batch's Sinkhorn
assignment statistics, which are computed per forward pass across ranks.
With the optional EMA teacher, weights are also updated per training forward;
accumulation is therefore not equivalent to one teacher update per large batch.

### Resume And Repeatability

```bash
python run.py --config-name stellar mounts.external="$EXTERNAL" \
  scratch.resume=/path/to/checkpoints/last.ckpt
```

Use the same architecture/data recipe. Lightning restores model, optimizer and
global step; `MomentumScheduleCallback` restores the EMA schedule position.
`trainer.max_epochs` and `trainer.max_steps` are total limits, not additional
work after resume. Increase any exhausted limit when extending a completed run.
For eval configs use `+scratch.resume=...` because those configs do not define
`scratch`. Only load trusted Lightning pickle checkpoints. A downloaded Hub
safetensors file is **not** a resumable training checkpoint.

Seed before dataset splitting; preserve file lists, world size, batch size,
precision, software versions and Hub revisions. The launcher saves resolved
configs and CSV logs under `outputs/stellar/<job_name>`. End-of-epoch checkpoints
are preferable: standard DataLoaders do not restore the exact mid-epoch cursor.
For stricter repeatability add `+trainer.deterministic=true`, recognizing that
some CUDA operations may reject deterministic mode. Cross-hardware bitwise
reproduction and original-paper metric reproduction are not guaranteed.

## Frozen-Feature Evaluation

```bash
python run.py --config-name eval_cls mounts.external="$EXTERNAL" \
  datamodule.dataloaders.train.batch_size=128 +scratch.seed=42
python run.py --config-name eval_recon mounts.external="$EXTERNAL" +scratch.seed=42
python run.py --config-name eval_seg mounts.external="$EXTERNAL" \
  datamodule.dataloaders.train.batch_size=16 +scratch.seed=42
```

These configs load the public B16 encoder and **train a new probe/decoder**.
Classification/segmentation do not require VQGAN. The ADE20K root is
`$EXTERNAL/ImageData/ADEChallengeData2016`, with `images/{training,validation}`
and `annotations/{training,validation}`. The official validation set is the
held-out test set; 10% of training is used for model selection.

For L/H, use `model.model.model=stellar-l16` (classification/reconstruction) or
`model.model_backbone.model=stellar-l16` (segmentation), and set
`model.feature_dim=1024` (L) or `1280` (H). Feature keys are `sparse`, `lowrank`,
and `dense`, respectively. Set `trainer.devices=N` for distributed probing.

To save classification/segmentation heads, set `trainer.enable_checkpointing=true`.
To evaluate a saved complete probe:

```bash
python run.py --config-name eval_cls mode=test mounts.external="$EXTERNAL" \
  +scratch.resume=/path/to/probe.ckpt
```

Without a trained probe checkpoint, `mode=test` evaluates a randomly initialized
head. `+scratch.test_after_fit=true` tests the final trained weights, not a
validation-selected best checkpoint. Do not select hyperparameters on the test
set. Paper Appendix A.2 swept learning rates from 1e-5 to 1e-2 and batch sizes
64 through 8192; a single config is not the entire paper evaluation protocol.

### Reconstruction Metrics

```bash
python -m pip install -r requirements-eval.txt
python -m examples.evaluate_reconstruction --data /path/to/held_out_images \
  --vq-model "$VQ_MODEL" --model stellar-b16 --device cuda \
  --output outputs/reconstruction_metrics.json
```

This streaming, single-device CLI reports TorchMetrics/torch-fidelity Inception
2048-dimensional **rFID** and AlexNet LPIPS. It uses deterministic resize/crop
preprocessing and the released pretraining decoder. H references are resized to
256 to match its output. It is not the separately finetuned B/L decoder from
Table 1, and preprocessing/LPIPS-network choices must match before comparing
numbers. Use the complete held-out set; FID on a handful of images is not
statistically meaningful. Distributed sampling/metric aggregation is not
implemented in this standalone CLI.
