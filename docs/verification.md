# Verification Ledger

Audit date: 2026-09-10. These checks distinguish executable workflows from
scientific reproduction. They are not new benchmark results.

## Environment

Linux; Python 3.10.19; one NVIDIA A100 80GB; PyTorch 2.10.0;
torchvision 0.25.0; Transformers 5.2.0; Lightning 2.6.1; Kornia 0.8.2;
huggingface-hub 1.4.1; safetensors 0.7.0; Olympus image components 0.0.79;
torch-fidelity 0.3.0.

STELLAR Hub revision: `6794be9a20fb9d3944c5bcc6003512a347fa518f`.
TiTok tokenizer revision: `ab646ed225080a3acb7c78440a574d7f67f16fa7`.

## Executed Checks

| Check | Result | Scope |
| :--- | :--- | :--- |
| Public Hub B8/B16/B24/L16/H16 | Passed | Actual downloads, required-weight validation, finite encoder outputs and expected shapes |
| B16 factorization | Passed | Spatial rows sum to one; `lowrank == spatial @ sparse` |
| B16 released reconstruction | Passed | Public VQGAN weights; two decoder calling forms agree; RGB 224x224 in [0,1] |
| B16 full pretraining | Passed | Released full weights; BF16 CUDA forward/backward; finite loss and sparse-token gradients |
| Label-free custom images | Passed | Recursive dataset, crop shapes, collation, finite SSL loss/backward |
| Local Lightning training/resume | Passed | Tiny model optimizer step 1; saved state resumed to step 2 |
| Distributed launch | Passed | `torchrun`, two CPU ranks, Gloo, EMA teacher and distributed SSL, two steps |
| Main Hydra recipes | Passed | All four resolve without AMLT environment variables; evaluator construction |
| Reconstruction probe evaluator | Passed | Hub encoder, pixel reconstruction probe, prediction and loss metrics |
| Classification/segmentation probes | Passed | Actual Hub-backed configs, synthetic-label loss/backward, frozen backbone gradients, evaluator metrics |
| Reconstruction metric CLI | Passed | Actual Inception and AlexNet weights; rFID/LPIPS calculation and JSON output on two images |
| Frozen probe modes | Passed | Classification, reconstruction and segmentation backbones stay in eval mode during frozen probing; fine-tuning still enables train mode |
| Lightweight imports | Passed | Cached Hub listing avoids PyTorch; B16 encoding avoids VQGAN imports; pinned B16 CUDA reconstruction still works |
| Dataset regressions | Passed | Image-like directory names excluded; missing custom root reported clearly; ImageNet SSL skips the discarded crop and preserves crop shapes |

The L16 download initially failed with an I/O error on a network-mounted cache;
L16/H16 both succeeded with `HF_HUB_CACHE` on local disk. This was not a state-dict
compatibility failure.

## Repeat Checks

```bash
python -m pip install -r requirements-inference.txt
# Also install the training dependencies for Hydra/Olympus config tests.
OMP_NUM_THREADS=2 python -m unittest discover -s tests -v

# Opt-in public download check (roughly 5.5 GB for all five models).
STELLAR_TEST_HUB=1 OMP_NUM_THREADS=2 python -m unittest discover -s tests -v

# A small labeled-free folder with at least 20 images is sufficient here.
python run.py --config-name smoke \
  scratch.data_root=/path/to/images scratch.test_data_root=/path/to/images

OMP_NUM_THREADS=2 torchrun --standalone --nproc_per_node=2 \
  run.py --config-name smoke \
  scratch.data_root=/path/to/images scratch.test_data_root=/path/to/images \
  trainer.devices=2 trainer.strategy=ddp_find_unused_parameters_true
```

The smoke recipe intentionally uses a tiny random ViT and disables
reconstruction. It tests the launcher/SSL/EMA/distribution, not pretrained
accuracy. Use separate real train/test folders outside smoke testing.

## Not Established

- A fresh-machine install of every legacy training dependency has not been
  executed; runtime tests used the existing STELLAR environment.
- Full ImageNet pretraining or the hyperparameter sweeps in Appendix A.2 were
  not rerun. The original benchmark numbers remain paper results.
- Two physical GPUs or multiple nodes were not available; CPU DDP is not a
  substitute for NCCL, cluster networking, or throughput validation.
- Full classification and ADE20K benchmark datasets, selected probe checkpoints,
  and decoder-finetuned B/L artifacts were not evaluated end to end here.
- No captioning/VQA benchmark was run and no trained VLM adapter ships.
- RAE-style image generation has not been experimentally evaluated. No
  image-generation implementation, trained prior or quality claim is provided.
- Mid-epoch DataLoader cursor/RNG replay and bitwise cross-device reproducibility
  are not guaranteed. Prefer end-of-epoch resumes and record the full protocol.
- Removing the unused ImageNet pretraining crop changes the random-number sequence
  relative to earlier code, even with the same seed; the returned augmentation recipe
  is unchanged. Record the code revision with each training run.
- No temporal/action model, object identity guarantee, or world-model result is
  provided by the image-only released weights.

The standalone metric CLI's deterministic crop, AlexNet LPIPS and released
decoder define a new explicit *measurement protocol*, not automatic equivalence
to every Table 1 number. Tiny-set FID values must not be advertised as benchmarks.