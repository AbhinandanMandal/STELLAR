"""Measure released-decoder reconstruction, with an explicit local protocol."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from examples.common import REVISION, image_paths, read_image
from load_stellar import load_stellar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--vq-model", required=True)
    parser.add_argument("--model", default="stellar-b16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default="outputs/reconstruction_metrics.json")
    args = parser.parse_args()
    paths = image_paths(args.data)
    if args.batch_size < 1 or len(paths) < 2:
        parser.error("Require positive batch size and at least two images")
    torch.manual_seed(42)
    model = load_stellar(args.model, purpose="reconstruct", vq_model=args.vq_model,
                         revision=REVISION, device=args.device)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(args.device).set_dtype(torch.float64)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(args.device)
    with torch.no_grad():
        for offset in range(0, len(paths), args.batch_size):
            images = torch.stack([read_image(path) for path in paths[offset:offset + args.batch_size]]).to(args.device)
            reconstructed = model.reconstruct(model.encode(images))["reconstruction"]
            reference = functional.interpolate(images, size=reconstructed.shape[-2:],
                                                mode="bicubic", align_corners=False).clamp(0, 1)
            fid.update(reference, real=True)
            fid.update(reconstructed, real=False)
            lpips.update(reconstructed, reference)
    result = dict(model=args.model, revision=REVISION, images=len(paths),
                  rfid=float(fid.compute()), lpips_alex=float(lpips.compute()),
                  preprocessing="RGB [0,1]; bicubic resize short side 256; center crop 224",
                  reference_size=list(reconstructed.shape[-2:]),
                  fid_implementation="torchmetrics / torch-fidelity Inception 2048",
                  decoder="released pretraining decoder, not a separately finetuned probe")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()