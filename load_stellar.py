"""Load the published STELLAR weights without downloading a second backbone."""

import json

from huggingface_hub import hf_hub_download


REPO_ID = "microsoft/STELLAR"


def model_index(repo_id=REPO_ID, revision=None, local_files_only=False):
    """Read the release manifest; use a commit revision for reproducible runs."""
    path = hf_hub_download(repo_id, "config.json", revision=revision,
                           local_files_only=local_files_only)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["models"]


def list_models(**kwargs):
    return list(model_index(**kwargs))


def load_stellar(model="stellar-b16", purpose="encode", vq_model=None,
                 repo_id=REPO_ID, device="cpu", revision=None,
                 local_files_only=False):
    """Load an encoder, reconstruction model, or full pretraining model.

    Inputs to ``encode`` are RGB tensors in [0, 1], without normalization.
    Reconstruction requires the external MaskGIT-VQGAN weights. Only pretraining
    returns a model in training mode. All required STELLAR tensors must load.
    """
    if purpose not in ("encode", "reconstruct", "pretrain"):
        raise ValueError("purpose must be encode, reconstruct, or pretrain")
    index = model_index(repo_id, revision, local_files_only)
    if model not in index:
        raise ValueError(f"Unknown model {model!r}; choose from {list(index)}")
    config = index[model]
    if purpose != "encode" and config["recon_type"] == "vq" and vq_model is None:
        raise ValueError("Pass vq_model=<MaskGIT-VQGAN weights> for reconstruction/pretraining")
    from safetensors.torch import load_file
    from src.models.stellar_model import STELLARModel

    dimensions = {768: (12, 12), 1024: (24, 16), 1280: (32, 16)}
    layers, heads = dimensions[config["embed_dim"]]
    net = STELLARModel(
        num_sparse_tokens=config["num_sparse_tokens"],
        num_decoder_layers=config["num_decoder_layers"],
        spatial_temp=config["spatial_temp"],
        do_cls=config["do_cls"],
        num_clusters=config["num_clusters"],
        do_recon=purpose != "encode",
        do_clustering=purpose == "pretrain",
        vq_model=vq_model if purpose != "encode" else None,
        vit_config=dict(hidden_size=config["embed_dim"],
                        intermediate_size=4 * config["embed_dim"],
                        num_hidden_layers=layers, num_attention_heads=heads,
                        patch_size=config["patch_size"],
                        image_size=config["image_size"], layer_norm_eps=1e-12),
    )
    path = hf_hub_download(repo_id, config["weights"], revision=revision,
                           local_files_only=local_files_only)
    missing, unexpected = net.load_state_dict(load_file(path), strict=False)
    missing = [key for key in missing if not key.startswith("tokenizer.")]
    skipped = ("decoder.", "decoder_proj.", "reconstruction_head.",
               "clustering_head.", "cls_cluster_head.") if purpose == "encode" else (
                   ("clustering_head.", "cls_cluster_head.") if purpose == "reconstruct" else ())
    unexpected = [key for key in unexpected if not key.startswith(skipped)]
    if missing or unexpected:
        raise RuntimeError(f"Incompatible checkpoint: missing={missing[:8]}, unexpected={unexpected[:8]}")
    net.to(device)
    net.train(purpose == "pretrain")
    return net