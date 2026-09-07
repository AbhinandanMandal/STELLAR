
"""
We've used STELLAR directly from github download
For our testcases, we've download it and run it on kaggle


code run on huggingface
------------------------
import json
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from src.models.stellar_model import STELLARModel # Loading the STELLAR-B16 model

repo = "microsoft/STELLAR"
config_path = hf_hub_download(
    repo_id=repo,
    filename="config.json"
)
with open(config_path, "r") as f:
    config = json.load(f)
print(config["models"].keys()) # Looking for all the STELLAR versions


# We'll be working with STELLAR-B16 model
cfg = config["models"]["stellar-b16"] 

# Then we've utilized cgf for model building as, 
model = STELLARModel(
    num_sparse_tokens=cfg["num_sparse_tokens"],
    num_decoder_layers=cfg["num_decoder_layers"],
    spatial_temp=cfg["spatial_temp"],
    vit_pretrained=cfg["backbone"],
    do_recon=False,
    do_clustering=False,
    vq_model=None,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
"""

#  For github upload part for soft code
# It is required to first download model from official STELLAT github source


def STELLARModel(num_sparse_tokens, num_decoder_layers, spatial_temp, vit_pretrained, do_recon, do_clustring, vq_model):
    return
