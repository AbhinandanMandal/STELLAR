
# Following operations has been done on kaggle cloud
"""
!git clone -q https://github.com/microsoft/STELLAR.git
!ls -la /kaggle/working/STELLAR
%cd /kaggle/working/STELLAR
!python -m pip install -r requirements-inference.txt
"""

# def STELLARModel(num_sparse_tokens, num_decoder_layers, spatial_temp, vit_pretrained, do_recon, do_clustring, vq_model):
#     return

import sys
import torch
from load_stellar import load_stellar

# Change with respect to kaggle path
sys.path.insert(0, "/kaggle/working/STELLAR")
device = "cuda" if torch.cuda.is_available() else "cpu"


def STELALR_model():
    model = load_stellar(
        "stellar-b16", revision="--key--", device=device,)
    return model
