
import torch 
from Hypothesis.STELLAR import STELLARModel
from Hypothesis.SegmentationProbing import SegmentationProbing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = STELLARModel(
    num_sparse_tokens=cfg["num_sparse_tokens"],
    num_decoder_layers=cfg["num_decoder_layers"],
    spatial_temp=cfg["spatial_temp"],
    vit_pretrained=cfg["backbone"],
    do_recon=False,
    do_clustering=False,
    vq_model=None,
)
model = model.to(device)
model.eval()


def dense_model(model_backbone, is_baseline, feature_key, feature_dim, num_classes, freeze_backbone, freeze_model, resize_output):
    dense_model = SegmentationProbing(
        model_backbone=model,
        is_baseline=False,
        feature_key="dense",
        feature_dim=768,
        num_classes=2,
        freeze_backbone=True,
        freeze_model=False,
        resize_output=(224, 224),
    ).to(device)
    return dense_model 

