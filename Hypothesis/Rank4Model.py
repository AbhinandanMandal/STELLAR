

import torch
import torch.nn as nn
from Hypothesis.RankTruncationBackbone import RankTruncatedBackbone
from Hypothesis.STELLAR import STELALR_model
from Hypothesis.utils.SegmentationProbing import SegmentationProbing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = STELALR_model()
model = model.to(device)
model.eval()


def rank4_model(model_backbone, is_baseline, feature_key, feature_dim, num_classes, freeze_backbone, freeze_model, resize_output):
    rank4_backbone = RankTruncatedBackbone(model, rank=4)
    rank4_model = SegmentationProbing(
        model_backbone=rank4_backbone,
        is_baseline=False,
        feature_key="dense",
        feature_dim=768,
        num_classes=2,
        freeze_backbone=True,
        freeze_model=False,
        resize_output=(224, 224),).to(device)
    return rank4_model
