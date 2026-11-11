
""" 
Segmentation metrics contains Gland Dice, IoU, Accuracy
"""
import numpy as np


def segmentation_metrics(pred, target, num_classes=2):
    pred = pred.detach().cpu().numpy().reshape(-1)
    target = target.detach().cpu().numpy().reshape(-1)

    metrics = {}
    ious = []

    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()

        if union == 0:
            iou = np.nan
        else:
            iou = intersection / union
        ious.append(iou)

    # Background + gland
    metrics["background_iou"] = ious[0]
    metrics["gland_iou"] = ious[1]
    valid_ious = [x for x in ious if not np.isnan(x)]
    metrics["miou"] = np.mean(valid_ious)

    # Gland Dice
    pred_gland = pred == 1
    target_gland = target == 1

    intersection = np.logical_and(pred_gland, target_gland).sum()

    pred_area = pred_gland.sum()
    target_area = target_gland.sum()
    denominator = pred_area + target_area

    if denominator == 0:
        dice = 1.0
    else:
        dice = (2.0 * intersection / denominator)

    metrics["gland_dice"] = dice
    metrics["pixel_accuracy"] = (pred == target).mean()
    return metrics
