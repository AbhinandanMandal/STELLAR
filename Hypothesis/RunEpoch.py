
""" 
Broiler plate code for model running
"""
import torch
from Hypothesis.SegmentationMetrics import segmentation_metrics


def run_epoch(model, loader, optimizer=None, criterion=None, device="cuda",):

    is_training = optimizer is not None
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model({"image": images})
            logits = outputs["predictions"]
            loss = criterion(logits, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = segmentation_metrics(all_preds, all_labels)

    metrics["loss"] = (total_loss / len(loader.dataset))
    return metrics
