
import random
import numpy as np
import torch
from train.RunEpoch import run_epoch

history = []


def train_spatial_probe(seed, model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    # Training randomness
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Best checkpoint in memory
    best_val_dice = -1.0
    best_epoch = -1
    best_state = None

    for epoch in range(num_epochs):
        train_metrics = run_epoch(model, train_loader, optimizer=optimizer,
                                  criterion=criterion, device=device,)

        val_metrics = run_epoch(model, val_loader, optimizer=None,
                                criterion=criterion, device=device,)

        history.append({
            "epoch": epoch + 1,
            "train_dice": train_metrics["gland_dice"],
            "train_miou": train_metrics["miou"],
            "val_dice": val_metrics["gland_dice"],
            "val_miou": val_metrics["miou"], })

        print(
            f"[Seed {seed}] "
            f"Epoch {epoch+1:02d}/{num_epochs} | "
            f"Train Dice: {train_metrics['gland_dice']:.4f} | "
            f"Val Dice: {val_metrics['gland_dice']:.4f}")

        if val_metrics["gland_dice"] > best_val_dice:
            best_val_dice = val_metrics["gland_dice"]
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()}

    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "history": history, }
