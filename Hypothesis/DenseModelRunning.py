

import torch
from Hypothesis.RunEpoch import run_epoch
from Hypothesis.ProbeTrainLoader import probe_train_loader, probe_val_loader
from Hypothesis.DenseModel import dense_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.AdamW(filter(
    lambda p: p.requires_grad, dense_model().parameters()), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()


num_epochs = 50  # Tunable metric, getting stablility at 50
best_val_dice = -1.0
history = []

for epoch in range(num_epochs):

    train_metrics = run_epoch(dense_model(), probe_train_loader(
    ), optimizer=optimizer, criterion=criterion, device=device,)
    val_metrics = run_epoch(dense_model(), probe_val_loader(
    ), optimizer=None, criterion=criterion, device=device,)

    row = {
        "epoch": epoch + 1,

        "train_loss": train_metrics["loss"],
        "train_dice": train_metrics["gland_dice"],
        "train_iou": train_metrics["gland_iou"],
        "train_miou": train_metrics["miou"],

        "val_loss": val_metrics["loss"],
        "val_dice": val_metrics["gland_dice"],
        "val_iou": val_metrics["gland_iou"],
        "val_miou": val_metrics["miou"],
    }

    history.append(row)
    print(
        f"Epoch {epoch+1:02d}/{num_epochs} | "
        f"Train Loss: {train_metrics['loss']:.4f} | "
        f"Train Dice: {train_metrics['gland_dice']:.4f} | "
        f"Train mIoU: {train_metrics['miou']:.4f} | "
        f"Val Loss: {val_metrics['loss']:.4f} | "
        f"Val Dice: {val_metrics['gland_dice']:.4f} | "
        f"Val mIoU: {val_metrics['miou']:.4f}"
    )

    if val_metrics["gland_dice"] > best_val_dice:
        best_val_dice = val_metrics["gland_dice"]
        torch.save(dense_model.state_dict(),
                   "/Hypothesis/working/stellar_dense_glas_probe_best.pth")
        print(f" Saved best checkpoint | (Val Dice = {best_val_dice:.4f})")
