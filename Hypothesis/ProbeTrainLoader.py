
"""
The code has been runned on Kaggle, use respective ROOT file for dataset
"""
import random
from Hypothesis.RunEpoch import run_epoch
from Hypothesis.DenseModel import dense_model
from Hypothesis.SegmentationProbing import SegmentationProbing
from TestLoader import testA_loader, testB_loader
import numpy as np
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import torch
from Hypothesis.GlasSegmentation import GlaSSegmentationDataset
ROOT = "/kaggle/input/datasets/sani84/glasmiccai2015-gland-segmentation/Warwick_QU_Dataset"

glas_train_aug = GlaSSegmentationDataset(
    root=ROOT, split="train", resolution=224, augment=True)
glas_train_eval = GlaSSegmentationDataset(
    root=ROOT, split="train", resolution=224, augment=False)

"""
We'll be using some specific train, val images and that has to be same
along all the ablation studies
"""

SEED = 42  # For same reproducibility
generator = torch.Generator().manual_seed(SEED)
indices = torch.randperm(len(glas_train_aug), generator=generator).tolist()

train_indices = indices[:70]
val_indices = indices[70:]

print("Train indices:", len(train_indices))
print("Validation indices:", len(val_indices))

print("Train examples:", train_indices[:10])
print("Val examples:", val_indices[:10])


train_subset = Subset(glas_train_aug, train_indices)
val_subset = Subset(glas_train_eval, val_indices)


def probe_train_loader(train_subset, batch_size, shuffle, num_workers, pin_memory):
    probe_train_loader = DataLoader(
        train_subset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    return probe_train_loader


def probe_val_loader(train_subset, batch_size, shuffle, num_workers, pin_memory):
    probe_val_loader = DataLoader(
        val_subset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    return probe_val_loader


"""# probe_train_loader = DataLoader(
#     train_subset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
# probe_val_loader = DataLoader(
#     val_subset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
"""


device = torch.device("cuda" if torch.cuda.is_available() else "gpu")


# Running for seed 42, 123, 2026
def train_dense_probe(seed, num_epochs=50):
    # Training randomness
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(filter(
        lambda p: p.requires_grad, dense_model().parameters()), lr=1e-3, weight_decay=1e-4,)
    criterion = torch.nn.CrossEntropyLoss()

    # Best checkpoint in memory
    best_val_dice = -1.0
    best_epoch = -1
    best_state = None
    history = []

    # Training
    for epoch in range(num_epochs):
        train_metrics = run_epoch(dense_model(), probe_train_loader(
        ), optimizer=optimizer, criterion=criterion, device=device,)
        val_metrics = run_epoch(dense_model(), probe_val_loader(
        ), optimizer=None, criterion=criterion, device=device,)

        history.append({
            "epoch": epoch + 1,
            "train_dice": train_metrics["gland_dice"],
            "train_miou": train_metrics["miou"],
            "val_dice": val_metrics["gland_dice"],
            "val_miou": val_metrics["miou"],
        })

        print(
            f"[Seed {seed}] "
            f"Epoch {epoch+1:02d}/{num_epochs} | "
            f"Train Dice: {train_metrics['gland_dice']:.4f} | "
            f"Val Dice: {val_metrics['gland_dice']:.4f}"
        )

        if val_metrics["gland_dice"] > best_val_dice:
            best_val_dice = val_metrics["gland_dice"]
            best_epoch = epoch + 1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in dense_model.state_dict().items()
            }

    # Restore best model
    dense_model().load_state_dict(best_state)
    dense_model().eval()

    # TestA
    testA_metrics = run_epoch(dense_model(), testA_loader(
    ), optimizer=None, criterion=criterion, device=device,)
    # TestB
    testB_metrics = run_epoch(dense_model(), testB_loader(
    ), optimizer=None, criterion=criterion, device=device,)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "testA": testA_metrics,
        "testB": testB_metrics,
        "history": history,
    }
