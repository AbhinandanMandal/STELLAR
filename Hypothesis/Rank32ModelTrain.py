

import torch
from Hypothesis.utils.DataLoader import train_loader, val_loader
from Hypothesis.Rank32Model import rank32_model
from Hypothesis.train.TrainSpatialProbe import train_spatial_probe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.AdamW(filter(
    lambda p: p.requires_grad, rank32_model().parameters()), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()


rank32_results = {}
for seed in [42, 123, 2026]:
    print("\n" + "=" * 70)
    print(f"STARTING RANK-32 PROBE — SEED {seed}")
    print("=" * 70)
    rank32_results[seed] = train_spatial_probe(seed, rank32_model(
    ), train_loader, val_loader, optimizer, criterion, device, 50)

for seed, result in rank32_results.items():
    print("\n" + "=" * 60)
    print(f"RANK-32 — SEED {seed}")
    print("=" * 60)
    print(f"Best epoch : {result['best_epoch']}")
    print(f"Val Dice   : {result['best_val_dice']:.4f}")
