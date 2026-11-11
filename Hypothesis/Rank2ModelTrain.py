

import torch
from Hypothesis.utils.DataLoader import train_loader, val_loader
from Hypothesis.Rank2Model import rank2_model
from Hypothesis.train.TrainSpatialProbe import train_spatial_probe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.AdamW(filter(
    lambda p: p.requires_grad, rank2_model().parameters()), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()


rank2_results = {}
for seed in [42, 123, 2026]:
    print("\n" + "=" * 70)
    print(f"STARTING RANK-2 PROBE — SEED {seed}")
    print("=" * 70)
    rank2_results[seed] = train_spatial_probe(seed, rank2_model(
    ), train_loader, val_loader, optimizer, criterion, device, 50)

for seed, result in rank2_results.items():
    print("\n" + "=" * 60)
    print(f"RANK-2 — SEED {seed}")
    print("=" * 60)
    print(f"Best epoch : {result['best_epoch']}")
    print(f"Val Dice   : {result['best_val_dice']:.4f}")

