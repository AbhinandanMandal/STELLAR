

import torch
from Hypothesis.utils.DataLoader import train_loader, val_loader
from Hypothesis.DenseRankModel import dense_model
from Hypothesis.train.TrainDenseProbe import train_dense_probe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.AdamW(filter(
    lambda p: p.requires_grad, dense_model().parameters()), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# For storing dense model performance
dense_results = {}
for seed in [42, 123, 2026]:
    print("\n" + "=" * 70)
    print(f"STARTING DENSE PROBE — SEED {seed}")
    print("=" * 70)
    dense_results[seed] = train_dense_probe(seed, dense_model(
    ), train_loader(), val_loader(), optimizer, criterion, device, 50)


for seed, result in dense_results.items():
    print("\n" + "=" * 60)
    print(f"SEED {seed}")
    print("=" * 60)
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Val Dice | {result['best_val_dice']:.4f}")
