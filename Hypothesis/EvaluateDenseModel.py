
from Hypothesis.DenseModel import dense_model
from Hypothesis.TestLoader import testA_loader, testB_loader
from Hypothesis.RunEpoch import run_epoch

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.AdamW(filter(
    lambda p: p.requires_grad, dense_model().parameters()), lr=1e-3, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss()

testA_metrics = run_epoch(dense_model(), testA_loader(
), optimizer=None, criterion=criterion, device=device)
print("\n======= GlaS TestA Dense Model Testing Performance ========")
for key, value in testA_metrics.items():
    print(f"{key:20s}: {value:.4f}")

testB_metrics = run_epoch(dense_model(), testB_loader(
), optimizer=None, criterion=criterion, device=device,)
print("\n=====-= GlaS TestB Dense Model Testing Performance =======")
for key, value in testB_metrics.items():
    print(f"{key:20s}: {value:.4f}")
