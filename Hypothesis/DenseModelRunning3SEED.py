
from Hypothesis.ProbeTrainLoader import train_dense_probe

dense_results = {}
for seed in [42, 123, 2026]:
    print(f"STARTING DENSE PROBE — SEED {seed}")
    dense_results[seed] = train_dense_probe(
        seed=seed,
        num_epochs=50,
    )


# Summarizing the results of 3 SEED
for seed, result in dense_results.items():
    print(f"SEED {seed}")
    print(f"Best epoch: {result['best_epoch']}")
    print(
        f"Val Dice: "
        f"{result['best_val_dice']:.4f}"
    )

    print("\nTestA")
    print(f"  Dice : {result['testA']['gland_dice']:.4f}")
    print(f"  IoU  : {result['testA']['gland_iou']:.4f}")
    print(f"  mIoU : {result['testA']['miou']:.4f}")

    print("\nTestB")
    print(f"  Dice : {result['testB']['gland_dice']:.4f}")
    print(f"  IoU  : {result['testB']['gland_iou']:.4f}")
    print(f"  mIoU : {result['testB']['miou']:.4f}")
