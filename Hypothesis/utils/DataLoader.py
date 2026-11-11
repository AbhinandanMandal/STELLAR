
"""
DataLoader for hypothesis testing
The specific train, testA, testB is specific to GlaS Segmentation Dataset also

"""

from torch.utils.data import DataLoader
from Hypothesis.utils.GlasSegmentation import GlaSSegmentationDataset

# ROOT path of GlaS dataset
ROOT = "/kaggle/input/datasets/sani84/glasmiccai2015-gland-segmentation/Warwick_QU_Dataset"

# For training
GlaS_train_aug = GlaSSegmentationDataset(
    root=ROOT, split="train", resolution=224, augment=True,)
# For validation
GlaS_train_eval = GlaSSegmentationDataset(
    root=ROOT, split="train", resolution=224, augment=False,)
# For testing
GlaS_testA_eval = GlaSSegmentationDataset(
    root=ROOT, split="testA", resolution=224, augment=False)
GlaS_testB_eval = GlaSSegmentationDataset(
    root=ROOT, split="testB", resolution=224, augment=False)


def train_loader(Glas_train_aug, batch_size=4, shuffle=True, num_workers=2, pin_memory=True):
    train_loader = DataLoader(GlaS_train_aug, batch_size=4,
                              shuffle=True, num_workers=2, pin_memory=True,)
    return train_loader


def val_loader(Glas_train_eval, batch_size=4, shuffle=False, num_workers=2, pin_memory=True):
    val_loader = DataLoader(GlaS_train_eval, batch_size=4,
                            shuffle=False, num_workers=2, pin_memory=True,)
    return val_loader


def testA_loader(Glas_testA_eval, batch_size=4, shuffle=False, num_workers=2, pin_memory=True):
    testA_loader = DataLoader(GlaS_testA_eval, batch_size=4,
                              shuffle=False, num_workers=2, pin_memory=True,)
    return testA_loader


def testB_loader(Glas_testB_eval, batch_size=4, shuffle=False, num_workers=2, pin_memory=True):
    testB_loader = DataLoader(GlaS_testB_eval, batch_size=4,
                              shuffle=False, num_workers=2, pin_memory=True,)
    return testB_loader
