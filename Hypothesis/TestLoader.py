
from Hypothesis.GlasSegmentation import GlaSSegmentationDataset
from torch.utils.data import DataLoader
ROOT = "/kaggle/input/datasets/sani84/glasmiccai2015-gland-segmentation/Warwick_QU_Dataset"

testA_dataset = GlaSSegmentationDataset(
    root=ROOT, split="testA", resolution=224, augment=False)
testB_dataset = GlaSSegmentationDataset(
    root=ROOT, split="testB", resolution=224, augment=False)


def testA_loader(testA_dataset, batch_size, shuffle, pin_memory,):
    testA_loader = DataLoader(
        testA_dataset, batch_size=4, shuffle=False, pin_memory=True)
    return testA_loader


def testB_loader(testB_dataset, batch_size, shuffle, pin_memory,):
    testB_loader = DataLoader(
        testB_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    return testB_loader

# testA_loader = DataLoader(testA_dataset, batch_size=4, shuffle=False, pin_memory=True)
# testB_loader = DataLoader(testB_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
