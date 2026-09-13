"""Label-free, recursive image folders using STELLAR's multi-crop contract."""

from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class ImageFolderSSL(Dataset):
    def __init__(self, root, resolution=224, num_global_views=2, num_local_views=8):
        if num_global_views < 2 or num_local_views < 1:
            raise ValueError("Need at least two global views and one local view")
        if root is None:
            raise ValueError("Set root to an image directory (scratch.data_root in the training config)")
        self.samples = sorted(path for path in Path(root).expanduser().rglob("*")
                              if path.suffix.lower() in
                              {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
                              and path.is_file())
        if not self.samples:
            raise ValueError(f"No images found under {root}")
        self.num_global_views = num_global_views
        self.num_local_views = num_local_views
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(resolution, scale=(0.36, 1.0),
                                         interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        ])
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=(0.06, 0.36),
                                         interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        global_views = torch.stack([self.global_transform(image)
                                    for _ in range(self.num_global_views)])
        local_views = torch.stack([self.local_transform(image)
                                   for _ in range(self.num_local_views)])
        return {"image": global_views[0], "labels": global_views[0],
                "global_views": global_views, "local_views": local_views,
                "filename": str(path)}