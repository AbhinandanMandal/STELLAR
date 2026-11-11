
"""Creating GlasSegmentationDataset class"""
import os
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


class GlaSSegmentationDataset(Dataset):
    def __init__(self, root, split="train", resolution=224, crop_scale=0.5, augment=False,):
        self.root = root
        self.split = split
        self.resolution = resolution
        self.crop_scale = crop_scale
        self.augment = augment

        # Select images belonging to the official split
        if split == "train":
            image_files = [f for f in os.listdir(root)if f.startswith("train_")
                           and f.endswith(".bmp") and not f.endswith("_anno.bmp")]

        elif split == "testA":
            image_files = [f for f in os.listdir(root) if f.startswith("testA_")
                           and f.endswith(".bmp") and not f.endswith("_anno.bmp")]

        elif split == "testB":
            image_files = [f for f in os.listdir(root) if f.startswith("testB_")
                           and f.endswith(".bmp") and not f.endswith("_anno.bmp")]

        else:
            raise ValueError(
                f"Unknown split: {split}. Expected train, testA or testB.")

        # Build image/mask pairs
        self.samples = []
        for image_file in sorted(image_files):
            mask_file = image_file.replace(".bmp", "_anno.bmp")
            image_path = os.path.join(root, image_file)
            mask_path = os.path.join(root, mask_file)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Missing mask for {image_file} | {mask_path}")
            self.samples.append((image_path, mask_path))
        print(
            f"GlaS {split} | {len(self.samples)} image/mask pairs | augment={self.augment}")

    def __len__(self):
        return len(self.samples)

    """
    Instance mask to binary semantic mask
    
    We'll be converting instance mask to binary semantic mask
    In original GlaS dataset, 
    0      = background
    1..32  = individual gland instances

    In binary semantic masking, 
    0 = background
    1 = gland
    """

    def _convert_mask(self, mask):
        mask = np.array(mask)
        mask = (mask > 0).astype(np.uint8)
        return torch.from_numpy(mask).long()

    # Training augmentation

    def _train_transform(self, image, mask):
        width, height = TF.get_image_size(image)

        # Random crop scale
        scale = torch.empty(1).uniform_(self.crop_scale, 1.0).item()

        crop_h = int(height * scale)
        crop_w = int(width * scale)

        crop_h = max(1, min(crop_h, height))
        crop_w = max(1, min(crop_w, width))

        # Random crop position
        if height == crop_h:
            top = 0
        else:
            top = torch.randint(0, height - crop_h + 1, (1,)).item()

        if width == crop_w:
            left = 0
        else:
            left = torch.randint(0, width - crop_w + 1, (1,)).item()

        # Image: bicubic
        image = TF.resized_crop(image, top, left, crop_h, crop_w,
                                [self.resolution, self.resolution],
                                interpolation=InterpolationMode.BICUBIC,)

        # Mask: nearest-neighbor
        mask = TF.resized_crop(mask.unsqueeze(0).float(), top, left, crop_h, crop_w,
                               [self.resolution, self.resolution],
                               interpolation=InterpolationMode.NEAREST,).squeeze(0).long()

        # Random horizontal flip
        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask

    # Deterministic validation/test transform
    def _eval_transform(self, image, mask):
        crop_res = int(self.resolution * 256 / 224)

        # Image: bicubic
        image = TF.resize(
            image,
            [crop_res, crop_res],
            interpolation=InterpolationMode.BICUBIC,
        )

        # Mask: nearest
        mask = TF.resize(
            mask.unsqueeze(0).float(),
            [crop_res, crop_res],
            interpolation=InterpolationMode.NEAREST,
        ).squeeze(0).long()

        # Center crop
        image = TF.center_crop(image, [self.resolution, self.resolution])
        mask = TF.center_crop(mask, [self.resolution, self.resolution])
        return image, mask

    # Get item
    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]

        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = self._convert_mask(mask)  # Instance to Semantic conversion

        # Apply transform
        if self.augment:
            image, mask = self._train_transform(image, mask)  # For training
        else:
            image, mask = self._eval_transform(
                image, mask)  # For testing/validation
        image = TF.to_tensor(image)  # Image to Tensor transform
        return {
            "image": image.clamp(0, 1),
            "labels": mask,
            "filename": idx,
        }
