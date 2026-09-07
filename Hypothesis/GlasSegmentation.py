
# Creating GlasSegmentationDataset class
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
        """ 
        In the original GlaS dataset, images were predefined splitted into
        train_, .bmp. _anno.bmp structure
        We'll utilize the strcutre to create train, testA, testB folder 
        accordingly
        """
        if split == "train":
            image_files = [
                f for f in os.listdir(root)
                if f.startswith("train_")
                and f.endswith(".bmp")
                and not f.endswith("_anno.bmp")
            ]

        elif split == "testA":
            image_files = [
                f for f in os.listdir(root)
                if f.startswith("testA_")
                and f.endswith(".bmp")
                and not f.endswith("_anno.bmp")
            ]

        elif split == "testB":
            image_files = [
                f for f in os.listdir(root)
                if f.startswith("testB_")
                and f.endswith(".bmp")
                and not f.endswith("_anno.bmp")
            ]

        else:
            raise ValueError(
                f"Unknown split: {split}. "
                "Expected train, testA or testB."
            )

        # Build image/mask pairs
        # Where .bmp is image and _anno.bmp is masked image
        self.samples = []
        for image_file in sorted(image_files):
            mask_file = image_file.replace(".bmp", "_anno.bmp")
            image_path = os.path.join(root, image_file)
            mask_path = os.path.join(root, mask_file)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Missing mask for {image_file}: "
                    f"{mask_path}"
                )
            self.samples.append((image_path, mask_path))
        print(
            f"GlaS {split}: "
            f"{len(self.samples)} image/mask pairs | "
            f"augment={self.augment}"
        )

    def __len__(self):
        return len(self.samples)

    # Creating binary segmentation mask from instance mask

    def _convert_mask(self, mask):
        mask = np.array(mask)

        """ 
        Original:
        0 = background, 1..32 = individual gland instance

        Semantic:
        0 = background, 1 = gland
        """
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

        # Bicubic image resize
        image = TF.resized_crop(image, top, left, crop_h, crop_w,
                                [self.resolution, self.resolution],
                                interpolation=InterpolationMode.BICUBIC,
                                )

        # Masking nearest neighbors
        mask = TF.resized_crop(mask.unsqueeze(0).float(), top, left, crop_h, crop_w,
                               [self.resolution, self.resolution],
                               interpolation=InterpolationMode.NEAREST,
                               ).squeeze(0).long()

        # Random horizontal flip
        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        return image, mask

    # Test/Validation augmentation
    def _eval_transform(self, image, mask):
        crop_res = int(self.resolution * 256 / 224)

        # Bicubic image resize
        image = TF.resize(image, [crop_res, crop_res],
                          interpolation=InterpolationMode.BICUBIC,)

        # Masking
        mask = TF.resize(mask.unsqueeze(0).float(), [crop_res, crop_res],
                         interpolation=InterpolationMode.NEAREST,).squeeze(0).long()

        # Center crop
        image = TF.center_crop(image, [self.resolution, self.resolution])
        mask = TF.center_crop(mask, [self.resolution, self.resolution])
        return image, mask

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")  # Loading image
        mask = Image.open(mask_path).convert("L")  # Loading mask
        mask = self._convert_mask(mask)  # Instance to semantic masking

        if self.augment:
            image, mask = self._train_transform(
                image, mask)  # Training transform
        else:
            image, mask = self._eval_transform(
                image, mask)  # Validation/Test transform
        image = TF.to_tensor(image)
        return {
            "image": image.clamp(0, 1),
            "labels": mask,
            "filename": idx,
        }
