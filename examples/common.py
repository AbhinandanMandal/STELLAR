"""Shared release revision and image preprocessing for runnable examples."""

from pathlib import Path

from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode


REVISION = "6794be9a20fb9d3944c5bcc6003512a347fa518f"


def read_image(path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    return transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
    ])(image)


def image_paths(root):
    paths = sorted(path for path in Path(root).expanduser().rglob("*")
                   if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                   and path.is_file())
    if not paths:
        raise ValueError(f"No images found under {root}")
    return paths