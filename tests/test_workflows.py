import tempfile
import unittest
import os
import json
from pathlib import Path
from unittest.mock import Mock

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader

from src.datasets.image_folder import ImageFolderSSL
from src.losses.stellar_loss import STELLAR_Loss
from src.models.stellar_model import STELLARModel


ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("STELLAR_TEST_HUB") == "1", "Set STELLAR_TEST_HUB=1 to download public weights")
    def test_public_hub(self):
        from examples.common import REVISION
        from load_stellar import load_stellar

        device = "cuda" if torch.cuda.is_available() else "cpu"
        for name, count, width, patches in [
            ("stellar-b16", 16, 768, 196), ("stellar-b8", 8, 768, 196),
            ("stellar-b24", 24, 768, 196), ("stellar-l16", 16, 1024, 196),
            ("stellar-h16", 16, 1280, 256),
        ]:
            model = load_stellar(name, revision=REVISION, device=device)
            with torch.no_grad():
                features = model.encode(torch.rand(1, 3, 224, 224, device=device))
            self.assertEqual(features["sparse"].shape, (1, count, width))
            self.assertEqual(features["spatial"].shape, (1, patches, count))
            self.assertTrue(torch.isfinite(features["lowrank"]).all())
            del model, features

    def test_configs(self):
        with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
            for name in ("stellar", "eval_cls", "eval_recon", "eval_seg"):
                config = compose(config_name=name)
                OmegaConf.to_container(config, resolve=True)
                instantiate(config.evaluator)
                instantiate(config.olympus_checkpoint,
                            _target_="azureml.acft.image.components.olympus.app.olympus_config._OlympusCheckpoint")

    def test_frozen_probe_modes(self):
        from src.models.downstream.classification import ClassificationProbing
        from src.models.downstream.reconstruction import ReconstructionProbing
        from src.models.downstream.segmentation import SegmentationProbing

        for frozen in (True, False):
            factories = (
                lambda backbone: ClassificationProbing(backbone, "sparse", 4, 2, freeze_model=frozen),
                lambda backbone: ReconstructionProbing(backbone, "lowrank", 4, num_decoder_layers=1, freeze_model=frozen),
                lambda backbone: SegmentationProbing(backbone, feature_dim=4, num_classes=2, freeze_backbone=frozen),
            )
            for factory in factories:
                backbone = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Dropout(0.5))
                probe = factory(backbone)
                self.assertIs(probe.train(), probe)
                self.assertTrue(probe.training)
                self.assertEqual(backbone.training, not frozen)
                if frozen:
                    running_mean = backbone[0].running_mean.clone()
                    inputs = torch.randn(8, 4)
                    torch.testing.assert_close(backbone(inputs), backbone(inputs))
                    torch.testing.assert_close(backbone[0].running_mean, running_mean)
                probe.eval()
                self.assertFalse(backbone.training)
                probe.train()
                self.assertEqual(backbone.training, not frozen)
        probe = SegmentationProbing(torch.nn.Identity(), feature_dim=4, freeze_model=True)
        probe.train()
        self.assertFalse(probe.training)
        self.assertFalse(probe.decoder_head.training)

    def test_image_folder_and_backward(self):
        from examples.common import image_paths

        torch.manual_seed(42)
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "Set root"):
                ImageFolderSSL(None)
            (Path(folder) / "not_an_image.jpg").mkdir()
            with self.assertRaisesRegex(ValueError, "No images"):
                ImageFolderSSL(folder)
            with self.assertRaisesRegex(ValueError, "No images"):
                image_paths(folder)
            for index in range(2):
                Image.new("RGB", (240, 260), (40 + index * 80, 90, 160)).save(
                    Path(folder) / f"{index}.png")
            self.assertEqual(len(image_paths(folder)), 2)
            batch = next(iter(DataLoader(ImageFolderSSL(folder), batch_size=2)))
            self.assertEqual(batch["global_views"].shape, (2, 2, 3, 224, 224))
            self.assertEqual(batch["local_views"].shape, (2, 8, 3, 96, 96))
            model = STELLARModel(
                4, vit_config=dict(hidden_size=48, num_hidden_layers=1,
                                   num_attention_heads=4, intermediate_size=96),
                num_clusters=16, prototype_dim=16, do_recon=False,
                num_masks=2, num_local_crops=2, do_global_align=True)
            predictions = model(batch)["predictions"]
            loss = STELLAR_Loss()(predictions, batch["labels"])
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(model.sparse_tokens.grad)
            features = model.eval().encode(batch["image"])
            torch.testing.assert_close(features["spatial"].sum(-1), torch.ones(2, 196))
            torch.testing.assert_close(features["lowrank"], features["spatial"] @ features["sparse"])

    def test_imagenet_ssl_skips_unused_crop(self):
        from src.datasets.imagenet_dataset import ImageNetKaggle

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            images = root / "ILSVRC/Data/CLS-LOC/train/n00000001"
            images.mkdir(parents=True)
            Image.new("RGB", (240, 260), (40, 90, 160)).save(images / "sample.jpg")
            (root / "imagenet_class_index.json").write_text(json.dumps({"0": ["n00000001", "fixture"]}))
            (root / "ILSVRC2012_val_labels.json").write_text("{}")
            dataset = ImageNetKaggle(root)
            dataset.transform = Mock(side_effect=AssertionError("Unused crop should not execute"))
            sample = dataset[0]
            dataset.transform.assert_not_called()
            self.assertEqual(sample["global_views"].shape, (8, 3, 224, 224))
            self.assertEqual(sample["local_views"].shape, (8, 3, 96, 96))
            torch.testing.assert_close(sample["image"], sample["labels"])


if __name__ == "__main__":
    unittest.main()