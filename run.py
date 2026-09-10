"""Seeded local entry point for the existing Olympus/Lightning recipes."""

from pathlib import Path

import hydra
import lightning as lightning
from omegaconf import OmegaConf


def run_experiment(config):
    from azureml.acft.image.components.olympus.app.main import parse_config

    options = config.get("scratch", {})
    lightning.seed_everything(options.get("seed", 42), workers=True)
    Path(config.experiment_output_dir).mkdir(parents=True, exist_ok=True)
    if config.olympus_checkpoint.load_checkpoint:
        raise ValueError("Use scratch.resume for a full Lightning resume, or a model checkpoint for weights only")
    experiment = parse_config(config)
    resume = options.get("resume")
    if config.mode == "train":
        experiment.trainer.fit(experiment.model, datamodule=experiment.data, ckpt_path=resume)
        if options.get("test_after_fit", False):
            experiment.trainer.test(experiment.model, datamodule=experiment.data)
    elif config.mode == "test":
        experiment.trainer.test(experiment.model, datamodule=experiment.data, ckpt_path=resume)
    else:
        raise ValueError("This launcher supports mode=train or mode=test")
    return experiment


@hydra.main(config_path="configs", config_name="stellar", version_base="1.3")
def main(config):
    if config.get("scratch", {}).get("print_config", False):
        print(OmegaConf.to_yaml(config, resolve=True))
        return
    run_experiment(config)


if __name__ == "__main__":
    main()