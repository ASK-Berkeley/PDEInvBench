import os
import logging
from functools import partial
import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from pdeinvbench.utils import validate_git_status
from pdeinvbench.utils.config_utils import (
    resolve_pde_resolution,
    resolve_grid_transform,
)
import sys
import pdeinvbench

# Add any other submodules you need


@hydra.main(
    version_base=None,
    config_path="configs/",
    config_name="1dkdv",
)
def main(cfg: DictConfig) -> None:
    # check git dirty status
    validate_git_status()

    hydra_cfg = HydraConfig.get()
    overrides = hydra_cfg.overrides.task
    overrides = [
        override
        for override in overrides
        if "data_root" not in override
        and "batch_size" not in override
        and "logging.project" not in override
        and "logging.run_name" not in override
        and "logging.save_dir" not in override
        and "num_nodes" not in override
        and "log_model" not in override
        and "supervised_learning_min_epoch" not in override
        and "supervised_learning_max_epoch" not in override
        and "system_params.data_root" not in override
    ]

    # First thing is to determine the constants
    resolve_pde_resolution(cfg)
    resolve_grid_transform(cfg)
    # Combine the config name with overrides:
    if "run_name" in cfg.logging.keys():
        wandb_name = cfg.logging.run_name
        del cfg.logging.run_name
    else:
        wandb_name = hydra_cfg.job["config_name"] + "_" + "_".join(overrides)

    # Fix random seed
    seed_everything(cfg.seed)

    # Define logger
    # Arguments for wandb.init
    wandb_args = OmegaConf.to_container(
        cfg.logging, resolve=True, throw_on_missing=True
    )
    # Remove the _target_ key which usually points to the lightning logger
    wandb_args.pop("_target_")
    # Rename the save_dir key to dir, which is what wandb.init expects
    # We keep save_dir in cfg.logging so that we can still resolve the lightning wandb logger
    wandb_args["dir"] = wandb_args.pop("save_dir")
    cfg_as_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # Manually instantiate the wandb experiment
    # We do this so that the wandb run is created with the correct parameters to recreate the run at the start.
    # Otherwise, the config is only saved at the end of the run.
    experiment = wandb.init(**wandb_args, name=wandb_name, config=cfg_as_dict)
    # Instantiate dataloaders
    train_dataloader = instantiate(cfg.data.train_dataloader)
    val_dataloaders = []
    test_dataloaders = []

    if "val_dataloader" in cfg.data.keys() and os.path.exists(cfg.data.val_data_root):
        val_dataloader = instantiate(cfg.data.val_dataloader)
        val_dataloaders.append(val_dataloader)
        # to run validation set at test time as well with best weights
        test_dataloaders.append(val_dataloader)

    if "ood_dataloader" in cfg.data.keys() and os.path.exists(cfg.data.ood_data_root):
        print("ood loader")
        test_dataloaders.append(instantiate(cfg.data.ood_dataloader))
    if "ood_dataloader_extreme" in cfg.data.keys() and os.path.exists(
        cfg.data.ood_data_root_extreme
    ):
        print("ood loader extreme")
        test_dataloaders.append(instantiate(cfg.data.ood_dataloader_extreme))
    if "test_dataloader" in cfg.data.keys() and os.path.exists(cfg.data.test_data_root):
        print("test iid loader")
        test_dataloaders.append(instantiate(cfg.data.test_dataloader))

    # Instantiate model and optimizer
    model = instantiate(cfg.model.model_config)
    checkpoint_path = None
    if "inverse_model_wandb_run" in cfg.keys() and cfg.inverse_model_wandb_run != "":
        logging.info("Loading inverse model checkpoint from wandb")

        inverse_model_run_path = cfg.inverse_model_wandb_run
        artifact = experiment.use_artifact(inverse_model_run_path, type="model")
        checkpoint_path = os.path.join(artifact.download(), "model.ckpt")
    elif (
        "inverse_model_checkpoint_path" in cfg.keys()
        and cfg.inverse_model_checkpoint_path != ""
    ):
        checkpoint_path = cfg.inverse_model_checkpoint_path

    if checkpoint_path is not None and "test_run" in cfg and cfg.test_run:
        state_dict = torch.load(checkpoint_path, weights_only=False)["state_dict"]
        state_dict = {k.partition("model.")[2]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    lr_scheduler = instantiate(cfg.lr_scheduler, optimizer=optimizer)

    module_kwargs = {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }

    if "tailoring_optimizer" in cfg.keys():
        # We use a partial function so that we can dynamically build new optimizers
        def tailoring_optimizer(x):
            return instantiate(cfg.tailoring_optimizer, params=x)

        module_kwargs["tailoring_optimizer"] = tailoring_optimizer
    else:
        tailoring_optimizer = None
    # Wraps the base model to perform prediction and losses
    if tailoring_optimizer != None:
        print(cfg.tailoring_optimizer)

    print("instantiate inverse module")
    Inverse_module = instantiate(
        cfg.lightning_module,
        **module_kwargs,
    )
    print("instantiate inverse module done")
    # Instantiate the lightning logger & force the lightning logger to use the wandb experiment manually created
    log_model = "all"
    if "log_model" in cfg:
        log_model = cfg.log_model
    logger = instantiate(cfg.logging, experiment=experiment, log_model=log_model)
    logger._save_dir += "/" + cfg_as_dict["logging"]["project"] + "/" + wandb_name
    logger.watch(model, log="all")
    # Lightning trainer
    if tailoring_optimizer is not None:
        L_trainer = instantiate(
            cfg.trainer,
            logger=logger,
            inference_mode=False,
        )
    else:
        L_trainer = instantiate(cfg.trainer, logger=logger)

    if "test_run" in cfg and cfg.test_run:
        L_trainer.test(
            model=Inverse_module,
            dataloaders=test_dataloaders,
        )
    else:
        # Train
        L_trainer.fit(
            Inverse_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloaders,
            ckpt_path=checkpoint_path,
        )
        L_trainer.test(dataloaders=test_dataloaders, ckpt_path="best")

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
