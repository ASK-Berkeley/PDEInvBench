import logging
import os

from git import Repo

"""
Collection of utility functions for running experiments.
"""


def is_wandb_online():
    """
    Check if wandb is online based on wandb setting file.
    """
    # Get W&B directory - default to current directory
    wandb_dir = os.environ.get("WANDB_DIR", ".")
    settings_file = os.path.join("wandb", "settings")
    disabled = False
    mode = "online"

    with open(os.path.join(wandb_dir, settings_file), "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("disabled"):
                disabled = "true" in line.split("=")[1].strip().lower()
            if line.startswith("mode"):
                mode = line.split("=")[1].strip().lower()
    return mode == "online" and not disabled


def validate_git_status():
    """
    Check if the git repository is clean to run experiments.
    """
    # Check the env variable if we are in a dev environment in which case,
    # we ignore any git dirty status
    is_dev_env = os.environ.get("META_DEV", "false").lower() == "true"
    if is_dev_env:
        return

    wandb_online = is_wandb_online()
    repo = Repo(".", search_parent_directories=True)
    repo_is_dirty = repo.is_dirty()
    if wandb_online:
        assert (
            not repo_is_dirty
        ), "Git repository is dirty! Please commit your changes before running wandb online experiments."
    elif repo_is_dirty:
        logging.warning(
            "Git repository is dirty! You may test out runs locally but commit your changes before running any wandb online experiments."
        )
