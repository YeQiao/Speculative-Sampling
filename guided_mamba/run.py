"""
Entry point for training the guided Mamba2 drafter.

Usage:
    python -m guided_mamba.run fit --config guided_mamba/config.yaml
    python -m guided_mamba.run fit --config guided_mamba/config.yaml --model.steer_z=true
"""

import os

import lightning as L
import torch
from datasets import DatasetDict
from lightning.pytorch.cli import LightningCLI

from guided_mamba.train import GuidedMambaTrainer


class DataLoader(L.LightningDataModule):
    """Loads pre-tokenised datasets saved by prepare_data.py."""

    def __init__(
        self,
        path: str,
        bsz: int = 32,
        n_train: int | None = None,
        n_val: int | None = None,
    ):
        super().__init__()
        self.path = path
        self.bsz = bsz
        self.n_train = n_train
        self.n_val = n_val

    def prepare_data(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Path {self.path} does not exist.")

    def setup(self, stage: str):
        self.ds = DatasetDict.load_from_disk(self.path).with_format("torch")
        to_keep = ["targets"]
        if "loss_mask" in self.ds["train"].features:
            to_keep.append("loss_mask")
        if "prompt" in self.ds["train"].features:
            to_keep.append("prompt")
        self.ds = self.ds.remove_columns(
            [c for c in self.ds["train"].column_names if c not in to_keep]
        )
        if self.n_train is not None:
            self.ds["train"] = self.ds["train"].select(range(self.n_train))
        if self.n_val is not None:
            self.ds["validation"] = self.ds["validation"].select(range(self.n_val))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds["train"],
            batch_size=self.bsz,
            num_workers=4,
            shuffle=False,
            drop_last=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.ds["validation"],
            batch_size=self.bsz,
            num_workers=4,
            shuffle=False,
            drop_last=True,
        )


def main():
    cli = LightningCLI(
        GuidedMambaTrainer,
        DataLoader,
        save_config_callback=None,
    )


if __name__ == "__main__":
    main()
