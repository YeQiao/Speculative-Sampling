"""
Entry point for training with device_map support (for 70B+ verifiers).

Usage:
    python -m spec_mamba.run fit --config guided_mamba/config_70b.yaml
    python -m spec_mamba.run fit --config guided_mamba/config_70b.yaml --device_map auto
"""

import os
import subprocess
import sys

import lightning as L
import torch
from datasets import DatasetDict
from lightning.pytorch.cli import LightningCLI

from spec_mamba.trainer import SpecMambaTrainer


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


def _detect_max_memory() -> dict:
    """Auto-detect per-GPU free memory via nvidia-smi.
    
    Reserves more headroom on GPU 0 (where trainable modules/optimizer live)
    and less on other GPUs (which only hold frozen verifier shards).
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        max_memory = {}
        for line in out.strip().splitlines():
            idx, free_mb = line.split(",")
            gpu_idx = int(idx.strip())
            free_gb = int(free_mb.strip()) / 1024
            # GPU 0 needs extra headroom for drafter, optimizer, and activations
            headroom = 18 if gpu_idx == 0 else 4
            usable = max(0, free_gb - headroom)
            max_memory[gpu_idx] = f"{int(usable)}GiB"
        return max_memory
    except Exception:
        return None


def main():
    # Extract --device_map and --quantize before LightningCLI parses args
    device_map = None
    quantize = None
    filtered_args = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--device_map":
            device_map = sys.argv[i + 1] if i + 1 < len(sys.argv) else "auto"
            i += 2
        elif sys.argv[i].startswith("--device_map="):
            device_map = sys.argv[i].split("=", 1)[1]
            i += 1
        elif sys.argv[i] == "--quantize":
            quantize = sys.argv[i + 1] if i + 1 < len(sys.argv) else "8bit"
            i += 2
        elif sys.argv[i].startswith("--quantize="):
            quantize = sys.argv[i].split("=", 1)[1]
            i += 1
        else:
            filtered_args.append(sys.argv[i])
            i += 1
    sys.argv = filtered_args

    if device_map:
        SpecMambaTrainer._verifier_device_map = device_map
        max_mem = _detect_max_memory()
        if max_mem:
            SpecMambaTrainer._verifier_max_memory = max_mem
            print(f"[run] device_map={device_map}, max_memory={max_mem}")
        else:
            print(f"[run] device_map={device_map}, max_memory=auto")

    if quantize:
        from transformers import BitsAndBytesConfig
        if quantize in ("8bit", "8"):
            SpecMambaTrainer._verifier_quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantize in ("4bit", "4"):
            SpecMambaTrainer._verifier_quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        print(f"[run] quantize={quantize}")
        # With quantization, device_map is required
        if not device_map:
            SpecMambaTrainer._verifier_device_map = "auto"
            print("[run] auto-enabling device_map=auto for quantization")

    cli = LightningCLI(
        SpecMambaTrainer,
        DataLoader,
        save_config_callback=None,
    )


if __name__ == "__main__":
    main()
