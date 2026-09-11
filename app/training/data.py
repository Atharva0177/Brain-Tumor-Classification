from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

CLASS_NAMES = ("glioma", "meningioma", "notumor", "pituitary")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


class CachedTensorDataset(Dataset):
    def __init__(self, manifest_path: Path, partition: str):
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.cache_root = Path(manifest_path).parent
        self.records = [record for record in manifest["records"] if record["partition"] == partition]
        self.partition = partition
        if not self.records:
            raise ValueError(f"No records found for partition {partition!r}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        tensor = torch.load(self.cache_root / record["processed_path"], weights_only=True)
        return tensor, CLASS_TO_INDEX[record["class_name"]]
