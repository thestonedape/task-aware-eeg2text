"""Lazy, row-addressable SemKey Stage-1 dataset backed by Kaggle NPZ shards."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

from data.canonical_contract import CANONICAL_FIELDS


PT_TARGET_KEYS = [
    "lexical simplification (v0)", "lexical simplification (v1)",
    "semantic clarity (v0)", "semantic clarity (v1)",
    "syntax simplification (v0)", "syntax simplification (v1)",
    "naive rewritten", "naive simplified",
]
TASK_PROMPTS = {
    "released": {"task1": "<NR>", "task2": "<NR>", "task3": "<TSR>"},
    "canonical": {"task1": "<SR>", "task2": "<NR>", "task3": "<TSR>"},
}


class ShardedZuCoDataset(Dataset):
    """SemKey-compatible dataset that loads only the NPZ shard needed by a sample."""

    def __init__(
        self,
        dataset_root: str | Path,
        phase: Literal["train", "val", "test"],
        classification_label_keys: list[str] | None = None,
        regression_label_keys: list[str] | None = None,
        task_prompt_mode: Literal["released", "canonical"] = "released",
        embeddings_dict: dict | None = None,
        use_zuco1_only: bool = False,
        signal_transform=None,
    ):
        self.root = Path(dataset_root)
        manifest_path = self.root / "metadata" / "shard_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        index_path = self.root / manifest["index"]
        with index_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows = [row for row in rows if row["phase"] == phase]
        if use_zuco1_only:
            rows = [row for row in rows if row["dataset"] == "ZuCo1"]
        if not rows:
            raise ValueError(f"no {phase!r} rows found in sharded dataset")
        if task_prompt_mode not in TASK_PROMPTS:
            raise ValueError(f"unknown task_prompt_mode: {task_prompt_mode}")

        self.phase = phase
        self.rows = rows
        self.classification_label_keys = (
            ["topic_label"] if classification_label_keys is None else classification_label_keys
        )
        self.regression_label_keys = [] if regression_label_keys is None else regression_label_keys
        required = {
            "sample_id", "source_dataframe_row_index", "shard", "offset", "dataset", "task",
            "subject", "text_uid", "input text", *PT_TARGET_KEYS,
            *self.classification_label_keys, *self.regression_label_keys,
        }
        missing = required - set(rows[0])
        if missing:
            raise ValueError(f"shard index missing SemKey fields: {sorted(missing)}")
        self.task_prompt_mode = task_prompt_mode
        self.embeddings_dict = embeddings_dict
        self.signal_transform = signal_transform
        self._archive_name = None
        self._archive = None
        self.n_target_text = len(PT_TARGET_KEYS)
        self._target_multiplier = self.n_target_text if phase == "train" else 1
        expanded = [row for row in rows for _ in range(self._target_multiplier)]
        self.data = {
            "text uid": [self._text_uid(row["text_uid"]) for row in expanded],
            "sample_id": [row["sample_id"] for row in expanded],
        }
        for key in CANONICAL_FIELDS:
            if key in rows[0] and key not in self.data:
                self.data[key] = [row[key] for row in expanded]
        for key in self.classification_label_keys + self.regression_label_keys:
            self.data[key] = [row[key] for row in expanded]

    def __len__(self) -> int:
        return len(self.rows) * self._target_multiplier

    def _load_arrays(self, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
        shard_name = row["shard"]
        if shard_name != self._archive_name:
            if self._archive is not None:
                self._archive.close()
            self._archive = np.load(self.root / "shards" / shard_name, allow_pickle=False)
            self._archive_name = shard_name
        offset = int(row["offset"])
        eeg = self._archive["eeg"][offset].astype(np.float32, copy=False)
        mask = self._archive["mask"][offset].astype(np.int8, copy=False)
        if self.signal_transform is not None:
            eeg = self.signal_transform(eeg)
        return eeg, mask

    @staticmethod
    def _number(raw: str) -> float:
        value = float(raw)
        if not np.isfinite(value):
            raise ValueError(f"non-finite regression label: {raw!r}")
        return value

    @staticmethod
    def _text_uid(raw: str):
        return int(raw) if raw.lstrip("-").isdigit() else raw

    def __getitem__(self, idx: int) -> dict:
        base_idx, variant_idx = divmod(idx, self._target_multiplier)
        row = self.rows[base_idx]
        eeg, mask = self._load_arrays(row)
        raw_text = row["input text"]
        target_text = row[PT_TARGET_KEYS[variant_idx]] if self.phase == "train" else raw_text
        task_prompt = TASK_PROMPTS[self.task_prompt_mode].get(row["task"])
        if task_prompt is None:
            raise ValueError(f"unknown raw task key: {row['task']!r}")
        item = {
            "eeg": torch.from_numpy(eeg).float(),
            "mask": torch.from_numpy(mask),
            "prompt": (task_prompt, row["dataset"], row["subject"]),
            "text uid": self._text_uid(row["text_uid"]),
            "sample_id": row["sample_id"],
            "source_dataframe_row_index": int(row["source_dataframe_row_index"]),
            "input text": f"To English: {raw_text}",
            "target text": target_text,
            "raw task key": row["task"],
            "raw input text": raw_text,
            "all target texts": tuple(row[key] for key in PT_TARGET_KEYS),
        }
        for key in CANONICAL_FIELDS:
            if key not in row or key in item:
                continue
            if key.startswith("mask_"):
                item[key] = int(row[key])
            elif key in {"source_dataframe_row_index", "length_words_whitespace_v1"}:
                item[key] = int(row[key])
            else:
                item[key] = row[key]
        for key in self.classification_label_keys:
            item[key] = row[key]
        for key in self.regression_label_keys:
            item[key] = self._number(row[key])
        if self.embeddings_dict is not None:
            embedding = self.embeddings_dict[self._text_uid(row["text_uid"])]
            item["sentence_embedding"] = torch.from_numpy(embedding["sentence"])
            item["keyword_embedding"] = torch.from_numpy(embedding["keyword"])
            item["keyword_text"] = tuple(row.get(f"keyword_{number}", "") for number in range(1, 4))
        return item

    def close(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None
            self._archive_name = None

    def __del__(self):
        self.close()
