"""Run one canonical shard row through the released GLIM representation path."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.sharded_dataset import ShardedZuCoDataset  # noqa: E402
from project_adapters.glim_representation import (  # noqa: E402
    CanonicalGLIMRepresentationAdapter,
    load_upstream_glim_class,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--glim-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--prompt-mode", choices=("released", "canonical", "task_masked"), default="canonical"
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    glim_root = args.glim_root.resolve()
    GLIM = load_upstream_glim_class(glim_root)

    dataset = ShardedZuCoDataset(
        args.dataset_root,
        args.phase,
        classification_label_keys=["sentiment label", "topic_label"],
        regression_label_keys=["length", "surprisal"],
        task_prompt_mode="canonical",
    )
    try:
        item = dataset[0]
        eeg = item["eeg"].unsqueeze(0)
        mask = item["mask"].unsqueeze(0)
        if tuple(eeg.shape[1:]) != (1280, 128) or tuple(mask.shape[1:]) != (1280,):
            raise ValueError(
                f"released GLIM expects EEG [1280,128] and mask [1280], got "
                f"{tuple(eeg.shape[1:])} and {tuple(mask.shape[1:])}"
            )

        device = torch.device(args.device)
        model = GLIM.load_from_checkpoint(str(args.checkpoint), map_location="cpu", strict=False)
        model.eval().to(device)
        adapter = CanonicalGLIMRepresentationAdapter(model).eval().to(device)
        with torch.no_grad():
            output = adapter(
                eeg.to(device),
                mask.to(device),
                item["prompt"],
                sample_ids=[item["sample_id"]],
                source_dataframe_row_indices=[item["source_dataframe_row_index"]],
                mode=args.prompt_mode,
            )
        report = {
            "status": "glim_batch1_pass",
            "sample_id": output["sample_id"][0],
            "source_dataframe_row_index": output["source_dataframe_row_index"][0],
            "canonical_prompt": item["prompt"][0],
            "canonical_task_id": int(output["canonical_task_id"][0]),
            "prompt_mode": args.prompt_mode,
            "eeg_shape": list(eeg.shape),
            "eeg_tokens_shape": list(output["eeg_tokens"].shape),
            "eeg_vector_shape": list(output["eeg_vector"].shape),
            "checkpoint_sha256": sha256(args.checkpoint),
            "glim_root": str(glim_root),
        }
        print(json.dumps(report, indent=2))
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
