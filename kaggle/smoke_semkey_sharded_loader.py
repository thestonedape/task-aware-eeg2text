"""Load one real row through the trainable shard-backed SemKey dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.sharded_dataset import ShardedZuCoDataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("train", "val", "test"), default="val")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dataset = ShardedZuCoDataset(
        args.dataset_root,
        args.phase,
        classification_label_keys=["sr_sentiment_3", "nr_relation_content", "tsr_instruction_relation"],
        regression_label_keys=["length_words_whitespace_v1"],
        task_prompt_mode="canonical",
    )
    try:
        item = dataset[0]
        report = {
            "status": "pass",
            "phase": args.phase,
            "rows": len(dataset),
            "sample_id": item["sample_id"],
            "prompt": list(item["prompt"]),
            "eeg_shape": list(item["eeg"].shape),
            "mask_shape": list(item["mask"].shape),
            "reading_task": item["reading_task"],
            "cohort": item["cohort"],
            "canonical_masks": {
                key: item[key] for key in (
                    "mask_sr_sentiment_3", "mask_nr_relation_content", "mask_tsr_instruction_relation"
                )
            },
        }
    finally:
        dataset.close()
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
