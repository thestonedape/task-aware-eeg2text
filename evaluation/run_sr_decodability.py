"""Run the plain SR-sentiment decodability check on the frozen GLIM vectors.

Mirrors ``run_frozen_factor_probes`` mounting: the same immutable frozen-vector
artifact (hash-checked) and the same rebuilt recoverability protocol, but instead of
the admission contrast battery it asks one question -- can a plain subject-grouped
classifier read SR three-class sentiment off the ``correct`` GLIM vector above a
label-permutation chance level, with the matched-wrong decoy control removed
(see ``sentiment_decodability_check`` for why the decoy is unfair to a 3-class
attribute). CPU-only. Never touches the held-out test (protocol rows are train/val).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.run_frozen_factor_probes import VectorStore, read_csv
from evaluation.sentiment_decodability_check import plain_decodability


def labels_and_groups(rows: list[dict[str, str]], ontology: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Map protocol rows to integer sentiment labels + subject groups (order preserved)."""
    index = {label: i for i, label in enumerate(ontology)}
    try:
        y = np.asarray([index[row["target_value"]] for row in rows], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"SR target outside ontology: {error}") from error
    groups = np.asarray([row["subject_id"] for row in rows])
    return y, groups


def load_features(
    vector_root: Path, protocol_root: Path, factor_id: str,
    expected_index_sha256: str | None, expected_vector_index_sha256: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    registry = json.loads((protocol_root / "recoverability_registry.json").read_text(encoding="utf-8"))
    ontology = [str(x) for x in registry["factors"][factor_id]["ontology"]]
    rows = [r for r in read_csv(protocol_root / "recoverability_rows.csv") if r["factor_id"] == factor_id]
    if any(r["split"] not in {"train", "val"} for r in rows):
        raise AssertionError("held-out row entered the SR decodability check")
    store = VectorStore(vector_root, expected_index_sha256, expected_vector_index_sha256)

    train = sorted([r for r in rows if r["split"] == "train"], key=lambda r: r["trial_id"])
    val = sorted([r for r in rows if r["split"] == "val"], key=lambda r: r["trial_id"])
    ordered = train + val                                   # correct vectors live under two conditions
    X = np.concatenate([
        store.matrix("correct_train", [r["trial_id"] for r in train]),
        store.matrix("correct_val", [r["trial_id"] for r in val]),
    ], axis=0)
    y, groups = labels_and_groups(ordered, ontology)
    return X, y, groups, ontology


def run(
    vector_root: Path, protocol_root: Path, output_root: Path,
    factor_id: str = "sr_sentiment_3", n_permutations: int = 200, seed: int = 20260729,
    expected_index_sha256: str | None = None, expected_vector_index_sha256: str | None = None,
) -> dict[str, Any]:
    X, y, groups, ontology = load_features(
        vector_root, protocol_root, factor_id, expected_index_sha256, expected_vector_index_sha256)
    result = plain_decodability(X, y, groups, n_permutations=n_permutations, seed=seed)
    result["factor_id"] = factor_id
    result["ontology"] = ontology
    result["held_out_test_accessed"] = False
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / "sr_sentiment_decodability.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["report_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--factor-id", default="sr_sentiment_3")
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--expected-index-sha256")
    parser.add_argument("--expected-vector-index-sha256")
    args = parser.parse_args()
    report = run(
        args.vector_root, args.protocol_root, args.output_root, args.factor_id,
        args.n_permutations, args.seed, args.expected_index_sha256, args.expected_vector_index_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["labels_and_groups", "load_features", "run"]
