"""Verify the preserved prompt-neutral EEG/text pilot-input artifact.

This verifier is standard-library only. It re-hashes every declared vector
chunk and checks the identity/mapping contracts without loading model weights,
raw EEG, NumPy arrays, or the held-out test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


EXPECTED = {
    "combined_manifest_sha256": "6c1fff8d2e89e33a72d03c39651e8ecce678c3b93cdb66747dd6dcc00538cddb",
    "project_commit": "75f79061c55ba370e1478b8c70622e613f52a215",
    "glim_commit": "e1f202cb793cfe7292fbc0072a4c26a7dd0660d9",
    "source_index_sha256": "bdaaaf5c91d3c9eec16a0727825da996fd2186867245951bfdfdc92aab7738b0",
    "checkpoint_sha256": "25fcd31d1d6cafc9a0656c50a4916ba6ee106884b269d347284784cc0522c8ba",
    "pilot_contract_sha256": "a7370a61921803fbeaaab874dcfef77d38d6dceb35b05e249b6f5548e8a2921e",
    "eeg_manifest_sha256": "37cb5e4461b1907190bea1c88772148d2bb88c29717be4b351bc97742234b3eb",
    "eeg_vector_index_sha256": "373e49da7d3d6d00aaae414437886033e7f2c6a938a92662f76a0973744e0ae9",
    "text_manifest_sha256": "c6510e5f5aeb3bfb2e70aa8f13080005878cec6c84afea085674e87895e30a7d",
}
EEG_COUNTS = {
    "correct_train": 17908,
    "correct_val": 2200,
    "gaussian_val": 2200,
    "matched_wrong_val": 2200,
    "zero_val": 2200,
}


def sha256(path: Path) -> str:
    state = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            state.update(block)
    return state.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, got {actual!r}")


def verify_chunk_files(root: Path, chunks: list[dict], kind: str) -> list[str]:
    hashes = []
    for chunk in chunks:
        number = int(chunk["chunk_number"])
        if kind == "eeg":
            relative = f"vectors/{chunk['condition']}_{number:05d}.npz"
        else:
            relative = f"vectors/text_{number:05d}.npz"
        path = root / relative
        actual = sha256(path)
        require_equal(actual, chunk["vector_npz_sha256"], f"chunk hash {relative}")
        hashes.append(actual)
    if len(hashes) != len(set(hashes)):
        raise ValueError(f"{kind} chunk files are not hash-distinct")
    return hashes


def verify_eeg(root: Path, combined: dict) -> dict:
    manifest_path = root / "vector_manifest.json"
    require_equal(sha256(manifest_path), EXPECTED["eeg_manifest_sha256"], "EEG manifest SHA256")
    require_equal(combined["eeg_manifest_sha256"], EXPECTED["eeg_manifest_sha256"], "combined EEG manifest SHA256")
    manifest = read_json(manifest_path)
    require_equal(manifest.get("status"), "pass", "EEG status")
    require_equal(manifest.get("run_mode"), "full_development", "EEG run mode")
    require_equal(manifest.get("prompt_mode"), "all_masked", "EEG prompt mode")
    require_equal(manifest.get("condition_counts"), EEG_COUNTS, "EEG condition counts")
    require_equal(manifest.get("vector_dim"), 1024, "EEG vector dimension")
    require_equal(manifest.get("dtype"), "float32", "EEG dtype")
    require_equal(manifest.get("source_index_sha256"), EXPECTED["source_index_sha256"], "EEG source index")
    require_equal(manifest.get("checkpoint_sha256"), EXPECTED["checkpoint_sha256"], "EEG checkpoint")
    require_equal(manifest.get("glim_commit"), EXPECTED["glim_commit"], "EEG GLIM commit")
    require_equal(manifest.get("checks", {}).get("held_out_test_accessed"), False, "EEG test access")
    chunks = manifest.get("chunks", [])
    require_equal(len(chunks), 212, "EEG chunk count")
    verify_chunk_files(root, chunks, "eeg")

    index_path = root / "vector_index.csv"
    actual_index_sha = sha256(index_path)
    require_equal(actual_index_sha, EXPECTED["eeg_vector_index_sha256"], "EEG vector index SHA256")
    require_equal(manifest.get("vector_index_sha256"), actual_index_sha, "EEG manifest index SHA256")
    require_equal(combined.get("eeg_vector_index_sha256"), actual_index_sha, "combined EEG index SHA256")
    rows = read_csv(index_path)
    require_equal(len(rows), 26708, "EEG vector-index rows")
    require_equal(dict(sorted(Counter(row["condition"] for row in rows).items())), EEG_COUNTS, "EEG CSV condition counts")
    require_equal({row["prompt_mode"] for row in rows}, {"all_masked"}, "EEG CSV prompt modes")
    if any(row["phase"] == "test" for row in rows):
        raise ValueError("held-out test row found in EEG vector index")
    if len({(row["condition"], row["target_trial_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate EEG condition/target identity")
    return {"rows": len(rows), "chunks": len(chunks), "vector_index_sha256": actual_index_sha}


def verify_text(root: Path, combined: dict) -> dict:
    manifest_path = root / "text_vector_manifest.json"
    require_equal(sha256(manifest_path), EXPECTED["text_manifest_sha256"], "text manifest SHA256")
    require_equal(combined["text_manifest_sha256"], EXPECTED["text_manifest_sha256"], "combined text manifest SHA256")
    manifest = read_json(manifest_path)
    require_equal(manifest.get("status"), "pass", "text status")
    require_equal(manifest.get("run_mode"), "full_development", "text run mode")
    require_equal(manifest.get("unique_text_identities"), 1347, "unique text identities")
    require_equal(manifest.get("mapped_trials"), 20108, "mapped text trials")
    require_equal(manifest.get("split_counts"), {"train": 17908, "val": 2200}, "text split counts")
    require_equal(manifest.get("text_model_id"), "google/flan-t5-large", "text model")
    require_equal(manifest.get("text_dtype"), "float16", "text dtype")
    require_equal(manifest.get("vector_dim"), 1024, "text vector dimension")
    require_equal(manifest.get("source_index_sha256"), EXPECTED["source_index_sha256"], "text source index")
    require_equal(manifest.get("checkpoint_sha256"), EXPECTED["checkpoint_sha256"], "text checkpoint")
    require_equal(manifest.get("glim_commit"), EXPECTED["glim_commit"], "text GLIM commit")
    require_equal(manifest.get("checks", {}).get("held_out_test_accessed"), False, "text test access")
    chunks = manifest.get("chunks", [])
    require_equal(len(chunks), 3, "text chunk count")
    verify_chunk_files(root, chunks, "text")

    index_path = root / "text_vector_index.csv"
    mapping_path = root / "trial_text_targets.csv"
    index_sha = sha256(index_path)
    mapping_sha = sha256(mapping_path)
    require_equal(manifest.get("text_vector_index_sha256"), index_sha, "text index SHA256")
    require_equal(manifest.get("trial_text_targets_sha256"), mapping_sha, "text mapping SHA256")
    require_equal(combined.get("text_vector_index_sha256"), index_sha, "combined text index SHA256")
    require_equal(combined.get("trial_text_targets_sha256"), mapping_sha, "combined text mapping SHA256")
    index_rows = read_csv(index_path)
    mapping_rows = read_csv(mapping_path)
    require_equal(len(index_rows), 1347, "text vector-index rows")
    require_equal(len(mapping_rows), 20108, "trial-text mapping rows")
    target_ids = {row["text_target_id"] for row in index_rows}
    require_equal(len(target_ids), 1347, "unique text target IDs")
    if any(row["text_target_id"] not in target_ids for row in mapping_rows):
        raise ValueError("trial mapping references a missing text target")
    if any(row["split"] == "test" for row in mapping_rows):
        raise ValueError("held-out test row found in trial-text mapping")
    require_equal(
        dict(sorted(Counter(row["split"] for row in mapping_rows).items())),
        {"train": 17908, "val": 2200},
        "trial-text CSV split counts",
    )
    require_equal(len({row["trial_id"] for row in mapping_rows}), 20108, "unique mapped trials")
    return {
        "unique_text_identities": len(index_rows),
        "mapped_trials": len(mapping_rows),
        "chunks": len(chunks),
        "text_vector_index_sha256": index_sha,
        "trial_text_targets_sha256": mapping_sha,
    }


def verify(artifact_root: Path, preserved_source_id: str) -> dict:
    combined_path = artifact_root / "pilot_input_manifest.json"
    require_equal(sha256(combined_path), EXPECTED["combined_manifest_sha256"], "combined manifest SHA256")
    combined = read_json(combined_path)
    for field in (
        "project_commit", "glim_commit", "source_index_sha256", "checkpoint_sha256",
        "pilot_contract_sha256",
    ):
        require_equal(combined.get(field), EXPECTED[field], f"combined {field}")
    require_equal(combined.get("status"), "pass", "combined status")
    require_equal(combined.get("schema_version"), 1, "combined schema")
    require_equal(combined.get("eeg_prompt_mode"), "all_masked", "combined prompt mode")
    require_equal(combined.get("held_out_test_accessed"), False, "combined test access")
    eeg = verify_eeg(artifact_root / "eeg", combined)
    text = verify_text(artifact_root / "text", combined)
    run_metadata = read_json(artifact_root / "run_metadata.json")
    require_equal(run_metadata.get("status"), "pass", "run metadata status")
    require_equal(run_metadata.get("project_commit"), EXPECTED["project_commit"], "run project commit")
    require_equal(run_metadata.get("glim_commit"), EXPECTED["glim_commit"], "run GLIM commit")
    require_equal(run_metadata.get("test_accessed"), False, "run test access")
    return {
        "status": "pass",
        "schema_version": 1,
        "preserved_source_id": preserved_source_id,
        "combined_manifest_sha256": EXPECTED["combined_manifest_sha256"],
        "project_commit": EXPECTED["project_commit"],
        "glim_commit": EXPECTED["glim_commit"],
        "eeg": eeg,
        "text": text,
        "checks": {
            "all_215_chunk_hashes_revalidated": True,
            "all_prompt_fields_masked": True,
            "all_trial_text_targets_resolve": True,
            "held_out_test_accessed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--preserved-source-id", required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.artifact_root, args.preserved_source_id)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
