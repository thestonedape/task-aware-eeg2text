import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data.sharded_dataset import ShardedZuCoDataset
from data.datamodule import GLIMDataModule
from data.canonical_contract import CANONICAL_MANIFEST_FIELDS, canonical_tsr


HERE = Path(__file__).resolve().parent


class KaggleIOTests(unittest.TestCase):
    def test_all_released_glim_tsr_labels_normalize(self):
        expected = {
            "awarding": "AWARD",
            "education": "EDUCATION",
            "employment": "EMPLOYER",
            "foundation": "FOUNDER",
            "job title": "JOB_TITLE",
            "nationality": "NATIONALITY",
            "political affiliation": "POLITICAL_AFFILIATION",
            "visit": "VISITED",
            "marriage": "WIFE",
        }
        self.assertEqual(
            {label: canonical_tsr(label) for label in expected},
            expected,
        )

    def test_tiny_pickle_to_real_batch1_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nr_text = "With his interest in race cars, he formed a second company, the Henry Ford Company."
            frame = pd.DataFrame({
                "eeg": [np.ones((8, 4), dtype=np.float32) * value for value in (1, 2, 3)],
                "mask": [np.ones(8, dtype=np.int8) for _ in range(3)],
                "dataset": ["ZuCo1", "ZuCo1", "ZuCo2"],
                "task": ["task1", "task2", "task3"],
                "subject": ["S1", "S1", "S2"],
                "phase": ["train", "val", "test"],
                "text uid": [1, 2, 3],
                "input text": ["a", nr_text, "c"],
                "raw text": ["a", nr_text, "c"],
                "sentiment label": ["neutral", np.nan, np.nan],
                "relation label": [np.nan, np.nan, "awarding"],
                "raw label": ["neutral", np.nan, "award"],
                "control": [False, False, False],
                "label id": [0, 1, 2],
            })
            for key in (
                "lexical simplification (v0)", "lexical simplification (v1)",
                "semantic clarity (v0)", "semantic clarity (v1)",
                "syntax simplification (v0)", "syntax simplification (v1)",
                "naive rewritten", "naive simplified",
            ):
                frame[key] = [f"{text}-{key}" for text in frame["input text"]]
            source = root / "source.df"
            output = root / "derived"
            frame.to_pickle(source)
            subprocess.run([
                sys.executable, str(HERE / "prepare_shards.py"), "--dataframe", str(source),
                "--output-root", str(output), "--rows-per-shard", "2",
            ], check=True, capture_output=True, text=True)
            result = subprocess.run([
                sys.executable, str(HERE / "smoke_input.py"), "--dataset-root", str(output),
                "--batch-size", "1",
            ], check=True, capture_output=True, text=True)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "batch1_pass")
            self.assertEqual(report["phase_counts"], {"test": 1, "train": 1, "val": 1})
            self.assertEqual(report["semkey_generated_labels"], "not_fabricated")
            self.assertEqual(report["eeg_shape"], [1, 8, 4])
            self.assertEqual(report["sample_id"], "ZuCo1::task1::S1::row000000")
            nested_input = root / "kaggle_input" / "datasets" / "owner" / "derived"
            shutil.copytree(output, nested_input)
            nested_result = subprocess.run([
                sys.executable, str(HERE / "smoke_input.py"),
                "--input-root", str(root / "kaggle_input"), "--batch-size", "1",
            ], check=True, capture_output=True, text=True)
            nested_report = json.loads(nested_result.stdout)
            self.assertEqual(nested_report["status"], "batch1_pass")
            self.assertEqual(nested_report["sample_id"], report["sample_id"])
            manifest = json.loads((output / "metadata" / "shard_manifest.json").read_text())
            self.assertEqual(manifest["schema_version"], 2)
            with (output / "manifests" / "canonical_full_manifest.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                canonical_rows = list(csv.DictReader(handle))
            self.assertEqual(len(canonical_rows), 3)
            self.assertEqual(canonical_rows[0]["sr_sentiment_3"], "neutral")
            self.assertEqual(canonical_rows[1]["nr_relation_content"], "FOUNDER")
            self.assertEqual(canonical_rows[2]["tsr_instruction_relation"], "AWARD")
            self.assertNotIn("semkey_sentiment_2", canonical_rows[0])
            self.assertNotIn("topic_label", canonical_rows[0])
            with (output / "shards" / "index.csv").open(encoding="utf-8", newline="") as handle:
                index_rows = list(csv.DictReader(handle))
            self.assertEqual(index_rows[0]["mask_semkey_sentiment_2"], "0")
            self.assertEqual(index_rows[0]["mask_topic_label"], "0")
            frozen_val = root / "frozen_val.csv"
            with frozen_val.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CANONICAL_MANIFEST_FIELDS)
                writer.writeheader()
                writer.writerow(canonical_rows[1])
            output_checked = root / "derived_checked"
            subprocess.run([
                sys.executable, str(HERE / "prepare_shards.py"), "--dataframe", str(source),
                "--output-root", str(output_checked), "--rows-per-shard", "2",
                "--expected-validation-manifest", str(frozen_val),
            ], check=True, capture_output=True, text=True)
            checked = json.loads(
                (output_checked / "metadata" / "canonical_full_contract_report.json").read_text()
            )
            self.assertEqual(checked["frozen_validation_check"]["status"], "pass")
            dataset = ShardedZuCoDataset(
                output, "val",
                classification_label_keys=[
                    "sr_sentiment_3", "nr_relation_content", "tsr_instruction_relation"
                ],
                regression_label_keys=["length_words_whitespace_v1"],
                task_prompt_mode="canonical",
            )
            item = dataset[0]
            self.assertEqual(item["prompt"], ("<NR>", "ZuCo1", "S1"))
            self.assertEqual(item["sample_id"], "ZuCo1::task2::S1::row000001")
            self.assertEqual(item["nr_relation_content"], "FOUNDER")
            self.assertEqual(item["mask_nr_relation_content"], 1)
            self.assertEqual(item["cohort"], "zuco1_nr_tsr_noncausal")
            self.assertEqual(item["length_words_whitespace_v1"], len(nr_text.split()))
            self.assertEqual(tuple(item["eeg"].shape), (8, 4))
            dataset.close()
            module = GLIMDataModule(
                data_path=output, bsz_train=1, bsz_val=1, bsz_test=1,
                classification_label_keys=[
                    "sr_sentiment_3", "nr_relation_content", "tsr_instruction_relation"
                ],
                regression_label_keys=["length_words_whitespace_v1"],
                task_prompt_mode="canonical", use_spectral_whitening=False,
                use_robust_normalize=False,
            )
            module.setup("fit")
            batch = next(iter(module.val_dataloader()))
            self.assertEqual(batch["sample_id"][0], "ZuCo1::task2::S1::row000001")
            self.assertEqual(batch["nr_relation_content"][0], "FOUNDER")
            module.val_set.close()


if __name__ == "__main__":
    unittest.main()
