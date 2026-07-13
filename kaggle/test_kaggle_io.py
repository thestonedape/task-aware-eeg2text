import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data.sharded_dataset import ShardedZuCoDataset
from data.datamodule import GLIMDataModule


HERE = Path(__file__).resolve().parent


class KaggleIOTests(unittest.TestCase):
    def test_tiny_pickle_to_real_batch1_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame({
                "eeg": [np.ones((8, 4), dtype=np.float32) * value for value in (1, 2, 3)],
                "mask": [np.ones(8, dtype=np.int8) for _ in range(3)],
                "dataset": ["ZuCo1", "ZuCo1", "ZuCo2"],
                "task": ["task1", "task2", "task3"],
                "subject": ["S1", "S1", "S2"],
                "phase": ["train", "val", "test"],
                "text uid": [1, 2, 3],
                "input text": ["a", "b", "c"],
                "sentiment label": ["neutral", "non_neutral", "neutral"],
                "topic_label": ["Biographies and Factual Knowledge"] * 3,
                "length": [1.0, 1.0, 1.0],
                "surprisal": [2.0, 2.0, 2.0],
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
            self.assertEqual(report["eeg_shape"], [1, 8, 4])
            self.assertEqual(report["sample_id"], "ZuCo1::task1::S1::row000000")
            dataset = ShardedZuCoDataset(
                output, "val", classification_label_keys=["sentiment label", "topic_label"],
                regression_label_keys=["length", "surprisal"], task_prompt_mode="canonical",
            )
            item = dataset[0]
            self.assertEqual(item["prompt"], ("<NR>", "ZuCo1", "S1"))
            self.assertEqual(item["sample_id"], "ZuCo1::task2::S1::row000001")
            self.assertEqual(tuple(item["eeg"].shape), (8, 4))
            dataset.close()
            module = GLIMDataModule(
                data_path=output, bsz_train=1, bsz_val=1, bsz_test=1,
                classification_label_keys=["sentiment label", "topic_label"],
                regression_label_keys=["length", "surprisal"],
                task_prompt_mode="canonical", use_spectral_whitening=False,
                use_robust_normalize=False,
            )
            module.setup("fit")
            batch = next(iter(module.val_dataloader()))
            self.assertEqual(batch["sample_id"][0], "ZuCo1::task2::S1::row000001")
            module.val_set.close()


if __name__ == "__main__":
    unittest.main()
