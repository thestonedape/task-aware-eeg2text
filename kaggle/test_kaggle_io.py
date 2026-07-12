import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


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
            })
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


if __name__ == "__main__":
    unittest.main()
