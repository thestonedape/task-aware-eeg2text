import tempfile
import unittest
from pathlib import Path

from evaluation.verify_prompt_neutral_pilot_inputs import require_equal, sha256


class PromptNeutralVerifierUnitTests(unittest.TestCase):
    def test_sha256_and_equality_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "value.txt"
            path.write_bytes(b"pilot\n")
            self.assertEqual(
                sha256(path),
                "0628e75497579c83f1c71deceb433ce3cb1e2f8675da03543e877a28f2712bd7",
            )
        require_equal({"a": 1}, {"a": 1}, "equal")
        with self.assertRaises(ValueError):
            require_equal("wrong", "expected", "tamper")


if __name__ == "__main__":
    unittest.main()
