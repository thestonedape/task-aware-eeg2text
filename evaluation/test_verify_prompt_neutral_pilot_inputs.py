import tempfile
import unittest
from pathlib import Path

from evaluation.verify_prompt_neutral_pilot_inputs import require_equal, sha256


class PromptNeutralVerifierUnitTests(unittest.TestCase):
    def test_sha256_and_equality_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "value.txt"
            path.write_text("pilot\n", encoding="utf-8")
            self.assertEqual(
                sha256(path),
                "0c4a4dabc303430307716b2cc2a74c7aedab6e4d636bd07f9f4464da9f2c9c09",
            )
        require_equal({"a": 1}, {"a": 1}, "equal")
        with self.assertRaises(ValueError):
            require_equal("wrong", "expected", "tamper")


if __name__ == "__main__":
    unittest.main()
