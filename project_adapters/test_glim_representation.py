import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from project_adapters.glim_representation import (
    CanonicalGLIMRepresentationAdapter,
    load_upstream_glim_class,
)


class DummyPromptEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 4
        self.prompt_keys = {
            "task": ["<UNK>", "<NR>", "<TSR>"],
            "dataset": ["<UNK>", "ZuCo1", "ZuCo2"],
            "subject": ["<UNK>", "S1", "S2"],
        }
        self.tables = nn.ModuleList([nn.Embedding(3, 4) for _ in range(3)])

    def encode(self, prompts, device=None):
        columns = []
        for values, keys in zip(prompts, self.prompt_keys.values()):
            columns.append(torch.tensor([keys.index(value) for value in values], device=device))
        return torch.stack(columns, dim=1)

    def forward(self, ids, _mode):
        return sum(table(ids[:, index]) for index, table in enumerate(self.tables))


class DummyEncoder(nn.Module):
    def forward(self, eeg, mask, prompt):
        return eeg + prompt.unsqueeze(1), {}


class DummyAligner(nn.Module):
    def embed_eeg(self, hidden):
        return hidden, hidden.mean(dim=1)


class DummyGLIM(nn.Module):
    def __init__(self):
        super().__init__()
        self.p_embedder = DummyPromptEmbedder()
        self.eeg_encoder = DummyEncoder()
        self.aligner = DummyAligner()
        self.eval_pembed = "src"


class CanonicalGLIMRepresentationTests(unittest.TestCase):
    def test_upstream_loader_rejects_non_glim_root(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_upstream_glim_class(Path(directory))

    def test_upstream_loader_scopes_released_torch_typing_workaround(self):
        with tempfile.TemporaryDirectory() as directory:
            model_root = Path(directory) / "model"
            model_root.mkdir()
            (model_root / "modules.py").write_text("", encoding="utf-8")
            (model_root / "glim.py").write_text(
                "import torch\n"
                "class GLIM:\n"
                "    def on_save_checkpoint(self, checkpoint: torch.Dict[str, torch.Any]) -> None:\n"
                "        pass\n",
                encoding="utf-8",
            )
            had_dict = hasattr(torch, "Dict")
            had_any = hasattr(torch, "Any")
            loaded = load_upstream_glim_class(Path(directory))
            self.assertEqual(loaded.__name__, "GLIM")
            if not had_dict:
                self.assertFalse(hasattr(torch, "Dict"))
            if not had_any:
                self.assertFalse(hasattr(torch, "Any"))

    def test_released_base_is_preserved_and_tasks_have_distinct_ids(self):
        adapter = CanonicalGLIMRepresentationAdapter(DummyGLIM())
        sr_base, sr_id = adapter.prompt_embedding(("<SR>", "ZuCo1", "S1"), "canonical")
        nr_base, nr_id = adapter.prompt_embedding(("<NR>", "ZuCo1", "S1"), "canonical")
        self.assertTrue(torch.equal(sr_base, nr_base))
        self.assertEqual(int(sr_id[0]), 0)
        self.assertEqual(int(nr_id[0]), 1)
        with torch.no_grad():
            adapter.task_delta.weight[0, 0] = 1.0
        sr_tuned, _ = adapter.prompt_embedding(("<SR>", "ZuCo1", "S1"), "canonical")
        self.assertFalse(torch.equal(sr_tuned, nr_base))

    def test_identity_safe_representation_output(self):
        adapter = CanonicalGLIMRepresentationAdapter(DummyGLIM())
        output = adapter(
            torch.ones(1, 8, 4),
            torch.ones(1, 8, dtype=torch.int8),
            ("<TSR>", "ZuCo2", "S2"),
            sample_ids=["sample-1"],
            source_dataframe_row_indices=[17],
        )
        self.assertEqual(output["sample_id"], ["sample-1"])
        self.assertEqual(output["source_dataframe_row_index"], [17])
        self.assertEqual(tuple(output["eeg_tokens"].shape), (1, 8, 4))
        self.assertEqual(tuple(output["eeg_vector"].shape), (1, 4))


if __name__ == "__main__":
    unittest.main()
