import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment as e


class FrozenProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data.csv"
        rng = np.random.default_rng(4)
        self.frame = pd.DataFrame(rng.normal(size=(120, 23)), columns=[f"f{i}" for i in range(23)])
        self.frame[e.core.TARGET] = np.tile([0, 1], 60)
        self.frame[e.core.ID] = np.arange(120)
        self.frame.to_csv(self.data, index=False)
        self.baseline = self.root / "baseline"
        self.baseline.mkdir()
        frame, columns, y = e.core.load_training(self.data)
        dev, holdout, folds, _ = e.core.make_splits(frame, columns, y, 0.2, 2, 42)
        ids = np.full(len(y), -1, dtype=int)
        for i, (_, valid) in enumerate(folds):
            ids[dev[valid]] = i
        pd.DataFrame({"row": np.arange(len(y)), e.core.ID: frame[e.core.ID],
                      "partition": np.where(ids == -1, "holdout", "development"),
                      "cv_fold": ids}).to_csv(self.baseline / "split.csv", index=False)
        e.save(self.baseline / "metrics.json", dict(data_sha256=e.sha(self.data),
                config=e.core.configs(1, 42)[0], arguments=dict(seed=42, valid_size=0.2, folds=2)))
        (self.baseline / "task1.pt").write_bytes(b"protected baseline sentinel")
        self.protocol = e.prepare(self.data, self.baseline, self.root / "protocol")
        self.path = self.root / "protocol" / "protocol.json"

    def args(self, seed=42, output="search"):
        return argparse.Namespace(data=str(self.data), protocol=str(self.path), config=None,
                                  output=str(self.root / output), model_seed=seed, epochs=1,
                                  patience=1, device="cpu", threads=1)

    def test_exact_original_split_preserved(self):
        self.assertEqual((self.baseline / "split.csv").read_bytes(),
                         (self.root / "protocol" / "split.csv").read_bytes())
        self.assertEqual(len(self.protocol["development_rows"]), 96)
        self.assertEqual(len(self.protocol["holdout_rows"]), 24)
        e.load_protocol(self.path, self.data)

    def test_seed_changes_only_model_and_never_touches_holdout(self):
        visits, seeds = [], []
        base = e.core.Preprocessor
        class Spy(base):
            def fit(self, frame):
                visits.append(("fit", frame.index.tolist()))
                return super().fit(frame)
            def transform(self, frame):
                visits.append(("transform", frame.index.tolist()))
                return super().transform(frame)
        def fake_fit(x, y, config, epochs, patience, seed, device, valid=None):
            self.assertIsNotNone(valid)
            seeds.append(seed)
            return None, 0.7, 1, [{"epoch": 1, "loss": 0.6, "auroc": 0.7}]
        digest = e.sha(self.path)
        with patch.object(e.core, "Preprocessor", Spy), patch.object(e.core, "fit_model", fake_fit), \
                patch.object(e.core, "probabilities", side_effect=AssertionError("Standalone holdout evaluation called")):
            e.search(self.args(42, "a"))
            first_visits = copy.deepcopy(visits)
            visits.clear()
            e.search(self.args(123, "b"))
        self.assertEqual(first_visits, visits)
        self.assertEqual(seeds, [42, 43, 123, 124])
        holdout = set(self.protocol["holdout_rows"])
        self.assertTrue(all(not holdout.intersection(rows) for _, rows in visits))
        self.assertEqual(digest, e.sha(self.path))
        for name in ("a", "b"):
            files = [p.name for p in (self.root / name).rglob("*") if p.is_file()]
            self.assertFalse(any(p.endswith(".pt") or p == "holdout_predictions.csv" for p in files))
            self.assertFalse(json.loads((self.root / name / "selection.json").read_text(encoding="utf-8"))["holdout_evaluated"])

    def test_changed_data_and_modified_protocol_rejected(self):
        changed = self.root / "changed.csv"
        self.frame.iloc[::-1].to_csv(changed, index=False)
        with self.assertRaisesRegex(ValueError, "Dataset content/order"):
            e.load_protocol(self.path, changed)
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        payload["split_seed"] = 999
        e.save(self.path, payload)
        with self.assertRaisesRegex(ValueError, "Protocol was modified"):
            e.load_protocol(self.path, self.data)

    def test_baseline_changes_and_output_overwrite_rejected(self):
        with self.assertRaisesRegex(ValueError, "new or empty"):
            e.prepare(self.data, self.baseline, self.root / "protocol")
        (self.baseline / "task1.pt").write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "Protected baseline"):
            e.load_protocol(self.path, self.data)

    def test_invalid_configs_rejected(self):
        config = copy.deepcopy(self.protocol["baseline_config"])
        config["dropout"] = float("nan")
        with self.assertRaises(ValueError):
            e.validate_config(config)
        config = copy.deepcopy(self.protocol["baseline_config"])
        config["unexpected"] = 1
        with self.assertRaises(ValueError):
            e.validate_config(config)

    def test_cv_partition_cannot_include_holdout(self):
        p = copy.deepcopy(self.protocol)
        p["cv_folds"][0]["valid_rows"][0] = p["holdout_rows"][0]
        frame, columns, y = e.core.load_training(self.data)
        with self.assertRaisesRegex(ValueError, "CV folds"):
            e.validate_partition(p, frame, columns, y)

    def test_actual_short_cv_run(self):
        e.torch.set_num_threads(1)
        before = {name: e.sha(self.baseline / name) for name in self.protocol["baseline_files_sha256"]}
        result = e.search(self.args())
        self.assertTrue(0 <= result["selected_cv_auroc"] <= 1)
        self.assertFalse(result["holdout_evaluated"])
        self.assertFalse((self.root / "search" / "task1.pt").exists())
        self.assertEqual(before, {name: e.sha(self.baseline / name) for name in before})


if __name__ == "__main__":
    unittest.main()
