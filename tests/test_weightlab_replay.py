"""Small deterministic fixtures for evidence boundaries, never navigation results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from weightlab.replay import BIAS_KEY, WEIGHT_KEY, ReplayValidationError, evaluate_head_replay


class HeadReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = self.root / "base.npz"
        self.patch = self.root / "patch.npz"
        self.batch = self.root / "batch.npz"
        self.output = self.root / "result"
        self.w = np.zeros((4, 512), dtype=np.float32)
        self.b = np.zeros(4, dtype=np.float32)
        self.h = np.zeros((3, 512), dtype=np.float32)
        self.batch_values = {
            "hidden": self.h, "scene_ids": np.asarray(["scene_a"] * 3),
            "episode_ids": np.asarray(["0", "0", "1"]),
            "split": np.asarray("diagnosis"), "step_ids": np.asarray([0, 1, 0]),
        }
        self.write_base()
        self.write_batch()

    def write_base(self, **extra):
        np.savez(self.base, **{WEIGHT_KEY: self.w, BIAS_KEY: self.b, **extra})

    def write_batch(self, **updates):
        self.batch_values.update(updates)
        np.savez(self.batch, **self.batch_values)

    def run_replay(self, patch=None, **kwargs):
        opts = {"allowed_keys": [("scene_a", "0"), ("scene_a", "1")],
                "expected_split": "diagnosis", **kwargs}
        return evaluate_head_replay(self.base, patch, self.batch, self.output, **opts)

    def rejected(self, pattern, patch=None, **kwargs):
        with self.assertRaisesRegex(ReplayValidationError, pattern):
            self.run_replay(patch, **kwargs)
        self.assertFalse(self.output.exists(), "Invalid input must not become a feedback artifact")

    def test_no_patch_is_identity_not_navigation(self):
        feedback = self.run_replay()
        self.assertTrue(feedback["no_model_navigation_executed"])
        self.assertIsNone(feedback["closed_loop_navigation_metrics"])
        self.assertEqual(feedback["metrics"]["mean_kl_base_to_candidate"], 0)
        self.assertEqual(feedback["metrics"]["argmax_action_change_count"], 0)
        self.assertIsNone(feedback["supervised_labels"])
        self.assertEqual(feedback["unique_episodes"], 2)
        self.assertEqual(feedback["counting"]["closed_loop_episodes"], 0)
        self.assertEqual(json.loads((self.output / "feedback.json").read_text()), feedback)

    def test_absolute_bias_replacement_and_known_distribution(self):
        self.b[:] = [0, 2, 0, 0]
        self.write_base()
        np.savez(self.patch, **{BIAS_KEY: np.asarray([0, 0, 3, 0], dtype=np.float32)})
        self.write_batch(target_actions=np.asarray([2, 2, 2]), actions=np.asarray([1, 1, 1]))
        feedback = self.run_replay(self.patch)
        with np.load(self.output / "paired_outputs.npz", allow_pickle=False) as result:
            np.testing.assert_array_equal(result["candidate_logits"], np.tile([0, 0, 3, 0], (3, 1)))
        self.assertEqual(feedback["metrics"]["argmax_action_change_count"], 3)
        self.assertAlmostEqual(feedback["supervised_labels"]["candidate_cross_entropy"],
                               float(np.log(np.exp(3) + 3) - 3), places=12)
        self.assertEqual(feedback["supervised_labels"]["candidate_top1_label_agreement"], 1)
        self.assertIn("not navigation gain", feedback["supervised_labels"]["interpretation"])
        self.assertIn("not correctness", feedback["recorded_actions"]["interpretation"])

    def test_rank_one_exactly_matches_external_linear_correction(self):
        rng = np.random.default_rng(21)
        self.w = rng.normal(size=(4, 512)).astype(np.float32) * .01
        self.h[:] = rng.normal(size=self.h.shape).astype(np.float32)
        self.write_base()
        self.write_batch()
        u = np.asarray([.1, -.2, .3, 0], dtype=np.float32)
        v = rng.normal(size=512).astype(np.float32) * .01
        np.savez(self.patch, **{WEIGHT_KEY: self.w + np.outer(u, v)})
        self.run_replay(self.patch)
        with np.load(self.output / "paired_outputs.npz", allow_pickle=False) as result:
            expected = result["base_logits"] + np.outer(self.h @ v, u)
            np.testing.assert_allclose(result["candidate_logits"], expected, atol=1e-6, rtol=1e-5)

    def test_recorded_logits_verified_with_explicit_float32_tolerance(self):
        self.write_batch(base_logits=np.full((3, 4), 5e-6, dtype=np.float32))
        feedback = self.run_replay()
        self.assertTrue(feedback["recorded_base_logits_check"]["verified"])

    def test_mismatched_recorded_logits_rejected(self):
        self.write_batch(base_logits=np.ones((3, 4), dtype=np.float32))
        self.rejected("Recorded base_logits")

    def test_recorded_logits_wrong_dtype_rejected(self):
        self.write_batch(base_logits=np.zeros((3, 4), dtype=np.float64))
        self.rejected("base_logits must be float32")

    def test_unknown_episode_key_rejected(self):
        self.write_batch(episode_ids=np.asarray(["0", "0", "secret_test_episode"]))
        self.rejected("outside the allowed manifest")

    def test_allowed_keys_are_mandatory_and_nonempty(self):
        with self.assertRaises(TypeError):
            evaluate_head_replay(self.base, None, self.batch, self.output, expected_split="diagnosis")
        self.rejected("cannot be empty", allowed_keys=[])

    def test_mixed_split_rejected_even_when_episode_ids_allowed(self):
        self.write_batch(split=np.asarray(["diagnosis", "diagnosis", "confirmation"]))
        self.rejected("Mixed splits")

    def test_wrong_single_split_rejected(self):
        self.write_batch(split=np.asarray("confirmation"))
        self.rejected("unexpected split")

    def test_unknown_batch_metrics_reject_fake_navigation_claim(self):
        self.write_batch(navigation_success=np.asarray(1.0))
        self.rejected("Unknown batch arrays")

    def test_nonhead_edit_cannot_be_replayed_on_cached_features(self):
        np.savez(self.patch, **{"net.state_encoder.rnn.weight_hh_l0": np.zeros((3, 3), dtype=np.float32)})
        self.rejected("only absolute head", self.patch)

    def test_nonhead_base_rejected_instead_of_silently_ignored(self):
        self.write_base(**{"net.visual_fc.0.weight": np.zeros((4, 4), dtype=np.float32)})
        self.rejected("exactly the two head")

    def test_nonfinite_features_rejected(self):
        self.h[0, 0] = np.nan
        self.write_batch()
        self.rejected("hidden contains nonfinite")

    def test_nonfinite_patch_rejected(self):
        np.savez(self.patch, **{BIAS_KEY: np.asarray([np.inf, 0, 0, 0], dtype=np.float32)})
        self.rejected("nonfinite", self.patch)

    def test_float32_forward_overflow_rejected(self):
        self.w.fill(np.finfo(np.float32).max)
        self.h.fill(2)
        self.write_base()
        self.write_batch()
        self.rejected("forward produced nonfinite")

    def test_pickle_metadata_is_never_loaded(self):
        self.write_batch(scene_ids=np.asarray(["scene_a"] * 3, dtype=object))
        self.rejected("object/pickle")

    def test_duplicate_timestep_rejected(self):
        self.write_batch(step_ids=np.asarray([0, 0, 0]))
        self.rejected("Duplicate")

    def test_action_labels_out_of_range_rejected(self):
        self.write_batch(target_actions=np.asarray([0, 4, 1]))
        self.rejected("outside \\[0, 3\\]")

    def test_batch_cap_is_enforced(self):
        self.rejected("exceeds the frozen call cap", max_batch_size=2)

    def test_inputs_unchanged_and_artifact_hash_matches(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.base, self.batch)}
        feedback = self.run_replay()
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})
        artifact = feedback["artifacts"]["paired_outputs"]
        self.assertEqual(artifact["sha256"], hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest())
        self.assertEqual(feedback["input_provenance"]["base"]["sha256"], before[self.base])
        with self.assertRaisesRegex(ReplayValidationError, "must not already exist"):
            self.run_replay()


if __name__ == "__main__":
    unittest.main()
