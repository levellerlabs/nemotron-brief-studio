"""Unit tests for the plan-building logic (no network, no API key needed)."""

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))

from _lib import briefstudio as bs  # noqa: E402
from _lib.briefstudio import (  # noqa: E402
    MalformedModelOutput,
    assemble,
    meta_messages,
    normalize_decomposition,
    normalize_metadata,
    parse_model_json,
    reasoning_messages,
    sample_payload,
    validate_brief,
)

GOOD_DECOMP = {
    "goals": ["Members can book slots online"],
    "scope": {"in_scope": ["slot picker"], "out_of_scope": ["mobile app"]},
    "milestones": [
        {"title": "Slot picker page", "detail": "Grid of open slots.",
         "depends_on": []},
        {"title": "Payment step", "detail": "Card checkout.",
         "depends_on": ["Slot picker page"]},
    ],
    "risks": [{"title": "No-shows", "mitigation": "Deposit on booking."}],
    "open_questions": ["Which payment provider?"],
}

GOOD_META = {
    "title": "Climbing gym booking page",
    "summary": "A booking flow with slot picking and card payment.",
    "tags": ["booking", "payments", "web"],
    "estimates": [
        {"milestone_title": "Slot picker page", "size": "M"},
        {"milestone_title": "Payment step", "size": "M"},
    ],
}


class TestValidateBrief(unittest.TestCase):
    def test_valid(self):
        brief, depth = validate_brief({"brief": "  build a kiosk app  ",
                                       "depth": "deep"})
        self.assertEqual(brief, "build a kiosk app")
        self.assertEqual(depth, "deep")

    def test_empty_rejected(self):
        with self.assertRaises(bs.BriefError):
            validate_brief({"brief": "   "})

    def test_missing_rejected(self):
        with self.assertRaises(bs.BriefError):
            validate_brief({})

    def test_too_long_rejected(self):
        with self.assertRaises(bs.BriefError):
            validate_brief({"brief": "x" * (bs.MAX_BRIEF + 1)})

    def test_bad_depth_rejected(self):
        with self.assertRaises(bs.BriefError):
            validate_brief({"brief": "hello", "depth": "ultra"})


class TestParseModelJson(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(parse_model_json('{"a": 1}'), {"a": 1})

    def test_fenced_json(self):
        self.assertEqual(parse_model_json('```json\n{"a": 1}\n```'), {"a": 1})

    def test_json_with_leading_prose(self):
        self.assertEqual(parse_model_json('Sure!\n{"a": 1}'), {"a": 1})

    def test_empty_raises(self):
        with self.assertRaises(MalformedModelOutput):
            parse_model_json("")

    def test_no_json_raises(self):
        with self.assertRaises(MalformedModelOutput):
            parse_model_json("I cannot help with that.")


class TestNormalizeDecomposition(unittest.TestCase):
    def test_good_shape_passes(self):
        out = normalize_decomposition(GOOD_DECOMP)
        self.assertEqual(out["goals"], ["Members can book slots online"])
        self.assertEqual(len(out["milestones"]), 2)
        self.assertEqual(out["milestones"][1]["depends_on"],
                         ["Slot picker page"])

    def test_no_goals_raises(self):
        with self.assertRaises(MalformedModelOutput):
            normalize_decomposition({"goals": [], "milestones": [{"title": "x"}]})

    def test_no_milestones_raises(self):
        with self.assertRaises(MalformedModelOutput):
            normalize_decomposition({"goals": ["g"], "milestones": []})

    def test_string_risks_coerced(self):
        out = normalize_decomposition(dict(GOOD_DECOMP,
                                           risks=["one flat risk"]))
        self.assertEqual(out["risks"][0]["title"], "one flat risk")


class TestNormalizeMetadata(unittest.TestCase):
    def test_sizes_matched_by_title(self):
        milestones = normalize_decomposition(GOOD_DECOMP)["milestones"]
        normalize_metadata(GOOD_META, milestones, "gym booking")
        self.assertEqual([m["size"] for m in milestones], ["M", "M"])

    def test_mismatched_estimates_degrade_to_blank(self):
        bad = dict(GOOD_META, estimates=[{"milestone_title": "nope", "size": "S"}])
        milestones = normalize_decomposition(GOOD_DECOMP)["milestones"]
        normalize_metadata(bad, milestones, "gym booking")
        self.assertTrue(all(m["size"] == "" for m in milestones))

    def test_fallback_title_from_brief(self):
        out = normalize_metadata({}, GOOD_DECOMP["milestones"],
                                 "Build a kiosk app for the office lobby")
        self.assertEqual(out["title"], "Build a kiosk app for the office lobby")
        self.assertEqual(out["tags"], ["plan"])


class TestAssemble(unittest.TestCase):
    def test_attribution_present(self):
        decomp = normalize_decomposition(GOOD_DECOMP)
        meta = normalize_metadata(GOOD_META, decomp["milestones"], "b")
        result = assemble(decomp, meta, "nvidia/big", "nvidia/small")
        self.assertEqual(result["attribution"]["decomposition_by"],
                         "nvidia/big")
        self.assertEqual(result["attribution"]["metadata_by"], "nvidia/small")
        self.assertEqual(result["attribution"]["platform"],
                         "Nebius Token Factory")


class TestSamplePayload(unittest.TestCase):
    def test_marked_and_shaped(self):
        sample = sample_payload("booking page for a gym")
        self.assertTrue(sample["sample_mode"])
        for key in ("title", "summary", "goals", "scope", "milestones",
                    "risks", "open_questions", "attribution"):
            self.assertIn(key, sample)
        self.assertIn("not model output", sample["summary"])


class TestPrompts(unittest.TestCase):
    def test_reasoning_prompt_carries_schema_and_brief(self):
        msgs = reasoning_messages("build a garden planner", "deep")
        self.assertIn('"goals"', msgs[0]["content"])
        self.assertIn("build a garden planner", msgs[1]["content"])
        self.assertIn("4-10 milestones", msgs[1]["content"])

    def test_meta_prompt_lists_milestones(self):
        msgs = meta_messages("brief", GOOD_DECOMP["milestones"])
        self.assertIn("Slot picker page", msgs[1]["content"])


if __name__ == "__main__":
    unittest.main()
