"""Frontend contract tests: parse index.html and pin the shipped UI shape.

These guard the parts judges and users actually see: the form, the sample-mode
labelling (never present live output as model output), the attribution badges
and the request target.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HTML = (ROOT / "public" / "index.html").read_text(encoding="utf-8")


class TestFrontendContract(unittest.TestCase):
    def test_form_posts_to_plan_endpoint(self):
        self.assertIn('fetch("/api/plan"', HTML)

    def test_brief_textarea_present_with_limit(self):
        self.assertIn('id="brief"', HTML)
        self.assertIn('maxlength="20000"', HTML)

    def test_depth_control_has_two_levels(self):
        self.assertIn('value="standard"', HTML)
        self.assertIn('value="deep"', HTML)

    def test_sample_banner_is_explicitly_labelled(self):
        self.assertIn("Sample mode.", HTML)
        self.assertIn("not</strong> model output", HTML)

    def test_sample_codes_list_matches_policy(self):
        # Client-side validation errors (400/413) must never trigger sample.
        self.assertIn('pair.code !== 400 && pair.code !== 413', HTML)
        self.assertIn("nebius_key_not_configured", HTML)
        self.assertIn("malformed_model_output", HTML)

    def test_attribution_badges_rendered(self):
        self.assertIn("decomposition_by", HTML)
        self.assertIn("metadata_by", HTML)
        self.assertIn('"badge"', HTML)

    def test_size_badges_rendered(self):
        self.assertIn('"size"', HTML)

    def test_no_em_or_en_dashes_in_copy(self):
        for glyph in ("\u2014", "\u2013"):
            self.assertNotIn(glyph, HTML)

    def test_no_literal_secrets(self):
        self.assertNotIn("Bearer ", HTML.replace('Bearer "', ""))
        self.assertFalse(re.search(r"(?i)(api[_-]?key\s*=\s*[\"'][^\"']{8,})",
                                   HTML), "hard-coded key found")


if __name__ == "__main__":
    unittest.main()
