"""Tests for the Token Factory client: error mapping and model resolution.

Network calls are mocked at urllib level, so these run with no credentials.
"""

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "api"))

from _lib import tf_client as tf  # noqa: E402
from _lib.tf_client import (  # noqa: E402
    NebiusAuthError,
    NebiusBadRequest,
    NebiusRateLimited,
    resolve_models,
)

MODELS = [
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/llama-3.1-nemotron-nano-8b-v1",
    "moonshotai/Kimi-K2.6",
    "deepseek-ai/DeepSeek-V3.2",
]


def fake_urlopen(status=200, payload=None):
    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(payload or {}).encode("utf-8")

    return mock.MagicMock(return_value=Resp())


class TestErrorMapping(unittest.TestCase):
    def run_call(self, code, payload=None):
        err = unittest.mock.Mock()
        err.read = lambda: (json.dumps(payload or {}).encode())
        error = tf.urllib.error.HTTPError(
            "url", code, "msg", hdrs=None, fp=None)
        # HTTPError doubles as the raised exception and the response object.
        with mock.patch.object(tf.urllib.request, "urlopen",
                               side_effect=error):
            with self.assertRaises(Exception) as ctx:
                tf.chat("https://x/v1/", "k", "m", [{"role": "user",
                                                     "content": "hi"}])
        return ctx.exception

    def test_401_maps_to_auth(self):
        exc = self.run_call(401)
        self.assertIsInstance(exc, NebiusAuthError)

    def test_429_maps_to_rate_limited(self):
        exc = self.run_call(429)
        self.assertIsInstance(exc, NebiusRateLimited)

    def test_400_maps_to_bad_request(self):
        exc = self.run_call(400)
        self.assertIsInstance(exc, NebiusBadRequest)

    def test_500_maps_to_unavailable(self):
        exc = self.run_call(500)
        self.assertIsInstance(exc, tf.NebiusUnavailable)

    def test_urlerror_maps_to_unavailable(self):
        with mock.patch.object(
                tf.urllib.request, "urlopen",
                side_effect=tf.urllib.error.URLError("no route")):
            with self.assertRaises(tf.NebiusUnavailable):
                tf.list_models("https://x/v1/", "k")


class TestChatShape(unittest.TestCase):
    def test_chat_sends_model_and_messages(self):
        captured = {}

        def resp_factory(req, timeout):
            captured["url"] = req.full_url
            captured["auth"] = req.headers.get("Authorization")
            captured["body"] = json.loads(req.data.decode("utf-8"))
            return fake_urlopen(payload={
                "choices": [{"message": {"content": "hello"}}]})()

        with mock.patch.object(tf.urllib.request, "urlopen",
                               side_effect=resp_factory):
            out = tf.chat("https://x/v1/", "secret-key", "nvidia/m",
                          [{"role": "user", "content": "hi"}],
                          max_tokens=64, temperature=0.1)
        self.assertEqual(out, "hello")
        self.assertEqual(captured["url"], "https://x/v1/chat/completions")
        self.assertEqual(captured["auth"], "Bearer secret-key")
        self.assertEqual(captured["body"]["model"], "nvidia/m")
        self.assertEqual(captured["body"]["max_tokens"], 64)

    def test_list_models_returns_ids(self):
        with mock.patch.object(
                tf.urllib.request, "urlopen",
                return_value=fake_urlopen(payload={"data": [
                    {"id": "a"}, {"id": "b"}]})()):
            self.assertEqual(tf.list_models("https://x/v1/", "k"), ["a", "b"])


class TestResolveModels(unittest.TestCase):
    def test_preferred_exact_match(self):
        r, f = resolve_models(MODELS)
        self.assertEqual(r, "nvidia/nemotron-3-super-120b-a12b")
        self.assertEqual(f, "nvidia/llama-3.1-nemotron-nano-8b-v1")

    def test_hint_overrides(self):
        models = MODELS + ["nvidia/custom-nemotron-42b"]
        r, f = resolve_models(models, reasoning_hint="nvidia/custom-nemotron-42b")
        self.assertEqual(r, "nvidia/custom-nemotron-42b")

    def test_nemotron_only_list_picks_biggest_class(self):
        r, f = resolve_models(["nvidia/nemotron-9b-nano", "other/model"])
        self.assertIn("nemotron", r.lower())
        self.assertEqual(r, "nvidia/nemotron-9b-nano")

    def test_no_nemotron_raises(self):
        with self.assertRaises(NebiusBadRequest):
            resolve_models(["openai/gpt-x", "other/model"])

    def test_small_falls_back_to_reasoning(self):
        r, f = resolve_models(["nvidia/nemotron-3-super-120b-a12b"])
        self.assertEqual(r, f)


if __name__ == "__main__":
    unittest.main()
