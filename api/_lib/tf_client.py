"""OpenAI-compatible client for Nebius Token Factory.

Token Factory serves open-weight models (including the NVIDIA Nemotron family)
behind an OpenAI-compatible REST API. This module is stdlib-only.

Model IDs change over time: the documented contract is to trust the live
``GET /models`` endpoint and resolve preferred candidates against it instead of
hard-coding one string (see ``resolve_models``).
"""

import json
import re
import urllib.error
import urllib.request

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

# Preferred candidates, first match wins. Verified against the live /models
# endpoint at integration time; never assumed.
PREFERRED_REASONING = [
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/llama-3.3-nemotron-super-49b-v1",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
]
PREFERRED_FAST = [
    "nvidia/llama-3.1-nemotron-nano-8b-v1",
    "nvidia/llama-3.1-nemotron-nano-4b-v1",
]

_RE_STRONG = re.compile(r"nemotron", re.I)
_RE_BIG = re.compile(r"nemotron.*(ultra|super|large|120b|253b|49b)", re.I)
_RE_SMALL = re.compile(r"nemotron.*(nano|mini|small|8b|4b)", re.I)


class NebiusError(Exception):
    """Base error for Token Factory client failures."""

    code = "nebius_upstream_error"


class NebiusAuthError(NebiusError):
    code = "nebius_upstream_auth"


class NebiusRateLimited(NebiusError):
    code = "nebius_upstream_rate_limited"


class NebiusUnavailable(NebiusError):
    code = "nebius_upstream_error"


class NebiusBadRequest(NebiusError):
    code = "nebius_bad_request"


def _headers(api_key):
    return {
        "Authorization": "Bearer " + api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _request(url, api_key, payload=None, timeout=45):
    """Single HTTP call against Token Factory. Returns parsed JSON."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=_headers(api_key),
                                 method="POST" if payload is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")[:300]
        except Exception:
            pass
        if exc.code in (401, 403):
            raise NebiusAuthError(
                "Token Factory rejected the credentials (HTTP %d)" % exc.code
            ) from exc
        if exc.code == 429:
            raise NebiusRateLimited(
                "Token Factory rate limited the request") from exc
        if 400 <= exc.code < 500:
            raise NebiusBadRequest(
                "Token Factory rejected the request (HTTP %d): %s"
                % (exc.code, body)) from exc
        raise NebiusUnavailable(
            "Token Factory unavailable (HTTP %d)" % exc.code) from exc
    except urllib.error.URLError as exc:
        raise NebiusUnavailable(
            "Token Factory not reachable: %s" % exc.reason) from exc
    except TimeoutError as exc:
        raise NebiusUnavailable("Token Factory timed out") from exc


def list_models(base_url, api_key, timeout=20):
    """Return the live model id list from GET /models."""
    data = _request(base_url.rstrip("/") + "/models", api_key, timeout=timeout)
    out = []
    for item in data.get("data", []) or []:
        mid = item.get("id") if isinstance(item, dict) else item
        if isinstance(mid, str):
            out.append(mid)
    return out


def chat(base_url, api_key, model, messages, max_tokens=1024, temperature=0.3,
         timeout=55):
    """One chat completion. Returns the assistant message content string."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    data = _request(base_url.rstrip("/") + "/chat/completions", api_key,
                    payload=payload, timeout=timeout)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise NebiusUnavailable(
            "Token Factory returned an unexpected completion shape") from exc
    if not isinstance(content, str):
        raise NebiusUnavailable("Token Factory returned a non-text completion")
    return content


def resolve_models(models, reasoning_hint=None, fast_hint=None):
    """Pick (reasoning_id, fast_id) from a live model list.

    Honours exact preferred matches first, then falls back to Nemotron family
    heuristics, then to the strongest available Nemotron id. Raises
    NebiusBadRequest when no Nemotron model is served at all (the hackathon
    rules require the Nemotron family, so running on anything else would be
    dishonest).
    """
    ids = list(models or [])
    if reasoning_hint and reasoning_hint in ids:
        reasoning = reasoning_hint
    else:
        reasoning = next((m for m in PREFERRED_REASONING if m in ids), None)
        if reasoning is None:
            strong = [m for m in ids if _RE_BIG.search(m)]
            reasoning = strong[0] if strong else None
        if reasoning is None:
            any_nemo = [m for m in ids if _RE_STRONG.search(m)]
            if any_nemo:
                reasoning = any_nemo[0]
    if fast_hint and fast_hint in ids:
        fast = fast_hint
    else:
        fast = next((m for m in PREFERRED_FAST if m in ids), None)
        if fast is None:
            small = [m for m in ids if _RE_SMALL.search(m)]
            if small:
                fast = small[0]
    if reasoning is None:
        raise NebiusBadRequest(
            "No NVIDIA Nemotron model found on this Token Factory endpoint")
    if fast is None or fast == reasoning:
        fast = reasoning
    return reasoning, fast
