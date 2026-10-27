"""Single Vercel entrypoint routing /api/health and /api/plan.

The Vercel Python runtime builds ONE Python app per project and needs the
entrypoint declared in pyproject.toml ([tool.vercel] entrypoint = "api.app:handler").
Static files (index.html) are served by the CDN; everything under /api/*
lands here.

Business logic lives in _lib (tf_client + briefstudio); this module only
routes, validates input, maps errors to codes and shapes responses.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from http.server import BaseHTTPRequestHandler  # noqa: E402

from _lib.tf_client import (  # noqa: E402
    DEFAULT_BASE_URL,
    NebiusAuthError,
    NebiusBadRequest,
    NebiusError,
    NebiusRateLimited,
    NebiusUnavailable,
    chat,
    list_models,
    resolve_models,
)
from _lib.briefstudio import (  # noqa: E402
    BriefError,
    MalformedModelOutput,
    assemble,
    meta_messages,
    normalize_decomposition,
    normalize_metadata,
    parse_model_json,
    reasoning_messages,
    validate_brief,
)

MAX_BODY = 24000
REASONING_TOKENS = {"standard": 1600, "deep": 2600}
SERVICE_NAME = "nemotron-brief-studio"


# ---------------------------------------------------------------------------
# Pure request logic (no HTTP wiring) - directly unit-testable
# ---------------------------------------------------------------------------

def health_response(environ=None):
    return 200, {
        "status": "ok",
        "service": SERVICE_NAME,
        "key_configured": bool(os.environ.get("NEBIUS_API_KEY")),
        "base_url": os.environ.get("NEBIUS_BASE_URL", DEFAULT_BASE_URL),
    }


def plan_response(body):
    """Full POST /api/plan logic. Returns (http_code, payload)."""
    try:
        brief, depth = validate_brief(body)
    except BriefError as exc:
        return 400, {"error": "invalid_request", "message": str(exc)}

    nf_credentials = os.environ.get("NEBIUS_API_KEY")
    if not nf_credentials:
        return 503, {
            "error": "nebius_key_not_configured",
            "message": "The service is not connected to Nebius Token "
                       "Factory yet. Check back shortly.",
        }

    base_url = os.environ.get("NEBIUS_BASE_URL", DEFAULT_BASE_URL)
    try:
        reasoning_id, fast_id = resolve_models(
            list_models(base_url, nf_credentials),
            reasoning_hint=os.environ.get("NEBIUS_REASONING_MODEL"),
            fast_hint=os.environ.get("NEBIUS_FAST_MODEL"),
        )
        result = build_plan(base_url, nf_credentials, brief, depth,
                            reasoning_id, fast_id)
    except NebiusAuthError as exc:
        return 503, {"error": exc.code, "message": str(exc)}
    except NebiusRateLimited as exc:
        return 503, {"error": exc.code, "message": str(exc)}
    except NebiusBadRequest as exc:
        return 503, {"error": exc.code, "message": str(exc)}
    except NebiusUnavailable as exc:
        return 503, {"error": exc.code, "message": str(exc)}
    except MalformedModelOutput as exc:
        return 503, {"error": "malformed_model_output", "message": str(exc)}
    except NebiusError as exc:
        return 503, {"error": exc.code, "message": str(exc)}
    return 200, {"mode": "live", "result": result}


def build_plan(base_url, nf_credentials, brief, depth, reasoning_id, fast_id):
    """Two-stage pipeline: decompose on the large Nemotron model, metadata on
    the small one. One repair retry on stage 1; graceful fallback on stage 2."""
    # Stage 1: decomposition.
    r_msgs = reasoning_messages(brief, depth)
    decomp_raw = None
    for attempt in range(2):
        text = chat(base_url, nf_credentials, reasoning_id, r_msgs,
                    max_tokens=REASONING_TOKENS[depth], temperature=0.2)
        try:
            decomp_raw = parse_model_json(text)
            break
        except MalformedModelOutput:
            if attempt == 0:
                r_msgs = r_msgs + [
                    {"role": "assistant", "content": text[:500]},
                    {"role": "user",
                     "content": "That was not valid JSON. Reply again with "
                                "ONLY the JSON object, no prose, no fences."},
                ]
    if decomp_raw is None:
        raise MalformedModelOutput(
            "reasoning stage returned no parsable JSON after one retry")
    decomp = normalize_decomposition(decomp_raw)

    # Stage 2: metadata. Failure degrades gracefully - the plan stands on
    # defaults, never fails the request.
    try:
        meta_text = chat(base_url, nf_credentials, fast_id,
                         meta_messages(brief, decomp["milestones"]),
                         max_tokens=500, temperature=0.3)
        meta = normalize_metadata(parse_model_json(meta_text),
                                  decomp["milestones"], brief)
    except (MalformedModelOutput, NebiusError):
        meta = normalize_metadata({}, decomp["milestones"], brief)
        fast_id = fast_id + " (metadata fell back to defaults)"
    return assemble(decomp, meta, reasoning_id, fast_id)


# ---------------------------------------------------------------------------
# HTTP wiring
# ---------------------------------------------------------------------------

class handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/health":
            code, payload = health_response()
            self._send(code, payload)
        elif path == "/api/plan":
            self._send(405, {
                "error": "method_not_allowed",
                "message": "POST a JSON body: {\"brief\": \"...\", "
                           "\"depth\": \"standard\"|\"deep\"}.",
            })
        else:
            self._send(404, {"error": "not_found",
                             "message": "Unknown API path."})

    def do_POST(self):
        path = self.path.split("?")[0]
        if path != "/api/plan":
            self._send(404, {"error": "not_found",
                             "message": "Unknown API path."})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY:
            self._send(413, {"error": "payload_too_large",
                             "message": "Body must be 1-24000 bytes."})
            return
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self._send(400, {"error": "invalid_json",
                             "message": "Body must be valid JSON."})
            return
        code, payload = plan_response(body)
        self._send(code, payload)
