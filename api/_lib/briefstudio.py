"""Core plan-building logic for Nemotron Brief Studio.

Two-stage pipeline, both stages on NVIDIA Nemotron open models served by
Nebius Token Factory:

1. Reasoning stage (large Nemotron model) decomposes the brief into goals,
   scope, milestones, risks and open questions.
2. Metadata stage (small Nemotron model) drafts the title, one-line summary,
   tags and a per-milestone size estimate.

The result carries a per-stage attribution block naming exactly which model
produced which part, because provenance is a first-class feature here.
"""

import json
import re

MAX_BRIEF = 20000
MAX_TITLE = 120
MAX_TAGS = 8

VALID_CODES_FOR_SAMPLE = (
    "nebius_key_not_configured",
    "nebius_upstream_auth",
    "nebius_upstream_rate_limited",
    "nebius_upstream_error",
    "nebius_bad_request",
    "malformed_model_output",
)


class BriefError(ValueError):
    """Client-side input problem (never triggers sample mode)."""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

PLAN_SCHEMA_DOC = """{
  "goals": ["string", "..."],
  "scope": {"in_scope": ["string", "..."], "out_of_scope": ["string", "..."]},
  "milestones": [{"title": "string", "detail": "string", "depends_on": ["string", "..."]}],
  "risks": [{"title": "string", "mitigation": "string"}],
  "open_questions": ["string", "..."]
}"""

META_SCHEMA_DOC = """{
  "title": "string (max 80 chars)",
  "summary": "one sentence",
  "tags": ["string", "..."],
  "estimates": [{"milestone_title": "string", "size": "S|M|L"}]
}"""


def reasoning_messages(brief, depth):
    detail = ("Aim for thorough coverage: 4-8 goals, 4-10 milestones with "
              "concrete detail, and named dependencies between milestones."
              if depth == "deep" else
              "Aim for a crisp first-pass plan: 3-6 goals, 3-6 milestones, "
              "short details.")
    return [
        {"role": "system",
         "content": (
             "You are a senior project planner. Decompose the user's brief "
             "into a structured plan. Reply with ONLY valid JSON matching "
             "this schema, no prose before or after, no markdown fences:\n"
             + PLAN_SCHEMA_DOC
             + "\nRules: goals are outcomes, not tasks. scope.out_of_scope "
               "states explicit exclusions. Each milestone is a deliverable "
               "chunk small enough to finish in one focused sitting. "
               "risks.mitigation must be an action, not a worry. "
               "open_questions lists what you would ask the stakeholder.")},
        {"role": "user",
         "content": "Brief:\n\n" + brief.strip() + "\n\n" + detail},
    ]


def meta_messages(brief, milestones):
    titles = "\n".join("- " + m.get("title", "") for m in milestones)
    return [
        {"role": "system",
         "content": (
             "You write tight product metadata. Reply with ONLY valid JSON "
             "matching this schema, no prose, no markdown fences:\n"
             + META_SCHEMA_DOC
             + "\nRules: title is plain and human (no colons, no buzzwords). "
               "summary is ONE sentence under 30 words. tags are lowercase, "
               "3-6 items. size is S (hours), M (1-3 days) or L (over 3 days) "
               "for each milestone, matched by exact milestone_title.")},
        {"role": "user",
         "content": ("Brief:\n" + brief.strip()[:1500]
                     + "\n\nMilestone titles:\n" + titles)},
    ]


# ---------------------------------------------------------------------------
# Model output parsing
# ---------------------------------------------------------------------------

class MalformedModelOutput(ValueError):
    """The model did not return usable JSON (triggers one repair retry)."""


def parse_model_json(text):
    """Parse strict JSON out of a model reply; tolerate one fenced block."""
    if not text or not text.strip():
        raise MalformedModelOutput("empty completion")
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        raise MalformedModelOutput("no JSON object found in completion")
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError as exc:
        raise MalformedModelOutput("JSON object could not be parsed: %s"
                                   % exc) from exc


def _as_str_list(value, limit=12):
    out = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip()[:300])
            elif isinstance(item, dict):
                text = item.get("title") or item.get("text") or ""
                if isinstance(text, str) and text.strip():
                    out.append(text.strip()[:300])
            if len(out) >= limit:
                break
    return out


def _as_pairs(value, limit=12):
    """Coerce a list into {title, mitigation/detail} dicts."""
    out = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                title = str(item.get("title") or item.get("name")
                            or "").strip()
                detail = str(item.get("mitigation") or item.get("detail")
                             or item.get("description") or "").strip()
            elif isinstance(item, str):
                title, detail = item.strip(), ""
            else:
                continue
            if title:
                out.append({"title": title[:200],
                            "mitigation": detail[:400]})
            if len(out) >= limit:
                break
    return out


def normalize_decomposition(raw):
    """Validate and coerce the reasoning-stage JSON into the canonical shape."""
    if not isinstance(raw, dict):
        raise MalformedModelOutput("decomposition is not an object")
    goals = _as_str_list(raw.get("goals"))
    if not goals:
        raise MalformedModelOutput("decomposition has no usable goals")
    scope_raw = raw.get("scope") if isinstance(raw.get("scope"), dict) else {}
    scope = {
        "in_scope": _as_str_list(scope_raw.get("in_scope")),
        "out_of_scope": _as_str_list(scope_raw.get("out_of_scope")),
    }
    milestones = []
    for pair in _as_pairs(raw.get("milestones")):
        milestones.append({
            "title": pair["title"],
            "detail": pair.get("mitigation", ""),
            "depends_on": [],
        })
    raw_ms = raw.get("milestones")
    if isinstance(raw_ms, list):
        for item, fixed in zip(raw_ms, milestones):
            if isinstance(item, dict) and isinstance(item.get("depends_on"),
                                                     list):
                fixed["depends_on"] = _as_str_list(item["depends_on"], 6)
    if not milestones:
        raise MalformedModelOutput("decomposition has no usable milestones")
    risks = _as_pairs(raw.get("risks"))
    questions = _as_str_list(raw.get("open_questions"))
    return {
        "goals": goals,
        "scope": scope,
        "milestones": milestones,
        "risks": risks,
        "open_questions": questions,
    }


def normalize_metadata(raw, milestones, brief):
    """Validate the metadata-stage JSON; fall back to brief-derived defaults."""
    meta = raw if isinstance(raw, dict) else {}
    title = str(meta.get("title") or "").strip()[:MAX_TITLE]
    if not title:
        words = re.sub(r"\s+", " ", brief.strip()).split(" ")[:9]
        title = (" ".join(words) or "Untitled brief")
    summary = str(meta.get("summary") or "").strip()[:400]
    tags = [t.lower()[:24] for t in _as_str_list(meta.get("tags"), MAX_TAGS)]
    if not tags:
        tags = ["plan"]
    estimates = {}
    raw_est = meta.get("estimates")
    if isinstance(raw_est, list):
        for item in raw_est:
            if isinstance(item, dict):
                key = str(item.get("milestone_title") or "").strip()
                size = str(item.get("size") or "").strip().upper()
                if key and size in ("S", "M", "L"):
                    estimates[key] = size
    for ms in milestones:
        ms["size"] = estimates.get(ms["title"], "")
    return {
        "title": title,
        "summary": summary,
        "tags": tags[:MAX_TAGS],
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble(decomp, meta, reasoning_model, fast_model):
    """Final API result with explicit per-stage model attribution."""
    return {
        "title": meta["title"],
        "summary": meta["summary"],
        "tags": meta["tags"],
        "goals": decomp["goals"],
        "scope": decomp["scope"],
        "milestones": decomp["milestones"],
        "risks": decomp["risks"],
        "open_questions": decomp["open_questions"],
        "attribution": {
            "decomposition_by": reasoning_model,
            "metadata_by": fast_model,
            "platform": "Nebius Token Factory",
        },
    }


def sample_payload(brief):
    """Deterministic illustrative sample, rendered client-side only when the
    live service is unavailable. Clearly labelled in the UI; never sent by the
    API as if it were model output."""
    snippet = re.sub(r"\s+", " ", (brief or "").strip())[:60]
    title = (snippet + ("..." if len(brief or "") > 60 else "")) or "Sample brief"
    return {
        "title": title or "Sample brief",
        "summary": ("Illustrative example of the plan shape. The live service "
                    "is not connected, so this sample stands in - it is not "
                    "model output."),
        "tags": ["sample", "illustrative", "plan"],
        "goals": [
            "Stand-in goal: show how a brief becomes structured outcomes",
            "Stand-in goal: demonstrate per-section model attribution",
        ],
        "scope": {
            "in_scope": ["Sample in-scope item one",
                         "Sample in-scope item two"],
            "out_of_scope": ["Sample exclusion (this is illustrative)"],
        },
        "milestones": [
            {"title": "Sample milestone one", "detail": "Illustrative detail.",
             "depends_on": [], "size": "S"},
            {"title": "Sample milestone two", "detail": "Illustrative detail.",
             "depends_on": ["Sample milestone one"], "size": "M"},
        ],
        "risks": [{"title": "Sample risk", "mitigation":
                   "Illustrative mitigation step."}],
        "open_questions": ["Sample open question (illustrative)"],
        "attribution": {
            "decomposition_by": "sample - not model output",
            "metadata_by": "sample - not model output",
            "platform": "none (sample mode)",
        },
        "sample_mode": True,
    }


def validate_brief(body):
    """Parse and validate the request body. Raises BriefError on bad input."""
    if not isinstance(body, dict):
        raise BriefError("request body must be a JSON object")
    brief = body.get("brief")
    if not isinstance(brief, str) or not brief.strip():
        raise BriefError("brief is required and must be a non-empty string")
    if len(brief) > MAX_BRIEF:
        raise BriefError("brief is too long (max %d characters)" % MAX_BRIEF)
    depth = body.get("depth", "standard")
    if depth not in ("standard", "deep"):
        raise BriefError("depth must be 'standard' or 'deep'")
    return brief.strip(), depth
