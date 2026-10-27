# Nemotron Brief Studio

Paste a rough project brief, get a structured plan: goals, scope, milestones
with size estimates, risks with mitigations, and open questions. Every section
is labelled with the model that produced it.

Built on NVIDIA Nemotron open models served by Nebius Token Factory
(OpenAI-compatible endpoint). Two-stage pipeline:

1. **Decomposition** - a large Nemotron model turns the brief into structured
   JSON (goals, scope, milestones, risks, open questions).
2. **Metadata** - a small Nemotron model drafts the title, one-line summary,
   tags and a per-milestone S/M/L size estimate.

The frontend renders per-section attribution badges naming the exact model id
behind each part, so a reader always knows which model produced what.

## Where Token Factory is used

- `api/_lib/tf_client.py` talks to `https://api.tokenfactory.nebius.com/v1/`
  (`/models` and `/chat/completions`) with a bearer API key.
- `api/plan.py` runs both stages over that endpoint on every request.
- Model ids are resolved against the **live `/models` list** at request time
  (preferred Nemotron candidates first, family heuristics second) instead of
  being hard-coded, because catalogue ids change.

## Setup

```bash
# environment
export NEBIUS_API_KEY=nf-...        # Token Factory API key
# optional overrides
export NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1/
export NEBIUS_REASONING_MODEL=nvidia/nemotron-3-super-120b-a12b
export NEBIUS_FAST_MODEL=nvidia/llama-3.1-nemotron-nano-8b-v1

# run the tests (stdlib only, no network, no key needed)
python3 -m unittest discover -s tests -v

# deploy (Vercel Python runtime - single entrypoint, see pyproject.toml)
npx vercel deploy --prod --yes
```

No third-party Python packages are required: the API and tests are stdlib-only
Python 3.9+. The Vercel Python runtime builds the project as a single Python
app with the entrypoint declared in `pyproject.toml`
(`api.app:handler` routes `/api/health` and `/api/plan`; static files are
served by the CDN).

## Sample mode

If the service is not connected (no API key configured, upstream auth, rate
limit, upstream outage or unparsable model output), the API returns an HTTP
503 with a machine-readable error code. The frontend then renders a clearly
labelled illustrative sample so the interface shape stays visible - it never
presents sample content as model output, and genuine client input errors
(HTTP 400/413) never trigger the sample path.

## API

- `POST /api/plan` - body `{"brief": "...", "depth": "standard"|"deep"}`.
  Returns `{"mode": "live", "result": {...}}` where `result.attribution`
  names the model behind each stage.
- `GET /api/health` - liveness plus whether the key is configured.

## Limits

- Briefs up to 20,000 characters.
- The metadata stage degrades gracefully: if the small model fails, the plan
  still ships with brief-derived defaults and the attribution states it.
- One repair retry if the reasoning stage returns unparsable JSON.

## License

MIT - see LICENSE.
