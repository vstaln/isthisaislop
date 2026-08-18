# Hit schema (v1) — Is This AI Slop? (ITAIS)

The two lanes never merge. The renderer must not emit a percentage-of-AI string.

## Request

```json
{ "text": "string", "artifact_sha256": "optional expected manifest hash" }
```

## Response

```json
{
  "status": "ok | unverified_artifact",
  "lean": "slop | human | mixed | unclear",
  "hits": [
    {
      "id": "colon",
      "lane": "style",
      "unit": "span",
      "quote": "The best part: it learns.",
      "say": "Rewrite as a plain sentence.",
      "fix": "Rewrite as a plain sentence."
    }
  ],
  "why_slop": [],
  "why_human": [
    {
      "id": "weekday",
      "lane": "construction",
      "unit": "span",
      "quote": "Thursday",
      "say": "A named weekday is a specific time, not a template."
    }
  ],
  "sentences": [
    {
      "text": "Thursday mornings at the clinic were empty.",
      "lean": "human",
      "why_slop": [],
      "why_human": []
    }
  ],
  "style_summary": "Nothing matched.",
  "resemblance": {
    "label": "matches_ai_pile",
    "text": "Resembles the AI pile more than 94% of human reference texts."
  }
}
```

Rules:

- `status: unverified_artifact` → `hits: []`, `resemblance: null`. Do not fall back to regex-only.
- Empty `hits` → show **Nothing matched.** Never "This looks human."
- `why_human` names checkable cues (a weekday, a number, a contraction). That is not an authorship claim.
- The model emits a one-word `id` plus a quote. `say` / `fix` are hardcoded. Never LLM prose.
- Never "87% AI." Never "AI-generated." Never "written by ChatGPT."
- Highlight units: style = smallest span, resemblance = sentence, construction = paragraph or piece.
- No extension code in v1. This file is the frozen contract only.
