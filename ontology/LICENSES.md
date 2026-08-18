# Ontology licenses

- `patterns.core.yaml` — MIT-compatible adaptations of editor skill lists.
- `patterns.rhetorical.yaml` — heuristic proxies for Reinhart et al. arXiv:2410.16107 / Biber dimensions. Pattern strings are functional.
- `patterns.wikipedia.yaml` — **CC BY-SA 4.0**. Derived from Wikipedia:Signs of AI writing. Share-alike applies to the descriptive `fix` blurbs in that file. Do not copy those blurbs into MIT source.
- `patterns.slop.yaml` — **Apache-2.0**. Generated from sam-paech/antislop-sampler (arXiv:2510.15061, ICLR 2026): top 300 over-used LLM phrases + 3 anti-slop regexes. Fix blurbs are generic. Regenerate with `scripts/emit_slop_ontology.py`.

Ids are append-only. Disable with `enabled: false`; never reuse a deleted id.
