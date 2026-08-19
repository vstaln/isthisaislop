"""One-word tags. The model emits an id + a quote. Copy never comes from the model.

Style hits come from the ontology regex. Construction hits come from heuristics
or the model. Both surface as a one-word id plus hardcoded say.
"""

from __future__ import annotations

from typing import Any

# Closed set the labeler may emit. Aliases collapse old snake_case / plural forms.
ALIAS = {
    "glue_ban": "glue",
    "glue_bans": "glue",
    "tapestry": "puffery",
    "delve": "glue",
    "leverage": "glue",
    "robust": "puffery",
    "seamless": "puffery",
    "pivotal": "puffery",
    "stock_frame": "frames",
    "stock_frames": "frames",
    "em_dash": "emdash",
    "em_dashes": "emdash",
    "body_cliche": "cliche",
    "body_cliches": "cliche",
    "thematic_explicitness": "moral",
    "narrator_commentary": "gloss",
    "tidy_ending": "recap",
    "even_shape": "even",
    "time_jumps": "jump",
    "named_anchors": "anchor",
    "unresolved": "open",
    "burstiness": "burst",
    "human_weekday": "weekday",
    "human_clock": "clock",
    "human_daypart": "daypart",
    "human_contraction": "spoken",
    "human_number": "number",
    "human_proper": "name",
    "human_first_person": "first",
    "human_burstiness": "burst",
    "human_adjacent_contrast": "contrast",
    "construction_recap": "recap",
    "construction_over_explain": "gloss",
    "construction_flat_rhythm": "even",
}

# Ontology regex id → one-word tag. Prefix rules cover the long tails.
STYLE_MAP = {
    "binary_contrast": "notx",
    "throat_clearing": "opener",
    "faux_insight": "insight",
    "colon_reveal": "colon",
    "superficial_analysis": "gloss",
    "importance_puffery": "puffery",
    "interpretive_metadiscourse": "gloss",
    "weasel_attribution": "weasel",
    "fake_strong_verb": "copula",
    "synonym_cycling": "synonyms",
    "negative_listing": "notx",
    "dramatic_fragmentation": "even",
    "robotic_rhythm": "even",
    "rhetorical_setup": "setup",
    "fake_profound_kicker": "kicker",
    "recap_ending": "recap",
    "formatting_slop": "format",
    "em_dash_cluster": "emdash",
    "ban_delve_class": "glue",
    "ban_puffery_noun": "puffery",
    "ban_puffery_adj": "puffery",
    "ban_journey_verb": "glue",
    "ban_showcase_verb": "glue",
    "ban_corporate": "corporate",
    "ban_game_changer": "puffery",
    "empty_adverb": "adverb",
    "rule_of_three": "three",
    "parataxis": "even",
    "hedging_seesaw": "hedge",
    "corporate_pep_talk": "corporate",
    "bullet_overuse": "bullets",
    "as_role_opener": "opener",
    "passive_construction": "passive",
    "mandatory_paragraph_transition": "transition",
    "punct_em_dash_budget": "emdash",
    "punct_exclamation_budget": "bang",
    "punct_ellipsis_budget": "ellipsis",
    "emoji_bullet": "format",
    "hashtag_stack": "format",
    "markdown_in_plain": "format",
    "copula_avoidance_surface": "copula",
    "in_the_realm_of": "puffery",
    "a_testament_to": "puffery",
    "rest_assured": "corporate",
    "it_goes_without_saying": "frames",
    "in_essence": "frames",
    "please_note_that": "frames",
    "as_mentioned_earlier": "gloss",
    "in_todays_digital_age": "frames",
    "phrase_unlock_the_power": "puffery",
    "phrase_bridge_the_gap": "puffery",
    "phrase_i_hope_this_helps": "recap",
    "phrase_i_hope_this_finds_you": "opener",
    "phrase_lets_dive_in": "opener",
    "phrase_without_further_ado": "opener",
    "phrase_in_a_nutshell": "recap",
    "phrase_please_dont_hesitate": "corporate",
    "phrase_going_forward": "corporate",
    "phrase_not_just_x_but_y": "notx",
    "phrase_firstly_secondly_thirdly": "three",
    "phrase_in_order_to": "to",
    "phrase_in_terms_of": "regard",
    "phrase_with_regard_to": "regard",
    "wiki_significance_puffery": "puffery",
    "wiki_promotional_tone": "puffery",
    "wiki_ai_vocabulary_cluster": "puffery",
    "wiki_valuable_insights": "puffery",
    "wiki_copula_avoidance": "copula",
    "wiki_negative_parallelism": "notx",
    "uniform_sentence_length": "even",
    "identical_paragraph_structure": "even",
    "cross_section_parallelism": "even",
    "rhet_present_participial": "gloss",
    "rhet_adj_stacking": "stack",
    "rhet_that_complement": "that",
}

_PREFIX = (
    ("slop_phrase_", "cliche"),
    ("slop_not_x", "notx"),
    ("slop_each_", "three"),
    ("slop_every_", "three"),
    ("opener_", "opener"),
    ("punct_em_dash", "emdash"),
    ("phrase_", "frames"),
    ("wiki_", "wiki"),
)

COPY: dict[str, dict[str, str]] = {
    "glue": {"lean": "slop", "lane": "style", "say": "Stock verb. Name the action."},
    "frames": {"lean": "human", "lane": "style", "say": "Discourse frame — a human academic habit (in this article, in essence)."},
    "emdash": {"lean": "slop", "lane": "style", "say": "Em-dash cluster. Use a period or a comma."},
    "cliche": {"lean": "slop", "lane": "style", "say": "Body cliche. Write the actual beat."},
    "puffery": {"lean": "slop", "lane": "style", "say": "Cut the adjective, or name the property."},
    "opener": {"lean": "slop", "lane": "style", "say": "Cut the opener. Start on the point."},
    "weasel": {"lean": "human", "lane": "style", "say": "Attributed claim, or source-less hedging that human writers use."},
    "colon": {"lean": "slop", "lane": "style", "say": "Rewrite as a plain sentence."},
    "insight": {"lean": "slop", "lane": "style", "say": "Drop the lone-expert setup."},
    "corporate": {"lean": "slop", "lane": "style", "say": "Say the work in ordinary words."},
    "notx": {"lean": "slop", "lane": "style", "say": "State Y. Drop the not-X frame."},
    "three": {"lean": "slop", "lane": "style", "say": "Break the default trio."},
    "adverb": {"lean": "slop", "lane": "style", "say": "Cut the adverb if it adds nothing."},
    "copula": {"lean": "slop", "lane": "style", "say": "Use is or has when that is what you mean."},
    "setup": {"lean": "slop", "lane": "style", "say": "Drop the setup. Make the point."},
    "kicker": {"lean": "slop", "lane": "style", "say": "Delete the mic-drop. End on the last fact."},
    "format": {"lean": "slop", "lane": "style", "say": "Write prose. Drop the decoration."},
    "hedge": {"lean": "slop", "lane": "style", "say": "Pick a side."},
    "bullets": {"lean": "slop", "lane": "style", "say": "Turn the list into sentences."},
    "transition": {"lean": "slop", "lane": "style", "say": "Let some paragraphs just stop."},
    "passive": {"lean": "human", "lane": "style", "say": "Passive reporting is a human academic convention here."},
    "synonyms": {"lean": "slop", "lane": "style", "say": "Repeat the clear word."},
    "wiki": {"lean": "slop", "lane": "style", "say": "Drop the brochure. Name a fact."},
    "stack": {"lean": "slop", "lane": "style", "say": "Keep one modifier that earns its place."},
    "bang": {"lean": "slop", "lane": "style", "say": "Cut the exclamation."},
    "ellipsis": {"lean": "slop", "lane": "style", "say": "One ellipsis, only for a real trailing-off."},
    "to": {"lean": "slop", "lane": "style", "say": "Rewrite as to plus the verb."},
    "regard": {"lean": "slop", "lane": "style", "say": "Name the topic and continue."},
    "that": {"lean": "slop", "lane": "rhetorical", "say": "Keep that when it prevents a garden path."},
    "moral": {"lean": "slop", "lane": "construction", "say": "States the lesson. Let the fact stand."},
    "gloss": {"lean": "slop", "lane": "construction", "say": "Explains the point after making it. Cut it."},
    "recap": {"lean": "slop", "lane": "construction", "say": "Wraps up. Stop on the last concrete point."},
    "even": {"lean": "slop", "lane": "construction", "say": "Same sentence shape repeated. Vary the pace."},
    "jump": {"lean": "human", "lane": "construction", "say": "Time jump. Templates usually stay linear."},
    "anchor": {"lean": "human", "lane": "construction", "say": "A named time, place, person, or number."},
    "open": {"lean": "human", "lane": "construction", "say": "Leaves it unresolved. No tidy close."},
    "burst": {"lean": "human", "lane": "construction", "say": "Short sentence next to a longer one."},
    "weekday": {
        "lean": "human",
        "lane": "construction",
        "say": "A named weekday is a specific time, not a template.",
    },
    "clock": {"lean": "human", "lane": "construction", "say": "A clock time is an anchor, not a vibe."},
    "daypart": {
        "lean": "human",
        "lane": "construction",
        "say": "A time of day attached to a place or habit.",
    },
    "spoken": {
        "lean": "human",
        "lane": "construction",
        "say": "A spoken contraction, not polished copula-avoidance.",
    },
    "number": {"lean": "human", "lane": "construction", "say": "A number the sentence is actually using."},
    "name": {
        "lean": "human",
        "lane": "construction",
        "say": "A mid-sentence proper name. Templates rarely need one.",
    },
    "first": {
        "lean": "human",
        "lane": "construction",
        "say": "First person on a short claim, not a role-opener.",
    },
    "contrast": {
        "lean": "human",
        "lane": "construction",
        "say": "A short sentence sits next to a much longer one.",
    },
    "moralize": {
        "lean": "slop",
        "lane": "storyscope",
        "say": "States the theme outright. Let the scene carry it.",
    },
    "sensory": {
        "lean": "slop",
        "lane": "storyscope",
        "say": "Layers sensory detail. Keep the one that matters.",
    },
    "causal": {
        "lean": "slop",
        "lane": "storyscope",
        "say": "Over-explains the causal chain. Trust the reader to connect.",
    },
    "realize": {
        "lean": "slop",
        "lane": "storyscope",
        "say": "Ends on the character realizing the theme. End on the last concrete event.",
    },
    "intro": {
        "lean": "slop",
        "lane": "storyscope",
        "say": "Introduces the character as a dossier. Show them acting.",
    },
    "agency": {
        "lean": "slop",
        "lane": "storyscope",
        "say": "Resolves through the protagonist choosing the tidy option. Let the ending stay messy.",
    },
    "reader": {
        "lean": "human",
        "lane": "storyscope",
        "say": "Talks to the reader directly. A human move.",
    },
    "dialogue": {
        "lean": "human",
        "lane": "storyscope",
        "say": "Mostly dialogue. Human stories lean on speech.",
    },
}

LLM_IDS = (
    "glue",
    "frames",
    "emdash",
    "cliche",
    "puffery",
    "opener",
    "weasel",
    "colon",
    "moral",
    "gloss",
    "recap",
    "even",
    "jump",
    "anchor",
    "open",
    "burst",
)


def style_word(raw: str | None) -> str:
    key = (raw or "").strip().lower().replace("-", "_")
    if not key:
        return "style"
    if key in COPY:
        return key
    mapped = ALIAS.get(key) or STYLE_MAP.get(key)
    if mapped:
        return mapped
    for prefix, word in _PREFIX:
        if key.startswith(prefix):
            return word
    return "style"


def canonical(raw: str | None) -> str | None:
    word = style_word(raw)
    if word in COPY:
        return word
    return None


def say(raw: str | None) -> str:
    pid = canonical(raw)
    if pid is None:
        return ""
    return COPY[pid]["say"]


def pack_style(hit: dict[str, Any]) -> dict[str, Any]:
    """Ontology hit → one-word id + hardcoded say (the pattern's fix)."""
    raw = str(hit.get("id") or hit.get("pattern") or "")
    word = style_word(raw)
    quote = str(hit.get("quote") or "").strip()
    text_say = str(hit.get("fix") or hit.get("say") or say(word) or "")
    out: dict[str, Any] = {
        "id": word,
        "pattern": raw,
        "quote": quote,
        "say": text_say,
        "fix": text_say,
        "lane": hit.get("lane") or COPY.get(word, {}).get("lane", "style"),
        "unit": hit.get("unit", "span"),
        "lean": COPY.get(word, {}).get("lean", "slop"),
    }
    if "start" in hit:
        out["start"] = hit["start"]
    if "end" in hit:
        out["end"] = hit["end"]
    return out


def hydrate(hit: dict[str, Any]) -> dict[str, Any] | None:
    """Keep id + quote from the model. Attach hardcoded say. Drop unknown ids."""
    pid = canonical(hit.get("id"))
    if pid is None:
        return None
    quote = str(hit.get("quote") or "").strip()
    if not quote:
        return None
    spec = COPY[pid]
    return {
        "id": pid,
        "quote": quote,
        "lean": spec["lean"],
        "lane": spec["lane"],
        "say": spec["say"],
    }


def hydrate_lanes(parsed: dict[str, Any]) -> dict[str, Any]:
    style: list[dict[str, Any]] = []
    construction: list[dict[str, Any]] = []
    for raw in list(parsed.get("style") or []) + list(parsed.get("construction") or []):
        hit = hydrate(raw) if isinstance(raw, dict) else None
        if hit is None:
            continue
        if hit["lane"] == "style":
            style.append(hit)
        else:
            construction.append(hit)
    out = dict(parsed)
    out["style"] = style
    out["construction"] = construction
    out.pop("reason", None)
    return out
