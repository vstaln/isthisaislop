#!/usr/bin/env python3
"""Emit ontology YAML from structured entries. Run from repo root. Not imported by slopdet."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def e(
    id: str,
    pattern: str,
    fix: str,
    *,
    lane: str = "style",
    unit: str = "span",
    detector: str = "regex",
    source: str = "no-ai-slop",
    license: str = "MIT-compatible",
    min_len_words: int = 0,
    paper: str | None = None,
    enabled: bool = True,
) -> dict:
    return {
        "id": id,
        "lane": lane,
        "unit": unit,
        "detector": detector,
        "pattern": pattern,
        "fix": fix,
        "source": source,
        "license": license,
        "min_len_words": min_len_words,
        "paper": paper,
        "enabled": enabled,
    }


CORE = [
    e("binary_contrast", r"(?i)\b(?:it(?:['’]s| is) not(?: just| only)?.{0,80}?\b(?:it['’]s|it is|but)\b|not (?:just|only)\b.{0,60}?\bbut(?: also)?\b|the question isn['’]?t\b.{0,60}?\bit['’]s\b)", "State Y directly. Drop the not-X-but-Y frame."),
    e("throat_clearing", r"(?i)\b(?:here['’]?s the thing|here['’]?s what I mean|let me be clear|I['’]ll be honest|the uncomfortable truth is|here['’]?s the deal)\b", "Cut the opener. Start on the point."),
    e("faux_insight", r"(?i)\b(?:this is the part most people skip|what most people get wrong|here['’]s what nobody tells you|the part everyone misses|what nobody tells you)\b", "Drop the lone-expert setup. Make the claim stand alone."),
    e("colon_reveal", r"(?m)^[A-Z][^.!?\n]{2,60}: [a-z]", "Rewrite as a plain sentence."),
    e("superficial_analysis", r"(?i),\s+(?:highlighting|underscoring|reflecting|showcasing|emphasizing|ensuring|symbolizing|demonstrating)\b", "Replace the trailing -ing clause with a concrete consequence."),
    e("importance_puffery", r"(?i)\b(?:stands as a testament|marks a pivotal moment|plays a vital role|solidifies its position|underscores its significance|crucial role|key turning point)\b", "State the fact. Let the reader judge if it matters."),
    e("interpretive_metadiscourse", r"(?i)\b(?:that last part matters(?: more than it sounds)?|the key point is|as you can see|this distinction matters|in other words)\b", "Delete the aside, or replace it with a fact."),
    e("weasel_attribution", r"(?i)\b(?:experts agree|industry reports suggest|many argue|widely regarded as|studies show|observers have cited|some critics argue)\b", "Name the source or cut the claim."),
    e("fake_strong_verb", r"(?i)\b(?:serves as a|functions as a|acts as a|operates as a|stands as a)\b", "Use is/has, or name the actual action."),
    e("synonym_cycling", r"(?i)\b(?:the agent|the assistant|the tool|the system|the platform)\b.{0,180}\b(?:the agent|the assistant|the tool|the system|the platform)\b", "Repeat the clear word. Do not rotate synonyms for style.", detector="heuristic", unit="sentence"),
    e("negative_listing", r"(?i)\bnot a\b.{0,50}\bnot a\b.{0,50}\b(?:a |an |the )", "Just say Z. Drop the not-X not-Y list."),
    e("dramatic_fragmentation", r"(?i)(?:\bthat['’]s it\. that['’]s the whole thing\b|(?m)^And [A-Z][^.!?\n]{0,50}\.\s*\nAnd )", "Use complete sentences. Stop stacking And-fragments."),
    e("robotic_rhythm", r"(?m)(?:^[A-Z][^.!?\n]{10,42}\.\s+){2}[A-Z][^.!?\n]{10,42}\.", "Vary sentence length and shape. Merge or split one of the three.", detector="heuristic", unit="paragraph", min_len_words=40),
    e("rhetorical_setup", r"(?i)\b(?:what if I told you|think about it:|plot twist:)\b", "Drop the setup. Make the point."),
    e("fake_profound_kicker", r"(?i)\b(?:and that(?:['’]s| is) the (?:whole |real )?point|the rest is (?:just )?noise|that(?:['’]s| is) the whole game)\b", "Delete the mic-drop. End on the last concrete sentence.", unit="sentence"),
    e("recap_ending", r"(?im)(?:^|(?<=[.!?]\s))(?:in conclusion|ultimately|overall|to sum up|in summary|to conclude)\s*[,:]", "End on the last concrete point. Do not restate the piece.", unit="paragraph"),
    e("formatting_slop", r"(?m)^#{1,3}\s+.+\n(?:.*\n){0,2}^#{1,3}\s+", "Drop headers over two-sentence sections. Write prose.", unit="paragraph"),
    e("em_dash_cluster", r"—[^.\n]{0,90}—", "Use a comma, period, or parentheses. Do not cluster dashes."),
    e("ban_delve_class", r"(?i)\b(?:delve|delves|delving|foster|fostering|leverage|leveraging|utilize|utilizing|facilitate|facilitating|empower|empowering|streamline|streamlining)\b", "Name the action. Use a concrete verb.", paper="kobak-2406.07016"),
    e("ban_puffery_noun", r"(?i)\b(?:tapestry|realm|beacon|paradigm(?: shift)?)\b", "Replace the metaphor with the actual thing.", paper="kobak-2406.07016"),
    e("ban_puffery_adj", r"(?i)\b(?:robust|cutting-edge|multifaceted|meticulous(?:ly)?|intricate|intricacies|paramount|transformative|vibrant|pivotal|groundbreaking|seamless(?:ly)?)\b", "Cut the adjective, or name the property it stands in for.", paper="kobak-2406.07016"),
    e("ban_journey_verb", r"(?i)\b(?:elevate|elevating|embark|embarking|supercharge|supercharging|harness|harnessing|ever-evolving)\b", "Use a plain verb. Say what actually happens.", paper="kobak-2406.07016"),
    e("ban_showcase_verb", r"(?i)\b(?:underscore|underscores|underscoring|showcase|showcases|showcasing|highlight|highlights|highlighting|emphasize|emphasizes|emphasizing)\b", "State the fact. Do not announce its importance.", paper="kobak-2406.07016"),
    e("ban_corporate", r"(?i)\b(?:synergy|synergies|pain points?|value proposition|thought leaders?(?:hip)?|circle back|touch base|move the needle)\b", "Say the work in ordinary words.", source="anti-ai-slop-writing"),
    e("ban_game_changer", r"(?i)\b(?:game[- ]chang(?:er|ing)|this is huge|this changes everything|unlock(?:s|ing)? the power)\b", "Name the change. Drop the slogan."),
    e("empty_adverb", r"(?i)\b(?:literally|honestly|simply|actually|truly|fundamentally|importantly|crucially|inherently|inevitably)\b", "Cut the adverb if it adds nothing."),
    e("phrase_worth_noting", r"(?i)\bit['’]?s worth noting\b", "Delete. Start with the fact."),
    e("phrase_important_to_note", r"(?i)\bit['’]?s important to note(?: that)?\b", "Delete. State the fact."),
    e("phrase_end_of_the_day", r"(?i)\bat the end of the day\b", "Cut the proverb. Make the claim."),
    e("phrase_when_it_comes_to", r"(?i)\bwhen it comes to\b", "Name the subject and start."),
    e("phrase_at_its_core", r"(?i)\bat its core\b", "Drop the frame. State the mechanism."),
    e("phrase_in_todays", r"(?i)\bin today['’]?s\b", "Cut the era opener. Name the situation."),
    e("phrase_in_the_age_of", r"(?i)\bin the age of\b", "Cut the era opener."),
    e("phrase_in_the_world_of", r"(?i)\bin the world of\b", "Name the field. Skip the tour."),
    e("phrase_the_reality_is", r"(?i)\bthe reality is\b", "Drop the drumroll. State the fact."),
    e("phrase_the_truth_is", r"(?i)\bthe truth is\b", "Drop the drumroll. State the fact."),
    e("phrase_in_terms_of", r"(?i)\bin terms of\b", "Rewrite with a direct object."),
    e("phrase_with_regard_to", r"(?i)\bwith regard to\b", "Name the topic and continue."),
    e("phrase_in_order_to", r"(?i)\bin order to\b", "Rewrite as 'to' plus the verb."),
    e("phrase_going_forward", r"(?i)\bgoing forward\b", "Cut it, or name the date."),
    e("phrase_in_this_article", r"(?i)\bin this article\b", "Do not announce the article."),
    e("phrase_lets_dive_in", r"(?i)\blet['’]?s dive (?:in|deeper|into)\b", "Start the first fact."),
    e("phrase_unlock_the_power", r"(?i)\bunlock(?:s|ing)? the power of\b", "Name the action the reader can take."),
    e("phrase_bridge_the_gap", r"(?i)\bbridge(?:s|ing)? the gap\b", "Name the two sides and the actual link."),
    e("phrase_i_hope_this_helps", r"(?i)\bI hope this helps\b", "End on the last useful sentence."),
    e("phrase_i_hope_this_finds_you", r"(?i)\bI hope this(?: email)? finds you well\b", "Open with the reason you wrote."),
    e("phrase_whether_youre", r"(?i)\bwhether you(?:['’]re| are) a\b.{0,40}\bor a\b", "Pick one reader. Write to them."),
    e("phrase_from_x_to_y_opener", r"(?i)^From .{2,40} to .{2,40}[,.]", "Open with the specific case, not a range."),
    e("phrase_this_is_where_x_comes_in", r"(?i)\bthis is where\b.{0,40}\bcomes in\b", "Introduce the thing without the drumroll."),
    e("phrase_firstly_secondly_thirdly", r"(?i)\b(?:firstly|secondly|thirdly)\b", "Use 1. 2. 3. or just write the points."),
    e("phrase_without_further_ado", r"(?i)\bwithout further ado\b", "Cut the drumroll and start."),
    e("phrase_in_a_nutshell", r"(?i)\bin a nutshell\b", "State the summary as a sentence."),
    e("phrase_please_dont_hesitate", r"(?i)\bplease don['’]?t hesitate to (?:reach out|contact)\b", "Give the actual next step."),
    e("phrase_not_just_x_but_y", r"(?i)\bit['’]?s not just about\b.{0,50}\bit['’]?s about\b", "State Y. Drop the not-just frame.", source="anti-ai-slop-writing"),
    e("rule_of_three", r"(?i)\b\w+,\s+\w+,\s+and\s+\w+\b", "Break the default trio. Use two, four, or one.", detector="heuristic", unit="sentence", source="anti-ai-slop-writing", min_len_words=20),
    e("uniform_sentence_length", r"(?s)(?=.{80,})", "Mix a short sentence with a long one.", detector="heuristic", unit="paragraph", source="anti-ai-slop-writing", min_len_words=80, lane="construction"),
    e("parataxis", r"(?m)(?:^[A-Z][^.!?\n]{0,28}\.\s+){2}[A-Z][^.!?\n]{0,28}\.", "Connect the thoughts. Add a because, but, or which.", detector="heuristic", unit="paragraph", source="anti-ai-slop-writing"),
    e("hedging_seesaw", r"(?i)\bon the one hand\b.{0,200}\bon the other hand\b", "Pick a side. Give the counterpoint one sentence.", detector="heuristic", unit="paragraph", source="anti-ai-slop-writing"),
    e("corporate_pep_talk", r"(?i)\b(?:together we can|exciting opportunity|passionate about|unlock(?:ing)? potential|drive(?:s|ing)? (?:impact|outcomes)|deliver(?:ing)? value)\b", "Write like someone who did the work, including the mess.", unit="sentence", source="anti-ai-slop-writing"),
    e("identical_paragraph_structure", r"(?s)(?=.{200,})", "Break the topic-explain-example-transition mold.", detector="heuristic", unit="piece", source="anti-ai-slop-writing", min_len_words=200, lane="construction"),
    e("bullet_overuse", r"(?m)(?:^[\t ]*(?:[-*]|\d+\.)\s+.+\n){6,}", "Turn the list into sentences, or cap it at five.", unit="paragraph", source="anti-ai-slop-writing"),
    e("as_role_opener", r"(?i)^As an? [A-Z][^,.]{2,40},\s+I\b", "Say the thing. Do not announce credentials.", source="anti-ai-slop-writing"),
    e("cross_section_parallelism", r"(?s)(?=.{300,})", "Give each section a different shape and length.", detector="heuristic", unit="piece", source="anti-ai-slop-writing", min_len_words=300, lane="construction"),
    e("passive_construction", r"(?i)\b(?:is being \w+ed|was found to be|are considered to be|has been shown to)\b", "Write the actor and the verb.", detector="heuristic", unit="sentence", source="anti-ai-slop-writing"),
    e("mandatory_paragraph_transition", r"(?i)(?:^|\n)\s*(?:moreover|furthermore|additionally|in addition|that said|with that in mind)\s*,", "Let some paragraphs just stop.", unit="sentence", source="anti-ai-slop-writing"),
    e("punct_em_dash_budget", r"(?:—.*){2,}", "At most one em dash per 500 words.", detector="heuristic", unit="piece", source="anti-ai-slop-writing", min_len_words=1),
    e("punct_exclamation_budget", r"(?:!.*){2,}", "At most one exclamation per 1,000 words.", detector="heuristic", unit="piece", source="anti-ai-slop-writing", min_len_words=1),
    e("punct_ellipsis_budget", r"(?:\.{3}|…).*(?:\.{3}|…)", "One ellipsis per piece, only for a real trailing-off.", detector="heuristic", unit="piece", source="anti-ai-slop-writing"),
    e("opener_certainly", r"(?im)^(?:certainly|absolutely|sure|great question|that['’]s a great point)[,!]", "Answer. Skip the cheer."),
    e("opener_moreover", r"(?im)^Moreover,", "Start with the next fact."),
    e("opener_furthermore", r"(?im)^Furthermore,", "Start with the next fact."),
    e("opener_additionally", r"(?im)^Additionally,", "Start with the next fact.", paper="kobak-2406.07016"),
    e("opener_interestingly", r"(?im)^Interestingly,", "State the interesting fact. Drop the label."),
    e("opener_notably", r"(?im)^Notably,", "State the fact. Drop the label."),
    e("opener_importantly", r"(?im)^Importantly,", "State the fact. Drop the label."),
    e("opener_indeed", r"(?im)^Indeed,", "Continue without the nod."),
    e("opener_as_an_ai", r"(?i)\b(?:as an AI|as a language model)\b", "Never announce the model."),
    e("emoji_bullet", r"(?m)^[\t ]*[✅🔥✨💡👉⭐️⭐]\s", "Write a sentence. Do not emoji-bullet."),
    e("hashtag_stack", r"(?:#[A-Za-z0-9_]+)(?:\s+#[A-Za-z0-9_]+){2,}", "Zero to two hashtags, in the sentence."),
    e("markdown_in_plain", r"(?m)^\*\*[^*]{3,40}\*\*\s*$", "Do not bold a whole line for emphasis."),
    e("copula_avoidance_surface", r"(?i)\b(?:serves as|stands as|functions as|operates as|boasts a|features a)\b", "Use is or has when that is what you mean.", paper="reinhart-2410.16107"),
    e("in_the_realm_of", r"(?i)\bin the realm of\b", "Name the field.", source="anti-ai-slop-writing"),
    e("a_testament_to", r"(?i)\ba testament to\b", "State what happened.", paper="kobak-2406.07016"),
    e("rest_assured", r"(?i)\brest assured\b", "Give the actual guarantee or drop it.", source="anti-ai-slop-writing"),
    e("it_goes_without_saying", r"(?i)\bit goes without saying\b", "If it goes without saying, delete it.", source="anti-ai-slop-writing"),
    e("in_essence", r"(?i)\bin essence\b", "State the claim.", source="anti-ai-slop-writing"),
    e("please_note_that", r"(?i)\bplease note that\b", "State the note as a fact."),
    e("as_mentioned_earlier", r"(?i)\bas (?:mentioned|noted|discussed) earlier\b", "Repeat the fact if needed. Do not point at the earlier sentence."),
    e("in_todays_digital_age", r"(?i)\bin today['’]?s (?:fast-paced |ever-changing |digital )?world\b", "Name the actual constraint."),
]


WIKI_HEADER = """\
# SPDX-License-Identifier: CC-BY-SA-4.0
#
# Derived from Wikipedia:Signs of AI writing
# https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing
# License: CC BY-SA 4.0
# https://creativecommons.org/licenses/by-sa/4.0/
#
# Share-alike applies to the descriptive text (fix blurbs) in this file.
# Regex strings are functional. Do not paste these descriptions into MIT source.
#
"""

WIKI = [
    e("wiki_significance_puffery", r"(?i)\b(?:stands as a testament|marking a pivotal moment|reflects broader|symbolizing its (?:ongoing|enduring|lasting)|setting the stage for|indelible mark|evolving landscape|focal point|deeply rooted)\b", "Cut the legacy sermon. Keep the dated fact.", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0"),
    e("wiki_notability_boilerplate", r"(?i)\b(?:independent coverage|active social media presence|profiled in|widely-read outlets|significant, substantial, secondary coverage)\b", "Cite the source. Do not recite notability policy.", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0"),
    e("wiki_promotional_tone", r"(?i)\b(?:nestled (?:in|within)|in the heart of|rich (?:cultural )?heritage|natural beauty|diverse array|boasts a|renowned for)\b", "Drop the brochure. Name one specific place or fact.", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0"),
    e("wiki_ai_vocabulary_cluster", r"(?i)(?:\b(?:delve|tapestry|underscore|pivotal|vibrant|intricate|meticulous|landscape|testament|showcase|foster|align with)\b.*){3,}", "One inflated word can be accident. Three in one passage is the tell. Rewrite with plain nouns.", detector="heuristic", unit="paragraph", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0", paper="kobak-2406.07016"),
    e("wiki_copula_avoidance", r"(?i)\b(?:serves as|stands as|marks a|functions as|operates as|holds the distinction of being|refers to)\b", "Use is or are. Stop dressing the copula.", unit="span", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0", paper="reinhart-2410.16107"),
    e("wiki_negative_parallelism", r"(?i)\b(?:not only .{0,40} but(?: also)?|it is not .{0,40}, it(?:['’]s| is)|rather than .{0,30}$)", "Drop the misconception-clearing frame. State the property.", unit="sentence", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0"),
    e("wiki_challenges_future", r"(?i)\b(?:challenges remain|future prospects|looking ahead|as .+ continues to evolve|more research is needed)\b", "Stop the outline close. End on what is known now.", unit="paragraph", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0"),
    e("wiki_awards_heading", r"(?i)^#+\s+Awards and recognition\s*$", "Merge awards into the career section if they are few.", unit="span", detector="regex", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0"),
    e("wiki_title_heading", r"(?m)^#\s+[A-Z].{0,80}\n\n", "Do not repeat the article title as the first heading.", unit="span", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0"),
    e("wiki_title_case_heading", r"(?m)^#{2,3}\s+(?:[A-Z][a-z]+\s+){2,}[A-Z][a-z]+$", "Use sentence case in headings.", unit="span", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0"),
    e("wiki_boldface_overuse", r"(?:\*\*[^*]{2,40}\*\*.*){3,}", "Bold once, if at all. Stop sprinkling emphasis.", unit="paragraph", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0"),
    e("wiki_inline_header_list", r"(?m)^[-*]\s+\*\*[^*]{2,40}\*\*:\s", "Turn canned bold-label bullets into sentences.", unit="paragraph", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0"),
    e("wiki_emoji_formatting", r"(?m)^[\t ]*(?:[✅❌⚠️📌🔍💡]|:[a-z_]+:)\s", "No emoji as structure.", unit="span", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0"),
    e("wiki_curly_quotes", r"[“”‘’].{0,80}[“”‘’]", "Straight quotes unless the house style needs curls.", unit="span", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0"),
    e("wiki_collaborative_you", r"(?i)\b(?:I hope this helps|let me know if|I['’]d be happy to|as you requested)\b", "This is article text, not a chat reply. Cut the assistant voice.", unit="sentence", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0"),
    e("wiki_knowledge_cutoff", r"(?i)\b(?:as of my last (?:training|update)|I don['’]t have (?:access|information)|my knowledge cutoff)\b", "Delete the model disclaimer. Write from sources.", unit="sentence", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0"),
    e("wiki_placeholder_text", r"(?i)\b(?:TODO|TBD|lorem ipsum|insert (?:text|citation|source) here|\[placeholder\])\b", "Replace placeholders before the text ships.", unit="span", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0"),
    e("wiki_skipping_heading_levels", r"(?m)^#\s+.+\n+(?:#{3,}\s+)", "Do not skip from H1 to H3.", unit="span", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0"),
    e("wiki_thematic_break_spam", r"(?m)(?:^---\s*\n){2,}", "Horizontal rules are not sectioning.", unit="piece", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0"),
    e("wiki_valuable_insights", r"(?i)\bvaluable insights\b", "Name the finding. Insights is empty.", source="wikipedia-signs-of-ai-writing", license="CC-BY-SA-4.0"),
]


RHET = [
    e("rhet_present_participial", r"(?i),\s+\w+ing\b.{0,80}(?:\.|$)", "The comma-VBG tag is a proxy for present-participial clauses. Replace with a finite clause.", lane="rhetorical", unit="sentence", detector="heuristic", source="reinhart-biber", paper="reinhart-2410.16107"),
    e("rhet_nominalization_density", r"(?i)\b\w{4,}(?:tion|sion|ment|ness|ity)s?\b", "High -tion/-ment/-ness/-ity rate. Prefer verbs over abstract nouns.", lane="rhetorical", unit="paragraph", detector="heuristic", source="reinhart-biber", paper="reinhart-2410.16107", min_len_words=80),
    e("rhet_copula_avoidance", r"(?i)\b(?:is|are|was|were|be|been|being)\b", "Low be-verb ratio vs lexical verbs is the tell. Restore is/are where they are clearer.", lane="rhetorical", unit="paragraph", detector="heuristic", source="reinhart-biber", paper="reinhart-2410.16107", min_len_words=80),
    e("rhet_that_complement", r"(?i)\b(?:said|argued|claimed|noted|reported|suggested|found|showed|believed) that\b", "That-complement rate vs human baseline. Keep that when it prevents a garden path.", lane="rhetorical", unit="sentence", detector="heuristic", source="reinhart-biber", paper="reinhart-2410.16107"),
    e("rhet_adj_stacking", r"(?i)\b(?:a|an|the)\s+[A-Za-z]+,\s+[A-Za-z]+(?:,|\s+and)\s+[A-Za-z]+\b", "Stacked attributive adjectives. Keep one modifier that earns its place.", lane="rhetorical", unit="span", detector="heuristic", source="reinhart-biber", paper="reinhart-2410.16107"),
]


def dump_list(path: Path, entries: list[dict], header: str = "") -> None:
    # Keep key order from `e()`.
    class InOrderDumper(yaml.SafeDumper):
        pass

    def represent_none(self, _):
        return self.represent_scalar("tag:yaml.org,2002:null", "null")

    InOrderDumper.add_representer(type(None), represent_none)
    body = yaml.dump(
        entries,
        Dumper=InOrderDumper,
        sort_keys=False,
        allow_unicode=True,
        width=1000,
        default_flow_style=False,
    )
    path.write_text(header + body, encoding="utf-8")


def main() -> None:
    dump_list(ROOT / "ontology" / "patterns.core.yaml", CORE)
    dump_list(ROOT / "ontology" / "patterns.wikipedia.yaml", WIKI, WIKI_HEADER)
    dump_list(ROOT / "ontology" / "patterns.rhetorical.yaml", RHET)
    ids = [x["id"] for x in CORE + WIKI + RHET]
    assert len(ids) == len(set(ids)), "duplicate ids in emitter"
    print(f"wrote {len(CORE)} core, {len(WIKI)} wikipedia, {len(RHET)} rhetorical ({len(ids)} total)")


if __name__ == "__main__":
    main()
