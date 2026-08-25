# Third-party notices

## Wikipedia:Signs of AI writing
CC BY-SA 4.0. Descriptive text and fix blurbs derived from
https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing
live only in `ontology/patterns.wikipedia.yaml`.

## jalaalrd/anti-ai-slop-writing
Pattern ids and structural rules adapted as functional regexes in
`ontology/patterns.core.yaml`. Check the upstream repository for its license.

## no-ai-slop / noslop
Named patterns adapted as functional regexes. MIT-compatible.

## Hello-SimpleAI/HC3
Question-paired human/ChatGPT answers, pulled by `scripts/fetch_v2.py` into the v2
training mix as the `hc3_*` registers.

## coai/ai-text-detection-training
arXiv abstracts vs LLM paraphrases. Downloaded on demand by
`scripts/train_cpu_scorer.py` for the CPU logistic floor, and carried into v2 as
the `coai` register.

## RAID (liamdugan/raid)
Train split only. RAID-test is never downloaded, never trained on.
