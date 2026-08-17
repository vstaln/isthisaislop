"""Colab smoke/full pipeline. Imported by the notebook after the repo is written to disk."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from slopdet.calibrate import calibration_record, human_percentile
from slopdet.construction import construction_stats
from slopdet.ontology import load_ontology
from slopdet.report import render_hits
from slopdet.weaklabel import label_text

SEED_HUMAN = [
    "Thursday mornings at the clinic were empty. Half the early slots sat unused, and the monthly average hid it.",
    "I cut the section. The reader already had the date, the name, and the number.",
    "Maya sent the booking export with names stripped. I marked the gaps in a shared sheet.",
    "The RPC timed out at 3am. I almost scrapped the feature, then cached the last good payload.",
    "We shipped on Tuesday. Review time went from 30 minutes to 8.",
    "He said the eval was the product. I believed him after the third silent failure.",
    "OakNorth's 2019 report listed the default rate by vintage, not a slogan.",
    "Put the conclusion in the first sentence if the rest is method.",
    "I don't know if FineWeb is clean after 2022. Filter the dumps anyway.",
    "The student is 40 million parameters because the embedding table is huge.",
]

SEED_MACHINE = [
    "Here's the thing, in today's competitive landscape we leverage robust pipelines to unlock the power of detection.",
    "It's worth noting that, at its core, this is a testament to seamless, cutting-edge architecture.",
    "Additionally, experts agree the launch marks a pivotal moment, highlighting the team's commitment to better workflows.",
    "In conclusion, we hope this helps. Please don't hesitate to reach out as we move the needle going forward.",
    "As an AI, I can say the platform serves as a centralized hub, fostering synergy across pain points.",
    "What if I told you it's not just about accuracy — it's about delivering value and supercharging outcomes?",
    "Certainly, the solution is multifaceted and meticulously designed to streamline your operations.",
    "In this article, we'll delve into how our transformative paradigm shift empowers teams.",
    "The company, nestled in the heart of the city, boasts a rich cultural heritage and a vibrant tapestry of innovation.",
    "Overall, studies show that utilizing these tools will elevate your process, underscoring its significance.",
]


def seed_docs() -> list[dict[str, Any]]:
    docs = []
    for i, text in enumerate(SEED_HUMAN):
        docs.append({"id": f"seed-human-{i}", "text": text, "source": "seed", "model": "human", "pile": 0})
    for i, text in enumerate(SEED_MACHINE):
        docs.append({"id": f"seed-ai-{i}", "text": text, "source": "seed", "model": "template", "pile": 1})
    return docs


def try_hc3(n: int, rng: random.Random) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed; using seed corpus only")
        return []
    configs = ("reddit_eli5", "finance", "open_qa", "medicine", "wiki_csai")
    docs: list[dict[str, Any]] = []
    for cfg in configs:
        try:
            ds = load_dataset("Hello-SimpleAI/HC3", cfg, split="train")
        except Exception as exc:
            print(f"HC3 {cfg} skipped: {exc}")
            continue
        rows = list(ds)
        rng.shuffle(rows)
        for row in rows:
            if len(docs) >= n:
                return docs
            human = row.get("human_answers") or []
            machine = row.get("chatgpt_answers") or []
            if human:
                docs.append(
                    {
                        "id": f"hc3-{cfg}-h-{len(docs)}",
                        "text": human[0],
                        "source": "hc3",
                        "model": "human",
                        "pile": 0,
                    }
                )
            if machine and len(docs) < n:
                docs.append(
                    {
                        "id": f"hc3-{cfg}-m-{len(docs)}",
                        "text": machine[0],
                        "source": "hc3",
                        "model": "chatgpt",
                        "pile": 1,
                    }
                )
        if docs:
            break
    return docs


def featurize(doc: dict[str, Any], id_index: dict[str, int], dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for hit in doc.get("style_hits") or []:
        idx = id_index.get(hit["id"])
        if idx is not None:
            vec[idx] += 1.0
    stats = doc.get("construction") or {}
    extra = [
        float(stats.get("burstiness") or 0.0),
        float(stats.get("evenness") or 0.0),
        float(stats.get("recap_closure") or 0.0),
        float(stats.get("over_explain") or 0.0),
        float(stats.get("portability") or 0.0),
        float(stats.get("n_words") or 0.0) / 1000.0,
    ]
    vec = np.concatenate([vec, np.array(extra, dtype=np.float32)])
    return vec


def build_and_train(root: Path, n_docs: int = 400) -> dict[str, Any]:
    rng = random.Random(0)
    onto = load_ontology(root / "ontology")
    ids = [p.id for p in onto.enabled_patterns()]
    id_index = {pid: i for i, pid in enumerate(ids)}

    docs = seed_docs()
    hc3 = try_hc3(max(0, n_docs - len(docs)), rng)
    docs.extend(hc3)
    print(f"corpus: {len(docs)} docs (seed={len(SEED_HUMAN)+len(SEED_MACHINE)} hc3={len(hc3)})")

    for doc in docs:
        text = doc["text"]
        doc["style_hits"] = [
            {k: h[k] for k in ("id", "start", "end", "unit")} for h in label_text(text, onto)
        ]
        doc["construction"] = construction_stats(text)
        doc["jspace"] = []

    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = data_dir / "corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    X = np.stack([featurize(d, id_index, len(ids)) for d in docs])
    y = np.array([d["pile"] for d in docs], dtype=np.int64)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xs, y)

    human_scores = clf.predict_proba(Xs[y == 0])[:, 1].tolist()
    calib = calibration_record(human_scores, 0.01)
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pattern_ids": ids,
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "calibration": calib,
        "ontology_sha256": onto.sha256,
        "n_docs": len(docs),
        "never_trained_on": ["raid-test"],
    }
    bundle_path = artifacts / "sklearn_bundle.json"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    print("wrote", bundle_path, "threshold", calib["threshold"])
    return {"docs": docs, "clf": clf, "scaler": scaler, "onto": onto, "bundle": bundle, "ids": ids}


def score_text(text: str, *, onto, clf, scaler, ids, human_scores: list[float]) -> dict[str, Any]:
    id_index = {pid: i for i, pid in enumerate(ids)}
    hits = label_text(text, onto)
    doc = {"style_hits": hits, "construction": construction_stats(text)}
    x = scaler.transform([featurize(doc, id_index, len(ids))])
    score = float(clf.predict_proba(x)[0, 1])
    pct = human_percentile(score, human_scores)
    result = render_hits(hits, resemblance={"human_percentile": pct})
    result["construction"] = doc["construction"]
    result["matches_ai_pile_score"] = score
    return result


def demo(state: dict[str, Any]) -> None:
    docs = state["docs"]
    human_scores = [
        float(state["clf"].predict_proba(state["scaler"].transform([
            featurize(d, {pid: i for i, pid in enumerate(state["ids"])}, len(state["ids"]))
        ]))[0, 1])
        for d in docs
        if d["pile"] == 0
    ]
    samples = [
        "Here's the thing, in today's world we leverage robust tools. In conclusion, experts agree.",
        "Thursday mornings at the clinic were empty. I marked the gaps on a sheet.",
    ]
    for text in samples:
        print("=" * 60)
        print(text)
        out = score_text(
            text,
            onto=state["onto"],
            clf=state["clf"],
            scaler=state["scaler"],
            ids=state["ids"],
            human_scores=human_scores,
        )
        print(out.get("style_summary") or "")
        for hit in out["hits"][:8]:
            print(f"  [{hit['id']}] {hit['quote']!r}")
            print(f"      {hit['fix']}")
        if out.get("resemblance") and len(human_scores) >= 30:
            print(out["resemblance"]["text"])
        elif out.get("resemblance"):
            print("(resemblance hidden until ≥30 human reference docs; HC3 will fill this on Colab)")


def try_gpu_distill(root: Path, docs: list[dict[str, Any]], max_docs: int = 64) -> None:
    """Best-effort teacher cache + 1-epoch student. Never fails the notebook."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from slopdet.student import Student, StudentConfig
    except Exception as exc:
        print("GPU distill skipped (imports):", exc)
        return
    if not torch.cuda.is_available():
        print("GPU distill skipped: no CUDA")
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    candidates = []
    if os.environ.get("FULL") == "1":
        candidates.append("google/gemma-3-4b-it")
    candidates.append("Qwen/Qwen2.5-0.5B-Instruct")

    model = None
    tokenizer = None
    name = None
    for cand in candidates:
        try:
            print("loading teacher", cand)
            tokenizer = AutoTokenizer.from_pretrained(cand, token=token)
            model = AutoModelForCausalLM.from_pretrained(
                cand,
                token=token,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            name = cand
            break
        except Exception as exc:
            print("teacher failed", cand, exc)
            model = None
    if model is None or tokenizer is None:
        print("GPU distill skipped: no teacher")
        return

    hidden = int(getattr(model.config, "hidden_size", 896))
    n_layers = int(getattr(model.config, "num_hidden_layers", 24))
    layer_idx = min(17, n_layers - 1)
    print("teacher", name, "hidden", hidden, "L", layer_idx)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    subset = docs[:max_docs]
    device = torch.device("cuda")
    cfg = StudentConfig(
        vocab_size=len(tokenizer),
        pad_token_id=int(tokenizer.pad_token_id or 0),
        max_length=128,
        output_dim=hidden,
        n_layers=2,
    )
    student = Student(cfg).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=3e-4)
    model.eval()
    student.train()
    losses = []
    for step, doc in enumerate(subset):
        enc = tokenizer(
            doc["text"],
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            target = out.hidden_states[layer_idx].float()
        pred = student(enc["input_ids"], enc["attention_mask"]).float()
        mask = enc["attention_mask"].unsqueeze(-1).float()
        loss = ((pred - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step % 16 == 0:
            print(f"distill {step}/{len(subset)} loss={losses[-1]:.4f}")
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    ckpt = artifacts / "student_smoke.pt"
    torch.save({"cfg": cfg.__dict__, "state": student.state_dict(), "teacher": name, "hidden": hidden}, ckpt)
    print("saved", ckpt, "mean loss", sum(losses) / max(len(losses), 1))
    del model
    torch.cuda.empty_cache()
