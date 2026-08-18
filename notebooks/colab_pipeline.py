"""ITAIS v1 — classify slop vs not, and name why (which sentences + which patterns)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from slopdet.calibrate import calibration_record, human_percentile
from slopdet.span import pure_docs, split_sentences, stitch_docs, token_labels


def _training_args(**kwargs: Any) -> Any:
    from transformers import TrainingArguments

    kwargs.setdefault("report_to", [])
    try:
        return TrainingArguments(**kwargs)
    except TypeError:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy", "epoch")
        return TrainingArguments(**kwargs)

COAI_BASE = "https://huggingface.co/datasets/coai/ai-text-detection-training/resolve/main/data"
COAI_FILES = {
    "train": "train-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}


def download_coai(data_dir: Path) -> dict[str, Path]:
    from urllib.request import urlopen

    data_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, fname in COAI_FILES.items():
        dest = data_dir / f"coai_{split}.parquet"
        if not dest.exists():
            print("downloading coai", split, fname)
            with urlopen(f"{COAI_BASE}/{fname}") as r, dest.open("wb") as out:
                out.write(r.read())
        paths[split] = dest
    return paths


def load_coai(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    import pandas as pd

    out: dict[str, list[dict[str, Any]]] = {}
    for split, path in download_coai(data_dir).items():
        df = pd.read_parquet(path)
        docs = [
            {"text": str(text), "label": int(label)}
            for text, label in zip(df["text"], df["label"])
        ]
        out[split] = docs
        print(split, len(docs), "docs")
    return out


def fine_tune_roberta(root: Path, epochs: int = 1, max_len: int = 192, batch_size: int = 32) -> dict[str, Any]:
    """Fine-tune roberta-base as a human/AI classifier on coai. Exports to artifacts/roberta/."""
    import torch
    from sklearn.metrics import roc_auc_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
    )

    data = load_coai(root / "data")
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=2)

    class PileDataset(torch.utils.data.Dataset):
        def __init__(self, docs: list[dict[str, Any]]) -> None:
            enc = tokenizer(
                [d["text"] for d in docs],
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            self.input_ids = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels = torch.tensor([d["label"] for d in docs])

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
            return {
                "input_ids": self.input_ids[i],
                "attention_mask": self.attention_mask[i],
                "labels": self.labels[i],
            }

    def compute_metrics(pred) -> dict[str, float]:
        logits, labels = pred.predictions, pred.label_ids
        probs = torch.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()
        return {
            "auc": float(roc_auc_score(labels, probs)),
            "acc": float((probs.round().astype(int) == labels).mean()),
        }

    out_dir = root / "artifacts" / "roberta"
    args = _training_args(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=PileDataset(data["train"]),
        eval_dataset=PileDataset(data["test"]),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    test = PileDataset(data["test"])
    logits = torch.softmax(torch.from_numpy(trainer.predict(test).predictions), dim=-1)[:, 1].numpy()
    human_scores = [float(s) for s, d in zip(logits, data["test"]) if d["label"] == 0]
    calib = calibration_record(human_scores, 0.01)
    calib["human_scores"] = human_scores
    calib["auc_test"] = float(roc_auc_score([d["label"] for d in data["test"]], logits))

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=1), encoding="utf-8")
    manifest = {
        "architecture": "FacebookAI/roberta-base",
        "trained_on": ["coai/ai-text-detection-training train split (62,460 docs)"],
        "never_trained_on": ["coai test split (11,022 docs)"],
        "n_train": len(data["train"]),
        "n_test": len(data["test"]),
        "auc_test": calib["auc_test"],
        "threshold": calib["threshold"],
        "fpr": 0.01,
        "epochs": epochs,
        "max_len": max_len,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("exported", out_dir, "auc", round(calib["auc_test"], 4), "threshold", round(calib["threshold"], 4))
    return {"dir": out_dir, "calibration": calib}


def demo_model(root: Path, info: dict[str, Any]) -> None:
    """Load the exported bundle with plain from_pretrained and score two samples."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out_dir = Path(info["dir"])
    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    model = AutoModelForSequenceClassification.from_pretrained(out_dir)
    calib = info["calibration"]

    samples = [
        "Here's the thing, in today's world we leverage robust tools. In conclusion, experts agree it's a pivotal moment.",
        "Thursday mornings at the clinic were empty. Half the early slots sat unused, and the monthly average hid it.",
    ]
    for text in samples:
        enc = tokenizer(text, return_tensors="pt", truncation=True)
        p = float(model(**enc).logits.softmax(dim=-1)[0, 1])
        verdict = "AI slop" if p >= calib["threshold"] else "Not slop"
        print("=" * 60)
        print(text)
        print(f"verdict: {verdict} | resembles the AI pile more than {human_percentile(p, calib['human_scores']):.0f}% of human references")


def train_span_roberta(
    root: Path,
    epochs: int = 1,
    max_len: int = 256,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Token-classifier on stitched piles -> per-sentence why. Exports artifacts/roberta-span/."""
    import torch
    from sklearn.metrics import roc_auc_score
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        Trainer,
    )

    data = load_coai(root / "data")
    rng = random.Random(0)
    train_mixed = stitch_docs(data["train"], rng, limit=8000)
    test_mixed = stitch_docs(data["test"], rng, limit=1500)
    print("span train", len(train_mixed), "test", len(test_mixed), "docs")

    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained("FacebookAI/roberta-base", num_labels=2)

    class SpanDataset(torch.utils.data.Dataset):
        def __init__(self, docs: list[dict[str, Any]]) -> None:
            self.items = [
                {k: torch.tensor(v) for k, v in token_labels(d, tokenizer, max_len).items()} for d in docs
            ]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
            return self.items[i]

    def doc_score(doc: dict[str, Any]) -> float:
        tl = token_labels(doc, tokenizer, max_len)
        with torch.no_grad():
            logits = model(
                torch.tensor([tl["input_ids"]]), torch.tensor([tl["attention_mask"]])
            ).logits[0]
        probs = logits.softmax(dim=-1)[:, 1]
        valid = tl["labels"]
        return float(probs[torch.tensor(valid) != -100].mean())

    def compute_metrics(pred) -> dict[str, float]:
        logits, labels = pred.predictions, pred.label_ids
        mask = labels != -100
        preds = logits.argmax(-1)
        acc = float((preds[mask] == labels[mask]).mean())
        tp = float(((preds == 1) & (labels == 1))[mask].sum())
        fp = float(((preds == 1) & (labels == 0))[mask].sum())
        fn = float(((preds == 0) & (labels == 1))[mask].sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return {"tok_acc": acc, "tok_f1": 2 * prec * rec / max(prec + rec, 1e-9)}

    out_dir = root / "artifacts" / "roberta-span"
    args = _training_args(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=16,
        fp16=torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=SpanDataset(train_mixed),
        eval_dataset=SpanDataset(test_mixed),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    human_pure, ai_pure = pure_docs(data["test"])
    human_scores = [doc_score(d) for d in human_pure[:400]]
    ai_scores = [doc_score(d) for d in ai_pure[:400]]
    calib = calibration_record(human_scores, 0.01)
    calib["human_scores"] = human_scores
    calib["auc_doc_test"] = float(roc_auc_score([0] * len(human_scores) + [1] * len(ai_scores), human_scores + ai_scores))

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "calibration.json").write_text(json.dumps(calib, indent=1), encoding="utf-8")
    manifest = {
        "architecture": "FacebookAI/roberta-base (token classification)",
        "trained_on": ["stitched coai train (3 human + 3 AI sentences per doc)"],
        "never_trained_on": ["coai test"],
        "n_train": len(train_mixed),
        "n_test": len(test_mixed),
        "auc_doc_test": calib["auc_doc_test"],
        "threshold": calib["threshold"],
        "fpr": 0.01,
        "epochs": epochs,
        "max_len": max_len,
    }
    manifest["trained_on"] = [t for t in manifest["trained_on"] if t]
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("exported", out_dir, "doc auc", round(calib["auc_doc_test"], 4), "threshold", round(calib["threshold"], 4))
    return {"dir": out_dir, "calibration": calib}


def demo_span(root: Path, info: dict[str, Any]) -> None:
    """Verdict + which sentences + named why (pattern id, quote, fix)."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from slopdet.explain import explain

    out_dir = Path(info["dir"])
    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    model = AutoModelForTokenClassification.from_pretrained(out_dir)
    calib = info["calibration"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    samples = [
        "Here's the thing, in today's world we leverage robust tools. In conclusion, experts agree it's a pivotal moment.",
        "Thursday mornings at the clinic were empty. Half the early slots sat unused, and the monthly average hid it.",
    ]
    for text in samples:
        sents = split_sentences(text)
        print("=" * 60)
        print(text)
        scores = []
        print("why:")
        for s in sents:
            enc = tokenizer(s, return_tensors="pt", truncation=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                p = float(model(**enc).logits[0].softmax(dim=-1)[:, 1].mean())
            scores.append(p)
            tag = "SLOP" if p >= calib["threshold"] else "not slop"
            print(f"  [{tag}] {s}")
            named = explain(s)
            for hit in named["why_slop"]:
                quote = hit.get("quote") or ""
                print(f"      slop  {hit['id']}: {quote!r}")
                if hit.get("say") or hit.get("fix"):
                    print(f"            {hit.get('say') or hit['fix']}")
            for hit in named["why_human"]:
                quote = hit.get("quote") or ""
                say = hit.get("say") or hit.get("fix") or ""
                print(f"      human {hit['id']}: {quote!r}  {say}")
        overall = float(sum(scores) / len(scores))
        verdict = "SLOP" if overall >= calib["threshold"] else "not slop"
        print(
            f"verdict: {verdict} | resembles the AI pile more than "
            f"{human_percentile(overall, calib['human_scores']):.0f}% of human references"
        )
