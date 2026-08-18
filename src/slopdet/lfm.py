"""LFM2 bidirectional encoder loading.

The trap: `AutoModel.from_pretrained("LiquidAI/LFM2.5-Encoder-230M", trust_remote_code=True)` — the
snippet the model card gives for downstream heads — returns an `Lfm2BidirectionalModel` whose weights
are **all freshly initialized**. The checkpoint stores body tensors under the `lfm2.` prefix, so the
base-model class reports every parameter missing and every checkpoint key unexpected, then hands back a
random encoder that trains and evaluates without ever erroring.

Load the masked-LM wrapper (which owns the `lfm2.` prefix) and take its body instead, and assert the
load was clean so this can never regress silently.
"""

from __future__ import annotations

try:
    import torch  # noqa: F401
    from transformers import AutoModelForMaskedLM
except ImportError as exc:  # pragma: no cover
    raise ImportError("lfm.py needs torch + transformers. Install the train extra.") from exc


def load_encoder_body(model_name: str, dtype=None):
    """Return a fully-loaded LFM2 bidirectional encoder body.

    Raises if any body tensor was initialized from scratch, which is what happens when the wrong
    auto-class is used.
    """
    mlm, info = AutoModelForMaskedLM.from_pretrained(
        model_name, trust_remote_code=True, dtype=dtype, output_loading_info=True
    )
    missing = [key for key in info.get("missing_keys", []) if not key.startswith("lm_head")]
    if missing:
        raise SystemExit(
            f"{model_name}: {len(missing)} body tensors were newly initialized "
            f"(first: {missing[:3]}). Refusing to train on a random encoder."
        )
    return mlm.lfm2
