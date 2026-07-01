"""Scaffolding shared by the demo scripts -- everything that is *not* the method.

The novel part of the paper (DAD / TEO) lives in :mod:`src.method`. This module
collects the non-novel plumbing so the demos stay readable:

* CLIP loading and embedding          (``load_clip``, ``build_text_embeddings``, ``encode_images``)
* on-disk image-embedding cache       (``get_image_embeddings``)
* result file writers                 (``save_results``)
* environment helpers                 (``set_seed``, ``get_device``, ``ensure_dir``, ``git_commit_hash``)
* the shared command-line parser      (``build_demo_parser``)

You normally do not need to read this file to understand the method; see
:mod:`src.method` and the ``demo_*.py`` scripts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import tempfile
from datetime import datetime, timezone

import numpy as np

DEFAULT_CLIP_MODEL = "ViT-L/14@336px"


# --------------------------------------------------------------------------- #
# Environment helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs for reproducibility.

    Torch RNG is not seeded here because the zero-shot pipeline is deterministic
    once the (cached) image embeddings exist: there is no sampling at inference
    time. The seed only affects optional sub-sampling of the dataset.
    """
    random.seed(seed)
    np.random.seed(seed)


def get_device(requested: str = "auto") -> str:
    """Resolve the compute device string.

    ``auto`` picks CUDA when available, otherwise CPU.
    """
    if requested not in ("auto", "cuda", "cpu"):
        raise ValueError(f"Unknown device '{requested}'. Use 'auto', 'cuda' or 'cpu'.")
    import torch

    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available on this machine.")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(path: str) -> str:
    """Create ``path`` (and parents) if missing and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def git_commit_hash() -> str | None:
    """Return the current git commit hash, or ``None`` if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# CLIP loading and embedding
# --------------------------------------------------------------------------- #
def load_clip(model_name: str = DEFAULT_CLIP_MODEL, device: str = "cpu"):
    """Load an OpenAI CLIP model and its preprocessing transform.

    Returns ``(model, preprocess)``. The model is set to eval mode.
    """
    import clip
    import torch

    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    torch.set_grad_enabled(False)
    return model, preprocess


def build_text_embeddings(model, class_prompts, templates, device: str = "cpu") -> np.ndarray:
    """Build per-class CLIP text embeddings.

    For each class the prompt is rendered through every template, encoded,
    L2-normalized, averaged over templates, and re-normalized. This mirrors the
    standard CLIP zero-shot classifier construction used in the reference
    implementation. (PE is applied upstream by choosing ``class_prompts``.)

    Returns an array of shape ``[num_classes, embed_dim]`` (rows are classes).
    """
    import clip
    import torch

    weights = []
    with torch.no_grad():
        for classname in class_prompts:
            texts = clip.tokenize([t.format(classname) for t in templates]).to(device)
            class_embeddings = model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding = class_embedding / class_embedding.norm()
            weights.append(class_embedding)
        text_weights = torch.stack(weights, dim=1)  # [D, C]
    return text_weights.t().cpu().numpy()  # [C, D]


def encode_images(model, preprocess, image_paths, device: str = "cpu", batch_size: int = 64):
    """Compute L2-normalized CLIP image embeddings for ``image_paths``.

    Images are loaded, preprocessed and encoded in batches. Each embedding is
    L2-normalized, matching the reference implementation. Returns an array of
    shape ``[N, D]`` in the natural dtype produced by CLIP (float16 on CUDA,
    float32 on CPU).
    """
    import torch
    from PIL import Image
    from tqdm import tqdm

    embeddings = []
    batch: list = []

    def _flush(batch_tensors):
        with torch.no_grad():
            inp = torch.stack(batch_tensors).to(device)
            feats = model.encode_image(inp)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()

    for path in tqdm(image_paths, desc="CLIP image embedding"):
        img = Image.open(path).convert("RGB")
        batch.append(preprocess(img))
        if len(batch) >= batch_size:
            embeddings.append(_flush(batch))
            batch = []
    if batch:
        embeddings.append(_flush(batch))

    return np.concatenate(embeddings, axis=0)


# --------------------------------------------------------------------------- #
# Image-embedding cache
# --------------------------------------------------------------------------- #
# CLIP image embedding is expensive, so embeddings are cached on disk and reused
# automatically. The cache is keyed by the ordered list of image paths plus the
# model / preprocessing settings, and is only reused when all of these match.
#
# Layout::
#     <cache-dir>/
#         image_embeddings.pt        # torch payload: embeddings + labels
#         image_embeddings_meta.json # validation metadata
CACHE_FORMAT_VERSION = 1
EMB_FILE = "image_embeddings.pt"
META_FILE = "image_embeddings_meta.json"


def _hash_paths(paths) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _clip_revision() -> str | None:
    try:
        import clip  # noqa: F401
        from importlib.metadata import version

        return version("clip")
    except Exception:
        return None


def _build_meta(dataset, data_root, samples, clip_model, preprocess_size, dtype) -> dict:
    paths = [s.path for s in samples]
    return {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "dataset": dataset,
        "data_root": os.path.abspath(os.path.expanduser(data_root)),
        "split": samples[0].split if samples else None,
        "num_images": len(samples),
        "image_paths_hash": _hash_paths(paths),
        "clip_model": clip_model,
        "clip_revision": _clip_revision(),
        "preprocess_size": preprocess_size,
        "dtype": dtype,
        "created": datetime.now(timezone.utc).isoformat(),
    }


def _validate(meta: dict, expected: dict) -> list[str]:
    """Return a list of mismatch reasons; empty list means the cache is valid."""
    keys = [
        ("cache_format_version", "cache format version"),
        ("dataset", "dataset name"),
        ("num_images", "image count"),
        ("image_paths_hash", "ordered image path hash"),
        ("clip_model", "CLIP model name"),
        ("preprocess_size", "preprocessing size"),
    ]
    reasons = []
    for key, label in keys:
        if meta.get(key) != expected.get(key):
            reasons.append(f"{label} changed ({meta.get(key)!r} -> {expected.get(key)!r})")
    return reasons


def get_image_embeddings(
    dataset,
    data_root,
    samples,
    clip_model,
    cache_dir,
    device="cpu",
    batch_size=64,
    preprocess_size=336,
    force_recompute=False,
):
    """Load cached image embeddings or compute and cache them.

    Returns ``(embeddings, subjects, class_idx)`` where ``embeddings`` is a
    ``[N, D]`` numpy array and the labels are aligned numpy arrays.
    """
    ensure_dir(cache_dir)
    emb_path = os.path.join(cache_dir, EMB_FILE)
    meta_path = os.path.join(cache_dir, META_FILE)

    # Build the expected metadata up front (dtype filled after a probe encode).
    expected = _build_meta(dataset, data_root, samples, clip_model, preprocess_size, dtype=None)

    if not force_recompute and os.path.isfile(emb_path) and os.path.isfile(meta_path):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            meta = None
        if meta is not None:
            # dtype is informational only; ignore it during validation.
            cmp_expected = dict(expected, dtype=meta.get("dtype"))
            reasons = _validate(meta, cmp_expected)
            if not reasons:
                payload = _load_payload(emb_path, meta)
                if payload is not None:
                    print(f"[cache] reusing {emb_path} ({meta['num_images']} images)")
                    return payload
                print(f"[cache] payload at {emb_path} is unreadable; recomputing.")
            else:
                print("[cache] invalidated; recomputing. Reasons:")
                for r in reasons:
                    print(f"        - {r}")

    print(f"[cache] computing image embeddings for {len(samples)} images ...")
    model, preprocess = load_clip(clip_model, device=device)
    embeddings = encode_images(
        model, preprocess, [s.path for s in samples], device=device, batch_size=batch_size
    )
    subjects = np.array([s.subject for s in samples])
    class_idx = np.array([s.class_idx for s in samples], dtype=np.int64)

    expected["dtype"] = str(embeddings.dtype)
    _save_atomic(emb_path, meta_path, embeddings, subjects, class_idx, expected)
    print(f"[cache] saved {emb_path}")
    return embeddings, subjects, class_idx


def _load_payload(emb_path, meta):
    import torch

    try:
        data = torch.load(emb_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    try:
        embeddings = np.asarray(data["embeddings"])
        subjects = np.asarray(data["subjects"])
        class_idx = np.asarray(data["class_idx"], dtype=np.int64)
    except (KeyError, TypeError):
        return None
    if embeddings.shape[0] != meta.get("num_images"):
        return None
    return embeddings, subjects, class_idx


def _save_atomic(emb_path, meta_path, embeddings, subjects, class_idx, meta):
    import torch

    cache_dir = os.path.dirname(emb_path)
    # Write the payload to a temp file then atomically rename it.
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(
            {
                "embeddings": embeddings,
                "subjects": subjects,
                "class_idx": class_idx,
            },
            tmp,
        )
        os.replace(tmp, emb_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".json.tmp")
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, meta_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# --------------------------------------------------------------------------- #
# Result file writers
# --------------------------------------------------------------------------- #
def save_results(output_dir, metrics, samples, logits, pred_labels, class_names):
    """Write metrics.json/.csv, class-wise metrics, confusion matrix, predictions."""
    ensure_dir(output_dir)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(output_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for key in (
            "top1_accuracy",
            "top3_accuracy",
            "macro_recall",
            "macro_precision",
            "binary_auprc",
            "binary_fnr",
            "num_samples",
        ):
            w.writerow([key, metrics[key]])

    with open(os.path.join(output_dir, "classwise_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_idx", "class_name", "recall", "precision", "support"])
        for row in metrics["classwise"]:
            w.writerow([row["class_idx"], row["class_name"], row["recall"], row["precision"], row["support"]])

    cm = np.asarray(metrics["confusion_matrix"])
    with open(os.path.join(output_dir, "confusion_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + [class_names[i] for i in range(cm.shape[1])])
        for i in range(cm.shape[0]):
            w.writerow([class_names[i]] + cm[i].tolist())

    _save_predictions(output_dir, samples, logits, pred_labels, class_names)


def _save_predictions(output_dir, samples, logits, pred_labels, class_names):
    logits = np.asarray(logits)
    with open(os.path.join(output_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "image_path",
                "subject",
                "gt_class_idx",
                "gt_class_name",
                "pred_class_idx",
                "pred_class_name",
                "top1_score",
                "top3_pred_classes",
            ]
        )
        for i, s in enumerate(samples):
            pred = int(pred_labels[i])
            top3 = np.argsort(logits[i])[-3:][::-1].tolist()
            w.writerow(
                [
                    s.path,
                    s.subject,
                    s.class_idx,
                    s.class_name,
                    pred,
                    class_names[pred],
                    f"{float(logits[i, pred]):.6f}",
                    "|".join(str(c) for c in top3),
                ]
            )


# --------------------------------------------------------------------------- #
# Command-line parser shared by the demo scripts
# --------------------------------------------------------------------------- #
def _add_toggle(parser, name, dest, default, help_text):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--enable-{name}", dest=dest, action="store_true", help=f"Enable {help_text}.")
    group.add_argument(f"--disable-{name}", dest=dest, action="store_false", help=f"Disable {help_text}.")
    parser.set_defaults(**{dest: default})


def build_demo_parser(dataset: str, description: str) -> argparse.ArgumentParser:
    """Build the argument parser shared by the demo scripts.

    PE/DAD/TEO default to enabled so the bare command reproduces the full
    proposed method ("Ours"); pass the matching ``--disable-*`` flag to turn a
    component off.
    """
    tag = dataset.replace("-", "_")
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data-root", required=True, help=f"Path to the {dataset} dataset.")
    p.add_argument("--cache-dir", default=f".cache/{tag}", help="Embedding cache directory.")
    p.add_argument("--output-dir", default=f"results/{tag}", help="Where to write result files.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Compute device.")
    p.add_argument("--batch-size", type=int, default=64, help="Image-encoding batch size.")
    p.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL, help="CLIP model name.")
    p.add_argument("--subsample-rate", type=float, default=1.0, help="Fraction of images per class (1.0 = all).")
    _add_toggle(p, "pe", "enable_pe", True, "Prompt Engineering (PE)")
    _add_toggle(p, "dad", "enable_dad", True, "Driver-specific Appearance Decoupling (DAD)")
    _add_toggle(p, "teo", "enable_teo", True, "Text Embedding Orthogonalization (TEO)")
    p.add_argument("--force-recompute-cache", action="store_true", help="Ignore any existing embedding cache.")
    p.add_argument("--dry-run", action="store_true", help="Print dataset statistics and exit (no CLIP).")
    p.add_argument("--seed", type=int, default=0, help="Random seed (affects sub-sampling only).")
    return p
