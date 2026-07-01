"""The proposed method: Double Decoupling (DAD + TEO).

This module holds the **novel** part of the paper and nothing else. Everything
here operates on already-computed CLIP embeddings; CLIP loading, encoding,
caching and reporting live in :mod:`src.utils`.

The full zero-shot classifier built on top of a frozen CLIP model is::

    img = apply_dad(image_embeddings, subjects)   # if DAD enabled
    txt = apply_teo(text_embeddings)              # if TEO enabled
    logits = img @ txt.T                          # cosine similarity
    prediction = logits.argmax(axis=1)

PE (Prompt Engineering) is not a transform on embeddings -- it only changes
which prompt list is encoded -- so it lives in :mod:`src.prompts` and is applied
when the text embeddings are built.

Components:

* **DAD** -- Driver-specific Appearance Decoupling: subtract each subject's mean
  image embedding, then L2-renormalize. Removes per-driver appearance bias.
* **TEO** -- Text Embedding Orthogonalization: replace the text-embedding matrix
  with its nearest orthonormal frame on the Stiefel manifold (orthogonal
  Procrustes solution). Spreads the class directions apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MethodConfig:
    """On/off switches for the three components."""

    enable_pe: bool = False
    enable_dad: bool = False
    enable_teo: bool = False

    def tag(self) -> str:
        return "+".join(
            name
            for name, on in (("PE", self.enable_pe), ("DAD", self.enable_dad), ("TEO", self.enable_teo))
            if on
        ) or "baseline"


def apply_dad(image_embeddings: np.ndarray, subjects) -> np.ndarray:
    """Driver-specific Appearance Decoupling (DAD).

    For each subject, subtract that subject's mean image embedding (computed over
    the images present in this evaluation set), then L2-normalize every vector.
    No behavior labels are used -- only the driver ID.

    ``image_embeddings`` is ``[N, D]``; ``subjects`` is a length-``N`` sequence of
    driver IDs. Returns a new ``[N, D]`` array.
    """
    subjects = np.asarray(subjects)
    centered = np.zeros_like(image_embeddings)
    for subj in sorted(set(subjects.tolist())):
        idx = np.where(subjects == subj)[0]
        subj_embs = image_embeddings[idx]
        centered[idx] = subj_embs - subj_embs.mean(axis=0)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / norms


def apply_teo(text_embeddings: np.ndarray) -> np.ndarray:
    """Text Embedding Orthogonalization (TEO).

    Given the text-embedding matrix ``T`` of shape ``[C, D]`` (rows = classes),
    take the thin SVD of ``T^T = U Sigma V^T`` and replace the embeddings with the
    rows of ``(U V^T)^T``. This is the orthogonal Procrustes / nearest-Stiefel
    solution: the closest matrix with orthonormal columns to ``T^T``. The result
    is re-normalized row-wise. SVD is done in float32 for numerical stability.
    """
    t = text_embeddings.astype(np.float32)
    u, _s, vt = np.linalg.svd(t.T, full_matrices=False)  # T^T is [D, C]
    q = (u @ vt).astype(np.float32)  # [D, C], columns orthonormal
    out = q.T  # [C, D]
    return out / np.linalg.norm(out, axis=1, keepdims=True)
