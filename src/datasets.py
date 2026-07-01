"""SAM-DD and StateFarm dataset loaders.

Each loader returns a list of :class:`Sample` records and the dataset's
human-readable class names. A sample exposes:

* ``path``        -- absolute image path
* ``class_idx``   -- integer class label in ``0..9``
* ``class_name``  -- human-readable class name
* ``subject``     -- subject / driver ID (required by DAD)
* ``split``       -- split name if available, else ``None``

DAD subtracts a per-subject mean image embedding, so the subject ID inference is
made explicit per dataset:

* SAM-DD    -- subject is the top-level folder name under ``--data-root``
               (e.g. ``Val1``, ``Val2`` ...).
* StateFarm -- subject is looked up from ``driver_imgs_list.csv`` by file name
               (the ``subject`` column, e.g. ``p002``).
"""

from __future__ import annotations

import os
import random
from collections import Counter
from dataclasses import dataclass

from . import prompts

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# SAM-DD camera modality folder used for evaluation (side RGB view).
SAM_DD_MODALITY = "side_RGB"


@dataclass(frozen=True)
class Sample:
    path: str
    class_idx: int
    class_name: str
    subject: str
    split: str | None = None


def load_dataset(dataset: str, data_root: str, subsample_rate: float = 1.0, seed: int = 0):
    """Dispatch to the dataset-specific loader.

    Returns ``(samples, class_names)``.
    """
    d = prompts._norm(dataset)
    if d == "sam-dd":
        return load_sam_dd(data_root, subsample_rate=subsample_rate, seed=seed)
    return load_statefarm(data_root, subsample_rate=subsample_rate, seed=seed)


def _subsample(paths: list[str], rate: float, rng: random.Random) -> list[str]:
    if rate >= 1.0:
        return list(paths)
    selected = [p for p in paths if rng.random() < rate]
    if not selected and paths:  # keep at least one image per group
        selected = [rng.choice(paths)]
    return selected


def load_sam_dd(data_root: str, subsample_rate: float = 1.0, seed: int = 0):
    """Load the SAM-DD evaluation split.

    ``data_root`` must point at the directory that directly contains the subject
    folders (e.g. ``.../SAM-DD/Val``). Each subject folder must contain class
    folders ``0``..``9``, each with a ``side_RGB`` sub-folder of images.
    """
    data_root = os.path.expanduser(data_root)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"SAM-DD data-root not found: {data_root}")

    split = os.path.basename(os.path.normpath(data_root)) or None
    class_names = prompts.get_display_names("sam-dd")
    rng = random.Random(seed)

    subjects = sorted(
        d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))
    )
    if not subjects:
        raise RuntimeError(
            f"No subject folders found under {data_root}. "
            "Expected sub-directories such as 'Val1', 'Val2', ..."
        )

    samples: list[Sample] = []
    for subj in subjects:
        for cls in range(10):
            folder = os.path.join(data_root, subj, str(cls), SAM_DD_MODALITY)
            if not os.path.isdir(folder):
                continue
            imgs = [
                os.path.join(root, fn)
                for root, _, files in os.walk(folder)
                for fn in files
                if fn.lower().endswith(VALID_EXTS)
            ]
            for path in _subsample(sorted(imgs), subsample_rate, rng):
                samples.append(
                    Sample(path, cls, class_names[cls], subj, split)
                )

    if not samples:
        raise RuntimeError(
            f"No images found under {data_root}. Expected layout: "
            f"<subject>/<class 0..9>/{SAM_DD_MODALITY}/*.jpg"
        )
    return samples, class_names


def load_statefarm(data_root: str, subsample_rate: float = 1.0, seed: int = 0):
    """Load the StateFarm training images used for zero-shot evaluation.

    ``data_root`` may point either at the dataset root (containing ``imgs/train``
    and ``driver_imgs_list.csv``) or directly at the ``imgs/train`` folder; in the
    latter case the CSV is searched two levels up.
    """
    import pandas as pd

    data_root = os.path.expanduser(data_root)
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"StateFarm data-root not found: {data_root}")

    train_dir, csv_path = _resolve_statefarm_paths(data_root)
    class_names = prompts.get_display_names("statefarm")
    rng = random.Random(seed)

    df = pd.read_csv(csv_path)
    if not {"subject", "img"}.issubset(df.columns):
        raise RuntimeError(
            f"Unexpected StateFarm CSV format at {csv_path}; "
            "expected 'subject', 'classname', 'img' columns."
        )
    img_to_subject = dict(zip(df["img"], df["subject"]))

    samples: list[Sample] = []
    for cls in range(10):
        folder = os.path.join(train_dir, f"c{cls}")
        if not os.path.isdir(folder):
            continue
        imgs = [
            os.path.join(folder, fn)
            for fn in os.listdir(folder)
            if fn.lower().endswith(VALID_EXTS)
        ]
        for path in _subsample(sorted(imgs), subsample_rate, rng):
            subj = img_to_subject.get(os.path.basename(path), "unknown")
            samples.append(Sample(path, cls, class_names[cls], subj, "train"))

    if not samples:
        raise RuntimeError(
            f"No images found under {train_dir}. Expected class folders c0..c9."
        )
    return samples, class_names


def _resolve_statefarm_paths(data_root: str) -> tuple[str, str]:
    """Return ``(train_dir, csv_path)`` from a flexible StateFarm root."""
    candidates = [
        (os.path.join(data_root, "imgs", "train"), os.path.join(data_root, "driver_imgs_list.csv")),
        (data_root, os.path.join(os.path.dirname(os.path.dirname(data_root)), "driver_imgs_list.csv")),
    ]
    for train_dir, csv_path in candidates:
        if os.path.isdir(train_dir) and os.path.isfile(csv_path):
            return train_dir, csv_path
    raise RuntimeError(
        f"Could not locate StateFarm 'imgs/train' and 'driver_imgs_list.csv' from {data_root}. "
        "Point --data-root at the dataset root containing 'imgs/' and 'driver_imgs_list.csv'."
    )


def describe(dataset: str, data_root: str, samples, class_names) -> str:
    """Return a human-readable dataset summary (used by ``--dry-run``)."""
    class_dist = Counter(s.class_idx for s in samples)
    subj_dist = Counter(s.subject for s in samples)
    lines = [
        f"Dataset: {dataset}",
        f"Root: {data_root}",
        f"Num images: {len(samples)}",
        f"Num classes: {len(class_names)}",
        f"Num subjects/drivers: {len(subj_dist)}",
        "Class distribution:",
    ]
    for idx in range(len(class_names)):
        lines.append(f"  [{idx}] {class_names[idx]}: {class_dist.get(idx, 0)}")
    lines.append("Subject distribution:")
    for subj in sorted(subj_dist):
        lines.append(f"  {subj}: {subj_dist[subj]}")
    return "\n".join(lines)
