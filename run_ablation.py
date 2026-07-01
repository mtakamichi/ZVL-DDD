#!/usr/bin/env python3
"""Run the full PE/DAD/TEO ablation and save a CSV.

Runs all eight on/off combinations of PE, DAD and TEO on a single dataset,
reusing one cached set of image embeddings, and writes one row per combination.

Example:
    uv run python run_ablation.py \\
        --dataset sam-dd \\
        --data-root /path/to/SAM-DD/Val \\
        --cache-dir .cache/sam_dd \\
        --output results/ablation_sam_dd.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os

from src import datasets, method, metrics, prompts, utils
from src.utils import DEFAULT_CLIP_MODEL


def main():
    p = argparse.ArgumentParser(description="PE/DAD/TEO ablation.")
    p.add_argument("--dataset", required=True, choices=["sam-dd", "statefarm"])
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--output", default=None, help="Output CSV path.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    p.add_argument("--subsample-rate", type=float, default=1.0)
    p.add_argument("--force-recompute-cache", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dataset = args.dataset
    cache_dir = args.cache_dir or f".cache/{dataset.replace('-', '_')}"
    output = args.output or f"results/ablation_{dataset.replace('-', '_')}.csv"

    utils.set_seed(args.seed)
    samples, class_names = datasets.load_dataset(
        dataset, args.data_root, subsample_rate=args.subsample_rate, seed=args.seed
    )
    device = utils.get_device(args.device)
    print(f"[info] device={device}  images={len(samples)}")

    image_embeddings, subjects, true_labels = utils.get_image_embeddings(
        dataset, args.data_root, samples, args.clip_model, cache_dir,
        device=device, batch_size=args.batch_size, force_recompute=args.force_recompute_cache,
    )

    # Build both PE-off and PE-on text embeddings once.
    model, _ = utils.load_clip(args.clip_model, device=device)
    templates = prompts.get_templates()
    text_by_pe = {
        pe: utils.build_text_embeddings(model, prompts.get_prompts(dataset, pe), templates, device=device)
        for pe in (False, True)
    }

    rows = []
    for pe, dad, teo in itertools.product([False, True], repeat=3):
        config = method.MethodConfig(pe, dad, teo)
        # Same Double Decoupling pipeline as the demos (see src/method.py).
        img = method.apply_dad(image_embeddings, subjects) if dad else image_embeddings
        txt = method.apply_teo(text_by_pe[pe]) if teo else text_by_pe[pe]
        logits = img @ txt.T
        preds = logits.argmax(axis=1)
        m = metrics.compute_metrics(true_labels, preds, logits, len(class_names), class_names)
        print(f"  {config.tag():16s}  {metrics.format_summary(m)}")
        rows.append(
            {
                "PE": "on" if pe else "off",
                "DAD": "on" if dad else "off",
                "TEO": "on" if teo else "off",
                "top1": round(m["top1_accuracy"] * 100, 2),
                "top3": round(m["top3_accuracy"] * 100, 2),
                "recall": round(m["macro_recall"] * 100, 2),
                "precision": round(m["macro_precision"] * 100, 2),
                "auprc": round(m["binary_auprc"] * 100, 2),
                "fnr": round(m["binary_fnr"] * 100, 2),
            }
        )

    utils.ensure_dir(os.path.dirname(output) or ".")
    with open(output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[info] ablation written to {output}")


if __name__ == "__main__":
    main()
