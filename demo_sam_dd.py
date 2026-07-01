#!/usr/bin/env python3
"""Zero-shot distracted driver detection on SAM-DD.

This script is intentionally flat: read it top-to-bottom and you see the whole
pipeline. The novel part of the paper is the two lines marked "proposed method"
(see :mod:`src.method`); everything else is standard CLIP zero-shot plumbing
living in :mod:`src.utils`.

Example:
    uv run python demo_sam_dd.py \\
        --data-root /path/to/SAM-DD/Val \\
        --cache-dir .cache/sam_dd \\
        --output-dir results/sam_dd
"""

from src import datasets, method, metrics, prompts, utils

DATASET = "sam-dd"


def main():
    args = utils.build_demo_parser(DATASET, "Zero-shot distracted driver detection on SAM-DD.").parse_args()
    utils.set_seed(args.seed)

    # 1. Index the dataset (image paths, class labels, driver IDs).
    samples, class_names = datasets.load_dataset(
        DATASET, args.data_root, subsample_rate=args.subsample_rate, seed=args.seed
    )
    if args.dry_run:  # just report what would be read, no CLIP.
        print(datasets.describe(DATASET, args.data_root, samples, class_names))
        return

    device = utils.get_device(args.device)
    cfg = method.MethodConfig(args.enable_pe, args.enable_dad, args.enable_teo)
    print(f"[info] device={device}  clip-model={args.clip_model}  images={len(samples)}  method={cfg.tag()}")

    # 2. CLIP image embeddings (expensive -> cached on disk and reused).
    image_emb, subjects, true_labels = utils.get_image_embeddings(
        DATASET, args.data_root, samples, args.clip_model, args.cache_dir,
        device=device, batch_size=args.batch_size, force_recompute=args.force_recompute_cache,
    )

    # 3. CLIP text embeddings. PE = which prompt list we encode (cheap -> no cache).
    clip_model, _ = utils.load_clip(args.clip_model, device=device)
    class_prompts = prompts.get_prompts(DATASET, cfg.enable_pe)
    text_emb = utils.build_text_embeddings(clip_model, class_prompts, prompts.get_templates(), device=device)

    # 4. ----- proposed method: Double Decoupling -----
    img = method.apply_dad(image_emb, subjects) if cfg.enable_dad else image_emb
    txt = method.apply_teo(text_emb) if cfg.enable_teo else text_emb
    logits = img @ txt.T          # cosine similarity (both sides are L2-normalized)
    pred_labels = logits.argmax(axis=1)
    # -------------------------------------------------

    # 5. Score and write result files.
    result = metrics.compute_metrics(true_labels, pred_labels, logits, len(class_names), class_names)
    result["method"], result["dataset"] = cfg.tag(), DATASET
    print("[result] " + metrics.format_summary(result))

    utils.save_results(args.output_dir, result, samples, logits, pred_labels, class_names)
    print(f"[info] results written to {args.output_dir}/")


if __name__ == "__main__":
    main()
