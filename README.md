# ZVL-DDD: Zero-Shot Distracted Driver Detection via Vision–Language Models with Double Decoupling

Public implementation of the paper:

> Takamichi Miyata, Sumiko Miyata, Andrew Morris,
> **"Zero-Shot Distracted Driver Detection via Vision Language Models with Double Decoupling,"**
> IEEE International Symposium on Communication Systems, Networks and Digital Signal Processing (CSNDSP), 2026

The method performs **zero-shot** distracted-driver classification with a frozen
CLIP model — no training, no fine-tuning — and improves over the DriveCLIP-style
naive zero-shot baseline with three lightweight, independently switchable
components:

- **PE — Prompt Engineering.** Dataset-specific class prompts.
- **DAD — Driver-specific Appearance Decoupling.** Subtract each driver's mean
  image embedding to remove appearance bias, then re-normalize.
- **TEO — Text Embedding Orthogonalization.** Replace the text-embedding matrix
  by its nearest orthonormal frame on the Stiefel manifold (orthogonal
  Procrustes solution).

A Japanese version of this document is available in [README-JP.md](README-JP.md).

---

## Overview

```
image ──► CLIP image encoder ──► (DAD) ──┐
                                         ├──► cosine similarity ──► argmax ──► class
prompts ─► CLIP text encoder ──► (TEO) ──┘
```

The baseline is the standard CLIP zero-shot classifier
`prediction = argmax_c cos(image_embedding, text_embedding_c)` using
**ViT-L/14@336px**. PE, DAD and TEO can each be toggled on or off.

---

## Code layout

```
demo_sam_dd.py       # SAM-DD evaluation
demo_statefarm.py    # StateFarm evaluation
run_ablation.py      # all 8 PE/DAD/TEO combinations on one cached embedding set
src/
  method.py          # DAD + TEO implementation
  datasets.py        # SAM-DD / StateFarm loaders
  prompts.py         # baseline vs PE prompt lists (PE lives here)
  metrics.py         # top-1/3, macro recall/precision, binary AUPRC/FNR
  utils.py           # CLIP load/encode, embedding cache,
                     #   result writers, device/seed, the CLI parser
```

## Installation

This project uses [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python demo_sam_dd.py --help
```

`uv sync` installs PyTorch (`2.6.0+cu126`), the OpenAI CLIP package, and — on
Linux/x86_64 — the matching NVIDIA CUDA runtime libraries. CLIP model weights are
downloaded automatically by the `clip` package on first use (cached under
`~/.cache/clip`).

### CUDA / CPU notes

- The pinned Torch build targets **CUDA 12.6**. The `nvidia-*-cu12` packages in
  `pyproject.toml` provide the CUDA runtime libraries that the `+cu126` wheels do
  not bundle (including `libcusparseLt.so.0`); they are platform-gated and are
  skipped on non-Linux platforms.
- To run on CPU, pass `--device cpu`. Note that ViT-L/14@336px on CPU is slow;
  GPU is strongly recommended for the full datasets.
- If you need a different CUDA version, adjust the `torch`/`torchvision`/`nvidia-*`
  pins and the `[[tool.uv.index]]` URL in `pyproject.toml` to match your system.

---

## Datasets

**This repository does not redistribute SAM-DD, StateFarm, or any other dataset.**
Please download each dataset from its official source and comply with its terms
of use. The repository license applies only to the implementation code, not to
the datasets or the pretrained CLIP weights.

Dataset paths are passed on the command line; nothing is hard-coded. In the
commands below, replace the placeholders with your own paths:

- `<SAM_DD_DATA_ROOT>` — the directory that directly contains the SAM-DD subject
  folders (e.g. `.../SAM-DD/Val`).
- `<STATEFARM_DATA_ROOT>` — the StateFarm dataset root containing `imgs/` and
  `driver_imgs_list.csv`.

### Expected layout

**SAM-DD** (`<SAM_DD_DATA_ROOT>` points at the split directory, e.g. `Val`):

```
<SAM_DD_DATA_ROOT>/
  Val1/                 # subject / driver ID
    0/ side_RGB/*.jpg   # class 0..9, side RGB camera view
    1/ side_RGB/*.jpg
    ...
  Val2/
  ...
```

The subject/driver ID is the top-level folder name (`Val1`, `Val2`, ...), which
DAD uses to compute per-driver mean embeddings.

**StateFarm** (`<STATEFARM_DATA_ROOT>` points at the dataset root):

```
<STATEFARM_DATA_ROOT>/
  imgs/train/c0/*.jpg   # class c0..c9
  imgs/train/c1/*.jpg
  ...
  driver_imgs_list.csv  # maps each image file name to its subject (e.g. p002)
```

The subject/driver ID is looked up from `driver_imgs_list.csv` by file name.

### Inspecting a dataset (no CLIP)

```bash
uv run python demo_sam_dd.py --data-root <SAM_DD_DATA_ROOT> --dry-run
```

This prints image counts, class distribution and subject distribution without
running any CLIP embedding.

---

## Running the evaluations

By default PE, DAD and TEO are **all enabled**, so the bare command reproduces
the full proposed method ("Ours"). Use the matching `--disable-*` flag to turn a
component off.

### SAM-DD

```bash
uv run python demo_sam_dd.py \
  --data-root <SAM_DD_DATA_ROOT> \
  --cache-dir .cache/sam_dd \
  --output-dir results/sam_dd \
  --enable-pe --enable-dad --enable-teo
```

### StateFarm

```bash
uv run python demo_statefarm.py \
  --data-root <STATEFARM_DATA_ROOT> \
  --cache-dir .cache/statefarm \
  --output-dir results/statefarm \
  --enable-pe --enable-dad --enable-teo
```

### Baseline (DriveCLIP-style naive zero-shot)

```bash
uv run python demo_sam_dd.py --data-root <SAM_DD_DATA_ROOT> \
  --disable-pe --disable-dad --disable-teo
```

### Ablation (all 8 PE/DAD/TEO combinations)

```bash
uv run python run_ablation.py \
  --dataset sam-dd \
  --data-root <SAM_DD_DATA_ROOT> \
  --cache-dir .cache/sam_dd \
  --output results/ablation_sam_dd.csv
```

The first run computes (and caches) the image embeddings once, then reuses them
for all eight combinations.

---

## Command-line options

Common to `demo_sam_dd.py` and `demo_statefarm.py`:

| Option | Description |
|---|---|
| `--data-root PATH` | Dataset path (required). |
| `--cache-dir PATH` | Image-embedding cache directory. |
| `--output-dir PATH` | Where result files are written. |
| `--device auto\|cuda\|cpu` | Compute device (`auto` picks CUDA if available). |
| `--batch-size INT` | Image-encoding batch size. |
| `--clip-model NAME` | CLIP model (default `ViT-L/14@336px`). |
| `--enable-pe / --disable-pe` | Toggle Prompt Engineering. |
| `--enable-dad / --disable-dad` | Toggle Driver-specific Appearance Decoupling. |
| `--enable-teo / --disable-teo` | Toggle Text Embedding Orthogonalization. |
| `--subsample-rate FLOAT` | Fraction of images per class (default `1.0` = all). |
| `--force-recompute-cache` | Ignore any existing embedding cache. |
| `--dry-run` | Print dataset statistics and exit (no CLIP). |
| `--seed INT` | Random seed (affects sub-sampling only). |

---

## Embedding cache

CLIP image embedding is expensive, so embeddings are cached and reused
automatically:

```
<cache-dir>/
  image_embeddings.pt          # embeddings + subject/class labels
  image_embeddings_meta.json   # validation metadata
```

- On the first run the embeddings are computed and saved.
- On later runs the cache is reused only if its metadata matches: dataset name,
  image count, the **ordered image-path hash**, the CLIP model name, the
  preprocessing size and the cache format version.
- If the cache is missing, invalid, or its image ordering has changed, it is
  recomputed automatically (with a printed explanation). `--force-recompute-cache`
  ignores any existing cache.
- The cache is written to a temporary file and atomically renamed, so a partial
  or corrupted cache is never reused.

Text embeddings are cheap and are always recomputed (they depend on the PE
switch).

---

## Expected results

### Main 10-class evaluation

| Method | Dataset | Top-1 | Top-3 | Recall | Precision |
|---|---|---:|---:|---:|---:|
| DriveCLIP (baseline) | SAM-DD | 66.5 | 85.8 | 44.8 | 44.7 |
| **Ours (PE+DAD+TEO)** | SAM-DD | **75.9** | **96.9** | **68.4** | **70.4** |
| DriveCLIP (baseline) | StateFarm | 45.5 | 76.6 | 44.4 | 48.4 |
| **Ours (PE+DAD+TEO)** | StateFarm | **54.6** | **89.3** | **54.6** | **55.7** |

### Binary safe-vs-distracted evaluation

(class 0 = safe driving, classes 1–9 = distracted)

| Method | Dataset | 2C-AUPRC | 2C-FNR |
|---|---|---:|---:|
| DriveCLIP (baseline) | SAM-DD | 90.6 | 32.6 |
| **Ours** | SAM-DD | **95.8** | **10.9** |
| DriveCLIP (baseline) | StateFarm | 95.6 | 20.9 |
| **Ours** | StateFarm | **97.1** | **11.9** |

### SAM-DD ablation

> **Note.** The intermediate ablation rows below are the values **produced by this
> implementation** (`run_ablation.py`). They supersede the corresponding table in
> the paper, whose intermediate single/double-component rows contain errors
> introduced during a last-minute revision. The headline rows (baseline, PE+DAD,
> and the full method) are unchanged and match the paper. The essential
> conclusion is also unchanged: **only the full PE+DAD+TEO combination achieves
> the best performance on every metric.**

| PE | DAD | TEO | Top-1 | Top-3 | Recall | Precision |
|:--:|:--:|:--:|---:|---:|---:|---:|
| off | off | off | 66.6 | 85.8 | 44.9 | 44.8 |
| on  | off | off | 66.3 | 89.3 | 40.2 | 53.7 |
| off | on  | off | 45.6 | 90.0 | 57.3 | 49.0 |
| off | off | on  | 53.6 | 78.1 | 29.7 | 36.8 |
| on  | on  | off | 57.2 | 94.9 | 63.6 | 66.2 |
| on  | off | on  | 64.5 | 84.3 | 39.3 | 52.4 |
| off | on  | on  | 40.0 | 88.0 | 53.5 | 47.7 |
| **on** | **on** | **on** | **76.0** | **96.9** | **68.4** | **70.4** |


---

## Output files

Each evaluation writes, under `--output-dir`:

```
results/<dataset>/
  metrics.json            # all metrics
  metrics.csv             # scalar metrics
  classwise_metrics.csv   # per-class recall / precision / support
  confusion_matrix.csv    # 10x10 confusion matrix
  predictions.csv         # per-image predictions
```

`predictions.csv` contains, per image: image path, subject/driver ID,
ground-truth class index and name, predicted class index and name, the top-1
score, and the top-3 predicted classes.

`run_ablation.py` writes a CSV with one row per PE/DAD/TEO combination.

---

## License

The implementation code is released under the **Apache License 2.0** (see
[LICENSE](LICENSE)).

**Disclaimer.** The Apache-2.0 license applies only to this repository's source
code. It does **not** apply to, and does not grant any rights over:

- the SAM-DD, StateFarm, or any other dataset — obtain these from their official
  sources under their own licenses/terms;
- the pretrained CLIP model weights — subject to OpenAI CLIP's license/terms;
- third-party libraries — subject to their respective licenses.

## Citation

```bibtex
@inproceedings{miyata2026_ZVL-DDD,
  author    = {Miyata, Takamichi and Miyata, Sumiko and Morris, Andrew},
  title     = {Zero-Shot Distracted Driver Detection via Vision Language Models with Double Decoupling},
  booktitle = {Proceedings of the IEEE International Symposium on Communication Systems, Networks and Digital Signal Processing (CSNDSP)},
  year      = {2026}
}
```
