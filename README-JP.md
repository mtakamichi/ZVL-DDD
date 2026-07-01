# ZVL-DDD: 二重デカップリングを用いた視覚言語モデルによるゼロショット脇見運転検出

論文の公開実装です。

> Takamichi Miyata, Sumiko Miyata, Andrew Morris, 
> **"Zero-Shot Distracted Driver Detection via Vision Language Models with Double Decoupling,"**
> IEEE International Symposium on Communication Systems, Networks and Digital Signal Processing (CSNDSP), 2026

本手法は、学習ずみのCLIP モデルを用いて運転者不注意状態を ゼロショット（学習・ファイン
チューニングなし）で検出します。DriveCLIP のゼロショット推論をベースライン
とし、以下の3つの軽量な要素で精度を向上させます。

- **PE — プロンプトエンジニアリング (Prompt Engineering).** データセットごとに調整したクラスプロンプトを用います。
- **DAD — 運転者固有外観デカップリング (Driver-specific Appearance Decoupling).** 各運転者の平均画像埋め込みを差し引くことで運転者の外観によるバイアスを除去します。
- **TEO — テキスト埋め込み直交化 (Text Embedding Orthogonalization).** テキスト埋め込み行列を、Stiefel多様体上で最も近い正規直交フレームに置き換えることでテキスト埋め込み間の相関を除去します。


英語版は [README.md](README.md) を参照してください。

---

## 概要

```
画像 ──► CLIP image enc. ──► (DAD)  ──┐
                                      ├──► cos類似度 ──► argmax ──► クラス
プロンプト ─► CLIP text Enc. ─► (TEO) ─┘
```

ベースラインは標準的な CLIP ゼロショット分類器
`予測 = argmax_c cos(画像埋め込み, テキスト埋め込み_c)` であり、
バックボーンには **ViT-L/14@336px** を用います。PE・DAD・TEO はそれぞれ
個別に有効/無効を切り替えられます。

---

## コード構成

```
demo_sam_dd.py       # SAM-DD データセットでの評価
demo_statefarm.py    # StateFarm データセットでの評価
run_ablation.py      # PE/DAD/TEO 全8通りのablationを実行
src/
  method.py          # DAD + TEOの実装はここ
  datasets.py        # SAM-DD / StateFarm のデータローダ
  prompts.py         # ベースライン / PE のプロンプト（PE はここ）
  metrics.py         # top-1/3、マクロ recall/precision、2 値 AUPRC/FNR
  utils.py           # CLIP ロード/エンコード、埋め込み
                     # キャッシュ、結果出力、デバイス/シード、CLI パーサ等
```


## インストール

本プロジェクトは [`uv`](https://docs.astral.sh/uv/) を使用します。

```bash
uv sync
uv run python demo_sam_dd.py --help
```

`uv sync` により PyTorch（`2.6.0+cu126`）、OpenAI CLIP パッケージ、そして
Linux/x86_64 では対応する NVIDIA CUDA ランタイムライブラリが導入されます。
CLIP の重みは初回利用時に `clip` パッケージが自動でダウンロードします
（`~/.cache/clip` にキャッシュ）。

### CUDA / CPU についての注意

- 固定している Torch ビルドは **CUDA 12.6** 向けです。`pyproject.toml` の
  `nvidia-*-cu12` パッケージ群は、`+cu126` ホイールが同梱していない CUDA ランタイム
  ライブラリ（`libcusparseLt.so.0` を含む）を補います。これらはプラットフォーム条件付き
  で、Linux 以外では導入されません。
- CPU で実行する場合は `--device cpu` を指定してください。ただし ViT-L/14@336px は
  CPU では非常に低速です。フルデータセットでは GPU を強く推奨します。
- 異なる CUDA バージョンが必要な場合は、`pyproject.toml` の `torch` /
  `torchvision` / `nvidia-*` のバージョン指定と `[[tool.uv.index]]` の URL を
  環境に合わせて変更してください。

---

## データセット

**本リポジトリは SAM-DD・StateFarm をはじめ、いかなるデータセットも再配布しません。**
各データセットは公式の配布元から入手し、その利用規約・ライセンスに従ってください。
本リポジトリのライセンスは実装コードにのみ適用され、データセットや学習済み CLIP の
重みには適用されません。

データセットのパスはコマンドライン引数で渡します（コード内にハードコードはしていま
せん）。以下のコマンド例では、プレースホルダをご自身のパスに置き換えてください。

- `<SAM_DD_DATA_ROOT>` … SAM-DD の被験者フォルダを直接含むディレクトリ
  （例: `.../SAM-DD/Val`）。
- `<STATEFARM_DATA_ROOT>` … `imgs/` と `driver_imgs_list.csv` を含む StateFarm の
  データセットルート。

### 想定するディレクトリ構成

**SAM-DD**（`<SAM_DD_DATA_ROOT>` は分割ディレクトリ、例えば `Val` を指す）:

```
<SAM_DD_DATA_ROOT>/
  Val1/                 # 被験者 / 運転者 ID
    0/ side_RGB/*.jpg   # クラス 0〜9、側方 RGB カメラ視点
    1/ side_RGB/*.jpg
    ...
  Val2/
  ...
```

被験者/運転者 ID は最上位のフォルダ名（`Val1`, `Val2`, …）であり、DAD はこれを用いて
運転者ごとの平均埋め込みを計算します。

**StateFarm**（`<STATEFARM_DATA_ROOT>` はデータセットルートを指す）:

```
<STATEFARM_DATA_ROOT>/
  imgs/train/c0/*.jpg   # クラス c0〜c9
  imgs/train/c1/*.jpg
  ...
  driver_imgs_list.csv  # 各画像ファイル名 → 被験者（例: p002）の対応表
```

被験者/運転者 ID は `driver_imgs_list.csv` からファイル名で参照します。

### データセットの確認（CLIP 不要）

```bash
uv run python demo_sam_dd.py --data-root <SAM_DD_DATA_ROOT> --dry-run
```

CLIP の埋め込みを一切実行せず、画像枚数・クラス分布・被験者分布を表示します。

---

## 評価の実行

既定では PE・DAD・TEO が **すべて有効** になっているため、引数なしのコマンドで提案
手法（フル手法、"Ours"）を再現します。要素を無効化するには対応する `--disable-*`
フラグを指定します。

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

### ベースライン（DriveCLIPのゼロショット認識）

```bash
uv run python demo_sam_dd.py --data-root <SAM_DD_DATA_ROOT> \
  --disable-pe --disable-dad --disable-teo
```

### アブレーション（PE/DAD/TEO の全 8 通り）

```bash
uv run python run_ablation.py \
  --dataset sam-dd \
  --data-root <SAM_DD_DATA_ROOT> \
  --cache-dir .cache/sam_dd \
  --output results/ablation_sam_dd.csv
```

初回に画像埋め込みを 1 度だけ計算してキャッシュし、8 通りすべてで再利用します。

---

## コマンドライン引数

`demo_sam_dd.py` と `demo_statefarm.py` で共通:

| 引数 | 説明 |
|---|---|
| `--data-root PATH` | データセットのパス（必須）。 |
| `--cache-dir PATH` | 画像埋め込みキャッシュのディレクトリ。 |
| `--output-dir PATH` | 結果ファイルの出力先。 |
| `--device auto\|cuda\|cpu` | 計算デバイス（`auto` は CUDA があれば使用）。 |
| `--batch-size INT` | 画像エンコードのバッチサイズ。 |
| `--clip-model NAME` | CLIP モデル名（既定 `ViT-L/14@336px`）。 |
| `--enable-pe / --disable-pe` | プロンプトエンジニアリングの切替。 |
| `--enable-dad / --disable-dad` | 運転者固有外観デカップリングの切替。 |
| `--enable-teo / --disable-teo` | テキスト埋め込み直交化の切替。 |
| `--subsample-rate FLOAT` | クラスあたりの画像使用率（既定 `1.0` = 全件）。 |
| `--force-recompute-cache` | 既存キャッシュを無視して再計算。 |
| `--dry-run` | データセット統計を表示して終了（CLIP 不要）。 |
| `--seed INT` | 乱数シード（サブサンプリングのみに影響）。 |

---

## 埋め込みキャッシュ

CLIP の画像埋め込みは計算コストが高いため、自動的にキャッシュして再利用します。

```
<cache-dir>/
  image_embeddings.pt          # 埋め込み + 被験者/クラスラベル
  image_embeddings_meta.json   # 検証用メタデータ
```

- 初回実行時に埋め込みを計算して保存します。
- 2 回目以降は、メタデータが一致した場合のみ再利用します（データセット名、画像枚数、
  **順序付き画像パスのハッシュ**、CLIP モデル名、前処理サイズ、キャッシュ形式バージョン）。
- キャッシュが存在しない・無効・画像の並び順が変わった場合は、理由を表示したうえで
  自動的に再計算します。`--force-recompute-cache` で既存キャッシュを無視できます。
- キャッシュは一時ファイルに書き出してからアトミックにリネームするため、破損・不完全
  なキャッシュが再利用されることはありません。

テキスト埋め込みは安価なので毎回再計算します（PE スイッチに依存するため）。

---

## 期待される結果

### 10 クラス評価（メイン）

| 手法 | データセット | Top-1 | Top-3 | Recall | Precision |
|---|---|---:|---:|---:|---:|
| DriveCLIP（ベースライン） | SAM-DD | 66.5 | 85.8 | 44.8 | 44.7 |
| **提案手法（PE+DAD+TEO）** | SAM-DD | **75.9** | **96.9** | **68.4** | **70.4** |
| DriveCLIP（ベースライン） | StateFarm | 45.5 | 76.6 | 44.4 | 48.4 |
| **提案手法（PE+DAD+TEO）** | StateFarm | **54.6** | **89.3** | **54.6** | **55.7** |

### 2 値（安全 vs 脇見）評価

（クラス 0 = 安全運転、クラス 1〜9 = 脇見運転）

| 手法 | データセット | 2C-AUPRC | 2C-FNR |
|---|---|---:|---:|
| DriveCLIP（ベースライン） | SAM-DD | 90.6 | 32.6 |
| **提案手法** | SAM-DD | **95.8** | **10.9** |
| DriveCLIP（ベースライン） | StateFarm | 95.6 | 20.9 |
| **提案手法** | StateFarm | **97.1** | **11.9** |

### SAM-DD アブレーション

> **注記.** 以下の中間行は、**本実装（`run_ablation.py`）が出力した値** です。論文の
> 対応する表のうち、単独・2 要素の中間行は最終改訂時の手違いにより誤りを含んでいた
> ため、本リポジトリの値を正としてこちらに掲載します。ヘッドライン行（ベースライン、
> PE+DAD、フル手法）は論文と一致しており変更ありません。本質的な結論も変わりません。
> すなわち **3 要素すべてを揃えたフル手法 PE+DAD+TEO のときに、あらゆる指標で最良の
> 性能が得られます。**

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

## 出力ファイル

各評価は `--output-dir` 以下に次を出力します。

```
results/<dataset>/
  metrics.json            # 全指標
  metrics.csv             # スカラー指標
  classwise_metrics.csv   # クラス別 recall / precision / 件数
  confusion_matrix.csv    # 10x10 混同行列
  predictions.csv         # 画像ごとの予測
```

`predictions.csv` には画像ごとに、画像パス・被験者/運転者 ID・正解クラス（番号と名称）・
予測クラス（番号と名称）・Top-1 スコア・Top-3 予測クラスが含まれます。

`run_ablation.py` は PE/DAD/TEO の組み合わせごとに 1 行の CSV を出力します。

---

## ライセンス

実装コードは **Apache License 2.0** で公開します（[LICENSE](LICENSE) を参照）。

**免責事項.** Apache-2.0 ライセンスは本リポジトリのソースコードにのみ適用されます。
次のものには適用されず、いかなる権利も付与しません。

- SAM-DD・StateFarm をはじめとするデータセット — 公式の配布元から、各自のライセンス/
  規約のもとで入手してください。
- 学習済み CLIP の重み — OpenAI CLIP のライセンス/規約に従います。
- サードパーティライブラリ — それぞれのライセンスに従います。


## 引用

```bibtex
@inproceedings{miyata2026_ZVL-DDD,
  author    = {Miyata, Takamichi and Miyata, Sumiko and Morris, Andrew},
  title     = {Zero-Shot Distracted Driver Detection via Vision Language Models with Double Decoupling},
  booktitle = {Proceedings of the IEEE International Symposium on Communication Systems, Networks and Digital Signal Processing (CSNDSP)},
  year      = {2026}
}
```
