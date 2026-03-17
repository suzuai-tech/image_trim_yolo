# facetrim_yoro

顔検出（YOLO系）を使って、画像を指定アスペクト比でトリムするツールです。  
単一画像・ディレクトリ再帰処理の両方に対応しています。

## セットアップ

1. 仮想環境を有効化
2. 依存関係をインストール

```bash
pip install -r requirements.txt
```

## 基本的な使い方

```bash
python -m facetrim_yoro.cli 入力 出力 --model モデルファイル
```

- 入力: 画像ファイル または ディレクトリ
- 出力: 画像ファイル または ディレクトリ

### 例（単一画像）

```bash
python -m facetrim_yoro.cli input.jpg output.jpg --model yoro.pt
```

### 例（ディレクトリ）

```bash
python -m facetrim_yoro.cli downloads/trimmed/raw downloads/trimmed/out --model yoro.pt
```

## アスペクト比指定

`--ratio` でトリム比率を指定できます（デフォルト: `1:1`）。

```bash
python -m facetrim_yoro.cli input.jpg output.jpg --model yoro.pt --ratio 16:9
```

指定例:

- `1:1`（正方形）
- `4:3`
- `16:9`
- `9:16`

## 主なオプション

- `--model`: 顔検出モデルのパス（例: `yoro.pt`）
- `--conf`: 検出しきい値（デフォルト `0.25`）
- `--iou`: NMS IoU（デフォルト `0.45`）
- `--device`: 推論デバイス（例: `cpu`, `0`）
- `--ratio`: 出力アスペクト比（例: `16:9`）

## 補足

- 入力がディレクトリの場合、画像拡張子（jpg / jpeg / png / bmp / webp / tif / tiff）を再帰的に処理します。
- 顔が検出されない画像は中央基準でトリムします。
