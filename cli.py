from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from PIL import Image

from .cropper import FaceBox, choose_crop
from .detector import Yoro26FaceDetector


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _parse_aspect_ratio(ratio_str: str) -> Tuple[int, int]:
    """
    比率文字列をパース。例: "16:9", "4:3", "1:1"
    """
    try:
        parts = ratio_str.split(":")
        if len(parts) != 2:
            raise ValueError
        w, h = int(parts[0]), int(parts[1])
        if w <= 0 or h <= 0:
            raise ValueError
        return w, h
    except (ValueError, IndexError):
        raise ValueError(f"Invalid aspect ratio format: {ratio_str}. Use format like '16:9', '4:3', '1:1'")


def _iter_images(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for p in sorted(path.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            yield p


def _load_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"画像を読み込めませんでした: {path}")
    return img


def _crop_one(detector: Yoro26FaceDetector, in_path: Path, out_path: Path, aspect_ratio: Tuple[int, int] = (1, 1)) -> None:
    bgr = _load_bgr(in_path)
    h, w = bgr.shape[:2]
    boxes: List[FaceBox] = detector.detect(bgr)

    left, top, crop_w, crop_h = choose_crop(w, h, boxes, aspect_ratio)

    with Image.open(in_path) as pil_img:
        pil_img = pil_img.convert("RGB")
        cropped = pil_img.crop((left, top, left + crop_w, top + crop_h))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(out_path)

    print(f"[OK] {in_path} -> {out_path} faces={len(boxes)} crop=({left},{top},{crop_w}x{crop_h}) ratio={aspect_ratio[0]}:{aspect_ratio[1]}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="yoro26 顔検出ベースのトリムツール（様々なアスペクト比に対応）")
    p.add_argument("input", type=Path, help="入力画像ファイル or ディレクトリ")
    p.add_argument("output", type=Path, help="出力画像ファイル or ディレクトリ")
    p.add_argument("--model", default="yoro26-face.pt", help="yoro26 顔モデルのパス")
    p.add_argument("--conf", type=float, default=0.25, help="検出 confidence しきい値")
    p.add_argument("--iou", type=float, default=0.45, help="NMS IoU")
    p.add_argument("--device", default=None, help="推論デバイス (例: cpu, 0)")
    p.add_argument("--ratio", type=str, default="1:1", help="アスペクト比 (例: 16:9, 4:3, 1:1, 9:16)")
    return p


def main() -> None:
    args = build_parser().parse_args()

    try:
        aspect_ratio = _parse_aspect_ratio(args.ratio)
    except ValueError as e:
        print(f"Error: {e}")
        return

    detector = Yoro26FaceDetector(
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )

    if args.input.is_file():
        out = args.output
        if out.suffix.lower() not in IMAGE_EXTS:
            out = out / args.input.name
        _crop_one(detector, args.input, out, aspect_ratio)
        return

    if not args.input.exists():
        raise FileNotFoundError(f"入力パスが存在しません: {args.input}")

    for in_img in _iter_images(args.input):
        rel = in_img.relative_to(args.input)
        out_img = args.output / rel
        _crop_one(detector, in_img, out_img, aspect_ratio)


if __name__ == "__main__":
    main()
