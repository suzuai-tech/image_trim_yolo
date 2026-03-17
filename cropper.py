from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class FaceBox:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def _intersection_area(a: Tuple[float, float, float, float], b: FaceBox) -> float:
    ax1, ay1, ax2, ay2 = a
    ix1 = max(ax1, b.x1)
    iy1 = max(ay1, b.y1)
    ix2 = min(ax2, b.x2)
    iy2 = min(ay2, b.y2)
    w = max(0.0, ix2 - ix1)
    h = max(0.0, iy2 - iy1)
    return w * h





def choose_crop(
    img_w: int, img_h: int, boxes: List[FaceBox], aspect_ratio: Tuple[int, int] = (1, 1)
) -> Tuple[int, int, int, int]:
    """
    指定されたアスペクト比でトリムする座標を返す。

    aspect_ratio: (width_ratio, height_ratio) tuple. 例: (16, 9), (4, 3), (1, 1)
    返り値: (left, top, width, height)
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError("Image size must be positive.")

    aspect_w, aspect_h = aspect_ratio
    if aspect_w <= 0 or aspect_h <= 0:
        raise ValueError("Aspect ratio must be positive.")

    # 目標アスペクト比
    target_ratio = aspect_w / aspect_h

    # 画像のアスペクト比
    img_ratio = img_w / img_h

    # 画像に合わせて幅または高さを決定
    if img_ratio > target_ratio:
        # 画像が横長 -> 高さに合わせて幅を制限
        crop_h = img_h
        crop_w = int(crop_h * target_ratio)
    else:
        # 画像が縦長または同じ -> 幅に合わせて高さを制限
        crop_w = img_w
        crop_h = int(crop_w / target_ratio)

    # クロップサイズ
    max_start_x = float(img_w - crop_w)
    max_start_y = float(img_h - crop_h)

    # 顔がない場合は中央トリム
    if not boxes:
        left = int(max_start_x / 2)
        top = int(max_start_y / 2)
        return left, top, crop_w, crop_h

    # 候補座標を生成
    candidates_x = {0.0, max_start_x}
    candidates_y = {0.0, max_start_y}

    for b in boxes:
        candidates_x.add(_clip(b.x1, 0.0, max_start_x))
        candidates_x.add(_clip(b.x2 - crop_w, 0.0, max_start_x))
        candidates_x.add(_clip(b.cx - crop_w / 2.0, 0.0, max_start_x))

        candidates_y.add(_clip(b.y1, 0.0, max_start_y))
        candidates_y.add(_clip(b.y2 - crop_h, 0.0, max_start_y))
        candidates_y.add(_clip(b.cy - crop_h / 2.0, 0.0, max_start_y))

    # 加重中央
    weighted_cx = sum(b.cx * max(0.05, b.conf) for b in boxes) / sum(max(0.05, b.conf) for b in boxes)
    weighted_cy = sum(b.cy * max(0.05, b.conf) for b in boxes) / sum(max(0.05, b.conf) for b in boxes)
    center_x = _clip(weighted_cx - crop_w / 2.0, 0.0, max_start_x)
    center_y = _clip(weighted_cy - crop_h / 2.0, 0.0, max_start_y)

    candidates_x.add(center_x)
    candidates_y.add(center_y)

    best_left = 0.0
    best_top = 0.0
    best_score = -1.0
    best_dist = float("inf")

    for left in sorted(candidates_x):
        for top in sorted(candidates_y):
            score = _score_crop(left, top, left + crop_w, top + crop_h, boxes)
            dist = ((left - center_x) ** 2 + (top - center_y) ** 2) ** 0.5
            if score > best_score or (score == best_score and dist < best_dist):
                best_score = score
                best_left = left
                best_top = top
                best_dist = dist

    return int(round(best_left)), int(round(best_top)), crop_w, crop_h


def _score_crop(
    left: float,
    top: float,
    right: float,
    bottom: float,
    boxes: Iterable[FaceBox],
) -> float:
    """矩形領域内の顔スコアを計算。"""
    rect = (left, top, right, bottom)
    score = 0.0
    for b in boxes:
        if b.area <= 0:
            continue
        conf_w = max(0.05, float(b.conf))
        score += _intersection_area(rect, b) * conf_w
    return score


def choose_square_crop(img_w: int, img_h: int, boxes: List[FaceBox]) -> Tuple[int, int, int]:
    """
    画像短辺サイズを維持して 1:1 にトリムする座標を返す。

    返り値: (left, top, size)
    """
    left, top, crop_w, crop_h = choose_crop(img_w, img_h, boxes, (1, 1))
    return left, top, crop_w
