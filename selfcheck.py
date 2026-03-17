from __future__ import annotations

from .cropper import FaceBox, choose_square_crop


def run_selfcheck() -> None:
    # 横長画像: 左右2人 -> 可能な限り両方を含む中央付近が選ばれる
    boxes = [
        FaceBox(60, 40, 220, 260, 0.95),
        FaceBox(360, 50, 520, 280, 0.90),
    ]
    left, top, size = choose_square_crop(640, 360, boxes)
    assert size == 360
    assert top == 0
    assert 40 <= left <= 120

    # 縦長画像: 下寄りの顔を優先
    boxes2 = [FaceBox(80, 420, 220, 620, 0.9)]
    left2, top2, size2 = choose_square_crop(300, 700, boxes2)
    assert size2 == 300
    assert left2 == 0
    assert 320 <= top2 <= 400

    # 顔なし -> 中央トリム
    left3, top3, size3 = choose_square_crop(1000, 600, [])
    assert size3 == 600
    assert left3 == 200
    assert top3 == 0

    print("selfcheck passed")


if __name__ == "__main__":
    run_selfcheck()
