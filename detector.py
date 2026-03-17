from __future__ import annotations

from typing import List, Optional

import numpy as np

from .cropper import FaceBox


class Yoro26FaceDetector:
    """YOLO系モデルで顔検出を行うラッパー。"""

    def __init__(
        self,
        model_path: str = "yoro26-face.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        device: Optional[str] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "ultralytics が見つかりません。`pip install ultralytics opencv-python pillow` を実行してください。"
            ) from e

        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device

    def detect(self, image_bgr: np.ndarray) -> List[FaceBox]:
        results = self.model.predict(
            source=image_bgr,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )
        if not results:
            return []

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        xyxy = r.boxes.xyxy.detach().cpu().numpy()
        confs = r.boxes.conf.detach().cpu().numpy() if r.boxes.conf is not None else np.ones((len(xyxy),), dtype=float)

        boxes: List[FaceBox] = []
        for i, b in enumerate(xyxy):
            x1, y1, x2, y2 = map(float, b.tolist())
            boxes.append(FaceBox(x1=x1, y1=y1, x2=x2, y2=y2, conf=float(confs[i])))
        return boxes
