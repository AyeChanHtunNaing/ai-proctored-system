from ultralytics import YOLO

class YoloPersonDetector:
    def __init__(self, weights: str = "weights/yolov8n.pt", conf: float = 0.35):
        self.model = YOLO(weights)
        self.conf = conf

    def count_persons(self, frame_bgr):
        """Return (count, boxes) where boxes are xyxy for class person."""
        results = self.model.predict(frame_bgr, conf=self.conf, verbose=False)
        r = results[0]
        count = 0
        boxes = []
        if r.boxes is None:
            return 0, []
        for b in r.boxes:
            cls = int(b.cls[0])
            # COCO class 0 == person
            if cls == 0:
                count += 1
                xyxy = b.xyxy[0].tolist()
                boxes.append(xyxy)
        # sort left-to-right for determinism
        boxes.sort(key=lambda x: x[0])
        return count, boxes
