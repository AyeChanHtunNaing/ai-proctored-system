import cv2
import numpy as np

class HaarFaceDetector:
    """Fast offline face detector using OpenCV Haar cascades."""

    def __init__(self, scaleFactor: float = 1.1, minNeighbors: int = 7, minSize=(60, 60)):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.minSize = minSize

    def _nms(self, boxes, threshold=0.35):
        """Apply Non-Maximum Suppression to eliminate overlapping bounding boxes."""
        if len(boxes) == 0:
            return []
        
        x1 = np.array([b[0] for b in boxes], dtype=float)
        y1 = np.array([b[1] for b in boxes], dtype=float)
        x2 = np.array([b[2] for b in boxes], dtype=float)
        y2 = np.array([b[3] for b in boxes], dtype=float)
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(areas)
        
        pick = []
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            
            inter = w * h
            overlap = inter / (areas[i] + areas[idxs[:last]] - inter + 1e-6)
            
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))
            
        return [boxes[i] for i in pick]

    def detect_one(self, frame_bgr, within_box_xyxy=None):
        """Return best face box (x1,y1,x2,y2) or None.
        If within_box_xyxy provided, detect inside that ROI.
        """
        if within_box_xyxy is not None:
            x1, y1, x2, y2 = map(int, within_box_xyxy)
            roi = frame_bgr[max(0,y1):max(0,y2), max(0,x1):max(0,x2)]
            offset = (x1, y1)
        else:
            roi = frame_bgr
            offset = (0, 0)

        if roi is None or roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=self.minSize
        )
        if faces is None or len(faces) == 0:
            return None
        
        # Convert to absolute xyxy boxes
        ox, oy = offset
        boxes = []
        for x, y, w, h in faces:
            boxes.append((ox + int(x), oy + int(y), ox + int(x + w), oy + int(y + h)))
            
        # Apply NMS to suppress overlapping duplicates
        refined_boxes = self._nms(boxes, threshold=0.35)
        if len(refined_boxes) == 0:
            return None
            
        # Return the largest box by area
        return max(refined_boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))

    def detect_all(self, frame_bgr, within_box_xyxy=None):
        """Return all face boxes [(x1,y1,x2,y2), ...]."""
        if within_box_xyxy is not None:
            x1, y1, x2, y2 = map(int, within_box_xyxy)
            roi = frame_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            offset = (x1, y1)
        else:
            roi = frame_bgr
            offset = (0, 0)

        if roi is None or roi.size == 0:
            return []

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=self.minSize,
        )
        if faces is None or len(faces) == 0:
            return []

        ox, oy = offset
        boxes = []
        for x, y, w, h in faces:
            boxes.append((ox + int(x), oy + int(y), ox + int(x + w), oy + int(y + h)))
            
        # Apply NMS to suppress duplicate detections of the same face
        nms_boxes = self._nms(boxes, threshold=0.35)
        if len(nms_boxes) <= 1:
            return nms_boxes

        # Discard background noise face candidates that are too small
        # compared to the largest detected face box in the frame
        areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in nms_boxes]
        max_area = max(areas)
        
        filtered_boxes = []
        for b, area in zip(nms_boxes, areas):
            if area >= 0.50 * max_area:
                filtered_boxes.append(b)
                
        return filtered_boxes
