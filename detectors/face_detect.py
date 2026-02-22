import cv2

class HaarFaceDetector:
    """Fast offline face detector using OpenCV Haar cascades."""

    def __init__(self, scaleFactor: float = 1.1, minNeighbors: int = 5, minSize=(60, 60)):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.minSize = minSize

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
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        ox, oy = offset
        return (ox + int(x), oy + int(y), ox + int(x + w), oy + int(y + h))

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
        out = []
        for x, y, w, h in faces:
            out.append((ox + int(x), oy + int(y), ox + int(x + w), oy + int(y + h)))
        return out
