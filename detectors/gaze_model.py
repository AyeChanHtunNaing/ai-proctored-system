import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense

# List of possible labels for the gaze classification
LABELS = ["focus", "left", "right", "updown"]


class DenseCompat(Dense):
    @classmethod
    def from_config(cls, config):
        # Some model files include Keras 3-only config keys that older Dense doesn't accept.
        config = dict(config)
        config.pop("quantization_config", None)
        return super().from_config(config)


def _load_model_compat(model_path: str):
    try:
        return load_model(model_path)
    except TypeError as e:
        msg = str(e)
        if "quantization_config" not in msg:
            raise
        return load_model(model_path, custom_objects={"Dense": DenseCompat})


class GazeClassifier:
    def __init__(self, model_path: str):
        # Load the model when the class is initialized
        self.model = _load_model_compat(model_path)

    def _preprocess(self, frame_bgr, face_box_xyxy=None, person_box_xyxy=None):
        """
        Preprocess the input frame by cropping the region of interest (ROI) based on the
        face_box_xyxy or person_box_xyxy coordinates and resizing it to 64x64 grayscale.
        """
        if face_box_xyxy is not None:
            x1, y1, x2, y2 = map(int, face_box_xyxy)
            roi = frame_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        elif person_box_xyxy is not None:
            x1, y1, x2, y2 = map(int, person_box_xyxy)
            y2_roi = y1 + int((y2 - y1) * 0.55)
            roi = frame_bgr[max(0, y1):max(0, y2_roi), max(0, x1):max(0, x2)]
        else:
            roi = frame_bgr

        if roi is None or roi.size == 0:
            return None

        # Convert to grayscale and resize to 64x64
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
        img = img[..., None]  # Add a channel dimension (e.g., grayscale image)
        img = np.expand_dims(img, 0)  # Add batch dimension
        return img

    def predict(self, frame_bgr, face_box_xyxy=None, person_box_xyxy=None):
        """
        Predict the gaze direction based on the input frame and optional face or person bounding boxes.
        Returns the predicted label and the associated probability.
        """
        x = self._preprocess(frame_bgr, face_box_xyxy=face_box_xyxy, person_box_xyxy=person_box_xyxy)
        if x is None:
            return None, 0.0

        # Make a prediction
        probs = self.model.predict(x, verbose=False)[0]
        idx = int(np.argmax(probs))  # Get the index of the highest probability
        return LABELS[idx], float(probs[idx])  # Return the label and probability
