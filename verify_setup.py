#!/usr/bin/env python
import sys
from pathlib import Path

def test_imports():
    print("=== Testing Imports ===")
    imports = [
        ("streamlit", "streamlit"),
        ("cv2 (OpenCV)", "cv2"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("Pillow (PIL)", "PIL"),
        ("scikit-learn", "sklearn"),
        ("tensorflow", "tensorflow"),
        ("streamlit-webrtc", "streamlit_webrtc"),
        ("av (PyAV)", "av"),
        ("aioice", "aioice"),
        ("ultralytics", "ultralytics"),
    ]
    
    all_ok = True
    for label, pkg in imports:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "unknown version")
            print(f"  [OK] {label} (version: {version})")
        except ImportError as e:
            print(f"  [FAIL] {label}: {e}")
            all_ok = False
            
    return all_ok

def test_models():
    print("\n=== Testing Model Loading ===")
    try:
        from detectors.face_detect import HaarFaceDetector
        from detectors.yolo_person import YoloPersonDetector
        from detectors.gaze_model import GazeClassifier
        print("  [OK] Successfully imported detector wrapper classes.")
    except Exception as e:
        print(f"  [FAIL] Detector imports failed: {e}")
        return False

    all_ok = True
    
    # Check Haar Face Detector init
    try:
        face_det = HaarFaceDetector()
        print("  [OK] HaarFaceDetector initialized successfully.")
    except Exception as e:
        print(f"  [FAIL] HaarFaceDetector init failed: {e}")
        all_ok = False
        
    # Check YOLO Detector and weights
    yolo_path = Path("models/yolov8n.pt")
    if not yolo_path.exists():
        print(f"  [FAIL] YOLOv8 weights missing at {yolo_path}")
        all_ok = False
    else:
        try:
            yolo = YoloPersonDetector(weights=str(yolo_path))
            print("  [OK] YoloPersonDetector loaded weights successfully.")
        except Exception as e:
            print(f"  [FAIL] YoloPersonDetector failed to load weights: {e}")
            all_ok = False
            
    # Check Gaze Classifier and weights
    gaze_path = Path("models/gaze_cnn.h5")
    if not gaze_path.exists():
        print(f"  [FAIL] Gaze CNN weights missing at {gaze_path}")
        all_ok = False
    else:
        try:
            gaze = GazeClassifier(str(gaze_path))
            print("  [OK] GazeClassifier loaded model successfully.")
        except Exception as e:
            print(f"  [FAIL] GazeClassifier failed to load model: {e}")
            all_ok = False
            
    return all_ok

def main():
    print("Checking python version...")
    print(f"Python: {sys.version}\n")
    
    imports_ok = test_imports()
    models_ok = test_models()
    
    print("\n=== Summary ===")
    if imports_ok and models_ok:
        print("SUCCESS: All packages imported and models loaded perfectly!")
        sys.exit(0)
    else:
        print("FAILURE: Some imports or model loads failed. Please inspect logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
