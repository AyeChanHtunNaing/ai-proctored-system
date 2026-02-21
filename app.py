import time, csv
from pathlib import Path

import streamlit as st
import cv2

from logic import Thresholds, ProctorState
from detectors.yolo_person import YoloPersonDetector
from detectors.face_detect import HaarFaceDetector
from detectors.gaze_model import GazeClassifier

st.set_page_config(page_title="AI Exam Proctor (Columbia + CNN)", layout="wide")

st.title("AI-Powered Exam Proctoring (YOLO + Haar + CNN)")
st.caption("Webcam → YOLO (person count) → Haar (face ROI) → CNN (gaze) → Rule Engine → Logs")

st.sidebar.header("Thresholds (seconds)")
absence_warn = st.sidebar.number_input("Absence warning", 1.0, 60.0, 5.0, 1.0)
absence_violate = st.sidebar.number_input("Absence violation", 2.0, 180.0, 20.0, 1.0)
away_warn = st.sidebar.number_input("Looking-away warning", 1.0, 60.0, 10.0, 1.0)
away_violate = st.sidebar.number_input("Looking-away violation", 2.0, 240.0, 30.0, 1.0)
yolo_conf = st.sidebar.slider("YOLO confidence", 0.10, 0.90, 0.35, 0.05)

st.sidebar.markdown("---")
st.sidebar.info("Ethics: Local processing only. Show 'AI Monitoring Active'.")

@st.cache_resource
def load_yolo(conf):
    return YoloPersonDetector(weights="models/yolov8n.pt", conf=conf)

yolo = load_yolo(yolo_conf)
face_det = HaarFaceDetector()
state = ProctorState(Thresholds(absence_warn, absence_violate, away_warn, away_violate))

model_path = Path("models/gaze_cnn.h5")
gaze = GazeClassifier(str(model_path)) if model_path.exists() else None

log_path = Path("logs/violations.csv")
log_path.parent.mkdir(parents=True, exist_ok=True)
if not log_path.exists():
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["timestamp","status","reason","seconds"])

def append_log(status, reason, seconds):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([ts, status, reason, f"{seconds:.2f}"])

def recent_logs(n=8):
    lines = log_path.read_text().splitlines()
    return lines[1:][-n:][::-1] if len(lines) > 1 else []

colA, colB = st.columns([2, 1])
frame_box = colA.empty()
status_box = colB.empty()
logs_box = colB.empty()

st.markdown("### 🔒 AI Monitoring Active (Local Processing Only)")

start = st.toggle("Start monitoring", value=False)

if start:
    if gaze is None:
        st.warning("CNN model missing: models/gaze_cnn.h5. Train it after preprocessing Columbia.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam. Close other apps using the camera and allow permissions.")
        st.stop()

    last_reason = None
    last_status = "OK"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        person_count, boxes = yolo.count_persons(frame)

        is_focused = True
        gaze_label, gaze_conf = None, 0.0
        face_box = None

        if person_count == 1 and boxes and gaze is not None:
            person_box = boxes[0]
            face_box = face_det.detect_one(frame, within_box_xyxy=person_box)

            x1,y1,x2,y2 = map(int, person_box)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 2)

            if face_box is not None:
                fx1,fy1,fx2,fy2 = map(int, face_box)
                cv2.rectangle(frame, (fx1,fy1), (fx2,fy2), (0,255,0), 2)

            gaze_label, gaze_conf = gaze.predict(frame, face_box_xyxy=face_box, person_box_xyxy=person_box)
            is_focused = (gaze_label == "focus") if gaze_label is not None else False

        status, reason, seconds = state.update(person_count, is_focused)

        cv2.putText(frame, f"Persons: {person_count}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)
        if gaze_label is not None:
            cv2.putText(frame, f"Gaze: {gaze_label} ({gaze_conf:.2f})", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
        cv2.putText(frame, f"{status} | {reason} | {seconds:.1f}s", (10, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_box.image(frame_rgb, channels="RGB")

        if status == "OK":
            status_box.success("✅ OK")
        elif status == "WARNING":
            status_box.warning(f"🟠 WARNING: {reason} ({seconds:.1f}s)")
        else:
            status_box.error(f"🔴 VIOLATION: {reason} ({seconds:.1f}s)")

        if status in ("WARNING","VIOLATION") and (status != last_status or reason != last_reason):
            append_log(status, reason, seconds)
            last_status, last_reason = status, reason
        if status == "OK":
            last_status, last_reason = "OK", None

        logs_box.markdown("### Recent Logs")
        logs_box.code("\n".join(recent_logs()) if recent_logs() else "No logs yet")

        # Stop if toggle turned off
        if not st.session_state.get("Start monitoring", True):
            break
        time.sleep(0.03)

    cap.release()
else:
    st.info("Toggle **Start monitoring** to begin.")
