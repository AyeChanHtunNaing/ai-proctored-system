import csv
import time
from pathlib import Path
from threading import Lock

import streamlit as st

st.set_page_config(page_title="AI Exam Proctor (Columbia + CNN)", layout="wide")

try:
    import av
    import cv2
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
except Exception as e:
    st.title("AI-Powered Exam Proctoring (YOLO + Haar + CNN)")
    st.error("Dependency import failed. Please check Streamlit Cloud build logs.")
    st.code(str(e))
    st.info(
        "Try these pins in requirements.txt: opencv-python-headless==4.8.1.78, "
        "tensorflow-cpu==2.17.0 and Python 3.10 via runtime.txt/.python-version."
    )
    st.stop()

from detectors.face_detect import HaarFaceDetector
from detectors.gaze_model import GazeClassifier
from detectors.yolo_person import YoloPersonDetector
from logic import ProctorState, Thresholds

st.title("AI-Powered Exam Proctoring (YOLO + Haar + CNN)")
st.caption("Webcam -> YOLO (person count) -> Haar (face ROI) -> CNN (gaze) -> Rule Engine -> Logs")

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
        csv.writer(f).writerow(["timestamp", "status", "reason", "seconds"])


def append_log(status, reason, seconds):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([ts, status, reason, f"{seconds:.2f}"])


def recent_logs(n=8):
    lines = log_path.read_text().splitlines()
    return lines[1:][-n:][::-1] if len(lines) > 1 else []


st.markdown("### AI Monitoring Active (Local Processing Only)")
if gaze is None:
    st.warning("CNN model missing: models/gaze_cnn.h5. Train it after preprocessing Columbia.")
st.info(
    "If webcam does not start on Streamlit Cloud: allow browser camera permission, "
    "then refresh once and click START again."
)

colA, colB = st.columns([2, 1])
status_box = colB.empty()
logs_box = colB.empty()

runtime = {
    "lock": Lock(),
    "status": "OK",
    "reason": "Waiting for camera",
    "seconds": 0.0,
    "gaze_label": None,
    "gaze_conf": 0.0,
    "person_count": 0,
    "last_reason": None,
    "last_status": "OK",
    "frame_idx": 0,
}


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    try:
        with runtime["lock"]:
            runtime["frame_idx"] += 1
            frame_idx = runtime["frame_idx"]
            prev_status = runtime["status"]
            prev_reason = runtime["reason"]
            prev_seconds = runtime["seconds"]

        # Keep webcam responsive on Streamlit Cloud by processing every 3rd frame.
        if frame_idx % 3 != 0:
            cv2.putText(
                img,
                f"{prev_status} | {prev_reason} | {prev_seconds:.1f}s",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        person_count, boxes = yolo.count_persons(img)

        is_focused = True
        gaze_label, gaze_conf = None, 0.0
        face_box = None

        if person_count == 1 and boxes:
            person_box = boxes[0]
            face_box = face_det.detect_one(img, within_box_xyxy=person_box)

            x1, y1, x2, y2 = map(int, person_box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

            if face_box is not None:
                fx1, fy1, fx2, fy2 = map(int, face_box)
                cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)

            if gaze is not None:
                gaze_label, gaze_conf = gaze.predict(img, face_box_xyxy=face_box, person_box_xyxy=person_box)
                is_focused = (gaze_label == "focus") if gaze_label is not None else False

        status, reason, seconds = state.update(person_count, is_focused)

        cv2.putText(img, f"Persons: {person_count}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        if gaze_label is not None:
            cv2.putText(img, f"Gaze: {gaze_label} ({gaze_conf:.2f})", (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(
            img,
            f"{status} | {reason} | {seconds:.1f}s",
            (10, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )

        with runtime["lock"]:
            runtime["status"] = status
            runtime["reason"] = reason
            runtime["seconds"] = seconds
            runtime["gaze_label"] = gaze_label
            runtime["gaze_conf"] = gaze_conf
            runtime["person_count"] = person_count

            if status in ("WARNING", "VIOLATION") and (
                status != runtime["last_status"] or reason != runtime["last_reason"]
            ):
                append_log(status, reason, seconds)
                runtime["last_status"] = status
                runtime["last_reason"] = reason
            if status == "OK":
                runtime["last_status"] = "OK"
                runtime["last_reason"] = None

        return av.VideoFrame.from_ndarray(img, format="bgr24")
    except Exception as e:
        cv2.putText(img, "Frame processing error", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(img, str(e)[:90], (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        with runtime["lock"]:
            runtime["status"] = "WARNING"
            runtime["reason"] = f"Frame error: {type(e).__name__}"
            runtime["seconds"] = 0.0
        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="ai-proctor-monitor",
    mode=WebRtcMode.SENDRECV,
    desired_playing_state=True,
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    },
    media_stream_constraints={
        "video": {
            "facingMode": "user",
            "width": {"ideal": 640},
            "height": {"ideal": 480},
        },
        "audio": False,
    },
    video_frame_callback=video_frame_callback,
    async_processing=True,
)

if webrtc_ctx.state.playing:
    while webrtc_ctx.state.playing:
        with runtime["lock"]:
            status = runtime["status"]
            reason = runtime["reason"]
            seconds = runtime["seconds"]

        if status == "OK":
            status_box.success("OK")
        elif status == "WARNING":
            status_box.warning(f"WARNING: {reason} ({seconds:.1f}s)")
        else:
            status_box.error(f"VIOLATION: {reason} ({seconds:.1f}s)")

        logs_box.markdown("### Recent Logs")
        logs_box.code("\n".join(recent_logs()) if recent_logs() else "No logs yet")
        time.sleep(0.5)
else:
    status_box.info("Waiting for webcam permission/start.")
    logs_box.markdown("### Recent Logs")
    logs_box.code("\n".join(recent_logs()) if recent_logs() else "No logs yet")
