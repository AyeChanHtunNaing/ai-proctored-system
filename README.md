# AI-Powered Automated Exam Proctoring System

Real-time local exam monitoring using:

- YOLOv8 for person detection
- OpenCV Haar Cascade for face detection
- CNN for gaze classification
- Streamlit dashboard
- Rule-based event logic
- CSV violation logging

## Project Overview

This app monitors webcam input and flags potentially suspicious behavior during an online exam session, with on-device processing and configurable timing thresholds.

### Detected behaviors

1. Candidate absence
2. Multiple people in frame
3. Prolonged looking away

### Privacy-focused design

- Local video processing only
- Grace-period timers to reduce false positives
- Lightweight violation logging (`csv`)

## System Flow

```text
Webcam
  -> YOLOv8 Person Detection
  -> Haar Face Detection
  -> CNN Gaze Classification
  -> Rule Engine
  -> Streamlit UI + CSV Logs
```

## Project Structure

```text
ai_proctor/
├── app.py
├── logic.py
├── requirements.txt
├── README.md
├── detectors/
│   ├── yolo_person.py
│   ├── face_detect.py
│   └── gaze_model.py
├── scripts/
│   └── train_gaze_cnn.py
├── models/
│   ├── gaze_cnn.h5
│   └── yolov8n.pt
├── dataset/
│   ├── focus/
│   ├── left/
│   ├── right/
│   └── updown/
└── logs/
    └── violations.csv
```

## Installation

### 1. Create virtual environment

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

Open: `http://localhost:8501`

## Train the Gaze CNN (Optional)

Use this if you want to retrain `models/gaze_cnn.h5` from your dataset folders:

```bash
python scripts/train_gaze_cnn.py
```

## Rule Logic Defaults

| Behavior         | Warning | Violation |
|------------------|---------|-----------|
| Absence          | 5s      | 20s       |
| Looking away     | 10s     | 30s       |
| Multiple persons | Immediate | Immediate |

Thresholds are adjustable from the Streamlit sidebar.

## Logs

Violations are appended to:

- `logs/violations.csv`

Columns:

- `timestamp`
- `status`
- `reason`
- `seconds`

## Notes

- If `models/gaze_cnn.h5` is missing, the app runs but warns that gaze CNN is unavailable.
- The gaze loader includes compatibility handling for model files containing `quantization_config` in Dense layer config.

## Ethical Considerations

- Local inference (no cloud video upload)
- Explicit monitoring notice in UI
- Timer-based warnings before violations
- Minimal retention through event-only logs

## Authors

- Nila Pyone
- Aye Chan Htun Naing
