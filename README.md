# Tennis Analysis

A computer vision pipeline that processes broadcast tennis footage and outputs an annotated video with player bounding boxes, ball trajectory, court keypoints, and a scaled mini-court overlay showing real-world player and ball positions frame by frame.

---

## Architecture

```
input-videos/input_video.mp4
        │
        ▼  PlayerTracker (YOLOv8x)
        │   detect_frames → choose_and_filter_players (proximity to court keypoints)
        │
        ▼  BallTracker (custom YOLO, models/last.pt)
        │   detect_frames → interpolate_ball_positions (fills missed detections)
        │
        ▼  CourtLineDetector (ResNet50, models/keypoints_model.pth)
        │   predict → 14 court keypoints from first frame
        │
        ▼  MiniCourt
        │   convert_bounding_boxes_to_mini_court_coords
        │   (maps pixel positions → scaled court coordinates)
        │
        ▼  draw_bboxes + draw_keypoints + draw_mini_court + draw_points
        │
        ▼  output-videos/output_video.avi
```

Detection stubs (`tracker_stubs/*.pkl`) cache per-frame detections so downstream steps can be exercised without running YOLO again. Only load stubs you created or otherwise trust: Python pickle files can execute code when loaded.

---

## Core Technical Stack

Python, PyTorch, Ultralytics YOLOv8, torchvision (ResNet50), OpenCV, pandas

---

## Key Methodologies

- **Two-model detection stack** — a pretrained YOLOv8x handles player detection (high recall on human figures); a separately trained custom YOLO model (`models/last.pt`) handles ball detection, where the ball is small, fast, and frequently occluded.

- **Player filtering by court proximity** — all persons detected by YOLO are filtered down to the two nearest to the court keypoints. This removes ball-boys, line judges, and crowd false positives without requiring a separate classification head.

- **Ball interpolation** — the ball tracker frequently misses frames due to motion blur and occlusion. Linear interpolation across the sequence of detected positions reconstructs a continuous trajectory, which is necessary for downstream shot-detection logic.

- **ResNet50 court keypoint detector** — trained on the [TennisCourtDetector dataset](https://github.com/yastrebksv/TennisCourtDetector) to regress 14 2D keypoints (court lines and corners) from a single frame. Keypoints anchor all spatial calculations, including the player-proximity filter and mini-court coordinate mapping.

- **Mini-court projection** — pixel-space bounding box centroids are mapped to a scaled diagram of a standard tennis court using the detected keypoints as reference anchors, producing a top-down view of ball and player movement.

---

## Demo Artifacts & Validation

- The repository includes a sample input video, a rendered output video, and cached detection stubs; these are existing demo artifacts, not evidence of a fresh end-to-end run.
- `models/last.pt`, `models/keypoints_model.pth`, `yolov8x.pt`, and `models/best.pt` are tracked, present in the checkout, and stored as binary weight files rather than Git LFS pointer text. The training directory also contains tracked `yolov5l6u.pt` and `yolov8n.pt`.
- No benchmark or performance claim is made here.

---

## Local Replication

Prerequisites: Python 3.8+ and the dependencies in `requirements.txt`; CUDA is recommended for faster YOLOv8x inference.

```bash
pip install -r requirements.txt
python main.py
```

`main.py` expects `input-videos/input_video.mp4` and uses the model weights and cached detection stubs at their repository paths. It writes `output-videos/output_video.avi`. The checked-in video, output, and cache are demo artifacts; do not overwrite or remove them if you want to preserve the supplied example. Back up the sample input before replacing it. Only use trusted detection stub pickle files.

The training notebooks are in `training/`. YOLOv8x weights are also tracked at `yolov8x.pt`; Ultralytics may download weights if it does not resolve the local file.

---

## Project Structure

```
├── main.py
├── trackers/
│   ├── player_tracker.py
│   └── ball_tracker.py
├── court_line_detector/
├── mini_court/
├── utils/
├── models/
│   ├── last.pt                  # Ball detector (custom YOLO)
│   └── keypoints_model.pth      # Court keypoint detector (ResNet50)
├── training/                    # Training notebooks
├── tracker_stubs/               # Cached detection results
├── input-videos/
└── output-videos/
```
