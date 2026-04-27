import os
import glob
import json
import math
import cv2
import uuid
from datetime import datetime
from ultralytics import YOLO

# =========================================================
# CONFIGURATION
# =========================================================
FRAMES_DIR = "data_samples/images"
OUTPUT_JSON_DIR = "results/json"
ANNOTATED_DIR = "results/annotated_frames"
MODEL_PATH = "weights/best.pt"
DEVICE = "cuda:1"

# --- DETECTION THRESHOLDS & GLARE/GHOSTING FIXES ---
CONF_THRESHOLD = 0.20
YOLO_NMS_IOU = 0.50

USE_TOP_ROI = True
TOP_ROI_RATIO = 0.68
UPSCALE_ROI = True
UPSCALE_FACTOR = 2.5

# detection filter
MIN_BOX_W = 3
MIN_BOX_H = 4
MAX_BOX_W = 200
MAX_BOX_H = 200
MAX_ASPECT_RATIO = 5.0

# --- TRACKER CONFIGURATION ---
IOU_MATCH_THRESHOLD = 0.10
MAX_CENTER_DIST = 80.0
MAX_MISSED_FRAMES = 4
MIN_HITS = 3
EMIT_MISSED_TRACKS = False

# --- CUSTOM JSON FORMAT CONFIG ---
SOURCE_ID = 1290217
SOURCE_TYPE = "DATA_FLOW"
SOURCE_NAME = "Without Task"
VALIDITY = "VALID"
DEVICE_NAME = "image_0"
CLASS_NUMBER = 29
CREATED_BY = 840012

os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# =========================================================
# HELPERS
# =========================================================
def load_frame_paths(frames_dir: str):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    frame_paths = []
    for ext in exts:
        frame_paths.extend(glob.glob(os.path.join(frames_dir, ext)))
    return sorted(frame_paths)

def compute_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0

def box_center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

def center_distance(box_a, box_b):
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    return math.hypot(ax - bx, ay - by)

def box_wh(box):
    return max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])

def normalize_box(box):
    x1, x2 = min(box[0], box[2]), max(box[0], box[2])
    y1, y2 = min(box[1], box[3]), max(box[1], box[3])
    return [float(x1), float(y1), float(x2), float(y2)]

def clamp_box_to_frame(box, frame_w, frame_h):
    x1 = max(0.0, min(float(box[0]), float(frame_w - 1)))
    x2 = max(0.0, min(float(box[2]), float(frame_w - 1)))
    y1 = max(0.0, min(float(box[1]), float(frame_h - 1)))
    y2 = max(0.0, min(float(box[3]), float(frame_h - 1)))
    return normalize_box([x1, y1, x2, y2])

def get_roi(frame):
    h, w = frame.shape[:2]
    if USE_TOP_ROI:
        return frame[:int(h * TOP_ROI_RATIO), :], 0, 0
    return frame, 0, 0

def upscale_image(img, factor):
    if factor == 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_LINEAR)

def is_valid_box(box):
    w, h = box_wh(box)

    if not ((MIN_BOX_W <= w <= MAX_BOX_W) and (MIN_BOX_H <= h <= MAX_BOX_H)):
        return False

    if w > 0 and h > 0:
        ratio = max(w / h, h / w)
        if ratio > MAX_ASPECT_RATIO:
            return False

    return True

def nms_boxes(detections, iou_thr=0.35):
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: x["score"], reverse=True)
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [
            det for det in detections
            if compute_iou(best["bbox"], det["bbox"]) < iou_thr
        ]
    return kept

def get_class_name(model, class_id):
    names = model.names
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)

def detect_objects(model, frame):
    roi, offset_x, offset_y = get_roi(frame)
    scale = UPSCALE_FACTOR if UPSCALE_ROI else 1.0
    roi_infer = upscale_image(roi, scale)

    results = model.predict(
        source=roi_infer,
        conf=CONF_THRESHOLD,
        iou=YOLO_NMS_IOU,
        agnostic_nms=True,
        verbose=False,
        device=DEVICE
    )[0]

    detections = []
    if results.boxes is None or len(results.boxes) == 0:
        return detections

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    clss = results.boxes.cls.cpu().numpy().astype(int)
    frame_h, frame_w = frame.shape[:2]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(float)
        x1, y1, x2, y2 = (
            (x1 / scale) + offset_x,
            (y1 / scale) + offset_y,
            (x2 / scale) + offset_x,
            (y2 / scale) + offset_y,
        )
        bbox = clamp_box_to_frame([x1, y1, x2, y2], frame_w, frame_h)

        if is_valid_box(bbox):
            class_id = int(clss[i])
            class_name = get_class_name(model, class_id)
            detections.append({
                "bbox": bbox,
                "score": float(confs[i]),
                "class_id": class_id,
                "class_name": class_name
            })

    return nms_boxes(detections, iou_thr=0.35)

def draw_box(frame, bbox, track_id, class_name, source="detected"):
    x1, y1, x2, y2 = map(int, bbox)
    box_color = (0, 255, 0) if source == "detected" else (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    label = f"ID:{track_id} {class_name}"
    (font_w, font_h), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    text_y = max(font_h + 2, y1 - 3)

    cv2.rectangle(
        frame,
        (x1, text_y - font_h - 2),
        (x1 + font_w + 4, text_y + baseline),
        box_color,
        -1
    )
    cv2.putText(
        frame,
        label,
        (x1 + 2, text_y - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA
    )

def bbox_xyxy_to_instance(track):
    x1, y1, x2, y2 = track["bbox"]

    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)

    w = abs(x_max - x_min)
    h = abs(y_max - y_min)
    area = int(round(w * h))
    now_ms = int(datetime.now().timestamp() * 1000)

    track_id_str = str(track["track_id"])

    return {
        "trackId": track_id_str,
        "trackName": track_id_str,
        "groups": [],
        "contour": {
            "area": area,
            "points": [
                {"x": round(x_min, 3), "y": round(y_min, 3)},
                {"x": round(x_max, 3), "y": round(y_min, 3)},
                {"x": round(x_max, 3), "y": round(y_max, 3)},
                {"x": round(x_min, 3), "y": round(y_max, 3)}
            ],
            "rotation": 0
        },
        "modelConfidence": float(track.get("score")) if track.get("score") is not None else None,
        "modelClass": None,
        "classVersion": 1,
        "isValid": None,
        "note": None,
        "start": None,
        "end": None,
        "deviceName": DEVICE_NAME,
        "deviceFrame": None,
        "bevFrameName": None,
        "index": None,
        "role": None,
        "content": None,
        "id": str(uuid.uuid4()),
        "type": "BOUNDING_BOX",
        "classId": int(track.get("class_id", -1)),
        "className": str(track.get("class_name", "unknown")),
        "classNumber": CLASS_NUMBER,
        "classValues": [],
        "createdAt": now_ms,
        "createdBy": CREATED_BY
    }

def build_output_json(frame_path, frame_index, tracks):
    instances = [bbox_xyxy_to_instance(track) for track in tracks]

    return [
        {
            "version": "1.0",
            "dataId": frame_index,
            "sourceId": SOURCE_ID,
            "sourceType": SOURCE_TYPE,
            "sourceName": SOURCE_NAME,
            "validity": VALIDITY,
            "classifications": [],
            "instances": instances,
            "segments": [],
            "entities": None,
            "relations": None
        }
    ]

# =========================================================
# TRACKER
# =========================================================
class StableTracker:
    def __init__(self, iou_threshold=0.10, max_missed=4, max_center_dist=80.0, min_hits=3):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.max_center_dist = max_center_dist
        self.min_hits = min_hits
        self.tracks = []
        self.next_id = 1

    def _predict_bbox(self, track):
        x1, y1, x2, y2 = track["bbox"]
        return [x1 + track["vx"], y1 + track["vy"], x2 + track["vx"], y2 + track["vy"]]

    def update(self, detections, frame_w, frame_h):
        matched_track_ids, matched_det_ids = set(), set()

        for track in self.tracks:
            if track["age"] >= 2:
                track["pred_bbox"] = clamp_box_to_frame(
                    self._predict_bbox(track), frame_w, frame_h
                )
            else:
                track["pred_bbox"] = track["bbox"][:]

        candidates = []
        for ti, track in enumerate(self.tracks):
            ref_box = track["pred_bbox"]
            for di, det in enumerate(detections):
                iou = compute_iou(ref_box, det["bbox"])
                dist = center_distance(ref_box, det["bbox"])

                if iou >= self.iou_threshold or dist <= self.max_center_dist:
                    score = iou - (dist / self.max_center_dist)
                    candidates.append((score, ti, di))

        candidates.sort(reverse=True, key=lambda x: x[0])

        for score, ti, di in candidates:
            if ti in matched_track_ids or di in matched_det_ids:
                continue

            track = self.tracks[ti]
            det = detections[di]

            old_cx, old_cy = box_center(track["bbox"])
            new_cx, new_cy = box_center(det["bbox"])
            raw_vx, raw_vy = new_cx - old_cx, new_cy - old_cy

            if track["age"] == 1:
                track["vx"], track["vy"] = raw_vx, raw_vy
            else:
                track["vx"] = (0.6 * track["vx"]) + (0.4 * raw_vx)
                track["vy"] = (0.6 * track["vy"]) + (0.4 * raw_vy)

            track["bbox"] = det["bbox"]
            track["score"] = det["score"]
            track["class_id"] = det["class_id"]
            track["class_name"] = det["class_name"]
            track["missed"] = 0
            track["age"] += 1
            track["hits"] += 1
            track["last_source"] = "detected"

            matched_track_ids.add(ti)
            matched_det_ids.add(di)

        for ti, track in enumerate(self.tracks):
            if ti not in matched_track_ids:
                track["missed"] += 1
                track["age"] += 1
                track["bbox"] = clamp_box_to_frame(track["pred_bbox"], frame_w, frame_h)
                track["last_source"] = "tracked"
                track["vx"] *= 0.30
                track["vy"] *= 0.30

        for di, det in enumerate(detections):
            if di not in matched_det_ids:
                self.tracks.append({
                    "track_id": self.next_id,
                    "bbox": det["bbox"],
                    "pred_bbox": det["bbox"][:],
                    "score": det["score"],
                    "class_id": det["class_id"],
                    "class_name": det["class_name"],
                    "missed": 0,
                    "age": 1,
                    "hits": 1,
                    "vx": 0.0,
                    "vy": 0.0,
                    "last_source": "detected"
                })
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t["missed"] <= self.max_missed]

        outputs = []
        for track in self.tracks:
            if track["hits"] >= self.min_hits:
                if track["missed"] == 0 or (EMIT_MISSED_TRACKS and track["missed"] <= self.max_missed):
                    outputs.append(track.copy())

        return outputs

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():
    if not os.path.isdir(FRAMES_DIR):
        raise FileNotFoundError(f"Dir not found: {FRAMES_DIR}")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    frame_paths = load_frame_paths(FRAMES_DIR)
    if not frame_paths:
        raise ValueError(f"No frames in: {FRAMES_DIR}")

    print(f"[INFO] Total frames: {len(frame_paths)} | Device: {DEVICE}")

    model = YOLO(MODEL_PATH)
    tracker = StableTracker(
        iou_threshold=IOU_MATCH_THRESHOLD,
        max_missed=MAX_MISSED_FRAMES,
        max_center_dist=MAX_CENTER_DIST,
        min_hits=MIN_HITS
    )

    for idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        frame_h, frame_w = frame.shape[:2]
        detections = detect_objects(model, frame)
        tracks = tracker.update(detections, frame_w, frame_h)

        num_detected, num_tracked = 0, 0

        for tr in tracks:
            track_id = tr["track_id"]
            x1, y1, x2, y2 = tr["bbox"]
            source = tr["last_source"]
            class_name = tr.get("class_name", "unknown")

            if source == "detected":
                num_detected += 1
            else:
                num_tracked += 1

            draw_box(frame, [x1, y1, x2, y2], track_id, class_name, source=source)

        if USE_TOP_ROI:
            y_line = int(frame_h * TOP_ROI_RATIO)
            cv2.line(frame, (0, y_line), (frame_w, y_line), (255, 255, 0), 2)

        json_data = build_output_json(frame_path, idx, tracks)

        json_path = os.path.join(
            OUTPUT_JSON_DIR,
            os.path.splitext(os.path.basename(frame_path))[0] + ".json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

        annotated_path = os.path.join(ANNOTATED_DIR, os.path.basename(frame_path))
        ok = cv2.imwrite(annotated_path, frame)

        if not ok:
            print(f"[ERROR] Failed to save annotated image: {annotated_path}")
        else:
            print(f"[SAVED] Annotated image: {annotated_path}")

        print(
            f"[INFO] {idx + 1}/{len(frame_paths)} | det={len(detections)} "
            f"out={len(tracks)} (D:{num_detected}, T:{num_tracked}) | {os.path.basename(frame_path)}"
        )

    print("[DONE] Finished processing sequence.")

if __name__ == "__main__":
    main()