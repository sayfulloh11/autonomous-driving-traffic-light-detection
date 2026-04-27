import os
import glob
import json
import math
import cv2
from ultralytics import YOLO

# =========================================================
# CONFIGURATION
# =========================================================
FRAMES_DIR = "data_samples/images"
OUTPUT_JSON_DIR = "results/json"
ANNOTATED_DIR = "results/annotated_frames"
MODEL_PATH = "weights/best.pt"
ONTOLOGY_PATH = "data_samples/dataset_ontology.json"
DEVICE = "cuda:1"

# --- DETECTION THRESHOLDS & NIGHTTIME FIXES ---
CONF_THRESHOLD = 0.22      
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

# --- ADVANCED TRACKER CONFIGURATION ---
IOU_MATCH_THRESHOLD = 0.10
MAX_CENTER_DIST = 80.0     
MAX_MISSED_FRAMES = 4      
MIN_HITS = 3               
EMIT_MISSED_TRACKS = False 

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
    if factor == 1.0: return img
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
    if not detections: return []
    detections = sorted(detections, key=lambda x: x["score"], reverse=True)
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [det for det in detections if compute_iou(best["bbox"], det["bbox"]) < iou_thr]
    return kept

def detect_objects(model, frame):
    roi, offset_x, offset_y = get_roi(frame)
    scale = UPSCALE_FACTOR if UPSCALE_ROI else 1.0
    roi_infer = upscale_image(roi, scale)

    results = model.predict(
        source=roi_infer, conf=CONF_THRESHOLD, iou=YOLO_NMS_IOU,
        agnostic_nms=True, verbose=False, device=DEVICE
    )[0]

    detections = []
    if results.boxes is None or len(results.boxes) == 0:
        return detections

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy() 

    frame_h, frame_w = frame.shape[:2]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.astype(float)
        x1, y1, x2, y2 = (x1/scale) + offset_x, (y1/scale) + offset_y, (x2/scale) + offset_x, (y2/scale) + offset_y
        bbox = clamp_box_to_frame([x1, y1, x2, y2], frame_w, frame_h)

        if is_valid_box(bbox):
            class_name = model.names[int(classes[i])]
            detections.append({
                "bbox": bbox, 
                "score": float(confs[i]), 
                "class_name": class_name 
            })

    return nms_boxes(detections, iou_thr=0.35)

def draw_box(frame, bbox, track_id, class_name, source="detected"):
    x1, y1, x2, y2 = map(int, bbox)
    box_color = (0, 255, 0) if source == "detected" else (0, 165, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    label = f"ID:{track_id} {class_name}"
    (font_w, font_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    text_y = max(font_h + 2, y1 - 3)

    cv2.rectangle(frame, (x1, text_y - font_h - 2), (x1 + font_w + 4, text_y + baseline), box_color, -1)
    cv2.putText(frame, label, (x1 + 2, text_y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

# =========================================================
# ADVANCED TRACKER
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
        px1, py1 = x1 + track["vx"], y1 + track["vy"]
        px2, py2 = x2 + track["vx"], y2 + track["vy"]
        half_vw, half_vh = track["vw"] / 2.0, track["vh"] / 2.0
        return [px1 - half_vw, py1 - half_vh, px2 + half_vw, py2 + half_vh]

    def _is_near_edge(self, box, frame_w, frame_h, margin=40):
        x1, y1, x2, y2 = box
        return (y1 <= margin) or (x1 <= margin) or (x2 >= frame_w - margin)

    def update(self, detections, frame_w, frame_h):
        matched_track_ids, matched_det_ids = set(), set()

        for track in self.tracks:
            if track["age"] >= 2:
                track["pred_bbox"] = clamp_box_to_frame(self._predict_bbox(track), frame_w, frame_h)
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
            if ti in matched_track_ids or di in matched_det_ids: continue

            track = self.tracks[ti]
            det = detections[di]

            old_cx, old_cy = box_center(track["bbox"])
            new_cx, new_cy = box_center(det["bbox"])
            old_w, old_h = box_wh(track["bbox"])
            new_w, new_h = box_wh(det["bbox"])

            raw_vx, raw_vy = new_cx - old_cx, new_cy - old_cy
            raw_vw, raw_vh = new_w - old_w, new_h - old_h

            if track["age"] == 1:
                track["vx"], track["vy"] = raw_vx, raw_vy
                track["vw"], track["vh"] = raw_vw, raw_vh
            else:
                track["vx"] = (0.6 * track["vx"]) + (0.4 * raw_vx)
                track["vy"] = (0.6 * track["vy"]) + (0.4 * raw_vy)
                track["vw"] = (0.6 * track["vw"]) + (0.4 * raw_vw)
                track["vh"] = (0.6 * track["vh"]) + (0.4 * raw_vh)

            track["bbox"] = det["bbox"]
            track["score"] = det["score"]
            track["class_name"] = det["class_name"] 
            track["missed"] = 0
            track["age"] += 1
            track["hits"] += 1
            track["last_source"] = "detected"

            matched_track_ids.add(ti)
            matched_det_ids.add(di)

        for ti, track in enumerate(self.tracks):
            if ti not in matched_track_ids:
                if self._is_near_edge(track["bbox"], frame_w, frame_h):
                    track["missed"] = self.max_missed + 1 
                    continue
                
                track["missed"] += 1
                track["age"] += 1
                track["bbox"] = clamp_box_to_frame(track["pred_bbox"], frame_w, frame_h)
                track["last_source"] = "tracked"
                track["vx"] *= 0.30
                track["vy"] *= 0.30
                track["vw"] *= 0.10 
                track["vh"] *= 0.10

        for di, det in enumerate(detections):
            if di not in matched_det_ids:
                self.tracks.append({
                    "track_id": self.next_id, 
                    "bbox": det["bbox"], 
                    "pred_bbox": det["bbox"][:],
                    "score": det["score"], 
                    "class_name": det["class_name"], 
                    "missed": 0, "age": 1, "hits": 1,
                    "vx": 0.0, "vy": 0.0, "vw": 0.0, "vh": 0.0, 
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
# MAIN EXECUTION (COCO-STYLE EXACT FORMAT MATCH)
# =========================================================
def main():
    if not os.path.isdir(FRAMES_DIR): raise FileNotFoundError(f"Dir not found: {FRAMES_DIR}")
    if not os.path.exists(MODEL_PATH): raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(ONTOLOGY_PATH): raise FileNotFoundError(f"Ontology JSON not found: {ONTOLOGY_PATH}")

    # 1. Load Ontology
    with open(ONTOLOGY_PATH, 'r', encoding='utf-8') as f:
        ontology_data = json.load(f)
    
    ontology_map = {cls["name"]: cls["id"] for cls in ontology_data["classes"]}
    fallback_class_name = ontology_data["classes"][0]["name"]
    fallback_class_id = ontology_data["classes"][0]["id"]

    # 2. Build the exact "categories" list block for the JSON
    output_categories = []
    for cls in ontology_data["classes"]:
        output_categories.append({
            "id": cls["id"],
            "name": cls["name"],
            "supercategory": "object"
        })

    frame_paths = load_frame_paths(FRAMES_DIR)
    if not frame_paths: raise ValueError(f"No frames in: {FRAMES_DIR}")

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
        if frame is None: continue

        frame_h, frame_w = frame.shape[:2]
        detections = detect_objects(model, frame)
        tracks = tracker.update(detections, frame_w, frame_h)

        # Rasm faylining nomidan main_id ni olish (masalan: 1764593607.206575.jpg -> 1764593607)
        file_name = os.path.basename(frame_path)
        try:
            main_id = int(file_name.split('.')[0])
        except ValueError:
            main_id = idx + 1 # Agar son bo'lmasa, ehtiyot chorasi sifatida idx beriladi

        annotations = []
        num_detected, num_tracked = 0, 0
        annotation_id_counter = 1

        for tr in tracks:
            track_id = tr["track_id"]
            x1, y1, x2, y2 = tr["bbox"]
            score = tr["score"]
            class_name = tr["class_name"] 
            source = tr["last_source"]
            
            if source == "detected": num_detected += 1
            else: num_tracked += 1

            class_id = ontology_map.get(class_name, fallback_class_id)
            
            # Formating bbox to COCO style: [x, y, width, height]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            area = w * h

            annotations.append({
                "id": annotation_id_counter,
                "image_id": 1,
                "category_id": int(class_id),
                "bbox": [
                    round(float(x1), 3),
                    round(float(y1), 3),
                    round(float(w), 3),
                    round(float(h), 3)
                ],
                "area": round(float(area), 3),
                "iscrowd": 0,
                "score": round(float(score), 5),
                "track_id": int(track_id),
                "main_id": main_id
            })
            annotation_id_counter += 1

            # Rasmga chizish
            draw_box(frame, [x1, y1, x2, y2], track_id, class_name, source=source)

        if USE_TOP_ROI:
            y_line = int(frame_h * TOP_ROI_RATIO)
            cv2.line(frame, (0, y_line), (frame_w, y_line), (255, 255, 0), 2)

        # Xuddi siz yuborgan NAMUNA (SAMPLE) strukturasi:
        json_data = {
            "info": {
                "description": "Single Output (ByteTrack)",
                "version": "1.0"
            },
            "images": [
                {
                    "id": 1,
                    "file_name": file_name,
                    "width": int(frame_w),
                    "height": int(frame_h)
                }
            ],
            "annotations": annotations,
            "categories": output_categories
        }

        # JSON ni saqlash
        json_name = os.path.splitext(file_name)[0] + ".json"
        json_path = os.path.join(OUTPUT_JSON_DIR, json_name)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)

        # Annotatsiya qilingan rasmni saqlash
        cv2.imwrite(os.path.join(ANNOTATED_DIR, file_name), frame)

        print(f"[INFO] {idx + 1}/{len(frame_paths)} | det={len(detections)} out={len(annotations)} (D:{num_detected}, T:{num_tracked}) | {file_name}")

    print("[DONE] Finished processing sequence successfully.")

if __name__ == "__main__":
    main()