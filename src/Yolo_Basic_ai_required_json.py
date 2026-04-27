import os
import json
import cv2
from ultralytics import YOLO

# =====================================================
# PATHS
# =====================================================
MODEL_PATH = "weights/best.pt"
INPUT_PATH = "data_samples/images"
OUTPUT_DIR = "results/json"
ONTOLOGY_PATH = "data_samples/dataset_ontology.json"  # test teamdan olgan ontology

CONF_THRES = 0.8

# =====================================================
def list_images(path: str):
    if os.path.isdir(path):
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(exts)]
        files.sort()
        return files
    return [path]

def load_ontology(ontology_path: str):
    with open(ontology_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = []
    class_map = {}  # name -> id

    for cls in data.get("classes", []):
        cid = int(cls["id"])
        name = str(cls["name"])
        categories.append({
            "id": cid,
            "name": name,
            "supercategory": "object"  # test JSON'dagi kabi
        })
        class_map[name] = cid

    return categories, class_map

# =====================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
names = model.names  # dict yoki list

def get_class_name(cls_id: int) -> str:
    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return names[cls_id]
    return str(cls_id)

categories, class_map = load_ontology(ONTOLOGY_PATH)

images = list_images(INPUT_PATH)
assert images, "❌ No images found in INPUT_PATH"

for img_path in images:
    img_name = os.path.basename(img_path)
    stem, _ = os.path.splitext(img_name)

    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Failed to read: {img_path}")
        continue

    h, w = img.shape[:2]

    results = model.predict(source=img, conf=CONF_THRES, verbose=False)
    r0 = results[0]

    coco_data = {
        "info": {"description": "Single Output", "version": "1.0"},
        "images": [{"id": 1, "file_name": img_name, "width": int(w), "height": int(h)}],
        "annotations": [],
        "categories": categories
    }

    ann_id = 1
    missing_names = set()

    if r0.boxes is not None and len(r0.boxes) > 0:
        xyxy = r0.boxes.xyxy.cpu().numpy()
        conf = r0.boxes.conf.cpu().numpy()
        cls  = r0.boxes.cls.cpu().numpy()

        for (x1, y1, x2, y2), sc, ci in zip(xyxy, conf, cls):
            sc = float(sc)
            ci = int(ci)

            class_name = get_class_name(ci)
            category_id = class_map.get(class_name, 0)
            if category_id == 0:
                missing_names.add(class_name)

            x = float(x1)
            y = float(y1)
            bw = float(max(0.0, x2 - x1))
            bh = float(max(0.0, y2 - y1))
            area = float(bw * bh)

            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": 1,
                "category_id": int(category_id),
                "bbox": [round(x, 3), round(y, 3), round(bw, 3), round(bh, 3)],
                "area": round(area, 3),
                "iscrowd": 0,
                "score": round(sc, 5)
            })
            ann_id += 1

    if missing_names:
        print(f"⚠️ category_id=0 (ontology'da yo'q nomlar): {sorted(missing_names)}")

    out_json = os.path.join(OUTPUT_DIR, f"{stem}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        # test team fayllari compact bo'lsa shunaqa qoldiramiz
        json.dump(coco_data, f, ensure_ascii=False, separators=(",", ":"))

    print(f"✅ Saved: {out_json} | anns={len(coco_data['annotations'])}")

print("\n✅ Done.")