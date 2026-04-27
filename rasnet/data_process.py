import os
import json
import shutil
from collections import defaultdict, Counter
from tqdm import tqdm

# =====================================================
# 🔧 FAQAT SHU PATHLARNI O'ZGARTIRASIZ
# =====================================================
ROOT_DATA_DIR = "/home/sayfulloh/swm_traffic/rasnet/datas/Labelled_data"
OUTPUT_DATASET = "/home/sayfulloh/swm_traffic/rasnet/dataset/withclasses"

# =====================================================
# 🔍 FIND COCO JSON
# =====================================================
json_files = []
for root, _, files in os.walk(ROOT_DATA_DIR):
    for f in files:
        if f.lower().endswith(".json"):
            json_files.append(os.path.join(root, f))

if not json_files:
    raise RuntimeError("❌ JSON file not found")

print("📄 JSON files found:")
for j in json_files:
    print("  -", j)

# =====================================================
# 📖 LOAD FIRST VALID COCO JSON
# =====================================================
data = None
ANNOT_FILE = None

for jf in json_files:
    try:
        with open(jf, "r") as f:
            tmp = json.load(f)
        if "images" in tmp and "annotations" in tmp and "categories" in tmp:
            data = tmp
            ANNOT_FILE = jf
            break
    except:
        pass

if data is None:
    raise RuntimeError("❌ COCO-style JSON not found")

print(f"\n✅ Using annotation file: {ANNOT_FILE}")

# =====================================================
# 🧠 BUILD CLASS MAP (AUTOMATIC)
# =====================================================
print("\n🧠 Extracting classes from JSON...")

categories = sorted(data["categories"], key=lambda x: x["id"])

CLASS_MAP = {cat["id"]: cat["name"] for cat in categories}

print("📂 Classes found:")
for k, v in CLASS_MAP.items():
    print(f"  - {k}: {v}")

# save mapping
os.makedirs(OUTPUT_DATASET, exist_ok=True)
CLASS_MAP_PATH = os.path.join(OUTPUT_DATASET, "classes.json")

with open(CLASS_MAP_PATH, "w") as f:
    json.dump(CLASS_MAP, f, indent=2)

print(f"\n💾 Saved class map → {CLASS_MAP_PATH}")

# =====================================================
# 🔍 INDEX ALL IMAGES RECURSIVELY
# =====================================================
print("\n🔍 Scanning images on disk...")

image_index = {}
for root, _, files in os.walk(ROOT_DATA_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            image_index[f] = os.path.join(root, f)

print(f"🖼 Total images found: {len(image_index)}")

# =====================================================
# 🧠 IMAGE NAME PARSER (ROBUST)
# =====================================================
def get_image_name(img):
    for k in ["file_name", "filename", "imagePath", "name"]:
        if k in img:
            return os.path.basename(img[k])
    raise KeyError(f"Image filename key not found: {img}")

# =====================================================
# 🗂 IMAGE ↔ ANNOTATION MAP
# =====================================================
image_id_to_name = {
    img["id"]: get_image_name(img) for img in data["images"]
}

image_to_anns = defaultdict(list)
for ann in data["annotations"]:
    image_to_anns[ann["image_id"]].append(ann)

# =====================================================
# 📁 OUTPUT STRUCTURE
# =====================================================
IMG_OUT = os.path.join(OUTPUT_DATASET, "images")
LBL_OUT = os.path.join(OUTPUT_DATASET, "labels")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

# =====================================================
# 📊 STATS
# =====================================================
total_images = 0
used_images = 0
total_boxes = 0
class_counter = Counter()
bbox_areas = []

# =====================================================
# 🚀 PROCESS DATASET
# =====================================================
print("\n🚀 Preparing dataset...")

for image_id, img_name in tqdm(image_id_to_name.items()):
    if img_name not in image_index:
        continue

    total_images += 1
    anns = image_to_anns.get(image_id, [])
    if not anns:
        continue

    used_images += 1

    # copy image
    shutil.copy(
        image_index[img_name],
        os.path.join(IMG_OUT, img_name)
    )

    # write label
    with open(os.path.join(LBL_OUT, img_name.rsplit(".", 1)[0] + ".txt"), "w") as f:
        for ann in anns:
            cls_id = ann["category_id"]   # ORIGINAL ID
            x, y, w, h = ann["bbox"]

            x1, y1 = x, y
            x2, y2 = x + w, y + h

            f.write(f"{cls_id} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

            total_boxes += 1
            class_counter[cls_id] += 1
            bbox_areas.append(w * h)

# =====================================================
# 📊 SUMMARY
# =====================================================
print("\n" + "=" * 60)
print("📊 DATASET SUMMARY")
print("=" * 60)
print(f"🖼 Total images matched     : {total_images}")
print(f"✅ Images used              : {used_images}")
print(f"📦 Total bounding boxes     : {total_boxes}")

if used_images > 0:
    print(f"📈 Avg boxes per image      : {total_boxes / used_images:.2f}")

print("\n📂 Class distribution:")
for cls, cnt in sorted(class_counter.items()):
    print(f"  - {cls} ({CLASS_MAP.get(cls,'?')}): {cnt}")

if bbox_areas:
    print("\n📐 Bounding box area stats:")
    print(f"  - Avg : {sum(bbox_areas)/len(bbox_areas):.1f}")
    print(f"  - Min : {min(bbox_areas):.1f}")
    print(f"  - Max : {max(bbox_areas):.1f}")

print("\n✅ Dataset preparation finished")
print(f"📁 Output folder: {OUTPUT_DATASET}")
print("=" * 60)
