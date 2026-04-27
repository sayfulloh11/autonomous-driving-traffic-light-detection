import os
import json
import cv2
import random
import shutil
from collections import Counter, defaultdict
from tqdm import tqdm
import albumentations as A

# =====================================================
# 🔧 PATHS (FAQAT SHUNI O'ZGARTIRING)
# =====================================================
SRC_DATASET = "/home/sayfulloh/swm_traffic/rasnet/dataset/withclasses"
OUT_DATASET = "/home/sayfulloh/swm_traffic/rasnet/dataset/dataset_augmented"

IMG_IN = os.path.join(SRC_DATASET, "images")
LBL_IN = os.path.join(SRC_DATASET, "labels")
CLS_JSON = os.path.join(SRC_DATASET, "classes.json")

IMG_OUT = os.path.join(OUT_DATASET, "images")
LBL_OUT = os.path.join(OUT_DATASET, "labels")

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

# =====================================================
# 📖 LOAD CLASSES
# =====================================================
with open(CLS_JSON) as f:
    CLASS_MAP = {int(k): v for k, v in json.load(f).items()}

# =====================================================
# 🔍 COLLECT IMAGE → CLASSES
# =====================================================
img_classes = {}
class_counts = Counter()

for lbl in os.listdir(LBL_IN):
    lbl_path = os.path.join(LBL_IN, lbl)
    img_name = lbl.rsplit(".", 1)[0] + ".jpg"
    if not os.path.exists(os.path.join(IMG_IN, img_name)):
        continue

    classes = set()
    with open(lbl_path) as f:
        for line in f:
            cid = int(line.split()[0])
            classes.add(cid)
            class_counts[cid] += 1

    img_classes[img_name] = classes

print("\n📊 ORIGINAL CLASS COUNTS")
for k, v in sorted(class_counts.items()):
    print(f"{k} ({CLASS_MAP.get(k,'?')}): {v}")

# =====================================================
# 🎯 TARGET MULTIPLIER LOGIC
# =====================================================
def target_multiplier(count):
    if count <= 1:
        return 10
    if count <= 5:
        return 5
    if count <= 10:
        return 3
    if count <= 19:
        return 2
    return 0  # no augment

small_classes = {c for c, n in class_counts.items() if n < 20}
print(f"\n🔥 Small classes (<20): {len(small_classes)}")

# =====================================================
# 🧪 AUGMENT PIPELINE (BBOX-SAFE)
# =====================================================
augment = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.RandomGamma(p=0.3),
    A.GaussNoise(p=0.3),
    A.MotionBlur(blur_limit=3, p=0.2),
    A.ImageCompression(quality_lower=60, quality_upper=95, p=0.3),
    A.ShiftScaleRotate(
        shift_limit=0.03,
        scale_limit=0.05,
        rotate_limit=0,
        border_mode=cv2.BORDER_CONSTANT,
        p=0.3
    )
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# =====================================================
# 🚀 COPY ORIGINAL DATA FIRST
# =====================================================
print("\n📦 Copying original dataset...")
for f in os.listdir(IMG_IN):
    shutil.copy(os.path.join(IMG_IN, f), os.path.join(IMG_OUT, f))
for f in os.listdir(LBL_IN):
    shutil.copy(os.path.join(LBL_IN, f), os.path.join(LBL_OUT, f))
shutil.copy(CLS_JSON, os.path.join(OUT_DATASET, "classes.json"))

# =====================================================
# 🚀 AUGMENT ONLY SMALL CLASSES
# =====================================================
aug_counter = Counter()

print("\n🚀 Augmenting small classes...")
for img_name, classes in tqdm(img_classes.items()):
    # check if image contains any small class
    relevant = [c for c in classes if c in small_classes]
    if not relevant:
        continue

    # choose the rarest class in this image
    rarest = min(relevant, key=lambda c: class_counts[c])
    mult = target_multiplier(class_counts[rarest])
    if mult <= 0:
        continue

    img = cv2.imread(os.path.join(IMG_IN, img_name))
    h, w = img.shape[:2]

    # load bboxes
    bboxes, labels = [], []
    with open(os.path.join(LBL_IN, img_name.rsplit(".",1)[0] + ".txt")) as f:
        for line in f:
            cid, x1, y1, x2, y2 = line.split()
            bboxes.append([float(x1), float(y1), float(x2), float(y2)])
            labels.append(int(cid))

    for i in range(mult):
        aug = augment(image=img, bboxes=bboxes, labels=labels)
        aug_img = aug["image"]
        aug_bboxes = aug["bboxes"]
        aug_labels = aug["labels"]

        new_name = img_name.rsplit(".",1)[0] + f"_aug_{i}.jpg"
        cv2.imwrite(os.path.join(IMG_OUT, new_name), aug_img)

        with open(os.path.join(LBL_OUT, new_name.rsplit(".",1)[0] + ".txt"), "w") as f:
            for (x1,y1,x2,y2), cid in zip(aug_bboxes, aug_labels):
                f.write(f"{cid} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")
                aug_counter[cid] += 1

# =====================================================
# 📊 SUMMARY
# =====================================================
print("\n================================================")
print("📊 AUGMENTATION SUMMARY")
print("================================================")
for k, v in sorted(aug_counter.items()):
    print(f"{k} ({CLASS_MAP.get(k,'?')}): +{v}")
print("================================================")

print("\n✅ Augmented dataset ready:")
print(OUT_DATASET)
