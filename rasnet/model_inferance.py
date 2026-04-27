


import os
import json
import cv2
import torch
import numpy as np
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import ResNet50_Weights

# =====================================================
# 🔧 PATHS (FAQAT SHULARNI O'ZGARTIRASIZ)
# =====================================================
MODEL_PATH = "/home/sayfulloh/swm_traffic/rasnet/model/fasterrcnn_detector.pth"
INPUT_PATH = "/home/sayfulloh/swm_traffic/all_data/unseen/Data_Set/20251129"
OUTPUT_DIR = "/home/sayfulloh/swm_traffic/rasnet/infer_out_20251129"
DATASET_DIR = "/home/sayfulloh/swm_traffic/rasnet/dataset/dataset_augmented"

CONF_THRES = 0.5
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# 📦 LOAD CHECKPOINT
# =====================================================
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
ID_MAP = ckpt["id_map"]
INV_ID_MAP = ckpt["inv_id_map"]

NUM_CLASSES = len(ID_MAP) + 1  # 🔥 MUHIM

print(f"✅ Loaded checkpoint | NUM_CLASSES={NUM_CLASSES}")

# =====================================================
# 📖 LOAD CLASS NAMES
# =====================================================
with open(os.path.join(DATASET_DIR, "classes.json")) as f:
    CLASS_NAMES = json.load(f)

# =====================================================
# 🧠 BUILD MODEL (TRAIN BILAN 100% MOS)
# =====================================================
anchor_generator = AnchorGenerator(
    sizes=((8,), (16,), (32,), (64,), (128,)),
    aspect_ratios=((0.5, 1.0, 2.0),) * 5
)

model = fasterrcnn_resnet50_fpn(
    weights=None,
    weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
    rpn_anchor_generator=anchor_generator
)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    NUM_CLASSES
)

model.load_state_dict(ckpt["model"])

# =====================================================
# 🔧 UPDATE: RPN PROPOSAL TUNING (FALSE NEGATIVE ↓)
# =====================================================
model.rpn.pre_nms_top_n_train = 4000
model.rpn.post_nms_top_n_train = 2000

model.rpn.pre_nms_top_n_test = 2000
model.rpn.post_nms_top_n_test = 1000

# (ixtiyoriy) NMS yumshatish
model.roi_heads.nms_thresh = 0.5

model.to(DEVICE)
model.eval()


print(f"🚀 Model ready on {DEVICE}")

# =====================================================
# 🖼 INPUT IMAGES
# =====================================================
if os.path.isdir(INPUT_PATH):
    images = [
        os.path.join(INPUT_PATH, f)
        for f in os.listdir(INPUT_PATH)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
else:
    images = [INPUT_PATH]

assert images, "❌ No images found"

# =====================================================
# 🎯 INFERENCE
# =====================================================
for img_path in images:
    img_name = os.path.basename(img_path)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).to(DEVICE)

    with torch.no_grad():
        output = model([img_tensor])[0]

    boxes = output["boxes"].cpu().numpy()
    scores = output["scores"].cpu().numpy()
    labels = output["labels"].cpu().numpy()

    for box, score, lbl in zip(boxes, scores, labels):
        if score < CONF_THRES:
            continue

        x1, y1, x2, y2 = box.astype(int)

        mapped_id = int(lbl - 1)
        orig_id = INV_ID_MAP[mapped_id]
        class_name = CLASS_NAMES.get(str(orig_id), f"id_{orig_id}")

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{class_name} {score:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1
        )

    out_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_path, img)
    print(f"🖼 Saved: {out_path}")

print("\n✅ Inference finished successfully")
