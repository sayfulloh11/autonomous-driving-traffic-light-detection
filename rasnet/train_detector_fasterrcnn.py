import os
import json
import random
import torch
import cv2
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import ResNet50_Weights

# =====================================================
# 🔧 PATHS (FAQAT SHULARNI TEKSHIRING)
# =====================================================
DATASET_DIR = "/home/sayfulloh/swm_traffic/rasnet/dataset/dataset_augmented"
MODEL_SAVE_PATH = "/home/sayfulloh/swm_traffic/rasnet/model/fasterrcnn_detector.pth"

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {DEVICE}")

# =====================================================
# 🔒 SEED
# =====================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# =====================================================
# 📖 LOAD CLASSES + BUILD CONTINUOUS ID MAP
# =====================================================
with open(os.path.join(DATASET_DIR, "classes.json")) as f:
    CLASS_MAP = json.load(f)

# original ids (sparse)
ORIG_CLASS_IDS = sorted(int(k) for k in CLASS_MAP.keys())

# remap: original_id -> 0..N-1
ID_MAP = {cid: i for i, cid in enumerate(ORIG_CLASS_IDS)}
INV_ID_MAP = {v: k for k, v in ID_MAP.items()}

NUM_CLASSES = len(ID_MAP) + 1   # + background

print("\n🔁 Class ID remapping (original → model):")
for k, v in ID_MAP.items():
    print(f"  {k} → {v}")

print(f"\n📂 Total classes: {len(ID_MAP)} | NUM_CLASSES={NUM_CLASSES}")

# =====================================================
# 📁 DATASET PATHS
# =====================================================
IMG_DIR = os.path.join(DATASET_DIR, "images")
LBL_DIR = os.path.join(DATASET_DIR, "labels")

# =====================================================
# 📦 DATASET
# =====================================================
class DetectionDataset(Dataset):
    def __init__(self):
        self.imgs = sorted([
            f for f in os.listdir(IMG_DIR)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        name = self.imgs[idx]

        img = cv2.imread(os.path.join(IMG_DIR, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = F.to_tensor(img)

        boxes, labels = [], []
        lbl_path = os.path.join(LBL_DIR, name.rsplit(".", 1)[0] + ".txt")

        with open(lbl_path) as f:
            for line in f:
                cid, x1, y1, x2, y2 = map(float, line.split())
                cid = int(cid)

                if cid not in ID_MAP:
                    raise ValueError(f"❌ Unknown class id {cid}")

                mapped = ID_MAP[cid]

                boxes.append([x1, y1, x2, y2])
                labels.append(mapped + 1)  # +1 for background

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))

# =====================================================
# 📦 DATALOADER
# =====================================================
loader = DataLoader(
    DetectionDataset(),
    batch_size=30,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

# =====================================================
# 🧠 MODEL (SMALL OBJECT FRIENDLY)
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

model.to(DEVICE)

# =====================================================
# ❄️ STAGE 1 — BACKBONE FREEZE
# =====================================================
for p in model.backbone.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

EPOCHS_STAGE1 = 80

print("\n🚀 Stage 1 training (backbone frozen)")
for epoch in range(EPOCHS_STAGE1):
    model.train()
    total_loss = 0.0

    for imgs, targets in tqdm(loader):
        imgs = [i.to(DEVICE) for i in imgs]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS_STAGE1} | Loss: {total_loss:.2f}")

# =====================================================
# 🔓 STAGE 2 — UNFREEZE LAST BLOCK
# =====================================================
for p in model.backbone.body.layer4.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)

EPOCHS_STAGE2 = 10

print("\n🚀 Stage 2 fine-tuning (layer4 unfrozen)")
for epoch in range(EPOCHS_STAGE2):
    model.train()
    total_loss = 0.0

    for imgs, targets in tqdm(loader):
        imgs = [i.to(DEVICE) for i in imgs]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS_STAGE2} | Loss: {total_loss:.2f}")

# =====================================================
# 💾 SAVE MODEL + REMAP (PROD UCHUN MUHIM)
# =====================================================
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save({
    "model": model.state_dict(),
    "id_map": ID_MAP,
    "inv_id_map": INV_ID_MAP
}, MODEL_SAVE_PATH)

print("\n✅ Detector training finished successfully")
print(f"💾 Model saved to: {MODEL_SAVE_PATH}")
