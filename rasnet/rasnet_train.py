# import os
# import random
# import torch
# import cv2
# import numpy as np
# from tqdm import tqdm

# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import functional as F
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from torchvision.models.detection.anchor_utils import AnchorGenerator
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models import ResNet50_Weights

# # =====================================================
# # 🔧 PATHLAR (FAQAT SHUNI TEKSHIRING)
# # =====================================================
# DATASET_DIR = "/home/sayfulloh/swm_traffic/rasnet/dataset"
# MODEL_SAVE_PATH = "/home/sayfulloh/swm_traffic/rasnet/model/2_fasterrcnn_resnet50.pth"

# # =====================================================
# # 🔧 TRAIN CONFIG (DATASETGA MOS)
# # =====================================================
# EPOCHS_STAGE1 = 30
# EPOCHS_STAGE2 = 80
# BATCH_SIZE = 2
# LR_STAGE1 = 1e-4
# LR_STAGE2 = 5e-6
# NUM_WORKERS = 4
# SEED = 42

# # =====================================================
# # 🔧 DEVICE (FAKAT GPU 0)
# # =====================================================
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(f"🔥 Using device: {DEVICE}")

# # =====================================================
# # 🔒 SEED
# # =====================================================
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# set_seed(SEED)

# # =====================================================
# # 📁 DATASET CHECK
# # =====================================================
# IMG_DIR = os.path.join(DATASET_DIR, "images")
# LBL_DIR = os.path.join(DATASET_DIR, "labels")

# assert os.path.isdir(IMG_DIR), "❌ images/ folder not found"
# assert os.path.isdir(LBL_DIR), "❌ labels/ folder not found"

# # =====================================================
# # 📦 DATASET
# # =====================================================
# class DetectionDataset(Dataset):
#     def __init__(self, img_dir, lbl_dir):
#         self.imgs = sorted(os.listdir(img_dir))
#         self.img_dir = img_dir
#         self.lbl_dir = lbl_dir

#     def __len__(self):
#         return len(self.imgs)

#     def __getitem__(self, idx):
#         name = self.imgs[idx]

#         img = cv2.imread(os.path.join(self.img_dir, name))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = F.to_tensor(img)

#         boxes, labels = [], []
#         label_path = os.path.join(self.lbl_dir, name.rsplit(".", 1)[0] + ".txt")

#         with open(label_path) as f:
#             for line in f:
#                 cls, x1, y1, x2, y2 = map(float, line.split())
#                 boxes.append([x1, y1, x2, y2])
#                 labels.append(int(cls) + 1)  # background = 0

#         target = {
#             "boxes": torch.tensor(boxes, dtype=torch.float32),
#             "labels": torch.tensor(labels, dtype=torch.int64)
#         }

#         return img, target

# def collate_fn(batch):
#     return tuple(zip(*batch))

# loader = DataLoader(
#     DetectionDataset(IMG_DIR, LBL_DIR),
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     num_workers=NUM_WORKERS,
#     collate_fn=collate_fn
# )

# # =====================================================
# # 🧠 MODEL (CORRECT CONFIG)
# # =====================================================
# # 🔥 Small-object anchor (traffic light)
# anchor_generator = AnchorGenerator(
#     sizes=(
#         (8,),    # P2  (very small objects)
#         (16,),   # P3
#         (32,),   # P4
#         (64,),   # P5
#         (128,)   # P6
#     ),
#     aspect_ratios=(
#         (0.5, 1.0, 2.0),
#         (0.5, 1.0, 2.0),
#         (0.5, 1.0, 2.0),
#         (0.5, 1.0, 2.0),
#         (0.5, 1.0, 2.0),
#     )
# )


# # ❗ COCO detector YO'Q
# # ✅ ImageNet backbone BOR
# model = fasterrcnn_resnet50_fpn(
#     weights=None,
#     weights_backbone=ResNet50_Weights.IMAGENET1K_V1,
#     rpn_anchor_generator=anchor_generator
# )

# # 🔥 Dataset:
# # category_id = 0..99  → 100 class
# # + background         → 101
# NUM_CLASSES = 101

# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(
#     in_features,
#     NUM_CLASSES
# )

# model.to(DEVICE)

# # =====================================================
# # ❄️ STAGE 1 — BACKBONE FREEZE
# # =====================================================
# for p in model.backbone.parameters():
#     p.requires_grad = False

# optimizer = torch.optim.AdamW(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=LR_STAGE1
# )

# print("\n🚀 Stage 1 training (backbone frozen)")
# for epoch in range(EPOCHS_STAGE1):
#     model.train()
#     total_loss = 0

#     for imgs, targets in tqdm(loader):
#         imgs = [i.to(DEVICE) for i in imgs]
#         targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

#         loss = sum(model(imgs, targets).values())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch+1}/{EPOCHS_STAGE1} | Loss: {total_loss:.2f}")

# # =====================================================
# # 🔓 STAGE 2 — UNFREEZE layer4
# # =====================================================
# for p in model.backbone.body.layer4.parameters():
#     p.requires_grad = True

# optimizer = torch.optim.AdamW(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=LR_STAGE2
# )

# print("\n🚀 Stage 2 fine-tuning (layer4 unfrozen)")
# for epoch in range(EPOCHS_STAGE2):
#     model.train()
#     total_loss = 0

#     for imgs, targets in tqdm(loader):
#         imgs = [i.to(DEVICE) for i in imgs]
#         targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

#         loss = sum(model(imgs, targets).values())
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     print(f"Epoch {epoch+1}/{EPOCHS_STAGE2} | Loss: {total_loss:.2f}")

# # =====================================================
# # 💾 SAVE MODEL
# # =====================================================
# os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
# torch.save(model.state_dict(), MODEL_SAVE_PATH)

# print("\n✅ Training finished successfully")
# print(f"💾 Model saved to: {MODEL_SAVE_PATH}")


import os
import cv2
import torch
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import ResNet50_Weights


# =========================
# CONFIG
# =========================
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

DATASET_ROOT = "/home/sayfulloh/swm_traffic/rasnet/dataset/dataset_detector"
TRAIN_IMG = os.path.join(DATASET_ROOT, "train/images")
TRAIN_LBL = os.path.join(DATASET_ROOT, "train/labels")
VAL_IMG   = os.path.join(DATASET_ROOT, "val/images")
VAL_LBL   = os.path.join(DATASET_ROOT, "val/labels")

NUM_CLASSES = 2          # background + traffic light
BATCH_SIZE = 10         # kernel crash bo‘lmasligi uchun 8 → 4
EPOCHS = 300
LR = 0.005
NUM_WORKERS = 4

# =========================
# SAVE CONFIG
# =========================
SAVE_DIR = "/home/sayfulloh/swm_traffic/rasnet/models"
MODEL_NAME = "best_detector_0202.pth"
os.makedirs(SAVE_DIR, exist_ok=True)
MODEL_PATH = os.path.join(SAVE_DIR, MODEL_NAME)


# =========================
# DATASET
# =========================
class TrafficLightDataset(Dataset):
    def __init__(self, img_dir, lbl_dir):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        lbl_path = os.path.join(
            self.lbl_dir,
            img_name.rsplit(".", 1)[0] + ".txt"
        )

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"❌ Image not found: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []

        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    cls, x1, y1, x2, y2 = line.strip().split()
                    boxes.append([float(x1), float(y1), float(x2), float(y2)])
                    labels.append(1)

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
        }

        img = F.to_tensor(img)
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


# =========================
# MODEL
# =========================
def get_model():
    anchor_generator = AnchorGenerator(
        sizes=((8, 16), (16, 32), (32, 64), (64, 128), (128, 256)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
        rpn_anchor_generator=anchor_generator,
        num_classes=NUM_CLASSES
    )

    # traffic light kichik obyekt → katta resize
    model.transform.min_size = (1100,)
    model.transform.max_size = 1800

    return model


# =========================
# TRAIN ONE EPOCH
# =========================
def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0

    for images, targets in tqdm(loader, leave=False):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================
# MAIN
# =========================
def main():
    print("🚀 Training started on device:", DEVICE)

    train_ds = TrafficLightDataset(TRAIN_IMG, TRAIN_LBL)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True
    )

    model = get_model().to(DEVICE)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=0.9,
        weight_decay=0.0005
    )

    best_loss = float("inf")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer)

        print(f"[Epoch {epoch+1:03d}/{EPOCHS}] Train Loss: {train_loss:.4f}")

        # save best
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Best model saved → {MODEL_PATH}")

    print("🎉 Training finished successfully!")


if __name__ == "__main__":
    main()