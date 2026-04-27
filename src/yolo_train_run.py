from ultralytics import YOLO
from pathlib import Path
import torch
import os

DATA_YAML  = "data_samples/yolo_data_example.yaml"
MODEL_PATH = "weights/yolo26m.pt"

PROJECT = "runs/yolo26x"
NAME    = "traffic_light_detection"

assert Path(DATA_YAML).exists()
assert Path(MODEL_PATH).exists()

assert torch.cuda.is_available()

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Visible GPU count =", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Visible GPU0 name =", torch.cuda.get_device_name(0))

model = YOLO(MODEL_PATH)

model.train(
    data=DATA_YAML,
    device="1",              # agar CUDA_VISIBLE_DEVICES ishlatilmagan bo'lsa

    project=PROJECT,
    name=NAME,

    # -----------------
    # BASIC TRAIN
    # -----------------
    epochs=200,
    imgsz=1280,
    batch=4,
    nbs=4,                   # LR scaling fix
    workers=4,

    # -----------------
    # OPTIMIZER
    # -----------------
    optimizer="AdamW",
    lr0=0.0006,              # stability uchun biroz past
    lrf=0.01,
    weight_decay=0.01,
    warmup_epochs=3.0,
    cos_lr=True,
    patience=50,

    # -----------------
    # AUGMENT
    # -----------------
    hsv_h=0.015,
    hsv_s=0.70,
    hsv_v=0.50,

    mosaic=0.15,             # small object uchun yumshoqroq
    mixup=0.05,
    copy_paste=0.05,
    close_mosaic=15,

    translate=0.08,
    scale=0.5,
    perspective=0.0005,

    erasing=0.20,

    # -----------------
    # PERFORMANCE
    # -----------------
    amp=True,
    cache="ram",

    # debugging uchun weights saqlash
    save=True,
    save_period=1
)

model.val(
    data=DATA_YAML,
    imgsz=1280,
    device="1"
)

# import os
# # ✅ Multi-GPU uchun ikkala GPUni ham ko‘rinadigan qiling
# # Agar sizda physical GPU0 va GPU1 bo'lsa:
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# from ultralytics import YOLO
# from pathlib import Path
# import torch

# DATA_YAML = "/home/sayfulloh/swm_traffic/all_data/dataset_resplit_4/data.yaml"
# MODEL_PATH = "/home/sayfulloh/swm_traffic/ultralytics/ultralytics/yolo11x.pt"
# CFG_PATH   = "/home/sayfulloh/swm_traffic/ultralytics/ultralytics/cfg/default.yaml"

# PROJECT = "runs/detection"
# NAME = "traffic_exp"
# EPOCHS = 500

# assert Path(DATA_YAML).exists(), f"❌ data.yaml not found: {DATA_YAML}"
# assert Path(CFG_PATH).exists(), f"❌ cfg.yaml not found: {CFG_PATH}"

# assert torch.cuda.is_available(), "❌ CUDA not available"
# print("✅ Visible GPUs:", torch.cuda.device_count())  # -> 2 bo‘lishi kerak

# if torch.cuda.device_count() >= 1:
#     print("✅ GPU0:", torch.cuda.get_device_name(0))
# if torch.cuda.device_count() >= 2:
#     print("✅ GPU1:", torch.cuda.get_device_name(1))

# model = YOLO(MODEL_PATH)

# # ✅ Multi-GPU train (DDP)
# model.train(
#     data=DATA_YAML,
#     cfg=CFG_PATH,
#     epochs=EPOCHS,
#     device="0,1",        # ✅ multi gpu
#     project=PROJECT,
#     name=NAME,
# )

# # ✅ Val (odatda 1 GPU yetadi, tezroq va barqaror)
# model.val(
#     data=DATA_YAML,
#     device=0,            # xohlasangiz "0,1" ham qilsa bo'ladi
# )
