from ultralytics import YOLO
from pathlib import Path
import json
import gc          # <<< SHU QATORNI QO‘SHING
import time


# =========================
# PATHS (shu yerlarni o'zgartirasiz)
# =========================
MODEL_PATH = "/home/sayfulloh/swm_traffic/codes/runs/detection/traffic_exp29/weights/best.pt"

# Bu papka ichida subfolderlar bo'lishi mumkin
SOURCE_ROOT = "/home/sayfulloh/swm_traffic/all_data/unseen/Data_Set/20251203"

OUT_DIR = "/home/sayfulloh/swm_traffic/ultralytics/ultralytics/runs/exp29_model/unseen_0120"

# =========================
# PARAMS
# =========================
IMGSZ = 1536
CONF = 0.4
IOU = 0.6
DEVICE = 1

SAVE_VIS = True
SAVE_TXT = True
SAVE_CONF = True

CHUNK_SIZE = 10          # <<< 500 tadan
SLEEP_SEC = 0.2           # chunklar orasida kichik tanaffus (ixtiyoriy)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

WRITE_JSON = True
JSON_PATH = Path(OUT_DIR) / "predictions.json"


def collect_images(root: Path):
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return sorted(files)


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]


def main():
    model_path = Path(MODEL_PATH)
    root = Path(SOURCE_ROOT)

    assert model_path.exists(), f"❌ Model not found: {model_path}"
    assert root.exists(), f"❌ SOURCE_ROOT not found: {root}"

    files = collect_images(root)
    if not files:
        raise SystemExit(f"❌ No images found under: {root}")

    print(f"✅ Found {len(files)} files under {root}")
    print(f"➡️ Running inference in chunks of {CHUNK_SIZE}...")

    model = YOLO(str(model_path))

    all_json = [] if WRITE_JSON else None
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(files)
    chunk_id = 0

    for batch_files in chunks(files, CHUNK_SIZE):
        chunk_id += 1
        start_idx = (chunk_id - 1) * CHUNK_SIZE + 1
        end_idx = min(chunk_id * CHUNK_SIZE, total)

        print(f"\n=== Chunk {chunk_id}: files {start_idx}-{end_idx} / {total} ===")

        LINE_WIDTH = 1  # <<< qo‘shing (parametrlar qismida)

        results = model.predict(
            source=[str(p) for p in batch_files],
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            project=str(out_dir.parent),
            name=out_dir.name,

            save=SAVE_VIS,
            save_txt=SAVE_TXT,
            save_conf=SAVE_CONF,

            line_width=1,          # <<< bbox ingichka
            show_labels=True,      # <<< class nomi ko‘rinsin
            show_conf=False,       # <<< confidence o‘chiriladi
            show_boxes=True,       # <<< bbox ko‘rinsin

            # Ultralytics versiyaga qarab ishlaydi
            #font_size=0.4,         # <<< label kichikroq (agar support bo‘lsa)

            stream=False,
            verbose=False,
        )

        if WRITE_JSON:
            for r in results:
                item = {"path": str(r.path), "boxes": []}
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().tolist()
                    confs = r.boxes.conf.cpu().tolist()
                    clss = r.boxes.cls.cpu().tolist()
                    names = r.names
                    for b, c, k in zip(xyxy, confs, clss):
                        k = int(k)
                        item["boxes"].append({
                            "cls_id": k,
                            "cls_name": names.get(k, str(k)),
                            "conf": float(c),
                            "xyxy": [float(x) for x in b]
                        })
                all_json.append(item)

        # free memory / file handles aggressively
        del results
        gc.collect()
        time.sleep(SLEEP_SEC)

    if WRITE_JSON:
        JSON_PATH.write_text(json.dumps(all_json, indent=2), encoding="utf-8")
        print("\n🧾 JSON saved:", JSON_PATH)

    print("\n✅ Done. Outputs:", OUT_DIR)


if __name__ == "__main__":
    main()
