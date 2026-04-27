import os, json, cv2, shutil, random, hashlib
from pathlib import Path
from collections import defaultdict
import pandas as pd

# ================= CONFIG =================
ROOT_DIR = "data_samples/coco_dataset"
OUTPUT_DIR = "results/tl_data_processed"

MIN_BOX = (3, 4)
MAX_BOX = (300, 300)
MAX_AR = 6.0
TRAIN_RATIO = 0.8

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")

DUPLICATE_COMPARE_SIZE = (320, 320)
DUPLICATE_THRESHOLD = 5


# ================= HELPERS =================
def mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def find_images(root=ROOT_DIR):
    print("[INFO] Searching images...")

    images = {}

    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(IMG_EXT):
                images[f] = os.path.join(r, f)

    print(f"[INFO] Found {len(images)} images")
    return images


def img_info(p):
    im = cv2.imread(p)
    if im is None:
        return None

    h, w = im.shape[:2]
    return w, h


def clamp_bbox(x, y, w, h, W, H):
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)

    new_w = x2 - x1
    new_h = y2 - y1

    if new_w <= 0 or new_h <= 0:
        return None

    return [x1, y1, new_w, new_h]


def is_similar(img1, img2, thr=DUPLICATE_THRESHOLD):
    """
    Resolution-independent near-duplicate detection.
    This does NOT affect training image size.
    It only compares resized preview images.
    """
    if img1 is None or img2 is None:
        return False

    img1_r = cv2.resize(img1, DUPLICATE_COMPARE_SIZE)
    img2_r = cv2.resize(img2, DUPLICATE_COMPARE_SIZE)

    diff = cv2.absdiff(img1_r, img2_r)
    return diff.mean() < thr


def coco_to_yolo(bbox, W, H):
    x, y, w, h = bbox

    x_center = (x + w / 2) / W
    y_center = (y + h / 2) / H
    bw = w / W
    bh = h / H

    return [x_center, y_center, bw, bh]


def quality_score(w, h, W, H):
    """
    Simple annotation quality score.
    Higher score = better training sample.
    """

    size_score = min(1.0, (w * h) / (W * H) * 20)

    aspect = max(w / (h + 1e-6), h / (w + 1e-6))
    aspect_score = max(0.0, 1.0 - aspect / 10.0)

    # This is not object center; it is a soft regularizer only.
    center_score = 0.5

    score = 0.6 * size_score + 0.3 * aspect_score + 0.1 * center_score
    return float(score)


def safe_copy(src, dst):
    mkdir(Path(dst).parent)
    shutil.copy2(src, dst)


# ================= CUSTOM MULTI JSON → COCO =================
def convert_multi_json_to_coco(root):
    print("[INFO] Converting MULTI JSON → COCO")

    images = []
    annotations = []
    categories = {}

    image_name_to_id = {}
    cat_id = 1
    ann_id = 1
    img_id = 1

    total_json = 0
    used_json = 0
    skipped_labels = 0

    for r, _, files in os.walk(root):
        for f in files:
            if not f.endswith(".json"):
                continue

            total_json += 1
            json_path = os.path.join(r, f)

            try:
                with open(json_path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
            except Exception:
                continue

            if not isinstance(data, dict):
                continue

            if "labels" not in data:
                continue

            used_json += 1

            # -------- detect image name --------
            img_name = None

            resources = data.get("resources")

            if isinstance(resources, str):
                img_name = resources

            elif isinstance(resources, dict):
                img_name = (
                    resources.get("file_name")
                    or resources.get("filename")
                    or resources.get("name")
                    or resources.get("path")
                )

            elif isinstance(resources, list) and len(resources) > 0:
                first_res = resources[0]
                if isinstance(first_res, dict):
                    img_name = (
                        first_res.get("file_name")
                        or first_res.get("filename")
                        or first_res.get("name")
                        or first_res.get("path")
                    )
                elif isinstance(first_res, str):
                    img_name = first_res

            if img_name:
                img_name = os.path.basename(img_name)
            else:
                img_name = f.replace(".json", ".jpg")

            if img_name not in image_name_to_id:
                image_name_to_id[img_name] = img_id

                images.append({
                    "id": img_id,
                    "file_name": img_name,
                    "width": 1920,
                    "height": 1080
                })

                img_id += 1

            current_img_id = image_name_to_id[img_name]

            labels = data.get("labels", [])

            if not isinstance(labels, list):
                continue

            for label in labels:
                if not isinstance(label, dict):
                    skipped_labels += 1
                    continue

                cls = (
                    label.get("class")
                    or label.get("className")
                    or label.get("category")
                    or label.get("category_name")
                    or label.get("label")
                    or "unknown"
                )

                if cls not in categories:
                    categories[cls] = cat_id
                    cat_id += 1

                bbox = (
                    label.get("bbox")
                    or label.get("box")
                    or label.get("bounding_box")
                    or label.get("BOUNDING_BOX")
                )

                # possible BOUNDING_BOX dict format
                if isinstance(bbox, dict):
                    if all(k in bbox for k in ["x", "y", "width", "height"]):
                        bbox = [bbox["x"], bbox["y"], bbox["width"], bbox["height"]]
                    elif all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
                        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                        bbox = [x1, y1, x2 - x1, y2 - y1]

                if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
                    skipped_labels += 1
                    continue

                try:
                    x, y, w, h = map(float, bbox)
                except Exception:
                    skipped_labels += 1
                    continue

                annotations.append({
                    "id": ann_id,
                    "image_id": current_img_id,
                    "category_id": categories[cls],
                    "bbox": [x, y, w, h],
                    "area": max(0.0, w) * max(0.0, h),
                    "iscrowd": 0
                })

                ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": v, "name": k} for k, v in categories.items()]
    }

    out = "results/merged_coco.json"

    with open(out, "w", encoding="utf-8") as fp:
        json.dump(coco, fp, indent=2, ensure_ascii=False)

    print(f"[INFO] JSON files found: {total_json}")
    print(f"[INFO] JSON files used: {used_json}")
    print(f"[INFO] COCO created: {len(images)} images, {len(annotations)} annotations")
    print(f"[INFO] Skipped labels: {skipped_labels}")
    print(f"[INFO] Saved merged COCO: {out}")

    return out


# ================= MAIN =================
def main():
    random.seed(42)

    output = Path(OUTPUT_DIR)
    reports = output / "reports"
    damaged_dir = output / "damaged"
    yolo = output / "yolo"

    mkdir(output)
    mkdir(reports)
    mkdir(damaged_dir)

    for d in [
        damaged_dir / "missing_image",
        damaged_dir / "corrupt",
        damaged_dir / "bad_bbox",
        damaged_dir / "near_duplicate",
        damaged_dir / "low_quality",
    ]:
        mkdir(d)

    # 1. Find all images
    imgs = find_images(ROOT_DIR)

    # 2. Convert all frame JSON files into one COCO
    coco_path = convert_multi_json_to_coco(ROOT_DIR)

    with open(coco_path, "r", encoding="utf-8") as fp:
        coco = json.load(fp)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    print(f"[INFO] Loaded COCO images: {len(images)}")
    print(f"[INFO] Loaded COCO annotations: {len(annotations)}")
    print(f"[INFO] Loaded categories: {len(categories)}")

    cat_map = {c["id"]: i for i, c in enumerate(categories)}

    anns_by_img = defaultdict(list)
    for ann in annotations:
        anns_by_img[ann["image_id"]].append(ann)

    clean_records = []
    weak_records = []
    damaged_records = []

    final_images = []
    final_annotations = []

    prev_img_small = None

    # 3. Validate images and annotations
    for img in images:
        img_id = img["id"]
        name = img["file_name"]

        if name not in imgs:
            damaged_records.append({
                "type": "missing_image",
                "file": name,
                "image_id": img_id,
                "reason": "Image referenced by JSON but not found on disk"
            })
            continue

        img_path = imgs[name]
        im = cv2.imread(img_path)

        if im is None:
            damaged_records.append({
                "type": "corrupt",
                "file": name,
                "image_id": img_id,
                "reason": "Image cannot be opened by OpenCV"
            })
            continue

        H, W = im.shape[:2]
        img["width"] = W
        img["height"] = H

        # Near duplicate check using resized preview
        im_small = cv2.resize(im, DUPLICATE_COMPARE_SIZE)

        if prev_img_small is not None:
            diff = cv2.absdiff(im_small, prev_img_small)
            if diff.mean() < DUPLICATE_THRESHOLD:
                damaged_records.append({
                    "type": "near_duplicate",
                    "file": name,
                    "image_id": img_id,
                    "reason": f"Near duplicate of previous image, diff_mean={diff.mean():.4f}"
                })

                safe_copy(img_path, damaged_dir / "near_duplicate" / name)
                continue

        prev_img_small = im_small

        valid_for_image = []

        for ann in anns_by_img.get(img_id, []):
            ann_id = ann["id"]

            if ann.get("category_id") not in cat_map:
                damaged_records.append({
                    "type": "unknown_category",
                    "file": name,
                    "image_id": img_id,
                    "annotation_id": ann_id,
                    "reason": f"Unknown category_id: {ann.get('category_id')}"
                })
                continue

            bbox = ann.get("bbox")

            if not bbox or len(bbox) != 4:
                damaged_records.append({
                    "type": "invalid_bbox_format",
                    "file": name,
                    "image_id": img_id,
                    "annotation_id": ann_id,
                    "reason": f"Invalid bbox: {bbox}"
                })
                continue

            try:
                x, y, w, h = map(float, bbox)
            except Exception:
                damaged_records.append({
                    "type": "invalid_bbox_value",
                    "file": name,
                    "image_id": img_id,
                    "annotation_id": ann_id,
                    "reason": f"Cannot parse bbox: {bbox}"
                })
                continue

            if w <= 0 or h <= 0:
                damaged_records.append({
                    "type": "bad_bbox",
                    "file": name,
                    "image_id": img_id,
                    "annotation_id": ann_id,
                    "reason": f"bbox width/height <= 0: {bbox}"
                })
                continue

            fixed = clamp_bbox(x, y, w, h, W, H)

            if fixed is None:
                damaged_records.append({
                    "type": "bbox_outside_image",
                    "file": name,
                    "image_id": img_id,
                    "annotation_id": ann_id,
                    "reason": f"bbox fully outside image: {bbox}"
                })
                safe_copy(img_path, damaged_dir / "bad_bbox" / name)
                continue

            x, y, w, h = fixed

            if w < MIN_BOX[0] or h < MIN_BOX[1]:
                damaged_records.append({
                    "type": "too_small_bbox",
                    "file": name,
                    "image_id": img_id,
                    "annotation_id": ann_id,
                    "reason": f"bbox too small after fixing: {[x, y, w, h]}"
                })
                continue

            if w > MAX_BOX[0] or h > MAX_BOX[1]:
                damaged_records.append({
                    "type": "too_large_bbox",
                    "file": name,
                    "image_id": img_id,
                    "annotation_id": ann_id,
                    "reason": f"bbox too large: {[x, y, w, h]}"
                })
                continue

            ar = max(w / (h + 1e-6), h / (w + 1e-6))

            if ar > MAX_AR:
                damaged_records.append({
                    "type": "bad_aspect_ratio",
                    "file": name,
                    "image_id": img_id,
                    "annotation_id": ann_id,
                    "reason": f"aspect_ratio={ar:.2f}, bbox={[x, y, w, h]}"
                })
                continue

            score = quality_score(w, h, W, H)

            ann["bbox"] = [x, y, w, h]
            ann["area"] = w * h
            ann["score"] = score

            record = {
                "file": name,
                "image_id": img_id,
                "annotation_id": ann_id,
                "category_id": ann["category_id"],
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "area": w * h,
                "aspect_ratio": ar,
                "score": score
            }

            if score >= 0.6:
                clean_records.append(record)
                valid_for_image.append(ann)

            elif score >= 0.3:
                weak_records.append(record)
                valid_for_image.append(ann)

            else:
                damaged_records.append({
                    "type": "low_quality",
                    "file": name,
                    "image_id": img_id,
                    "annotation_id": ann_id,
                    "reason": f"quality_score={score:.4f}, bbox={[x, y, w, h]}"
                })

        if valid_for_image:
            final_images.append(img)
            final_annotations.extend(valid_for_image)
        else:
            damaged_records.append({
                "type": "no_valid_annotation",
                "file": name,
                "image_id": img_id,
                "reason": "Image has no valid annotations after filtering"
            })

    print(f"[INFO] Final images with valid annotations: {len(final_images)}")
    print(f"[INFO] Final valid annotations: {len(final_annotations)}")
    print(f"[INFO] Clean annotations: {len(clean_records)}")
    print(f"[INFO] Weak annotations: {len(weak_records)}")
    print(f"[INFO] Damaged records: {len(damaged_records)}")

    # 4. Export clean COCO
    clean_coco = {
        "images": final_images,
        "annotations": final_annotations,
        "categories": categories
    }

    with open(output / "cleaned_coco.json", "w", encoding="utf-8") as fp:
        json.dump(clean_coco, fp, indent=2, ensure_ascii=False)

    # 5. YOLO export: use clean annotations only score >= 0.6
    for split in ["train", "val"]:
        mkdir(yolo / "images" / split)
        mkdir(yolo / "labels" / split)

    final_anns_by_img = defaultdict(list)
    for ann in final_annotations:
        if ann.get("score", 0) >= 0.6:
            final_anns_by_img[ann["image_id"]].append(ann)

    yolo_items = []

    for img in final_images:
        img_id = img["id"]
        name = img["file_name"]

        if img_id not in final_anns_by_img:
            continue

        if name not in imgs:
            continue

        img_path = imgs[name]
        W, H = img_info(img_path)

        label_lines = []

        for ann in final_anns_by_img[img_id]:
            box = coco_to_yolo(ann["bbox"], W, H)

            if not all(0 <= v <= 1 for v in box):
                damaged_records.append({
                    "type": "invalid_yolo_box",
                    "file": name,
                    "image_id": img_id,
                    "annotation_id": ann["id"],
                    "reason": f"YOLO box invalid: {box}"
                })
                continue

            cls = cat_map[ann["category_id"]]
            label_lines.append(
                f"{cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
            )

        if label_lines:
            yolo_items.append((img_path, name, label_lines))

    random.shuffle(yolo_items)
    split_idx = int(len(yolo_items) * TRAIN_RATIO)

    train_items = yolo_items[:split_idx]
    val_items = yolo_items[split_idx:]

    for split, items in [("train", train_items), ("val", val_items)]:
        for img_path, name, lines in items:
            safe_copy(img_path, yolo / "images" / split / name)

            label_name = Path(name).stem + ".txt"
            with open(yolo / "labels" / split / label_name, "w", encoding="utf-8") as fp:
                fp.write("\n".join(lines))

    # 6. data.yaml
    with open(yolo / "data.yaml", "w", encoding="utf-8") as fp:
        fp.write(f"path: {yolo}\n")
        fp.write("train: images/train\n")
        fp.write("val: images/val\n")
        fp.write(f"nc: {len(categories)}\n")
        fp.write("names:\n")

        for i, c in enumerate(categories):
            fp.write(f"  {i}: {c['name']}\n")

    # 7. Reports
    pd.DataFrame(clean_records).to_csv(reports / "clean.csv", index=False)
    pd.DataFrame(weak_records).to_csv(reports / "weak.csv", index=False)
    pd.DataFrame(damaged_records).to_csv(reports / "damaged.csv", index=False)

    summary = {
        "total_images_in_coco": len(images),
        "total_annotations_in_coco": len(annotations),
        "categories": len(categories),
        "final_images": len(final_images),
        "final_annotations": len(final_annotations),
        "clean_annotations": len(clean_records),
        "weak_annotations": len(weak_records),
        "damaged_records": len(damaged_records),
        "yolo_train_images": len(train_items),
        "yolo_val_images": len(val_items),
    }

    with open(reports / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    with open(reports / "summary.md", "w", encoding="utf-8") as fp:
        fp.write("# Dataset Processing Summary\n\n")
        for k, v in summary.items():
            fp.write(f"- {k}: {v}\n")

    print("\n✅ DONE - FINAL STRONG PIPELINE")
    print(f"OUTPUT: {OUTPUT_DIR}")
    print(f"YOLO: {yolo}")
    print(f"REPORTS: {reports}")


if __name__ == "__main__":
    main()