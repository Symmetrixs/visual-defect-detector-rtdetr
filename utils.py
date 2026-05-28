import os
import json
import shutil
import zipfile
from pathlib import Path
from collections import Counter
from config import ALIAS_MAP, LABEL2ID, ID2LABEL, DEFECT_CLASSES

SCRIPT_DIR  = Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR.parent / "dataset"
MERGED_DIR  = DATASET_DIR / "merged"
OUTPUT_DIR  = SCRIPT_DIR / "outputs" / "vida-rtdetrv2"

EXPECTED_DATASETS = [
    "Concrete defect detection.v1i.coco",
    "detr_crack_dataset.v1i.coco",
]


def get_gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
            return f"{name} — {vram:.1f} GB VRAM"
        return "No GPU detected (CPU only — training will be slow)"
    except Exception:
        return "GPU info unavailable"


def resolve_class(raw: str):
    name   = raw.lower().strip().replace("-", " ").replace("_", " ")
    name_u = raw.lower().strip().replace(" ", "_").replace("-", "_")
    if name_u in ALIAS_MAP: return ALIAS_MAP[name_u]
    if name   in ALIAS_MAP: return ALIAS_MAP[name]
    for alias, target in ALIAS_MAP.items():
        if alias in name or name in alias:
            return target
    return None


def find_datasets():
    results = []
    if not DATASET_DIR.exists():
        return results, [f"Dataset directory not found: {DATASET_DIR}"]

    errors = []
    for name in EXPECTED_DATASETS:
        folder = DATASET_DIR / name
        zp     = DATASET_DIR / f"{name}.zip"
        if folder.exists():
            results.append(folder)
        elif zp.exists():
            try:
                with zipfile.ZipFile(zp, "r") as z:
                    z.extractall(DATASET_DIR)
                if folder.exists():
                    results.append(folder)
                else:
                    matches = [f for f in DATASET_DIR.iterdir() if name.split(".")[0].lower() in f.name.lower() and f.is_dir()]
                    if matches:
                        results.append(matches[0])
                    else:
                        errors.append(f"Extracted {name}.zip but could not find output folder")
            except Exception as e:
                errors.append(f"Failed to extract {name}.zip: {e}")
        else:
            errors.append(f"Not found: {name}")

    return results, errors


def validate_dataset(ds_path: Path):
    report = []
    issues = []
    for split in ("train", "valid"):
        split_dir  = ds_path / split
        images_dir = split_dir / "images"
        ann_file   = split_dir / "_annotations.coco.json"

        if not split_dir.exists():
            issues.append(f"Missing {split}/ folder")
            continue

        imgs = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.webp"))
        n_imgs = len(imgs)

        if not images_dir.exists():
            issues.append(f"Missing {split}/images/")
        elif n_imgs == 0:
            issues.append(f"{split}/images/ is empty")

        if not ann_file.exists():
            issues.append(f"Missing {split}/_annotations.coco.json")
        else:
            try:
                with open(ann_file) as f:
                    coco = json.load(f)
                cats    = [c["name"] for c in coco.get("categories", [])]
                n_anns  = len(coco.get("annotations", []))
                n_coco  = len(coco.get("images", []))
                report.append(f"{split}: {n_coco} images | {n_anns} annotations | classes: {cats}")
            except json.JSONDecodeError:
                issues.append(f"{split}/_annotations.coco.json is corrupted")

    return report, issues


def get_class_distribution(ann_file: str):
    counts = {c: 0 for c in DEFECT_CLASSES}
    skipped = []
    try:
        with open(ann_file) as f:
            coco = json.load(f)
        cat_map = {}
        for cat in coco.get("categories", []):
            r = resolve_class(cat["name"])
            if r and r in LABEL2ID:
                cat_map[cat["id"]] = r
            else:
                skipped.append(cat["name"])
        for ann in coco.get("annotations", []):
            cls = cat_map.get(ann.get("category_id"))
            if cls:
                counts[cls] += 1
    except Exception:
        pass
    return counts, skipped


def do_merge(datasets: list, progress_fn=None):
    log_lines = []

    def log(msg):
        log_lines.append(msg)
        if progress_fn:
            progress_fn("\n".join(log_lines))

    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)

    for split in ("train", "valid"):
        out_img_dir = MERGED_DIR / split / "images"
        out_ann     = MERGED_DIR / split / "_annotations.coco.json"
        out_img_dir.mkdir(parents=True, exist_ok=True)

        merged_images = []
        merged_anns   = []
        img_offset    = 0
        ann_offset    = 0
        class_counts  = {c: 0 for c in DEFECT_CLASSES}
        skipped_cats  = set()

        for ds_idx, ds_path in enumerate(datasets):
            ann_file = ds_path / split / "_annotations.coco.json"
            img_dir  = ds_path / split / "images"

            if not ann_file.exists():
                log(f"[{ds_path.name}] No {split} split — skipped")
                continue

            with open(ann_file) as f:
                coco = json.load(f)

            cat_map = {}
            for cat in coco.get("categories", []):
                r = resolve_class(cat["name"])
                if r and r in LABEL2ID:
                    cat_map[cat["id"]] = LABEL2ID[r]
                else:
                    skipped_cats.add(cat["name"])

            orig_to_new = {}
            copied = 0

            for img in coco.get("images", []):
                src = img_dir / img["file_name"]
                if not src.exists():
                    alts = list(img_dir.glob(f"*{Path(img['file_name']).stem}*"))
                    if not alts:
                        continue
                    src = alts[0]

                new_id   = img["id"] + img_offset
                new_name = f"d{ds_idx}_{img['file_name']}"
                shutil.copy2(src, out_img_dir / new_name)
                orig_to_new[img["id"]] = new_id
                merged_images.append({"id": new_id, "file_name": new_name,
                                      "width": img.get("width", 640), "height": img.get("height", 640)})
                copied += 1

            for ann in coco.get("annotations", []):
                if ann["category_id"] not in cat_map:     continue
                if ann["image_id"]    not in orig_to_new: continue
                new_cat = cat_map[ann["category_id"]]
                merged_anns.append({"id": ann["id"] + ann_offset, "image_id": orig_to_new[ann["image_id"]],
                                    "category_id": new_cat, "bbox": ann["bbox"],
                                    "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]), "iscrowd": 0})
                class_counts[ID2LABEL[new_cat]] += 1

            max_img = max((i["id"] for i in coco.get("images", [])), default=0)
            max_ann = max((a["id"] for a in coco.get("annotations", [])), default=0)
            img_offset += max_img + 1
            ann_offset += max_ann + 1
            log(f"[{ds_path.name}] {split}: {copied} images copied")

        cats = [{"id": i, "name": c} for i, c in ID2LABEL.items()]
        with open(out_ann, "w") as f:
            json.dump({"images": merged_images, "annotations": merged_anns, "categories": cats}, f)

        total = sum(class_counts.values())
        log(f"\n{split.upper()}: {len(merged_images)} images | {len(merged_anns)} annotations")
        for cls in DEFECT_CLASSES:
            count = class_counts[cls]
            bar   = "█" * int(count / max(total, 1) * 20)
            flag  = " ← LOW" if count < 100 else ""
            log(f"  {cls:<20}: {count:>5}  {bar}{flag}")

        if skipped_cats:
            log(f"  Skipped: {skipped_cats}")

    log("\nMerge complete.")
    return "\n".join(log_lines)


def get_latest_checkpoint():
    if not OUTPUT_DIR.exists():
        return str(OUTPUT_DIR)
    checkpoints = sorted(OUTPUT_DIR.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0)
    if checkpoints:
        return str(checkpoints[-1])
    return str(OUTPUT_DIR)


def get_merged_paths():
    return (
        str(MERGED_DIR / "train" / "images"),
        str(MERGED_DIR / "train" / "_annotations.coco.json"),
        str(MERGED_DIR / "valid" / "images"),
        str(MERGED_DIR / "valid" / "_annotations.coco.json"),
    )


def merged_exists():
    ti, ta, vi, va = get_merged_paths()
    return all(Path(p).exists() for p in [ti, ta, vi, va])
