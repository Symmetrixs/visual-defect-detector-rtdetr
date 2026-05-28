import os
import json
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from config import LABEL2ID, ID2LABEL, DEFECT_CLASSES, InferConfig


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RT-DETRv2 VIDA model")
    p.add_argument("--model_path", required=True, help="Local path or HuggingFace repo ID")
    p.add_argument("--val_images", required=True)
    p.add_argument("--val_ann",    required=True)
    p.add_argument("--conf",       type=float, default=InferConfig.confidence_threshold)
    return p.parse_args()


def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model_path}")
    processor = RTDetrImageProcessor.from_pretrained(args.model_path)
    model     = RTDetrV2ForObjectDetection.from_pretrained(args.model_path).to(device)
    model.eval()

    with open(args.val_ann) as f:
        coco = json.load(f)

    images_info = {img["id"]: img for img in coco["images"]}

    cat_map = {}
    for cat in coco.get("categories", []):
        name = cat["name"].lower().strip().replace(" ", "_").replace("-", "_")
        for cls in DEFECT_CLASSES:
            if name == cls or name in cls or cls in name:
                cat_map[cat["id"]] = LABEL2ID[cls]
                break

    gt_by_img = {}
    for ann in coco["annotations"]:
        if ann["category_id"] not in cat_map:
            continue
        img_id = ann["image_id"]
        if img_id not in gt_by_img:
            gt_by_img[img_id] = {"boxes": [], "labels": []}
        x, y, w, h = ann["bbox"]
        gt_by_img[img_id]["boxes"].append([x, y, x + w, y + h])
        gt_by_img[img_id]["labels"].append(cat_map[ann["category_id"]])

    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    preds, targets = [], []

    print(f"Running inference on {len(images_info)} images (conf ≥ {args.conf})...")

    for img_id, img_info in tqdm(images_info.items()):
        img_path = os.path.join(args.val_images, img_info["file_name"])
        if not os.path.exists(img_path):
            continue

        image  = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, threshold=args.conf, target_sizes=target_sizes
        )[0]

        preds.append({
            "boxes":  results["boxes"].cpu(),
            "scores": results["scores"].cpu(),
            "labels": results["labels"].cpu(),
        })

        gt = gt_by_img.get(img_id, {"boxes": [], "labels": []})
        targets.append({
            "boxes":  torch.tensor(gt["boxes"],  dtype=torch.float32) if gt["boxes"]  else torch.zeros((0, 4)),
            "labels": torch.tensor(gt["labels"], dtype=torch.long)    if gt["labels"] else torch.zeros(0, dtype=torch.long),
        })

    metric.update(preds, targets)
    r = metric.compute()

    print("\n" + "=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(f"  mAP@50        : {r['map_50'].item():.4f}  ({r['map_50'].item()*100:.1f}%)")
    print(f"  mAP@50:95     : {r['map'].item():.4f}  ({r['map'].item()*100:.1f}%)")
    print(f"  mAR@100       : {r['mar_100'].item():.4f}")
    print(f"  mAP (small)   : {r['map_small'].item():.4f}")
    print(f"  mAP (medium)  : {r['map_medium'].item():.4f}")
    print(f"  mAP (large)   : {r['map_large'].item():.4f}")
    print("-" * 55)
    print("  Per-class AP@50:")
    if "map_per_class" in r and r["map_per_class"] is not None:
        for i, ap in enumerate(r["map_per_class"]):
            label = ID2LABEL.get(i, f"class_{i}")
            bar   = "█" * int(ap.item() * 30)
            print(f"    {label:<20} {ap.item():.4f}  {bar}")
    print("=" * 55)


if __name__ == "__main__":
    main()
