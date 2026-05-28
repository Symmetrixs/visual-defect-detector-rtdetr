import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from config import LABEL2ID, DEFECT_CLASSES, ALIAS_MAP


def resolve_class(raw_name: str):
    name = raw_name.lower().strip().replace("-", " ").replace("_", " ")
    name_underscore = raw_name.lower().strip().replace(" ", "_").replace("-", "_")
    if name_underscore in ALIAS_MAP:
        return ALIAS_MAP[name_underscore]
    if name in ALIAS_MAP:
        return ALIAS_MAP[name]
    for alias, target in ALIAS_MAP.items():
        if alias in name or name in alias:
            return target
    return None


class DefectDataset(Dataset):
    def __init__(self, image_dir: str, annotation_file: str, processor):
        self.image_dir = image_dir
        self.processor = processor

        with open(annotation_file) as f:
            coco = json.load(f)

        self.images_info = {img["id"]: img for img in coco["images"]}

        self.cat_map = {}
        skipped_cats = []
        for cat in coco.get("categories", []):
            resolved = resolve_class(cat["name"])
            if resolved and resolved in LABEL2ID:
                self.cat_map[cat["id"]] = LABEL2ID[resolved]
            else:
                skipped_cats.append(cat["name"])

        if skipped_cats:
            print(f"  Skipped categories (not in 7 classes): {skipped_cats}")

        self.image_ids = []
        self.anns_by_image = {}

        for ann in coco.get("annotations", []):
            cat_id = ann.get("category_id")
            if cat_id not in self.cat_map:
                continue
            img_id = ann["image_id"]
            if img_id not in self.anns_by_image:
                self.anns_by_image[img_id] = []
                self.image_ids.append(img_id)
            x, y, w, h = ann["bbox"]
            self.anns_by_image[img_id].append({
                "bbox":        [x, y, x + w, y + h],
                "category_id": self.cat_map[cat_id],
            })

        self.image_ids = sorted(set(self.image_ids))
        print(f"  Loaded {len(self.image_ids)} images | {sum(len(v) for v in self.anns_by_image.values())} annotations")
        self._print_class_distribution()

    def _print_class_distribution(self):
        from collections import Counter
        from config import ID2LABEL
        counts = Counter()
        for anns in self.anns_by_image.values():
            for ann in anns:
                counts[ID2LABEL[ann["category_id"]]] += 1
        print("  Class distribution:")
        total = sum(counts.values())
        for cls in DEFECT_CLASSES:
            count = counts.get(cls, 0)
            bar = "█" * int(count / max(total, 1) * 30)
            print(f"    {cls:<20}: {count:>5}  {bar}")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id   = self.image_ids[idx]
        img_info = self.images_info[img_id]
        img_path = os.path.join(self.image_dir, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")
        anns  = self.anns_by_image.get(img_id, [])

        encoding = self.processor(
            images=image,
            annotations={"image_id": img_id, "annotations": anns},
            return_tensors="pt",
        )
        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels":       encoding["labels"][0],
        }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels":       [b["labels"] for b in batch],
    }
