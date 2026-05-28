import sys
import os
import argparse
import torch
from PIL import Image, ImageDraw
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from config import ID2LABEL, DEFECT_COLORS, InferConfig


def parse_args():
    p = argparse.ArgumentParser(description="Run RT-DETRv2 inference on an image or folder")
    p.add_argument("model_path",  help="Local model path or HuggingFace repo ID")
    p.add_argument("input",       help="Image file or folder of images")
    p.add_argument("--conf",      type=float, default=InferConfig.confidence_threshold)
    p.add_argument("--output",    default="./outputs/infer", help="Output directory for visualized images")
    p.add_argument("--no_save",   action="store_true", help="Print results only, do not save images")
    return p.parse_args()


def load_model(model_path: str):
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    processor = RTDetrImageProcessor.from_pretrained(model_path)
    model     = RTDetrV2ForObjectDetection.from_pretrained(model_path).to(device)
    model.eval()
    return model, processor, device


def predict_image(model, processor, device, image: Image.Image, conf: float):
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, threshold=conf, target_sizes=target_sizes
    )[0]
    detections = []
    for score, label_id, box in zip(results["scores"], results["labels"], results["boxes"]):
        label = ID2LABEL.get(label_id.item(), "unknown")
        x1, y1, x2, y2 = box.tolist()
        detections.append({
            "label": label,
            "score": round(score.item(), 4),
            "box":   {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
        })
    detections.sort(key=lambda d: -d["score"])
    return detections


def visualize(image: Image.Image, detections: list) -> Image.Image:
    img  = image.copy()
    draw = ImageDraw.Draw(img)
    for det in detections:
        box   = det["box"]
        label = det["label"]
        score = det["score"]
        color = DEFECT_COLORS.get(label, "#6366f1")
        draw.rectangle([box["xmin"], box["ymin"], box["xmax"], box["ymax"]], outline=color, width=3)
        text = f"{label.replace('_', ' ')} {score:.0%}"
        tw   = len(text) * 7
        draw.rectangle([box["xmin"], box["ymin"] - 22, box["xmin"] + tw + 6, box["ymin"]], fill=color)
        draw.text((box["xmin"] + 3, box["ymin"] - 18), text, fill="white")
    return img


def process_single(model, processor, device, img_path: str, conf: float, output_dir: str, no_save: bool):
    image = Image.open(img_path).convert("RGB")
    dets  = predict_image(model, processor, device, image, conf)

    name = os.path.basename(img_path)
    print(f"\n{name}")
    if not dets:
        print("  No defects detected.")
    else:
        print(f"  {len(dets)} defect(s):")
        for d in dets:
            b = d["box"]
            print(f"  [{d['label']:<20}] conf={d['score']:.2%}  ({b['xmin']:.0f},{b['ymin']:.0f}) → ({b['xmax']:.0f},{b['ymax']:.0f})")

    if not no_save and dets:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, name)
        visualize(image, dets).save(out_path)
        print(f"  Saved → {out_path}")

    return dets


def main():
    args = parse_args()
    print(f"Loading model from: {args.model_path}")
    model, processor, device = load_model(args.model_path)
    print(f"Device: {device}  |  Confidence threshold: {args.conf}")

    if os.path.isdir(args.input):
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = [f for f in os.listdir(args.input) if os.path.splitext(f)[1].lower() in exts]
        files.sort()
        print(f"Processing {len(files)} images in {args.input}...")
        total_dets = 0
        for fname in files:
            dets = process_single(model, processor, device, os.path.join(args.input, fname), args.conf, args.output, args.no_save)
            total_dets += len(dets)
        print(f"\nTotal: {total_dets} detection(s) across {len(files)} images.")
    else:
        process_single(model, processor, device, args.input, args.conf, args.output, args.no_save)


if __name__ == "__main__":
    main()
