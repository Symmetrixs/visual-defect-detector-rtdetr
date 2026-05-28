import argparse
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from huggingface_hub import HfApi
from config import ID2LABEL, LABEL2ID, DEFECT_CLASSES, DEFECT_DESCRIPTIONS


MODEL_CARD_TEMPLATE = """---
language: en
license: apache-2.0
tags:
  - object-detection
  - building-defects
  - rt-detrv2
  - computer-vision
  - utem
datasets:
  - custom
model-index:
  - name: VIDA RT-DETRv2 Building Defect Detector
    results: []
---

# VIDA — RT-DETRv2 Building Defect Detector

Fine-tuned **RT-DETRv2** for detecting building surface defects on the UTeM campus.
Part of the VIDA (Visual Infrastructure Defect Analyzer) FYP project.

## Detected Classes ({num_classes})

| Class | Description |
|-------|-------------|
{class_table}

## Usage

```python
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
from PIL import Image
import torch

model_id = "{model_id}"
processor = RTDetrImageProcessor.from_pretrained(model_id)
model = RTDetrV2ForObjectDetection.from_pretrained(model_id)
model.eval()

image = Image.open("building.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_object_detection(
    outputs, threshold=0.3, target_sizes=torch.tensor([image.size[::-1]])
)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    print(f"{{model.config.id2label[label.item()]}}: {{score:.2%}} | {{box.tolist()}}")
```

## Training

Base model: `PekingU/rtdetr_v2_r50vd`
Framework: HuggingFace Transformers ≥ 4.45
"""


def parse_args():
    p = argparse.ArgumentParser(description="Push VIDA model to HuggingFace Hub")
    p.add_argument("model_path",  help="Local path to trained model")
    p.add_argument("hub_model_id", help="HuggingFace repo ID (e.g. YourName/vida-rtdetrv2-defect)")
    p.add_argument("--private",   action="store_true", help="Make the repo private")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading model from: {args.model_path}")
    processor = RTDetrImageProcessor.from_pretrained(args.model_path)
    model     = RTDetrV2ForObjectDetection.from_pretrained(args.model_path)

    model.config.id2label   = ID2LABEL
    model.config.label2id   = LABEL2ID
    model.config.num_labels = len(DEFECT_CLASSES)

    class_table = "\n".join(
        f"| `{cls}` | {DEFECT_DESCRIPTIONS[cls]} |"
        for cls in DEFECT_CLASSES
    )
    model_card = MODEL_CARD_TEMPLATE.format(
        num_classes=len(DEFECT_CLASSES),
        class_table=class_table,
        model_id=args.hub_model_id,
    )

    print(f"Pushing to: {args.hub_model_id}")
    model.push_to_hub(args.hub_model_id, private=args.private)
    processor.push_to_hub(args.hub_model_id, private=args.private)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=args.hub_model_id,
        repo_type="model",
    )

    print(f"\nModel live at: https://huggingface.co/{args.hub_model_id}")
    print(f"\nAdd to VIDA .env:")
    print(f"  HF_API_URL=https://api-inference.huggingface.co/models/{args.hub_model_id}")


if __name__ == "__main__":
    main()
