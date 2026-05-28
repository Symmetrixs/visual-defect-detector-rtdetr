# VIDA — Visual Defect Detector (RT-DETRv2)

Standalone training app for the VIDA building defect detection model.
Fine-tunes **RT-DETRv2** on 7 building surface defect classes.

## Defect Classes

| Class | Description |
|---|---|
| `crack` | Structural or hairline crack |
| `faded_paint` | Peeling or deteriorated paint |
| `spalling` | Concrete flaking or chipping |
| `water_stain` | Dampness or moisture marks |
| `rust` | Corrosion on metal or rebar bleed-through |
| `mold` | Mold, algae, or biological growth |
| `efflorescence` | White chalky salt deposits on concrete/brick |

---

## Setup

```bash
pip install -r requirements.txt
huggingface-cli login   # needed for push_to_hub
```

---

## Dataset Format

Export from Roboflow / CVAT / Label Studio as **COCO JSON**.

```
data/
  train/
    images/         ← .jpg / .png files
    _annotations.coco.json
  valid/
    images/
    _annotations.coco.json
```

Category names in the JSON must match one of the 7 class names above.
The loader does fuzzy matching so `"Crack"`, `"CRACK"`, and `"crack"` all work.

---

## Training

```bash
python train.py \
  --train_images data/train/images \
  --train_ann    data/train/_annotations.coco.json \
  --val_images   data/valid/images \
  --val_ann      data/valid/_annotations.coco.json \
  --output_dir   outputs/vida-rtdetrv2 \
  --epochs       50 \
  --batch_size   4
```

Resume from checkpoint:
```bash
python train.py ... --resume outputs/vida-rtdetrv2/checkpoint-500
```

Train + push to HuggingFace in one step:
```bash
python train.py ... --push_to_hub --hub_model_id YourName/vida-rtdetrv2-defect
```

---

## Evaluation

```bash
python evaluate.py \
  --model_path outputs/vida-rtdetrv2 \
  --val_images data/valid/images \
  --val_ann    data/valid/_annotations.coco.json
```

Outputs: mAP@50, mAP@50:95, per-class AP with bar chart.

---

## Local Inference

Single image:
```bash
python infer.py outputs/vida-rtdetrv2 path/to/photo.jpg
```

Whole folder:
```bash
python infer.py outputs/vida-rtdetrv2 data/test/images --output results/
```

---

## Push to HuggingFace Hub

```bash
python export.py outputs/vida-rtdetrv2 YourName/vida-rtdetrv2-defect
```

Then set in VIDA `Backend/.env`:
```
HF_API_URL=https://api-inference.huggingface.co/models/YourName/vida-rtdetrv2-defect
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
CONFIDENCE_THRESHOLD=0.3
```

---

## Recommended Training Setup

- **Google Colab** (free T4 GPU) — sufficient for batch_size=4, 50 epochs
- **Kaggle** (free P100 GPU) — slightly faster
- Minimum dataset: ~100 images per class (700 total)
- Recommended: 300+ per class (2100+ total)

---

## File Structure

```
├── config.py        ← All class names, colors, and default hyperparameters
├── dataset.py       ← COCO dataset loader
├── train.py         ← Fine-tuning script
├── evaluate.py      ← mAP evaluation
├── infer.py         ← Local inference + visualisation
├── export.py        ← Push to HuggingFace Hub
├── requirements.txt
└── data/            ← Put your dataset here
```
