from dataclasses import dataclass
from typing import List

DEFECT_CLASSES: List[str] = [
    "crack",
    "faded_paint",
    "spalling",
    "water_stain",
    "rust",
    "mold",
    "efflorescence",
]

ID2LABEL = {i: c for i, c in enumerate(DEFECT_CLASSES)}
LABEL2ID = {c: i for i, c in enumerate(DEFECT_CLASSES)}

ALIAS_MAP = {
    "crack":                  "crack",
    "stairstep_crack":        "crack",
    "stairstep crack":        "crack",
    "hairline":               "crack",
    "diagonal crack":         "crack",
    "horizontal crack":       "crack",
    "vertical crack":         "crack",
    "multi-branched crack":   "crack",
    "peeling_paint":          "faded_paint",
    "peeling paint":          "faded_paint",
    "paint peeling":          "faded_paint",
    "flaking plaster":        "faded_paint",
    "flaking paint":          "faded_paint",
    "faded_paint":            "faded_paint",
    "spalling":               "spalling",
    "concrete spalling":      "spalling",
    "brick spalling":         "spalling",
    "spall":                  "spalling",
    "water_seepage":          "water_stain",
    "water seepage":          "water_stain",
    "water_stain":            "water_stain",
    "dampness":               "water_stain",
    "damp":                   "water_stain",
    "wet spot":               "water_stain",
    "ruststain":              "rust",
    "ruststrain":             "rust",
    "rust stain":             "rust",
    "rust":                   "rust",
    "corrosion":              "rust",
    "corrosion stain":        "rust",
    "mold":                   "mold",
    "moss":                   "mold",
    "algae":                  "mold",
    "mildew":                 "mold",
    "dampness with fungus":   "mold",
    "biological growth":      "mold",
    "efflorescence":          "efflorescence",
}

DEFECT_DESCRIPTIONS = {
    "crack":         "Structural or hairline crack on building surface or material",
    "faded_paint":   "Deteriorated, peeling, or faded paint on walls or structures",
    "spalling":      "Concrete or plaster breaking off, flaking, or chipping from surface",
    "water_stain":   "Water staining, dampness marks, or moisture damage on surfaces",
    "rust":          "Rust or corrosion on metallic elements or rebar bleeding through concrete",
    "mold":          "Mold, mildew, algae, or biological growth on surfaces",
    "efflorescence": "White chalky salt deposits on concrete, brick, or masonry surfaces",
}

DEFECT_COLORS = {
    "crack":         "#ef4444",
    "faded_paint":   "#f59e0b",
    "spalling":      "#8b5cf6",
    "water_stain":   "#3b82f6",
    "rust":          "#b45309",
    "mold":          "#16a34a",
    "efflorescence": "#64748b",
}


@dataclass
class TrainConfig:
    base_model:          str   = "PekingU/rtdetr_v2_r50vd"
    output_dir:          str   = "./outputs/vida-rtdetrv2"
    epochs:              int   = 60
    batch_size:          int   = 6
    learning_rate:       float = 1e-4
    weight_decay:        float = 1e-4
    warmup_ratio:        float = 0.1
    lr_scheduler:        str   = "cosine"
    max_grad_norm:       float = 0.1
    save_total_limit:    int   = 3
    seed:                int   = 42
    fp16:                bool  = True
    dataloader_workers:  int   = 4
    pin_memory:          bool  = True
    log_steps:           int   = 10


@dataclass
class InferConfig:
    confidence_threshold: float = 0.30
    iou_threshold:        float = 0.50
    max_detections:       int   = 100
