import os
import argparse
import torch
from transformers import (
    RTDetrV2ForObjectDetection,
    RTDetrImageProcessor,
    TrainingArguments,
    Trainer,
)
from dataset import DefectDataset, collate_fn
from config import TrainConfig, ID2LABEL, LABEL2ID, DEFECT_CLASSES


def parse_args():
    p = argparse.ArgumentParser(description="Train RT-DETRv2 on VIDA building defects")
    p.add_argument("--train_images",  required=True)
    p.add_argument("--train_ann",     required=True)
    p.add_argument("--val_images",    required=True)
    p.add_argument("--val_ann",       required=True)
    p.add_argument("--output_dir",    default=TrainConfig.output_dir)
    p.add_argument("--base_model",    default=TrainConfig.base_model)
    p.add_argument("--epochs",        type=int,   default=TrainConfig.epochs)
    p.add_argument("--batch_size",    type=int,   default=TrainConfig.batch_size)
    p.add_argument("--lr",            type=float, default=TrainConfig.learning_rate)
    p.add_argument("--resume",        default=None)
    p.add_argument("--push_to_hub",   action="store_true")
    p.add_argument("--hub_model_id",  default="")
    return p.parse_args()


def main():
    args   = parse_args()
    cfg    = TrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("  VIDA RT-DETRv2 Training")
    print(f"  Device      : {device}")
    if device == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Base model  : {args.base_model}")
    print(f"  Output      : {args.output_dir}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  FP16        : {cfg.fp16 and device == 'cuda'}")
    print(f"  Classes ({len(DEFECT_CLASSES)}): {', '.join(DEFECT_CLASSES)}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nLoading processor and model...")
    processor = RTDetrImageProcessor.from_pretrained(args.base_model)
    model     = RTDetrV2ForObjectDetection.from_pretrained(
        args.base_model,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        num_labels=len(DEFECT_CLASSES),
        ignore_mismatched_sizes=True,
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,} total | {trainable:,} trainable")

    print("\nLoading datasets...")
    train_dataset = DefectDataset(args.train_images, args.train_ann, processor)
    val_dataset   = DefectDataset(args.val_images,   args.val_ann,   processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type=cfg.lr_scheduler,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=cfg.log_steps,
        fp16=cfg.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=cfg.dataloader_workers,
        dataloader_pin_memory=cfg.pin_memory,
        seed=cfg.seed,
        push_to_hub=args.push_to_hub and bool(args.hub_model_id),
        hub_model_id=args.hub_model_id or None,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        processing_class=processor,
    )

    print("\nStarting training... (monitor GPU: nvidia-smi dmon -s u)")
    trainer.train(resume_from_checkpoint=args.resume)

    print(f"\nSaving best model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if args.push_to_hub and args.hub_model_id:
        print(f"\nPushing to HuggingFace Hub: {args.hub_model_id}")
        trainer.push_to_hub(commit_message="Training complete")
        processor.push_to_hub(args.hub_model_id)
        print(f"Model live at: https://huggingface.co/{args.hub_model_id}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
