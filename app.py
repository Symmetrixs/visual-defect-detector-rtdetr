import sys
import os
import subprocess
import threading
from pathlib import Path
import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    get_gpu_info, find_datasets, validate_dataset, do_merge,
    get_class_distribution, get_latest_checkpoint, get_merged_paths,
    merged_exists, DATASET_DIR, MERGED_DIR, OUTPUT_DIR, SCRIPT_DIR
)
from config import DEFECT_CLASSES, TrainConfig, InferConfig


def tab_dataset():
    with gr.Tab("📁 Dataset"):
        gr.Markdown("## Dataset Management")
        gr.Markdown(f"**Dataset directory:** `{DATASET_DIR}`")

        with gr.Row():
            btn_check = gr.Button("🔍 Check & Validate Datasets", variant="primary")
            btn_merge = gr.Button("🔗 Merge Datasets",            variant="primary")

        check_output = gr.Textbox(label="Validation Report", lines=14, interactive=False)
        merge_output = gr.Textbox(label="Merge Log",         lines=14, interactive=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Train Split — Class Distribution")
                train_chart = gr.BarPlot(visible=False)
            with gr.Column():
                gr.Markdown("### Valid Split — Class Distribution")
                valid_chart = gr.BarPlot(visible=False)

        def check_datasets():
            datasets, errors = find_datasets()
            lines = []
            if errors:
                lines.append("❌ ERRORS:")
                lines.extend(f"  {e}" for e in errors)
            else:
                lines.append(f"✅ Found {len(datasets)} dataset(s):\n")
            for ds in datasets:
                lines.append(f"📂 {ds.name}")
                report, issues = validate_dataset(ds)
                for r in report:
                    lines.append(f"   ✓ {r}")
                for i in issues:
                    lines.append(f"   ✗ {i}")
                lines.append("")
            return "\n".join(lines)

        def merge_datasets_ui():
            datasets, errors = find_datasets()
            if errors:
                return "❌ Cannot merge — datasets not found:\n" + "\n".join(errors)
            log = []

            def progress(msg):
                log.clear()
                log.append(msg)

            result = do_merge(datasets, progress_fn=progress)
            return result

        btn_check.click(check_datasets, outputs=check_output)
        btn_merge.click(merge_datasets_ui, outputs=merge_output)


def tab_train():
    cfg = TrainConfig()
    with gr.Tab("🏋️ Train"):
        gr.Markdown("## Training Configuration")

        with gr.Row():
            epochs     = gr.Slider(10, 200, value=cfg.epochs,    step=5,   label="Epochs")
            batch_size = gr.Slider(1,  16,  value=cfg.batch_size, step=1,  label="Batch Size")
        with gr.Row():
            lr         = gr.Number(value=cfg.learning_rate, label="Learning Rate", precision=6)
            base_model = gr.Textbox(value=cfg.base_model,   label="Base Model (HuggingFace)")
        with gr.Row():
            resume     = gr.Textbox(value="", label="Resume from checkpoint (leave blank for fresh start)", placeholder="outputs/vida-rtdetrv2/checkpoint-xxx")
            push_hub   = gr.Checkbox(value=False, label="Push to HuggingFace Hub after training")
            hub_id     = gr.Textbox(value="", label="HuggingFace Model ID (e.g. YourName/vida-rtdetrv2)", visible=False)

        push_hub.change(lambda x: gr.update(visible=x), inputs=push_hub, outputs=hub_id)

        with gr.Row():
            btn_train = gr.Button("🚀 Start Training", variant="primary", scale=3)
            btn_stop  = gr.Button("⏹ Stop",           variant="stop",    scale=1)

        status  = gr.Markdown("**Status:** Ready")
        log_box = gr.Textbox(label="Training Log", lines=22, interactive=False, autoscroll=True)

        proc_state = gr.State(value=None)

        def start_training(epochs, batch_size, lr, base_model, resume, push_hub, hub_id):
            ti, ta, vi, va = get_merged_paths()
            if not merged_exists():
                yield "❌ Merged dataset not found. Go to 📁 Dataset tab and run Merge first.", gr.update(value="**Status:** ❌ No merged dataset"), None
                return

            cmd = [
                sys.executable, str(SCRIPT_DIR / "train.py"),
                "--train_images", ti,
                "--train_ann",    ta,
                "--val_images",   vi,
                "--val_ann",      va,
                "--output_dir",   str(OUTPUT_DIR),
                "--epochs",       str(int(epochs)),
                "--batch_size",   str(int(batch_size)),
                "--lr",           str(lr),
                "--base_model",   base_model,
            ]
            if resume.strip():
                cmd += ["--resume", resume.strip()]
            if push_hub and hub_id.strip():
                cmd += ["--push_to_hub", "--hub_model_id", hub_id.strip()]

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env, cwd=str(SCRIPT_DIR)
            )

            log_text = ""
            for line in proc.stdout:
                log_text += line
                yield log_text, gr.update(value="**Status:** 🏋️ Training..."), proc

            proc.wait()
            if proc.returncode == 0:
                yield log_text + "\n✅ Training complete!", gr.update(value="**Status:** ✅ Complete"), None
            else:
                yield log_text + "\n❌ Training failed.", gr.update(value="**Status:** ❌ Failed"), None

        def stop_training(proc):
            if proc is not None:
                proc.terminate()
                return "⏹ Training stopped by user.", gr.update(value="**Status:** Stopped"), None
            return "No training running.", gr.update(value="**Status:** Idle"), None

        btn_train.click(
            start_training,
            inputs=[epochs, batch_size, lr, base_model, resume, push_hub, hub_id],
            outputs=[log_box, status, proc_state]
        )
        btn_stop.click(stop_training, inputs=[proc_state], outputs=[log_box, status, proc_state])


def tab_evaluate():
    with gr.Tab("📊 Evaluate"):
        gr.Markdown("## Model Evaluation")

        with gr.Row():
            model_path = gr.Textbox(value=str(OUTPUT_DIR), label="Model Path (local dir or HuggingFace repo)")
            conf       = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="Confidence Threshold")

        btn_eval = gr.Button("▶ Run Evaluation", variant="primary")
        eval_log = gr.Textbox(label="Evaluation Results", lines=20, interactive=False)

        def run_eval(model_path, conf):
            _, _, vi, va = get_merged_paths()
            if not Path(vi).exists():
                yield "❌ Merged dataset not found. Run merge first."
                return

            cmd = [
                sys.executable, str(SCRIPT_DIR / "evaluate.py"),
                "--model_path", model_path.strip(),
                "--val_images", vi,
                "--val_ann",    va,
                "--conf",       str(conf),
            ]

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env=env, cwd=str(SCRIPT_DIR))
            log = ""
            for line in proc.stdout:
                log += line
                yield log
            proc.wait()

        btn_eval.click(run_eval, inputs=[model_path, conf], outputs=eval_log)


def tab_inference():
    with gr.Tab("🔍 Inference"):
        gr.Markdown("## Test Your Model")

        with gr.Row():
            with gr.Column(scale=1):
                inp_image  = gr.Image(label="Upload Image", type="pil")
                model_path = gr.Textbox(value=str(OUTPUT_DIR), label="Model Path")
                conf       = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="Confidence Threshold")
                btn_detect = gr.Button("🔍 Detect Defects", variant="primary")

            with gr.Column(scale=1):
                out_image = gr.Image(label="Detection Result", type="pil")
                det_table = gr.Dataframe(
                    headers=["#", "Class", "Confidence", "x1", "y1", "x2", "y2"],
                    label="Detected Defects",
                    interactive=False,
                )

        def detect(image, model_path, conf):
            if image is None:
                return None, []
            try:
                from infer import load_model, predict_image, visualize
                model, processor, device = load_model(model_path.strip())
                dets    = predict_image(model, processor, device, image, conf)
                vis_img = visualize(image.copy(), dets)
                table   = [
                    [i+1, d["label"], f"{d['score']:.2%}",
                     f"{d['box']['xmin']:.0f}", f"{d['box']['ymin']:.0f}",
                     f"{d['box']['xmax']:.0f}", f"{d['box']['ymax']:.0f}"]
                    for i, d in enumerate(dets)
                ]
                return vis_img, table
            except Exception as e:
                return None, [[str(e), "", "", "", "", "", ""]]

        btn_detect.click(detect, inputs=[inp_image, model_path, conf], outputs=[out_image, det_table])


def tab_export():
    with gr.Tab("🚀 Export"):
        gr.Markdown("## Push Model to HuggingFace Hub")

        with gr.Row():
            model_path = gr.Textbox(value=str(OUTPUT_DIR), label="Model Path (local)")
            hub_id     = gr.Textbox(value="", label="HuggingFace Repo ID  (e.g. YourName/vida-rtdetrv2-defect)", placeholder="YourName/vida-rtdetrv2-defect")

        private   = gr.Checkbox(value=True, label="Private repository")
        btn_push  = gr.Button("⬆ Push to HuggingFace Hub", variant="primary")
        push_log  = gr.Textbox(label="Export Log", lines=10, interactive=False)

        def push(model_path, hub_id, private):
            if not hub_id.strip():
                return "❌ Please enter a HuggingFace repo ID"
            cmd = [
                sys.executable, str(SCRIPT_DIR / "export.py"),
                model_path.strip(), hub_id.strip(),
            ]
            if private:
                cmd.append("--private")
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1, env=env, cwd=str(SCRIPT_DIR))
            log = ""
            for line in proc.stdout:
                log += line
                yield log
            proc.wait()

        btn_push.click(push, inputs=[model_path, hub_id, private], outputs=push_log)


def build_app():
    gpu_info = get_gpu_info()

    with gr.Blocks(
        title="VIDA — RT-DETRv2 Training",
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=".gradio-container { max-width: 1200px !important; }"
    ) as demo:

        gr.Markdown(f"""
# 🔬 VIDA — RT-DETRv2 Building Defect Detector
**Visual Infrastructure Defect Analyzer · UTeM FYP**

💻 **{gpu_info}** &nbsp;|&nbsp; 🏷️ **7 Classes:** {' · '.join(DEFECT_CLASSES)}
""")

        tab_dataset()
        tab_train()
        tab_evaluate()
        tab_inference()
        tab_export()

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
