"""
export_yolo.py

Export a YOLO model (YOLO26 / YOLO11 compatible) into selected formats.

Example:
    python export_yolo.py \
        --weights runs/train/exp/weights/best.pt \
        --formats onnx tfjs tflite \
        --imgsz 640
"""

import argparse
from pathlib import Path
import subprocess

# Supported formats from Ultralytics export
SUPPORTED_FORMATS = {
    "onnx",
    "openvino",
    "engine",        # TensorRT
    "coreml",
    "saved_model",
    "pb",
    "tflite",
    "tfjs"           # handled separately
}


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO model to selected formats")

    parser.add_argument("--weights", type=str, required=True, help="Path to .pt model")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--output", type=str, default="exports", help="Output directory")

    parser.add_argument(
        "--formats",
        nargs="+",
        required=True,
        help=f"Formats to export: {', '.join(SUPPORTED_FORMATS)}"
    )

    return parser.parse_args()


def validate_formats(formats):
    formats = set(f.lower() for f in formats)

    invalid = formats - SUPPORTED_FORMATS
    if invalid:
        raise ValueError(f"Unsupported formats: {invalid}")

    # TFJS requires saved_model
    if "tfjs" in formats:
        formats.add("saved_model")

    return formats


def export_formats(model, args, formats):
    """
    Export selected formats using Ultralytics
    """
    for fmt in formats:
        if fmt == "tfjs":
            continue  # handled later

        try:
            print(f"\n🚀 Exporting to {fmt}...")
            model.export(
                format=fmt,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                project=args.output,
                name=fmt
            )
        except Exception as e:
            print(f"❌ Failed to export {fmt}: {e}")


def export_tfjs(output_dir):
    """
    Convert SavedModel → TensorFlow.js
    """
    saved_model_dir = Path(output_dir) / "saved_model"
    output_path = Path(output_dir) / "tfjs"

    if not saved_model_dir.exists():
        print("❌ SavedModel not found, cannot export TFJS.")
        return

    print("\n🌐 Converting to TensorFlow.js (tfjs)...")

    cmd = [
        "tensorflowjs_converter",
        "--input_format=tf_saved_model",
        "--output_format=tfjs_graph_model",
        "--signature_name=serving_default",
        "--saved_model_tags=serve",
        str(saved_model_dir),
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"✅ TFJS model saved to: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"❌ TFJS conversion failed: {e}")


def main():
    args = parse_args()
    formats = validate_formats(args.formats)

    from ultralytics import YOLO

    print(f"📦 Loading model: {args.weights}")
    model = YOLO(args.weights)

    # Step 1: Export selected formats (except tfjs)
    export_formats(model, args, formats)

    # Step 2: TFJS conversion if requested
    if "tfjs" in formats:
        export_tfjs(args.output)


if __name__ == "__main__":
    main()