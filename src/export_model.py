"""
Utilities to export YOLO models to ONNX and TFLite for mobile deployment.

Example:
  python -m src.export_model --model ./src/yolov10x.pt --onnx ./src/yolov10x.onnx --dynamic
  python -m src.export_model --model ./src/yolov8n.pt --tflite ./src/yolov8n_fp16.tflite --fp16
"""

import os
import argparse
from typing import Optional

from ultralytics import YOLO


def export_to_onnx(model_path: str, onnx_path: str, dynamic: bool = True, opset: int = 13):
    model = YOLO(model_path)
    model.export(format="onnx", dynamic=dynamic, opset=opset, imgsz=640, half=False, simplify=True)
    # ultralytics saves next to the model with suffix; move/rename if requested
    produced = os.path.splitext(model_path)[0] + ".onnx"
    if os.path.exists(produced) and produced != onnx_path:
        os.replace(produced, onnx_path)


def export_to_tflite(model_path: str, tflite_path: str, fp16: bool = True, int8: bool = False):
    model = YOLO(model_path)
    args = {"format": "tflite", "imgsz": 640}
    if fp16:
        args["half"] = True
    if int8:
        args["int8"] = True
    model.export(**args)
    produced = os.path.splitext(model_path)[0] + ("_int8.tflite" if int8 else "_fp16.tflite" if fp16 else ".tflite")
    if os.path.exists(produced) and produced != tflite_path:
        os.replace(produced, tflite_path)


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model for mobile deployment")
    parser.add_argument("--model", required=True, help="Path to .pt model")
    parser.add_argument("--onnx", type=str, default=None, help="Output ONNX path")
    parser.add_argument("--tflite", type=str, default=None, help="Output TFLite path")
    parser.add_argument("--dynamic", action="store_true", help="Use dynamic axes for ONNX")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--fp16", action="store_true", help="Export half precision (TFLite)")
    parser.add_argument("--int8", action="store_true", help="Export int8 quantized (TFLite)")
    args = parser.parse_args()

    if args.onnx:
        export_to_onnx(args.model, args.onnx, dynamic=args.dynamic, opset=args.opset)
    if args.tflite:
        export_to_tflite(args.model, args.tflite, fp16=args.fp16, int8=args.int8)
    if not args.onnx and not args.tflite:
        parser.error("Specify at least one of --onnx or --tflite")


if __name__ == "__main__":
    main()



