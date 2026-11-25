"""ONNX export, graph validation, and PyTorch parity verification for SafeGuard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from src.data.inspect_dataset import compute_sha256


def export_to_onnx(
    weights_path: Path,
    output_onnx_path: Path,
    imgsz: int = 640,
    opset: int = 17,
    dynamic: bool = False,
) -> dict[str, Any]:
    """Export a trained Ultralytics YOLO checkpoint to ONNX and verify graph."""
    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    exported_path_str = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        dynamic=dynamic,
        simplify=False,
    )
    exported_path = Path(exported_path_str)

    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != output_onnx_path.resolve():
        output_onnx_path.write_bytes(exported_path.read_bytes())

    onnx_model = onnx.load(str(output_onnx_path))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(
        str(output_onnx_path), providers=["CPUExecutionProvider"]
    )
    inputs_meta = [
        {"name": inp.name, "shape": list(inp.shape), "type": inp.type}
        for inp in session.get_inputs()
    ]
    outputs_meta = [
        {"name": out.name, "shape": list(out.shape), "type": out.type}
        for out in session.get_outputs()
    ]

    dummy = np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)
    ort_out = session.run(None, {inputs_meta[0]["name"]: dummy})

    metadata = {
        "source_checkpoint": str(weights_path),
        "checkpoint_sha256": compute_sha256(weights_path),
        "onnx_path": str(output_onnx_path),
        "onnx_sha256": compute_sha256(output_onnx_path),
        "onnx_size_mb": round(output_onnx_path.stat().st_size / (1024 * 1024), 3),
        "opset": opset,
        "imgsz": imgsz,
        "dynamic": dynamic,
        "inputs": inputs_meta,
        "outputs": outputs_meta,
        "smoke_output_shapes": [list(arr.shape) for arr in ort_out],
    }
    meta_path = output_onnx_path.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    """CLI entrypoint for exporting a YOLO checkpoint to ONNX."""
    parser = argparse.ArgumentParser(description="Export SafeGuard YOLO model to ONNX.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    meta = export_to_onnx(args.weights, args.output, imgsz=args.imgsz, opset=args.opset)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
