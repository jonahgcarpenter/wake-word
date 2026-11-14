"""Export model artifacts. Library module, called via main.py only.

Export step only: .onnx -> .tflite/.oww + versioned copy to data/models/<model_name>/.
No training or evaluation logic here.
"""

from pathlib import Path


def run_export(model_path: Path, models_dir: Path) -> Path:
    """Convert .onnx to .tflite/.oww and version the output under models_dir."""
    raise NotImplementedError("export not implemented yet")
