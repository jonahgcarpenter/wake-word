"""Train wake-word model. Library module, called via main.py only.

Training wrapper only: reads configs/custom_model.yml, runs OWW loop.
No data download, synthesis, evaluation, or export logic here.
"""

from pathlib import Path


def run_train(model_config: Path, synthetic_dir: Path, models_dir: Path) -> Path:
    """Run OWW training from synthetic_dir, write versioned output under models_dir."""
    raise NotImplementedError("train not implemented yet")
