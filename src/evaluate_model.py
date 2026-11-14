"""Evaluate model on real clips. Library module, called via main.py only.

Validation only on data/classified/: recall, FP/hr, ROC sweep -> metrics.json.
Never trains; consumes models from data/models/<model_name>/.
"""

from pathlib import Path


def run_evaluate(model_path: Path, classified_dir: Path, output_json: Path) -> dict:
    """Score model_path against data/classified/, write metrics to output_json."""
    raise NotImplementedError("evaluate not implemented yet")
