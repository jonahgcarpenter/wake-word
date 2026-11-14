"""Synthesize training clips. Library module, called via main.py only.

Synthetic data generation only: Piper TTS positives + negatives.
Output -> data/synthetic/.
Reads configs/voices.yml; no augmentation, training, or evaluation here.
"""

from pathlib import Path


def run_synthesize(
    model_config: Path, voices_config: Path, synthetic_dir: Path
) -> Path:
    """Generate Piper TTS positives + adversarial negatives into synthetic_dir."""
    raise NotImplementedError("synthesize not implemented yet")
