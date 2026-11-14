# Training container only: CUDA 11.7 + Python 3.10 base for the pinned OWW stack.
# System deps (ffmpeg, espeak-ng, build-essential) live here so the host stays clean.
# Python deps come from requirements.txt (torch 1.13.1, TF 2.8.1, etc).
# Expects data/, configs/, src/ mounted; outputs to data/models/<model_name>/.
