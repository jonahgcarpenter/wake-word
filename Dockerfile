FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3.10-venv build-essential ffmpeg espeak-ng && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /work
COPY requirements.txt ./
RUN uv venv /opt/venv && uv pip install -r requirements.txt
COPY configs/ ./configs/
COPY src/ ./src/
COPY main.py ./
CMD ["python", "main.py", "--help"]
