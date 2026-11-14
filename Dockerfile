FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04
ENV UV_PROJECT_ENVIRONMENT=/opt/venv PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PIPER_VERSION=2023.11.14-2
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3.10-venv build-essential ffmpeg espeak-ng curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/piper.tar.gz https://github.com/rhasspy/piper/releases/download/${PIPER_VERSION}/piper_linux_x86_64.tar.gz \
 && mkdir -p /opt && tar -xzf /tmp/piper.tar.gz -C /opt \
 && BIN=$(find /opt/piper* -maxdepth 2 -name piper -type f 2>/dev/null | head -1) \
 && [ -n "$BIN" ] && ln -sf "$BIN" /usr/local/bin/piper \
 && rm -f /tmp/piper.tar.gz \
 && piper --help >/dev/null && ffmpeg -version >/dev/null
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /work
COPY requirements.txt ./
RUN uv venv /opt/venv && uv pip install -r requirements.txt
COPY configs/ ./configs/
COPY src/ ./src/
COPY main.py ./
CMD ["python", "main.py", "--help"]
