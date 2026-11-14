"""Wake-word training orchestrator. Run with `uv run python main.py <command>`.

Main orchestrator only: fetch -> label -> synthesize -> train -> evaluate -> export.
Sole CLI entrypoint via `uv run python main.py ...`.
src/* expose run_*() library functions.
"""

import argparse
import os
import sys
from pathlib import Path

from src.evaluate_model import run_evaluate
from src.export_model import run_export
from src.fetch_clips import run_download
from src.label_clips import run_label
from src.synthesize_clips import run_synthesize
from src.train_model import run_train


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Wake-word pipeline (uv + .venv).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch", help="download + dedupe satellite clips -> data/raw/")
    p.add_argument("--hosts", default=os.environ.get("KIOSK_HOSTS", ""))
    p.add_argument(
        "--port", type=int, default=int(os.environ.get("KIOSK_PORT", "2324"))
    )
    p.add_argument("--password", default=os.environ.get("KIOSK_PASSWORD", ""))
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("label", help="triage data/raw/ -> data/classified/")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--classified-dir", default="data/classified")
    p.add_argument("--no-play", action="store_true")
    p.add_argument("--limit", type=int, default=0)

    p = sub.add_parser("synthesize", help="Piper TTS -> data/synthetic/")
    p.add_argument("--config", default="configs/custom_model.yml")
    p.add_argument("--voices", default="configs/voices.yml")
    p.add_argument("--synthetic-dir", default="data/synthetic")

    p = sub.add_parser("train", help="OWW training -> data/models/")
    p.add_argument("--config", default="configs/custom_model.yml")
    p.add_argument("--synthetic-dir", default="data/synthetic")
    p.add_argument("--models-dir", default="data/models")

    p = sub.add_parser("evaluate", help="validate model on data/classified/")
    p.add_argument("--model", required=True)
    p.add_argument("--classified-dir", default="data/classified")
    p.add_argument("--output", default="data/models/metrics.json")

    p = sub.add_parser("export", help="onnx -> tflite/oww")
    p.add_argument("--model", required=True)
    p.add_argument("--models-dir", default="data/models")
    return ap


def main(argv: list[str] | None = None) -> int:
    load_dotenv(Path(".env"))
    args = build_parser().parse_args(argv)
    if args.cmd == "fetch":
        hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
        if not hosts or not args.password:
            print("ERROR: set KIOSK_HOSTS and KIOSK_PASSWORD", file=sys.stderr)
            return 2
        run_download(
            hosts, args.port, args.password, Path(args.raw_dir), force=args.force
        )
    elif args.cmd == "label":
        run_label(
            Path(args.raw_dir),
            Path(args.classified_dir),
            no_play=args.no_play,
            limit=args.limit,
        )
    elif args.cmd == "synthesize":
        run_synthesize(Path(args.config), Path(args.voices), Path(args.synthetic_dir))
    elif args.cmd == "train":
        run_train(Path(args.config), Path(args.synthetic_dir), Path(args.models_dir))
    elif args.cmd == "evaluate":
        run_evaluate(Path(args.model), Path(args.classified_dir), Path(args.output))
    elif args.cmd == "export":
        run_export(Path(args.model), Path(args.models_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
