#!/usr/bin/env python3
"""Iterate raw clips, play audio, collect T/F human labels.

Reads:  <raw_dir>/<device>/<activations|near_misses>/<id>.wav (+.json)
Writes: <classified_dir>/true/<device>__<id>.wav (+.json)
        <classified_dir>/false/<device>__<id>.wav (+.json)
        <classified_dir>/manifest.csv

Keys (single press, no Enter needed on Linux tty):
  T = true activation (really "Hey Oswald")   F = false activation
  S = skip   R = replay   Q = quit
Standalone: python classify.py [--no-play] [--limit N]
Stdlib only.
"""

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

MANIFEST_FIELDS = [
    "source_wav",
    "device",
    "orig_label",
    "score",
    "trigger",
    "human_label",
    "classified_wav",
]


def find_clips(raw_dir: Path) -> list[Path]:
    return sorted(raw_dir.glob("*/*/*.wav"))


def load_manifest(manifest: Path) -> dict:
    done = {}
    if manifest.exists():
        with manifest.open() as f:
            for row in csv.DictReader(f):
                done[row["source_wav"]] = row
    return done


def detect_player(explicit: str | None) -> list[str] | None:
    if explicit:
        return [explicit]
    for cand in (
        ["aplay"],
        ["paplay"],
        ["play"],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ):
        if shutil.which(cand[0]):
            return cand
    return None


def play_audio(player: list[str] | None, wav: Path, no_play: bool) -> None:
    if no_play or player is None:
        return
    try:
        subprocess.run(
            [*player, str(wav)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"  ! play failed ({e}), continuing", file=sys.stderr)


def get_key(prompt: str) -> str:
    """Single-keypress read on Linux tty, falls back to input()."""
    try:
        import termios
        import tty

        sys.stdout.write(prompt)
        sys.stdout.flush()
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write(f"{ch}\n")
        return ch.lower()
    except Exception:
        return input(prompt).strip().lower()[:1]


def sidecar_info(wav: Path) -> dict:
    j = wav.with_suffix(".json")
    if j.exists():
        try:
            return json.loads(j.read_text())
        except Exception:
            pass
    return {}


def run_label(
    raw_dir: Path,
    classified_dir: Path,
    player: str | None = None,
    no_play: bool = False,
    limit: int = 0,
    relabel: bool = False,
    device_filter: str = "",
) -> tuple[int, int, int]:
    clips = find_clips(raw_dir)
    if device_filter:
        clips = [c for c in clips if device_filter in c.parts[-3]]
    if limit:
        clips = clips[:limit]
    if not clips:
        print(f"No clips in {raw_dir}/*/*/*.wav")
        return 0, 0, 0

    manifest = classified_dir / "manifest.csv"
    done = {} if relabel else load_manifest(manifest)
    player_cmd = None if no_play else detect_player(player)
    if not no_play and player_cmd is None:
        print(
            "WARNING: no audio player found (aplay/paplay/play/ffplay). "
            "Continuing without audio; use --no-play to silence this.",
            file=sys.stderr,
        )

    true_d = classified_dir / "true"
    false_d = classified_dir / "false"
    true_d.mkdir(parents=True, exist_ok=True)
    false_d.mkdir(parents=True, exist_ok=True)
    new_rows = not manifest.exists()
    f = manifest.open("a", newline="") if manifest.exists() else None
    writer = None
    if f is None:
        f = manifest.open("w", newline="")
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
    else:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)

    n_true = n_false = n_skip = 0
    pending = [c for c in clips if relabel or str(c) not in done]
    print(
        f"{len(pending)} to review ({len(clips)} total, {len(clips) - len(pending)} already labeled)"
    )

    try:
        for i, wav in enumerate(pending, 1):
            info = sidecar_info(wav)
            device = wav.parts[-3]
            orig = wav.parts[-2]
            print(
                f"\n[{i}/{len(pending)}] {device}/{orig}/{wav.name} "
                f"score={info.get('score', '?')} trigger={info.get('trigger', '?')}"
            )
            play_audio(player_cmd, wav, no_play)
            while True:
                k = get_key("  [T]rue  [F]alse  [S]kip  [R]eplay  [Q]uit ? ")
                if k == "r":
                    play_audio(player_cmd, wav, no_play)
                    continue
                if k == "q":
                    print("Quit, progress saved.")
                    f.close()
                    return n_true, n_false, n_skip
                if k in ("t", "f", "s"):
                    break
                print("  press T, F, S, R or Q")
            if k == "s":
                n_skip += 1
                continue
            human = "true" if k == "t" else "false"
            dest_dir = true_d if k == "t" else false_d
            dest = dest_dir / f"{device}__{wav.stem}.wav"
            shutil.copy2(wav, dest)
            src_json = wav.with_suffix(".json")
            if src_json.exists():
                labeled = dict(info)
                labeled["human_label"] = human
                labeled["source_wav"] = str(wav)
                (dest.with_suffix(".json")).write_text(json.dumps(labeled, indent=2))
            writer.writerow(
                {
                    "source_wav": str(wav),
                    "device": device,
                    "orig_label": orig,
                    "score": info.get("score", ""),
                    "trigger": info.get("trigger", ""),
                    "human_label": human,
                    "classified_wav": str(dest),
                }
            )
            f.flush()
            if k == "t":
                n_true += 1
            else:
                n_false += 1
    except KeyboardInterrupt:
        print("\nInterrupted, progress saved.")
    finally:
        f.close()
    print(
        f"\nLabeled this run: {n_true} true, {n_false} false, {n_skip} skipped -> {classified_dir}/"
    )
    return n_true, n_false, n_skip


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Play + T/F label raw/ -> classified/")
    ap.add_argument("--raw-dir", default="raw")
    ap.add_argument("--classified-dir", default="classified")
    ap.add_argument(
        "--player",
        default=None,
        help="audio player binary (default: auto-detect aplay/paplay/play/ffplay)",
    )
    ap.add_argument(
        "--no-play", action="store_true", help="don't play audio (test manifest flow)"
    )
    ap.add_argument("--limit", type=int, default=0, help="only review first N clips")
    ap.add_argument(
        "--relabel", action="store_true", help="review even already-labeled clips"
    )
    ap.add_argument(
        "--device", default="", help="only clips whose device contains this string"
    )
    args = ap.parse_args()
    run_label(
        Path(args.raw_dir),
        Path(args.classified_dir),
        player=args.player,
        no_play=args.no_play,
        limit=args.limit,
        relabel=args.relabel,
        device_filter=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
