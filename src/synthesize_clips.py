"""Synthesize training clips. Library module, called via main.py only.

Synthetic data generation only: Piper TTS positives + negatives.
Output -> data/synthetic/.
Reads configs/voices.yaml; no augmentation, training, or evaluation here.
"""

import csv
import os
import shutil
import subprocess
import sys
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MANIFEST_FIELDS = ["file", "split", "kind", "text", "voice", "voice_file"]


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise RuntimeError(
            f"pyyaml is required to read {path} (`uv pip install -r requirements.txt`)"
        ) from None
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} did not parse to a mapping")
    return data


def _voice_keys(voices_cfg: dict, section: str) -> list[str]:
    group = voices_cfg.get(section) or {}
    models = group.get("models") or {}
    return [str(k) for k in models]


_QUALITIES = ("medium", "high", "low", "x_low")


def _head_ok(url: str, timeout: int) -> bool:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001
        return False


def download_voices(
    voices_config: Path, voice_cache: Path, timeout: int = 120
) -> tuple[dict[str, Path], dict[str, str]]:
    """Download every voice in voices.yaml.

    Voice quality varies per speaker (e.g. libritts ships only `high`),
    so each voice probes medium -> high -> low -> x_low and uses the
    first quality with downloadable files.

    Returns (paths_by_resolved_key, alias_wanted_to_resolved).
    """
    cfg = _load_yaml(voices_config)
    wanted: list[tuple[str, str, str]] = []
    for group in ((cfg.get("primary") or {}), (cfg.get("secondary") or {})):
        base = (group.get("base_url") or "").rstrip("/")
        for key, rel in (group.get("models") or {}).items():
            wanted.append((str(key), base, str(rel)))
    if not wanted:
        raise RuntimeError(f"no voices defined in {voices_config}")
    voice_cache.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    alias: dict[str, str] = {}
    for want_key, base, rel in wanted:
        parent = rel.rpartition("/")[0] or rel
        stem = want_key.rpartition("-")[0] or want_key
        cands = [(f"{stem}-{q}", f"{base}/{parent}/{q}/{stem}-{q}") for q in _QUALITIES]
        cands.sort(key=lambda c: c[0] != want_key)
        chosen: tuple[str, str] | None = None
        for key, stem_url in cands:
            if _head_ok(stem_url + ".onnx.json", timeout):
                chosen = (key, stem_url)
                break
        if chosen is None:
            raise RuntimeError(f"no downloadable quality for {want_key}")
        key, stem_url = chosen
        onnx = voice_cache / f"{key}.onnx"
        conf = voice_cache / f"{key}.onnx.json"
        for local, url in (
            (onnx, stem_url + ".onnx"),
            (conf, stem_url + ".onnx.json"),
        ):
            if local.exists() and local.stat().st_size > 0:
                continue
            print(f"  dl {url}", flush=True)
            last_exc: Exception | None = None
            for _ in range(3):
                try:
                    urllib.request.urlretrieve(url, local)
                    last_exc = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if local.exists():
                        local.unlink()
            if last_exc is not None:
                raise RuntimeError(
                    f"failed to download {url}: {last_exc}"
                ) from last_exc
        resolved[key] = onnx
        alias[want_key] = key
    print(f"voices ready: {len(resolved)} in {voice_cache}/")
    return resolved, alias


def _resolve_bin(name: str, label: str) -> str:
    if Path(name).is_file():
        return name
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f"{label} binary {name!r} not found (run inside Docker)")


def _piper_cmd(piper_bin: str, voice_onnx: Path, tmp: Path) -> list[str]:
    """Prefer the `piper-tts` pip module (matches retrained HF voices).

    Falls back to the piper binary pinned to its bundled espeak-ng-data,
    since system espeak-ng phoneme maps are incompatible and crash it.
    """
    base = ["--model", str(voice_onnx), "--output_file", str(tmp)]
    try:
        import piper  # noqa: F401

        return [sys.executable, "-m", "piper", *base]
    except ImportError:
        espeak_data = os.environ.get("PIPER_ESPEAK_DATA", "/opt/piper/espeak-ng-data")
        return [
            _resolve_bin(piper_bin, "piper"),
            *base,
            "--espeak_data",
            espeak_data,
        ]


def _render_clip(
    piper_bin: str, voice_onnx: Path, text: str, out_wav: Path, timeout: int = 180
) -> None:
    tmp = out_wav.with_suffix(".tmp.wav")
    try:
        proc = subprocess.run(
            _piper_cmd(piper_bin, voice_onnx, tmp),
            input=text.encode("utf-8"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if proc.returncode != 0 or not tmp.exists():
            err = proc.stderr.decode(errors="replace")[-500:]
            raise RuntimeError(f"piper failed ({voice_onnx.name}): {err}")
        res = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(tmp),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                str(out_wav),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if res.returncode != 0 or not out_wav.exists():
            err = res.stderr.decode(errors="replace")[-500:]
            raise RuntimeError(f"ffmpeg failed for {out_wav.name}: {err}")
        with wave.open(str(out_wav), "rb") as w:
            if w.getframerate() != 16000 or w.getnchannels() != 1:
                raise RuntimeError(f"bad format for {out_wav.name}")
    finally:
        if tmp.exists():
            tmp.unlink()


def _phrase_variants(targets: list[str]) -> list[str]:
    variants: list[str] = []
    for raw in targets:
        text = str(raw).strip()
        if not text:
            continue
        for cand in (text, text.title(), f"{text}.", f"{text}!", f"{text}?"):
            if cand not in variants:
                variants.append(cand)
    if not variants:
        raise RuntimeError("target_phrase must list at least one phrase")
    return variants


def _plan_clips(
    cfg: dict, primary: list[str], secondary: list[str]
) -> list[tuple[str, str, int, str, str]]:
    """Return (kind, split, index, text, voice) for every clip to render."""
    n_train = int(cfg.get("n_samples", 0))
    n_val = int(cfg.get("n_samples_val", 0))
    if n_train <= 0:
        raise RuntimeError("n_samples must be > 0")
    raw_targets = cfg.get("target_phrase", [])
    targets = raw_targets if isinstance(raw_targets, list) else [raw_targets]
    pos_variants = _phrase_variants([str(t) for t in targets])
    customs = [
        str(x).strip()
        for x in (cfg.get("custom_negative_phrases") or [])
        if str(x).strip()
    ]
    if not customs and not secondary:
        raise RuntimeError(
            "need custom_negative_phrases or secondary voices for negatives"
        )
    plan: list[tuple[str, str, int, str, str]] = []
    for split, count in (("train", n_train), ("val", n_val)):
        for i in range(count):
            plan.append(
                (
                    "pos",
                    split,
                    i,
                    pos_variants[i % len(pos_variants)],
                    primary[i % len(primary)],
                )
            )
    for split, count in (("train", n_train), ("val", n_val)):
        for i in range(count):
            if customs and secondary and i % 3 == 2:
                text = pos_variants[i % len(pos_variants)]
                voice = secondary[i % len(secondary)]
            elif customs:
                text = customs[i % len(customs)]
                voice = primary[i % len(primary)]
            else:
                text = pos_variants[i % len(pos_variants)]
                voice = secondary[i % len(secondary)]
            plan.append(("neg", split, i, text, voice))
    return plan


def _load_done(manifest: Path, force: bool) -> set[str]:
    if force or not manifest.exists():
        with manifest.open("w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS).writeheader()
        return set()
    done: set[str] = set()
    with manifest.open() as fh:
        for row in csv.DictReader(fh):
            if row.get("file"):
                done.add(row["file"])
    return done


def run_synthesize(
    model_config: Path,
    voices_config: Path,
    synthetic_dir: Path,
    *,
    preview: int = 0,
    limit: int = 0,
    force: bool = False,
    jobs: int | None = None,
    piper_bin: str = "piper",
    voice_cache: Path | None = None,
) -> Path:
    """Download voices, render TTS clips into synthetic_dir. Returns it."""
    cfg = _load_yaml(model_config)
    voices_cfg = _load_yaml(voices_config)
    primary = _voice_keys(voices_cfg, "primary")
    secondary = _voice_keys(voices_cfg, "secondary")
    if not primary:
        raise RuntimeError(f"no primary voices in {voices_config}")
    cache = voice_cache or (synthetic_dir.parent / "background" / "piper")
    voice_paths, alias = download_voices(voices_config, cache)
    primary = [alias[k] for k in primary]
    secondary = [alias[k] for k in secondary]
    missing = [k for k in primary + secondary if k not in voice_paths]
    if missing:
        raise RuntimeError(f"voices missing after download: {missing}")
    piper = piper_bin
    _resolve_bin("ffmpeg", "ffmpeg")
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    workers = jobs or (os.cpu_count() or 4)

    if preview > 0:
        targets = _phrase_variants(
            [
                str(t)
                for t in (
                    cfg.get("target_phrase", [])
                    if isinstance(cfg.get("target_phrase", []), list)
                    else [cfg.get("target_phrase", "")]
                )
            ]
        )
        out_dir = synthetic_dir / "preview"
        out_dir.mkdir(parents=True, exist_ok=True)
        keys = primary + secondary
        jobs_list = [
            (
                out_dir / f"preview_{i:03d}_{keys[i % len(keys)]}.wav",
                targets[i % len(targets)],
                keys[i % len(keys)],
            )
            for i in range(min(preview, 500))
        ]
        for wav, text, voice in jobs_list:
            _render_clip(piper, voice_paths[voice], text, wav)
            print(f"  preview {wav.name}: {text!r} [{voice}]")
        print(f"preview ready: {len(jobs_list)} clips -> {out_dir}/")
        return synthetic_dir

    plan = _plan_clips(cfg, primary, secondary)
    if limit > 0:
        plan = [p for p in plan if p[2] < limit]
    manifest = synthetic_dir / "manifest.csv"
    done = _load_done(manifest, force)
    todo = [
        (kind, split, i, text, voice)
        for kind, split, i, text, voice in plan
        if force or f"{kind}_{split}_{i:06d}_{voice}.wav" not in done
    ]
    print(
        f"plan: {len(plan)} clips "
        f"({len(todo)} to render, {len(plan) - len(todo)} cached)",
        flush=True,
    )
    if not todo:
        return synthetic_dir

    def _one(item: tuple[str, str, int, str, str]) -> dict:
        kind, split, i, text, voice = item
        name = f"{kind}_{split}_{i:06d}_{voice}.wav"
        out = synthetic_dir / split / kind / name
        out.parent.mkdir(parents=True, exist_ok=True)
        _render_clip(piper, voice_paths[voice], text, out)
        return {
            "file": name,
            "split": split,
            "kind": kind,
            "text": text,
            "voice": voice,
            "voice_file": voice_paths[voice].name,
        }

    ok = fail = 0
    errors: list[str] = []
    with manifest.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one, item): item for item in todo}
            for n, fut in enumerate(as_completed(futures), 1):
                try:
                    writer.writerow(fut.result())
                    ok += 1
                except Exception as exc:  # noqa: BLE001
                    fail += 1
                    if len(errors) < 5:
                        errors.append(f"{futures[fut][:3]}: {exc}")
                if n % 500 == 0:
                    fh.flush()
                    print(f"  ... {n}/{len(todo)} (ok={ok} fail={fail})", flush=True)
    print(f"done: ok={ok} fail={fail} -> {synthetic_dir}/", flush=True)
    for err in errors:
        print(f"  ! {err}", file=sys.stderr)
    if ok == 0:
        raise RuntimeError("no clips rendered, see errors above")
    return synthetic_dir
