"""Synthesize training clips. Library module, called via main.py only.

Synthetic data generation only: Piper TTS positives + negatives.
Output -> data/synthetic/.
Reads configs/voices.yaml; no augmentation, training, or evaluation here.
"""

import csv
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


def _load_piper_voice(onnx_path: Path):
    try:
        from piper import PiperVoice
    except ImportError:
        raise RuntimeError(
            "piper-tts is required (`uv pip install -r requirements.txt`)"
        ) from None
    return PiperVoice.load(str(onnx_path))


def _synth_tmp(voice, text: str, tmp: Path) -> None:
    wrote = False
    with wave.open(str(tmp), "wb") as w:
        for chunk in voice.synthesize(text):
            if not wrote:
                w.setframerate(chunk.sample_rate)
                w.setsampwidth(chunk.sample_width)
                w.setnchannels(chunk.sample_channels)
                wrote = True
            w.writeframes(chunk.audio_int16_bytes)
    if not wrote:
        raise RuntimeError(f"piper produced no audio for {text!r}")


def _resample_16k(tmp: Path, out_wav: Path, timeout: int = 120) -> None:
    try:
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


def _voice_worker(
    voice_key: str,
    onnx_path: Path,
    assignments: list[tuple[str, Path, dict]],
) -> list[dict]:
    """Load one voice once, render all its clips. One thread per voice."""
    voice = _load_piper_voice(onnx_path)
    rows: list[dict] = []
    for text, out_wav, meta in assignments:
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_wav.with_suffix(".tmp.wav")
        _synth_tmp(voice, text, tmp)
        _resample_16k(tmp, out_wav)
        rows.append(
            {**meta, "text": text, "voice": voice_key, "voice_file": onnx_path.name}
        )
    return rows


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
    _resolve_bin("ffmpeg", "ffmpeg")
    synthetic_dir.mkdir(parents=True, exist_ok=True)

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
        by_voice: dict[str, list[tuple[str, Path, dict]]] = {}
        for i in range(min(preview, 500)):
            voice = keys[i % len(keys)]
            wav = out_dir / f"preview_{i:03d}_{voice}.wav"
            meta = {"file": wav.name, "split": "preview", "kind": "preview"}
            by_voice.setdefault(voice, []).append(
                (targets[i % len(targets)], wav, meta)
            )
        _run_pool(voice_paths, by_voice, jobs, manifest=None)
        print(f"preview ready -> {out_dir}/")
        for voice, items in by_voice.items():
            for text, wav, _meta in items:
                print(f"  preview {wav.name}: {text!r} [{voice}]")
        return synthetic_dir

    plan = _plan_clips(cfg, primary, secondary)
    if limit > 0:
        plan = [p for p in plan if p[2] < limit]
    manifest = synthetic_dir / "manifest.csv"
    done = _load_done(manifest, force)
    by_voice = {}
    for kind, split, i, text, voice in plan:
        name = f"{kind}_{split}_{i:06d}_{voice}.wav"
        if not force and name in done:
            continue
        out = synthetic_dir / split / kind / name
        meta = {"file": name, "split": split, "kind": kind}
        by_voice.setdefault(voice, []).append((text, out, meta))
    print(
        f"plan: {len(plan)} clips "
        f"({sum(len(v) for v in by_voice.values())} to render, "
        f"{len(plan) - sum(len(v) for v in by_voice.values())} cached) "
        f"across {len(by_voice)} voices",
        flush=True,
    )
    if not by_voice:
        return synthetic_dir
    with manifest.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        _run_pool(voice_paths, by_voice, jobs, manifest_writer=writer)
    return synthetic_dir


def _run_pool(
    voice_paths: dict[str, Path],
    by_voice: dict[str, list[tuple[str, Path, dict]]],
    jobs: int | None,
    manifest=None,
    manifest_writer=None,
) -> tuple[int, int]:
    """Render one thread per voice (model loaded once). Returns (ok, fail)."""
    total = sum(len(v) for v in by_voice.values())
    workers = min(jobs or len(by_voice), len(by_voice))
    ok = fail = 0
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_voice_worker, voice, voice_paths[voice], items): voice
            for voice, items in by_voice.items()
        }
        finished = 0
        for fut in as_completed(futures):
            voice = futures[fut]
            try:
                for row in fut.result():
                    if manifest_writer is not None:
                        manifest_writer.writerow(row)
                    ok += 1
                    finished += 1
                    if finished % 500 == 0:
                        if manifest is not None:
                            manifest.flush()
                        print(
                            f"  ... {finished}/{total} (ok={ok} fail={fail})",
                            flush=True,
                        )
            except Exception as exc:  # noqa: BLE001
                fail += len(by_voice[voice])
                finished += len(by_voice[voice])
                if len(errors) < 5:
                    errors.append(f"[{voice}]: {exc}")
    print(f"done: ok={ok} fail={fail}", flush=True)
    for err in errors:
        print(f"  ! {err}", file=sys.stderr)
    if ok == 0 and total > 0:
        raise RuntimeError("no clips rendered, see errors above")
    return ok, fail
