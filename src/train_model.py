"""Train wake-word model on GPU. Library module, called via main.py only.

Official openWakeWord PyTorch recipe applied to our synthetic clips:
  background assets -> augment -> frozen-embedding features ->
  auto_train with false-positive targeting -> ONNX export
  (HA/Wyoming-compatible: same input shape as stock OWW models).

Requires CUDA; fails loudly without it (CPU training is not supported here).
"""

import datetime
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# Required bundled assets. ACAV/FP features are mandatory; background audio
# and RIRs degrade gracefully (augmentation runs without the missing piece).
ACAV_URL = (
    "https://huggingface.co/datasets/davidscripka/openwakeword_features"
    "/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
)
FPVAL_URL = (
    "https://huggingface.co/datasets/davidscripka/openwakeword_features"
    "/resolve/main/validation_set_features.npy"
)
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
RIR_URLS = [
    "https://www.openslr.org/resources/28/rirs_noises.zip",
]

MUSAN_SUBDIRS = ("music", "noise", "speech")


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


def _resolve(p: str | Path, base: Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else base / p


def _download(url: str, dest: Path, min_bytes: int = 1024) -> Path:
    """Download with resume + retries. Returns dest. Raises on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    for attempt in (1, 2, 3):
        try:
            req = urllib.request.Request(url)
            if have > 0:
                req.add_header("Range", f"bytes={have}-")
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status == 206:
                    mode, done = "ab", have
                elif r.status == 200 and have > 0:
                    mode, done = "wb", 0
                    print("  server ignored resume, restarting", flush=True)
                elif r.status == 200:
                    mode, done = "wb", 0
                else:
                    raise RuntimeError(f"HTTP {r.status}")
                total = r.headers.get("content-length")
                total = (int(total) + done) if total else None
                last_mark = done // (256 * 1024 * 1024)
                with dest.open(mode) as fh:
                    while True:
                        chunk = r.read(1024 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
                        done += len(chunk)
                        mark = done // (256 * 1024 * 1024)
                        if mark != last_mark:
                            last_mark = mark
                            gb = done / 1024**3
                            extra = f"/{total / 1024**3:.1f}GB" if total else ""
                            print(f"  {dest.name}: {gb:.1f}GB{extra}", flush=True)
            if dest.stat().st_size < min_bytes:
                raise RuntimeError("file too small, likely an error page")
            print(f"  ok {dest.name} ({dest.stat().st_size / 1024**3:.2f}GB)")
            return dest
        except Exception as exc:  # noqa: BLE001
            have = dest.stat().st_size if dest.exists() else 0
            print(f"  ! attempt {attempt}: {exc} (kept {have} bytes)", flush=True)
    raise RuntimeError(f"failed to download {url}")


def _extracted_wav_dirs(root: Path) -> list[Path]:
    return sorted({p.parent for p in root.rglob("*.wav")})


def _flatten_wavs(sources: list[Path], dest: Path) -> Path:
    """Symlink every wav under sources into one flat dir (official loaders
    scandir a single level and choke on subdirectories). Idempotent."""
    dest.mkdir(parents=True, exist_ok=True)
    want = {}
    for src in sources:
        if src.is_file() and src.suffix == ".wav":
            want[f"{src.parent.name}__{src.name}"] = src
        elif src.is_dir():
            for wav in sorted(src.rglob("*.wav")):
                rel = wav.relative_to(src).parent
                tag = "__".join(rel.parts) if rel.parts else src.name
                want[f"{tag}__{wav.name}"] = wav
    have = {p.name for p in dest.iterdir()}
    for name, src in want.items():
        if name not in have:
            try:
                (dest / name).symlink_to(src)
            except FileExistsError:
                pass
    for stale in dest.iterdir():
        if stale.name not in want:
            stale.unlink()
    return dest


def _ensure_musan(background_dir: Path) -> list[Path]:
    """Download + extract MUSAN. Returns usable background dirs (maybe empty)."""
    dest = background_dir / "musan"
    if not _extracted_wav_dirs(dest):
        archive = background_dir / "musan.tar.gz"
        print(f"downloading MUSAN (~3GB) -> {archive}", flush=True)
        _download(MUSAN_URL, archive, min_bytes=1024**3)
        print("extracting MUSAN (one time, a few minutes)...", flush=True)
        with tarfile.open(archive) as tf:
            tf.extractall(dest)
    dirs = [dest / s for s in MUSAN_SUBDIRS if (dest / s).exists()]
    found = dirs or _extracted_wav_dirs(dest)
    print(f"MUSAN dirs: {[str(d) for d in found]}")
    return found


def _ensure_rirs(background_dir: Path) -> tuple[list[Path], list[Path]]:
    """Download + extract RIRs. Returns (rir_dirs, extra_noise_dirs)."""
    dest = background_dir / "rirs"
    if not _extracted_wav_dirs(dest):
        for url in RIR_URLS:
            archive = background_dir / Path(url).name
            try:
                print(f"downloading RIRs -> {archive}", flush=True)
                _download(url, archive, min_bytes=1024**2)
                print("extracting RIRs...", flush=True)
                if archive.suffix == ".zip":
                    with zipfile.ZipFile(archive) as zf:
                        zf.extractall(dest)
                else:
                    with tarfile.open(archive) as tf:
                        tf.extractall(dest)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  ! RIR source failed: {exc}", file=sys.stderr)
        else:
            print(
                "WARNING: no RIRs available, continuing without reverb", file=sys.stderr
            )
            return [], []
    wav_dirs = _extracted_wav_dirs(dest)
    rir_dirs = [
        d
        for d in wav_dirs
        if any(k in str(d).lower() for k in ("rir", "impulse", "reverb"))
    ]
    noise_dirs = [d for d in wav_dirs if d not in rir_dirs]
    print(f"RIR dirs: {[str(d) for d in rir_dirs]}")
    return rir_dirs or wav_dirs[:1], noise_dirs


def _ensure_assets(cfg: dict, repo: Path, force: bool = False) -> dict:
    """Download required assets. Returns resolved asset paths."""
    background_dir = repo / "data" / "background"
    background_dir.mkdir(parents=True, exist_ok=True)
    acav = _resolve(cfg["feature_data_files"]["ACAV100M_sample"], repo)
    fpval = _resolve(cfg["false_positive_validation_data_path"], repo)
    print(f"asset: ACAV features -> {acav}", flush=True)
    if force or not acav.exists():
        _download(ACAV_URL, acav, min_bytes=1024**3)
    print(f"asset: FP validation features -> {fpval}", flush=True)
    if force or not fpval.exists():
        _download(FPVAL_URL, fpval, min_bytes=1024**2)
    background_paths = [str(_resolve(p, repo)) for p in cfg.get("background_paths", [])]
    rir_paths = [str(_resolve(p, repo)) for p in cfg.get("rir_paths", [])]
    try:
        for d in _ensure_musan(background_dir):
            if str(d) not in background_paths:
                background_paths.append(str(d))
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: MUSAN unavailable ({exc}), continuing", file=sys.stderr)
    try:
        rirs, noises = _ensure_rirs(background_dir)
        for d in rirs:
            if str(d) not in rir_paths:
                rir_paths.append(str(d))
        for d in noises:
            if str(d) not in background_paths:
                background_paths.append(str(d))
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: RIRs unavailable ({exc}), continuing", file=sys.stderr)
    flat = background_dir / "_flat"
    _flatten_wavs([Path(p) for p in rir_paths], flat / "rirs")
    _flatten_wavs([Path(p) for p in background_paths], flat / "background")
    # Official train.py scandirs configured dirs into FILE lists before
    # augment_clips (it random.choice()s + loads entries directly, so any
    # subdirectory entry crashes it). Our flat dirs contain only wav symlinks.
    import os

    rir_files = [i.path for i in os.scandir(flat / "rirs") if i.is_file()]
    print(
        f"background wavs: {len(list((flat / 'background').iterdir()))}, "
        f"RIR wavs: {len(rir_files)}",
        flush=True,
    )
    if not list((flat / "background").iterdir()):
        print("WARNING: no background audio, augmentation runs dry", file=sys.stderr)
    return {
        "acav": acav,
        "fpval": fpval,
        "background_paths": [str(flat / "background")],
        "rir_paths": rir_files,
    }


def _ensure_oww_resources() -> None:
    """Download the frozen mel/embedding ONNX models the pip package omits."""
    import openwakeword

    target = Path(openwakeword.__file__).parent.resolve() / "resources" / "models"
    target.mkdir(parents=True, exist_ok=True)
    for name, mdl in openwakeword.FEATURE_MODELS.items():
        url = mdl["download_url"]
        if url.endswith(".tflite"):
            url = url[: -len(".tflite")] + ".onnx"
        dest = target / url.split("/")[-1]
        if dest.exists() and dest.stat().st_size > 1024:
            print(f"cached: {dest.name}")
            continue
        print(f"asset: OWW {name} -> {dest}", flush=True)
        _download(url, dest, min_bytes=1024)


def _collect_wavs(d: Path) -> list[str]:
    wavs = sorted(str(p) for p in d.glob("*.wav"))
    if not wavs:
        raise RuntimeError(f"no .wav files in {d} (run synthesize first)")
    return wavs


def _median_total_length(sample_wavs: list[str], n: int = 50) -> int:
    from scipy.io import wavfile

    import numpy as np
    import random

    durations = []
    for _ in range(min(n, len(sample_wavs))):
        _sr, dat = wavfile.read(random.choice(sample_wavs))
        durations.append(len(dat))
    total = int(round(float(np.median(durations)) / 1000) * 1000) + 12000
    if total < 32000:
        total = 32000
    elif abs(total - 32000) <= 4000:
        total = 32000
    print(f"clip total_length: {total} samples ({total / 16000:.2f}s)")
    return total


def _ort_has_cuda() -> bool:
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:  # noqa: BLE001
        return False


def _featurize_set(
    name: str,
    wav_paths: list[str],
    total_length: int,
    background_paths: list[str],
    rir_paths: list[str],
    batch_size: int,
    rounds: int,
    feature_file: Path,
    force: bool,
) -> None:
    done_marker = feature_file.with_suffix(feature_file.suffix + ".done")
    if done_marker.exists() and feature_file.exists() and not force:
        print(f"features cached: {feature_file.name}, skipping (use --force)")
        return
    if feature_file.exists():
        feature_file.unlink()
    if done_marker.exists():
        done_marker.unlink()
    from openwakeword.data import augment_clips
    from openwakeword.utils import compute_features_from_generator

    gen = augment_clips(
        wav_paths * max(1, rounds),
        total_length=total_length,
        batch_size=batch_size,
        background_clip_paths=background_paths,
        RIR_paths=rir_paths,
    )
    use_gpu = _ort_has_cuda()
    print(
        f"featurizing {name}: {len(wav_paths)} clips "
        f"(device={'gpu' if use_gpu else 'cpu'})",
        flush=True,
    )
    compute_features_from_generator(
        gen,
        n_total=len(wav_paths),
        clip_duration=total_length,
        output_file=str(feature_file),
        device="gpu" if use_gpu else "cpu",
        ncpu=1 if use_gpu else 8,
    )
    done_marker.touch()


def run_train(
    model_config: Path,
    synthetic_dir: Path,
    models_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Download assets, featurize, train on GPU, export ONNX. Returns model path."""
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA GPU visible (need --gpus all / nvidia runtime)")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f}GB)", flush=True)
    from src.audio_shim import install as _install_audio_shim

    _install_audio_shim()
    try:
        from openwakeword.train import Model
    except ImportError:
        raise RuntimeError(
            "openwakeword training code not importable "
            "(`uv pip install -r requirements.txt` inside the image)"
        ) from None

    cfg = _load_yaml(model_config)
    repo = Path.cwd()
    model_name = str(cfg.get("model_name", "wakeword"))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = models_dir / f"{model_name}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_config, run_dir / "custom_model.yaml")
    print(f"run dir: {run_dir}", flush=True)

    pos_train = _collect_wavs(synthetic_dir / "train" / "pos")
    neg_train = _collect_wavs(synthetic_dir / "train" / "neg")
    pos_test = _collect_wavs(synthetic_dir / "val" / "pos")
    neg_test = _collect_wavs(synthetic_dir / "val" / "neg")
    print(
        f"clips: {len(pos_train)} train pos, {len(neg_train)} train neg, "
        f"{len(pos_test)} val pos, {len(neg_test)} val neg"
    )

    assets = _ensure_assets(cfg, repo, force=force)
    _ensure_oww_resources()
    total_length = _median_total_length(pos_test)
    rounds = int(cfg.get("augmentation_rounds", 1))
    aug_batch = int(cfg.get("augmentation_batch_size", 16))
    feat = {
        "positive_train": run_dir / "positive_features_train.npy",
        "negative_train": run_dir / "negative_features_train.npy",
        "positive_test": run_dir / "positive_features_test.npy",
        "negative_test": run_dir / "negative_features_test.npy",
    }
    _featurize_set(
        "positive_train",
        pos_train,
        total_length,
        assets["background_paths"],
        assets["rir_paths"],
        aug_batch,
        rounds,
        feat["positive_train"],
        force,
    )
    _featurize_set(
        "negative_train",
        neg_train,
        total_length,
        assets["background_paths"],
        assets["rir_paths"],
        aug_batch,
        rounds,
        feat["negative_train"],
        force,
    )
    _featurize_set(
        "positive_test",
        pos_test,
        total_length,
        assets["background_paths"],
        assets["rir_paths"],
        aug_batch,
        rounds,
        feat["positive_test"],
        force,
    )
    _featurize_set(
        "negative_test",
        neg_test,
        total_length,
        assets["background_paths"],
        assets["rir_paths"],
        aug_batch,
        rounds,
        feat["negative_test"],
        force,
    )

    from openwakeword.data import mmap_batch_generator

    input_shape = np.load(feat["positive_test"], mmap_mode="r").shape[1:]
    print(f"model input shape: {input_shape}", flush=True)
    oww = Model(
        n_classes=1,
        input_shape=input_shape,
        model_type=str(cfg.get("model_type", "dnn")),
        layer_dim=int(cfg.get("layer_size", 32)),
    )

    def _reshape(x, n=input_shape[0]):
        if n > x.shape[1] or n < x.shape[1]:
            x = np.vstack(x)
            return np.array([x[i : i + n, :] for i in range(0, x.shape[0] - n, n)])
        return x

    feature_files = dict(cfg.get("feature_data_files", {}))
    feature_files = {k: str(_resolve(v, repo)) for k, v in feature_files.items()}
    data_transforms = {key: _reshape for key in feature_files}
    label_transforms = {}
    for key in ["positive", *feature_files, "adversarial_negative"]:
        label_transforms[key] = (
            (lambda x: [1 for _ in x])
            if key == "positive"
            else (lambda x: [0 for _ in x])
        )
    feature_files["positive"] = str(feat["positive_train"])
    feature_files["adversarial_negative"] = str(feat["negative_train"])

    batch_gen = mmap_batch_generator(
        feature_files,
        n_per_class=dict(cfg.get("batch_n_per_class", {})),
        data_transform_funcs=data_transforms,
        label_transform_funcs=label_transforms,
    )

    import torch.utils.data

    class IterDataset(torch.utils.data.IterableDataset):
        def __init__(self, generator):
            self.generator = generator

        def __iter__(self):
            return self.generator

    import os

    n_cpus = (os.cpu_count() or 2) // 2
    X_train = torch.utils.data.DataLoader(
        IterDataset(batch_gen),
        batch_size=None,
        num_workers=n_cpus,
        prefetch_factor=16,
    )

    X_val_fp = np.load(str(assets["fpval"]))
    X_val_fp = np.array(
        [
            X_val_fp[i : i + input_shape[0]]
            for i in range(0, X_val_fp.shape[0] - input_shape[0], 1)
        ]
    )
    X_val_fp_labels = np.zeros(X_val_fp.shape[0]).astype(np.float32)
    X_val_fp = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(X_val_fp), torch.from_numpy(X_val_fp_labels)
        ),
        batch_size=len(X_val_fp_labels),
    )

    X_val_pos = np.load(feat["positive_test"])
    X_val_neg = np.load(feat["negative_test"])
    labels = np.hstack(
        (np.ones(X_val_pos.shape[0]), np.zeros(X_val_neg.shape[0]))
    ).astype(np.float32)
    X_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(np.vstack((X_val_pos, X_val_neg))),
            torch.from_numpy(labels),
        ),
        batch_size=len(labels),
    )

    import json

    steps = int(cfg.get("steps", 50000))
    print(
        f"auto_train: steps={steps} "
        f"neg_weight={cfg.get('max_negative_weight')} "
        f"target_fp/hr={cfg.get('target_false_positives_per_hour')}",
        flush=True,
    )
    best_model = oww.auto_train(
        X_train=X_train,
        X_val=X_val,
        false_positive_val_data=X_val_fp,
        steps=steps,
        max_negative_weight=int(cfg.get("max_negative_weight", 1000)),
        target_fp_per_hour=float(cfg.get("target_false_positives_per_hour", 0.2)),
    )
    oww.export_model(model=best_model, model_name=model_name, output_dir=str(run_dir))
    with (run_dir / "history.json").open("w") as fh:
        json.dump(
            {
                k: [
                    float(np.asarray(v).ravel()[0]) if np.asarray(v).size else None
                    for v in vals
                ]
                if isinstance(vals, list)
                else str(vals)
                for k, vals in oww.history.items()
            },
            fh,
            indent=2,
        )
    onnx_path = run_dir / f"{model_name}.onnx"
    print(f"model saved: {onnx_path}", flush=True)
    return onnx_path
