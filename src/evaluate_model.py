"""Evaluate model on real clips. Library module, called via main.py only.

Validation only on data/classified/: recall, FP/hr, ROC sweep -> metrics.json.
Never trains; consumes models from data/models/<model_name>/.
"""

import csv
import json
import wave
from pathlib import Path


def _clip_hours(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                return 0.0
            return w.getnframes() / float(w.getframerate() or 16000) / 3600
    except Exception:  # noqa: BLE001
        return 0.0


def _max_score(predictions: list) -> float:
    best = 0.0
    for frame in predictions:
        for score in frame.values():
            try:
                best = max(best, float(score))
            except (TypeError, ValueError):
                continue
    return best


def run_evaluate(model_path: Path, classified_dir: Path, output_json: Path) -> dict:
    """Score model_path against data/classified/, write metrics to output_json."""
    try:
        import openwakeword
        from openwakeword import Model
    except ImportError:
        raise RuntimeError(
            "openwakeword is required (`uv pip install -r requirements.txt`)"
        ) from None

    # The pip package omits the frozen feature models; fetch them on demand.
    target = Path(openwakeword.__file__).parent.resolve() / "resources" / "models"
    target.mkdir(parents=True, exist_ok=True)
    import urllib.request

    for mdl in openwakeword.FEATURE_MODELS.values():
        url = mdl["download_url"]
        if url.endswith(".tflite"):
            url = url[: -len(".tflite")] + ".onnx"
        dest = target / url.split("/")[-1]
        if not dest.exists() or dest.stat().st_size < 1024:
            print(f"asset: OWW {dest.name}", flush=True)
            urllib.request.urlretrieve(url, dest)

    manifest = classified_dir / "manifest.csv"
    if not manifest.exists():
        raise RuntimeError(f"no manifest at {manifest} (run label first)")
    oww = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
    name = next(iter(oww.models.keys()))
    print(f"loaded model: {name} ({model_path})", flush=True)

    scored: list[tuple[str, float]] = []
    hours = {"true": 0.0, "false": 0.0}

    def _resolve_clip(raw: str) -> Path | None:
        # Manifests may store absolute paths or paths relative to an older
        # layout (classified/... vs data/classified/...). Try each base.
        candidates = [Path(raw)]
        if not candidates[0].is_absolute():
            candidates.append(Path.cwd() / raw)
            candidates.append(classified_dir / Path(raw).name)
            candidates.append(classified_dir / raw)
            candidates.append(classified_dir.parent / raw)
            for sub in ("true", "false", "ambient", "stt_only"):
                candidates.append(classified_dir / sub / Path(raw).name)
        for cand in candidates:
            if cand.is_file():
                return cand
        return None

    with manifest.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("human_label")]
    print(f"scoring {len(rows)} labeled clips...", flush=True)
    for n, row in enumerate(rows, 1):
        label = row["human_label"]
        clip = _resolve_clip(row.get("classified_wav") or "")
        if clip is None:
            print(f"  ! missing, skipping: {row.get('classified_wav')}", flush=True)
            continue
        try:
            preds = oww.predict_clip(str(clip))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! failed {clip.name}: {exc}", flush=True)
            continue
        scored.append((label, _max_score(preds)))
        hours[label] = hours.get(label, 0.0) + _clip_hours(clip)
        if n % 100 == 0:
            print(f"  ... {n}/{len(rows)}", flush=True)

    true_scores = sorted(s for lab, s in scored if lab == "true")
    false_scores = sorted(s for lab, s in scored if lab == "false")
    n_true, n_false = len(true_scores), len(false_scores)
    print(
        f"scored: {n_true} true, {n_false} false "
        f"({hours.get('true', 0):.2f}h true audio, "
        f"{hours.get('false', 0):.2f}h false audio)",
        flush=True,
    )
    if not true_scores and not false_scores:
        raise RuntimeError("no clips could be scored")

    thresholds = [round(t / 100, 2) for t in range(5, 100, 5)]
    table = []
    for thr in thresholds:
        recall = (sum(s >= thr for s in true_scores) / n_true) if n_true else 0.0
        fp = sum(s >= thr for s in false_scores)
        fp_rate = (fp / n_false) if n_false else 0.0
        fp_per_hr = (fp / hours["false"]) if hours.get("false") else 0.0
        table.append(
            {
                "threshold": thr,
                "recall": round(recall, 4),
                "false_positives": fp,
                "false_positive_rate": round(fp_rate, 4),
                "false_positives_per_hour": round(fp_per_hr, 2),
            }
        )
    # Recommended operating point: best recall with zero false positives,
    # falling back to the best Youden index (recall + (1 - fp_rate) - 1).
    zero_fp = [r for r in table if r["false_positives"] == 0]
    if zero_fp:
        best = max(zero_fp, key=lambda r: r["recall"])
        reason = "highest recall with zero false positives"
    else:
        best = max(table, key=lambda r: r["recall"] - r["false_positive_rate"])
        reason = "best youden index (no zero-FP threshold exists)"
    metrics = {
        "model": str(model_path),
        "n_true_clips": n_true,
        "n_false_clips": n_false,
        "true_audio_hours": round(hours.get("true", 0), 3),
        "false_audio_hours": round(hours.get("false", 0), 3),
        "note": (
            "false clips are adversarially selected activations, "
            "so FP/hr here is pessimistic vs random ambient audio"
        ),
        "thresholds": table,
        "recommended_threshold": best["threshold"],
        "recommended_reason": reason,
        "recommended_recall": best["recall"],
        "recommended_false_positives_per_hour": best["false_positives_per_hour"],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as fh:
        json.dump(metrics, fh, indent=2)
    print(
        f"recommended threshold {best['threshold']} ({reason}): "
        f"recall={best['recall']}, "
        f"FP/hr={best['false_positives_per_hour']}",
        flush=True,
    )
    print(f"metrics -> {output_json}", flush=True)
    return metrics
