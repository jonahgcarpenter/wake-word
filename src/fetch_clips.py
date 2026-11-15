"""Fetch satellite debug clips into data/raw/. Library module, called via main.py only.

Raw (immutable) layout:
  data/raw/<device>/activations/<id>.wav + <id>.json
  data/raw/<device>/near_misses/<id>.wav + <id>.json
"""

import base64
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def api_post(
    host: str,
    port: int,
    endpoint: str,
    payload: dict,
    token: str | None = None,
    timeout: int = 15,
) -> dict:
    url = f"http://{host}:{port}{endpoint}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        raise RuntimeError(f"HTTP {e.code} {url}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Connection failed {url}: {e}") from e


def extract_token(login_resp: dict) -> str:
    if isinstance(login_resp.get("token"), str):
        return login_resp["token"]
    d = login_resp.get("data")
    if isinstance(d, str):
        return d
    if isinstance(d, dict) and isinstance(d.get("token"), str):
        return d["token"]
    raise RuntimeError(f"Login response has no token: {login_resp}")


def sanitize(name: str) -> str:
    return (
        "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_")
        or "device"
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(
    host,
    port,
    token,
    clip_id,
    wav_path: Path,
    json_path: Path,
    meta: dict,
    timeout: int,
    seen_hashes: dict,
) -> str | None:
    """Fetch one clip. Returns 'downloaded', 'deduped', or None on failure.

    Dedupe rules (no device-side delete needed, ring buffer overwrites itself):
    - same ID + identical audio hash on disk -> 'deduped', no rewrite
    - same audio hash under a different ID -> still saved (labels may differ)
      but sidecar records duplicate_of so training can drop it
    """
    resp = api_post(
        host,
        port,
        "/api/commands/getWakeWordActivationAudio",
        {"id": clip_id},
        token=token,
        timeout=timeout,
    )
    data = resp.get("data", resp)
    b64 = data.get("base64") if isinstance(data, dict) else None
    if not b64:
        print(f"  ! {clip_id}: no base64 in response, skipping", file=sys.stderr)
        return None
    try:
        audio = base64.b64decode(b64)
    except Exception as e:
        print(f"  ! {clip_id}: base64 decode failed: {e}", file=sys.stderr)
        return None
    new_hash = hashlib.sha256(audio).hexdigest()

    # Same ID already on disk with identical bytes -> dedupe, keep existing.
    if wav_path.exists():
        try:
            if file_sha256(wav_path) == new_hash:
                if not json_path.exists():
                    _write_sidecar(
                        host,
                        clip_id,
                        meta,
                        data,
                        wav_path,
                        json_path,
                        new_hash,
                        seen_hashes.get(new_hash),
                    )
                print(f"  = [{meta.get('label', '?')}] {clip_id}.wav (dedupe)")
                seen_hashes.setdefault(new_hash, clip_id)
                return "deduped"
        except Exception as e:
            print(
                f"  ! {clip_id}: hash check failed ({e}), re-writing", file=sys.stderr
            )

    wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav_path.write_bytes(audio)
    _write_sidecar(
        host,
        clip_id,
        meta,
        data,
        wav_path,
        json_path,
        new_hash,
        seen_hashes.get(new_hash),
    )
    seen_hashes[new_hash] = clip_id
    return "downloaded"


def _write_sidecar(
    host, clip_id, meta, data, wav_path, json_path, audio_hash, duplicate_of
) -> None:
    sidecar = {
        "id": clip_id,
        "device_host": host,
        **meta,
        "audio_file": wav_path.name,
        "audio_sha256": audio_hash,
        "raw_response_keys": list(data.keys()) if isinstance(data, dict) else [],
    }
    if isinstance(data, dict):
        for k, v in data.items():
            if k != "base64" and k not in sidecar:
                sidecar[k] = v
    if duplicate_of is not None and duplicate_of != clip_id:
        sidecar["duplicate_of"] = duplicate_of
    if "wakeWord" not in sidecar or "engine" not in sidecar:
        print(
            f"  ! {clip_id}: API response has no model info (wakeWord/engine missing)",
            file=sys.stderr,
        )
    json_path.write_text(json.dumps(sidecar, indent=2))


def process_device(
    host: str,
    port: int,
    password: str,
    raw_base: Path,
    timeout: int,
    force: bool = False,
) -> tuple[int, int, int]:
    """Returns (new, deduped, failed). Idempotent: existing IDs are skipped
    unless force=True."""
    host = host.strip()
    if not host:
        return 0, 0, 0
    dev_dir = raw_base / sanitize(host)
    print(f"== {host} -> {dev_dir}/")
    login_resp = api_post(
        host, port, "/api/login", {"password": password}, timeout=timeout
    )
    token = extract_token(login_resp)
    resp = api_post(
        host,
        port,
        "/api/commands/getWakeWordActivations",
        {},
        token=token,
        timeout=timeout,
    )
    data = resp.get("data", resp)
    activations = data.get("activations", []) if isinstance(data, dict) else []
    near_misses = (
        data.get("nearMisses", data.get("near_misses", []))
        if isinstance(data, dict)
        else []
    )
    print(f"  found {len(activations)} activations, {len(near_misses)} near misses")
    new = deduped = fail = 0
    # Seed content hashes from what's already on disk so re-runs and
    # cross-kind repeats (same bytes, different ID) are caught.
    seen_hashes: dict = {}
    for existing in sorted(dev_dir.rglob("*.wav")):
        try:
            seen_hashes.setdefault(file_sha256(existing), existing.stem)
        except Exception:
            pass
    for kind, clips in (("activations", activations), ("near_misses", near_misses)):
        for a in clips:
            cid = str(a.get("id", "unknown"))
            wav_path = dev_dir / kind / f"{cid}.wav"
            json_path = dev_dir / kind / f"{cid}.json"
            if not force and wav_path.exists():
                deduped += 1
                continue
            meta = {k: v for k, v in a.items() if k != "id"}
            meta["label"] = kind
            try:
                status = download_one(
                    host,
                    port,
                    token,
                    a["id"],
                    wav_path,
                    json_path,
                    meta,
                    timeout,
                    seen_hashes,
                )
                if status == "downloaded":
                    new += 1
                    print(f"  + [{kind}] {cid}.wav")
                elif status == "deduped":
                    deduped += 1
                else:
                    fail += 1
            except Exception as e:
                fail += 1
                print(f"  ! [{kind}] {cid}: {e}", file=sys.stderr)
    print(f"  done {host}: {new} new, {deduped} deduped, {fail} failed")
    return new, deduped, fail


def run_download(
    hosts: list[str],
    port: int,
    password: str,
    raw_dir: Path,
    timeout: int = 15,
    force: bool = False,
) -> tuple[int, int, int]:
    total_new = total_deduped = total_fail = 0
    for h in hosts:
        try:
            new, deduped, fail = process_device(
                h, port, password, raw_dir, timeout, force
            )
            total_new += new
            total_deduped += deduped
            total_fail += fail
        except Exception as e:
            total_fail += 1
            print(f"!! {h} failed: {e}", file=sys.stderr)
    print(
        f"\nTOTAL: {total_new} new, {total_deduped} deduped, "
        f"{total_fail} failed -> {raw_dir}/"
    )
    return total_new, total_deduped, total_fail
