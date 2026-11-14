#!/usr/bin/env python3
"""Download wake-word diagnostics clips from kiosk satellites.

Raw (immutable) layout:
  <raw_dir>/<device>/activations/<id>.wav + <id>.json
  <raw_dir>/<device>/near_misses/<id>.wav + <id>.json

Standalone: python download.py [--force]
Env (.env): KIOSK_HOSTS, KIOSK_PORT, KIOSK_PASSWORD
Stdlib only.
"""

import base64
import hashlib
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path


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


def main() -> int:
    import argparse

    load_dotenv(Path(".env"))
    ap = argparse.ArgumentParser(
        description="Download wake-word clips from kiosks -> raw/"
    )
    ap.add_argument("--hosts", default=os.environ.get("KIOSK_HOSTS", ""))
    ap.add_argument(
        "--port", type=int, default=int(os.environ.get("KIOSK_PORT", "2324"))
    )
    ap.add_argument("--password", default=os.environ.get("KIOSK_PASSWORD", ""))
    ap.add_argument("--raw-dir", default="raw")
    ap.add_argument("--timeout", type=int, default=15)
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the wav already exists",
    )
    args = ap.parse_args()

    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    if not hosts:
        print("ERROR: no hosts. Set KIOSK_HOSTS or use --hosts", file=sys.stderr)
        return 2
    if not args.password:
        print(
            "ERROR: no password. Set KIOSK_PASSWORD or use --password", file=sys.stderr
        )
        return 2
    new, _, fail = run_download(
        hosts, args.port, args.password, Path(args.raw_dir), args.timeout, args.force
    )
    return 1 if fail and not new else 0


if __name__ == "__main__":
    sys.exit(main())
