#!/usr/bin/env python3
import argparse, json, os, sys, time, shutil, subprocess
from urllib.request import urlopen, Request

DEFAULT_MANIFEST = "/opt/assets/manifest.json"  # baked or mounted path
WORKSPACE = os.environ.get("WORKSPACE_DIR", "/workspace")

def human(n): 
    return f"{n/1024/1024:.1f} MB"

def stream_download(url, out_path, chunk=1024*1024):
    tmp = out_path + ".part"
    with urlopen(Request(url, headers={"User-Agent": "wan2gp-bootstrap"})) as r, open(tmp, "wb") as f:
        total = 0
        while True:
            b = r.read(chunk)
            if not b: break
            f.write(b)
            total += len(b)
            if total % (50*chunk) == 0:
                print(f"… {human(total)}")
    os.replace(tmp, out_path)

def curl_fallback(url, out_path):
    cmd = ["curl", "-L", "--fail", "--retry", "3", "--retry-delay", "2", "-o", out_path, url]
    subprocess.check_call(cmd)

def ensure_file(item):
    dest_rel = item["dest"]
    url = item["url"]
    must_size = int(float(item.get("size_mb", 0)) * 1024 * 1024) if item.get("size_mb") else None

    dest_abs = os.path.join(WORKSPACE, dest_rel)
    os.makedirs(os.path.dirname(dest_abs), exist_ok=True)

    if os.path.exists(dest_abs):
        if must_size and os.path.getsize(dest_abs) >= must_size * 0.98:
            print(f"[OK] {dest_rel} already present ({human(os.path.getsize(dest_abs))})")
            return
        else:
            print(f"[WARN] {dest_rel} exists but looks incomplete — redownloading.")
            try: os.remove(dest_abs)
            except: pass

    print(f"[DL] {url}  ->  {dest_rel}")
    try:
        stream_download(url, dest_abs)
    except Exception as e:
        print(f"[WARN] urllib failed: {e}. Trying curl…")
        curl_fallback(url, dest_abs)

    if must_size:
        got = os.path.getsize(dest_abs)
        if got < must_size * 0.95:
            raise RuntimeError(f"Downloaded size too small for {dest_rel}: {human(got)} < ~{item['size_mb']} MB")
    print(f"[OK] {dest_rel} ready ({human(os.path.getsize(dest_abs))})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Path to manifest.json")
    args = ap.parse_args()

    if not os.path.exists(args.manifest):
        print(f"[ERR] Manifest not found: {args.manifest}", file=sys.stderr)
        sys.exit(2)

    with open(args.manifest, "r") as f:
        manifest = json.load(f)

    items = manifest.get("items", [])
    if not items:
        print("[INFO] Manifest has no items. Nothing to download.")
        return

    print(f"[BOOTSTRAP] Using workspace: {WORKSPACE}")
    for it in items:
        ensure_file(it)

if __name__ == "__main__":
    main()
