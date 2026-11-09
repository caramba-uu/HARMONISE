# Rebuilds the large published files from the compressed chunks in files/chunks/.
# GitHub doesn't accept files over 100 MB, so the big models and result files are stored gzipped and split.
#
#   pixi run merge                    # rebuild the files (run once after pixi install)
#   python merge_files.py --split     # re-create the chunks after changing a large file
import argparse
import gzip
import hashlib
import json
import os

FILES_DIR = "files"
CHUNK_DIR = "files/chunks"
MANIFEST = f"{CHUNK_DIR}/manifest.json"
CHUNK_SIZE = 45 * 1024 * 1024
MIN_SIZE = 50 * 1024 * 1024  # files above this size get split


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split():
    big = []
    for root, _, names in os.walk(FILES_DIR):
        if root.startswith(CHUNK_DIR):
            continue
        for name in names:
            path = os.path.join(root, name)
            if os.path.getsize(path) > MIN_SIZE:
                big.append(path)

    manifest = {}
    for path in sorted(big):
        rel = os.path.relpath(path, FILES_DIR)
        with open(path, "rb") as f:
            packed = gzip.compress(f.read(), compresslevel=6)
        out_dir = os.path.join(CHUNK_DIR, os.path.dirname(rel))
        os.makedirs(out_dir, exist_ok=True)
        parts = []
        for i in range(0, len(packed), CHUNK_SIZE):
            part = os.path.join(CHUNK_DIR, f"{rel}.gz.part{i // CHUNK_SIZE:02d}")
            with open(part, "wb") as f:
                f.write(packed[i:i + CHUNK_SIZE])
            parts.append(os.path.relpath(part, CHUNK_DIR))
        manifest[rel] = {"sha256": sha256(path), "parts": parts}
        print(f"{rel}: {os.path.getsize(path) / 1e6:.0f} MB -> {len(parts)} parts")

    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {MANIFEST} ({len(manifest)} files)")


def merge():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    for rel, info in manifest.items():
        target = os.path.join(FILES_DIR, rel)
        if os.path.exists(target) and sha256(target) == info["sha256"]:
            print(f"{rel}: already there")
            continue
        packed = b"".join(open(os.path.join(CHUNK_DIR, p), "rb").read() for p in info["parts"])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(gzip.decompress(packed))
        if sha256(target) != info["sha256"]:
            raise RuntimeError(f"{rel}: checksum mismatch, the chunks may be incomplete")
        print(f"{rel}: rebuilt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true", help="split the large files into chunks instead of merging")
    split() if parser.parse_args().split else merge()
