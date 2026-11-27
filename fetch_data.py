"""Retrieve the three source files and verify their SHA-256 digests."""

import argparse
import hashlib
from pathlib import Path
import shutil
from urllib.request import Request, urlopen


STEVENS_REV = "29a03d0f5746a79359cb96faf3f8b4281d63f162"
STEVENS_ROOT = f"https://raw.githubusercontent.com/netlab-stevens/cloud-edge-latency/{STEVENS_REV}"
SOURCES = {
    "intel_lab.txt.gz": (
        "https://db.csail.mit.edu/labdata/data.txt.gz",
        "d99288c8f406ca6604d359ceaa0d8adfffa79e7095061a1e27dc4399f48c7225",
    ),
    "edge.csv": (
        f"{STEVENS_ROOT}/edge.csv",
        "bfc2792b8d1480c4689208c517d06b507002e707d445e623bea0a2ded4b3ffc1",
    ),
    "cloud.csv": (
        f"{STEVENS_ROOT}/cloud.csv",
        "94ce0e33e8796a4c9ccc10dfae7dd80e0834a6139ada18a622b42ab179d3c342",
    ),
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in SOURCES.items():
        path = data_dir / name
        if path.is_file() and digest(path) == expected:
            print(f"verified {name}")
            continue
        temporary = path.with_suffix(path.suffix + ".part")
        try:
            request = Request(url, headers={"User-Agent": "edge-cloud-iot-study/1.0"})
            with urlopen(request, timeout=90) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            if digest(temporary) != expected:
                raise ValueError(f"Source checksum changed for {name}. Source: {url}")
            temporary.replace(path)
            print(f"downloaded and verified {name}")
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    fetch(parser.parse_args().data_dir)
