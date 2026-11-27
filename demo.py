"""Build the offline live-demo page from the study's aggregate CSV files."""

import argparse
import json
from pathlib import Path

import pandas as pd


def build(results, output):
    metadata = json.loads((results / "run.json").read_text(encoding="utf-8"))
    tables = {name: json.loads(pd.read_csv(results / f"{name}.csv").to_json(orient="records", double_precision=10))
              for name in ("network", "matrix", "sensitivity")}
    data = json.dumps({**tables, "meta": {key: metadata[key] for key in ("network", "cloud_speed")}},
                      separators=(",", ":"))
    template = Path(__file__).with_name("demo_template.html").read_text(encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.replace("__STUDY_DATA__", data), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("Demo.html"))
    args = parser.parse_args()
    build(args.results, args.out)
