"""Independently check saved tables against the two raw sources and timings."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def read_minimum(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        next(reader)
        seen = set()
        valid = {}
        for row in reader:
            key = int(row[0])
            assert key not in seen, f"Repeated source probe: {key}"
            seen.add(key)
            try:
                value = float(row[1])
            except ValueError:
                continue
            if math.isfinite(value) and value > 0:
                valid[key] = value
    return seen, valid


def verify(data, results):
    meta = json.loads((results / "run.json").read_text(encoding="utf-8"))
    edges, edge_by_id = read_minimum(data / "edge.csv")
    clouds, cloud_by_id = read_minimum(data / "cloud.csv")
    ordered = sorted(edge_by_id.keys() & cloud_by_id.keys())
    assert (len(edges), len(clouds), len(ordered)) == (
        meta["network"]["edge_rows"], meta["network"]["cloud_rows"], meta["network"]["usable_pairs"],
    )
    edge_values = np.array([edge_by_id[key] for key in ordered])
    cloud_values = np.array([cloud_by_id[key] for key in ordered])
    bands = ("Good", "Normal", "Poor")
    cuts = np.quantile(cloud_values, [1 / 3, 2 / 3])
    np.testing.assert_allclose(cuts, meta["cloud_rtt_cuts_ms"])
    group = np.searchsorted(cuts, cloud_values, side="left")
    benchmark = pd.read_csv(results / "benchmark.csv")
    repetitions = [f"repeat_{j}_ms" for j in range(1, meta["repeats"] + 1)]
    np.testing.assert_allclose(benchmark[repetitions].median(axis=1), benchmark["median_ms"])
    assert set(benchmark.groupby("workload").size()) == {160}
    assert meta["sensor"]["raw"] == sum(meta["sensor"].get(key, 0) for key in
                                        ("malformed", "outside_bounds", "duplicate_mote_epoch", "clean"))

    matrix = pd.read_csv(results / "matrix.csv")
    sensitivity = pd.read_csv(results / "sensitivity.csv")
    assert len(matrix) == 9
    assert len(sensitivity) == 18 * sensitivity["cloud_speed"].nunique()
    assert (matrix["delta_ci_low_ms"] <= matrix["delta_mean_ms"]).all()
    assert (matrix["delta_ci_high_ms"] >= matrix["delta_mean_ms"]).all()
    assert (matrix["holm_p"] >= matrix["p_paired"]).all()
    reference = sensitivity[(sensitivity.grouping == "Cloud RTT")
                            & (sensitivity.cloud_speed == meta["cloud_speed"])]
    np.testing.assert_allclose(matrix["delta_mean_ms"], reference["mean_delta_ms"])
    np.testing.assert_allclose(matrix["delta_p95_ms"], reference["p95_delta_ms"])
    parity = sensitivity[(sensitivity.grouping == "Cloud RTT") & (sensitivity.cloud_speed == 1)]
    rng = np.random.default_rng(meta["seed"] + 30)
    assigned = np.resize(rng.permutation(160), len(ordered))
    rng.shuffle(assigned)
    for k, band in enumerate(bands):
        mask = group == k
        gap = cloud_values[mask] - edge_values[mask]
        np.testing.assert_allclose(parity.loc[parity.network == band, "mean_delta_ms"], gap.mean())
        for row in matrix[matrix.network == band].itertuples():
            assert row.n == mask.sum()
            timing = benchmark[benchmark.workload == row.workload].sort_values("window_id" if "window_id" in benchmark else "window")
            cpu = timing.median_ms.to_numpy()[assigned[mask]]
            edge_total = edge_values[mask] + cpu
            cloud_total = cloud_values[mask] + cpu / row.cloud_speed
            expected = (edge_total.mean(), cloud_total.mean(), np.percentile(edge_total, 95),
                        np.percentile(cloud_total, 95), 100 * np.mean(cloud_total > edge_total))
            observed = (row.edge_mean_ms, row.cloud_mean_ms, row.edge_p95_ms, row.cloud_p95_ms, row.edge_win_pct)
            np.testing.assert_allclose(observed, expected, atol=1e-8)
            if np.isfinite(row.mean_crossover_speed) and row.mean_crossover_speed > 1:
                np.testing.assert_allclose(gap.mean(), cpu.mean() * (1 - 1 / row.mean_crossover_speed), atol=1e-9)
    print(f"Verified {len(ordered):,} probe pairs, {len(matrix)} baseline cells, and {len(sensitivity)} sensitivity rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    verify(args.data_dir, args.results)
