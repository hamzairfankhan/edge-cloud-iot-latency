"""Benchmark IoT kernels and replay paired public edge and cloud RTTs."""

import argparse
from collections import Counter
from datetime import date
import gzip
import json
from pathlib import Path
import platform
from statistics import median
from time import process_time_ns

import numpy as np
import pandas as pd
from scipy import stats

from fetch_data import SOURCES, digest


SEED = 20261127
BANDS = ("Good", "Normal", "Poor")
WORKLOADS = ("Light", "Medium", "Heavy")
WINDOW_SIZE = 64
WINDOW_COUNT = 160
REPEATS = 5
BOOTSTRAPS = 2000
CLOUD_SPEED = 2.0
SPEEDS = (1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 6, 8)


def load_sensor_windows(path):
    counts = Counter()
    unique = {}
    bounds = ((-10, 60), (0, 100), (0, 100000), (1.5, 3.5))
    with gzip.open(path, "rt", encoding="ascii", errors="replace") as stream:
        for line in stream:
            counts["raw"] += 1
            fields = line.split()
            if len(fields) != 8:
                counts["malformed"] += 1
                continue
            day, clock, epoch, mote, *features = fields
            try:
                day = date.fromisoformat(day)
                hour, minute, second = (float(value) for value in clock.split(":"))
                epoch, mote = int(epoch), int(mote)
                values = tuple(map(float, features))
            except (ValueError, OverflowError):
                counts["malformed"] += 1
                continue
            if not (date(2004, 2, 28) <= day <= date(2004, 4, 5)
                    and 0 <= hour < 24 and 0 <= minute < 60 and 0 <= second < 60
                    and epoch >= 0 and 1 <= mote <= 54
                    and all(lo <= value <= hi for value, (lo, hi) in zip(values, bounds))):
                counts["outside_bounds"] += 1
                continue
            key = (mote, epoch)
            if key in unique:
                counts["duplicate_mote_epoch"] += 1
                continue
            unique[key] = (day.toordinal() * 86400 + hour * 3600 + minute * 60 + second, values)
    counts["clean"] = len(unique)
    counts["clean_motes"] = len({key[0] for key in unique})
    candidates = []
    current = []
    previous = (-1, -2, 0)
    for (mote, epoch), (stamp, values) in sorted(unique.items()):
        if mote != previous[0] or not 1 <= epoch - previous[1] <= 5 or not 0 < stamp - previous[2] <= 180:
            current = []
        current.append((mote, epoch, *values))
        if len(current) == WINDOW_SIZE:
            candidates.append(current)
            current = []
        previous = (mote, epoch, stamp)
    counts["eligible_windows"] = len(candidates)
    if len(candidates) < WINDOW_COUNT:
        raise ValueError(f"Only {len(candidates)} eligible sensor windows; need {WINDOW_COUNT}.")
    index = np.random.default_rng(SEED).choice(len(candidates), WINDOW_COUNT, replace=False)
    chosen = [candidates[i] for i in index]
    payloads = [json.dumps(window, separators=(",", ":")).encode("ascii") for window in chosen]
    byte_sizes = [len(payload) for payload in payloads]
    counts.update(sample_rows=len(chosen) * WINDOW_SIZE, sample_motes=len({w[0][0] for w in chosen}),
                  sample_windows=len(chosen), json_bytes_min=min(byte_sizes), json_bytes_max=max(byte_sizes),
                  json_bytes_median=int(median(byte_sizes)))
    return payloads, dict(counts)


def load_network(data_dir):
    edge = pd.read_csv(data_dir / "edge.csv", usecols=["prb", "Minlatency"])
    cloud = pd.read_csv(data_dir / "cloud.csv", usecols=["prb", "minMedian"])
    pairs = edge.merge(cloud, on="prb", validate="one_to_one").sort_values("prb").reset_index(drop=True)
    n_matched = len(pairs)
    pairs = pairs.rename(columns={"Minlatency": "edge_rtt", "minMedian": "cloud_rtt"})
    for name in ("edge_rtt", "cloud_rtt"):
        pairs[name] = pd.to_numeric(pairs[name], errors="coerce")
    valid = np.isfinite(pairs[["edge_rtt", "cloud_rtt"]]).all(axis=1)
    valid &= (pairs[["edge_rtt", "cloud_rtt"]] > 0).all(axis=1)
    pairs = pairs.loc[valid].reset_index(drop=True)
    return pairs.drop(columns="prb"), {
        "edge_rows": len(edge), "cloud_rows": len(cloud), "matched": n_matched,
        "invalid_pairs": n_matched - len(pairs), "usable_pairs": len(pairs),
        "unmatched_edge": len(edge) - n_matched, "unmatched_cloud": len(cloud) - n_matched,
    }


def light(rows):
    temperatures = [row[2] for row in rows]
    return sum(temperatures) / len(rows), sum(temp > 30 for temp in temperatures)


def medium(rows):
    # Smooth all four channels and score deviations using a nine-reading median.
    channels = tuple(zip(*(row[2:] for row in rows)))
    return tuple(sum(abs(series[i] - median(series[i - 4:i + 5]))
                     for i in range(4, len(series) - 4)) for series in channels)


def reference_profiles():
    # Seeded synthetic exemplars are a compute workload, not training data.
    rng = np.random.default_rng(SEED + 10)
    anchors = rng.uniform(-0.15, 1.15, size=(2048, 8, 4))
    x = np.linspace(0, 7, WINDOW_SIZE)
    return [tuple(np.stack([np.interp(x, np.arange(8), profile[:, c])
                            for c in range(4)], axis=1).ravel().tolist())
            for profile in anchors]


def heavy(rows, profiles):
    # Brute-force nearest reference-profile search with squared Euclidean distance.
    features = tuple(value for row in rows
                     for value in (row[2] / 45, row[3] / 100, row[4] / 5000, (row[5] - 1.5) / 2))
    best = (float("inf"), -1)
    for index, profile in enumerate(profiles):
        distance = 0.0
        for actual, target in zip(features, profile):
            difference = actual - target
            distance += difference * difference
        if distance < best[0]:
            best = (distance, index)
    return best


def benchmark(payloads):
    profiles = reference_profiles()
    kernels = (light, medium, lambda rows: heavy(rows, profiles))
    rng = np.random.default_rng(SEED + 20)
    records = []
    for name, kernel in zip(WORKLOADS, kernels):
        for payload in payloads[:5]:
            kernel(json.loads(payload))
        timings = np.empty((len(payloads), REPEATS))
        for repeat in range(REPEATS):
            for index in rng.permutation(len(payloads)):
                payload = payloads[index]
                start = process_time_ns()
                kernel(json.loads(payload))
                timings[index, repeat] = (process_time_ns() - start) / 1e6
        # process_time_ns is coarse on some Windows hosts. Batch invocations to
        # get a positive, sufficiently resolved per-request CPU time.
        if (timings <= 0).mean() > 0.01 or np.median(timings) < 1:
            batch = {"Light": 768, "Medium": 192, "Heavy": 8}[name]
            for repeat in range(REPEATS):
                for index in rng.permutation(len(payloads)):
                    payload = payloads[index]
                    start = process_time_ns()
                    for _ in range(batch):
                        kernel(json.loads(payload))
                    timings[index, repeat] = (process_time_ns() - start) / (batch * 1e6)
        else:
            batch = 1
        for index, values in enumerate(timings):
            records.append({"workload": name, "window": index, "invocations_per_timing": batch,
                            **{f"repeat_{j + 1}_ms": value for j, value in enumerate(values)},
                            "median_ms": float(np.median(values))})
        print(f"{name}: {np.median(timings):.4f} ms per request ({batch} per timing)", flush=True)
    result = pd.DataFrame.from_records(records)
    if not np.isfinite(result["median_ms"]).all() or (result["median_ms"] <= 0).any():
        raise ValueError("CPU timing resolution was insufficient for at least one window.")
    return result


def bands_for(rtt):
    lower, upper = np.quantile(rtt, [1 / 3, 2 / 3])
    band = np.select([rtt <= lower, rtt <= upper], BANDS[:2], default=BANDS[2])
    return band, (lower, upper)


def bootstrap_means(edge, cloud, rng, b=BOOTSTRAPS):
    edge_means = []
    cloud_means = []
    for start in range(0, b, 100):
        samples = rng.integers(len(edge), size=(min(100, b - start), len(edge)))
        edge_means.extend(edge[samples].mean(axis=1))
        cloud_means.extend(cloud[samples].mean(axis=1))
    edge_means, cloud_means = np.asarray(edge_means), np.asarray(cloud_means)
    return tuple(np.quantile(x, [.025, .975]) for x in (edge_means, cloud_means, cloud_means - edge_means))


def holm(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    corrected = np.empty_like(p_values)
    corrected[order] = np.minimum(1, np.maximum.accumulate(p_values[order] * (len(order) - np.arange(len(order)))))
    return corrected


def metrics(values, prefix):
    q25, q50, q75, q95 = np.quantile(values, [.25, .5, .75, .95])
    return {f"{prefix}_mean_ms": values.mean(), f"{prefix}_median_ms": q50,
            f"{prefix}_p95_ms": q95, f"{prefix}_sd_ms": values.std(ddof=1),
            f"{prefix}_iqr_ms": q75 - q25}


def classify(value, tolerance=1e-9):
    if abs(value) <= tolerance:
        return "Tie"
    return "Edge" if value > 0 else "Cloud"


def analyze(pairs, timings, cloud_speed=CLOUD_SPEED):
    rng = np.random.default_rng(SEED + 30)
    edge = pairs["edge_rtt"].to_numpy()
    cloud = pairs["cloud_rtt"].to_numpy()
    gap = cloud - edge
    labels, cuts = bands_for(cloud)
    alt_labels, alt_cuts = bands_for((cloud + edge) / 2)
    windows = np.resize(rng.permutation(WINDOW_COUNT), len(pairs))
    rng.shuffle(windows)
    network = []
    summary = []
    sensitivity = []
    for band in BANDS:
        mask = labels == band
        network.append({"network": band, "n": mask.sum(), **metrics(edge[mask], "edge"),
                        **metrics(cloud[mask], "cloud"), **metrics(gap[mask], "gap"),
                        "network_edge_win_pct": 100 * (gap[mask] > 0).mean()})
        for workload in WORKLOADS:
            cpu = timings.loc[timings.workload == workload].sort_values("window")["median_ms"].to_numpy()[windows]
            edge_total, cloud_total = edge[mask] + cpu[mask], cloud[mask] + cpu[mask] / cloud_speed
            delta = cloud_total - edge_total
            edge_ci, cloud_ci, delta_ci = bootstrap_means(edge_total, cloud_total, rng)
            test = stats.ttest_rel(cloud_total, edge_total)
            p95_delta = np.quantile(cloud_total, .95) - np.quantile(edge_total, .95)
            mean_gap = gap[mask].mean()
            mean_cpu = cpu[mask].mean()
            if mean_gap <= 0:
                crossover = 1.0
            elif mean_gap < mean_cpu:
                crossover = mean_cpu / (mean_cpu - mean_gap)
            else:
                crossover = None
            summary.append({
                "network": band, "workload": workload, "n": mask.sum(), "cloud_speed": cloud_speed,
                **metrics(edge_total, "edge"), **metrics(cloud_total, "cloud"),
                "edge_ci_low_ms": edge_ci[0], "edge_ci_high_ms": edge_ci[1],
                "cloud_ci_low_ms": cloud_ci[0], "cloud_ci_high_ms": cloud_ci[1],
                "delta_mean_ms": delta.mean(), "delta_ci_low_ms": delta_ci[0], "delta_ci_high_ms": delta_ci[1],
                "delta_p95_ms": p95_delta, "relative_edge_saving_pct": 100 * delta.mean() / cloud_total.mean(),
                "edge_win_pct": 100 * (delta > 0).mean(), "tie_pct": 100 * (delta == 0).mean(),
                "mean_network_gap_ms": mean_gap, "edge_cpu_mean_ms": mean_cpu, "mean_crossover_speed": crossover,
                "p_paired": test.pvalue, "mean_winner": classify(delta.mean()), "p95_winner": classify(p95_delta),
            })
            for grouping, groups in (("Cloud RTT", labels), ("Mean paired RTT", alt_labels)):
                subset = groups == band
                for speed in sorted(set(SPEEDS + (cloud_speed,))):
                    et = edge[subset] + cpu[subset]
                    ct = cloud[subset] + cpu[subset] / speed
                    advantages = ct - et
                    sensitivity.append({"grouping": grouping, "network": band, "workload": workload,
                                        "cloud_speed": speed, "n": subset.sum(),
                                        "mean_delta_ms": advantages.mean(),
                                        "p95_delta_ms": np.quantile(ct, .95) - np.quantile(et, .95),
                                        "edge_win_pct": 100 * (advantages > 0).mean(),
                                        "mean_winner": classify(advantages.mean())})
    summary = pd.DataFrame.from_records(summary)
    summary["holm_p"] = holm(summary["p_paired"])
    return pd.DataFrame.from_records(network), summary, pd.DataFrame.from_records(sensitivity), cuts, alt_cuts


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    sha = {name: digest(args.data_dir / name) for name in SOURCES}
    if any(sha[name] != SOURCES[name][1] for name in SOURCES):
        raise ValueError("Dataset checksum mismatch. Run fetch_data.py to restore the exact sources.")
    pairs, network_counts = load_network(args.data_dir)
    if args.replay_timing:
        timings = pd.read_csv(args.replay_timing)
    else:
        payloads, sensor_counts = load_sensor_windows(args.data_dir / "intel_lab.txt.gz")
        print(f"{sensor_counts['clean']:,} valid sensor rows, {len(payloads)} selected windows", flush=True)
        timings = benchmark(payloads)
        timing_path = args.out / "benchmark.csv"
        timings.to_csv(timing_path, index=False)
        metadata = {"sensor": sensor_counts, "system": platform.platform(), "cpu": platform.processor(),
                    "python": platform.python_version(), "seed": SEED, "repeats": REPEATS,
                    "clock": "process_time_ns"}
        timing_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    timing_source = args.replay_timing or timing_path
    provenance = json.loads(timing_source.with_suffix(".json").read_text(encoding="utf-8"))
    if (len(timings) != len(WORKLOADS) * WINDOW_COUNT
            or set(timings["workload"]) != set(WORKLOADS)
            or any(set(group["window"]) != set(range(WINDOW_COUNT)) for _, group in timings.groupby("workload"))
            or not np.isfinite(timings["median_ms"]).all()
            or (timings["median_ms"] <= 0).any()):
        raise ValueError("Invalid benchmark timing data.")
    network, matrix, sensitivity, cuts, alt_cuts = analyze(pairs, timings, args.cloud_speed)
    for name, table in (("network", network), ("matrix", matrix), ("sensitivity", sensitivity)):
        table.to_csv(args.out / f"{name}.csv", index=False)
    metadata = dict(provenance, network=network_counts, cloud_rtt_cuts_ms=list(cuts),
                    mean_rtt_cuts_ms=list(alt_cuts), bootstrap_draws=BOOTSTRAPS,
                    cloud_speed=args.cloud_speed, input_sha256=sha, timing_sha256=digest(timing_source))
    (args.out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(matrix[["network", "workload", "edge_mean_ms", "cloud_mean_ms", "delta_mean_ms",
                  "edge_win_pct", "mean_winner", "p95_winner"]].round(2).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--replay-timing", type=Path, help="Use recorded benchmark CSV and matching JSON")
    parser.add_argument("--cloud-speed", type=float, default=CLOUD_SPEED, help="Simulated speed relative to edge")
    args = parser.parse_args()
    if not np.isfinite(args.cloud_speed) or args.cloud_speed < 1:
        parser.error("--cloud-speed must be finite and at least 1")
    run(args)
