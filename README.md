# Edge or Cloud? IoT latency trace replay

This R&D experiment asks when a faster processor makes up for a longer network round trip. It combines historical RIPE Atlas RTT summaries with local CPU timings for three workloads using real Intel Berkeley Lab sensor payloads.

At an illustrative 2× cloud CPU speed, **edge had lower mean latency in seven of nine scenarios**. Cloud led on heavy work in the good and normal RTT bands. For good-network light and medium work, edge led on mean while cloud led on P95.

| Nearest cloud RTT band | Light | Medium | Heavy |
| --- | --- | --- | --- |
| Good, ≤11.11 ms | Edge by 0.69 ms | Edge by 0.67 ms | Cloud by 6.99 ms |
| Normal, through 21.51 ms | Edge by 4.87 ms | Edge by 4.84 ms | Cloud by 2.81 ms |
| Poor, >21.51 ms | Edge by 22.75 ms | Edge by 22.72 ms | Edge by 15.06 ms |

The numbers are differences between the edge and cloud **means** at 2×. The cutoffs are rounded. Source output includes exact cutoffs, tail performance and uncertainty.

## Live demo

Open **Demo.html** in a browser. It works offline. Select Good / Heavy at **2×**, change to **1×**, then compare Poor / Heavy at **2×**. Switch to Good / Light to see why mean and P95 can favor different tiers. Edit RTT and CPU inputs in the calculator to test a single request live.

## Reproduce the published numbers

Use Python 3.12. In a virtual environment, run:

~~~bash
python -m pip install -r requirements.txt
python fetch_data.py
python study.py --replay-timing results/benchmark.csv --out replay
python verify.py
python figures.py --out replay/figures
python demo.py --out replay/Demo.html
~~~

The download script checks SHA-256 for all three files (about 52 MB). If the host is temporarily unavailable, place an original copy in the data folder and run the script to verify it. The exact replay uses 480 saved timing summaries. The independent verifier rebuilds the raw CSV pairing and checks the means, P95, win shares, parity and analytical crossovers.

### Measure the local host

~~~bash
python study.py --out rerun
python verify.py --results rerun
python figures.py --results rerun --out rerun/figures
python demo.py --results rerun --out rerun/Demo.html
~~~

This path cleans the full compressed sensor archive and measures 160 windows per workload. It can take several minutes. A fresh benchmark measures the reader's host and can shift the crossover. Set the optional cloud-speed parameter to a modeled ratio of at least 1. The executed notebook gives a portable, read-only walkthrough. Install requirements-notebook.txt for its dependencies.

## Method and scope

- **Network.** Inner join the nearest observed edge and cloud RTT by Atlas probe. The result contains 8,456 usable probes. Bands use empirical cloud RTT tertiles. A second grouping uses the mean paired RTT. Neither grouping applies experimental network degradation.
- **Payload.** A seeded sample of 160 non-overlapping, same-mote windows (10,240 readings) becomes compact JSON. Each probe receives the same window across tiers and workloads.
- **Compute.** Local CPU time covers JSON decode and kernel execution. A seeded reference bank creates the heavy workload. Cloud CPU time is the recorded local time divided by a modeled speed factor.
- **Stats.** Positive cloud-minus-edge values favor edge. Outputs report mean, median, P95, SD, IQR, paired bootstrap intervals, paired t-tests with Holm adjustment, and probe win shares.
- **Scope.** RTT is historical, CPU timings come from one host, and the replay omits queueing, transfer, protocol setup and sensor acquisition. Probe dependence and window reuse constrain inference.

The mean rule is simple. Prefer edge when its RTT saving exceeds its extra compute time. Compare the percentile tied to the operational objective before choosing a deployment.

## Files

| File | Purpose |
| --- | --- |
| fetch_data.py and study.py | Pinned inputs, cleaning, CPU benchmark and paired replay |
| verify.py | Independent numerical checks |
| figures.py and figures/ | Chart generator and saved figures |
| Demo.html, demo.py and demo_template.html | Offline demo and its generator |
| Edge_Cloud_IoT_Analysis.ipynb | Executed methodology and reproduction |
| results/benchmark.csv and benchmark.json | Recorded timing summaries and setup metadata |
| results/network.csv and matrix.csv | Network strata and nine reference comparisons |
| results/sensitivity.csv and run.json | Speed sweep, alternative grouping and run manifest |

## Primary sources

- [Intel Berkeley Lab sensor archive, MIT CSAIL](https://db.csail.mit.edu/labdata/labdata.html). We acknowledge the original dataset contributors.
- [Stevens Network Lab latency repository at the analyzed revision](https://github.com/netlab-stevens/cloud-edge-latency/tree/29a03d0f5746a79359cb96faf3f8b4281d63f162)
- [Charyyev, Arslan and Gunes, IEEE GLOBECOM 2020](https://doi.org/10.1109/GLOBECOM42002.2020.9322406). This describes the original RTT measurements.
- [Altaie, Tahir and Alfathe, JESA 2026](https://doi.org/10.18280/jesa.590710). This benchmarks an IoT application using an alternative simulation design.
- [Ali-Eldin, Wang and Shenoy, SC21](https://doi.org/10.1145/3458817.3476142). This demonstrates queueing-driven edge performance inversion.
- [Zhang et al., IEEE Transactions on Cloud Computing 2024](https://doi.org/10.1109/TCC.2024.3452094). This measures network-delay variation at scale.

Raw data remain available from their original sources under their original terms.
