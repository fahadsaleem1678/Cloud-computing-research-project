"""Aggregate per-request logs and calculate paired HKD-versus-greedy effects."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lower, upper = int(position), min(int(position) + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def one_run(path: Path, tolerance: float) -> dict:
    metadata = json.loads(path.with_suffix(".metadata.json").read_text())
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(records) != metadata["requests"]:
        raise ValueError(f"{path}: expected {metadata['requests']} records, found {len(records)}")
    completed = [r for r in records if r["status"] == "ok"]
    latencies = [r["latency_ms"] for r in completed]
    duration_min = metadata["requests"] / metadata["rate_hz"] / 60
    mismatches = sum(r["max_abs_diff"] > tolerance for r in completed)
    return {"file": str(path), "model": metadata["model_name"],
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "trace": Path(metadata["trace"]).stem, "policy": metadata["policy"],
            "seed": metadata["seed"], "requests": len(records), "completed": len(completed),
            "p50_ms": percentile(latencies, .50), "p95_ms": percentile(latencies, .95),
            "p99_ms": percentile(latencies, .99),
            "miss_pct": 100 * sum(r["deadline_violation"] for r in records) / len(records),
            "switches_per_min": sum(r["changed"] for r in records) / duration_min,
            "throughput_per_s": len(completed) / (duration_min * 60),
            "mean_decision_ms": mean(r["decision_ms"] for r in records),
            "max_schedule_lag_ms": max(r["schedule_lag_ms"] for r in records),
            "mean_wire_bytes": mean(r["sent_bytes"] + r["received_bytes"] for r in completed) if completed else None,
            "accuracy_pct": 100 * sum(r["prediction"] == r["target"] for r in completed) / len(completed) if completed else None,
            "max_abs_diff": max((r["max_abs_diff"] for r in completed), default=None),
            "numerical_mismatches": mismatches, "shaping_enabled": metadata["shaping_enabled"]}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_effects(runs: list[dict], bootstrap_samples: int) -> list[dict]:
    grouped = defaultdict(dict)
    for row in runs:
        key = (row["model"], row["checkpoint_sha256"], row["trace"], row["seed"])
        grouped[key][row["policy"]] = row
    by_setting = defaultdict(list)
    for (model, digest, trace, seed), pair in grouped.items():
        if "HKD" in pair and "greedy" in pair:
            for metric in ("p95_ms", "miss_pct", "switches_per_min"):
                a, b = pair["HKD"][metric], pair["greedy"][metric]
                if a is not None and b is not None:
                    by_setting[(model, digest, trace, metric)].append(a - b)
    rng = random.Random(1731)
    rows = []
    for (model, digest, trace, metric), diffs in sorted(by_setting.items()):
        boot = sorted(mean(rng.choices(diffs, k=len(diffs))) for _ in range(bootstrap_samples))
        rows.append({"model": model, "checkpoint_sha256": digest, "trace": trace,
                     "metric": metric, "paired_runs": len(diffs), "hkd_minus_greedy": mean(diffs),
                     "ci_low": boot[int(.025 * (bootstrap_samples - 1))],
                     "ci_high": boot[int(.975 * (bootstrap_samples - 1))]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--runs-output", type=Path, required=True)
    parser.add_argument("--paired-output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("At least 100 bootstrap samples are required")
    runs = [one_run(path, args.tolerance) for path in args.logs]
    write_csv(args.runs_output, runs)
    write_csv(args.paired_output, paired_effects(runs, args.bootstrap_samples))
    print(f"Analyzed {len(runs)} runs")


if __name__ == "__main__":
    main()
