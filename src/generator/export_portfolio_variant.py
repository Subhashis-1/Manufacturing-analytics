"""Generate a separate, reproducible synthetic input variant for comparison.

The canonical baseline configuration and its regression outputs are untouched.
All generated files are written under the gitignored data/synthetic directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC / "generator") not in sys.path:
    sys.path.insert(0, str(_SRC / "generator"))

from factory_generator import (  # noqa: E402
    default_config,
    portfolio_variant_config,
    simulate,
    theoretical_utilization,
)


OUT = Path(__file__).resolve().parents[2] / "data" / "synthetic"


def summarize(log, lifecycle, cfg) -> dict:
    """Calculate comparable steady-state KPIs for one generated run."""
    t0, t1 = cfg.warmup_hours, cfg.horizon_hours
    window = t1 - t0
    completed = lifecycle.dropna(subset=["completion_time"])
    in_window = completed[
        (completed["completion_time"] >= t0)
        & (completed["completion_time"] <= t1)
    ]
    cycle_time = in_window["completion_time"] - in_window["arrival_time"]
    throughput = len(in_window) / window

    arrival = lifecycle["arrival_time"].to_numpy()
    completion = lifecycle["completion_time"].fillna(t1).to_numpy()
    overlap = np.clip(
        np.minimum(completion, t1) - np.maximum(arrival, t0), 0, None
    )
    measured_wip = float(overlap.sum() / window)
    predicted_wip = float(throughput * cycle_time.mean())
    ll_gap = (
        abs(measured_wip - predicted_wip) / measured_wip
        if measured_wip > 0
        else float("nan")
    )

    utilization = {}
    for station, station_cfg in cfg.stations.items():
        operations = log[log["station"] == station]
        start = operations["process_start_time"].clip(lower=t0, upper=t1)
        end = operations["process_complete_time"].clip(lower=t0, upper=t1)
        busy = float((end - start).clip(lower=0).sum())
        slots = station_cfg.n_tools * station_cfg.batch_size * window
        utilization[station] = busy / slots

    return {
        "seed": cfg.seed,
        "arrival_rate_lots_per_hour": cfg.arrival_rate,
        "litho_mean_processing_hours": cfg.stations["LITHO"].pt_mean,
        "completed_lots_in_window": int(len(in_window)),
        "throughput_lots_per_hour": float(throughput),
        "mean_cycle_time_hours": float(cycle_time.mean()),
        "little_law_relative_gap": float(ll_gap),
        "empirical_bottleneck": max(utilization, key=utilization.get),
        "station_slot_utilization": {
            station: float(value) for station, value in utilization.items()
        },
        "theoretical_slot_utilization": theoretical_utilization(cfg),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_cfg = default_config(seed=42)
    variant_cfg = portfolio_variant_config(seed=42)

    baseline_log, baseline_lifecycle, _ = simulate(baseline_cfg)
    variant_log, variant_lifecycle, variant_meta = simulate(variant_cfg)
    baseline = summarize(baseline_log, baseline_lifecycle, baseline_cfg)
    variant = summarize(variant_log, variant_lifecycle, variant_cfg)

    event_path = OUT / "portfolio_variant_event_log.csv"
    lifecycle_path = OUT / "portfolio_variant_lot_lifecycle.csv"
    summary_path = OUT / "portfolio_variant_comparison.json"
    variant_log.to_csv(event_path, index=False)
    variant_lifecycle.to_csv(lifecycle_path, index=False)

    comparison = {
        "description": (
            "Synthetic variant using the same methodology and seed, with small "
            "arrival-rate and LITHO processing-time changes."
        ),
        "baseline": baseline,
        "variant": variant,
        "delta_variant_minus_baseline": {
            "throughput_lots_per_hour": (
                variant["throughput_lots_per_hour"]
                - baseline["throughput_lots_per_hour"]
            ),
            "mean_cycle_time_hours": (
                variant["mean_cycle_time_hours"]
                - baseline["mean_cycle_time_hours"]
            ),
        },
        "variant_metadata": variant_meta,
    }
    summary_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Synthetic portfolio variant generated (seed 42).")
    print(f"  Baseline bottleneck : {baseline['empirical_bottleneck']}")
    print(f"  Variant bottleneck  : {variant['empirical_bottleneck']}")
    print(
        "  Throughput          : "
        f"{baseline['throughput_lots_per_hour']:.4f} -> "
        f"{variant['throughput_lots_per_hour']:.4f} lots/hour"
    )
    print(
        "  Mean cycle time     : "
        f"{baseline['mean_cycle_time_hours']:.3f} -> "
        f"{variant['mean_cycle_time_hours']:.3f} hours"
    )
    print(f"  Event log           : {event_path}")
    print(f"  Lot lifecycle       : {lifecycle_path}")
    print(f"  Comparison summary  : {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
