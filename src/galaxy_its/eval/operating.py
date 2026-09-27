"""Operating-point evaluation: event detection and latency at a false-alarm budget.

Window-level F1 thresholds give unusable false-alarm rates in a vehicle (hundreds per hour).
Here an alarm is an *event*: the k-of-n rule fires when >= k of the last n windows exceed the
threshold; a false alarm is a rising edge of that rule on attack-free traffic. For each budget
(false alarms per hour) the threshold is the lowest one meeting the budget on validation data,
then test attacks are scored by whether/when the rule first fires inside the attack episode.

Resolution caveat: the budget is only as precise as the amount of attack-free validation
driving allows (1 alarm in 10 min of driving = 6/h).
"""
from __future__ import annotations

import numpy as np


def k_of_n(above: np.ndarray, k: int, n: int) -> np.ndarray:
    c = np.concatenate([[0], np.cumsum(above.astype(np.int64))])
    idx = np.arange(1, len(above) + 1)
    return (c[idx] - c[np.maximum(idx - n, 0)]) >= k


def _rising(a: np.ndarray) -> int:
    return int(a[0]) + int((a[1:] & ~a[:-1]).sum()) if len(a) else 0


def false_alarms_per_hour(seqs: list[tuple[np.ndarray, np.ndarray]], thr: float, k: int, n: int, dt: float):
    """seqs: (scores, inj) per sequence in time order; counts alarm events on windows before
    any attack (whole sequence if attack-free). Also returns the alarm duty cycle: a threshold so
    low that the alarm never clears has just one rising edge per recording and would otherwise
    look like a very low false-alarm rate."""
    events, hours, on, tot = 0, 0.0, 0, 0
    for s, inj in seqs:
        clean_end = int(np.argmax(inj)) if inj.any() else len(inj)
        a = k_of_n(s[:clean_end] >= thr, k, n)
        events += _rising(a); hours += clean_end * dt / 3600
        on += int(a.sum()); tot += len(a)
    return events / max(hours, 1e-9), hours, on / max(tot, 1)


def threshold_for_budget(val_seqs, budget: float, k: int, n: int, dt: float,
                         max_duty: float = 0.01) -> float:
    """Most sensitive threshold whose false-alarm rate meets the budget and whose alarm is active
    for at most `max_duty` of attack-free time."""
    cand = np.unique(np.concatenate([s for s, _ in val_seqs]))
    cand = cand[np.linspace(0, len(cand) - 1, min(len(cand), 400)).astype(int)]
    for t in cand:                                        # ascending: first feasible = most sensitive
        fa, _, duty = false_alarms_per_hour(val_seqs, t, k, n, dt)
        if fa <= budget and duty <= max_duty:
            return float(t)
    return float(cand[-1]) + 1e-6


def event_detection(test_seqs, names, thr: float, k: int, n: int, dt: float) -> list[dict]:
    rows = []
    for (s, inj), name in zip(test_seqs, names):
        a = k_of_n(s >= thr, k, n)
        if not inj.any():
            continue
        on = int(np.argmax(inj))
        off = on + int(np.argmax(inj[on:] == 0)) if (inj[on:] == 0).any() else len(inj)
        hit = np.where(a[on:off])[0]
        rows.append({"name": name, "detected": bool(len(hit)),
                     "latency_ms": float(hit[0] * dt * 1000) if len(hit) else np.nan,
                     "alarm_before_onset": bool(a[:on].any())})
    return rows


def event_curve(test_seqs, names, k: int, n: int, dt: float, n_points: int = 60) -> list[dict]:
    """Event-level trade-off: events detected vs false alarms/h, sweeping the threshold on test.

    Model quality independent of threshold calibration (which transfers imperfectly between
    recordings). The calibrated operating point from `operating_report` is the deployable
    estimate; this curve is the ranking.
    """
    cand = np.unique(np.concatenate([s for s, _ in test_seqs]))
    cand = cand[np.linspace(0, len(cand) - 1, min(len(cand), n_points)).astype(int)]
    out = []
    for t in cand:
        fa, hrs, duty = false_alarms_per_hour(test_seqs, t, k, n, dt)
        rows = event_detection(test_seqs, names, t, k, n, dt)
        out.append({"threshold": float(t), "fa_per_h": fa, "duty": duty,
                    "detected": sum(r["detected"] for r in rows), "events": len(rows)})
    return out


def operating_report(val_seqs, test_seqs, test_names, dt: float, budgets=(60, 10),
                     rules=((1, 1), (2, 3), (3, 5))) -> list[dict]:
    out = []
    for k, n in rules:
        for b in budgets:
            thr = threshold_for_budget(val_seqs, b, k, n, dt)
            fa, hrs, duty = false_alarms_per_hour(test_seqs, thr, k, n, dt)
            rows = event_detection(test_seqs, test_names, thr, k, n, dt)
            out.append({"rule": f"{k}-of-{n}", "budget_per_h": b, "threshold": thr,
                        "test_false_alarms_per_h": fa, "test_clean_hours": hrs,
                        "test_alarm_duty": duty, "events": rows})
    return out
