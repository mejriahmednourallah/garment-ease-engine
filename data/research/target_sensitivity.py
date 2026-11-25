#!/usr/bin/env python3
"""How much would using Dik et al.'s published ease preferences change a recommendation?

The Appendix 2 values are per-panel pattern adjustments, so they cannot be pasted into
EaseLab's circumference targets. This script quantifies the two things that *can* be
stated without inventing a mapping:

  1. the spread of each girth-like parameter across the five psychographic groups,
     compared with a typical size-grade step (~5 cm);
  2. how often the top-ranked size changes when the chest/waist/hip target is shifted
     across that spread, holding body and garment fixed.

No production constants are changed. Synthetic bodies are used only as fixed probe
points; this is a sensitivity analysis, not a validation.

Run:  python target_sensitivity.py
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "dik_2023_appendix2_pattern_ease_by_group.csv"
GROUPS = ["P1_cm", "P2_cm", "P3_cm", "P4_cm", "P5_cm"]
GIRTH_HINTS = ("chest", "waist", "hip", "sweep")
GRADE_STEP_CM = 5.0     # typical grading increment used by the synthetic charts (3-7 cm)

# EaseLab's current invented targets (app/main.py) and weights.
TARGETS = {"slim": (2, 1, 2), "regular": (8, 5, 8), "loose": (14, 12, 14)}
WEIGHTS = (0.4, 0.3, 0.3)

# Fixed probe: one plausible body and a graded chart around it.
BODY = {"chest": 96.0, "waist": 80.0, "hip": 102.0}
CHART = [("XS", 88.0, 72.0, 94.0), ("S", 94.0, 78.0, 100.0), ("M", 100.0, 84.0, 106.0),
         ("L", 106.0, 90.0, 112.0), ("XL", 112.0, 96.0, 118.0)]


def rank(body: dict, chart, targets, weights=WEIGHTS, stretch=0.0) -> list[str]:
    scored = []
    for label, c, w, h in chart:
        s = 0.0
        for value, garment, target, weight in zip(("chest", "waist", "hip"), (c, w, h), targets, weights):
            required = body[value] + target - body[value] * stretch / 100.0
            s += weight * (garment - required) ** 2
        scored.append((round(s, 6), label))
    scored.sort()
    return [label for _, label in scored]


def girth_rows():
    with SRC.open() as fh:
        for row in csv.DictReader(fh):
            if any(h in row["parameter"].lower() for h in GIRTH_HINTS):
                vals = [float(row[g]) for g in GROUPS]
                yield row["garment"], row["parameter"], vals


def main() -> int:
    print("## 1. Spread across psychographic groups, girth-like parameters\n")
    print(f"{'garment':22s} {'parameter':30s} {'min':>6s} {'max':>6s} {'spread':>7s} {'spread/grade':>13s}")
    spreads = []
    for garment, parameter, vals in girth_rows():
        lo, hi = min(vals), max(vals)
        spread = hi - lo
        spreads.append(spread)
        print(f"{garment:22s} {parameter[:30]:30s} {lo:6.2f} {hi:6.2f} {spread:7.2f} {spread / GRADE_STEP_CM:13.2f}")
    print(f"\nmedian spread {statistics.median(spreads):.2f} cm, max {max(spreads):.2f} cm, "
          f"grade step {GRADE_STEP_CM:.1f} cm")

    print("\n## 2. Does the top-ranked size change when the target moves across that spread?\n")
    base = TARGETS["regular"]
    flips = 0
    total = 0
    rows = []
    for garment, parameter, vals in girth_rows():
        lo, hi = min(vals), max(vals)
        regime = "circumference-like" if max(vals) >= GRADE_STEP_CM / 2 else "panel-like"
        for delta in (lo, (lo + hi) / 2, hi):
            target = (base[0] + delta, base[1] + delta, base[2] + delta)
            top = rank(BODY, CHART, target)[0]
            rows.append((garment, parameter[:26], regime, f"{delta:+.2f}", top))
        tops = {r[4] for r in rows if r[0] == garment and r[1] == parameter[:26]}
        total += 1
        flips += len(tops) > 1

    for r in rows:
        print(f"{r[0]:22s} {r[1]:26s} {r[2]:18s} delta {r[3]:>6s} -> top {r[4]}")
    print(f"\n{flips}/{total} parameters change the recommended size across their group range.")

    print("\n## 3. Reference: current invented targets vs published group extremes\n")
    for pref, t in TARGETS.items():
        print(f"{pref:8s} chest/waist/hip {t} -> top {rank(BODY, CHART, t)[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
