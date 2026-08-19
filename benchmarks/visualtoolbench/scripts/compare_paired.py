#!/usr/bin/env python3
"""Paired bootstrap CI and sign-flip test for two rubric-scored methods."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    weight = index - lo
    return values[lo] * (1 - weight) + values[hi] * weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--method-a", required=True)
    parser.add_argument("--method-b", required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    rows = {}
    for line in Path(args.scores).read_text().splitlines():
        row = json.loads(line)
        if row.get("ok"):
            rows[row["id"]] = row
    differences = []
    for row in rows.values():
        methods = row["audit"]["methods"]
        left = methods[args.method_a]["rubric_correct"]
        right = methods[args.method_b]["rubric_correct"]
        differences.append(sum(right) / len(right) - sum(left) / len(left))

    rng = random.Random(args.seed)
    n = len(differences)
    observed = sum(differences) / n
    bootstrap = [
        sum(differences[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(args.bootstrap)
    ]
    extreme = 0
    for _ in range(args.permutations):
        statistic = sum(
            value if rng.getrandbits(1) else -value for value in differences
        ) / n
        extreme += abs(statistic) >= abs(observed)
    report = {
        "n": n,
        "method_a": args.method_a,
        "method_b": args.method_b,
        "mean_difference_percentage_points": observed * 100,
        "bootstrap_95_ci_percentage_points": [
            percentile(bootstrap, 0.025) * 100,
            percentile(bootstrap, 0.975) * 100,
        ],
        "two_sided_sign_flip_p": (extreme + 1) / (args.permutations + 1),
        "significant_at_0.05": (extreme + 1) / (args.permutations + 1) < 0.05,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
