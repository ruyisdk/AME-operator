#!/usr/bin/env python3
"""Independent logical INT32 GEMM: Python integers, no AME/model code."""
import argparse
import json
from pathlib import Path


def gemm(a, b):
    if len(a) != 4 or len(b) != 4 or any(len(row) != 4 for row in a + b):
        raise ValueError("reference requires two 4x4 matrices")
    if any(type(v) is not int or not -(1 << 31) <= v < (1 << 31)
           for row in a + b for v in row):
        raise ValueError("input is not INT32")
    result = [[0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            # This bound also excludes overflowing intermediate sums.
            bound = 0
            for k in range(4):
                product = a[i][k] * b[k][j]
                result[i][j] += product
                bound += abs(product)
            if bound > (1 << 31) - 1:
                raise ValueError("input exceeds exact no-overflow tutorial contract")
    return result


def compare(observation):
    expected = gemm(observation["a"], observation["b"])
    actual = observation["c"]
    if len(actual) != 4 or any(len(row) != 4 for row in actual):
        raise ValueError("output is not 4x4")
    mismatches = [dict(row=i, column=j, expected=expected[i][j], actual=actual[i][j])
                  for i in range(4) for j in range(4)
                  if expected[i][j] != actual[i][j]]
    return dict(passed=not mismatches, checked_elements=16,
                expected=expected, actual=actual, mismatches=mismatches,
                arithmetic="Python arbitrary precision; exact INT32 no-overflow")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation", type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.observation.read_text()))
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
