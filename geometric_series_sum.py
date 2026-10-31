#!/usr/bin/env python3
"""Compute the sum of a geometric series.

Given an initial term (a), common ratio (r), and number of terms (n),
this script outputs the sum of the finite geometric series:
    S_n = a * (1 - r^n) / (1 - r),  when r ≠ 1
    S_n = a * n,                    when r = 1

For an infinite geometric series (|r| < 1):
    S = a / (1 - r)
"""

import sys


def finite_sum(a: float, r: float, n: int) -> float:
    """Sum of the first n terms of a geometric series."""
    if n <= 0:
        raise ValueError("Number of terms n must be a positive integer.")
    if r == 1:
        return a * n
    return a * (1 - r ** n) / (1 - r)


def infinite_sum(a: float, r: float) -> float:
    """Sum of an infinite geometric series (converges only when |r| < 1)."""
    if abs(r) >= 1:
        raise ValueError(
            f"Infinite series diverges for |r| = {abs(r)} >= 1. "
            "Use finite_sum() instead."
        )
    return a / (1 - r)


def main():
    print("Geometric Series Sum Calculator")
    print("=" * 40)

    try:
        a = float(input("Enter the initial term (a): "))
        r = float(input("Enter the common ratio (r): "))
        mode = input("Finite or infinite series? (f/i) [f]: ").strip().lower() or "f"

        if mode == "i":
            result = infinite_sum(a, r)
            print(f"\nSum of infinite geometric series: {result}")
        else:
            n = int(input("Enter the number of terms (n): "))
            result = finite_sum(a, r, n)
            print(f"\nSum of first {n} terms: {result}")

    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
