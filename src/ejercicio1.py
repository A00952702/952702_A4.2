# src/ejercicio1.py
"""Compute statisticsmean, median, mode, standard deviation, variance."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


RESULTS_FILENAME = "StatisticsResults.txt"


# pylint: disable=too-many-instance-attributes
@dataclass
class StatsResult:
    """Contiene los resultados estadísticos del archivo."""
    total_items_in_file: int
    valid_count: int
    mean: Optional[float]
    median: Optional[float]
    mode: Optional[float]
    std_dev: Optional[float]   # population SD (divide by n)
    variance: Optional[float]  # sample variance (divide by n-1)
    elapsed_seconds: float


def _is_letter(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


def parse_number_line(line: str) -> Tuple[Optional[float], Optional[str]]:
    """Parses a line into a float"""
    raw = line.strip()
    if not raw:
        return None, "Empty line"

    # First try direct float
    try:
        return float(raw), None
    except ValueError:
        pass

    # Try: numeric prefix + trailing letters only (e.g., 405s)
    idx = 0
    n = len(raw)

    # optional sign
    if idx < n and raw[idx] in "+-":
        idx += 1

    saw_digit = False
    saw_dot = False

    while idx < n:
        ch = raw[idx]
        if "0" <= ch <= "9":
            saw_digit = True
            idx += 1
            continue
        if ch == "." and not saw_dot:
            saw_dot = True
            idx += 1
            continue
        break

    if not saw_digit:
        return None, f"Invalid number: {raw}"

    num_part = raw[:idx]
    suffix = raw[idx:]

    # suffix must be letters only
    if suffix and not all(_is_letter(c) for c in suffix):
        return None, f"Invalid number format: {raw}"

    try:
        return float(num_part), f"Coerced '{raw}' -> {num_part}"
    except ValueError:
        return None, f"Invalid number: {raw}"


def merge_sort(values: List[float]) -> List[float]:
    """Basic merge sort (avoids relying on sorted())."""
    if len(values) <= 1:
        return values[:]
    mid = len(values) // 2
    left = merge_sort(values[:mid])
    right = merge_sort(values[mid:])
    return merge(left, right)


def merge(left: List[float], right: List[float]) -> List[float]:
    """Basic merge """
    merged: List[float] = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    while i < len(left):
        merged.append(left[i])
        i += 1
    while j < len(right):
        merged.append(right[j])
        j += 1
    return merged


def compute_mean(values: List[float]) -> float:
    """Basic compute_mean"""
    total = 0.0
    for x in values:
        total += x
    return total / float(len(values))


def compute_median(sorted_values: List[float]) -> float:
    """Basic compute_median"""
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def compute_mode(values: List[float]) -> Optional[float]:
    """Basic compute_mode"""
    counts: dict[float, int] = {}
    for x in values:
        counts[x] = counts.get(x, 0) + 1

    best_value: Optional[float] = None
    best_count = 0

    for val, cnt in counts.items():
        if cnt > best_count:
            best_count = cnt
            best_value = val
        elif cnt == best_count and cnt > 1 and best_value is not None:
            # Tie-breaker: choose the largest value
            best_value = max(best_value, val)

    if best_count <= 1:
        return None
    return best_value


def compute_variance_sample(values: List[float], mean: float) -> Optional[float]:
    """Basic variance_sample"""
    n = len(values)
    if n < 2:
        return None
    total = 0.0
    for x in values:
        diff = x - mean
        total += diff * diff
    return total / float(n - 1)


def compute_std_dev_population(values: List[float], mean: float) -> Optional[float]:
    """Basic compute_std_dev_population"""
    n = len(values)
    if n < 1:
        return None
    total = 0.0
    for x in values:
        diff = x - mean
        total += diff * diff
    variance_pop = total / float(n)
    # sqrt without math.sqrt (basic algorithm): Newton's method
    return sqrt_newton(variance_pop)


def sqrt_newton(value: float) -> float:
    """Basic sqrt_newton"""
    if value <= 0.0:
        return 0.0
    guess = value
    for _ in range(30):
        guess = 0.5 * (guess + value / guess)
    return guess


def format_number(value: Optional[float]) -> str:
    """Basic format_number"""
    if value is None:
        return "#N/A"
    # Match the style in your errata (keeps decimals reasonable)
    return f"{value:.10f}".rstrip("0").rstrip(".")


# pylint: disable=too-many-locals
def run(file_path: str) -> StatsResult:
    """Procesa el archivo y calcula estadísticas."""
    start = time.perf_counter()

    total_lines = 0
    values: List[float] = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line_num, line in enumerate(file, start=1):
            total_lines += 1
            val, err = parse_number_line(line)
            if val is None:
                print(f"[Line {line_num}] Invalid data -> '{line.strip()}': {err}")
                continue
            if err is not None:
                print(f"[Line {line_num}] Warning: {err}")
            values.append(val)

    if not values:
        elapsed = time.perf_counter() - start
        return StatsResult(
            total_items_in_file=total_lines,
            valid_count=0,
            mean=None,
            median=None,
            mode=None,
            std_dev=None,
            variance=None,
            elapsed_seconds=elapsed,
        )

    sorted_values = merge_sort(values)
    mean = compute_mean(values)
    median = compute_median(sorted_values)
    mode = compute_mode(values)
    variance = compute_variance_sample(values, mean)
    std_dev = compute_std_dev_population(values, mean)

    elapsed = time.perf_counter() - start

    return StatsResult(
        total_items_in_file=total_lines,
        valid_count=len(values),
        mean=mean,
        median=median,
        mode=mode,
        std_dev=std_dev,
        variance=variance,
        elapsed_seconds=elapsed,
    )


def write_results(result: StatsResult) -> None:
    """write_results"""
    lines = [
        f"COUNT: {result.total_items_in_file}",
        f"VALID_COUNT: {result.valid_count}",
        f"MEAN: {format_number(result.mean)}",
        f"MEDIAN: {format_number(result.median)}",
        f"MODE: {format_number(result.mode)}",
        f"SD: {format_number(result.std_dev)}",
        f"VARIANCE: {format_number(result.variance)}",
        f"ELAPSED_SECONDS: {result.elapsed_seconds:.6f}",
    ]

    for line in lines:
        print(line)

    with open(RESULTS_FILENAME, "w", encoding="utf-8") as out:
        for line in lines:
            out.write(line + "\n")


def main() -> None:
    """wmain"""
    if len(sys.argv) != 2:
        print("Usage: python ejercicio1.py fileWithData.txt")
        sys.exit(1)

    file_path = sys.argv[1]
    result = run(file_path)
    write_results(result)


if __name__ == "__main__":
    main()
