# src/ejercicio2.py
"""Reads integers from a file and converts each to binary and hexadecimal."""

from __future__ import annotations

import sys
import time
from typing import Optional, Tuple


RESULTS_FILENAME = "ConvertionResults.txt"
HEX_DIGITS = "0123456789ABCDEF"


def parse_int_line(line: str) -> Tuple[Optional[int], Optional[str]]:
    """parse_int_line"""
    raw = line.strip()
    if not raw:
        return None, "Empty line"

    # Reject floats explicitly
    if "." in raw:
        return None, f"Not an integer: {raw}"

    # Accept integer with optional sign
    try:
        return int(raw), None
    except ValueError:
        return None, f"Invalid integer: {raw}"


def to_base(number: int, base: int) -> str:
    """to_base"""
    if number == 0:
        return "0"

    is_negative = number < 0
    n = -number if is_negative else number

    digits = []
    while n > 0:
        n, rem = divmod(n, base)
        if base == 16:
            digits.append(HEX_DIGITS[rem])
        else:
            digits.append(str(rem))

    # reverse digits (manual)
    out = ""
    for i in range(len(digits) - 1, -1, -1):
        out += digits[i]

    if is_negative:
        out = "-" + out
    return out


def run(file_path: str) -> float:
    """run"""
    start = time.perf_counter()

    outputs = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line_num, line in enumerate(file, start=1):
            value, err = parse_int_line(line)
            if value is None:
                print(f"[Line {line_num}] Invalid data -> '{line.strip()}': {err}")
                continue

            b_str = to_base(value, 2)
            h_str = to_base(value, 16)
            outputs.append(f"{value}\tBIN={b_str}\tHEX={h_str}")

    elapsed = time.perf_counter() - start

    for row in outputs:
        print(row)

    with open(RESULTS_FILENAME, "w", encoding="utf-8") as out:
        for row in outputs:
            out.write(row + "\n")
        out.write(f"ELAPSED_SECONDS: {elapsed:.6f}\n")

    print(f"ELAPSED_SECONDS: {elapsed:.6f}")
    return elapsed


def main() -> None:
    """main"""
    if len(sys.argv) != 2:
        print("Usage: python ejercicio2.py fileWithData.txt")
        sys.exit(1)

    run(sys.argv[1])


if __name__ == "__main__":
    main()
