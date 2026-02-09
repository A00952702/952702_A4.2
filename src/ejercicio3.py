# pylint: disable=duplicate-code
# src/ejercicio3.py
"""Reads words from a file and counts distinct words and their frequency"""

from __future__ import annotations

import sys
import time
from typing import Dict, List


RESULTS_FILENAME = "WordCountResults.txt"


def clean_word(token: str) -> str:
    """clean_word"""
    # Basic cleanup: keep letters/numbers and apostrophe, drop punctuation
    cleaned = ""
    for ch in token.strip():
        if ch.isalnum() or ch == "'":
            cleaned += ch
    return cleaned.lower()


def merge_sort_keys(keys: List[str]) -> List[str]:
    """merge_sort_keys"""
    if len(keys) <= 1:
        return keys[:]
    mid = len(keys) // 2
    left = merge_sort_keys(keys[:mid])
    right = merge_sort_keys(keys[mid:])
    return merge_keys(left, right)


def merge_keys(left: List[str], right: List[str]) -> List[str]:
    """merge_keys"""
    out: List[str] = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            out.append(left[i])
            i += 1
        else:
            out.append(right[j])
            j += 1
    while i < len(left):
        out.append(left[i])
        i += 1
    while j < len(right):
        out.append(right[j])
        j += 1
    return out


def run(file_path: str) -> float:
    """run"""
    start = time.perf_counter()

    counts: Dict[str, int] = {}

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        for line_num, line in enumerate(file, start=1):
            if not line.strip():
                print(f"[Line {line_num}] Empty line -> ignored")
                continue

            tokens = line.split()
            for tok in tokens:
                w = clean_word(tok)
                if not w:
                    continue
                counts[w] = counts.get(w, 0) + 1

    elapsed = time.perf_counter() - start

    keys = merge_sort_keys(list(counts.keys()))
    rows = [f"{k}\t{counts[k]}" for k in keys]

    for row in rows:
        print(row)

    with open(RESULTS_FILENAME, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(row + "\n")
        out.write(f"ELAPSED_SECONDS: {elapsed:.6f}\n")

    print(f"ELAPSED_SECONDS: {elapsed:.6f}")
    return elapsed


def main() -> None:
    """main"""
    if len(sys.argv) != 2:
        print("Usage: python ejercicio3.py fileWithData.txt")
        sys.exit(1)

    run(sys.argv[1])


if __name__ == "__main__":
    main()
