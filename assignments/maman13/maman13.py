import csv
from pathlib import Path

import numpy as np

filename = "diabetes.csv"

def load_data(path: str | Path) -> list[list[str]]:
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        return list(reader)

def calc_y_decile_ranges(data: list[list[str]]):
    # read y_values (still as strings) from all rows, skipping the first header row
    y_values = [row[-1] for row in data[1:] if row]
    # convert y values to ints
    y_values = [int(y) for y in y_values]
    # 10 deciles correspond to percentiles 0, 10, 20, ..., 100
    cutoffs = np.percentile(y_values, np.linspace(0, 100, 11))
    # Consecutive pairs (low, high) form the 10 decile ranges
    return list(zip(cutoffs[:-1], cutoffs[1:]))

def find_decile_index(decile: int, decile_ranges: list[list[int]]) -> int:
    for i in range(len(decile_ranges)):
        if decile_ranges[i][0] <= int(decile) <= decile_ranges[i][1]:
            return i
    return 0

def add_class_col(data, decile_ranges):
    data[0].append("Class")
    for row in data[1:]:
        decile_index = find_decile_index(row[-1], decile_ranges)
        row.append(decile_index + 1)    # 1-based decile labels [1..10]


def main() -> None:
    data = load_data(filename)
    decile_ranges = calc_y_decile_ranges(data)
    add_class_col(data, decile_ranges)


    pass



if __name__ == "__main__":
    main()