import csv
from pathlib import Path
import numpy as np


class DataLoader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data: list[list[int]] = []
        self.decile_ranges: list[tuple[int, int]] = []

    def load_data(self):
        with open(self.path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter = '\t')
            self.data = list(reader)

    def calc_y_decile_ranges(self):
        # read y_values (still as strings) from all rows, skipping the first header row
        y_values = [row[-1] for row in self.data[1:] if row]
        # convert y values to ints
        y_values = [int(y) for y in y_values]
        # 10 deciles correspond to percentiles 0, 10, 20, ..., 100
        cutoffs = np.percentile(y_values, np.linspace(0, 100, 11))
        # Consecutive pairs (low, high) form the 10 decile ranges
        self.decile_ranges = list(zip(cutoffs[:-1], cutoffs[1:]))

    def find_decile_index(self, decile):
        for i in range(len(self.decile_ranges)):
            if self.decile_ranges[i][0] <= int(decile) <= self.decile_ranges[i][1]:
                return i
        return 0

    def add_class_col(self):
        self.data[0].append("Class")
        for row in self.data[1:]:
            decile_index = self.find_decile_index(row[-1])
            row.append(decile_index + 1)    # 1-based decile labels [1..10]


def main() -> None:
    data_loader = DataLoader("diabetes.csv")
    data_loader.load_data()
    data_loader.calc_y_decile_ranges()
    data_loader.add_class_col()
    pass



if __name__ == "__main__":
    main()