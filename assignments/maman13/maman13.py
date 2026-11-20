import csv
from pathlib import Path
import numpy as np


class DataLoader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data: list[list[int]] = []

    def load_data(self):
        with open(self.path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter = '\t')
            next(reader) # skip header row
            self.data = list(reader)

    def calc_y_decile_ranges(self):
        y_values = [row[-1] for row in self.data if row]
        # convert to ints
        y_values = [int(y) for y in y_values]
        # 10 deciles correspond to percentiles 0, 10, 20, ..., 100
        cutoffs = np.percentile(y_values, np.linspace(0, 100, 11))

        # Consecutive pairs (low, high) form the 10 decile ranges
        decile_ranges = list(zip(cutoffs[:-1], cutoffs[1:]))


def main() -> None:
    data_loader = DataLoader("diabetes.csv")
    data_loader.load_data()
    data_loader.calc_y_decile_ranges()



if __name__ == "__main__":
    main()