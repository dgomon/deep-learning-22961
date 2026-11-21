import csv
from pathlib import Path

import numpy as np
import torch

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

filename = "diabetes.csv"

def load_data(path: str | Path) -> list[list[str]]:
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        next(reader)
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
    # data[0].append("Class")
    for row in data:
        decile_index = find_decile_index(row[-1], decile_ranges)
        row.append(decile_index + 1)    # 1-based decile labels [1..10]

class DiabetesDataset(Dataset):
    def __init__(self, rows: list[list], zero_indexed_labels: bool = True) -> None:
        """
        rows: data rows excluding the header row (e.g., data_loader.data[1:]).
        zero_indexed_labels: if True, converts 1-based class labels [1..10] to [0..9]
                             for standard PyTorch cross-entropy loss functions.
        """
        self.samples = [row for row in rows if row]
        self.zero_indexed_labels = zero_indexed_labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.samples[idx]
        features = torch.tensor([float(val) for val in row[:-1]], dtype=torch.float32)
        label_val = int(row[-1]) - 1 if self.zero_indexed_labels else int(row[-1])
        label = torch.tensor(label_val, dtype=torch.long)
        return features, label

def main() -> None:
    train_data = load_data(filename)
    decile_ranges = calc_y_decile_ranges(train_data)
    add_class_col(train_data, decile_ranges)

    diabetes_dataset = DiabetesDataset(train_data, zero_indexed_labels=False)

    train_dataloader = DataLoader(diabetes_dataset,  batch_size=10)

    iterator = iter(train_dataloader)
    x = next(iterator)
    print(x)

    pass



if __name__ == "__main__":
    main()