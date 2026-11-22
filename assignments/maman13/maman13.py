import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

filename = "diabetes.csv"

def load_data(path: str | Path) -> list[list[str]]:
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter = '\t')
        # next(reader)
        return list(reader)

def calc_y_decile_ranges(data: list[list[str]]):
    # read y_values (still as strings) from all rows
    y_values = [row[-1] for row in data if row]
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
        row.append(decile_index)    # 0-based decile labels [0..9]

class DiabetesDataset(Dataset):
    def __init__(self, rows: list[list]) -> None:
        """
        rows: data rows excluding the header row (e.g., data_loader.data[1:]).
        """
        self.samples = [row for row in rows if row]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.samples[index]
        features = torch.tensor([float(val) for val in row[:-1]], dtype=torch.float32)
        return features, row[-1]

def iterate_batch(optimizer, model, loss_fn, features, deciles):
        optimizer.zero_grad()
        y_model = model(features)
        loss = loss_fn(y_model, deciles)
        loss.backward()
        optimizer.step()
        predicted_labels = torch.argmax(y_model, dim=1)
        accuracy = (predicted_labels == deciles).float().mean()

        return loss.item(), accuracy.item()

def main() -> None:
    train_data = load_data(filename)
    decile_ranges = calc_y_decile_ranges(train_data[1:])
    add_class_col(train_data, decile_ranges)

    diabetes_dataset = DiabetesDataset(train_data[1:])

    train_dataloader = DataLoader(diabetes_dataset,  batch_size=10, shuffle=True)

    iterator = iter(train_dataloader)
    (features, deciles) = next(iterator)
    print("index: ", deciles, " features ", features)

    # 8.
    model = nn.Sequential(
        nn.Linear(11, 11),
        nn.ReLU(),
        nn.Linear(11, 10),
        nn.LogSoftmax(dim=1),
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    epochs = 0

    loss = 0
    accuracy = 0
    while accuracy < 0.90:
        iterator = iter(train_dataloader)
        epochs += 1
        for i in range(len(iterator)):
            features, deciles = next(iterator)
            optimizer.zero_grad()
            loss, accuracy = iterate_batch(optimizer, model, nn.NLLLoss(), features, deciles)
        print(epochs, " loss: ", loss, " accuracy: ", accuracy)



if __name__ == "__main__":
    main()