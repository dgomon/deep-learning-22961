import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset

filename = "diabetes.csv"

def load_data(path: str | Path) -> list[list[str]]:
    with open(path, newline='') as csvfile:
        return list(csv.reader(csvfile, delimiter = '\t'))


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

        valid_rows = [row for row in rows if row]

        # row[:-1] contains the 11 feature values; row[-1] is the 0..9 class label
        raw_features = torch.tensor(
            [[float(val) for val in row[:-1]] for row in valid_rows],
            dtype=torch.float32,
        )
        self.targets = torch.tensor(
            [int(row[-1]) for row in valid_rows],
            dtype=torch.long,
        )

        # normalize data
        mean = raw_features.mean(dim=0, keepdim=True)
        std = raw_features.std(dim=0, keepdim=True, unbiased=False)
        std = torch.where(std == 0, torch.ones_like(std), std)

        self.features = (raw_features - mean) / std


    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.features[index], self.targets[index]


def train_model(model: nn.Module, optimizer: torch.optim.Optimizer, dataloader: DataLoader,
                target_accuracy: float, ignore_y: bool) -> tuple[int, float | int]:
    epochs = 0
    epoch_accuracy = 0
    loss_fn = nn.NLLLoss()

    while epoch_accuracy < target_accuracy:
        epochs += 1
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for features, deciles in dataloader:
            optimizer.zero_grad()
            y_model = model(features) if not ignore_y else model(features[:, :-1])
            loss = loss_fn(y_model, deciles)
            loss.backward()
            optimizer.step()

            batch_size = deciles.size(0)
            running_loss += loss.item() * batch_size
            predicted_labels = torch.argmax(y_model, dim=1)
            correct_predictions += (predicted_labels == deciles).sum().item()
            total_samples += batch_size

        epoch_accuracy = correct_predictions / total_samples

    return epochs, epoch_accuracy


def subparagraph_8() -> None:
    train_data = load_data(filename)
    decile_ranges = calc_y_decile_ranges(train_data[1:])
    add_class_col(train_data, decile_ranges)

    diabetes_dataset = DiabetesDataset(train_data[1:])
    train_dataloader = DataLoader(diabetes_dataset,  batch_size=10, shuffle=False)

    model = nn.Sequential(
        nn.Linear(11, 11),
        nn.ReLU(),
        nn.Linear(11, 10),
        nn.LogSoftmax(dim=1),
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs, epochs_accuracy = train_model(model, optimizer, train_dataloader,  0.1, False)
    print(f"epochs: {epochs}   accuracy: {epochs_accuracy:.4f}")


def subparagraph_9() -> None:
    train_data = load_data(filename)
    decile_ranges = calc_y_decile_ranges(train_data[1:])
    add_class_col(train_data, decile_ranges)

    diabetes_dataset = DiabetesDataset(train_data[1:])
    train_dataloader = DataLoader(diabetes_dataset,  batch_size=10, shuffle=False)

    # 9.
    model = nn.Sequential(
        nn.Linear(10, 10),
        nn.ReLU(),
        nn.Linear(10, 10),
        nn.LogSoftmax(dim=1),
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs, epochs_accuracy = train_model(model, optimizer, train_dataloader,  0.1, True)
    print(f"epochs: {epochs}   accuracy: {epochs_accuracy:.4f}")


def subparagraph_10() -> None:
    train_data = load_data(filename)
    decile_ranges = calc_y_decile_ranges(train_data[1:])
    add_class_col(train_data, decile_ranges)


    generator = torch.Generator().manual_seed(42)

    diabetes_dataset = DiabetesDataset(train_data[1:])

    train_ds, test_ds = random_split(
        diabetes_dataset, [0.8, 0.2], generator=generator
    )

    train_dataloader = DataLoader(train_ds,  batch_size=10, shuffle=False)
    test_dataloader = DataLoader(test_ds,  batch_size=1, shuffle=False)

    # 10.
    model = nn.Sequential(
        nn.Linear(11, 11),
        nn.ReLU(),
        nn.Linear(11, 10),
        nn.LogSoftmax(dim=1),
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    loss_fn = nn.NLLLoss()
    epochs = 0
    epoch_accuracy = 0.0

    while epoch_accuracy < 0.5:
        epochs += 1
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for features, deciles in train_dataloader:
            optimizer.zero_grad()
            y_model = model(features)
            loss = loss_fn(y_model, deciles)
            loss.backward()
            optimizer.step()

            batch_size = deciles.size(0)
            running_loss += loss.item() * batch_size
            predicted_labels = torch.argmax(y_model, dim=1)
            correct_predictions += (predicted_labels == deciles).sum().item()
            total_samples += batch_size

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        print(f"{epochs}  loss: {epoch_loss:.4f}  accuracy: {epoch_accuracy:.4f}")

    print("test set")
    correct_predictions = 0
    total_samples = 0
    running_loss = 0
    for features, deciles in test_dataloader:
        batch_size = deciles.size(0)
        y_model = model(features)
        predicted_labels = torch.argmax(y_model, dim=1)
        correct_predictions += (predicted_labels == deciles).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    print(f"{epochs}  loss: {epoch_loss:.4f}  accuracy: {epoch_accuracy:.4f}")


if __name__ == "__main__":
    subparagraph_9()