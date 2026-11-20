import csv
from pathlib import Path


class DataLoader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data: list[dict[str, str]] = []

    def load_data(self) -> list[dict[str, str]]:
        with self.path.open(mode="r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            self.data = list(reader)
        return self.data


def main() -> None:
    data_loader = DataLoader("diabetes.csv")
    rows = data_loader.load_data()
    _ = rows
    pass


if __name__ == "__main__":
    main()