import pandas as pd
from pathlib import Path


class RCCMetadataDataset:
    """
    Класс для управления метаданными WSI сканов почечно-клеточного рака (RCC).
    """

    def __init__(self, csv_file: str, split: str = 'train'):
        """
        :param csv_file: путь к метаданным (data/metadata.csv)
        :param split: выборка ('train', 'val' или 'test')
        """
        self.csv_file = Path(csv_file)
        self.split = split

        if not self.csv_file.exists():
            raise FileNotFoundError(f"Файл метаданных не найден: {self.csv_file}")

        # Загружаем CSV манифест
        self.df = pd.read_csv(self.csv_file)

        # Фильтруем по split ('train' / 'val')
        if 'split' in self.df.columns:
            self.df = self.df[self.df['split'] == self.split].reset_index(drop=True)

        # Соответствие меток и названий подтипов
        self.label_map = {
            0: "Normal",
            1: "ccRCC",
            2: "pRCC",
            3: "chRCC",
            4: "Oncocytoma"
        }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        return {
            "slide_id": row["slide_id"],
            "patient_id": row["patient_id"],
            "label": int(row["subtype_label"]),
            "subtype_name": row["subtype_name"]
        }


if __name__ == "__main__":
    # Простой тест работы класса
    ROOT = Path(__file__).resolve().parent.parent
    dataset = RCCMetadataDataset(csv_file=ROOT / "data" / "metadata.csv", split="train")
    print(f"✅ В выборке '{dataset.split}' найдено {len(dataset)} слайдов.")
    if len(dataset) > 0:
        print(f"📄 Пример первого элемента: {dataset[0]}")