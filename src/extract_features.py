"""Запуск: один скан -> файл с признаками.

Пример (из корня проекта):
    python -m src.extract_features --slide data/raw_wsi/СКАН.svs --out data/features
"""
import argparse
from pathlib import Path
import numpy as np

from src.wsi import open_slide, find_tissue_coords
from src.features import load_encoder, extract_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide", required=True, help="путь к файлу .svs")
    parser.add_argument("--out", required=True, help="папка для результата")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--min-tissue", type=float, default=0.5)
    args = parser.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Считаем на:", device)

    slide = open_slide(args.slide)
    coords, region = find_tissue_coords(slide, args.patch_size,
                                        min_tissue_ratio=args.min_tissue)
    print("Патчей с тканью:", len(coords))

    encoder = load_encoder(device)
    features = extract_features(slide, coords, region, encoder, device, args.patch_size)
    print("Размер таблицы признаков:", features.shape)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(args.slide).name.split(".")[0]
    np.save(out_dir / f"{name}_features.npy", features)
    np.save(out_dir / f"{name}_coords.npy", np.array(coords))
    print("Сохранено в", out_dir)


if __name__ == "__main__":
    main()