"""Картинка «до / после» нормализации окраски по Макенко (для отчёта и слайда).

Берёт несколько срезов (лучше из разных больниц и подтипов), для каждого - несколько случайных патчей с тканью,
и рисует их до и после нормализации.

Запуск в Colab (скачает срезы по манифестам и удалит после использования):
    python -m src.stain_check --slide-ids TCGA-A3-3383-01Z-00-DX1 TCGA-5P-A9K2-01Z-00-DX1 TCGA-KN-8426-01Z-00-DX1 \
        --manifests data/manifests/kirc_new_manifest.txt data/manifests/kirp_new_manifest.txt data/manifests/kich_new_manifest.txt \
        --out /content/drive/MyDrive/kidney_results/stain_check.png
"""
import argparse
import os
import random
from pathlib import Path

import numpy as np


def make_figure(rows, out_path, n):
    """rows: список (название, [патчи до], [патчи после]); патчи - массивы (H, W, 3) uint8."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(rows), 2 * n, figsize=(2.2 * 2 * n, 2.3 * len(rows)), squeeze=False)
    for r, (title, before, after) in enumerate(rows):
        for j in range(n):
            axes[r][j].imshow(before[j]); axes[r][n + j].imshow(after[j])
        axes[r][0].set_ylabel(title, fontsize=8, rotation=0, ha="right", va="center", labelpad=6)
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
    axes[0][n // 2].set_title("до нормализации", fontsize=11, loc="center")
    axes[0][n + n // 2].set_title("после (Макенко)", fontsize=11, loc="center")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide-ids", nargs="+", required=True)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/tmp/wsi")
    ap.add_argument("--n", type=int, default=3, help="патчей на срез")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    from src.batch_extract import read_manifest, slide_name, download
    from src.stain import MacenkoNormalizer
    from src.wsi import open_slide, find_tissue_coords, read_patch

    items = {}
    for m in a.manifests:
        for it in read_manifest(m):
            items[slide_name(it["filename"])] = it
    stain = MacenkoNormalizer()
    rng = random.Random(a.seed)
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in a.slide_ids:
        if sid not in items:
            raise SystemExit(f"Срез {sid} не найден в манифестах")
        svs = work / items[sid]["filename"]
        try:
            print("Скачиваю", sid)
            download(items[sid]["id"], svs)
            slide = open_slide(svs)
            coords, region = find_tissue_coords(slide, a.patch_size, min_tissue_ratio=0.9)
            pick = rng.sample(coords, min(a.n, len(coords)))
            before = [np.array(read_patch(slide, x, y, region, a.patch_size)) for x, y in pick]
            after = [stain(p) for p in before]
            rows.append((sid[:12], before, after))
        finally:
            if svs.exists():
                os.remove(svs)
    make_figure(rows, a.out, a.n)
    print("Сохранено:", a.out)


if __name__ == "__main__":
    main()
