"""Карта внимания для одного среза: какие участки модель считала главными.

Запуск (в Colab), пример для светлоклеточного среза:
  python -m src.attention_map --slide-id TCGA-B8-4154-01Z-00-DX1 \
      --manifests data/manifests/kirc_manifest.txt data/manifests/kirp_manifest.txt data/manifests/kich_manifest.txt \
      --feat-dir /content/drive/MyDrive/kidney_features \
      --model /content/drive/MyDrive/kidney_results/mil_final.pt \
      --out /content/drive/MyDrive/kidney_results/attention
Скан скачивается заново (он не хранится), рисунок сохраняется в --out.
ВАЖНО: финальная модель обучена на всех 30 срезах, поэтому на этих же срезах карта
показывает механизм работы, но не проверку качества.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from src.batch_extract import read_manifest, slide_name, download, project_of
from src.heatmap import attention_overlay, save_figure
from src.model import AttentionMIL
from src.metrics import NAMES
from src.wsi import open_slide, get_magnification, read_patch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide-id", required=True)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/tmp/wsi")
    ap.add_argument("--patch-size", type=int, default=256)
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()

    item, true_name = None, None
    for m in a.manifests:
        for it in read_manifest(m):
            if slide_name(it["filename"]) == a.slide_id:
                item, true_name = it, project_of(m)[1]
    if item is None:
        raise SystemExit(f"Срез {a.slide_id} не найден в манифестах")

    feat_dir, out, work = Path(a.feat_dir), Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True)
    feats = torch.from_numpy(np.load(feat_dir / f"{a.slide_id}_features.npy")).float()
    coords = np.load(feat_dir / f"{a.slide_id}_coords.npy")

    model = AttentionMIL()
    model.load_state_dict(torch.load(a.model, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits, att = model(feats)
    probs = torch.softmax(logits, -1).numpy(); att = att.numpy()

    svs = work / item["filename"]
    print("Скачиваю скан для рисунка...")
    download(item["id"], svs)
    try:
        slide = open_slide(svs)
        region = int(a.patch_size * get_magnification(slide) / 20.0)
        w, h = slide.dimensions
        thumb = np.array(slide.get_thumbnail((1400, 1400)).convert("RGB"))
        sx, sy = w / thumb.shape[1], h / thumb.shape[0]
        overlay = attention_overlay(thumb, coords, att, region, sx, sy)
        top_idx = np.argsort(att)[::-1][: a.top]
        top_patches = [np.array(read_patch(slide, int(coords[i][0]), int(coords[i][1]), region, a.patch_size))
                       for i in top_idx]
    finally:
        if svs.exists():
            os.remove(svs)

    probs_text = "Вероятности: " + ", ".join(f"{n} {p:.2f}" for n, p in zip(NAMES, probs)) + \
                 f"  |  предсказание: {NAMES[int(probs.argmax())]}"
    path = out / f"{a.slide_id}_attention.png"
    save_figure(path, thumb, overlay, top_patches, f"{a.slide_id} (метка: {true_name})", probs_text)
    print("Сохранено:", path)


if __name__ == "__main__":
    main()
