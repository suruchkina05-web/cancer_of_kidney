"""Исследовательский прототип: подтип ПКР по гистологическому срезу + карта внимания.

ИССЛЕДОВАТЕЛЬСКИЙ ПРОТОТИП, НЕ МЕДИЦИНСКОЕ ИЗДЕЛИЕ. Не заменяет заключение патолога.

Запуск (в Colab, из корня проекта):
  !pip -q install gradio
  !python -m src.app --feat-dir /content/drive/MyDrive/kidney_features \
      --results-dir /content/drive/MyDrive/kidney_results \
      --manifests data/manifests/kirc_manifest.txt data/manifests/kirp_manifest.txt data/manifests/kich_manifest.txt
Gradio напечатает временную публичную ссылку, её можно открыть в браузере.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.batch_extract import read_manifest, slide_name, download, project_of
from src.heatmap import attention_overlay, save_figure, scatter_map
from src.metrics import NAMES
from src.model import AttentionMIL
from src.wsi import open_slide, get_magnification, read_patch

DISCLAIMER = ("**Исследовательский прототип, не медицинское изделие.** Не заменяет заключение патолога. "
              "Модель учили всего на 30 срезах TCGA, качество по кросс-валидации скромное (macro-AUC около 0,74).")


def build(args):
    import gradio as gr
    feat_dir, res_dir = Path(args.feat_dir), Path(args.results_dir)
    model = AttentionMIL()
    model.load_state_dict(torch.load(res_dir / "mil_final.pt", map_location="cpu"))
    model.eval()

    items, truth = {}, {}
    for m in args.manifests:
        for it in read_manifest(m):
            sid = slide_name(it["filename"])
            if (feat_dir / f"{sid}_features.npy").exists():
                items[sid] = it
                truth[sid] = project_of(m)[1]
    oof_path = res_dir / "oof_predictions.csv"
    oof = pd.read_csv(oof_path).set_index("slide_id") if oof_path.exists() else None
    ids = sorted(items)

    def predict(sid, with_patches):
        feats = torch.from_numpy(np.load(feat_dir / f"{sid}_features.npy")).float()
        coords = np.load(feat_dir / f"{sid}_coords.npy")
        with torch.no_grad():
            logits, att = model(feats)
        probs = torch.softmax(logits, -1).numpy(); att = att.numpy()
        region = int(args.patch_size * 2)  # 40x -> 20x, как при извлечении признаков
        label = {n: float(p) for n, p in zip(NAMES, probs)}
        text = f"Истинный подтип (по клиническим данным TCGA): **{truth[sid]}**.\n\n"
        text += "Вероятности выше посчитаны моделью, которая обучалась на этом срезе, поэтому они завышены."
        if oof is not None and sid in oof.index:
            r = oof.loc[sid]
            text += (f"\n\n**Предсказание при кросс-валидации** (срез в обучении не участвовал): "
                     f"{r['pred']} (ccRCC {r['p_ccRCC']:.2f}, pRCC {r['p_pRCC']:.2f}, chRCC {r['p_chRCC']:.2f}).")
        quick = scatter_map(coords, att, region)
        if not with_patches:
            return label, quick, None, text
        svs = Path(args.work) / items[sid]["filename"]
        Path(args.work).mkdir(parents=True, exist_ok=True)
        download(items[sid]["id"], svs)
        try:
            slide = open_slide(svs)
            region = int(args.patch_size * get_magnification(slide) / 20.0)
            w, h = slide.dimensions
            thumb = np.array(slide.get_thumbnail((1400, 1400)).convert("RGB"))
            overlay = attention_overlay(thumb, coords, att, region, w / thumb.shape[1], h / thumb.shape[0])
            top = np.argsort(att)[::-1][:6]
            patches = [np.array(read_patch(slide, int(coords[i][0]), int(coords[i][1]), region, args.patch_size))
                       for i in top]
        finally:
            if svs.exists():
                os.remove(svs)
        out = Path(args.work) / f"{sid}_ui.png"
        save_figure(out, thumb, overlay, patches, sid, "карта внимания и самые важные патчи")
        return label, quick, str(out), text

    with gr.Blocks(title="ИИ для гистологии рака почки") as demo:
        gr.Markdown("# ИИ для гистологии рака почки (прототип)\n" + DISCLAIMER)
        with gr.Row():
            sid = gr.Dropdown(ids, value=ids[0], label="Срез (из 30 обработанных)")
            full = gr.Checkbox(False, label="Показать срез и важные патчи (скачивает скан, 1–2 минуты)")
            btn = gr.Button("Показать", variant="primary")
        with gr.Row():
            lab = gr.Label(num_top_classes=3, label="Вероятности подтипов")
            img = gr.Image(label="Карта внимания (быстрая)")
        full_img = gr.Image(label="Срез, карта внимания и самые важные патчи")
        note = gr.Markdown()
        btn.click(predict, [sid, full], [lab, img, full_img, note])
        gr.Markdown(DISCLAIMER)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--work", default="/tmp/wsi")
    ap.add_argument("--patch-size", type=int, default=256)
    args = ap.parse_args()
    build(args).launch(share=True)


if __name__ == "__main__":
    main()
