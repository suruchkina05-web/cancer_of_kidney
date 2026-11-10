"""Сервис: загрузите свой гистологический срез (.svs/.tif/.ndpi) и получите подтип ПКР + карту внимания.

ИССЛЕДОВАТЕЛЬСКИЙ ПРОТОТИП, НЕ МЕДИЦИНСКОЕ ИЗДЕЛИЕ. Не заменяет заключение патолога.

Запуск в Colab (GPU T4), из корня проекта:
  !pip -q install gradio
  !python -m src.app_upload --model /content/drive/MyDrive/kidney_results/mil_final.pt
Gradio напечатает публичную ссылку (живёт, пока работает ячейка). Загруженные файлы не сохраняются.
"""
import argparse
import os
import time

import numpy as np
import torch

from src.features import load_encoder, extract_features
from src.heatmap import attention_overlay, save_figure
from src.metrics import NAMES
from src.model import AttentionMIL
from src.wsi import open_slide, find_tissue_coords, get_magnification, read_patch

DISCLAIMER = ("**Исследовательский прототип, не медицинское изделие.** Не заменяет заключение патолога. "
              "Модель учили на 30 срезах TCGA трёх подтипов (светлоклеточный, папиллярный, хромофобный). "
              "На других органах, окрасках и сканерах результат ненадёжен. Файл не сохраняется.")


def build(args):
    import gradio as gr
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttentionMIL()
    model.load_state_dict(torch.load(args.model, map_location="cpu"))
    model.eval()
    encoder = load_encoder(device)

    def predict(file, max_patches, progress=gr.Progress()):
        if file is None:
            raise gr.Error("Загрузите файл среза.")
        path = file if isinstance(file, str) else file.name
        t0 = time.time()
        progress(0.05, desc="Открываю срез и ищу ткань")
        slide = open_slide(path)
        coords, region = find_tissue_coords(slide, args.patch_size, min_tissue_ratio=0.5)
        if len(coords) == 0:
            raise gr.Error("Ткань на срезе не найдена. Проверьте файл.")
        if len(coords) > max_patches:  # равномерная подвыборка для скорости
            idx = np.sort(np.random.default_rng(0).choice(len(coords), int(max_patches), replace=False))
            coords = [coords[i] for i in idx]
        progress(0.2, desc=f"Считаю признаки по {len(coords)} патчам")
        feats = extract_features(slide, coords, region, encoder, device, args.patch_size)
        progress(0.8, desc="Запускаю модель")
        with torch.no_grad():
            logits, att = model(torch.from_numpy(np.asarray(feats)).float())
        probs = torch.softmax(logits, -1).numpy(); att = att.numpy()
        coords = np.array(coords)
        w, h = slide.dimensions
        thumb = np.array(slide.get_thumbnail((1400, 1400)).convert("RGB"))
        sx, sy = w / thumb.shape[1], h / thumb.shape[0]
        overlay = attention_overlay(thumb, coords, att, region, sx, sy)
        top = np.argsort(att)[::-1][:6]
        patches = [np.array(read_patch(slide, int(coords[i][0]), int(coords[i][1]), region, args.patch_size)) for i in top]
        pt = "Вероятности: " + ", ".join(f"{n} {p:.2f}" for n, p in zip(NAMES, probs))
        out = "/tmp/upload_attention.png"
        save_figure(out, thumb, overlay, patches, "Загруженный срез", pt)
        note = (f"Патчей: {len(coords)}, время: {time.time()-t0:.0f} с. "
                f"Предсказание модели: **{NAMES[int(probs.argmax())]}** (не диагноз).")
        return {n: float(p) for n, p in zip(NAMES, probs)}, out, note

    with gr.Blocks(title="ИИ для гистологии рака почки") as demo:
        gr.Markdown("# ИИ для гистологии рака почки: загрузите свой срез")
        gr.Markdown(DISCLAIMER)
        f = gr.File(label="Файл среза (.svs, .tif, .ndpi)", file_count="single")
        n = gr.Slider(200, 3000, value=1000, step=100, label="Сколько патчей анализировать (меньше = быстрее)")
        b = gr.Button("Анализировать", variant="primary")
        lab = gr.Label(label="Вероятности подтипов")
        img = gr.Image(label="Срез, карта внимания и самые важные патчи")
        txt = gr.Markdown()
        b.click(predict, [f, n], [lab, img, txt])
        gr.Markdown(DISCLAIMER)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--patch-size", type=int, default=256)
    a = ap.parse_args()
    build(a).launch(share=True)


if __name__ == "__main__":
    main()
