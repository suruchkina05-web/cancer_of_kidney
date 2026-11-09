"""Лёгкий прототип: работает без PyTorch, без сканов и без Colab. Читает папку demo_data.

ИССЛЕДОВАТЕЛЬСКИЙ ПРОТОТИП, НЕ МЕДИЦИНСКОЕ ИЗДЕЛИЕ. Не заменяет заключение патолога.

Локально:  pip install gradio numpy pandas matplotlib
           python -m src.app_demo --demo-dir demo_data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.heatmap import scatter_map

NAMES = ["ccRCC", "pRCC", "chRCC"]
DISCLAIMER = ("**Исследовательский прототип, не медицинское изделие.** Не заменяет заключение патолога. "
              "Модель учили всего на 30 срезах TCGA, качество по кросс-валидации скромное (macro-AUC около 0,74).")


def load_slide(demo_dir, sid):
    """Возвращает (вероятности финальной модели, изображение карты, путь к рисунку или None, текст)."""
    demo_dir = Path(demo_dir)
    meta = pd.read_csv(demo_dir / "meta.csv").set_index("slide_id")
    d = np.load(demo_dir / "slides" / f"{sid}.npz")
    probs = {n: float(p) for n, p in zip(NAMES, d["probs"])}
    img = scatter_map(d["coords"], d["att"], 512)
    r = meta.loc[sid]
    text = (f"Истинный подтип (по клиническим данным TCGA): **{r['truth']}**.\n\n"
            "Вероятности выше посчитаны моделью, которая обучалась на этом срезе, поэтому они завышены.")
    if "oof_pred" in meta.columns and isinstance(r.get("oof_pred"), str):
        text += (f"\n\n**Предсказание при кросс-валидации** (срез в обучении не участвовал): "
                 f"{r['oof_pred']} (ccRCC {r['oof_ccRCC']:.2f}, pRCC {r['oof_pRCC']:.2f}, "
                 f"chRCC {r['oof_chRCC']:.2f}).")
    fig = demo_dir / "figures" / f"{sid}_attention.png"
    return probs, img, (str(fig) if fig.exists() else None), text


def build(demo_dir):
    import gradio as gr
    ids = sorted(pd.read_csv(Path(demo_dir) / "meta.csv")["slide_id"])
    with gr.Blocks(title="ИИ для гистологии рака почки") as demo:
        gr.Markdown("# ИИ для гистологии рака почки (прототип)\n" + DISCLAIMER)
        sid = gr.Dropdown(ids, value=ids[0], label="Срез (из 30 обработанных)")
        btn = gr.Button("Показать", variant="primary")
        with gr.Row():
            lab = gr.Label(num_top_classes=3, label="Вероятности подтипов")
            img = gr.Image(label="Карта внимания (светлее = важнее)")
        fig = gr.Image(label="Срез, карта внимания и самые важные патчи (есть не для всех срезов)")
        note = gr.Markdown()
        btn.click(lambda s: load_slide(demo_dir, s), [sid], [lab, img, fig, note])
        gr.Markdown(DISCLAIMER)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-dir", default="demo_data")
    ap.add_argument("--share", action="store_true", help="временная публичная ссылка")
    a = ap.parse_args()
    build(a.demo_dir).launch(share=a.share)


if __name__ == "__main__":
    main()
