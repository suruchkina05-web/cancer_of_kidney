"""Рисование карты внимания по миниатюре среза. Без torch и openslide, чтобы можно было тестировать отдельно."""
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def attention_overlay(thumb, coords, attn, region, scale_x, scale_y, blur=7, alpha=0.55):
    """Накладывает на миниатюру карту внимания.

    thumb      - миниатюра RGB (H, W, 3)
    coords     - (N, 2) координаты патчей на самом детальном уровне скана
    attn       - (N,) веса внимания
    region     - сторона патча на детальном уровне (в пикселях скана)
    scale_x/y  - во сколько раз скан больше миниатюры
    Цвет показывает РАНГ патча (0 - наименее важный, 1 - самый важный),
    потому что абсолютные веса внимания очень малы и зависят от числа патчей.
    """
    h, w = thumb.shape[:2]
    order = np.argsort(np.argsort(attn))
    rank = order / max(len(attn) - 1, 1)
    heat = np.zeros((h, w), np.float32)
    seen = np.zeros((h, w), np.float32)
    for (x, y), r in zip(coords, rank):
        x0, x1 = int(x / scale_x), max(int((x + region) / scale_x), int(x / scale_x) + 1)
        y0, y1 = int(y / scale_y), max(int((y + region) / scale_y), int(y / scale_y) + 1)
        heat[y0:y1, x0:x1] = r
        seen[y0:y1, x0:x1] = 1
    if blur:
        k = blur * 2 + 1
        heat = cv2.GaussianBlur(heat, (k, k), 0)
        seen = cv2.GaussianBlur(seen, (k, k), 0)
    colored = (plt.get_cmap("magma")(heat)[:, :, :3] * 255).astype(np.uint8)
    a = (alpha * np.clip(seen * 1.5, 0, 1))[:, :, None]
    return (thumb * (1 - a) + colored * a).astype(np.uint8)


def save_figure(path, thumb, overlay, top_patches, title, probs_text):
    """Сохраняет рисунок: миниатюра | карта внимания, ниже самые важные патчи."""
    n = len(top_patches)
    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, max(n, 2), height_ratios=[3, 1.1], hspace=0.15, wspace=0.05)
    ax1 = fig.add_subplot(gs[0, : max(n, 2) // 2]); ax1.imshow(thumb); ax1.set_title("Миниатюра среза"); ax1.axis("off")
    ax2 = fig.add_subplot(gs[0, max(n, 2) // 2:]); ax2.imshow(overlay)
    ax2.set_title("Карта внимания (светлее = важнее)"); ax2.axis("off")
    for i, p in enumerate(top_patches):
        ax = fig.add_subplot(gs[1, i]); ax.imshow(p); ax.axis("off")
        ax.set_title(f"топ-{i + 1}", fontsize=8)
    fig.suptitle(f"{title}\n{probs_text}", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def scatter_map(coords, attn, region):
    """Быстрая карта внимания БЕЗ скана: патчи в своих координатах, цвет = ранг важности.
    Возвращает изображение RGB (numpy)."""
    coords = np.asarray(coords)
    rank = np.argsort(np.argsort(attn)) / max(len(attn) - 1, 1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(coords[:, 0], coords[:, 1], c=rank, cmap="magma", marker="s",
               s=max(2, 3000 / max(len(coords) ** 0.5, 1)), linewidths=0)
    ax.invert_yaxis(); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Карта внимания (светлее = важнее)")
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img
