"""Работа с настоящими гистологическими сканами (WSI): поиск ткани и чтение патчей.

Скан огромный, поэтому целиком в память он не загружается:
1) ткань ищется на маленькой копии (миниатюре);
2) настоящие патчи читаются из файла по одному, только там, где есть ткань.
"""
import numpy as np
import cv2


def open_slide(path):
    """Открывает файл .svs и возвращает объект OpenSlide."""
    import openslide  # импорт здесь, чтобы остальной код работал и без openslide
    return openslide.OpenSlide(str(path))


def get_magnification(slide, default=40.0):
    """Увеличение скана (обычно 20 или 40)."""
    return float(slide.properties.get("openslide.objective-power", default))


def tissue_mask(slide, thumb_size=2000):
    """Маска ткани по миниатюре. Возвращает (маска bool, масштаб по x, масштаб по y)."""
    width, height = slide.dimensions
    thumb = np.array(slide.get_thumbnail((thumb_size, thumb_size)).convert("RGB"))
    th, tw = thumb.shape[:2]
    saturation = cv2.cvtColor(thumb, cv2.COLOR_RGB2HSV)[:, :, 1]
    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask > 0, width / tw, height / th


def find_tissue_coords(slide, patch_size=256, target_mag=20.0,
                       min_tissue_ratio=0.5, thumb_size=2000):
    """Находит координаты патчей, где достаточно ткани.

    Возвращает (coords, region):
      coords - список (x, y) в пикселях самого детального уровня;
      region - сторона квадрата в этих пикселях, которую надо прочитать,
               чтобы после сжатия до patch_size получилось увеличение target_mag.
    """
    width, height = slide.dimensions
    region = int(patch_size * get_magnification(slide) / target_mag)
    mask, fx, fy = tissue_mask(slide, thumb_size)

    coords = []
    for y in range(0, height - region, region):
        for x in range(0, width - region, region):
            window = mask[int(y / fy):int((y + region) / fy),
                          int(x / fx):int((x + region) / fx)]
            if window.size and window.mean() >= min_tissue_ratio:
                coords.append((x, y))
    return coords, region


def read_patch(slide, x, y, region, patch_size=256):
    """Читает один патч из файла и приводит его к размеру patch_size x patch_size."""
    img = slide.read_region((x, y), 0, (region, region)).convert("RGB")
    return img.resize((patch_size, patch_size))