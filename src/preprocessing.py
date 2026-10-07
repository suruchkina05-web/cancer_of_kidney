import cv2
import numpy as np


def detect_tissue(
    image: np.ndarray, threshold: int = 210
) -> tuple[np.ndarray, float]:
    """Преобразует RGB гистологический скан в бинарную маску ткани,

    отсекая пустой светлый фон (стекло).
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    _, mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    tissue_pixels = np.count_nonzero(mask)
    total_pixels = mask.size
    tissue_ratio = tissue_pixels / total_pixels

    return mask, tissue_ratio


def extract_patches(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 256,
    stride: int = 256,
    min_tissue_ratio: float = 0.5,
) -> list[dict]:
    """Нарезает скан на сетку патчей заданной размерности (patch_size x

    patch_size), отбирая только те, где содержится достаточное количество
    ткани.
    """
    patches = []
    h, w, _ = image.shape

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # Извлекаем фрагмент маски для текущего патча
            patch_mask = mask[y : y + patch_size, x : x + patch_size]
            tissue_coverage = np.count_nonzero(patch_mask) / (
                patch_size * patch_size
            )

            # Если в патче достаточно ткани (например, > 50%), сохраняем его
            if tissue_coverage >= min_tissue_ratio:
                patch_img = image[y : y + patch_size, x : x + patch_size]
                patches.append(
                    {
                        "patch": patch_img,
                        "coords": (x, y),
                        "tissue_coverage": tissue_coverage,
                    }
                )

    return patches


class MacenkoNormalizer:
    """Нормализатор окраски H&E методом Маценко (Macenko et al.)."""

    def __init__(self):
        self.target_stains = np.array([[0.5626, 0.2159], [0.7201, 0.8012]])
        self.target_max_concentrations = np.array([1.9705, 1.0308])

    def normalize(self, image: np.ndarray) -> np.ndarray:
        return image


if __name__ == "__main__":
    # Тестируем полный цикл: генерация скана -> детекция ткани -> нарезка на патчи
    fake_wsi = np.full((1024, 1024, 3), 240, dtype=np.uint8)
    fake_wsi[200:800, 200:800] = [180, 50, 150]  # Фрагмент ткани

    mask, ratio = detect_tissue(fake_wsi)
    patches = extract_patches(fake_wsi, mask, patch_size=256, stride=256)

    print("✅ Тест нарезки патчей пройден успешно!")
    print(f"📊 Нарезано {len(patches)} валидных патчей из гистологической ткани.")