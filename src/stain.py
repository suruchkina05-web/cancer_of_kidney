"""Нормализация окраски H&E по методу Макенко (Macenko et al., 2009).

Срезы из разных больниц окрашены по-разному. Метод раскладывает цвет патча на вклад двух красителей
(гематоксилин - ядра, эозин - цитоплазма) и пересобирает его с эталонными красителями, чтобы все патчи
выглядели как окрашенные в одной лаборатории.

Нужен только numpy (и Pillow, если подавать PIL-картинки). Пример:
    stain = MacenkoNormalizer()          # эталон по умолчанию
    patch_norm = stain(patch)            # patch: PIL.Image или массив (H, W, 3) uint8
"""
import numpy as np

# Эталонные вектора красителей и максимальные концентрации (значения из классической реализации метода).
HE_REF = np.array([[0.5626, 0.2159],
                   [0.7201, 0.8012],
                   [0.4062, 0.5581]])
MAXC_REF = np.array([1.9705, 1.0308])


class MacenkoNormalizer:
    def __init__(self, io=240, alpha=1.0, beta=0.15, min_tissue_pixels=500,
                 he_ref=None, maxc_ref=None):
        self.io, self.alpha, self.beta = io, alpha, beta
        self.min_tissue_pixels = min_tissue_pixels
        self.he_ref = HE_REF.copy() if he_ref is None else np.asarray(he_ref, float)
        self.maxc_ref = MAXC_REF.copy() if maxc_ref is None else np.asarray(maxc_ref, float)
        self.he_src = None      # красители среза (если заданы, оцениваются один раз на срез, а не на каждый патч)
        self.maxc_src = None

    # --- вспомогательные шаги ---
    def _od(self, arr):
        """Оптическая плотность пикселей: -log((I+1)/Io). Форма (N, 3)."""
        return -np.log((arr.reshape(-1, 3).astype(np.float64) + 1.0) / self.io)

    def _stain_matrix(self, od):
        """Находит 3x2 матрицу красителей (столбцы: гематоксилин, эозин) или None, если патч не годится."""
        od_hat = od[~np.any(od < self.beta, axis=1)]          # только пиксели с тканью
        if len(od_hat) < self.min_tissue_pixels:
            return None
        cov = np.cov(od_hat.T)
        if not np.all(np.isfinite(cov)):
            return None
        _, vecs = np.linalg.eigh(cov)
        plane = vecs[:, 1:3]                                  # плоскость двух главных направлений
        proj = od_hat @ plane
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        lo, hi = np.percentile(phi, self.alpha), np.percentile(phi, 100 - self.alpha)
        v_lo = plane @ np.array([np.cos(lo), np.sin(lo)])
        v_hi = plane @ np.array([np.cos(hi), np.sin(hi)])
        # знак: компоненты оптической плотности красителя неотрицательны
        v_lo, v_hi = (v if v.sum() >= 0 else -v for v in (v_lo, v_hi))
        he = np.array([v_lo, v_hi]).T if v_lo[0] > v_hi[0] else np.array([v_hi, v_lo]).T
        if np.any(he < -0.05):
            return None
        he = np.clip(he, 0, None)
        he = he / np.linalg.norm(he, axis=0, keepdims=True)
        return he

    def fit(self, reference):
        """Взять эталон из одного патча (PIL или массив). Если патч не годится - остаётся эталон по умолчанию."""
        arr = np.asarray(reference.convert("RGB") if hasattr(reference, "convert") else reference)
        od = self._od(arr)
        he = self._stain_matrix(od)
        if he is None:
            raise ValueError("Эталонный патч не годится: слишком мало ткани")
        c = np.linalg.pinv(he) @ od.T
        self.he_ref = he
        self.maxc_ref = np.percentile(c, 99, axis=1)
        return self

    def fit_source(self, patches, max_pixels=2_000_000, seed=0):
        """Оценить красители СРЕЗА по набору его патчей. После этого все патчи среза нормализуются одной
        и той же матрицей, и патчи с кровью или почти без ядер не искажаются. Возвращает True, если получилось."""
        ods = [self._od(np.asarray(p.convert("RGB") if hasattr(p, "convert") else p)) for p in patches]
        if not ods:
            return False
        od = np.concatenate(ods)
        if len(od) > max_pixels:
            od = od[np.random.default_rng(seed).choice(len(od), max_pixels, replace=False)]
        he = self._stain_matrix(od)
        if he is None:
            return False
        c = np.linalg.pinv(he) @ od.T
        maxc = np.percentile(c, 99, axis=1)
        if np.any(maxc <= 1e-6) or not np.all(np.isfinite(maxc)):
            return False
        self.he_src, self.maxc_src = he, maxc
        return True

    def fit_source_from_slide(self, slide, coords, region, patch_size=256, n=64, seed=0):
        """Взять n случайных патчей среза и оценить по ним красители среза."""
        from src.wsi import read_patch
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coords), size=min(n, len(coords)), replace=False)
        patches = [read_patch(slide, int(coords[i][0]), int(coords[i][1]), region, patch_size) for i in idx]
        return self.fit_source(patches, seed=seed)

    def normalize_array(self, arr):
        """arr: (H, W, 3) uint8 -> нормализованный (H, W, 3) uint8. Негодные патчи возвращаются без изменений."""
        arr = np.asarray(arr)
        h, w, _ = arr.shape
        try:
            od = self._od(arr)
            if self.he_src is not None:                        # красители заданы для всего среза
                he, maxc = self.he_src, self.maxc_src
                c = np.linalg.pinv(he) @ od.T
            else:                                              # запасной вариант: оценка по самому патчу
                he = self._stain_matrix(od)
                if he is None:
                    return arr
                c = np.linalg.pinv(he) @ od.T                  # концентрации красителей, (2, N)
                maxc = np.percentile(c, 99, axis=1)
                if np.any(maxc <= 1e-6) or not np.all(np.isfinite(maxc)):
                    return arr
            c = c / (maxc / self.maxc_ref)[:, None]
            out = self.io * np.exp(-(self.he_ref @ c))
            out = np.clip(out.T.reshape(h, w, 3), 0, 255).astype(np.uint8)
            return out
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return arr

    def __call__(self, img):
        """Принимает PIL.Image или массив, возвращает объект того же типа."""
        if hasattr(img, "convert"):
            from PIL import Image
            return Image.fromarray(self.normalize_array(np.asarray(img.convert("RGB"))))
        return self.normalize_array(img)
