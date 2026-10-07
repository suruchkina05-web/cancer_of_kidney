"""Превращение патчей в числа с помощью готовой нейросети ResNet50 («глаза» модели).

torch импортируется внутри функций: так остальной код проекта
запускается и на компьютере, где torch не установлен.
"""
import numpy as np
from src.wsi import read_patch


def load_encoder(device="cpu"):
    """ResNet50, предобученная на ImageNet, без последнего слоя (выход: 2048 чисел)."""
    import torch.nn as nn
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Identity()
    return model.to(device).eval()


def extract_features(slide, coords, region, encoder, device="cpu",
                     patch_size=256, batch_size=32):
    """Возвращает массив размера (число патчей, 2048)."""
    import torch
    from torchvision import transforms
    from tqdm import tqdm

    prep = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    result = []
    with torch.no_grad():
        for i in tqdm(range(0, len(coords), batch_size)):
            batch = torch.stack([
                prep(read_patch(slide, x, y, region, patch_size))
                for x, y in coords[i:i + batch_size]
            ]).to(device)
            result.append(encoder(batch).cpu().numpy())
    return np.concatenate(result)