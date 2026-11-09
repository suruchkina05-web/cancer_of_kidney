"""Обучение attention-MIL с кросс-валидацией по пациентам + метрики с доверительными интервалами.

Запуск (в Colab):
  python -m src.train_mil --feat-dir /content/drive/MyDrive/kidney_features \
      --out /content/drive/MyDrive/kidney_results
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from src.model import AttentionMIL
from src.metrics import NAMES, metrics, bootstrap


def load_bag(feat_dir, slide_id):
    return torch.from_numpy(np.load(Path(feat_dir) / f"{slide_id}_features.npy")).float()


def train_one(bags, labels, epochs, device, max_patches, lr, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AttentionMIL().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for i in rng.permutation(len(bags)):
            x = bags[i]
            if len(x) > max_patches:
                x = x[torch.from_numpy(rng.choice(len(x), max_patches, replace=False))]
            logits, _ = model(x.to(device))
            loss = loss_fn(logits.unsqueeze(0), torch.tensor([labels[i]], device=device))
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def predict(model, bag, device):
    model.eval()
    logits, att = model(bag.to(device))
    return torch.softmax(logits, -1).cpu().numpy(), att.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--meta", default=None, help="по умолчанию <feat-dir>/metadata_real.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-patches", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.meta or Path(a.feat_dir) / "metadata_real.csv")
    df = df[df["fold"].notna()].reset_index(drop=True)
    df = df[[(Path(a.feat_dir) / f"{s}_features.npy").exists() for s in df["slide_id"]]].reset_index(drop=True)
    y = (df["subtype_label"].astype(int) - 1).to_numpy()
    print(f"Срезов с признаками: {len(df)}  | на устройстве: {device}")
    bags = [load_bag(a.feat_dir, s) for s in df["slide_id"]]

    prob = np.zeros((len(df), 3))
    for k in sorted(df["fold"].unique()):
        te = np.where(df["fold"] == k)[0]
        tr = np.where(df["fold"] != k)[0]
        model = train_one([bags[i] for i in tr], y[tr], a.epochs, device, a.max_patches, a.lr, a.seed)
        for i in te:
            prob[i], _ = predict(model, bags[i], device)
        print(f"фолд {int(k)}: обучено на {len(tr)}, проверено на {len(te)}")

    m = metrics(y, prob)
    ci = bootstrap(y, prob)
    rows = [{"metric": k, "value": v,
             "ci_low": ci[k].quantile(0.025) if k in ci else np.nan,
             "ci_high": ci[k].quantile(0.975) if k in ci else np.nan} for k, v in m.items()]
    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics_cv.csv", index=False)
    pd.DataFrame(prob, columns=[f"p_{n}" for n in NAMES]).assign(
        slide_id=df["slide_id"], true=[NAMES[i] for i in y], pred=[NAMES[i] for i in prob.argmax(1)]
    ).to_csv(out / "oof_predictions.csv", index=False)
    print("\n=== Метрики (кросс-валидация по пациентам, предсказания на отложенных фолдах) ===")
    print(res.round(3).to_string(index=False))
    print("\nМатрица ошибок (строки - истина, столбцы - предсказание):")
    print(pd.DataFrame(confusion_matrix(y, prob.argmax(1), labels=[0, 1, 2]), index=NAMES, columns=NAMES))

    final = train_one(bags, y, a.epochs, device, a.max_patches, a.lr, a.seed)
    torch.save(final.state_dict(), out / "mil_final.pt")
    print("\nФинальная модель (на всех срезах) сохранена:", out / "mil_final.pt")


if __name__ == "__main__":
    main()
