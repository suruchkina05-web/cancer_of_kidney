"""Назначает срезам фолды кросс-валидации ПО ПАЦИЕНТАМ (стратификация по подтипу).

Запуск (в Colab):
  python -m src.make_splits --meta /content/drive/MyDrive/kidney_features/metadata_real.csv --folds 5
Колонка split -> 'cv', колонка fold -> 0..K-1. Один пациент всегда в одном фолде.
"""
import argparse
import numpy as np
import pandas as pd


def assign_folds(df, n_folds=5, seed=42):
    rng = np.random.default_rng(seed)
    fold_of = {}
    for label, g in df.groupby("subtype_label"):
        pats = sorted(g["patient_id"].unique())
        rng.shuffle(pats)
        for i, p in enumerate(pats):
            fold_of[p] = i % n_folds
    out = df.copy()
    out["fold"] = out["patient_id"].map(fold_of)
    out["split"] = "cv"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = pd.read_csv(a.meta)
    out = assign_folds(df, a.folds, a.seed)
    out.to_csv(a.meta, index=False)
    print(pd.crosstab(out["subtype_name"], out["fold"]))
    print("\nБольницы по подтипам:")
    print(pd.crosstab(out["subtype_name"], out["tss_code"]))
    leak = out.groupby("patient_id")["fold"].nunique().max()
    assert leak == 1, "один пациент попал в разные фолды!"
    print("\nПроверка утечки по пациентам: ок. Сохранено в", a.meta)


if __name__ == "__main__":
    main()
