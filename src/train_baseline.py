"""Базовая модель БЕЗ внимания: среднее признаков всех патчей среза + логистическая регрессия.
Те же фолды и те же признаки, что у attention-MIL, поэтому сравнение честное.

Запуск (в Colab):
  python -m src.train_baseline --feat-dir /content/drive/MyDrive/kidney_features \
      --out /content/drive/MyDrive/kidney_results
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.metrics import NAMES, metrics, bootstrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, default=0.1, help="сила регуляризации (меньше = сильнее)")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.meta or Path(a.feat_dir) / "metadata_real.csv")
    df = df[df["fold"].notna()].reset_index(drop=True)
    df = df[[(Path(a.feat_dir) / f"{s}_features.npy").exists() for s in df["slide_id"]]].reset_index(drop=True)
    y = (df["subtype_label"].astype(int) - 1).to_numpy()
    X = np.stack([np.load(Path(a.feat_dir) / f"{s}_features.npy").mean(axis=0) for s in df["slide_id"]])
    print(f"Срезов: {len(df)}, признаков на срез: {X.shape[1]}")

    prob = np.zeros((len(df), 3))
    for k in sorted(df["fold"].unique()):
        te = np.where(df["fold"] == k)[0]; tr = np.where(df["fold"] != k)[0]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=a.C, max_iter=2000))
        clf.fit(X[tr], y[tr])
        prob[te] = clf.predict_proba(X[te])

    m = metrics(y, prob); ci = bootstrap(y, prob)
    res = pd.DataFrame([{"metric": k, "value": v,
                         "ci_low": ci[k].quantile(0.025) if k in ci else np.nan,
                         "ci_high": ci[k].quantile(0.975) if k in ci else np.nan} for k, v in m.items()])
    res.to_csv(out / "metrics_baseline_cv.csv", index=False)
    pd.DataFrame(prob, columns=[f"p_{n}" for n in NAMES]).assign(
        slide_id=df["slide_id"], true=[NAMES[i] for i in y], pred=[NAMES[i] for i in prob.argmax(1)]
    ).to_csv(out / "oof_predictions_baseline.csv", index=False)
    print("\n=== БАЗОВАЯ модель (среднее признаков + логистическая регрессия), кросс-валидация по пациентам ===")
    print(res.round(3).to_string(index=False))
    print("\nМатрица ошибок:")
    print(pd.DataFrame(confusion_matrix(y, prob.argmax(1), labels=[0, 1, 2]), index=NAMES, columns=NAMES))

    mil = out / "metrics_cv.csv"
    if mil.exists():
        a_ = pd.read_csv(mil).set_index("metric"); b_ = res.set_index("metric")
        cmp = pd.DataFrame({"MIL_с_вниманием": a_["value"], "базовая": b_["value"]}).loc[["macro_AUC", "macro_F1", "accuracy"]]
        print("\n=== Сравнение моделей ===")
        print(cmp.round(3).to_string())


if __name__ == "__main__":
    main()
