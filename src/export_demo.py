"""Выгружает из Colab маленькую папку для лёгкого прототипа (без PyTorch и без сканов).

Для каждого среза сохраняет: координаты патчей, веса внимания, вероятности финальной модели,
истинный подтип и предсказание кросс-валидации. Если в results-dir/attention есть готовые рисунки
карт внимания, копирует и их.

Запуск (в Colab):
  !python -m src.export_demo --feat-dir /content/drive/MyDrive/kidney_features \
      --results-dir /content/drive/MyDrive/kidney_results \
      --manifests data/manifests/kirc_manifest.txt data/manifests/kirp_manifest.txt data/manifests/kich_manifest.txt \
      --out /content/drive/MyDrive/kidney_demo
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.batch_extract import read_manifest, slide_name, project_of
from src.metrics import NAMES
from src.model import AttentionMIL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    feat_dir, res_dir, out = Path(a.feat_dir), Path(a.results_dir), Path(a.out)
    (out / "slides").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    model = AttentionMIL()
    model.load_state_dict(torch.load(res_dir / "mil_final.pt", map_location="cpu"))
    model.eval()
    oof_path = res_dir / "oof_predictions.csv"
    oof = pd.read_csv(oof_path).set_index("slide_id") if oof_path.exists() else None

    rows = []
    for m in a.manifests:
        truth = project_of(m)[1]
        for it in read_manifest(m):
            sid = slide_name(it["filename"])
            f = feat_dir / f"{sid}_features.npy"
            if not f.exists():
                continue
            feats = torch.from_numpy(np.load(f)).float()
            coords = np.load(feat_dir / f"{sid}_coords.npy").astype(np.int32)
            with torch.no_grad():
                logits, att = model(feats)
            probs = torch.softmax(logits, -1).numpy()
            np.savez_compressed(out / "slides" / f"{sid}.npz", coords=coords,
                                att=att.numpy().astype(np.float32), probs=probs.astype(np.float32))
            row = {"slide_id": sid, "truth": truth, "pred_final": NAMES[int(probs.argmax())]}
            if oof is not None and sid in oof.index:
                r = oof.loc[sid]
                row.update({"oof_pred": r["pred"], "oof_ccRCC": r["p_ccRCC"], "oof_pRCC": r["p_pRCC"],
                            "oof_chRCC": r["p_chRCC"]})
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "meta.csv", index=False)

    for png in (res_dir / "attention").glob("*_attention.png"):
        shutil.copy(png, out / "figures" / png.name)
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"Готово: {len(rows)} срезов, папка {out}, размер {size:.1f} МБ")


if __name__ == "__main__":
    main()
