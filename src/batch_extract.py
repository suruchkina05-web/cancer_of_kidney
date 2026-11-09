"""Пакетная обработка срезов по манифестам GDC.

Для каждого среза: скачать -> посчитать признаки -> сохранить -> удалить скан.
Уже посчитанные срезы пропускаются, поэтому после обрыва Colab можно
просто запустить скрипт ещё раз.

Пример (из корня проекта, в Colab):
    python -m src.batch_extract \
        --manifests data/manifests/kirc_manifest.txt data/manifests/kirp_manifest.txt data/manifests/kich_manifest.txt \
        --out /content/drive/MyDrive/kidney_features
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np

API = "https://api.gdc.cancer.gov/data/"
# часть имени манифеста -> (метка, название подтипа)
PROJECTS = {"kirc": (1, "ccRCC"), "kirp": (2, "pRCC"), "kich": (3, "chRCC")}


def read_manifest(path):
    """Читает манифест GDC (таблица с табуляцией: id, filename, md5, size, state)."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def project_of(manifest_path):
    """Определяет подтип по имени файла манифеста (kirc / kirp / kich)."""
    name = Path(manifest_path).name.lower()
    for key, value in PROJECTS.items():
        if key in name:
            return value
    raise ValueError(f"В имени манифеста {name} нет kirc, kirp или kich")


def slide_name(filename):
    """TCGA-A3-3336-01Z-00-DX1.F284....svs -> TCGA-A3-3336-01Z-00-DX1"""
    return filename.split(".")[0]


def metadata_rows(manifest_paths, features_dir):
    """Строки для metadata_real.csv по всем срезам из манифестов."""
    rows = []
    for mpath in manifest_paths:
        label, subtype = project_of(mpath)
        for item in read_manifest(mpath):
            name = slide_name(item["filename"])
            feat = f"{name}_features.npy"
            rows.append({
                "slide_id": name,
                "patient_id": name[:12],
                "tss_code": name[5:7],
                "subtype_label": label,
                "subtype_name": subtype,
                "isup_grade": "",
                "split": "pending",
                "features_file": feat if (Path(features_dir) / feat).exists() else "",
            })
    return rows


def write_metadata(rows, path):
    fields = ["slide_id", "patient_id", "tss_code", "subtype_label",
              "subtype_name", "isup_grade", "split", "features_file"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def download(file_id, dest, retries=3):
    import requests
    last = None
    for attempt in range(retries):
        try:
            with requests.get(API + file_id, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            return
        except Exception as e:  # повторяем при обрыве связи
            last = e
            print(f"  попытка {attempt + 1} не удалась: {e}")
    raise RuntimeError(f"Не удалось скачать {file_id}: {last}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--out", required=True, help="папка для признаков (Google Drive)")
    parser.add_argument("--work", default="/tmp/wsi", help="временная папка для сканов")
    parser.add_argument("--limit", type=int, default=None,
                        help="обработать только первые N срезов каждого манифеста (для проверки)")
    parser.add_argument("--patch-size", type=int, default=256)
    args = parser.parse_args()

    import torch
    from src.wsi import open_slide, find_tissue_coords
    from src.features import load_encoder, extract_features

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Считаем на:", device)
    out_dir, work_dir = Path(args.out), Path(args.work)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    encoder = load_encoder(device)

    done, skipped, failed = 0, 0, []
    for mpath in args.manifests:
        items = read_manifest(mpath)[: args.limit]
        print(f"\n== {Path(mpath).name}: {len(items)} срезов")
        for i, item in enumerate(items, 1):
            name = slide_name(item["filename"])
            feat_path = out_dir / f"{name}_features.npy"
            if feat_path.exists():
                print(f"[{i}/{len(items)}] {name}: уже посчитан, пропускаю")
                skipped += 1
                continue
            svs = work_dir / item["filename"]
            try:
                print(f"[{i}/{len(items)}] {name}: скачиваю...")
                download(item["id"], svs)
                slide = open_slide(svs)
                coords, region = find_tissue_coords(slide, args.patch_size)
                print(f"  патчей с тканью: {len(coords)}")
                feats = extract_features(slide, coords, region, encoder, device, args.patch_size)
                np.save(feat_path, feats)
                np.save(out_dir / f"{name}_coords.npy", np.array(coords))
                print(f"  готово: {feats.shape}")
                done += 1
            except Exception as e:
                print(f"  ОШИБКА: {e}")
                failed.append(name)
            finally:
                if svs.exists():
                    os.remove(svs)

    rows = metadata_rows(args.manifests, out_dir)
    write_metadata(rows, out_dir / "metadata_real.csv")
    print(f"\nИтог: посчитано {done}, пропущено {skipped}, ошибок {len(failed)}")
    if failed:
        print("Не получились:", failed)
    print("Таблица метаданных:", out_dir / "metadata_real.csv")


if __name__ == "__main__":
    main()
