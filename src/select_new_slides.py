"""Отбор новых пациентов для расширения выборки (20 срезов на подтип).

Читает клинические метки и текущие манифесты, исключает уже взятых пациентов и
выбирает кандидатов так, чтобы в каждом подтипе было как можно больше РАЗНЫХ больниц
(tss_code): жадно берётся пациент из больницы, которая в подтипе представлена меньше всего.
Часть кандидатов запасная: у некоторых пациентов в GDC нет диагностического среза.

Запуск из корня проекта (нужна только стандартная библиотека):
    python -m src.select_new_slides --new 10 --reserve 6 --seed 42
Результат: data/new_slide_candidates.csv и три файла data/new_cases_<проект>.txt
(списки пациентов для загрузки в GDC).
"""
import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECTS = ["kirc", "kirp", "kich"]
NAMES = {"kirc": "ccRCC", "kirp": "pRCC", "kich": "chRCC"}


def read_tsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick(pool, used_tss, n, rng):
    """Жадный выбор: каждый раз берём пациента из наименее представленной больницы."""
    by_tss = defaultdict(list)
    for r in pool:
        by_tss[r["tss_code"]].append(r["patient_id"])
    for v in by_tss.values():
        rng.shuffle(v)
    chosen = []
    counts = Counter(used_tss)
    while len(chosen) < n and any(by_tss.values()):
        cands = [t for t, v in by_tss.items() if v]
        # меньше всего уже взято; при равенстве - где больше пациентов; затем по коду (воспроизводимость)
        t = min(cands, key=lambda x: (counts[x], -len(by_tss[x]), x))
        chosen.append((by_tss[t].pop(), t))
        counts[t] += 1
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--new", type=int, default=10, help="сколько новых срезов на подтип")
    ap.add_argument("--reserve", type=int, default=6, help="запасных кандидатов на подтип")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    data = Path(a.data)
    rng = random.Random(a.seed)
    out_rows = []
    for p in PROJECTS:
        taken = {r["filename"][:12] for r in read_tsv(data / "manifests" / f"{p}_manifest.txt")}
        taken_tss = [t[5:7] for t in taken]
        pool = [r for r in read_csv(data / f"{p}_clinical_labels.csv")
                if r["patient_id"] not in taken and r.get("use_for_training", "да") not in ("нет", "0")]
        chosen = pick(pool, taken_tss, a.new + a.reserve, rng)
        print(f"{p}: уже взято {len(taken)} пациентов из больниц {sorted(set(taken_tss))}")
        print(f"   новые: {Counter(t for _, t in chosen[:a.new])}")
        with open(data / f"new_cases_{p}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(pid for pid, _ in chosen) + "\n")
        for i, (pid, t) in enumerate(chosen, 1):
            out_rows.append({"subtype": NAMES[p], "project": p, "rank": i, "role": "основной" if i <= a.new else "запасной",
                             "patient_id": pid, "tss_code": t})
    with open(data / "new_slide_candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    print("Готово: data/new_slide_candidates.csv и data/new_cases_<проект>.txt")


if __name__ == "__main__":
    main()
