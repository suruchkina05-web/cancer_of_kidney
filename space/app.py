"""Постоянный сервис загрузки среза (Hugging Face Space, CPU).

В Space лежат только 4 небольших файла: app.py, requirements.txt, packages.txt, README.md.
Код и обученная модель при запуске скачиваются из GitHub (ветка develop, models/mil_final.pt).
"""
import argparse
import os
import subprocess
import sys

REPO = "https://github.com/suruchkina05-web/cancer_of_kidney.git"
BRANCH = "develop"

if not os.path.isdir("repo"):
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO, "repo"], check=True)
sys.path.insert(0, os.path.abspath("repo"))

from src.app_upload import build  # noqa: E402

args = argparse.Namespace(model="repo/models/mil_final.pt", patch_size=256, stain_norm=True)
demo = build(args)

if __name__ == "__main__":
    demo.launch()
