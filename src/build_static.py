"""Собирает СТАТИЧЕСКИЙ сайт-прототип из папки demo_data (без сервера, PyTorch и Gradio).

Результат (папка static_site) можно открыть двойным щелчком по index.html, выложить на
бесплатный Static Space на Hugging Face или опубликовать как веб-страницу.

Запуск:  python -m src.build_static --demo-dir demo_data --out static_site
"""
import argparse
import base64
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src.heatmap import scatter_map

NAMES = ["ccRCC", "pRCC", "chRCC"]
TITLE_NAMES = {"ccRCC": "светлоклеточный", "pRCC": "папиллярный", "chRCC": "хромофобный"}

HTML = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ИИ для гистологии рака почки</title>
<style>
:root{--bg:#f6e3ec;--card:#fcf4f8;--ink:#2e2150;--acc:#c8386f;--pur:#5b3f9e;--line:#e3b8cc}
@media(prefers-color-scheme:dark){:root{--bg:#1d1530;--card:#2a2044;--ink:#f3e6f0;--acc:#f08ab0;--pur:#b9a3ee;--line:#45386a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:24px 16px 48px}
h1{font-family:Georgia,serif;font-size:28px;margin:0 0 8px}
.warn{background:var(--ink);color:var(--bg);border-radius:14px;padding:12px 16px;margin:12px 0 20px;font-size:15px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:16px;margin-bottom:16px}
select{width:100%;padding:10px;font-size:16px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:700px){.grid{grid-template-columns:1fr}}
.bar{display:flex;align-items:center;gap:8px;margin:8px 0}.bar span.n{width:70px;font-weight:600}
.track{flex:1;height:18px;background:var(--line);border-radius:9px;overflow:hidden}.fill{height:100%;background:var(--pur)}
.fill.top{background:var(--acc)}.bar span.v{width:48px;text-align:right}
img{max-width:100%;border-radius:10px;display:block}h2{font-size:17px;margin:0 0 10px}
.note{font-size:15px}.small{font-size:13px;opacity:.85}
</style></head><body><div class="wrap">
<h1>ИИ для гистологии рака почки (прототип)</h1>
<div class="warn"><b>Исследовательский прототип, не медицинское изделие.</b> Не заменяет заключение патолога. Модель учили на 30 срезах TCGA, качество по кросс-валидации скромное (macro-AUC около 0,74).</div>
<div class="card"><label for="s"><b>Срез</b> (из 30 обработанных)</label><br><select id="s"></select></div>
<div class="grid">
<div class="card"><h2>Вероятности подтипов</h2><div id="bars"></div><p class="small">Считает модель, обучавшаяся на всех 30 срезах, в том числе на этом, поэтому вероятности завышены.</p></div>
<div class="card"><h2>Карта внимания (светлее = важнее)</h2><img id="sc" alt="Карта внимания"></div>
</div>
<div class="card note" id="txt"></div>
<div class="card" id="figcard" style="display:none"><h2>Срез, карта внимания и самые важные патчи</h2><img id="fig" alt="Срез с картой внимания"></div>
<div class="warn">Исследовательский прототип, не медицинское изделие. Результаты: 5-кратная кросс-валидация по пациентам на 30 срезах TCGA (светлоклеточный, папиллярный, хромофобный).</div>
</div>
<script>
fetch('data.json').then(r=>r.json()).then(D=>{
const sel=document.getElementById('s');
D.slides.forEach(s=>{const o=document.createElement('option');o.value=s.id;o.textContent=s.id;sel.appendChild(o)});
function show(){
 const s=D.slides.find(x=>x.id===sel.value);
 const bars=document.getElementById('bars');bars.innerHTML='';
 const top=Math.max(...s.probs);
 D.names.forEach((n,i)=>{const p=s.probs[i];const d=document.createElement('div');d.className='bar';
  d.innerHTML='<span class="n">'+n+'</span><div class="track"><div class="fill'+(p===top?' top':'')+'" style="width:'+(p*100).toFixed(0)+'%"></div></div><span class="v">'+p.toFixed(2)+'</span>';bars.appendChild(d)});
 document.getElementById('sc').src=s.scatter;
 let t='<button id="rv" style="padding:10px 16px;font-size:16px;border-radius:10px;border:0;background:var(--acc);color:#fff;cursor:pointer">Показать правильный ответ</button><div id="ans" style="display:none;margin-top:12px">Истинный подтип (по клиническим данным TCGA): <b>'+s.truth+'</b> ('+s.truth_ru+').';
 if(s.oof){t+='<br><br><b>Предсказание при кросс-валидации</b> (срез в обучении не участвовал): '+s.oof.pred+' (ccRCC '+s.oof.p[0].toFixed(2)+', pRCC '+s.oof.p[1].toFixed(2)+', chRCC '+s.oof.p[2].toFixed(2)+').'}
 t+='</div>';
 document.getElementById('txt').innerHTML=t;
 document.getElementById('rv').onclick=function(){document.getElementById('ans').style.display='block';this.style.display='none'};
 const fc=document.getElementById('figcard');
 if(s.figure){document.getElementById('fig').src=s.figure;fc.style.display='block'}else{fc.style.display='none'}
}
sel.addEventListener('change',show);show();
});
</script></body></html>
"""

README = """---
title: ИИ для гистологии рака почки
emoji: 🔬
colorFrom: pink
colorTo: purple
sdk: static
pinned: false
---

Исследовательский прототип: подтип почечно-клеточного рака по срезу TCGA и карта внимания модели.
Не медицинское изделие. Не заменяет заключение патолога.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-dir", default="demo_data")
    ap.add_argument("--out", default="static_site")
    a = ap.parse_args()
    demo, out = Path(a.demo_dir), Path(a.out)
    (out / "scatter").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    meta = pd.read_csv(demo / "meta.csv")
    slides = []
    for _, r in meta.iterrows():
        sid = r["slide_id"]
        d = np.load(demo / "slides" / f"{sid}.npz")
        buf = io.BytesIO()
        Image.fromarray(scatter_map(d["coords"], d["att"], 512)).convert("RGB").resize((600, 600)).save(buf, "JPEG", quality=85)
        scatter = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
        fig_src = demo / "figures" / f"{sid}_attention.png"
        figure = None
        if fig_src.exists():
            im = Image.open(fig_src).convert("RGB")
            im.thumbnail((1200, 1200)); b2 = io.BytesIO(); im.save(b2, "JPEG", quality=78)
            figure = "data:image/jpeg;base64," + base64.b64encode(b2.getvalue()).decode()
        item = {"id": sid, "truth": r["truth"], "truth_ru": TITLE_NAMES.get(r["truth"], ""),
                "probs": [float(x) for x in d["probs"]], "scatter": scatter, "figure": figure, "oof": None}
        if "oof_pred" in meta.columns and isinstance(r.get("oof_pred"), str):
            item["oof"] = {"pred": r["oof_pred"], "p": [float(r["oof_ccRCC"]), float(r["oof_pRCC"]), float(r["oof_chRCC"])]}
        slides.append(item)
    (out / "data.json").write_text(json.dumps({"names": NAMES, "slides": slides}, ensure_ascii=False), encoding="utf-8")
    (out / "index.html").write_text(HTML, encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"Готово: {len(slides)} срезов, папка {out}, {size:.1f} МБ. Откройте {out}/index.html")


if __name__ == "__main__":
    main()
