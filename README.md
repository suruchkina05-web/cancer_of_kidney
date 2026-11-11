# ИИ для гистологического исследования при раке почки

Исследовательский прототип: классификация подтипа почечно-клеточного рака (светлоклеточный ccRCC, папиллярный pRCC, хромофобный chRCC) по гистологическому срезу (WSI, TCGA) с картой внимания модели.

**Не медицинское изделие. Не заменяет заключение патолога.**

- Демо-страница (30 срезов, карта внимания): https://cancerofkidney.netlify.app
- Код: https://github.com/suruchkina05-web/cancer_of_kidney

## Пайплайн
срез .svs → маска ткани (HSV + Otsu) → патчи 256 px при 20x → признаки ResNet50 (ImageNet, 2048) → Attention-MIL (gated attention) → вероятности подтипов + карта внимания.

## Структура
- `src/wsi.py`, `preprocessing.py`, `features.py` – чтение среза, маска ткани, патчи, признаки
- `src/batch_extract.py` – пакетная обработка по манифестам GDC
- `src/model.py`, `train_mil.py`, `train_baseline.py`, `make_splits.py`, `metrics.py` – модель, обучение, 5-fold CV по пациентам, метрики
- `src/attention_map.py`, `heatmap.py` – карты внимания
- `src/app_upload.py` – сервис: врач загружает свой срез (Gradio, нужен GPU/Colab)
- `src/export_demo.py`, `build_static.py` – сборка статической демо-страницы
- `data/` – манифесты, клинические метки, таблица метаданных (сканы и признаки в Git не хранятся)

## Запуск (Colab)
```
git clone -b develop https://github.com/suruchkina05-web/cancer_of_kidney.git
cd cancer_of_kidney
apt-get -qq install -y openslide-tools
pip -q install -r requirements.txt gradio
python -m src.app_upload --model <путь>/mil_final.pt
```

## Результаты (5-fold CV по пациентам, 30 срезов)
Attention-MIL: macro-AUC 0,74 (95% ДИ 0,59–0,89), F1 0,50. Базовая модель (усреднённые признаки + логистическая регрессия): macro-AUC 0,79 (0,66–0,92), F1 0,63. Выборка мала, доверительные интервалы широкие.
