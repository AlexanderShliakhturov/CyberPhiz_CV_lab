# Experiment Description (YOLOv11, COCO128)

## Данные и цель
- Датасет: `coco128.yaml`
- Задача: детекция объектов (bbox + class)
- Основная метрика: `mAP50-95`
- Доп. метрики: `mAP50`, `precision`, `recall`, `F1`

## Какие модели обучались

### 1) Baseline: `coco128_v2_baseline`
- Модель: `yolo11n.pt`
- Метапараметры:
- `epochs=10`, `imgsz=640`, `batch=8`, `device=cpu`
- `optimizer=SGD`, `lr0=0.01`
- Аугментации отключены: `mosaic=0`, `mixup=0`, `fliplr=0`, `hsv*=0`
- Коротко по архитектуре:
- YOLO11n (nano): компактная архитектура с меньшим числом параметров, быстрый и стабильный baseline.
- Результат:
- `mAP50-95=0.6047`, `mAP50=0.7875`, `precision=0.8016`, `recall=0.6774`, `F1=0.7343`

### 2) Candidate: `coco128_v2_improved_yolo11n_adamw_aug`
- Модель: `yolo11n.pt`
- Метапараметры:
- `epochs=16`, `imgsz=640`, `batch=8`, `device=cpu`
- `optimizer=AdamW`, `lr0=0.004`, `cos_lr=True`
- Аугментации: `mosaic=0.3`, `mixup=0.05`, `fliplr=0.5`, `hsv_h=0.015`, `hsv_s=0.4`, `hsv_v=0.3`
- Коротко по архитектуре:
- Архитектура та же (YOLO11n), изменены только оптимизация и стратегия обучения.
- Результат:
- `mAP50-95=0.1887`, `mAP50=0.3226`, `precision=0.5611`, `recall=0.2945`, `F1=0.3862`

### 3) Candidate: `coco128_v2_improved_yolo11s_adamw_aug`
- Модель: `yolo11s.pt`
- Метапараметры:
- `epochs=16`, `imgsz=640`, `batch=8`, `device=cpu`
- `optimizer=AdamW`, `lr0=0.0035`, `cos_lr=True`
- Аугментации: `mosaic=0.3`, `mixup=0.05`, `fliplr=0.5`, `hsv_h=0.015`, `hsv_s=0.4`, `hsv_v=0.3`
- Коротко по архитектуре:
- YOLO11s (small): больше параметров/ширина и глубина сети выше, чем у YOLO11n; потенциально выше качество при достаточном времени обучения.
- Результат:
- `mAP50-95=0.1482`, `mAP50=0.2455`, `precision=0.6216`, `recall=0.2176`, `F1=0.3224`

### 4) Candidate: `coco128_v2_improved_baseline_finetune` (выбран как improved)
- Модель: дообучение от чекпоинта baseline  
  `/home/alexander/cyberphiz_cv2/runs/baseline_vs_improved/coco128_v2_baseline/weights/best.pt`
- Метапараметры:
- `epochs=8`, `imgsz=640`, `batch=8`, `device=cpu`
- `optimizer=SGD`, `lr0=0.002`
- Умеренные аугментации: `mosaic=0.15`, `mixup=0`, `fliplr=0.5`, `hsv_h=0.01`, `hsv_s=0.2`, `hsv_v=0.15`
- Коротко по архитектуре:
- Архитектура не менялась (YOLO11n), улучшение достигнуто через fine-tuning от сильной стартовой точки + более мягкий learning rate.
- Результат:
- `mAP50-95=0.6412`, `mAP50=0.8177`, `precision=0.7985`, `recall=0.7395`, `F1=0.7679`

## Основные изменения в архитектуре и обучении
- Архитектурно сравнивались `YOLO11n` и `YOLO11s`:
- `YOLO11s` более емкая модель, но на CPU и при заданном бюджете эпох не показала прироста.
- Наилучший результат получен не сменой архитектуры, а режимом обучения:
- Fine-tuning baseline-чекпоинта + уменьшение `lr0` + умеренные аугментации.

## Выводы
- Лучший итог: `baseline_finetune`.
- Прирост относительно baseline:
- `mAP50-95: +0.0366`
- `mAP50: +0.0302`
- `recall: +0.0620`
- `F1: +0.0335`
- Агрессивные аугментации и AdamW в этом эксперименте оказались хуже baseline.
- Для данного вычислительного режима (CPU, ограниченные эпохи) стратегия «сначала стабильный baseline, затем аккуратный fine-tuning» оказалась наиболее эффективной.
