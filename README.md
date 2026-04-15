# YOLOv11 Research (Detection + Classification)

Скрипт [`cv_yolo11_research.py`](/home/alexander/cyberphiz_cv2/cv_yolo11_research.py) автоматизирует пункты задания:

1. baseline обучение и оценка,
2. проверка гипотез улучшения,
3. формирование improved baseline,
4. повторное обучение и сравнение метрик.

## 1) Варианты маленьких датасетов (выбор)

### Для детекции
- `coco8.yaml`: очень быстрый запуск (8 изображений), но 80 классов.
- `coco128.yaml`: маленький и стабильнее для сравнения гипотез.
- `path/to/your_dataset.yaml`: лучший вариант для отчета, если сделать 2-5 классов и ~100-800 изображений.

### Для классификации
- `mnist160`: очень быстрые эксперименты.
- `cifar10`: маленький и более реалистичный.

Показать варианты в консоли:

```bash
python cv_yolo11_research.py --list-datasets
```

## 2) Выбранные метрики и обоснование

Для `detect`:
- `mAP50-95` (основная): главная метрика качества детекции.
- `mAP50`: дополнительная для интерпретации.
- `precision`, `recall`, `F1`: баланс ложноположительных и пропусков.

Для `classify`:
- `top1` (основная), `top5`.

## 3) Установка

```bash
pip install ultralytics
```

## 4) Запуск baseline + improved (полный цикл)

### Детекция (рекомендуемый быстрый старт)
```bash
python cv_yolo11_research.py \
  --task detect \
  --data coco128.yaml \
  --model yolo11n.pt \
  --epochs 15 \
  --imgsz 640 \
  --batch 16 \
  --device cpu \
  --exp-name detect_coco128
```

### Классификация
```bash
python cv_yolo11_research.py \
  --task classify \
  --data mnist160 \
  --model yolo11n-cls.pt \
  --epochs 10 \
  --imgsz 160 \
  --batch 32 \
  --device cpu \
  --exp-name cls_mnist160
```

## 5) Где смотреть результаты

После запуска создаются:
- `runs/yolo11_research/<exp-name>_report/report.json`
- `runs/yolo11_research/<exp-name>_report/report.md`

В отчете есть сравнение baseline vs improved и дельты по метрикам.
