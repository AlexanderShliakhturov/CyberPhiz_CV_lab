"""
Исследование детекции/классификации на Ultralytics YOLOv11.

Что делает скрипт:
1) Обучает baseline.
2) Проверяет набор гипотез улучшения.
3) Выбирает лучшую гипотезу по целевой метрике.
4) Обучает improved baseline.
5) Формирует JSON и Markdown отчеты с сравнением метрик.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent


DETECTION_DATASET_OPTIONS = [
    {
        "name": "COCO8 (ultralytics)",
        "data": "coco8.yaml",
        "classes": 80,
        "size": "очень маленький (8 изображений)",
        "why": "самый быстрый старт для проверки пайплайна, но классов много",
    },
    {
        "name": "COCO128 (ultralytics)",
        "data": "coco128.yaml",
        "classes": 80,
        "size": "маленький (128 изображений)",
        "why": "быстро учится даже на слабом железе, метрики стабильнее чем у COCO8",
    },
    {
        "name": "Кастомный mini-dataset (рекомендуется для курсовой)",
        "data": "path/to/your_dataset.yaml",
        "classes": "2-5",
        "size": "100-800 изображений",
        "why": "лучший баланс: мало классов и реалистичные выводы",
    },
]

CLASSIFICATION_DATASET_OPTIONS = [
    {
        "name": "MNIST160 (ultralytics)",
        "data": "mnist160",
        "classes": 10,
        "size": "очень маленький",
        "why": "очень быстрые эксперименты для задачи классификации",
    },
    {
        "name": "CIFAR10 (ultralytics)",
        "data": "cifar10",
        "classes": 10,
        "size": "маленький",
        "why": "более реалистичная классификация, чем MNIST",
    },
]


@dataclass
class ExperimentResult:
    stage: str
    run_name: str
    task: str
    model: str
    metrics: Dict[str, float]
    params: Dict[str, Any]
    score: float


def list_dataset_options() -> None:
    print("\nВарианты маленьких датасетов для DETECTION:")
    for idx, item in enumerate(DETECTION_DATASET_OPTIONS, start=1):
        print(
            f"{idx}. {item['name']} | data={item['data']} | classes={item['classes']} | "
            f"size={item['size']}\n   Причина: {item['why']}"
        )
    print("\nВарианты маленьких датасетов для CLASSIFICATION:")
    for idx, item in enumerate(CLASSIFICATION_DATASET_OPTIONS, start=1):
        print(
            f"{idx}. {item['name']} | data={item['data']} | classes={item['classes']} | "
            f"size={item['size']}\n   Причина: {item['why']}"
        )
    print()


def _to_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
        if math.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _pick_metric(raw: Dict[str, Any], contains_any: List[str]) -> Optional[float]:
    keys = list(raw.keys())
    for key in keys:
        lkey = key.lower()
        if any(token in lkey for token in contains_any):
            val = _to_float(raw[key])
            if val is not None:
                return val
    return None


def parse_metrics(task: str, raw_metrics: Dict[str, Any]) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    if task == "detect":
        precision = _pick_metric(raw_metrics, ["precision"])
        recall = _pick_metric(raw_metrics, ["recall"])
        map50 = _pick_metric(raw_metrics, ["map50("]) or _pick_metric(raw_metrics, ["map50"])
        map5095 = _pick_metric(raw_metrics, ["map50-95", "map"])

        if precision is not None:
            parsed["precision"] = precision
        if recall is not None:
            parsed["recall"] = recall
        if map50 is not None:
            parsed["mAP50"] = map50
        if map5095 is not None:
            parsed["mAP50-95"] = map5095
        if precision is not None and recall is not None and (precision + recall) > 0:
            parsed["F1"] = 2 * precision * recall / (precision + recall)
    else:
        top1 = _pick_metric(raw_metrics, ["top1", "accuracy"])
        top5 = _pick_metric(raw_metrics, ["top5"])
        if top1 is not None:
            parsed["top1"] = top1
        if top5 is not None:
            parsed["top5"] = top5
    return parsed


def get_primary_metric(task: str, metrics: Dict[str, float]) -> float:
    if task == "detect":
        return float(metrics.get("mAP50-95", -1.0))
    return float(metrics.get("top1", -1.0))


def default_hypotheses(task: str) -> List[Tuple[str, Dict[str, Any], Optional[str]]]:
    if task == "detect":
        return [
            (
                "h1_aug_light",
                {"hsv_h": 0.02, "hsv_s": 0.6, "hsv_v": 0.4, "fliplr": 0.5, "mosaic": 0.5, "mixup": 0.1},
                None,
            ),
            ("h2_imgsz_512", {"imgsz": 512}, None),
            ("h3_model_yolo11s", {}, "yolo11s.pt"),
            ("h4_optimizer_adamw_coslr", {"optimizer": "AdamW", "cos_lr": True, "lr0": 0.005}, None),
        ]
    return [
        ("h1_aug_light", {"erasing": 0.2, "hsv_h": 0.01, "hsv_s": 0.3, "hsv_v": 0.2}, None),
        ("h2_imgsz_192", {"imgsz": 192}, None),
        ("h3_model_yolo11s_cls", {}, "yolo11s-cls.pt"),
        ("h4_optimizer_adamw_coslr", {"optimizer": "AdamW", "cos_lr": True, "lr0": 0.003}, None),
    ]


def run_train_and_val(
    *,
    task: str,
    data: str,
    model_name: str,
    run_name: str,
    project: str,
    base_params: Dict[str, Any],
    extra_params: Optional[Dict[str, Any]] = None,
) -> ExperimentResult:
    params = dict(base_params)
    if extra_params:
        params.update(extra_params)

    model = YOLO(model_name)
    train_args: Dict[str, Any] = {
        "data": data,
        "project": project,
        "name": run_name,
        **params,
    }
    model.train(**train_args)

    val_name = f"{run_name}_val"
    val_results = model.val(data=data, split="val", project=project, name=val_name)
    results_dict = getattr(val_results, "results_dict", {}) or {}
    parsed = parse_metrics(task, results_dict)
    score = get_primary_metric(task, parsed)

    return ExperimentResult(
        stage="train_val",
        run_name=run_name,
        task=task,
        model=model_name,
        metrics=parsed,
        params=params,
        score=score,
    )


def compare_metrics(baseline: Dict[str, float], improved: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    all_keys = sorted(set(baseline.keys()) | set(improved.keys()))
    for key in all_keys:
        b = baseline.get(key)
        i = improved.get(key)
        if b is not None and i is not None:
            out[key] = i - b
    return out


def dump_reports(
    output_dir: Path,
    report: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    md_path = output_dir / "report.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# YOLOv11 Research Report",
        "",
        f"- Date: {report['meta']['date_utc']}",
        f"- Task: {report['meta']['task']}",
        f"- Dataset: {report['meta']['dataset']}",
        f"- Baseline model: {report['baseline']['model']}",
        f"- Improved model: {report['improved']['model']}",
        "",
        "## Baseline metrics",
    ]
    for k, v in report["baseline"]["metrics"].items():
        lines.append(f"- {k}: {v:.6f}")

    lines.extend(["", "## Hypotheses", ""])
    for h in report["hypotheses"]:
        lines.append(
            f"- {h['name']} | model={h['model']} | score={h['score']:.6f} | "
            f"metrics={h['metrics']}"
        )

    lines.extend(["", "## Improved metrics", ""])
    for k, v in report["improved"]["metrics"].items():
        lines.append(f"- {k}: {v:.6f}")

    lines.extend(["", "## Delta (improved - baseline)", ""])
    for k, v in report["comparison"]["delta"].items():
        lines.append(f"- {k}: {v:+.6f}")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_research(args: argparse.Namespace) -> Dict[str, Any]:
    task = args.task
    data = args.data
    project_input = Path(args.project)
    project = str((ROOT_DIR / project_input).resolve()) if not project_input.is_absolute() else str(project_input)
    exp_name = args.exp_name

    base_model = args.model or ("yolo11n.pt" if task == "detect" else "yolo11n-cls.pt")
    baseline_params: Dict[str, Any] = {
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "patience": args.patience,
        "verbose": True,
    }
    if task == "detect":
        baseline_params["plots"] = True

    baseline_run = f"{exp_name}_baseline"
    baseline = run_train_and_val(
        task=task,
        data=data,
        model_name=base_model,
        run_name=baseline_run,
        project=project,
        base_params=baseline_params,
    )

    hypotheses_results: List[Dict[str, Any]] = []
    best_hyp: Optional[ExperimentResult] = None
    hypotheses = default_hypotheses(task)
    for hyp_name, hyp_params, hyp_model in hypotheses:
        model_name = hyp_model or base_model
        run_name = f"{exp_name}_{hyp_name}"
        hyp = run_train_and_val(
            task=task,
            data=data,
            model_name=model_name,
            run_name=run_name,
            project=project,
            base_params=baseline_params,
            extra_params=hyp_params,
        )
        hypotheses_results.append(
            {
                "name": hyp_name,
                "run_name": hyp.run_name,
                "model": hyp.model,
                "params": hyp.params,
                "metrics": hyp.metrics,
                "score": hyp.score,
            }
        )
        if best_hyp is None or hyp.score > best_hyp.score:
            best_hyp = hyp

    assert best_hyp is not None, "Не удалось вычислить лучшую гипотезу."

    improved_epochs = int(max(args.epochs, round(args.epochs * 1.3)))
    improved_params = dict(best_hyp.params)
    improved_params["epochs"] = improved_epochs

    improved_model = best_hyp.model
    improved_run = f"{exp_name}_improved"
    improved = run_train_and_val(
        task=task,
        data=data,
        model_name=improved_model,
        run_name=improved_run,
        project=project,
        base_params=improved_params,
    )

    delta = compare_metrics(baseline.metrics, improved.metrics)
    report = {
        "meta": {
            "date_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "task": task,
            "dataset": data,
            "project": project,
            "experiment_name": exp_name,
            "primary_metric": "mAP50-95" if task == "detect" else "top1",
            "metrics_rationale": (
                "Для детекции выбраны mAP50-95, mAP50, precision, recall, F1: "
                "они отражают точность локализации и баланс ложноположительных/пропусков."
                if task == "detect"
                else "Для классификации выбраны top1 и top5 accuracy: "
                "они отражают качество распознавания классов."
            ),
        },
        "baseline": {
            "run_name": baseline.run_name,
            "model": baseline.model,
            "params": baseline.params,
            "metrics": baseline.metrics,
            "score": baseline.score,
        },
        "hypotheses": hypotheses_results,
        "best_hypothesis": {
            "run_name": best_hyp.run_name,
            "model": best_hyp.model,
            "params": best_hyp.params,
            "metrics": best_hyp.metrics,
            "score": best_hyp.score,
        },
        "improved": {
            "run_name": improved.run_name,
            "model": improved.model,
            "params": improved.params,
            "metrics": improved.metrics,
            "score": improved.score,
        },
        "comparison": {"delta": delta},
    }

    out_dir = Path(project) / f"{exp_name}_report"
    dump_reports(out_dir, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Исследование детекции/классификации на Ultralytics YOLOv11."
    )
    parser.add_argument("--list-datasets", action="store_true", help="Показать варианты маленьких датасетов.")
    parser.add_argument("--task", choices=["detect", "classify"], default="detect", help="Тип задачи.")
    parser.add_argument("--data", type=str, default="coco128.yaml", help="Путь к data yaml или имя датасета.")
    parser.add_argument("--model", type=str, default="", help="Начальная модель.")
    parser.add_argument("--project", type=str, default="runs/yolo11_research", help="Папка экспериментов.")
    parser.add_argument("--exp-name", type=str, default="exp", help="Префикс имени эксперимента.")
    parser.add_argument("--epochs", type=int, default=15, help="Число эпох для baseline.")
    parser.add_argument("--imgsz", type=int, default=640, help="Размер изображения.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", type=str, default="cpu", help="Устройство: cpu, 0, 0,1 ...")
    parser.add_argument("--workers", type=int, default=2, help="Число workers.")
    parser.add_argument("--patience", type=int, default=15, help="Ранняя остановка.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_datasets:
        list_dataset_options()
        return

    if not args.model:
        args.model = "yolo11n.pt" if args.task == "detect" else "yolo11n-cls.pt"

    project_input = Path(args.project)
    project_abs = (ROOT_DIR / project_input).resolve() if not project_input.is_absolute() else project_input
    args.project = str(project_abs)
    os.makedirs(args.project, exist_ok=True)
    report = run_research(args)

    print("\n=== Исследование завершено ===")
    print(f"Task: {report['meta']['task']}")
    print(f"Dataset: {report['meta']['dataset']}")
    print(f"Baseline score: {report['baseline']['score']:.6f}")
    print(f"Improved score: {report['improved']['score']:.6f}")
    print("Отчет:")
    print(f"- {Path(args.project) / (args.exp_name + '_report') / 'report.json'}")
    print(f"- {Path(args.project) / (args.exp_name + '_report') / 'report.md'}")


if __name__ == "__main__":
    main()
