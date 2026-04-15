from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

ROOT_DIR = Path(__file__).resolve().parent


def _to_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
        if math.isnan(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def parse_detect_metrics(raw: Dict[str, Any]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for k, v in raw.items():
        lk = k.lower()
        fv = _to_float(v)
        if fv is None:
            continue
        if "precision" in lk:
            metrics["precision"] = fv
        elif "recall" in lk:
            metrics["recall"] = fv
        elif "map50-95" in lk or "map50-95(" in lk:
            metrics["mAP50-95"] = fv
        elif "map50(" in lk or "map50" in lk:
            metrics["mAP50"] = fv
    p = metrics.get("precision")
    r = metrics.get("recall")
    if p is not None and r is not None and (p + r) > 0:
        metrics["F1"] = 2 * p * r / (p + r)
    return metrics


def class_name(names: Any, idx: int) -> str:
    if isinstance(names, list):
        return names[idx] if 0 <= idx < len(names) else str(idx)
    if isinstance(names, dict):
        return str(names.get(idx, names.get(str(idx), idx)))
    return str(idx)


def save_prediction_examples(
    *,
    weights_path: Path,
    data: str,
    project: str,
    pred_name: str,
    imgsz: int,
    max_images: int,
    conf: float = 0.25,
) -> Dict[str, Any]:
    dataset = check_det_dataset(data)
    source = dataset.get("val")
    if isinstance(source, list):
        source = source[0]

    infer_model = YOLO(str(weights_path))
    results = infer_model.predict(
        source=source,
        save=True,
        project=project,
        name=pred_name,
        exist_ok=True,
        imgsz=imgsz,
        conf=conf,
        stream=False,
    )

    pred_dir = Path(project) / pred_name
    images = sorted(pred_dir.glob("*.jpg")) + sorted(pred_dir.glob("*.png"))
    image_paths = [str(p) for p in images[:max_images]]

    predicted_counts: Counter[int] = Counter()
    for r in results:
        if getattr(r, "boxes", None) is None or r.boxes.cls is None:
            continue
        predicted_counts.update([int(x) for x in r.boxes.cls.tolist()])

    names = dataset.get("names", {})
    top_classes = [
        {"class_id": cid, "class_name": class_name(names, cid), "count": cnt}
        for cid, cnt in predicted_counts.most_common(15)
    ]

    return {
        "saved_dir": str(pred_dir),
        "images": image_paths,
        "num_saved_images": len(image_paths),
        "unique_predicted_classes": len(predicted_counts),
        "top_predicted_classes": top_classes,
    }


def train_and_eval(
    model_name: str,
    run_name: str,
    train_args: Dict[str, Any],
    *,
    data: str,
    project: str,
    max_examples: int,
) -> Dict[str, Any]:
    model = YOLO(model_name)
    model.train(data=data, project=project, name=run_name, plots=True, **train_args)
    val = model.val(data=data, split="val", project=project, name=f"{run_name}_val", plots=True)
    raw = getattr(val, "results_dict", {}) or {}
    metrics = parse_detect_metrics(raw)
    weights_path = Path(project) / run_name / "weights" / "best.pt"
    pred_name = f"{run_name}_pred_examples"
    pred_info = save_prediction_examples(
        weights_path=weights_path,
        data=data,
        project=project,
        pred_name=pred_name,
        imgsz=int(train_args.get("imgsz", 640)),
        max_images=max_examples,
    )
    return {
        "run_name": run_name,
        "model": model_name,
        "train_args": train_args,
        "metrics": metrics,
        "primary_score": metrics.get("mAP50-95", -1.0),
        "weights_path": str(weights_path),
        "prediction_examples": pred_info,
    }


def choose_improved(
    baseline: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str]:
    best_candidate = max(candidates, key=lambda x: float(x.get("primary_score", -1.0)))
    if float(best_candidate.get("primary_score", -1.0)) >= float(baseline.get("primary_score", -1.0)):
        return best_candidate, "best_candidate"
    # Если все гипотезы хуже baseline, фиксируем improved = baseline (не деградируем).
    return baseline, "baseline_floor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline and improved YOLOv11 on detection dataset.")
    parser.add_argument("--data", type=str, default="coco128.yaml", help="Dataset yaml/path.")
    parser.add_argument(
        "--project",
        type=str,
        default=str((ROOT_DIR / "runs" / "baseline_vs_improved").resolve()),
        help="Absolute/relative output directory.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu or gpu index.")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--baseline-epochs", type=int, default=10, help="Baseline epochs.")
    parser.add_argument("--improved-epochs", type=int, default=16, help="Improved candidate epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Seed.")
    parser.add_argument("--examples", type=int, default=24, help="How many saved prediction images per model.")
    parser.add_argument("--tag", type=str, default="exp", help="Run prefix.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = (ROOT_DIR / project_path).resolve()
    project = str(project_path)
    Path(project).mkdir(parents=True, exist_ok=True)

    print(f"Saving all artifacts to: {project}")

    baseline_args = {
        "epochs": args.baseline_epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "seed": args.seed,
        "patience": args.baseline_epochs,
        "optimizer": "SGD",
        "lr0": 0.01,
        "mosaic": 0.0,
        "mixup": 0.0,
        "fliplr": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
    }

    baseline = train_and_eval(
        "yolo11n.pt",
        f"{args.tag}_baseline",
        baseline_args,
        data=args.data,
        project=project,
        max_examples=args.examples,
    )

    improved_candidates_cfg = [
        (
            "yolo11n_adamw_aug",
            "yolo11n.pt",
            {
                "epochs": args.improved_epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": args.device,
                "workers": args.workers,
                "seed": args.seed,
                "patience": args.improved_epochs,
                "optimizer": "AdamW",
                "lr0": 0.004,
                "cos_lr": True,
                "mosaic": 0.3,
                "mixup": 0.05,
                "fliplr": 0.5,
                "hsv_h": 0.015,
                "hsv_s": 0.4,
                "hsv_v": 0.3,
            },
        ),
        (
            "yolo11s_adamw_aug",
            "yolo11s.pt",
            {
                "epochs": args.improved_epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": args.device,
                "workers": args.workers,
                "seed": args.seed,
                "patience": args.improved_epochs,
                "optimizer": "AdamW",
                "lr0": 0.0035,
                "cos_lr": True,
                "mosaic": 0.3,
                "mixup": 0.05,
                "fliplr": 0.5,
                "hsv_h": 0.015,
                "hsv_s": 0.4,
                "hsv_v": 0.3,
            },
        ),
        (
            "baseline_finetune",
            baseline["weights_path"],
            {
                "epochs": max(6, args.improved_epochs // 2),
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": args.device,
                "workers": args.workers,
                "seed": args.seed,
                "patience": max(6, args.improved_epochs // 2),
                "optimizer": "SGD",
                "lr0": 0.002,
                "mosaic": 0.15,
                "mixup": 0.0,
                "fliplr": 0.5,
                "hsv_h": 0.01,
                "hsv_s": 0.2,
                "hsv_v": 0.15,
            },
        ),
    ]

    improved_candidates: List[Dict[str, Any]] = []
    for cfg_name, cfg_model, cfg_args in improved_candidates_cfg:
        result = train_and_eval(
            cfg_model,
            f"{args.tag}_improved_{cfg_name}",
            cfg_args,
            data=args.data,
            project=project,
            max_examples=args.examples,
        )
        result["candidate_name"] = cfg_name
        improved_candidates.append(result)

    improved, improved_policy = choose_improved(baseline, improved_candidates)

    delta: Dict[str, float] = {}
    for k in sorted(set(baseline["metrics"]) | set(improved["metrics"])):
        if k in baseline["metrics"] and k in improved["metrics"]:
            delta[k] = improved["metrics"][k] - baseline["metrics"][k]

    report = {
        "meta": {
            "date_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "dataset": args.data,
            "task": "detect",
            "primary_metric": "mAP50-95",
            "project_dir": project,
            "selection_policy": improved_policy,
        },
        "baseline": baseline,
        "improved_candidates": improved_candidates,
        "improved": improved,
        "comparison": {"delta": delta},
    }

    out = Path(project) / "report.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# Baseline vs Improved (YOLOv11)",
        "",
        f"- Date: {report['meta']['date_utc']}",
        f"- Dataset: {report['meta']['dataset']}",
        f"- Selection policy: {report['meta']['selection_policy']}",
        "",
        "## Baseline metrics",
    ]
    for k, v in report["baseline"]["metrics"].items():
        md_lines.append(f"- {k}: {v:.6f}")

    md_lines.extend(["", "## Improved metrics"])
    for k, v in report["improved"]["metrics"].items():
        md_lines.append(f"- {k}: {v:.6f}")

    md_lines.extend(["", "## Delta (improved - baseline)"])
    for k, v in report["comparison"]["delta"].items():
        md_lines.append(f"- {k}: {v:+.6f}")

    md_lines.extend(["", "## Predicted Class Coverage"])
    b_cov = report["baseline"]["prediction_examples"]["unique_predicted_classes"]
    i_cov = report["improved"]["prediction_examples"]["unique_predicted_classes"]
    md_lines.append(f"- Baseline unique predicted classes: {b_cov}")
    md_lines.append(f"- Improved unique predicted classes: {i_cov}")

    md_lines.extend(["", "## Prediction Examples (Baseline)"])
    for p in report["baseline"]["prediction_examples"]["images"]:
        md_lines.append(f"- {p}")

    md_lines.extend(["", "## Prediction Examples (Improved)"])
    for p in report["improved"]["prediction_examples"]["images"]:
        md_lines.append(f"- {p}")

    out_md = Path(project) / "report.md"
    with out_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved report: {out}")
    print(f"Saved report: {out_md}")


if __name__ == "__main__":
    main()
