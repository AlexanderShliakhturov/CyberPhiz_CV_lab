from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou
from torchvision.transforms.functional import pil_to_tensor
from ultralytics.data.utils import check_det_dataset


ROOT_DIR = Path(__file__).resolve().parent


def yolo_xywh_to_xyxy(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    xc = boxes[:, 0] * w
    yc = boxes[:, 1] * h
    bw = boxes[:, 2] * w
    bh = boxes[:, 3] * h
    x1 = np.clip(xc - bw / 2, 0, w - 1)
    y1 = np.clip(yc - bh / 2, 0, h - 1)
    x2 = np.clip(xc + bw / 2, 0, w - 1)
    y2 = np.clip(yc + bh / 2, 0, h - 1)
    return np.stack([x1, y1, x2, y2], axis=1)


def collect_images(path_like: str) -> List[Path]:
    p = Path(path_like)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted([x for x in p.rglob("*") if x.suffix.lower() in exts])
    if p.is_file():
        lines = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        return [Path(x) for x in lines]
    raise FileNotFoundError(f"Path not found: {path_like}")


def img_to_label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return img_path.with_suffix(".txt")


class YOLOTxtDetectionDataset(Dataset):
    def __init__(
        self,
        images: Sequence[Path],
        names: Sequence[str],
        train_mode: bool = False,
        repeat: int = 1,
        hflip_prob: float = 0.5,
    ):
        self.images = list(images)
        self.names = list(names)
        self.train_mode = train_mode
        self.repeat = max(1, int(repeat))
        self.hflip_prob = float(hflip_prob)

    def __len__(self) -> int:
        return len(self.images) * self.repeat

    def __getitem__(self, idx: int):
        img_path = self.images[idx % len(self.images)]
        lbl_path = img_to_label_path(img_path)
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        labels: List[int] = []
        boxes_xywh: List[List[float]] = []
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0]))
                x, y, bw, bh = map(float, parts[1:])
                labels.append(cls)
                boxes_xywh.append([x, y, bw, bh])

        if boxes_xywh:
            boxes = yolo_xywh_to_xyxy(np.array(boxes_xywh, dtype=np.float32), w, h)
            labels_np = np.array(labels, dtype=np.int64) + 1  # background=0 in torchvision
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels_np = np.zeros((0,), dtype=np.int64)

        # Light augmentation for tiny datasets.
        if self.train_mode and boxes.shape[0] > 0 and random.random() < self.hflip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            x1 = boxes[:, 0].copy()
            x2 = boxes[:, 2].copy()
            boxes[:, 0] = (w - 1) - x2
            boxes[:, 2] = (w - 1) - x1

        image_t = pil_to_tensor(image).float() / 255.0
        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels_np),
            "image_id": torch.tensor([idx], dtype=torch.int64),
        }
        return image_t, target, str(img_path)


def collate_fn(batch):
    images, targets, paths = zip(*batch)
    return list(images), list(targets), list(paths)


def build_model(num_classes_with_bg: int, pretrained: bool = True) -> nn.Module:
    if pretrained:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
    else:
        model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
    return model


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for images, targets, _ in loader:
        images = [x.to(device) for x in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(1, len(loader))


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[Dict], List[Dict], List[str]]:
    model.eval()
    all_preds: List[Dict] = []
    all_targets: List[Dict] = []
    all_paths: List[str] = []
    with torch.no_grad():
        for images, targets, paths in loader:
            outputs = model([x.to(device) for x in images])
            for out, tgt, path in zip(outputs, targets, paths):
                pred = {
                    "boxes": out["boxes"].cpu(),
                    "scores": out["scores"].cpu(),
                    "labels": (out["labels"].cpu() - 1).clamp(min=0),
                }
                gt = {
                    "boxes": tgt["boxes"].cpu(),
                    "labels": (tgt["labels"].cpu() - 1).clamp(min=0),
                }
                all_preds.append(pred)
                all_targets.append(gt)
                all_paths.append(path)
    return all_preds, all_targets, all_paths


def ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    # 101-point interpolation (COCO-like integration).
    if precision.size == 0:
        return 0.0
    rec_points = np.linspace(0.0, 1.0, 101)
    vals = []
    for r in rec_points:
        p = precision[recall >= r]
        vals.append(float(np.max(p)) if p.size else 0.0)
    return float(np.mean(vals))


def compute_ap_for_class(
    cls_id: int,
    preds: List[Dict],
    targets: List[Dict],
    iou_thr: float,
) -> float:
    gt_count = 0
    gt_per_image: Dict[int, Dict[str, np.ndarray]] = {}
    pred_records: List[Tuple[int, float, np.ndarray]] = []

    for img_idx, (p, t) in enumerate(zip(preds, targets)):
        gt_mask = (t["labels"].numpy() == cls_id)
        gt_boxes = t["boxes"].numpy()[gt_mask]
        gt_count += len(gt_boxes)
        gt_per_image[img_idx] = {"boxes": gt_boxes, "matched": np.zeros(len(gt_boxes), dtype=bool)}

        pr_mask = (p["labels"].numpy() == cls_id)
        pr_boxes = p["boxes"].numpy()[pr_mask]
        pr_scores = p["scores"].numpy()[pr_mask]
        for b, s in zip(pr_boxes, pr_scores):
            pred_records.append((img_idx, float(s), b))

    if gt_count == 0:
        return float("nan")
    if len(pred_records) == 0:
        return 0.0

    pred_records.sort(key=lambda x: x[1], reverse=True)
    tp = np.zeros(len(pred_records), dtype=np.float32)
    fp = np.zeros(len(pred_records), dtype=np.float32)

    for i, (img_idx, _score, pbox) in enumerate(pred_records):
        g = gt_per_image[img_idx]
        gboxes = g["boxes"]
        if len(gboxes) == 0:
            fp[i] = 1
            continue
        ious = box_iou(torch.tensor([pbox], dtype=torch.float32), torch.tensor(gboxes, dtype=torch.float32))[0].numpy()
        best_idx = int(np.argmax(ious))
        best_iou = float(ious[best_idx])
        if best_iou >= iou_thr and not g["matched"][best_idx]:
            tp[i] = 1
            g["matched"][best_idx] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(gt_count, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
    return ap_from_pr(precision, recall)


def compute_precision_recall_f1(
    preds: List[Dict],
    targets: List[Dict],
    iou_thr: float = 0.5,
    score_thr: float = 0.25,
) -> Dict[str, float]:
    tp = 0
    fp = 0
    fn = 0
    for p, t in zip(preds, targets):
        gt_boxes = t["boxes"]
        gt_labels = t["labels"]
        pred_mask = p["scores"] >= score_thr
        pr_boxes = p["boxes"][pred_mask]
        pr_labels = p["labels"][pred_mask]
        used = torch.zeros(len(gt_boxes), dtype=torch.bool)

        order = torch.argsort(p["scores"][pred_mask], descending=True)
        pr_boxes = pr_boxes[order]
        pr_labels = pr_labels[order]

        for pb, pl in zip(pr_boxes, pr_labels):
            same_cls = torch.where(gt_labels == pl)[0]
            same_cls = same_cls[~used[same_cls]]
            if len(same_cls) == 0:
                fp += 1
                continue
            ious = box_iou(pb.unsqueeze(0), gt_boxes[same_cls]).squeeze(0)
            best_local = int(torch.argmax(ious))
            best_iou = float(ious[best_local].item())
            best_gt = int(same_cls[best_local].item())
            if best_iou >= iou_thr:
                tp += 1
                used[best_gt] = True
            else:
                fp += 1
        fn += int((~used).sum().item())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = (2 * precision * recall / max(precision + recall, 1e-9)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "F1": f1}


def evaluate_detection(
    preds: List[Dict],
    targets: List[Dict],
    num_classes: int,
) -> Dict[str, float]:
    iou_thresholds = [0.5 + i * 0.05 for i in range(10)]
    aps_per_thr: Dict[float, List[float]] = {thr: [] for thr in iou_thresholds}

    for thr in iou_thresholds:
        for cls_id in range(num_classes):
            ap = compute_ap_for_class(cls_id, preds, targets, thr)
            if not np.isnan(ap):
                aps_per_thr[thr].append(ap)

    map50 = float(np.mean(aps_per_thr[0.5])) if aps_per_thr[0.5] else 0.0
    map5095 = float(np.mean([np.mean(v) for v in aps_per_thr.values() if len(v) > 0])) if aps_per_thr else 0.0
    prf = compute_precision_recall_f1(preds, targets, iou_thr=0.5, score_thr=0.25)
    return {
        "precision": float(prf["precision"]),
        "recall": float(prf["recall"]),
        "mAP50": map50,
        "mAP50-95": map5095,
        "F1": float(prf["F1"]),
    }


def draw_predictions(
    model: nn.Module,
    image_paths: Sequence[str],
    names: Sequence[str],
    out_dir: Path,
    device: torch.device,
    conf_thr: float,
    max_images: int,
) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    model.eval()
    with torch.no_grad():
        for p in image_paths[:max_images]:
            img = Image.open(p).convert("RGB")
            x = pil_to_tensor(img).float() / 255.0
            pred = model([x.to(device)])[0]
            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = (pred["labels"].cpu().numpy() - 1).clip(min=0)

            draw = ImageDraw.Draw(img)
            for b, s, l in zip(boxes, scores, labels):
                if s < conf_thr:
                    continue
                x1, y1, x2, y2 = b.tolist()
                cname = names[int(l)] if 0 <= int(l) < len(names) else str(int(l))
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1 + 2, y1 + 2), f"{cname} {s:.2f}", fill="yellow")

            save_path = out_dir / Path(p).name
            img.save(save_path)
            saved.append(str(save_path))
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom detector training on COCO8 with manual metrics.")
    parser.add_argument("--data", type=str, default="coco8.yaml", help="YOLO dataset yaml (default: coco8.yaml).")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay.")
    parser.add_argument("--workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="", help="cpu or cuda.")
    parser.add_argument("--examples", type=int, default=8, help="How many predicted images to save.")
    parser.add_argument("--project", type=str, default="runs/custom_detector_coco8", help="Output directory.")
    parser.add_argument("--run-name", type=str, default="fasterrcnn_mobilenetv3", help="Run folder name.")
    parser.add_argument("--repeat-train", type=int, default=8, help="Repeat factor for tiny train sets.")
    parser.add_argument("--hflip-prob", type=float, default=0.5, help="Horizontal flip probability for train.")
    parser.add_argument("--freeze-backbone-epochs", type=int, default=8, help="Warmup epochs with frozen backbone.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    out_root = Path(args.project)
    if not out_root.is_absolute():
        out_root = (ROOT_DIR / out_root).resolve()
    run_dir = out_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = check_det_dataset(args.data)
    train_imgs = collect_images(str(dataset_cfg["train"]))
    val_imgs = collect_images(str(dataset_cfg["val"]))
    names = dataset_cfg["names"]
    if isinstance(names, dict):
        class_names = [names.get(i, names.get(str(i), str(i))) for i in range(len(names))]
    else:
        class_names = [names[i] for i in range(len(names))]
    num_classes = len(class_names)

    train_ds = YOLOTxtDetectionDataset(
        train_imgs,
        class_names,
        train_mode=True,
        repeat=args.repeat_train,
        hflip_prob=args.hflip_prob,
    )
    val_ds = YOLOTxtDetectionDataset(val_imgs, class_names)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes_with_bg=num_classes + 1, pretrained=True).to(device)

    # Warmup: first learn the newly initialized classifier/regressor heads.
    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.1)

    history = []
    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
        loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({"epoch": epoch, "loss": loss})
        print(f"Epoch {epoch}/{args.epochs} | loss={loss:.4f} | lr={current_lr:.6f}")

    preds, targets, paths = collect_predictions(model, val_loader, device)
    metrics = evaluate_detection(preds, targets, num_classes=num_classes)

    examples_dir = run_dir / "pred_examples"
    saved_examples = draw_predictions(
        model=model,
        image_paths=paths,
        names=class_names,
        out_dir=examples_dir,
        device=device,
        conf_thr=0.25,
        max_images=args.examples,
    )

    report = {
        "meta": {
            "date_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "dataset": args.data,
            "task": "detect",
            "implementation": "custom_torchvision_fasterrcnn_mobilenet_v3_320_fpn",
            "project_dir": str(run_dir),
            "device": str(device),
            "num_classes": num_classes,
        },
        "train_config": {
            "epochs": args.epochs,
            "batch": args.batch,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "workers": args.workers,
            "seed": args.seed,
            "repeat_train": args.repeat_train,
            "hflip_prob": args.hflip_prob,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
        },
        "metrics": metrics,
        "history": history,
        "prediction_examples": {
            "saved_dir": str(examples_dir),
            "images": saved_examples,
        },
    }

    report_json = run_dir / "report.json"
    report_md = run_dir / "report.md"
    checkpoint_path = run_dir / "model_final.pth"
    torch.save(model.state_dict(), checkpoint_path)

    with report_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_lines = [
        "# Custom Detector Report",
        "",
        f"- Dataset: {args.data}",
        f"- Model: {report['meta']['implementation']}",
        f"- Device: {report['meta']['device']}",
        "",
        "## Metrics",
    ]
    for k, v in metrics.items():
        md_lines.append(f"- {k}: {v:.6f}")
    md_lines.extend(["", "## Prediction Examples"])
    for p in saved_examples:
        md_lines.append(f"- {p}")
    with report_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\nTraining and evaluation finished.")
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Report JSON: {report_json}")
    print(f"Report MD:   {report_md}")


if __name__ == "__main__":
    main()
