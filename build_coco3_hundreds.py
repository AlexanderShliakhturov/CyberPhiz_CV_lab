from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, ImageEnhance


ROOT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = Path("/home/alexander/datasets/coco128")


def parse_label_file(path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        p = line.strip().split()
        if len(p) != 5:
            continue
        cls = int(float(p[0]))
        x, y, w, h = map(float, p[1:])
        rows.append((cls, x, y, w, h))
    return rows


def write_label_file(path: Path, rows: Sequence[Tuple[int, float, float, float, float]]) -> None:
    lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def augment_image(img: Image.Image, aug_id: int) -> Tuple[Image.Image, bool]:
    # 0 = original. Others are deterministic, light augmentations.
    if aug_id == 0:
        return img, False

    do_flip = aug_id % 2 == 1
    out = img.transpose(Image.FLIP_LEFT_RIGHT) if do_flip else img.copy()

    if aug_id % 3 == 0:
        out = ImageEnhance.Brightness(out).enhance(1.20)
    elif aug_id % 3 == 1:
        out = ImageEnhance.Contrast(out).enhance(1.20)
    else:
        out = ImageEnhance.Color(out).enhance(1.20)
    return out, do_flip


def remap_and_filter_rows(
    rows: Sequence[Tuple[int, float, float, float, float]],
    class_map: Dict[int, int],
    do_flip: bool,
) -> List[Tuple[int, float, float, float, float]]:
    out = []
    for cls, x, y, w, h in rows:
        if cls not in class_map:
            continue
        nx = 1.0 - x if do_flip else x
        out.append((class_map[cls], nx, y, w, h))
    return out


def create_dataset(
    *,
    selected_classes: Sequence[int],
    target_total_images: int,
    val_fraction: float,
    seed: int,
    out_root: Path,
) -> Path:
    src_img = SOURCE_ROOT / "images" / "train2017"
    src_lbl = SOURCE_ROOT / "labels" / "train2017"
    if not src_img.exists() or not src_lbl.exists():
        raise FileNotFoundError(f"Source dataset not found at {SOURCE_ROOT}")

    out_img_train = out_root / "images" / "train"
    out_img_val = out_root / "images" / "val"
    out_lbl_train = out_root / "labels" / "train"
    out_lbl_val = out_root / "labels" / "val"

    if out_root.exists():
        shutil.rmtree(out_root)
    out_img_train.mkdir(parents=True, exist_ok=True)
    out_img_val.mkdir(parents=True, exist_ok=True)
    out_lbl_train.mkdir(parents=True, exist_ok=True)
    out_lbl_val.mkdir(parents=True, exist_ok=True)

    class_map = {src_cls: new_cls for new_cls, src_cls in enumerate(selected_classes)}
    selected_imgs: List[Path] = []
    src_rows: Dict[str, List[Tuple[int, float, float, float, float]]] = {}
    for ip in sorted(src_img.glob("*.jpg")):
        lp = src_lbl / f"{ip.stem}.txt"
        if not lp.exists():
            continue
        rows = parse_label_file(lp)
        filtered = [r for r in rows if r[0] in class_map]
        if not filtered:
            continue
        selected_imgs.append(ip)
        src_rows[ip.stem] = filtered

    if not selected_imgs:
        raise RuntimeError("No images found for selected classes.")

    random.seed(seed)
    random.shuffle(selected_imgs)

    n_val = max(1, int(round(len(selected_imgs) * val_fraction)))
    val_base = selected_imgs[:n_val]
    train_base = selected_imgs[n_val:]

    total_base = len(selected_imgs)
    repeats = max(1, target_total_images // total_base)
    # keep val smaller to avoid huge eval time
    val_repeats = max(1, repeats // 2)
    train_repeats = repeats

    def dump_split(
        base_imgs: Sequence[Path],
        out_img_dir: Path,
        out_lbl_dir: Path,
        n_repeats: int,
    ) -> int:
        count = 0
        for img_path in base_imgs:
            base_rows = src_rows[img_path.stem]
            img = Image.open(img_path).convert("RGB")
            for r in range(n_repeats):
                aug_img, do_flip = augment_image(img, r)
                rows = remap_and_filter_rows(base_rows, class_map, do_flip)
                if not rows:
                    continue
                name = f"{img_path.stem}_r{r:02d}.jpg"
                aug_img.save(out_img_dir / name, quality=95)
                write_label_file(out_lbl_dir / f"{Path(name).stem}.txt", rows)
                count += 1
        return count

    n_train = dump_split(train_base, out_img_train, out_lbl_train, train_repeats)
    n_val_new = dump_split(val_base, out_img_val, out_lbl_val, val_repeats)

    names = {i: n for i, n in enumerate(["person", "car", "dog"][: len(selected_classes)])}
    yaml_text = "\n".join(
        [
            f"path: {out_root.as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            *[f"  {k}: {v}" for k, v in names.items()],
            "",
        ]
    )
    (out_root / "data.yaml").write_text(yaml_text, encoding="utf-8")

    summary = "\n".join(
        [
            f"base_images_selected: {len(selected_imgs)}",
            f"train_base: {len(train_base)}",
            f"val_base: {len(val_base)}",
            f"train_images_generated: {n_train}",
            f"val_images_generated: {n_val_new}",
            f"total_images_generated: {n_train + n_val_new}",
            f"repeats_train: {train_repeats}",
            f"repeats_val: {val_repeats}",
            f"classes_src_to_new: {class_map}",
        ]
    )
    (out_root / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    return out_root / "data.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reduced COCO dataset with 3 classes and hundreds of images.")
    parser.add_argument("--classes", type=str, default="0,2,16", help="Source class ids from COCO.")
    parser.add_argument("--target-total", type=int, default=450, help="Approx total generated images.")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction by base images.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--out-dir", type=str, default="datasets/coco3_hundreds", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    classes = [int(x.strip()) for x in args.classes.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (ROOT_DIR / out_dir).resolve()

    data_yaml = create_dataset(
        selected_classes=classes,
        target_total_images=args.target_total,
        val_fraction=args.val_fraction,
        seed=args.seed,
        out_root=out_dir,
    )
    print(f"Dataset created: {out_dir}")
    print(f"YAML: {data_yaml}")
    print((out_dir / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
