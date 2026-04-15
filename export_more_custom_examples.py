from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import pil_to_tensor

from custom_detector_coco8 import build_model, collect_images


ROOT_DIR = Path(__file__).resolve().parent


def parse_names_from_yaml(yaml_path: Path) -> List[str]:
    names = []
    in_names = False
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        s = line.rstrip()
        if not s:
            continue
        if s.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if not s.startswith("  "):
                break
            if ":" in s:
                _, value = s.split(":", 1)
                names.append(value.strip())
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Export more prediction examples from trained custom detector.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint.")
    parser.add_argument("--data-yaml", type=str, required=True, help="Dataset yaml used for training.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="Split to visualize.")
    parser.add_argument("--max-images", type=int, default=80, help="Number of images to save.")
    parser.add_argument("--conf", type=float, default=0.25, help="Score threshold.")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory for images.")
    parser.add_argument("--device", type=str, default="", help="cpu or cuda.")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    data_yaml = Path(args.data_yaml)
    if not data_yaml.is_absolute():
        data_yaml = (ROOT_DIR / data_yaml).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    root = data_yaml.parent
    names = parse_names_from_yaml(data_yaml)
    num_classes = len(names)
    if num_classes == 0:
        raise RuntimeError("Failed to parse class names from data.yaml")

    split_dir = root / "images" / args.split
    image_paths = [str(p) for p in collect_images(str(split_dir))]
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = ckpt.parent / f"pred_examples_{args.split}_extra"
    if not out_dir.is_absolute():
        out_dir = (ROOT_DIR / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(num_classes_with_bg=num_classes + 1, pretrained=False).to(device)
    state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state)
    model.eval()

    saved = 0
    with torch.no_grad():
        for p in image_paths:
            if saved >= args.max_images:
                break
            img = Image.open(p).convert("RGB")
            x = pil_to_tensor(img).float() / 255.0
            pred = model([x.to(device)])[0]

            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = (pred["labels"].cpu().numpy() - 1).clip(min=0)

            draw = ImageDraw.Draw(img)
            for b, s, l in zip(boxes, scores, labels):
                if float(s) < args.conf:
                    continue
                x1, y1, x2, y2 = b.tolist()
                cname = names[int(l)] if 0 <= int(l) < len(names) else str(int(l))
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1 + 2, y1 + 2), f"{cname} {float(s):.2f}", fill="yellow")

            save_path = out_dir / Path(p).name
            img.save(save_path)
            saved += 1

    print(f"Saved {saved} images to: {out_dir}")


if __name__ == "__main__":
    main()
