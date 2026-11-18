"""Frozen SSDLite on a deterministic COCO validation subset used for development.

This is an engineering probe on two traffic classes with fixed size, score and
IoU rules. Images and pretrained weights are cached locally.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
    ssdlite320_mobilenet_v3_large,
)
from torchvision.ops import box_iou

from rareprior.operators.blur import GaussianBlur
from rareprior.solvers.cg import tikhonov
from rareprior.utils.manifest import atomic_output

SIZE = 320
CATEGORIES = {10: "traffic_light", 13: "stop_sign"}


def scaled_box(ann, image, pixels=SIZE):
    x, y, w, h = ann["bbox"]
    sx, sy = pixels / image["width"], pixels / image["height"]
    return [x * sx, y * sy, (x + w) * sx, (y + h) * sy]


def small(ann, image):
    x1, y1, x2, y2 = scaled_box(ann, image)
    w, h = x2 - x1, y2 - y1
    return min(w, h) >= 6 and w * h < 32 ** 2


def dev_subset(annotations, per_class):
    images = {im["id"]: im for im in annotations["images"] if im["license"] in range(1, 7)}
    by_image = defaultdict(list)
    crowd_ids = set()
    for a in annotations["annotations"]:
        if a["image_id"] in images and a["category_id"] in CATEGORIES:
            if a["iscrowd"]:
                crowd_ids.add(a["image_id"])
            else:
                by_image[a["image_id"]].append(a)
    selected = set()
    for c in CATEGORIES:
        eligible = [i for i in sorted(by_image) if i not in crowd_ids
                    and any(a["category_id"] == c and small(a, images[i]) for a in by_image[i])]
        if len(eligible) < per_class:
            raise ValueError(f"Insufficient annotated development images for category {c}.")
        selected.update(eligible[:per_class])
    return [(images[i], sorted(by_image[i], key=lambda a: a["id"])) for i in sorted(selected)]


def cached_image(img, cache, pixels):
    path = cache / img["file_name"]
    if not path.exists():
        url = f"https://s3.amazonaws.com/images.cocodataset.org/val2017/{path.name}"
        with atomic_output(path) as temp, urllib.request.urlopen(url, timeout=30) as response:
            temp.write_bytes(response.read())
    with Image.open(path) as raw:
        return torchvision.transforms.functional.pil_to_tensor(
            raw.convert("RGB").resize((pixels, pixels), Image.Resampling.BILINEAR)
        ).float() / 255


def counts(prediction, annotations, image, category, pixels):
    selected = prediction["labels"].eq(category) & prediction["scores"].ge(.25)
    boxes = prediction["boxes"][selected]
    relevant = [a for a in annotations if a["category_id"] == category]
    targets = torch.tensor([scaled_box(a, image, pixels) for a in relevant]).reshape(-1, 4)
    iou = box_iou(boxes, targets)
    matched = set()
    for row in iou:
        if len(row):
            for j in row.argsort(descending=True).tolist():
                if row[j] < .5:
                    break
                if j not in matched:
                    matched.add(j)
                    break
    small_targets = {j for j, a in enumerate(relevant) if small(a, image)}
    return dict(small_boxes=len(small_targets), detected=len(small_targets & matched),
                unmatched_predictions=len(boxes) - len(matched),
                small_box_ids=[relevant[j]["id"] for j in sorted(small_targets)],
                detected_small_box_ids=[relevant[j]["id"] for j in sorted(small_targets & matched)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=12)
    parser.add_argument("--detector", choices=["ssdlite", "mobile_frcnn"], default="ssdlite")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    subset = dev_subset(json.loads(args.instances.read_text()), args.per_class)
    if args.detector == "ssdlite":
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        net = ssdlite320_mobilenet_v3_large(weights=weights).eval()
        pixels = 320
    else:
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1
        net = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, min_size=640, max_size=640).eval()
        pixels = 640
    op = GaussianBlur(31, 2.0 * pixels / SIZE, .0125)
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        imgs = pool.map(lambda pair: cached_image(pair[0], args.cache, pixels), subset)
        for (image, annotations), clean in zip(subset, imgs, strict=True):
            generator = torch.Generator().manual_seed(image["id"])
            blurred = op.add_noise(op.A(clean[None]), generator=generator)
            lin = tikhonov(op, blurred, .01, clean[None].shape, iters=60)
            variants = {"clean": clean, "blur": blurred[0], "linear": lin[0]}
            with torch.no_grad():
                preds = net([x.clamp(0, 1) for x in variants.values()])
            for (condition, _), pred in zip(variants.items(), preds, strict=True):
                for c, label in CATEGORIES.items():
                    rows.append(dict(image_id=image["id"], condition=condition, category=label,
                                     **counts(pred, annotations, image, c, pixels)))
            print(image["id"], flush=True)
    summary = {"n_images": len(subset), "dataset": "COCO val2017 annotated development subset",
               "model": str(weights), "torchvision": torchvision.__version__, "size_px": pixels, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
