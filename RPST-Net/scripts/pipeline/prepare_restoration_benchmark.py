                     
                       

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
MASK_EXTS = IMAGE_EXTS


DEFAULT_PROJECT_ROOT = Path(r"<PROJECT_ROOT>")


ROI_TYPE_TO_SEM_VALUE = {
    "background": 0,
    "face": 10,
    "head": 20,
    "figure": 30,
    "hand": 40,
    "cloth": 50,
    "object": 60,
    "animal": 70,
    "plant": 80,
    "architecture": 90,
    "ornament": 100,
}


@dataclass
class CropTransform:
    crop_xmin: int
    crop_ymin: int
    crop_xmax: int
    crop_ymax: int
    pad_left: int
    pad_top: int
    crop_square_size: int
    output_size: int
    scale: float


def norm_str(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def parse_list_arg(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def ensure_clean_dir(path: Path, overwrite: bool = False) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_files_recursive(root: Path, exts: set[str]) -> List[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def build_image_lookup(image_root: Path) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for p in list_files_recursive(image_root, IMAGE_EXTS):
                                                                    
        lookup.setdefault(p.name, p)
    return lookup


def find_image_path(row: pd.Series, image_root: Path, image_lookup: Dict[str, Path]) -> Optional[Path]:
    rel = norm_str(row.get("relative_path", ""))
    if rel:
        p = image_root / rel.replace("/", "\\")
        if p.exists():
            return p
        p2 = image_root / rel.replace("\\", "/")
        if p2.exists():
            return p2

    filename = norm_str(row.get("filename", ""))
    if filename in image_lookup:
        return image_lookup[filename]

    return None


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_binary_mask(path: Path, auto_invert: bool = True, threshold: int = 127) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"))
    mask = (arr > threshold).astype(np.uint8) * 255

    if auto_invert:
        ratio = float((mask > 0).mean())
                                                                         
                                                      
        if ratio > 0.50:
            mask = 255 - mask

    return mask


def get_nonzero_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def square_crop_with_context(
    img: Image.Image,
    bbox: Tuple[int, int, int, int],
    output_size: int,
    context_factor: float,
    min_square_size: int,
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Image.Image, CropTransform]:
    w, h = img.size
    xmin, ymin, xmax, ymax = bbox
    bw = max(1, xmax - xmin)
    bh = max(1, ymax - ymin)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0

    square_size = int(math.ceil(max(bw, bh) * context_factor))
    square_size = max(square_size, int(min_square_size))
    square_size = min(square_size, max(w, h))

    crop_xmin = int(round(cx - square_size / 2))
    crop_ymin = int(round(cy - square_size / 2))
    crop_xmax = crop_xmin + square_size
    crop_ymax = crop_ymin + square_size

                                                                
    src_xmin = max(0, crop_xmin)
    src_ymin = max(0, crop_ymin)
    src_xmax = min(w, crop_xmax)
    src_ymax = min(h, crop_ymax)

    crop = img.crop((src_xmin, src_ymin, src_xmax, src_ymax))

    pad_left = src_xmin - crop_xmin
    pad_top = src_ymin - crop_ymin
    pad_right = crop_xmax - src_xmax
    pad_bottom = crop_ymax - src_ymax

    if any(v > 0 for v in [pad_left, pad_top, pad_right, pad_bottom]):
        padded = Image.new("RGB", (square_size, square_size), pad_color)
        padded.paste(crop, (pad_left, pad_top))
        crop = padded

    scale = output_size / float(square_size)
    crop_resized = crop.resize((output_size, output_size), Image.Resampling.LANCZOS)

    tfm = CropTransform(
        crop_xmin=crop_xmin,
        crop_ymin=crop_ymin,
        crop_xmax=crop_xmax,
        crop_ymax=crop_ymax,
        pad_left=pad_left,
        pad_top=pad_top,
        crop_square_size=square_size,
        output_size=output_size,
        scale=scale,
    )
    return crop_resized, tfm


def bbox_to_crop_coords(
    bbox: Tuple[int, int, int, int],
    tfm: CropTransform,
) -> Tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = bbox
    x1 = (xmin - tfm.crop_xmin) * tfm.scale
    y1 = (ymin - tfm.crop_ymin) * tfm.scale
    x2 = (xmax - tfm.crop_xmin) * tfm.scale
    y2 = (ymax - tfm.crop_ymin) * tfm.scale
    out = (
        int(max(0, min(tfm.output_size - 1, round(x1)))),
        int(max(0, min(tfm.output_size - 1, round(y1)))),
        int(max(1, min(tfm.output_size, round(x2)))),
        int(max(1, min(tfm.output_size, round(y2)))),
    )
    if out[2] <= out[0]:
        out = (out[0], out[1], min(tfm.output_size, out[0] + 1), out[3])
    if out[3] <= out[1]:
        out = (out[0], out[1], out[2], min(tfm.output_size, out[1] + 1))
    return out


def random_transform_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    im = Image.fromarray(mask)
    if rng.random() < 0.5:
        im = ImageOps.mirror(im)
    if rng.random() < 0.5:
        im = ImageOps.flip(im)
    angle = rng.uniform(-35, 35)
    im = im.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True, fillcolor=0)
    arr = np.array(im.convert("L"))
    arr = (arr > 127).astype(np.uint8) * 255
    return arr


def make_synthetic_irregular_mask(
    output_size: int,
    target_bbox: Tuple[int, int, int, int],
    rng: random.Random,
) -> np.ndarray:
    """Fallback irregular blob mask for debugging only."""
    mask = Image.new("L", (output_size, output_size), 0)
    draw = ImageDraw.Draw(mask)
    tx1, ty1, tx2, ty2 = target_bbox
    tw = max(8, tx2 - tx1)
    th = max(8, ty2 - ty1)
    cx = rng.randint(tx1, max(tx1, tx2 - 1))
    cy = rng.randint(ty1, max(ty1, ty2 - 1))
    radius = int(rng.uniform(0.20, 0.45) * max(tw, th))
    n = rng.randint(8, 16)
    pts = []
    for i in range(n):
        theta = 2 * math.pi * i / n
        rr = radius * rng.uniform(0.55, 1.25)
        x = cx + rr * math.cos(theta)
        y = cy + rr * math.sin(theta)
        pts.append((x, y))
    draw.polygon(pts, fill=255)

                       
    for _ in range(rng.randint(1, 3)):
        x0 = rng.randint(max(0, tx1 - tw // 3), min(output_size - 1, tx2 + tw // 3))
        y0 = rng.randint(max(0, ty1 - th // 3), min(output_size - 1, ty2 + th // 3))
        points = [(x0, y0)]
        for _j in range(rng.randint(3, 7)):
            x0 += rng.randint(-tw // 5, tw // 5)
            y0 += rng.randint(-th // 5, th // 5)
            x0 = max(0, min(output_size - 1, x0))
            y0 = max(0, min(output_size - 1, y0))
            points.append((x0, y0))
        draw.line(points, fill=255, width=max(2, output_size // 150))
    return np.array(mask)


def transfer_external_mask_to_target(
    external_mask: np.ndarray,
    output_size: int,
    target_bbox: Tuple[int, int, int, int],
    rng: random.Random,
    min_mask_ratio: float,
    max_mask_ratio: float,
    min_target_overlap: float = 0.55,
    min_target_coverage: float = 0.06,
    placement_jitter: float = 0.18,
    max_tries: int = 80,
) -> Tuple[np.ndarray, Dict[str, object]]:
    tx1, ty1, tx2, ty2 = target_bbox
    tw = max(1, tx2 - tx1)
    th = max(1, ty2 - ty1)
    target_area = max(1, tw * th)
    target_bool = np.zeros((output_size, output_size), dtype=bool)
    target_bool[ty1:ty2, tx1:tx2] = True

    bbox = get_nonzero_bbox(external_mask)
    if bbox is None:
        raise ValueError("External mask has no foreground.")
    mx1, my1, mx2, my2 = bbox
    mask_crop = external_mask[my1:my2, mx1:mx2]

    best_mask = None
    best_meta = None
    best_score = -1.0

    for attempt in range(max_tries):
        arr = random_transform_mask(mask_crop, rng)
        bbox2 = get_nonzero_bbox(arr)
        if bbox2 is None:
            continue
        ax1, ay1, ax2, ay2 = bbox2
        arr = arr[ay1:ay2, ax1:ax2]

        ah, aw = arr.shape[:2]
        if aw <= 0 or ah <= 0:
            continue

                                                                                          
        max_dim = max(aw, ah)
        target_dim = max(tw, th)
        scale = rng.uniform(0.35, 1.05) * target_dim / max(1, max_dim)
        new_w = max(4, int(round(aw * scale)))
        new_h = max(4, int(round(ah * scale)))
        new_w = min(output_size, new_w)
        new_h = min(output_size, new_h)

        resized = Image.fromarray(arr).resize((new_w, new_h), Image.Resampling.BILINEAR)
        resized_arr = (np.array(resized.convert("L")) > 127).astype(np.uint8) * 255

                                                                          
        tcx = (tx1 + tx2) / 2.0
        tcy = (ty1 + ty2) / 2.0
        jitter_x = rng.uniform(-placement_jitter, placement_jitter) * tw
        jitter_y = rng.uniform(-placement_jitter, placement_jitter) * th
        px = int(round(tcx + jitter_x - new_w / 2))
        py = int(round(tcy + jitter_y - new_h / 2))
        px = max(0, min(output_size - new_w, px))
        py = max(0, min(output_size - new_h, py))

        canvas = np.zeros((output_size, output_size), dtype=np.uint8)
        canvas[py:py + new_h, px:px + new_w] = np.maximum(
            canvas[py:py + new_h, px:px + new_w],
            resized_arr,
        )

        mask_bool = canvas > 0
        ratio = float(mask_bool.mean())
        if ratio <= 0:
            continue
        overlap = float((mask_bool & target_bool).sum()) / float(max(1, mask_bool.sum()))
        target_coverage = float((mask_bool & target_bool).sum()) / float(target_area)

                                                                           
                                                                               
                                              
        score = (2.0 * overlap) + target_coverage - abs(ratio - ((min_mask_ratio + max_mask_ratio) / 2.0))
        if score > best_score:
            best_score = score
            best_mask = canvas
            best_meta = dict(
                attempt=attempt,
                placed_x=px,
                placed_y=py,
                placed_w=new_w,
                placed_h=new_h,
                mask_ratio=ratio,
                target_overlap=overlap,
                target_coverage=target_coverage,
            )

        if (
            min_mask_ratio <= ratio <= max_mask_ratio
            and overlap >= min_target_overlap
            and target_coverage >= min_target_coverage
        ):
            return canvas, best_meta or {}

    if best_mask is None:
        raise ValueError("Failed to transfer external mask.")

                                                                               
                                                         
    if best_meta is None or float(best_meta.get("target_overlap", 0.0)) < min_target_overlap or float(best_meta.get("target_coverage", 0.0)) < min_target_coverage:
        raise ValueError(
            f"Failed strict ROI placement: best_overlap={best_meta.get('target_overlap', '') if best_meta else ''}, "
            f"best_coverage={best_meta.get('target_coverage', '') if best_meta else ''}, "
            f"required_overlap={min_target_overlap}, required_coverage={min_target_coverage}"
        )

    return best_mask, best_meta or {}


def apply_damage_fill(clean: Image.Image, mask: np.ndarray, fill_mode: str, rng: random.Random) -> Image.Image:
    arr = np.array(clean).copy()
    m = mask > 0
    if not np.any(m):
        return clean.copy()

    if fill_mode == "black":
        fill = np.array([0, 0, 0], dtype=np.uint8)
    elif fill_mode == "white":
        fill = np.array([255, 255, 255], dtype=np.uint8)
    elif fill_mode == "gray":
        fill = np.array([128, 128, 128], dtype=np.uint8)
    elif fill_mode == "noise":
        noise = rng.randint(40, 220)
        fill = np.array([noise, noise, noise], dtype=np.uint8)
    elif fill_mode == "mean":
        bg = arr[~m]
        if bg.size == 0:
            fill = np.array([128, 128, 128], dtype=np.uint8)
        else:
            fill = np.mean(bg.reshape(-1, 3), axis=0).astype(np.uint8)
    else:
        raise ValueError(f"Unknown fill_mode: {fill_mode}")

    arr[m] = fill
    return Image.fromarray(arr)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    try:
        import cv2
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)
    except Exception:
        im = Image.fromarray(mask)
                                            
        from PIL import ImageFilter
        return np.array(im.filter(ImageFilter.MaxFilter(size=2 * radius + 1)))


def extract_visible_edge(damaged: Image.Image, mask: np.ndarray, edge_method: str, mask_dilate_radius: int) -> Image.Image:
    arr = np.array(damaged.convert("RGB"))
    try:
        import cv2
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        if edge_method == "canny":
            edges = cv2.Canny(gray, 80, 160)
        elif edge_method == "sobel":
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            edges = (255 * (mag / (mag.max() + 1e-6))).astype(np.uint8)
        else:
            edges = cv2.Canny(gray, 80, 160)
    except Exception:
        from PIL import ImageFilter
        gray_im = damaged.convert("L")
        edges = np.array(gray_im.filter(ImageFilter.FIND_EDGES))

    dm = dilate_mask(mask, mask_dilate_radius) > 0
    edges[dm] = 0
    return Image.fromarray(edges).convert("L")


def crop_query_roi(damaged: Image.Image, mask: np.ndarray, query_size: int, expand_factor: float) -> Tuple[Image.Image, Dict[str, int]]:
    bbox = get_nonzero_bbox(mask)
    if bbox is None:
        return damaged.resize((query_size, query_size), Image.Resampling.LANCZOS), {
            "query_xmin": 0,
            "query_ymin": 0,
            "query_xmax": damaged.size[0],
            "query_ymax": damaged.size[1],
        }

    x1, y1, x2, y2 = bbox
    w = damaged.size[0]
    h = damaged.size[1]
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    side = int(math.ceil(max(bw, bh) * expand_factor))
    side = max(side, min(w, h) // 4)
    side = min(side, max(w, h))

    qx1 = int(round(cx - side / 2))
    qy1 = int(round(cy - side / 2))
    qx2 = qx1 + side
    qy2 = qy1 + side

    src_x1 = max(0, qx1)
    src_y1 = max(0, qy1)
    src_x2 = min(w, qx2)
    src_y2 = min(h, qy2)
    crop = damaged.crop((src_x1, src_y1, src_x2, src_y2))

    pad_left = src_x1 - qx1
    pad_top = src_y1 - qy1
    pad_right = qx2 - src_x2
    pad_bottom = qy2 - src_y2
    if any(v > 0 for v in [pad_left, pad_top, pad_right, pad_bottom]):
        padded = Image.new("RGB", (side, side), (0, 0, 0))
        padded.paste(crop, (pad_left, pad_top))
        crop = padded

    crop = crop.resize((query_size, query_size), Image.Resampling.LANCZOS)
    return crop, {"query_xmin": qx1, "query_ymin": qy1, "query_xmax": qx2, "query_ymax": qy2}


def save_preview_sheet(rows: List[Dict[str, str]], output_path: Path, max_rows: int = 24, thumb_w: int = 180) -> None:
    if not rows:
        return

    chosen = rows[:max_rows]
    cols = ["clean_path", "damaged_path", "mask_path", "visible_edge_path", "query_roi_path"]
    col_titles = ["clean", "damaged", "mask", "visible_edge", "query_roi"]
    title_h = 26
    label_h = 40
    thumb_h = thumb_w

    canvas = Image.new("RGB", (thumb_w * len(cols), (thumb_h + label_h) * len(chosen) + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    for c, t in enumerate(col_titles):
        draw.text((c * thumb_w + 5, 5), t, fill=(0, 0, 0))

    for r_idx, row in enumerate(chosen):
        y0 = title_h + r_idx * (thumb_h + label_h)
        for c_idx, col in enumerate(cols):
            p = Path(row[col])
            if not p.exists():
                im = Image.new("RGB", (thumb_w, thumb_h), (220, 220, 220))
            else:
                im = Image.open(p).convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x0 = c_idx * thumb_w
            canvas.paste(im, (x0, y0))
        label = f"{row.get('sample_id','')} | {row.get('roi_type','')} | {row.get('final_style_group','')} | {row.get('split','')}"
        draw.text((5, y0 + thumb_h + 4), label[:120], fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def read_annotations(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input annotations CSV not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

                             
    for col in ["xmin", "ymin", "xmax", "ymax", "image_width", "image_height"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def filter_targets(
    df: pd.DataFrame,
    target_roi_types: List[str],
    target_quality_types: List[str],
    clean_candidate_values: List[str],
    include_splits: List[str],
    exclude_style_uk: bool,
) -> pd.DataFrame:
    d = df.copy()

    if target_roi_types:
        d = d[d["roi_type"].astype(str).str.lower().isin([x.lower() for x in target_roi_types])]
    if target_quality_types:
        d = d[d["quality_type"].astype(str).str.lower().isin([x.lower() for x in target_quality_types])]
    if clean_candidate_values and "is_clean_candidate" in d.columns:
        d = d[d["is_clean_candidate"].astype(str).str.lower().isin([x.lower() for x in clean_candidate_values])]
    if include_splits:
        d = d[d["split"].astype(str).str.lower().isin([x.lower() for x in include_splits])]

    if exclude_style_uk and "final_style_group" in d.columns:
        d = d[d["final_style_group"].astype(str).str.upper() != "S_UK"]

                                        
    d = d.dropna(subset=["xmin", "ymin", "xmax", "ymax"])
    d = d[(d["xmax"] > d["xmin"]) & (d["ymax"] > d["ymin"])]

    return d.reset_index(drop=True)


def build_output_dirs(output_dir: Path, splits: Iterable[str], overwrite: bool) -> Dict[str, Path]:
    subdirs = ["clean", "damaged", "masks", "visible_edges", "query_roi", "semantic_part_masks"]
    paths: Dict[str, Path] = {}
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for sd in subdirs:
        for sp in splits:
            p = output_dir / sd / sp
            p.mkdir(parents=True, exist_ok=True)
        paths[sd] = output_dir / sd
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (output_dir / "preview").mkdir(parents=True, exist_ok=True)
    return paths


def save_semantic_part_mask(output_size: int, target_bbox: Tuple[int, int, int, int], roi_type: str, out_path: Path) -> None:
    sem = np.zeros((output_size, output_size), dtype=np.uint8)
    x1, y1, x2, y2 = target_bbox
    value = ROI_TYPE_TO_SEM_VALUE.get(roi_type, 255)
    sem[y1:y2, x1:x2] = value
    Image.fromarray(sem).save(out_path)



def roi_specific_mask_thresholds(
    roi_type: str,
    base_overlap: float,
    base_coverage: float,
    base_jitter: float,
) -> Tuple[float, float, float]:
    """Return stricter placement thresholds for small semantic targets.

    overlap = fraction of mask pixels that fall inside the target bbox.
    coverage = fraction of target bbox covered by mask pixels.

    Face/head/hand benchmarks are sensitive to off-target masks, so they use
    tighter overlap and lower jitter. Large regions such as figure/cloth are
    allowed more flexibility.
    """
    r = (roi_type or "").lower()
    if r == "face":
        return max(base_overlap, 0.68), max(base_coverage, 0.10), min(base_jitter, 0.12)
    if r == "head":
        return max(base_overlap, 0.62), max(base_coverage, 0.08), min(base_jitter, 0.14)
    if r == "hand":
        return max(base_overlap, 0.58), max(base_coverage, 0.07), min(base_jitter, 0.14)
    if r == "cloth":
        return max(min(base_overlap, 0.48), 0.45), max(min(base_coverage, 0.06), 0.04), min(base_jitter, 0.20)
    if r == "figure":
        return max(min(base_overlap, 0.40), 0.35), max(min(base_coverage, 0.05), 0.03), min(base_jitter, 0.22)
    return base_overlap, base_coverage, base_jitter

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--annotations_csv", type=str, default="")
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--external_mask_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--output_size", type=int, default=512)
    parser.add_argument("--query_size", type=int, default=256)
    parser.add_argument("--context_factor", type=float, default=2.8)
    parser.add_argument("--min_crop_size", type=int, default=256)
    parser.add_argument("--samples_per_annotation", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--target_roi_types", type=str, default="face,head,hand,cloth,figure")
    parser.add_argument("--target_quality_types", type=str, default="normal")
    parser.add_argument("--clean_candidate_values", type=str, default="yes,roi_only")
    parser.add_argument("--include_splits", type=str, default="train,val,test")
    parser.add_argument("--exclude_style_uk", action="store_true")
    parser.add_argument("--min_mask_ratio", type=float, default=0.025)
    parser.add_argument("--max_mask_ratio", type=float, default=0.35)
    parser.add_argument("--min_target_overlap", type=float, default=0.55)
    parser.add_argument("--min_target_coverage", type=float, default=0.06)
    parser.add_argument("--placement_jitter", type=float, default=0.18)
    parser.add_argument("--damage_fill", type=str, default="mean", choices=["mean", "black", "white", "gray", "noise"])
    parser.add_argument("--edge_method", type=str, default="canny", choices=["canny", "sobel"])
    parser.add_argument("--mask_dilate_radius", type=int, default=5)
    parser.add_argument("--query_expand_factor", type=float, default=2.2)
    parser.add_argument("--allow_synthetic_fallback", action="store_true")
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root)
    annotations_csv = Path(args.annotations_csv) if args.annotations_csv else project_root / "splits" / "annotations_master_with_split.csv"
    image_root = Path(args.image_root) if args.image_root else project_root / "curated_originals"
    external_mask_dir = Path(args.external_mask_dir) if args.external_mask_dir else project_root / "external_masks" / "dunhuang_damage_masks"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "restoration_benchmark"

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    target_roi_types = parse_list_arg(args.target_roi_types)
    target_quality_types = parse_list_arg(args.target_quality_types)
    clean_candidate_values = parse_list_arg(args.clean_candidate_values)
    include_splits = parse_list_arg(args.include_splits)

    print(f"[INFO] Project root: {project_root}")
    print(f"[INFO] Annotations CSV: {annotations_csv}")
    print(f"[INFO] Image root: {image_root}")
    print(f"[INFO] External mask dir: {external_mask_dir}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Target ROI types: {target_roi_types}")
    print(f"[INFO] Target quality types: {target_quality_types}")

    df = read_annotations(annotations_csv)
    targets = filter_targets(
        df,
        target_roi_types=target_roi_types,
        target_quality_types=target_quality_types,
        clean_candidate_values=clean_candidate_values,
        include_splits=include_splits,
        exclude_style_uk=args.exclude_style_uk,
    )

    if args.max_samples and args.max_samples > 0:
                                                              
        targets = targets.sample(n=min(args.max_samples, len(targets)), random_state=args.seed).reset_index(drop=True)

    print(f"[INFO] Candidate target annotations: {len(targets)}")
    if len(targets) == 0:
        raise RuntimeError("No target annotations after filtering.")

    external_masks = list_files_recursive(external_mask_dir, MASK_EXTS)
    if not external_masks and not args.allow_synthetic_fallback:
        raise FileNotFoundError(
            f"No external mask images found in: {external_mask_dir}\n"
            f"Put binary damage masks there, or run with --allow_synthetic_fallback for debugging."
        )
    print(f"[INFO] External masks found: {len(external_masks)}")

    splits = sorted(set(norm_str(x) for x in targets["split"].tolist() if norm_str(x)))
    build_output_dirs(output_dir, splits, overwrite=args.overwrite)

    image_lookup = build_image_lookup(image_root)
    print(f"[INFO] Images indexed by filename: {len(image_lookup)}")

    records: List[Dict[str, object]] = []
    issues: List[Dict[str, object]] = []
    preview_records: List[Dict[str, str]] = []

    sample_counter = 0
    total_iterations = len(targets) * max(1, args.samples_per_annotation)

    pbar = tqdm(total=total_iterations, desc="Preparing inpaint benchmark")
    for idx, row in targets.iterrows():
        image_path = find_image_path(row, image_root, image_lookup)
        if image_path is None or not image_path.exists():
            issues.append({
                "row_index": idx,
                "image_id": norm_str(row.get("image_id", "")),
                "filename": norm_str(row.get("filename", "")),
                "issue": "image_not_found",
            })
            pbar.update(max(1, args.samples_per_annotation))
            continue

        try:
            img = load_rgb(image_path)
        except Exception as e:
            issues.append({
                "row_index": idx,
                "image_id": norm_str(row.get("image_id", "")),
                "filename": norm_str(row.get("filename", "")),
                "issue": f"image_load_error: {e}",
            })
            pbar.update(max(1, args.samples_per_annotation))
            continue

        bbox_orig = (
            int(row["xmin"]),
            int(row["ymin"]),
            int(row["xmax"]),
            int(row["ymax"]),
        )

                                                                                                   
        try:
            clean_crop, tfm = square_crop_with_context(
                img,
                bbox=bbox_orig,
                output_size=args.output_size,
                context_factor=args.context_factor,
                min_square_size=args.min_crop_size,
                pad_color=(0, 0, 0),
            )
        except Exception as e:
            issues.append({
                "row_index": idx,
                "image_id": norm_str(row.get("image_id", "")),
                "filename": norm_str(row.get("filename", "")),
                "label": norm_str(row.get("label", "")),
                "issue": f"crop_error: {e}",
            })
            pbar.update(max(1, args.samples_per_annotation))
            continue

        target_bbox_crop = bbox_to_crop_coords(bbox_orig, tfm)
        split = norm_str(row.get("split", "train")) or "train"
        roi_type = norm_str(row.get("roi_type", "unknown")) or "unknown"
        style_group = norm_str(row.get("final_style_group", "unknown")) or "unknown"

        for sidx in range(args.samples_per_annotation):
            sample_counter += 1
            sample_id = f"{split}_{sample_counter:07d}_{roi_type}"

            try:
                if external_masks:
                    mask_path = rng.choice(external_masks)
                    ext_mask = load_binary_mask(mask_path, auto_invert=True)
                    roi_min_overlap, roi_min_coverage, roi_jitter = roi_specific_mask_thresholds(
                        roi_type,
                        base_overlap=args.min_target_overlap,
                        base_coverage=args.min_target_coverage,
                        base_jitter=args.placement_jitter,
                    )
                    mask_arr, mask_meta = transfer_external_mask_to_target(
                        ext_mask,
                        output_size=args.output_size,
                        target_bbox=target_bbox_crop,
                        rng=rng,
                        min_mask_ratio=args.min_mask_ratio,
                        max_mask_ratio=args.max_mask_ratio,
                        min_target_overlap=roi_min_overlap,
                        min_target_coverage=roi_min_coverage,
                        placement_jitter=roi_jitter,
                    )
                    mask_meta["required_target_overlap"] = roi_min_overlap
                    mask_meta["required_target_coverage"] = roi_min_coverage
                    mask_meta["placement_jitter"] = roi_jitter
                    mask_origin = "external"
                    mask_source_path = str(mask_path)
                else:
                    mask_arr = make_synthetic_irregular_mask(args.output_size, target_bbox_crop, rng)
                    mask_meta = {
                        "mask_ratio": float((mask_arr > 0).mean()),
                        "target_overlap": "",
                        "target_coverage": "",
                    }
                    mask_origin = "synthetic_fallback"
                    mask_source_path = ""
            except Exception as e:
                issues.append({
                    "row_index": idx,
                    "image_id": norm_str(row.get("image_id", "")),
                    "filename": norm_str(row.get("filename", "")),
                    "label": norm_str(row.get("label", "")),
                    "issue": f"mask_transfer_error: {e}",
                })
                pbar.update(1)
                continue

            mask_bin = (mask_arr > 0).astype(np.uint8) * 255
            damaged = apply_damage_fill(clean_crop, mask_bin, args.damage_fill, rng)
            visible_edge = extract_visible_edge(
                damaged=damaged,
                mask=mask_bin,
                edge_method=args.edge_method,
                mask_dilate_radius=args.mask_dilate_radius,
            )
            query_roi, query_meta = crop_query_roi(
                damaged=damaged,
                mask=mask_bin,
                query_size=args.query_size,
                expand_factor=args.query_expand_factor,
            )

            clean_out = output_dir / "clean" / split / f"{sample_id}_clean.png"
            damaged_out = output_dir / "damaged" / split / f"{sample_id}_damaged.png"
            mask_out = output_dir / "masks" / split / f"{sample_id}_mask.png"
            edge_out = output_dir / "visible_edges" / split / f"{sample_id}_visible_edge.png"
            query_out = output_dir / "query_roi" / split / f"{sample_id}_query_roi.png"
            sem_out = output_dir / "semantic_part_masks" / split / f"{sample_id}_semantic_part.png"

            clean_crop.save(clean_out)
            damaged.save(damaged_out)
            Image.fromarray(mask_bin).save(mask_out)
            visible_edge.save(edge_out)
            query_roi.save(query_out)
            save_semantic_part_mask(args.output_size, target_bbox_crop, roi_type, sem_out)

            mask_bbox = get_nonzero_bbox(mask_bin)
            if mask_bbox is None:
                mask_bbox = (0, 0, 0, 0)

            rec: Dict[str, object] = {
                "sample_id": sample_id,
                "split": split,
                "source_annotation_index": idx,
                "image_id": norm_str(row.get("image_id", "")),
                "filename": norm_str(row.get("filename", "")),
                "relative_path": norm_str(row.get("relative_path", "")),
                "image_path": str(image_path),
                "tomb_id": norm_str(row.get("tomb_id", "")),
                "tomb_name": norm_str(row.get("tomb_name", row.get("tomb_name_auto", ""))),
                "final_style_group": style_group,
                "scene_type": norm_str(row.get("scene_type", "")),
                "label": norm_str(row.get("label", "")),
                "roi_type": roi_type,
                "quality_type": norm_str(row.get("quality_type", "")),
                "crop_role": norm_str(row.get("crop_role", "")),
                "gender": norm_str(row.get("gender", "")),
                "view": norm_str(row.get("view", "")),
                "identity": norm_str(row.get("identity", "")),
                "feature": norm_str(row.get("feature", "")),
                "clean_path": str(clean_out),
                "damaged_path": str(damaged_out),
                "mask_path": str(mask_out),
                "visible_edge_path": str(edge_out),
                "query_roi_path": str(query_out),
                "semantic_part_mask_path": str(sem_out),
                "output_size": args.output_size,
                "query_size": args.query_size,
                "source_xmin": bbox_orig[0],
                "source_ymin": bbox_orig[1],
                "source_xmax": bbox_orig[2],
                "source_ymax": bbox_orig[3],
                "crop_xmin": tfm.crop_xmin,
                "crop_ymin": tfm.crop_ymin,
                "crop_xmax": tfm.crop_xmax,
                "crop_ymax": tfm.crop_ymax,
                "crop_square_size": tfm.crop_square_size,
                "crop_scale": tfm.scale,
                "target_xmin": target_bbox_crop[0],
                "target_ymin": target_bbox_crop[1],
                "target_xmax": target_bbox_crop[2],
                "target_ymax": target_bbox_crop[3],
                "mask_xmin": mask_bbox[0],
                "mask_ymin": mask_bbox[1],
                "mask_xmax": mask_bbox[2],
                "mask_ymax": mask_bbox[3],
                "mask_area": int((mask_bin > 0).sum()),
                "mask_ratio": float((mask_bin > 0).mean()),
                "mask_origin": mask_origin,
                "mask_source_path": mask_source_path,
                "mask_transform_meta": json.dumps(mask_meta, ensure_ascii=False),
                "mask_target_overlap": mask_meta.get("target_overlap", ""),
                "mask_target_coverage": mask_meta.get("target_coverage", ""),
                "damage_fill": args.damage_fill,
                "edge_method": args.edge_method,
                "mask_dilate_radius": args.mask_dilate_radius,
                **query_meta,
            }
            records.append(rec)
            if len(preview_records) < 36:
                preview_records.append({k: str(v) for k, v in rec.items()})
            pbar.update(1)

    pbar.close()

    metadata_dir = output_dir / "metadata"
    master_csv = metadata_dir / "inpaint_benchmark_master.csv"
    pd.DataFrame(records).to_csv(master_csv, index=False, encoding="utf-8-sig")

    if records:
        rec_df = pd.DataFrame(records)
        for sp, g in rec_df.groupby("split"):
            g.to_csv(metadata_dir / f"{sp}.csv", index=False, encoding="utf-8-sig")

                                                                       
        query_cols = [
            "sample_id", "split", "image_id", "tomb_id", "final_style_group",
            "roi_type", "gender", "view", "identity", "feature",
            "damaged_path", "mask_path", "query_roi_path", "visible_edge_path",
            "semantic_part_mask_path",
        ]
        rec_df[[c for c in query_cols if c in rec_df.columns]].to_csv(
            metadata_dir / "retrieval_queries.csv",
            index=False,
            encoding="utf-8-sig",
        )

        summary = {
            "num_samples": int(len(rec_df)),
            "num_source_annotations": int(rec_df["source_annotation_index"].nunique()),
            "output_size": args.output_size,
            "query_size": args.query_size,
            "target_roi_types": target_roi_types,
            "target_quality_types": target_quality_types,
            "clean_candidate_values": clean_candidate_values,
            "include_splits": include_splits,
            "external_mask_dir": str(external_mask_dir),
            "external_mask_count": len(external_masks),
            "count_by_split": rec_df["split"].value_counts().to_dict(),
            "count_by_roi_type": rec_df["roi_type"].value_counts().to_dict(),
            "count_by_style": rec_df["final_style_group"].value_counts().to_dict(),
            "mask_ratio_mean": float(rec_df["mask_ratio"].mean()),
            "mask_ratio_min": float(rec_df["mask_ratio"].min()),
            "mask_ratio_max": float(rec_df["mask_ratio"].max()),
            "num_issues": len(issues),
        }
    else:
        summary = {
            "num_samples": 0,
            "num_issues": len(issues),
        }

    with open(metadata_dir / "inpaint_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame(issues).to_csv(metadata_dir / "inpaint_benchmark_issues.csv", index=False, encoding="utf-8-sig")

    save_preview_sheet(
        preview_records,
        output_path=output_dir / "preview" / "inpaint_benchmark_preview_sheet.jpg",
        max_rows=36,
    )

    print("\n[DONE] 07 inpainting benchmark preparation finished.")
    print(f"[DONE] Samples: {summary.get('num_samples', 0)}")
    print(f"[DONE] Master CSV: {master_csv}")
    print(f"[DONE] Retrieval queries: {metadata_dir / 'retrieval_queries.csv'}")
    print(f"[DONE] Preview: {output_dir / 'preview' / 'inpaint_benchmark_preview_sheet.jpg'}")
    print(f"[DONE] Issues: {len(issues)}")


if __name__ == "__main__":
    main()
