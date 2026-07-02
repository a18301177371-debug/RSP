                     
                       
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    import cv2                
except Exception:                    
    cv2 = None

DEFAULT_PROJECT_ROOT = Path(r"<PROJECT_ROOT>")
STYLE_ORDER = ["S0", "S1", "S2", "S3"]
STYLE_ALIAS = {
    "s uk": "S_UK", "s_uk": "S_UK", "suk": "S_UK", "unknown": "S_UK", "": "S_UK",
    "s0": "S0", "s1": "S1", "s2": "S2", "s3": "S3",
}

STRICT_ROIS = {"face", "head", "hand"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def norm_text(x: object) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def safe_name(x: object, default: str = "sample") -> str:
    s = norm_text(x) or default
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or default


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_clear_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        try:
            shutil.rmtree(path)
        except OSError:
            backup = path.with_name(path.name + "_old")
            if backup.exists():
                shutil.rmtree(backup, ignore_errors=True)
            path.rename(backup)
    path.mkdir(parents=True, exist_ok=True)


def norm_style(x: object) -> str:
    s = norm_text(x)
    key = s.lower().replace("-", "_").strip()
    return STYLE_ALIAS.get(key, s if s else "S_UK")


def style_compat(a: object, b: object) -> float:
    sa, sb = norm_style(a), norm_style(b)
    if sa == sb:
        return 1.0
    if "S_UK" in {sa, sb}:
        return 0.45
    if {sa, sb} in [{"S1", "S2"}, {"S2", "S3"}]:
        return 0.82
    if sa in STYLE_ORDER and sb in STYLE_ORDER:
        d = abs(STYLE_ORDER.index(sa) - STYLE_ORDER.index(sb))
        return {1: 0.65, 2: 0.35, 3: 0.18}.get(d, 0.15)
    return 0.25


def norm_gender(x: object) -> str:
    s = norm_text(x).lower()
    if s in {"male", "man"}:
        return "man"
    if s in {"female", "woman", "w", "f"}:
        return "woman"
    return "unknown"


def gender_compat(query_gender: object, ref_gender: object, roi_type: str) -> float:
    q = norm_gender(query_gender)
    r = norm_gender(ref_gender)
    if q == "unknown" or r == "unknown":
        return 0.85 if roi_type in STRICT_ROIS else 0.95
    if q == r:
        return 1.0
    return 0.35 if roi_type == "face" else (0.55 if roi_type == "head" else 0.75)


def load_gray(path: Path, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("L")
        if size is not None:
            im = im.resize(size, Image.Resampling.BILINEAR)
        return im


def load_rgb(path: Path, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size is not None:
            im = im.resize(size, Image.Resampling.LANCZOS)
        return im


def compute_edge(gray: Image.Image, method: str = "canny", low: int = 60, high: int = 160, auto_canny: bool = True) -> Image.Image:
    arr = np.asarray(gray, dtype=np.uint8)
    if method == "canny" and cv2 is not None:
        if auto_canny:
            med = float(np.median(arr))
            lo = int(max(10, 0.66 * med))
            hi = int(min(255, 1.33 * med))
            if hi <= lo:
                lo, hi = low, high
        else:
            lo, hi = int(low), int(high)
        edge = cv2.Canny(arr, lo, hi)
        edge = cv2.dilate(edge, np.ones((2, 2), np.uint8), iterations=1)
        return Image.fromarray(edge, mode="L")
    edge = gray.filter(ImageFilter.FIND_EDGES)
    edge = ImageOps.autocontrast(edge)
    return edge


def mask_bbox(mask: Image.Image) -> Tuple[int, int, int, int]:
    arr = np.asarray(mask.convert("L"))
    ys, xs = np.where(arr > 127)
    if len(xs) == 0:
        return (0, 0, mask.width, mask.height)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def dilate_mask(mask: Image.Image, radius: int) -> Image.Image:
    arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    bin_arr = (arr > 127).astype(np.uint8) * 255
    if radius <= 0:
        return Image.fromarray(bin_arr, mode="L")
    if cv2 is not None:
        k = max(1, int(radius) * 2 + 1)
        out = cv2.dilate(bin_arr, np.ones((k, k), np.uint8), iterations=1)
        return Image.fromarray(out, mode="L")
    return Image.fromarray(bin_arr, mode="L").filter(ImageFilter.MaxFilter(max(3, radius * 2 + 1)))


def softmax_weights(scores: Sequence[float], tau: float) -> np.ndarray:
    arr = np.asarray([float(s) for s in scores], dtype=np.float64)
    if len(arr) == 0:
        return arr
    if (not np.isfinite(arr).all()) or np.allclose(arr.max(), arr.min()):
        return np.ones(len(arr), dtype=np.float64) / len(arr)
    z = (arr - arr.max()) / max(float(tau), 1e-6)
    z = np.clip(z, -50, 50)
    e = np.exp(z)
    return e / max(e.sum(), 1e-12)


def entropy_norm(weights: np.ndarray) -> float:
    if len(weights) <= 1:
        return 0.0
    w = np.clip(weights.astype(np.float64), 1e-12, 1.0)
    h = -float(np.sum(w * np.log(w)))
    return h / math.log(len(w))


def nonzero_bbox(edge: Image.Image, thr: int = 16) -> Optional[Tuple[int, int, int, int]]:
    arr = np.asarray(edge.convert("L"), dtype=np.uint8)
    ys, xs = np.where(arr > thr)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def bbox_metrics(box: Tuple[int, int, int, int], canvas_w: int, canvas_h: int) -> Dict[str, float]:
    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0 / max(1.0, canvas_w)
    cy = (y1 + y2) / 2.0 / max(1.0, canvas_h)
    return {
        "w_frac": bw / max(1.0, canvas_w),
        "h_frac": bh / max(1.0, canvas_h),
        "area_frac": (bw * bh) / max(1.0, canvas_w * canvas_h),
        "aspect": bw / max(1.0, bh),
        "cx": cx,
        "cy": cy,
        "bw": bw,
        "bh": bh,
    }


def geometry_compat(ref_box: Tuple[int, int, int, int], tgt_box: Tuple[int, int, int, int], canvas_size: int, roi_type: str) -> Tuple[float, Dict[str, float]]:
    rm = bbox_metrics(ref_box, canvas_size, canvas_size)
    tm = bbox_metrics(tgt_box, canvas_size, canvas_size)
    scale_w = math.exp(-abs(math.log(max(rm["w_frac"], 1e-6) / max(tm["w_frac"], 1e-6))))
    scale_h = math.exp(-abs(math.log(max(rm["h_frac"], 1e-6) / max(tm["h_frac"], 1e-6))))
    aspect = math.exp(-abs(math.log(max(rm["aspect"], 1e-6) / max(tm["aspect"], 1e-6))))
    fill = math.exp(-abs(math.log(max(rm["area_frac"], 1e-6) / max(tm["area_frac"], 1e-6))))
    cdist = math.sqrt((rm["cx"] - tm["cx"]) ** 2 + (rm["cy"] - tm["cy"]) ** 2)
    center = math.exp(-cdist * (9.0 if roi_type in STRICT_ROIS else 5.0))
    if roi_type == "face":
        score = 0.30 * aspect + 0.25 * scale_w + 0.20 * scale_h + 0.15 * center + 0.10 * fill
    elif roi_type == "head":
        score = 0.25 * aspect + 0.25 * scale_w + 0.20 * scale_h + 0.15 * center + 0.15 * fill
    else:
        score = 0.22 * aspect + 0.22 * scale_w + 0.18 * scale_h + 0.18 * center + 0.20 * fill
    diag = {
        "geom_scale_w": float(scale_w),
        "geom_scale_h": float(scale_h),
        "geom_aspect": float(aspect),
        "geom_center": float(center),
        "geom_fill": float(fill),
    }
    return float(np.clip(score, 0.0, 1.0)), diag


def choose_target_box(row: pd.Series, mask_img: Image.Image, output_size: int, roi_type: str) -> Tuple[int, int, int, int]:
    tx1 = row.get("target_xmin", np.nan); ty1 = row.get("target_ymin", np.nan)
    tx2 = row.get("target_xmax", np.nan); ty2 = row.get("target_ymax", np.nan)
    if pd.isna(tx1) or pd.isna(ty1) or pd.isna(tx2) or pd.isna(ty2):
        tx1, ty1, tx2, ty2 = mask_bbox(mask_img)
    target = (int(tx1), int(ty1), int(tx2), int(ty2))
                                                                                                               
    mx1, my1, mx2, my2 = mask_bbox(mask_img)
    if roi_type in STRICT_ROIS:
        x1 = int(round(0.65 * target[0] + 0.35 * mx1))
        y1 = int(round(0.65 * target[1] + 0.35 * my1))
        x2 = int(round(0.65 * target[2] + 0.35 * mx2))
        y2 = int(round(0.65 * target[3] + 0.35 * my2))
        x1 = max(0, min(x1, output_size - 1)); y1 = max(0, min(y1, output_size - 1))
        x2 = max(x1 + 1, min(x2, output_size)); y2 = max(y1 + 1, min(y2, output_size))
        return (x1, y1, x2, y2)
    return target


def align_reference_edge(edge: Image.Image, ref_box: Tuple[int, int, int, int], target_box: Tuple[int, int, int, int], output_size: int, roi_type: str) -> Image.Image:
    x1, y1, x2, y2 = ref_box
    crop = edge.crop((x1, y1, x2, y2))
    tw = max(1, int(target_box[2] - target_box[0]))
    th = max(1, int(target_box[3] - target_box[1]))
                                                                                   
    scale = 0.98 if roi_type == "face" else (1.00 if roi_type == "head" else 1.03)
    rw = max(1, int(round(tw * scale)))
    rh = max(1, int(round(th * scale)))
    crop = crop.resize((rw, rh), Image.Resampling.BILINEAR)
    cx = (target_box[0] + target_box[2]) / 2.0
    cy = (target_box[1] + target_box[3]) / 2.0
    px1 = int(round(cx - rw / 2.0))
    py1 = int(round(cy - rh / 2.0))
    px2 = min(output_size, px1 + rw)
    py2 = min(output_size, py1 + rh)
    px1 = max(0, px1); py1 = max(0, py1)
    canvas = Image.new("L", (output_size, output_size), 0)
    crop = crop.crop((0, 0, max(1, px2 - px1), max(1, py2 - py1)))
    canvas.paste(crop, (px1, py1))
    return canvas


def combine_structure_condition(visible_edge: Image.Image, ref_prior_masked: Image.Image, confidence: float, min_ref_gain: float) -> Image.Image:
    v = np.asarray(visible_edge.convert("L"), dtype=np.float32)
    r = np.asarray(ref_prior_masked.convert("L"), dtype=np.float32)
                                                                                       
    r = (r / 255.0) ** 0.9 * 255.0 * max(min_ref_gain, float(confidence))
    out = np.maximum(v, r)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="L")


def masked_prior(prior: Image.Image, mask: Image.Image, dilate_radius: int) -> Image.Image:
    m = dilate_mask(mask, dilate_radius)
    p = np.asarray(prior.convert("L"), dtype=np.uint8)
    ma = (np.asarray(m, dtype=np.uint8) > 127).astype(np.uint8)
    out = (p * ma).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def make_overlay(damaged_rgb: Image.Image, mask_img: Image.Image, ref_prior_masked: Image.Image, structure_img: Image.Image) -> Image.Image:
    base = damaged_rgb.convert("RGB")
    b = np.asarray(base, dtype=np.uint8).copy()
    m = np.asarray(mask_img.convert("L"), dtype=np.uint8)
    r = np.asarray(ref_prior_masked.convert("L"), dtype=np.uint8)
    s = np.asarray(structure_img.convert("L"), dtype=np.uint8)
                  
    red = m > 127
    b[red, 0] = np.clip(0.60 * b[red, 0] + 0.40 * 255, 0, 255)
    b[red, 1] = np.clip(0.60 * b[red, 1], 0, 255)
    b[red, 2] = np.clip(0.60 * b[red, 2], 0, 255)
                               
    green = r > 32
    b[green, 1] = np.clip(0.45 * b[green, 1] + 0.55 * 255, 0, 255)
    b[green, 0] = np.clip(0.70 * b[green, 0], 0, 255)
                                      
    cyan = s > 80
    b[cyan, 1] = np.clip(0.55 * b[cyan, 1] + 0.45 * 255, 0, 255)
    b[cyan, 2] = np.clip(0.55 * b[cyan, 2] + 0.45 * 255, 0, 255)
    return Image.fromarray(b.astype(np.uint8), mode="RGB")


def draw_text_panel(img: Image.Image, lines: List[str], width: int = 256, height: int = 256) -> Image.Image:
    panel = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    y = 8
    for line in lines:
        draw.text((8, y), line[:42], fill=(0, 0, 0), font=font)
        y += 14
    return panel


def make_preview_sheet(rows: List[Dict[str, object]], out_path: Path, max_rows: int = 12) -> None:
    if not rows:
        return
    tile = 192
    cols = 7
    rows = rows[:max_rows]
    sheet = Image.new("RGB", (cols * tile, len(rows) * tile), (245, 245, 245))
    for r_idx, rec in enumerate(rows):
        paths = [
            rec.get("clean_path", ""),
            rec.get("damaged_path", ""),
            rec.get("mask_path", ""),
            rec.get("visible_edge_path", ""),
            rec.get("reference_structural_prior_masked_path", ""),
            rec.get("structure_condition_path", ""),
            rec.get("overlay_path", ""),
        ]
        titles = ["clean", "damaged", "mask", "visible", "ref_prior", "structure", "overlay"]
        for c_idx, (ptxt, title) in enumerate(zip(paths, titles)):
            p = Path(str(ptxt)) if ptxt else None
            try:
                if p is not None and p.exists():
                    im = Image.open(p).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
                else:
                    im = draw_text_panel(Image.new("RGB", (tile, tile), "white"), [title, "missing"], tile, tile)
            except Exception:
                im = draw_text_panel(Image.new("RGB", (tile, tile), "white"), [title, "error"], tile, tile)
                           
            bar = Image.new("RGB", (tile, 18), (255, 255, 255))
            d = ImageDraw.Draw(bar)
            d.text((4, 2), title, fill=(0, 0, 0), font=ImageFont.load_default())
            combo = Image.new("RGB", (tile, tile), (255, 255, 255))
            combo.paste(im.crop((0, 18, tile, tile)) if im.height >= tile else im, (0, 18))
            combo.paste(bar, (0, 0))
            sheet.paste(combo.resize((tile, tile)), (c_idx * tile, r_idx * tile))
    ensure_dir(out_path.parent)
    sheet.save(out_path, quality=92)


def collect_refs(row: pd.Series, top_k: int) -> List[Dict[str, object]]:
    refs: List[Dict[str, object]] = []
    for rank in range(1, top_k + 1):
        ptxt = norm_text(row.get(f"ref_{rank:02d}_path", ""))
        if not ptxt:
            continue
        p = Path(ptxt)
        if not p.exists():
            continue
        rec = {
            "rank": rank,
            "path": p,
            "patch_id": norm_text(row.get(f"ref_{rank:02d}_patch_id", "")),
            "style_group": norm_text(row.get(f"ref_{rank:02d}_style_group", "")),
            "gender": norm_text(row.get(f"ref_{rank:02d}_gender", "")),
            "final_score": float(row.get(f"ref_{rank:02d}_final_score", 0.0) or 0.0),
            "visual_score": float(row.get(f"ref_{rank:02d}_visual_score", 0.0) or 0.0),
        }
        refs.append(rec)
    return refs


def build_subdirs(base: Path) -> Dict[str, Path]:
    return {
        "aligned": base / "aligned_reference_edges",
        "fused": base / "fused_reference_edge_patch",
        "prior_canvas": base / "reference_structural_prior_canvas",
        "prior_masked": base / "reference_structural_prior_masked",
        "structure": base / "structure_condition",
        "overlays": base / "overlays",
        "preview": base / "preview",
        "metadata": base / "metadata",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Refine Reference Structural Prior to geometry-aware v2")
    ap.add_argument("--project_root", type=str, default=str(DEFAULT_PROJECT_ROOT))
    ap.add_argument("--encoder_tag", type=str, default="dinov2_small")
    ap.add_argument("--top_k", type=int, default=10)
    ap.add_argument("--select_top_n", type=int, default=5)
    ap.add_argument("--output_size", type=int, default=512)
    ap.add_argument("--edge_method", type=str, default="canny", choices=["canny", "sobel", "pil"])
    ap.add_argument("--canny_low", type=int, default=60)
    ap.add_argument("--canny_high", type=int, default=160)
    ap.add_argument("--disable_auto_canny", action="store_true")
    ap.add_argument("--softmax_tau", type=float, default=0.10)
    ap.add_argument("--geom_power", type=float, default=1.0)
    ap.add_argument("--min_geom_score", type=float, default=0.34)
    ap.add_argument("--mask_dilate_radius", type=int, default=8)
    ap.add_argument("--min_ref_gain", type=float, default=0.20)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--preview_rows", type=int, default=12)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.project_root)
    bench_csv = root / "restoration_benchmark" / "metadata" / "inpaint_benchmark_master.csv"
    retr_csv = root / "restoration_benchmark" / "retrieval" / args.encoder_tag / "inpaint_benchmark_with_references.csv"
    out_root = root / "restoration_benchmark" / "reference_structural_prior_v2" / args.encoder_tag

    if not bench_csv.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {bench_csv}")
    if not retr_csv.exists():
        raise FileNotFoundError(f"Retrieval CSV not found: {retr_csv}")

    maybe_clear_output_dir(out_root, overwrite=args.overwrite)
    subdirs = build_subdirs(out_root)
    for p in subdirs.values():
        ensure_dir(p)

    bench_df = read_csv_flexible(bench_csv)
    retr_df = read_csv_flexible(retr_csv)
    if args.max_samples and args.max_samples > 0:
        retr_df = retr_df.head(int(args.max_samples)).copy()

    bench_map = {norm_text(r.get("sample_id", "")): r for _, r in bench_df.iterrows()}

    records: List[Dict[str, object]] = []
    issues: List[Dict[str, object]] = []
    preview_rows: List[Dict[str, object]] = []

    for _, row in tqdm(retr_df.iterrows(), total=len(retr_df), desc="Refining RSP v2"):
        sample_id = safe_name(row.get("sample_id", "sample"))
        split = norm_text(row.get("split", "train")) or "train"
        bench = bench_map.get(sample_id)
        if bench is None:
            issues.append({"sample_id": sample_id, "issue": "sample_not_found_in_benchmark_csv"})
            continue

        clean_path = Path(norm_text(bench.get("clean_path", "")))
        damaged_path = Path(norm_text(bench.get("damaged_path", "")))
        mask_path = Path(norm_text(bench.get("mask_path", "")))
        visible_path = Path(norm_text(bench.get("visible_edge_path", "")))
        if not (clean_path.exists() and damaged_path.exists() and mask_path.exists() and visible_path.exists()):
            issues.append({"sample_id": sample_id, "issue": "missing_input_paths"})
            continue

        roi_type = norm_text(bench.get("roi_type", row.get("roi_type", "unknown")))
        query_style = norm_style(bench.get("final_style_group", row.get("final_style_group", "")))
        query_gender = bench.get("gender", row.get("gender", ""))

        mask_img = load_gray(mask_path, (args.output_size, args.output_size))
        visible_img = load_gray(visible_path, (args.output_size, args.output_size))
        damaged_img = load_rgb(damaged_path, (args.output_size, args.output_size))
        target_box = choose_target_box(bench, mask_img, args.output_size, roi_type)

        refs = collect_refs(row, args.top_k)
        if not refs:
            issues.append({"sample_id": sample_id, "issue": "no_reference_paths"})
            continue

        aligned_dir = subdirs["aligned"] / split / sample_id
        ensure_dir(aligned_dir)

        cand_rows: List[Dict[str, object]] = []
        for ref in refs:
            try:
                gray = load_gray(Path(ref["path"]), (args.output_size, args.output_size))
                edge = compute_edge(gray, args.edge_method, args.canny_low, args.canny_high, not args.disable_auto_canny)
                rb = nonzero_bbox(edge)
                if rb is None:
                    issues.append({"sample_id": sample_id, "issue": "reference_edge_empty", "ref_path": str(ref["path"])})
                    continue
                gscore, gdiag = geometry_compat(rb, target_box, args.output_size, roi_type)
                if gscore < args.min_geom_score:
                    issues.append({"sample_id": sample_id, "issue": "geometry_score_below_threshold", "ref_path": str(ref["path"]), "geom_score": gscore})
                    continue
                scomp = style_compat(query_style, ref.get("style_group", ""))
                gcom = gender_compat(query_gender, ref.get("gender", ""), roi_type)
                base_score = max(float(ref.get("final_score", 0.0)), 1e-6)
                combined = base_score * (gscore ** float(args.geom_power)) * scomp * gcom
                aligned = align_reference_edge(edge, rb, target_box, args.output_size, roi_type)
                save_path = aligned_dir / f"ref_aligned_{int(ref['rank']):02d}.png"
                aligned.save(save_path)
                cand_rows.append({
                    **ref,
                    **gdiag,
                    "geom_score": float(gscore),
                    "style_compat": float(scomp),
                    "gender_compat": float(gcom),
                    "combined_score": float(combined),
                    "aligned_path": str(save_path),
                })
            except Exception as e:
                issues.append({"sample_id": sample_id, "issue": "reference_process_error", "ref_path": str(ref.get('path', '')), "error": str(e)})

        if not cand_rows:
            issues.append({"sample_id": sample_id, "issue": "no_reference_survived_geometry_filter"})
            continue

        cand_rows = sorted(cand_rows, key=lambda x: (-float(x["combined_score"]), int(x["rank"])))[: max(1, int(args.select_top_n))]
        weights = softmax_weights([float(x["combined_score"]) for x in cand_rows], args.softmax_tau)

        accum = np.zeros((args.output_size, args.output_size), dtype=np.float32)
        for i, c in enumerate(cand_rows):
            apath = Path(str(c["aligned_path"]))
            arr = np.asarray(load_gray(apath, (args.output_size, args.output_size)), dtype=np.float32) / 255.0
            accum += float(weights[i]) * arr
        q = np.quantile(accum, 0.995) if float(accum.max()) > 0 else 1.0
        if q > 1e-6:
            accum = np.clip(accum / q, 0, 1)
        prior_canvas = Image.fromarray((accum * 255.0).astype(np.uint8), mode="L")

                                                       
        tx1, ty1, tx2, ty2 = target_box
        patch = prior_canvas.crop((tx1, ty1, tx2, ty2))
        prior_masked = masked_prior(prior_canvas, mask_img, args.mask_dilate_radius)

        top1_w = float(weights.max()) if len(weights) else 0.0
        ent = entropy_norm(weights)
        mean_geom = float(np.average([float(x["geom_score"]) for x in cand_rows], weights=weights))
        mean_style = float(np.average([float(x["style_compat"]) for x in cand_rows], weights=weights))
        mean_gender = float(np.average([float(x["gender_compat"]) for x in cand_rows], weights=weights))
                                                                                       
        if roi_type == "face":
            conf = 0.30 * top1_w + 0.20 * (1.0 - ent) + 0.30 * mean_geom + 0.10 * mean_style + 0.10 * mean_gender
        elif roi_type == "head":
            conf = 0.28 * top1_w + 0.18 * (1.0 - ent) + 0.27 * mean_geom + 0.15 * mean_style + 0.12 * mean_gender
        else:
            conf = 0.24 * top1_w + 0.16 * (1.0 - ent) + 0.24 * mean_geom + 0.20 * mean_style + 0.16 * mean_gender
        conf = float(np.clip(conf, 0.05, 1.0))

        structure_img = combine_structure_condition(visible_img, prior_masked, conf, args.min_ref_gain)
        overlay_img = make_overlay(damaged_img, mask_img, prior_masked, structure_img)

        ensure_dir(subdirs["fused"] / split)
        ensure_dir(subdirs["prior_canvas"] / split)
        ensure_dir(subdirs["prior_masked"] / split)
        ensure_dir(subdirs["structure"] / split)
        ensure_dir(subdirs["overlays"] / split)
        fused_path = subdirs["fused"] / split / f"{sample_id}.png"
        prior_canvas_path = subdirs["prior_canvas"] / split / f"{sample_id}.png"
        prior_masked_path = subdirs["prior_masked"] / split / f"{sample_id}.png"
        structure_path = subdirs["structure"] / split / f"{sample_id}.png"
        overlay_path = subdirs["overlays"] / split / f"{sample_id}.jpg"
        patch.save(fused_path)
        prior_canvas.save(prior_canvas_path)
        prior_masked.save(prior_masked_path)
        structure_img.save(structure_path)
        overlay_img.save(overlay_path, quality=92)

        rec = dict(bench.to_dict())
        rec.update({
            "selected_reference_count": int(len(cand_rows)),
            "selected_reference_patch_ids_json": json.dumps([str(x.get("patch_id", "")) for x in cand_rows], ensure_ascii=False),
            "selected_reference_aligned_paths_json": json.dumps([str(x.get("aligned_path", "")) for x in cand_rows], ensure_ascii=False),
            "selected_reference_geom_scores_json": json.dumps([float(x.get("geom_score", 0.0)) for x in cand_rows], ensure_ascii=False),
            "selected_reference_combined_scores_json": json.dumps([float(x.get("combined_score", 0.0)) for x in cand_rows], ensure_ascii=False),
            "selected_reference_weights_json": json.dumps([float(x) for x in weights], ensure_ascii=False),
            "reference_weight_top1": top1_w,
            "reference_weight_entropy_norm": ent,
            "reference_mean_geometry_compat": mean_geom,
            "reference_mean_style_compat": mean_style,
            "reference_mean_gender_compat": mean_gender,
            "ref_confidence": conf,
            "reference_prior_target_box_xmin": int(target_box[0]),
            "reference_prior_target_box_ymin": int(target_box[1]),
            "reference_prior_target_box_xmax": int(target_box[2]),
            "reference_prior_target_box_ymax": int(target_box[3]),
            "fused_reference_edge_patch_path": str(fused_path),
            "reference_structural_prior_canvas_path": str(prior_canvas_path),
            "reference_structural_prior_masked_path": str(prior_masked_path),
            "structure_condition_path": str(structure_path),
            "overlay_path": str(overlay_path),
        })
        records.append(rec)
        if len(preview_rows) < args.preview_rows:
            preview_rows.append(rec)

    out_df = pd.DataFrame(records)
    issue_df = pd.DataFrame(issues)
    meta_dir = subdirs["metadata"]
    out_csv = meta_dir / "reference_structural_prior_v2_master.csv"
    issues_csv = meta_dir / "reference_structural_prior_v2_issues.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    issue_df.to_csv(issues_csv, index=False, encoding="utf-8-sig")

    by_split = Counter(out_df.get("split", pd.Series(dtype=str)).astype(str)) if not out_df.empty else Counter()
    by_roi = Counter(out_df.get("roi_type", pd.Series(dtype=str)).astype(str)) if not out_df.empty else Counter()
    conf_stats = {
        "mean": float(out_df["ref_confidence"].mean()) if (not out_df.empty and "ref_confidence" in out_df.columns) else 0.0,
        "median": float(out_df["ref_confidence"].median()) if (not out_df.empty and "ref_confidence" in out_df.columns) else 0.0,
        "min": float(out_df["ref_confidence"].min()) if (not out_df.empty and "ref_confidence" in out_df.columns) else 0.0,
        "max": float(out_df["ref_confidence"].max()) if (not out_df.empty and "ref_confidence" in out_df.columns) else 0.0,
    }
    summary = {
        "project_root": str(root),
        "encoder_tag": args.encoder_tag,
        "benchmark_samples_input": int(len(retr_df)),
        "benchmark_samples_output": int(len(out_df)),
        "issues": int(len(issue_df)),
        "counts_by_split": dict(by_split),
        "counts_by_roi_type": dict(by_roi),
        "confidence": conf_stats,
        "parameters": {
            "top_k": int(args.top_k),
            "select_top_n": int(args.select_top_n),
            "output_size": int(args.output_size),
            "edge_method": args.edge_method,
            "softmax_tau": float(args.softmax_tau),
            "geom_power": float(args.geom_power),
            "min_geom_score": float(args.min_geom_score),
            "mask_dilate_radius": int(args.mask_dilate_radius),
            "min_ref_gain": float(args.min_ref_gain),
        },
    }
    with open(meta_dir / "reference_structural_prior_v2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    make_preview_sheet(preview_rows, subdirs["preview"] / "reference_structural_prior_v2_preview_sheet.jpg", max_rows=args.preview_rows)

    print("\n[DONE] 09b Reference Structural Prior v2 finished.")
    print(f"[DONE] Samples: {len(out_df)}")
    print(f"[DONE] Master CSV: {out_csv}")
    print(f"[DONE] Preview: {subdirs['preview'] / 'reference_structural_prior_v2_preview_sheet.jpg'}")
    print(f"[DONE] Issues: {len(issue_df)}")


if __name__ == "__main__":
    main()
