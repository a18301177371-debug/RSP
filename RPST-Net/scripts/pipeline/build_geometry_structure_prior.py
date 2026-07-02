                     
                       

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
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageFile
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

ROI_EXPAND = {
    "face": 1.35,
    "head": 1.25,
    "hand": 1.45,
    "cloth": 1.15,
    "figure": 1.08,
    "object": 1.20,
    "architecture": 1.08,
    "ornament": 1.18,
    "animal": 1.15,
    "plant": 1.15,
}

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
                                                                                    
        kernel = np.ones((2, 2), np.uint8)
        edge = cv2.dilate(edge, kernel, iterations=1)
        return Image.fromarray(edge, mode="L")
    if method == "sobel" and cv2 is not None:
        gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        if mag.max() > 0:
            mag = mag / mag.max() * 255.0
        return Image.fromarray(mag.astype(np.uint8), mode="L")
                        
    edge = gray.filter(ImageFilter.FIND_EDGES)
    edge = ImageOps.autocontrast(edge)
    return edge


def softmax_weights(scores: Sequence[float], tau: float) -> np.ndarray:
    arr = np.array([float(s) for s in scores], dtype=np.float64)
    if len(arr) == 0:
        return arr
    if not np.isfinite(arr).all() or np.allclose(arr.max(), arr.min()):
        return np.ones(len(arr), dtype=np.float64) / len(arr)
    tau = max(float(tau), 1e-6)
    z = (arr - arr.max()) / tau
    z = np.clip(z, -50, 50)
    exp = np.exp(z)
    return exp / max(exp.sum(), 1e-12)


def entropy_norm(weights: np.ndarray) -> float:
    if len(weights) <= 1:
        return 0.0
    w = np.clip(weights.astype(np.float64), 1e-12, 1.0)
    h = -float(np.sum(w * np.log(w)))
    return h / math.log(len(w))


def expand_box(xmin: float, ymin: float, xmax: float, ymax: float, scale: float, w: int, h: int) -> Tuple[int, int, int, int]:
    bw = max(1.0, xmax - xmin)
    bh = max(1.0, ymax - ymin)
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    size_w = bw * scale
    size_h = bh * scale
    x1 = int(round(max(0, cx - size_w / 2.0)))
    y1 = int(round(max(0, cy - size_h / 2.0)))
    x2 = int(round(min(w, cx + size_w / 2.0)))
    y2 = int(round(min(h, cy + size_h / 2.0)))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def mask_bbox(mask: Image.Image) -> Tuple[int, int, int, int]:
    arr = np.asarray(mask.convert("L"))
    ys, xs = np.where(arr > 127)
    if len(xs) == 0:
        return (0, 0, mask.width, mask.height)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def dilate_mask(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask.convert("L")
    arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    bin_arr = (arr > 127).astype(np.uint8) * 255
    if cv2 is not None:
        k = max(1, int(radius) * 2 + 1)
        kernel = np.ones((k, k), np.uint8)
        out = cv2.dilate(bin_arr, kernel, iterations=1)
        return Image.fromarray(out, mode="L")
    return Image.fromarray(bin_arr, mode="L").filter(ImageFilter.MaxFilter(max(3, radius * 2 + 1)))


def collect_ref_paths_and_scores(row: pd.Series, top_k: int, score_kind: str) -> Tuple[List[Path], List[float], List[str], List[str]]:
    paths: List[Path] = []
    scores: List[float] = []
    styles: List[str] = []
    labels: List[str] = []
    for rank in range(1, top_k + 1):
        ptxt = norm_text(row.get(f"ref_{rank:02d}_path", ""))
        if not ptxt:
            continue
        p = Path(ptxt)
        if not p.exists():
                                                                                
            continue
        paths.append(p)
        score = row.get(f"ref_{rank:02d}_{score_kind}_score", np.nan)
        if pd.isna(score):
            score = row.get(f"ref_{rank:02d}_final_score", np.nan)
        if pd.isna(score):
            score = row.get(f"ref_{rank:02d}_visual_score", 0.0)
        try:
            scores.append(float(score))
        except Exception:
            scores.append(0.0)
        styles.append(norm_text(row.get(f"ref_{rank:02d}_style_group", "")))
        labels.append(norm_text(row.get(f"ref_{rank:02d}_patch_id", "")))
    return paths, scores, styles, labels


def fuse_reference_edges(
    ref_paths: List[Path],
    weights: np.ndarray,
    output_size: int,
    edge_method: str,
    canny_low: int,
    canny_high: int,
    auto_canny: bool,
    edge_save_dir: Optional[Path] = None,
) -> Tuple[Image.Image, List[str]]:
    if len(ref_paths) == 0:
        return Image.new("L", (output_size, output_size), 0), []
    accum = np.zeros((output_size, output_size), dtype=np.float32)
    saved: List[str] = []
    for i, p in enumerate(ref_paths):
        gray = load_gray(p, (output_size, output_size))
        edge = compute_edge(gray, edge_method, canny_low, canny_high, auto_canny)
        edge_arr = np.asarray(edge, dtype=np.float32) / 255.0
        w = float(weights[i]) if i < len(weights) else 1.0 / len(ref_paths)
        accum += w * edge_arr
        if edge_save_dir is not None:
            ensure_dir(edge_save_dir)
            edge_path = edge_save_dir / f"ref_edge_{i+1:02d}.png"
            edge.save(edge_path)
            saved.append(str(edge_path))
                                                         
    q = np.quantile(accum, 0.995) if accum.max() > 0 else 1.0
    if q > 1e-6:
        accum = np.clip(accum / q, 0, 1)
    out = (accum * 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="L"), saved


def paste_prior_to_canvas(fused_patch: Image.Image, box: Tuple[int, int, int, int], canvas_size: int) -> Image.Image:
    x1, y1, x2, y2 = box
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    resized = fused_patch.resize((bw, bh), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (canvas_size, canvas_size), 0)
    canvas.paste(resized, (x1, y1))
    return canvas


def masked_prior(prior: Image.Image, mask: Image.Image, dilate_radius: int) -> Image.Image:
    m = dilate_mask(mask, dilate_radius)
    p = np.asarray(prior.convert("L"), dtype=np.uint8)
    ma = (np.asarray(m, dtype=np.uint8) > 127).astype(np.uint8)
    out = (p * ma).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def combine_structure_condition(visible_edge: Image.Image, ref_prior_masked: Image.Image, confidence: float, min_ref_gain: float) -> Image.Image:
    v = np.asarray(visible_edge.convert("L"), dtype=np.float32)
    r = np.asarray(ref_prior_masked.convert("L"), dtype=np.float32)
    gain = float(min_ref_gain + (1.0 - min_ref_gain) * max(0.0, min(1.0, confidence)))
    out = np.maximum(v, r * gain)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def make_overlay(damaged: Image.Image, mask: Image.Image, prior: Image.Image, condition: Image.Image) -> Image.Image:
    base = damaged.convert("RGB")
    m = np.asarray(mask.convert("L"), dtype=np.uint8)
    p = np.asarray(prior.convert("L"), dtype=np.uint8)
    c = np.asarray(condition.convert("L"), dtype=np.uint8)
    arr = np.asarray(base, dtype=np.uint8).copy()
                           
    mask_region = m > 127
    arr[mask_region, 0] = np.maximum(arr[mask_region, 0], 180)
    arr[mask_region, 1] = (arr[mask_region, 1] * 0.55).astype(np.uint8)
    arr[mask_region, 2] = (arr[mask_region, 2] * 0.55).astype(np.uint8)
                                  
    prior_region = p > 80
    arr[prior_region, 1] = 255
    arr[prior_region, 0] = (arr[prior_region, 0] * 0.55).astype(np.uint8)
                                  
    cond_region = c > 180
    arr[cond_region, 2] = np.maximum(arr[cond_region, 2], 220)
    return Image.fromarray(arr, mode="RGB")


def draw_text_block(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, width: int, font: ImageFont.ImageFont) -> None:
    x, y = xy
    line = ""
    for token in text.split():
        trial = token if not line else line + " " + token
        try:
            w = draw.textbbox((0, 0), trial, font=font)[2]
        except Exception:
            w = len(trial) * 7
        if w <= width:
            line = trial
        else:
            draw.text((x, y), line, fill=(0, 0, 0), font=font)
            y += 14
            line = token
    if line:
        draw.text((x, y), line, fill=(0, 0, 0), font=font)


def make_preview_sheet(rows: List[Dict[str, object]], output_path: Path, max_rows: int = 40, thumb: int = 160) -> None:
    if not rows:
        return
    try:
        font = ImageFont.truetype("arial.ttf", 11)
        font_bold = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_bold = ImageFont.load_default()
    cols = ["clean", "damaged", "mask", "visible", "ref_prior", "structure", "overlay"]
    n = min(max_rows, len(rows))
    margin = 12
    text_h = 48
    cell_w = thumb + margin
    cell_h = thumb + text_h + margin
    sheet = Image.new("RGB", (len(cols) * cell_w + margin, n * cell_h + margin), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows[:n]):
        y = margin + i * cell_h
        title = f"{row.get('sample_id','')} | {row.get('roi_type','')} | conf={float(row.get('ref_confidence',0)):.2f}"
        draw.text((margin, y), title, fill=(0, 0, 0), font=font_bold)
        for j, col in enumerate(cols):
            path_col = {
                "clean": "clean_path",
                "damaged": "damaged_path",
                "mask": "mask_path",
                "visible": "visible_edge_path",
                "ref_prior": "reference_structural_prior_masked_path",
                "structure": "structure_condition_path",
                "overlay": "overlay_path",
            }[col]
            x = margin + j * cell_w
            py = y + 18
            p = Path(str(row.get(path_col, "")))
            try:
                im = load_rgb(p, (thumb, thumb)) if p.exists() else Image.new("RGB", (thumb, thumb), (230, 230, 230))
            except Exception:
                im = Image.new("RGB", (thumb, thumb), (230, 230, 230))
            sheet.paste(im, (x, py))
            draw.rectangle([x, py, x + thumb - 1, py + thumb - 1], outline=(180, 180, 180))
            draw.text((x, py + thumb + 3), col, fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Reference Structural Prior for inpainting benchmark samples.")
    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--encoder_tag", type=str, default="dinov2_small")
    parser.add_argument("--benchmark_dir", type=Path, default=None)
    parser.add_argument("--retrieval_csv", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--score_kind", choices=["final", "visual"], default="final")
    parser.add_argument("--softmax_tau", type=float, default=0.07)
    parser.add_argument("--edge_method", choices=["canny", "sobel", "pil"], default="canny")
    parser.add_argument("--canny_low", type=int, default=60)
    parser.add_argument("--canny_high", type=int, default=160)
    parser.add_argument("--disable_auto_canny", action="store_true")
    parser.add_argument("--mask_dilate_radius", type=int, default=9)
    parser.add_argument("--min_ref_gain", type=float, default=0.45)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--preview_rows", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root
    benchmark_dir = args.benchmark_dir or (project_root / "restoration_benchmark")
    retrieval_csv = args.retrieval_csv or (benchmark_dir / "retrieval" / args.encoder_tag / "inpaint_benchmark_with_references.csv")
    output_dir = args.output_dir or (benchmark_dir / "reference_structural_prior" / args.encoder_tag)

    if not retrieval_csv.exists():
        raise FileNotFoundError(f"Retrieval wide CSV not found: {retrieval_csv}")

    maybe_clear_output_dir(output_dir, args.overwrite)
    subdirs = {
        "reference_edges": output_dir / "reference_edges",
        "fused_patch": output_dir / "fused_reference_edge_patch",
        "prior_box": output_dir / "reference_structural_prior_box",
        "prior_masked": output_dir / "reference_structural_prior_masked",
        "structure": output_dir / "structure_condition",
        "overlays": output_dir / "overlays",
        "metadata": output_dir / "metadata",
        "preview": output_dir / "preview",
    }
    for p in subdirs.values():
        ensure_dir(p)

    df = read_csv_flexible(retrieval_csv).reset_index(drop=True)
    if args.max_samples is not None:
        df = df.head(int(args.max_samples)).copy()

    records: List[Dict[str, object]] = []
    issues: List[Dict[str, object]] = []
    preview_rows: List[Dict[str, object]] = []

    print(f"[INFO] Retrieval CSV: {retrieval_csv}")
    print(f"[INFO] Samples: {len(df)}")
    print(f"[INFO] Output dir: {output_dir}")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing reference structural priors"):
        sample_id = safe_name(row.get("sample_id", f"sample_{idx:06d}"))
        split = safe_name(row.get("split", "unknown"), "unknown")
        roi_type = norm_text(row.get("roi_type", "unknown")) or "unknown"
        output_size = int(row.get("output_size", 512) if not pd.isna(row.get("output_size", 512)) else 512)

        damaged_path = Path(norm_text(row.get("damaged_path", "")))
        clean_path = Path(norm_text(row.get("clean_path", "")))
        mask_path = Path(norm_text(row.get("mask_path", "")))
        visible_path = Path(norm_text(row.get("visible_edge_path", "")))
        if not damaged_path.exists() or not mask_path.exists() or not visible_path.exists():
            issues.append({"sample_id": sample_id, "issue": "missing_core_image", "damaged_path": str(damaged_path), "mask_path": str(mask_path), "visible_edge_path": str(visible_path)})
            continue

        ref_paths, scores, ref_styles, ref_labels = collect_ref_paths_and_scores(row, args.top_k, args.score_kind)
        if len(ref_paths) == 0:
            issues.append({"sample_id": sample_id, "issue": "no_existing_reference_paths"})
            continue
        weights = softmax_weights(scores, args.softmax_tau)
        h_norm = entropy_norm(weights)
        top1_w = float(weights.max()) if len(weights) else 0.0
        style_vals = [style_compat(row.get("final_style_group", ""), s) for s in ref_styles]
        mean_style = float(np.average(style_vals, weights=weights[:len(style_vals)])) if len(style_vals) == len(weights) and len(weights) else 0.0
        ref_conf = float(np.clip(0.45 * top1_w + 0.35 * (1.0 - h_norm) + 0.20 * mean_style, 0.05, 1.0))

        edge_dir = subdirs["reference_edges"] / split / sample_id
        fused_patch, saved_ref_edges = fuse_reference_edges(
            ref_paths=ref_paths,
            weights=weights,
            output_size=output_size,
            edge_method=args.edge_method,
            canny_low=args.canny_low,
            canny_high=args.canny_high,
            auto_canny=not args.disable_auto_canny,
            edge_save_dir=edge_dir,
        )

                                                                 
        tx1 = row.get("target_xmin", np.nan); ty1 = row.get("target_ymin", np.nan)
        tx2 = row.get("target_xmax", np.nan); ty2 = row.get("target_ymax", np.nan)
        if pd.isna(tx1) or pd.isna(ty1) or pd.isna(tx2) or pd.isna(ty2):
            mb = mask_bbox(load_gray(mask_path, (output_size, output_size)))
            tx1, ty1, tx2, ty2 = mb
        scale = ROI_EXPAND.get(roi_type, 1.2)
        prior_box = expand_box(float(tx1), float(ty1), float(tx2), float(ty2), scale, output_size, output_size)

        prior_box_img = paste_prior_to_canvas(fused_patch, prior_box, output_size)
        mask_img = load_gray(mask_path, (output_size, output_size))
        visible_edge = load_gray(visible_path, (output_size, output_size))
        prior_masked_img = masked_prior(prior_box_img, mask_img, args.mask_dilate_radius)
        structure_img = combine_structure_condition(visible_edge, prior_masked_img, ref_conf, args.min_ref_gain)
        damaged_img = load_rgb(damaged_path, (output_size, output_size))
        overlay_img = make_overlay(damaged_img, mask_img, prior_masked_img, structure_img)

                       
        ensure_dir(subdirs["fused_patch"] / split)
        ensure_dir(subdirs["prior_box"] / split)
        ensure_dir(subdirs["prior_masked"] / split)
        ensure_dir(subdirs["structure"] / split)
        ensure_dir(subdirs["overlays"] / split)
        fused_path = subdirs["fused_patch"] / split / f"{sample_id}.png"
        prior_box_path = subdirs["prior_box"] / split / f"{sample_id}.png"
        prior_masked_path = subdirs["prior_masked"] / split / f"{sample_id}.png"
        structure_path = subdirs["structure"] / split / f"{sample_id}.png"
        overlay_path = subdirs["overlays"] / split / f"{sample_id}.jpg"
        fused_patch.save(fused_path)
        prior_box_img.save(prior_box_path)
        prior_masked_img.save(prior_masked_path)
        structure_img.save(structure_path)
        overlay_img.save(overlay_path, quality=92)

        rec = row.to_dict()
        rec.update({
            "num_reference_edges_used": len(ref_paths),
            "reference_weight_top1": top1_w,
            "reference_weight_entropy_norm": h_norm,
            "reference_mean_style_compat": mean_style,
            "ref_confidence": ref_conf,
            "reference_prior_box_xmin": prior_box[0],
            "reference_prior_box_ymin": prior_box[1],
            "reference_prior_box_xmax": prior_box[2],
            "reference_prior_box_ymax": prior_box[3],
            "reference_edge_weights_json": json.dumps([float(x) for x in weights], ensure_ascii=False),
            "reference_edge_paths_json": json.dumps(saved_ref_edges, ensure_ascii=False),
            "fused_reference_edge_patch_path": str(fused_path),
            "reference_structural_prior_box_path": str(prior_box_path),
            "reference_structural_prior_masked_path": str(prior_masked_path),
            "structure_condition_path": str(structure_path),
            "overlay_path": str(overlay_path),
        })
        records.append(rec)
        if len(preview_rows) < args.preview_rows:
            preview_rows.append(rec)

    out_df = pd.DataFrame(records)
    issues_df = pd.DataFrame(issues)
    out_csv = subdirs["metadata"] / "reference_structural_prior_master.csv"
    issues_csv = subdirs["metadata"] / "reference_structural_prior_issues.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    issues_df.to_csv(issues_csv, index=False, encoding="utf-8-sig")

    summary = {
        "encoder_tag": args.encoder_tag,
        "input_retrieval_csv": str(retrieval_csv),
        "output_dir": str(output_dir),
        "input_samples": int(len(df)),
        "output_samples": int(len(out_df)),
        "issues": int(len(issues_df)),
        "edge_method": args.edge_method,
        "top_k": int(args.top_k),
        "softmax_tau": float(args.softmax_tau),
        "mask_dilate_radius": int(args.mask_dilate_radius),
        "counts_split": Counter(out_df.get("split", pd.Series(dtype=str)).astype(str)).most_common() if not out_df.empty else [],
        "counts_roi_type": Counter(out_df.get("roi_type", pd.Series(dtype=str)).astype(str)).most_common() if not out_df.empty else [],
        "counts_style_group": Counter(out_df.get("final_style_group", pd.Series(dtype=str)).astype(str)).most_common() if not out_df.empty else [],
        "ref_confidence_mean": float(out_df["ref_confidence"].mean()) if "ref_confidence" in out_df and not out_df.empty else None,
        "ref_confidence_min": float(out_df["ref_confidence"].min()) if "ref_confidence" in out_df and not out_df.empty else None,
        "ref_confidence_max": float(out_df["ref_confidence"].max()) if "ref_confidence" in out_df and not out_df.empty else None,
    }
    with open(subdirs["metadata"] / "reference_structural_prior_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    make_preview_sheet(preview_rows, subdirs["preview"] / "reference_structural_prior_preview_sheet.jpg", max_rows=args.preview_rows)

    print("\n[DONE] 09 Reference Structural Prior prepared.")
    print(f"[DONE] Output samples: {len(out_df)}")
    print(f"[DONE] Issues: {len(issues_df)}")
    print(f"[DONE] Master CSV: {out_csv}")
    print(f"[DONE] Summary: {subdirs['metadata'] / 'reference_structural_prior_summary.json'}")
    print(f"[DONE] Preview: {subdirs['preview'] / 'reference_structural_prior_preview_sheet.jpg'}")


if __name__ == "__main__":
    main()
