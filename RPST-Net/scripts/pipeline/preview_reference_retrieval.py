                     
                       

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

STYLE_ORDER = ["S0", "S1", "S2", "S3"]
STYLE_ALIAS = {
    "s uk": "S_UK",
    "s_uk": "S_UK",
    "suk": "S_UK",
    "unknown": "S_UK",
    "s0": "S0",
    "s1": "S1",
    "s2": "S2",
    "s3": "S3",
}


def norm_text(x: object) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x).strip()


def norm_style(x: object) -> str:
    s = norm_text(x)
    key = s.lower().replace("-", "_").strip()
    return STYLE_ALIAS.get(key, s if s else "S_UK")


def style_score(a: object, b: object) -> float:
    """Soft style compatibility score in [0, 1]."""
    sa, sb = norm_style(a), norm_style(b)
    if not sa or not sb:
        return 0.35
    if sa == sb:
        return 1.0
    if "S_UK" in {sa, sb}:
        return 0.45
                                                          
    if {sa, sb} in [{"S1", "S2"}, {"S2", "S3"}]:
        return 0.80
    if sa in STYLE_ORDER and sb in STYLE_ORDER:
        da = abs(STYLE_ORDER.index(sa) - STYLE_ORDER.index(sb))
        if da == 1:
            return 0.62
        if da == 2:
            return 0.35
        return 0.20
    return 0.30


def gender_score(a: object, b: object) -> float:
    ga, gb = norm_text(a).lower(), norm_text(b).lower()
    def gnorm(g: str) -> str:
        if g in {"man", "male"}:
            return "man"
        if g in {"woman", "female"}:
            return "woman"
        return "unknown" if not g or g in {"nan", "none", "unknown"} else g
    ga, gb = gnorm(ga), gnorm(gb)
    if ga == "unknown" or gb == "unknown":
        return 0.50
    return 1.0 if ga == gb else 0.0


def role_score(role: object) -> float:
    r = norm_text(role).lower()
    if r == "clean_reference":
        return 1.0
    if r == "partial_reference":
        return 0.65
    if r == "weak_structure":
        return 0.10
    return 0.40


def read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def resolve_patch_path(row: pd.Series, reference_bank_dir: Path) -> Optional[Path]:
    vals = []
    for col in ["patch_path", "patch_rel_path"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            vals.append(str(row[col]).strip())
    candidates: List[Path] = []
    for v in vals:
        p = Path(v)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(reference_bank_dir / p)
    for p in candidates:
        if p.exists():
            return p
    return candidates[0] if candidates else None


def load_thumb(path: Path, size: int = 160) -> Image.Image:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (size, size), (245, 245, 245))
            x = (size - im.width) // 2
            y = (size - im.height) // 2
            canvas.paste(im, (x, y))
            return canvas
    except Exception:
        canvas = Image.new("RGB", (size, size), (230, 230, 230))
        d = ImageDraw.Draw(canvas)
        d.text((10, size // 2 - 10), "MISSING", fill=(0, 0, 0))
        return canvas


def draw_text_block(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, width: int, font) -> None:
    x, y = xy
    lines: List[str] = []
    for raw in text.split("\n"):
        raw = raw.strip()
        if not raw:
            lines.append("")
            continue
        line = ""
        for token in re.split(r"(\s+|[_/\\.-])", raw):
            cand = line + token
                                                                             
            try:
                length = draw.textlength(cand, font=font)
            except Exception:
                length = len(cand) * 7
            if length <= width or not line:
                line = cand
            else:
                lines.append(line)
                line = token.lstrip()
        if line:
            lines.append(line)
    for line in lines[:5]:
        draw.text((x, y), line, fill=(0, 0, 0), font=font)
        y += 14


def make_contact_sheet(
    query_row: pd.Series,
    result_rows: List[pd.Series],
    result_scores: List[Dict[str, float]],
    reference_bank_dir: Path,
    output_path: Path,
    thumb_size: int = 160,
) -> None:
    n = 1 + len(result_rows)
    label_h = 82
    margin = 10
    gap = 8
    w = margin * 2 + n * thumb_size + (n - 1) * gap
    h = margin * 2 + thumb_size + label_h
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 11)
        font_bold = ImageFont.truetype("arialbd.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_bold = font

    all_rows = [query_row] + result_rows
    for i, row in enumerate(all_rows):
        x = margin + i * (thumb_size + gap)
        y = margin
        p = resolve_patch_path(row, reference_bank_dir)
        thumb = load_thumb(p, size=thumb_size) if p is not None else load_thumb(Path("__missing__"), size=thumb_size)
        sheet.paste(thumb, (x, y))
        draw.rectangle([x, y, x + thumb_size - 1, y + thumb_size - 1], outline=(80, 80, 80), width=1)
        if i == 0:
            title = "QUERY"
            subtitle = f"{row.get('roi_type','')}|{row.get('quality_type','')}|{row.get('final_style_group','')}\n{row.get('patch_id','')}"
        else:
            sc = result_scores[i - 1]
            title = f"R{i:02d} final={sc['final_score']:.3f}"
            subtitle = (
                f"vis={sc['visual_score']:.3f} style={sc['style_score']:.2f} gender={sc['gender_score']:.2f}\n"
                f"{row.get('roi_type','')}|{row.get('quality_type','')}|{row.get('final_style_group','')}\n"
                f"{row.get('patch_id','')}"
            )
        draw.text((x, y + thumb_size + 4), title, fill=(0, 0, 0), font=font_bold)
        draw_text_block(draw, (x, y + thumb_size + 20), subtitle, thumb_size, font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def choose_queries(
    df: pd.DataFrame,
    patch_size: int,
    query_roles: Sequence[str],
    samples_per_roi: int,
    max_queries: Optional[int],
    seed: int,
) -> pd.DataFrame:
    q = df.copy()
    if "patch_size" in q.columns:
        q = q[q["patch_size"].astype(int) == int(patch_size)]
    if query_roles:
        q = q[q["crop_role"].astype(str).isin(set(query_roles))]
    if q.empty:
        raise ValueError("No query patches after filtering.")

    rng = np.random.default_rng(seed)
    selected = []
                                                       
    group_cols = [c for c in ["roi_type", "quality_type"] if c in q.columns]
    for _, g in q.groupby(group_cols, dropna=False):
        n = min(samples_per_roi, len(g))
        selected.extend(rng.choice(g.index.to_numpy(), size=n, replace=False).tolist())
    rng.shuffle(selected)
    if max_queries is not None:
        selected = selected[:max_queries]
    return q.loc[selected].copy().reset_index(drop=True)


def filter_candidate_indices(
    meta: pd.DataFrame,
    query_row: pd.Series,
    candidate_mask_base: np.ndarray,
    roi_policy: str,
    min_candidates: int,
    exclude_same_image: bool,
    exclude_same_tomb: bool,
) -> np.ndarray:
    mask = candidate_mask_base.copy()
    if exclude_same_image and "image_id" in meta.columns:
        mask &= meta["image_id"].astype(str).values != str(query_row.get("image_id", ""))
    if exclude_same_tomb and "tomb_id" in meta.columns:
        mask &= meta["tomb_id"].astype(str).values != str(query_row.get("tomb_id", ""))

    if roi_policy == "required" and "roi_type" in meta.columns:
        roi_mask = meta["roi_type"].astype(str).values == str(query_row.get("roi_type", ""))
        mask2 = mask & roi_mask
        if mask2.sum() >= min_candidates:
            mask = mask2
    elif roi_policy == "soft":
        pass
    return np.where(mask)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview style-soft retrieval top-k results from official reference bank index.")
    parser.add_argument("--project_root", type=Path, default=Path(r"<PROJECT_ROOT>"))
    parser.add_argument("--encoder_tag", type=str, default="dinov2_small")
    parser.add_argument("--retrieval_dir", type=Path, default=None)
    parser.add_argument("--reference_bank_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--query_patch_size", type=int, default=256)
    parser.add_argument("--candidate_patch_size", type=int, default=256)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--candidate_k", type=int, default=300)
    parser.add_argument("--samples_per_roi", type=int, default=3)
    parser.add_argument("--max_queries", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--query_roles", nargs="+", default=["clean_reference", "partial_reference", "weak_structure"])
    parser.add_argument("--candidate_roles", nargs="+", default=["clean_reference", "partial_reference"])
    parser.add_argument("--candidate_split", type=str, default="train", help="Use 'all' to allow all splits.")
    parser.add_argument("--roi_policy", choices=["required", "soft"], default="required")
    parser.add_argument("--min_candidates", type=int, default=10)
    parser.add_argument("--exclude_same_image", action="store_true", default=True)
    parser.add_argument("--allow_same_image", action="store_true")
    parser.add_argument("--exclude_same_tomb", action="store_true", help="Stricter check; may reduce candidates too much.")
    parser.add_argument("--style_weight", type=float, default=0.08)
    parser.add_argument("--gender_weight", type=float, default=0.03)
    parser.add_argument("--role_weight", type=float, default=0.02)
    parser.add_argument("--roi_weight", type=float, default=0.04)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root
    retrieval_dir = args.retrieval_dir or (project_root / "retrieval" / args.encoder_tag)
    reference_bank_dir = args.reference_bank_dir or (project_root / "reference_bank")
    output_dir = args.output_dir or (retrieval_dir / "retrieval_preview")

    meta_path = retrieval_dir / "index_metadata.csv"
    emb_path = retrieval_dir / "embeddings.npy"
    if not meta_path.exists():
        raise FileNotFoundError(f"index_metadata.csv not found: {meta_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"embeddings.npy not found: {emb_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output dir exists and is not empty: {output_dir}. Use --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "contact_sheets").mkdir(parents=True, exist_ok=True)

    meta = read_csv_flexible(meta_path).reset_index(drop=True)
    emb = np.load(emb_path).astype("float32")
    if len(meta) != emb.shape[0]:
        raise ValueError(f"Metadata rows {len(meta)} != embeddings rows {emb.shape[0]}")
    if "index_id" not in meta.columns:
        meta.insert(0, "index_id", np.arange(len(meta), dtype=np.int64))

                     
    queries = choose_queries(
        meta,
        patch_size=args.query_patch_size,
        query_roles=args.query_roles,
        samples_per_roi=args.samples_per_roi,
        max_queries=args.max_queries,
        seed=args.seed,
    )
    queries.to_csv(output_dir / "retrieval_preview_queries.csv", index=False, encoding="utf-8-sig")

                          
    cand_mask = np.ones(len(meta), dtype=bool)
    if "patch_size" in meta.columns:
        cand_mask &= meta["patch_size"].astype(int).values == int(args.candidate_patch_size)
    if args.candidate_roles:
        cand_mask &= meta["crop_role"].astype(str).isin(set(args.candidate_roles)).values
    if args.candidate_split.lower() != "all" and "split" in meta.columns:
        cand_mask &= meta["split"].astype(str).str.lower().values == args.candidate_split.lower()

    exclude_same_image = args.exclude_same_image and not args.allow_same_image
    rows_out: List[Dict[str, object]] = []
    warning_rows: List[Dict[str, object]] = []

    for _, qrow in tqdm(queries.iterrows(), total=len(queries), desc="Retrieval preview"):
        q_idx = int(qrow["index_id"])
        cand_idxs = filter_candidate_indices(
            meta=meta,
            query_row=qrow,
            candidate_mask_base=cand_mask,
            roi_policy=args.roi_policy,
            min_candidates=args.min_candidates,
            exclude_same_image=exclude_same_image,
            exclude_same_tomb=args.exclude_same_tomb,
        )
        if len(cand_idxs) == 0:
            warning_rows.append({"query_index_id": q_idx, "patch_id": qrow.get("patch_id", ""), "warning": "no_candidates"})
            continue

                                                                                    
        scores_visual = emb[cand_idxs] @ emb[q_idx]
                                                             
        n_cand = min(args.candidate_k, len(cand_idxs))
        top_local = np.argpartition(-scores_visual, np.arange(n_cand))[:n_cand]
        pool_idxs = cand_idxs[top_local]
        pool_visual = scores_visual[top_local]

        rescored = []
        q_roi = str(qrow.get("roi_type", ""))
        for idx, visual in zip(pool_idxs, pool_visual):
            r = meta.iloc[int(idx)]
            ss = style_score(qrow.get("final_style_group", ""), r.get("final_style_group", ""))
            gs = gender_score(qrow.get("gender", ""), r.get("gender", ""))
            rs = role_score(r.get("crop_role", ""))
            roi_s = 1.0 if str(r.get("roi_type", "")) == q_roi else 0.0
            final = float(visual) + args.style_weight * ss + args.gender_weight * gs + args.role_weight * rs + args.roi_weight * roi_s
            rescored.append((final, int(idx), float(visual), ss, gs, rs, roi_s))
        rescored.sort(key=lambda x: x[0], reverse=True)
        rescored = rescored[: args.top_k]

        result_rows = []
        result_scores = []
        for rank, (final, idx, visual, ss, gs, rs, roi_s) in enumerate(rescored, start=1):
            r = meta.iloc[idx]
            result_rows.append(r)
            score_dict = {
                "final_score": final,
                "visual_score": visual,
                "style_score": ss,
                "gender_score": gs,
                "role_score": rs,
                "roi_score": roi_s,
            }
            result_scores.append(score_dict)
            rows_out.append({
                "query_index_id": q_idx,
                "query_patch_id": qrow.get("patch_id", ""),
                "query_image_id": qrow.get("image_id", ""),
                "query_split": qrow.get("split", ""),
                "query_roi_type": qrow.get("roi_type", ""),
                "query_quality_type": qrow.get("quality_type", ""),
                "query_crop_role": qrow.get("crop_role", ""),
                "query_style_group": qrow.get("final_style_group", ""),
                "query_gender": qrow.get("gender", ""),
                "rank": rank,
                "ref_index_id": idx,
                "ref_patch_id": r.get("patch_id", ""),
                "ref_image_id": r.get("image_id", ""),
                "ref_split": r.get("split", ""),
                "ref_roi_type": r.get("roi_type", ""),
                "ref_quality_type": r.get("quality_type", ""),
                "ref_crop_role": r.get("crop_role", ""),
                "ref_style_group": r.get("final_style_group", ""),
                "ref_gender": r.get("gender", ""),
                "visual_score": visual,
                "style_score": ss,
                "gender_score": gs,
                "role_score": rs,
                "roi_score": roi_s,
                "final_score": final,
                "ref_patch_path": r.get("patch_path", ""),
                "ref_patch_rel_path": r.get("patch_rel_path", ""),
            })

        safe_patch = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(qrow.get("patch_id", f"q{q_idx}")))[:120]
        sheet_path = output_dir / "contact_sheets" / f"{safe_patch}_top{args.top_k}.jpg"
        make_contact_sheet(qrow, result_rows, result_scores, reference_bank_dir, sheet_path)

    topk_df = pd.DataFrame(rows_out)
    topk_df.to_csv(output_dir / "retrieval_preview_topk.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(warning_rows).to_csv(output_dir / "retrieval_preview_warnings.csv", index=False, encoding="utf-8-sig")

    summary = {
        "retrieval_dir": str(retrieval_dir),
        "reference_bank_dir": str(reference_bank_dir),
        "output_dir": str(output_dir),
        "metadata_rows": int(len(meta)),
        "embedding_shape": list(map(int, emb.shape)),
        "query_count": int(len(queries)),
        "topk_rows": int(len(topk_df)),
        "warnings": int(len(warning_rows)),
        "query_patch_size": args.query_patch_size,
        "candidate_patch_size": args.candidate_patch_size,
        "candidate_roles": args.candidate_roles,
        "candidate_split": args.candidate_split,
        "roi_policy": args.roi_policy,
        "exclude_same_image": bool(exclude_same_image),
        "exclude_same_tomb": bool(args.exclude_same_tomb),
        "counts_query_roi_type": Counter(queries.get("roi_type", pd.Series(dtype=str)).astype(str)).most_common(),
        "counts_query_quality_type": Counter(queries.get("quality_type", pd.Series(dtype=str)).astype(str)).most_common(),
        "counts_ref_roi_type": Counter(topk_df.get("ref_roi_type", pd.Series(dtype=str)).astype(str)).most_common() if not topk_df.empty else [],
        "counts_ref_style_group": Counter(topk_df.get("ref_style_group", pd.Series(dtype=str)).astype(str)).most_common() if not topk_df.empty else [],
    }
    with open(output_dir / "retrieval_preview_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n[DONE] Retrieval preview finished.")
    print(f"[DONE] Queries: {len(queries)}")
    print(f"[DONE] Top-k rows: {len(topk_df)}")
    print(f"[DONE] Output: {output_dir}")
    print(f"[DONE] Contact sheets: {output_dir / 'contact_sheets'}")


if __name__ == "__main__":
    main()
