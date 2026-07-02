                     
                       


from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision.models import inception_v3

from skimage.metrics import structural_similarity as ssim_fn
from scipy.linalg import sqrtm


                               
      
                               
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=r"<PROJECT_ROOT>")
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--output_col", type=str, default="rsptnet_output_path")

    parser.add_argument("--compute_lpips", action="store_true")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--mask_threshold", type=int, default=127)
    parser.add_argument("--roi_margin", type=int, default=16)

    return parser.parse_args()


                               
             
                               
def load_img(path, size=None):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.BICUBIC)
    return np.array(img)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse < 1e-8:
        return 99.0
    return 20 * math.log10(255.0 / math.sqrt(mse))


def mae(a, b):
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def mask_psnr(a, b, mask):
    if mask.sum() == 0:
        return float("nan")
    a, b = a[mask], b[mask]
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse < 1e-8:
        return 99.0
    return 20 * math.log10(255.0 / math.sqrt(mse))


def bbox(mask, margin=16):
    ys, xs = np.where(mask)
    h, w = mask.shape
    if len(xs) == 0:
        return 0, 0, w, h
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    return x1, y1, x2, y2


                               
                  
                               
class FID:
    def __init__(self, device):
        self.device = device
        self.net = inception_v3(pretrained=True, transform_input=False)
        self.net.fc = torch.nn.Identity()
        self.net.to(device)
        self.net.eval()

        self.real_feats = []
        self.fake_feats = []

    @torch.no_grad()
    def extract(self, img):
        x = torch.from_numpy(img).permute(2, 0, 1).float()
        x = F.interpolate(x.unsqueeze(0), (299, 299), mode="bilinear", align_corners=False)
        x = x / 255.0
        x = x.to(self.device)
        feat = self.net(x).cpu().numpy()
        return feat

    def add(self, real, fake):
        self.real_feats.append(self.extract(real))
        self.fake_feats.append(self.extract(fake))

    def compute(self):
        real = np.concatenate(self.real_feats, axis=0)
        fake = np.concatenate(self.fake_feats, axis=0)

        mu1, mu2 = real.mean(0), fake.mean(0)
        sigma1, sigma2 = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)

        covmean = sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)


                               
      
                               
def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"

    df = pd.read_csv(args.results_csv)

    if args.output_col not in df.columns:
        raise ValueError(f"Missing column: {args.output_col}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_csv).parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    fid = FID(device=device)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        clean = load_img(row["clean_path"])
        pred = load_img(row[args.output_col], size=(clean.shape[1], clean.shape[0]))
        mask = load_img(row["mask_path"], size=(clean.shape[1], clean.shape[0]))[:, :, 0] > 127

        p = psnr(clean, pred)
        m = mask_psnr(clean, pred, mask)
        a = mae(clean, pred)

        x1, y1, x2, y2 = bbox(mask)
        c_crop = clean[y1:y2, x1:x2]
        p_crop = pred[y1:y2, x1:x2]

        fid.add(clean, pred)

        results.append({
            "psnr": p,
            "mask_psnr": m,
            "mae": a,
            "bbox_psnr": psnr(c_crop, p_crop)
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_dir / "per_sample.csv", index=False)

    fid_score = fid.compute()

    summary = {
        "mean_psnr": float(df_out["psnr"].mean()),
        "mean_mask_psnr": float(df_out["mask_psnr"].mean()),
        "mean_mae": float(df_out["mae"].mean()),
        "mean_bbox_psnr": float(df_out["bbox_psnr"].mean()),
        "mean_full_ssim": float(metrics_df["full_ssim"].mean()),
        "mean_full_lpips": float(metrics_df["full_lpips"].mean()) if "full_lpips" in metrics_df else None,
        "FID": fid_score,
        "time": str(datetime.now())
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== RESULTS =====")
    print(summary)


if __name__ == "__main__":
    main()