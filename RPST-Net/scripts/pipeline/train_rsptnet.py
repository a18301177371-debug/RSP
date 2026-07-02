                     
                       

from __future__ import annotations

import argparse
import json
import math
import random
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEFAULT_PROJECT_ROOT = Path(r"<PROJECT_ROOT>")


                              
     
                              

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", type=str, default=str(DEFAULT_PROJECT_ROOT))
    p.add_argument("--package_tag", type=str, default="dinov2_normal_all_pseclean")
    p.add_argument("--roi_metadata_csv", type=str, default=None)
    p.add_argument("--output_dir", type=str, default=None)

    p.add_argument("--target_roi_types", type=str, default="face,head")
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--ref_size", type=int, default=224)
    p.add_argument("--style_ref_n", type=int, default=5)
    p.add_argument("--style_token_grids", type=str, default="8,4,2")

    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--stage1_epochs", type=int, default=100)
    p.add_argument("--gan_start_epoch", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--base_ch", type=int, default=48)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--d_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)

                                                           
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--cosine_t0", type=int, default=30)
    p.add_argument("--cosine_t_mult", type=int, default=2)
    p.add_argument("--eta_min_ratio", type=float, default=0.01)
    p.add_argument("--disable_lr_scheduler", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--max_infer_samples", type=int, default=0)
    p.add_argument("--infer_splits", type=str, default="test")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--train_only", action="store_true")
    p.add_argument("--infer_only", action="store_true")
    p.add_argument("--overwrite_outputs", action="store_true")

               
    p.add_argument("--disable_style_refs", action="store_true")
    p.add_argument("--disable_visible_edge", action="store_true")
    p.add_argument("--disable_semantic_mask", action="store_true")
    p.add_argument("--style_dropout_prob", type=float, default=0.05)

                                 
    p.add_argument("--style_strength", type=float, default=0.18)
    p.add_argument("--attention_strength", type=float, default=0.20)
    p.add_argument("--context_style_blend", type=float, default=0.55)

                  
    p.add_argument("--lambda_mask_l1", type=float, default=1.0)
    p.add_argument("--lambda_context", type=float, default=0.08)
    p.add_argument("--lambda_vgg", type=float, default=0.08)
    p.add_argument("--lambda_style_gram", type=float, default=0.06)
    p.add_argument("--lambda_dino", type=float, default=0.0)
    p.add_argument("--lambda_edge", type=float, default=0.18)
    p.add_argument("--lambda_lap", type=float, default=0.18)
    p.add_argument("--lambda_gan", type=float, default=0.002)
    p.add_argument("--lambda_residual", type=float, default=0.004)

                                 
    p.add_argument("--gan_mask_dilate_kernel", type=int, default=31,
                   help="PatchGAN loss is weighted by dilated damage mask. Use odd kernel, e.g. 31.")
    p.add_argument("--d_update_every", type=int, default=3,
                   help="Update discriminator once every N generator steps. 3 means G:G:G:D rhythm.")
    p.add_argument("--disable_masked_gan", action="store_true",
                   help="Use unmasked full-image GAN loss. Not recommended for small masks.")

                              
    p.add_argument("--val_lpips_every", type=int, default=10)
    p.add_argument("--val_lpips_max", type=int, default=24)
    p.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])

              
    p.add_argument("--vgg_no_pretrained", action="store_true")
    p.add_argument("--dino_model", type=str, default="dinov2_vits14")
    p.add_argument("--dino_hub_repo", type=str, default="facebookresearch/dinov2")

    p.add_argument("--mask_threshold", type=int, default=127)
    p.add_argument("--preview_rows", type=int, default=32)

                               
    p.add_argument("--preview_every", type=int, default=50,
                   help="Save one JPG preview from validation set every N epochs. 0 disables.")
    p.add_argument("--save_every", type=int, default=50,
                   help="Save checkpoint_XXX.pt every N epochs. 0 disables.")
    return p.parse_args()


                              
           
                              

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_flexible(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def norm_text(x) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    return str(x).strip()


def safe_name(x) -> str:
    s = norm_text(x) or "sample"
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "sample"


def as_float(x, default=0.0) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def load_rgb(path: str | Path, size: Tuple[int, int]) -> Image.Image:
    im = Image.open(path).convert("RGB")
    if im.size != size:
        im = im.resize(size, Image.Resampling.BICUBIC)
    return im


def load_l(path: str | Path, size: Tuple[int, int]) -> Image.Image:
    im = Image.open(path).convert("L")
    if im.size != size:
        im = im.resize(size, Image.Resampling.BILINEAR)
    return im


def pil_rgb_to_tensor(im: Image.Image) -> torch.Tensor:
    arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def pil_l_to_tensor(im: Image.Image, binarize: bool = False, threshold: float = 0.5) -> torch.Tensor:
    arr = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    if binarize:
        arr = (arr > threshold).astype(np.float32)
    return torch.from_numpy(arr)[None, :, :]


def tensor_to_pil_rgb(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().clamp(0, 1)
    arr = (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def psnr_np(a: np.ndarray, b: np.ndarray, max_val: float = 1.0) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def masked_psnr_np(a: np.ndarray, b: np.ndarray, mask: np.ndarray, max_val: float = 1.0) -> float:
    m = mask > 0.5
    if m.sum() == 0:
        return float("nan")
    mse = float(np.mean((a[m].astype(np.float32) - b[m].astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(max_val / math.sqrt(mse))


def set_optimizer_lr(optimizer: Optional[torch.optim.Optimizer], lr: float) -> None:
    if optimizer is None:
        return
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def get_optimizer_lr(optimizer: Optional[torch.optim.Optimizer]) -> float:
    if optimizer is None or not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def update_learning_rates(
    epoch: int,
    args: argparse.Namespace,
    opt_g: torch.optim.Optimizer,
    sch_g: Optional[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts],
    opt_d: Optional[torch.optim.Optimizer] = None,
    sch_d: Optional[torch.optim.lr_scheduler.CosineAnnealingWarmRestarts] = None,
) -> Dict[str, float]:
    """
    Epoch-level schedule:
    - epoch 1..warmup_epochs: linear warmup to base LR
    - after warmup: CosineAnnealingWarmRestarts starts at t=0
      with T_0=args.cosine_t0 and T_mult=args.cosine_t_mult.
    """
    if args.disable_lr_scheduler:
        return {"lr_g": get_optimizer_lr(opt_g), "lr_d": get_optimizer_lr(opt_d)}

    warmup = max(0, int(args.warmup_epochs))
    if warmup > 0 and epoch <= warmup:
        factor = max(1e-4, float(epoch) / float(warmup))
        set_optimizer_lr(opt_g, args.lr * factor)
        set_optimizer_lr(opt_d, args.d_lr * factor)
    else:
                                                       
        t = max(0, epoch - warmup - 1)
        if sch_g is not None:
            sch_g.step(t)
        if sch_d is not None:
            sch_d.step(t)

    return {"lr_g": get_optimizer_lr(opt_g), "lr_d": get_optimizer_lr(opt_d)}


def parse_grids(s: str) -> List[int]:
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if x:
            out.append(max(1, int(x)))
    return out or [8, 4, 2]


                              
                    
                              

def load_metadata(args: argparse.Namespace) -> pd.DataFrame:
    root = Path(args.project_root)
    csv_path = Path(args.roi_metadata_csv) if args.roi_metadata_csv else (
        root / "restoration_benchmark" / "roi_structure_benchmark" / args.package_tag / "metadata" / "roi_structure_benchmark_master.csv"
    )
    if not csv_path.exists():
        raise FileNotFoundError(f"ROI benchmark metadata not found: {csv_path}")
    df = read_csv_flexible(csv_path)
    target_rois = {x.strip().lower() for x in args.target_roi_types.split(",") if x.strip()}
    if target_rois and "roi_type" in df.columns:
        df = df[df["roi_type"].astype(str).str.lower().isin(target_rois)].copy()
        df = df.reset_index(drop=True)
    return df


class StyleContextROIDataset(Dataset):
    def __init__(self, df: pd.DataFrame, args: argparse.Namespace, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.args = args
        self.train = train
        self.size = (int(args.image_size), int(args.image_size))
        self.ref_size = (int(args.ref_size), int(args.ref_size))

    def __len__(self):
        return len(self.df)

    def _optional_l(self, path_text: str, disabled: bool = False, fill: int = 0) -> Image.Image:
        if disabled:
            return Image.new("L", self.size, fill)
        p = norm_text(path_text)
        if not p or not Path(p).exists():
            return Image.new("L", self.size, fill)
        return load_l(p, self.size)

    def _load_style_refs(self, row: pd.Series) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        imgs, scores, valid = [], [], []
        use_dropout = self.train and self.args.style_dropout_prob > 0 and random.random() < self.args.style_dropout_prob
        for i in range(1, int(self.args.style_ref_n) + 1):
            p = norm_text(row.get(f"style_ref_{i:02d}_ref_letterbox_path", ""))
            score = as_float(row.get(f"style_ref_{i:02d}_final_role_score", 0.0))
            style_score = as_float(row.get(f"style_ref_{i:02d}_style_priority_score", 0.0))
            if self.args.disable_style_refs or use_dropout or not p or not Path(p).exists():
                imgs.append(torch.zeros(3, self.args.ref_size, self.args.ref_size, dtype=torch.float32))
                scores.append([0.0, 0.0])
                valid.append(0.0)
                continue
            try:
                imgs.append(pil_rgb_to_tensor(load_rgb(p, self.ref_size)))
                scores.append([score, style_score])
                valid.append(1.0)
            except Exception:
                imgs.append(torch.zeros(3, self.args.ref_size, self.args.ref_size, dtype=torch.float32))
                scores.append([0.0, 0.0])
                valid.append(0.0)
        if len(valid) > 0 and sum(valid) == 0:
            valid[0] = 1.0
        return torch.stack(imgs, dim=0), torch.tensor(scores, dtype=torch.float32), torch.tensor(valid, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        clean = load_rgb(row["clean_roi_path"], self.size)
        damaged = load_rgb(row["damaged_roi_path"], self.size)
        mask = load_l(row["mask_roi_path"], self.size)
        visible = self._optional_l(row.get("visible_edge_roi_path", ""), disabled=self.args.disable_visible_edge)
        semantic = self._optional_l(row.get("semantic_mask_roi_path", ""), disabled=self.args.disable_semantic_mask)
        style_imgs, style_scores, style_valid = self._load_style_refs(row)

        mask_t = pil_l_to_tensor(mask, binarize=True)
        x = torch.cat([
            pil_rgb_to_tensor(damaged),
            mask_t,
            pil_l_to_tensor(visible),
            pil_l_to_tensor(semantic),
        ], dim=0)              

        return {
            "input": x,
            "clean": pil_rgb_to_tensor(clean),
            "damaged": pil_rgb_to_tensor(damaged),
            "mask": mask_t,
            "visible_edge": pil_l_to_tensor(visible),
            "style_imgs": style_imgs,
            "style_scores": style_scores,
            "style_valid": style_valid,
            "sample_id": str(row.get("sample_id", idx)),
            "split": str(row.get("split", "")),
            "row_json": json.dumps(row.to_dict(), ensure_ascii=False),
        }


                              
                            
                              

def try_load_vgg19(pretrained: bool = True) -> nn.Module:
    try:
        from torchvision.models import vgg19, VGG19_Weights
        weights = VGG19_Weights.DEFAULT if pretrained else None
        return vgg19(weights=weights).features
    except Exception as e:
        warnings.warn(f"Could not load pretrained VGG19 weights ({e}); falling back to untrained VGG19.")
        try:
            from torchvision.models import vgg19
            return vgg19(weights=None).features
        except TypeError:
            from torchvision.models import vgg19
            return vgg19(pretrained=False).features


class VGGFeatureExtractor(nn.Module):
    """
    Frozen VGG feature slices. Returns relu1_2, relu2_2, relu3_4-like features.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        features = try_load_vgg19(pretrained=pretrained)
        self.slice1 = nn.Sequential(*[features[i] for i in range(0, 4)])
        self.slice2 = nn.Sequential(*[features[i] for i in range(4, 9)])
        self.slice3 = nn.Sequential(*[features[i] for i in range(9, 18)])
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean.to(x.device)) / self.std.to(x.device)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.norm(x)
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        return [f1, f2, f3]


class DinoFeatureLoss(nn.Module):
    def __init__(self, repo: str, model_name: str, device: str):
        super().__init__()
        self.enabled = False
        self.model = None
        try:
            model = torch.hub.load(repo, model_name, pretrained=True)
            model.eval().to(device)
            for p in model.parameters():
                p.requires_grad = False
            self.model = model
            self.enabled = True
            print(f"[INFO] Loaded DINO model from torch.hub: {repo} {model_name}")
        except Exception as e:
            warnings.warn(f"DINO model could not be loaded. DINO loss disabled. Error: {e}")
            self.enabled = False
            self.model = None
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229,0.224,0.225]).view(1,3,1,1), persistent=False)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.model is None:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        x224 = F.interpolate(x.float(), size=(224,224), mode="bilinear", align_corners=False)
        y224 = F.interpolate(y.float(), size=(224,224), mode="bilinear", align_corners=False)
        x224 = (x224 - self.mean.to(x.device)) / self.std.to(x.device)
        y224 = (y224 - self.mean.to(y.device)) / self.std.to(y.device)
        fx = self.model(x224)
        fy = self.model(y224)
        if isinstance(fx, dict):
                                                               
            fx = next(iter(fx.values()))
            fy = next(iter(fy.values()))
        return F.l1_loss(fx.float(), fy.float())


                              
                    
                              

class ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        groups = 8 if out_ch >= 8 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(groups, out_ch),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.AvgPool2d(2), ConvGNAct(in_ch, out_ch))
    def forward(self, x):
        return self.net(x)


class UpBilinear(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvGNAct(in_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class VGGMultiScaleStyleEncoder(nn.Module):
    """
    Encodes style references with frozen VGG and trainable projections.
    Returns:
      tokens: B,N,C
      key_padding_mask: B,N bool
      style_vec: B,C
    """
    def __init__(self, token_dim: int, grids: List[int], pretrained: bool = True):
        super().__init__()
        self.vgg = VGGFeatureExtractor(pretrained=pretrained)
        self.grids = grids
        feat_dims = [64, 128, 256]
        self.proj = nn.ModuleList([nn.Linear(d, token_dim) for d in feat_dims])
        self.score_mlp = nn.Sequential(nn.Linear(2, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, token_dim))
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, refs: torch.Tensor, scores: torch.Tensor, valid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                         
        b, k, c, h, w = refs.shape
        x = refs.reshape(b*k, c, h, w)

        with torch.no_grad():
            feats = self.vgg(x.float())

        all_tokens = []
        token_counts = []
        for fi, feat in enumerate(feats):
            grid = self.grids[min(fi, len(self.grids)-1)]
            pooled = F.adaptive_avg_pool2d(feat, (grid, grid))
            pooled = pooled.flatten(2).transpose(1, 2)          
            tok = self.proj[fi](pooled)                         
            t = tok.shape[1]
            tok = tok.reshape(b, k, t, -1)
            all_tokens.append(tok)
            token_counts.append(t)

        tokens = torch.cat(all_tokens, dim=2)                       
        t_sum = tokens.shape[2]
        score_bias = self.score_mlp(scores).unsqueeze(2)
        tokens = tokens + score_bias
        tokens = tokens.reshape(b, k*t_sum, -1)
        tokens = self.norm(tokens)

        valid_tokens = valid.repeat_interleave(t_sum, dim=1)
        key_padding_mask = valid_tokens < 0.5
        all_bad = key_padding_mask.all(dim=1)
        if all_bad.any():
            key_padding_mask[all_bad, 0] = False

                                       
        tok_by_ref = tokens.reshape(b, k, t_sum, -1).mean(dim=2)
        weights = valid / valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        style_vec = (tok_by_ref * weights.unsqueeze(-1)).sum(dim=1)
        return tokens, key_padding_mask, style_vec


class StyleReferenceEncoderV2UNet(nn.Module):
    def __init__(self, in_ch: int = 6, base_ch: int = 48, num_heads: int = 8, style_grids: List[int] = [8,4,2],
                 style_strength: float = 0.18, attention_strength: float = 0.20, vgg_pretrained: bool = True):
        super().__init__()
        c = int(base_ch)
        self.token_dim = c * 8
        self.style_strength = float(style_strength)
        self.attention_strength = float(attention_strength)

        self.e0 = ConvGNAct(in_ch, c)
        self.e1 = Down(c, c*2)
        self.e2 = Down(c*2, c*4)
        self.e3 = Down(c*4, c*8)
        self.bot = ConvGNAct(c*8, c*8)

        self.style_encoder = VGGMultiScaleStyleEncoder(self.token_dim, grids=style_grids, pretrained=vgg_pretrained)
        self.query_norm = nn.LayerNorm(self.token_dim)
        self.style_norm = nn.LayerNorm(self.token_dim)
        self.cross_attn = nn.MultiheadAttention(self.token_dim, num_heads=num_heads, batch_first=True)

        self.film8 = nn.Linear(self.token_dim, c*8*2)
        self.film4 = nn.Linear(self.token_dim, c*4*2)
        self.film2 = nn.Linear(self.token_dim, c*2*2)
        self.film1 = nn.Linear(self.token_dim, c*1*2)
        self.attn_gate = nn.Sequential(
            nn.Linear(self.token_dim*2 + 1, self.token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.token_dim, self.token_dim),
            nn.Sigmoid(),
        )

        self.u3 = UpBilinear(c*8, c*4, c*4)
        self.u2 = UpBilinear(c*4, c*2, c*2)
        self.u1 = UpBilinear(c*2, c, c)
        self.coarse_head = nn.Conv2d(c, 3, 1)

        refine_in = in_ch + 3
        self.r0 = ConvGNAct(refine_in, c)
        self.r1 = Down(c, c*2)
        self.r2 = Down(c*2, c*4)
        self.rb = ConvGNAct(c*4, c*4)
        self.ru2 = UpBilinear(c*4, c*2, c*2)
        self.ru1 = UpBilinear(c*2, c, c)
        self.res_head = nn.Conv2d(c, 3, 1)

    def apply_film(self, feat: torch.Tensor, style_vec: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        b, c, _, _ = feat.shape
        gb = proj(style_vec)
        gamma, beta = gb.chunk(2, dim=-1)
        gamma = torch.tanh(gamma).view(b, c, 1, 1)
        beta = torch.tanh(beta).view(b, c, 1, 1)
        return feat * (1.0 + self.style_strength * gamma) + self.style_strength * beta

    def forward(self, x: torch.Tensor, mask: torch.Tensor, style_imgs: torch.Tensor, style_scores: torch.Tensor, style_valid: torch.Tensor) -> Dict[str, torch.Tensor]:
        s0 = self.e0(x)
        s1 = self.e1(s0)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        bot = self.bot(s3)
        b, c, h, w = bot.shape

        style_tokens, key_mask, style_vec = self.style_encoder(style_imgs, style_scores, style_valid)
        style_tokens = style_tokens.to(dtype=bot.dtype)
        style_vec = style_vec.to(dtype=bot.dtype)
        style_tokens = self.style_norm(style_tokens)

        q = bot.flatten(2).transpose(1, 2)
        qn = self.query_norm(q)
        attn, _ = self.cross_attn(qn, style_tokens, style_tokens, key_padding_mask=key_mask, need_weights=False)

        mask_small = F.interpolate(mask, size=(h,w), mode="nearest")
        mask_tok = mask_small.flatten(2).transpose(1, 2)
        gate = self.attn_gate(torch.cat([qn, attn, mask_tok], dim=-1)) * mask_tok
        fused = q + self.attention_strength * gate * attn
        bot = fused.transpose(1, 2).reshape(b, c, h, w)
        bot = self.apply_film(bot, style_vec, self.film8)

        y = self.u3(bot, s2)
        y = self.apply_film(y, style_vec, self.film4)
        y = self.u2(y, s1)
        y = self.apply_film(y, style_vec, self.film2)
        y = self.u1(y, s0)
        y = self.apply_film(y, style_vec, self.film1)

        coarse = torch.sigmoid(self.coarse_head(y))
        r_in = torch.cat([x, coarse], dim=1)
        r0 = self.r0(r_in)
        r1 = self.r1(r0)
        r2 = self.r2(r1)
        rb = self.rb(r2)
        rr = self.ru2(rb, r1)
        rr = self.ru1(rr, r0)
        residual = torch.tanh(self.res_head(rr)) * 0.12
        refined = torch.clamp(coarse + residual, 0.0, 1.0)

        return {
            "coarse": coarse,
            "residual": residual,
            "refined": refined,
            "style_vec_norm": style_vec.norm(dim=-1).mean(),
            "style_gate_mean": gate.mean(),
        }


class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base*2, 4, 2, 1),
            nn.GroupNorm(8, base*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*2, base*4, 4, 2, 1),
            nn.GroupNorm(8, base*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*4, base*4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*4, 1, 3, 1, 1),
        )
    def forward(self, x):
        return self.net(x)


                              
        
                              

_sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
_sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
_lap = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32).view(1,1,3,3)

def rgb_to_gray(x):
    return 0.299*x[:,0:1] + 0.587*x[:,1:2] + 0.114*x[:,2:3]

def sobel_mag(x):
    g = rgb_to_gray(x)
    sx = F.conv2d(g, _sobel_x.to(x.device, x.dtype), padding=1)
    sy = F.conv2d(g, _sobel_y.to(x.device, x.dtype), padding=1)
    return torch.sqrt(sx*sx + sy*sy + 1e-6)

def laplacian(x):
    g = rgb_to_gray(x)
    return F.conv2d(g, _lap.to(x.device, x.dtype), padding=1)

def mask_context(mask, k=31):
    if k % 2 == 0:
        k += 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=k//2)


def set_requires_grad(module: Optional[nn.Module], flag: bool) -> None:
    if module is None:
        return
    for p in module.parameters():
        p.requires_grad_(flag)


def gan_weight_from_mask(mask: torch.Tensor, logits: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """
    Convert Bx1xHxW damage mask to discriminator-logit resolution.
    For small masks, direct downsampling can erase the mask; therefore we first
    dilate the mask in image space and then resize it to the PatchGAN map.
    """
    if args.disable_masked_gan:
        return torch.ones_like(logits, dtype=logits.dtype, device=logits.device)

    k = int(args.gan_mask_dilate_kernel)
    if k < 3:
        m = mask
    else:
        if k % 2 == 0:
            k += 1
        m = F.max_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)

    w = F.interpolate(m, size=logits.shape[-2:], mode="nearest")
    w = (w > 0.01).to(dtype=logits.dtype, device=logits.device)

                                                                                
                                                                 
    if float(w.sum().detach().cpu()) < 1.0:
        soft = F.interpolate(m, size=logits.shape[-2:], mode="bilinear", align_corners=False)
        flat = soft.reshape(soft.shape[0], -1)
        idx = flat.argmax(dim=1)
        w = torch.zeros_like(soft, dtype=logits.dtype, device=logits.device)
        for bi in range(w.shape[0]):
            w.reshape(w.shape[0], -1)[bi, idx[bi]] = 1.0
    return w.expand_as(logits)


def weighted_mean(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (x * weight).sum() / weight.sum().clamp_min(1.0)


def masked_hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    real_loss = weighted_mean(F.relu(1.0 - real_logits), weight)
    fake_loss = weighted_mean(F.relu(1.0 + fake_logits), weight)
    return real_loss + fake_loss


def masked_hinge_g_loss(fake_logits: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return weighted_mean(-fake_logits, weight)


def masked_charbonnier(pred, target, mask, eps=1e-3):
    m = mask.expand_as(pred)
    return (torch.sqrt((pred-target)**2 + eps*eps) * m).sum() / m.sum().clamp_min(1.0)

def feature_masked_l1(fx, fy, mask):
    m = F.interpolate(mask, size=fx.shape[-2:], mode="nearest")
    return (torch.abs(fx-fy) * m).sum() / (m.sum() * fx.shape[1]).clamp_min(1.0)

def gram_matrix(feat):
    b, c, h, w = feat.shape
    f = feat.reshape(b, c, h*w)
    return torch.bmm(f, f.transpose(1,2)) / float(c*h*w)

def masked_mean_std(feat, mask, eps=1e-6):
    m = F.interpolate(mask, size=feat.shape[-2:], mode="nearest")
    denom = m.sum(dim=(2,3), keepdim=True).clamp_min(eps)
    mean = (feat * m).sum(dim=(2,3), keepdim=True) / denom
    var = (((feat - mean) ** 2) * m).sum(dim=(2,3), keepdim=True) / denom
    return mean.squeeze(-1).squeeze(-1), torch.sqrt(var.squeeze(-1).squeeze(-1) + eps)

def vgg_perceptual_style_losses(vgg_loss_net, final, clean, damaged, mask, style_imgs, style_valid, blend=0.55):
                                                          
    feats_f = vgg_loss_net(final)
    feats_c = vgg_loss_net(clean)
    feats_d = vgg_loss_net(damaged)

                         
    b,k,c,h,w = style_imgs.shape
    refs_flat = style_imgs.reshape(b*k, c, h, w)
    feats_r_flat = vgg_loss_net(refs_flat)

    percept = 0.0
    style = 0.0
    ctx = (mask_context(mask, 31) - mask).clamp(0,1)
    if float(ctx.sum().detach().cpu()) < 1.0:
        ctx = (1.0-mask).clamp(0,1)
    weights = style_valid / style_valid.sum(dim=1, keepdim=True).clamp_min(1.0)

    for li, (ff, fc, fd, fr_flat) in enumerate(zip(feats_f, feats_c, feats_d, feats_r_flat)):
        percept = percept + feature_masked_l1(ff, fc, mask)

                            
        mf, sf = masked_mean_std(ff, mask)
        md, sd = masked_mean_std(fd, ctx)

                         
        brk, ch, hh, ww = fr_flat.shape
        fr = fr_flat.reshape(b, k, ch, hh, ww)
        mr = fr.mean(dim=(3,4))         
        sr = fr.std(dim=(3,4))          
        mr_w = (mr * weights.unsqueeze(-1)).sum(dim=1)
        sr_w = (sr * weights.unsqueeze(-1)).sum(dim=1)

        target_m = blend * md + (1-blend) * mr_w
        target_s = blend * sd + (1-blend) * sr_w
        style = style + F.l1_loss(mf, target_m) + F.l1_loss(sf, target_s)

    return percept / len(feats_f), style / len(feats_f)

def get_stage_weights(epoch: int, args: argparse.Namespace) -> Dict[str, float]:
                                                                
    if epoch <= int(args.stage1_epochs):
        return {
            "edge": 0.0,
            "lap": 0.0,
            "gan": 0.0,
            "style_gram": 0.5 * args.lambda_style_gram,
        }
    return {
        "edge": args.lambda_edge,
        "lap": args.lambda_lap,
        "gan": args.lambda_gan if epoch >= int(args.gan_start_epoch) else 0.0,
        "style_gram": args.lambda_style_gram,
    }

def compute_g_loss_core(out, clean, damaged, mask, visible_edge, style_imgs, style_valid, args, stage_w, vgg_loss_net=None, dino_loss_net=None):
    pred = out["refined"]
    coarse = out["coarse"]
    final = pred * mask + damaged * (1-mask)
    coarse_final = coarse * mask + damaged * (1-mask)

    l_mask = masked_charbonnier(final, clean, mask)
    l_coarse = 0.20 * masked_charbonnier(coarse_final, clean, mask)

    ctx = mask_context(mask, 31)
    l_ctx = (torch.abs(final-clean) * ctx.expand_as(final)).sum() / ctx.expand_as(final).sum().clamp_min(1.0)

    l_edge = (torch.abs(sobel_mag(final)-sobel_mag(clean)) * mask_context(mask, 9)).sum() / mask_context(mask, 9).sum().clamp_min(1.0)
    l_lap = (torch.abs(laplacian(final)-laplacian(clean)) * mask_context(mask, 9)).sum() / mask_context(mask, 9).sum().clamp_min(1.0)

    l_vgg = torch.tensor(0.0, device=clean.device, dtype=clean.dtype)
    l_style = torch.tensor(0.0, device=clean.device, dtype=clean.dtype)
    if vgg_loss_net is not None and (args.lambda_vgg > 0 or args.lambda_style_gram > 0):
        l_vgg, l_style = vgg_perceptual_style_losses(
            vgg_loss_net, final, clean, damaged, mask, style_imgs, style_valid, blend=args.context_style_blend
        )
        l_vgg = l_vgg.to(dtype=clean.dtype)
        l_style = l_style.to(dtype=clean.dtype)

    l_dino = torch.tensor(0.0, device=clean.device, dtype=clean.dtype)
    if dino_loss_net is not None and args.lambda_dino > 0 and getattr(dino_loss_net, "enabled", False):
        l_dino = dino_loss_net(final, clean).to(dtype=clean.dtype)

    l_res = torch.mean(torch.abs(out["residual"]) * mask.expand_as(out["residual"]))

    loss = (
        args.lambda_mask_l1 * (l_mask + l_coarse)
        + args.lambda_context * l_ctx
        + args.lambda_vgg * l_vgg
        + stage_w["style_gram"] * l_style
        + args.lambda_dino * l_dino
        + stage_w["edge"] * l_edge
        + stage_w["lap"] * l_lap
        + args.lambda_residual * l_res
    )

    logs = {
        "mask_l1": float(l_mask.detach().cpu()),
        "context": float(l_ctx.detach().cpu()),
        "vgg": float(l_vgg.detach().cpu()),
        "style": float(l_style.detach().cpu()),
        "dino": float(l_dino.detach().cpu()),
        "edge": float(l_edge.detach().cpu()),
        "lap": float(l_lap.detach().cpu()),
        "res": float(l_res.detach().cpu()),
        "style_norm": float(out["style_vec_norm"].detach().cpu()),
        "style_gate": float(out["style_gate_mean"].detach().cpu()),
    }
    return loss, final, logs


@torch.no_grad()
def validate(model, loader, device, args, lpips_model=None, lpips_limit=0):
    model.eval()
    fpsnrs, mpsnrs, lpips_vals, norms, gates = [], [], [], [], []
    count_lpips = 0
    for batch in loader:
        x = batch["input"].to(device)
        clean = batch["clean"].to(device)
        damaged = batch["damaged"].to(device)
        mask = batch["mask"].to(device)
        style_imgs = batch["style_imgs"].to(device)
        style_scores = batch["style_scores"].to(device)
        style_valid = batch["style_valid"].to(device)

        out = model(x, mask, style_imgs, style_scores, style_valid)
        final = out["refined"] * mask + damaged * (1-mask)
        norms.append(float(out["style_vec_norm"].detach().cpu()))
        gates.append(float(out["style_gate_mean"].detach().cpu()))

        for i in range(final.shape[0]):
            o = final[i].detach().cpu().permute(1,2,0).numpy()
            c = clean[i].detach().cpu().permute(1,2,0).numpy()
            m = mask[i,0].detach().cpu().numpy()
            fpsnrs.append(psnr_np(c, o, 1.0))
            mpsnrs.append(masked_psnr_np(c, o, m, 1.0))

        if lpips_model is not None and (lpips_limit <= 0 or count_lpips < lpips_limit):
            vals = lpips_model(final * 2 - 1, clean * 2 - 1).detach().cpu().view(-1).numpy().tolist()
            for v in vals:
                if lpips_limit <= 0 or count_lpips < lpips_limit:
                    lpips_vals.append(float(v))
                    count_lpips += 1

    out = {
        "val_full_psnr": float(np.mean(fpsnrs)) if fpsnrs else float("nan"),
        "val_mask_psnr": float(np.nanmean(mpsnrs)) if mpsnrs else float("nan"),
        "val_style_vec_norm": float(np.mean(norms)) if norms else 0.0,
        "val_style_gate": float(np.mean(gates)) if gates else 0.0,
    }
    if lpips_model is not None:
        out["val_full_lpips"] = float(np.mean(lpips_vals)) if lpips_vals else float("nan")
    return out


                              
                               
                              

def tensor_l_to_pil_rgb(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().float().clamp(0, 1)
    if t.ndim == 3 and t.shape[0] == 1:
        arr = (t[0].numpy() * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")
    if t.ndim == 2:
        arr = (t.numpy() * 255.0 + 0.5).astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")
    return tensor_to_pil_rgb(t)


def make_style_strip_from_tensor(style_imgs: torch.Tensor, style_valid: torch.Tensor, tile: int = 128) -> Image.Image:
                         
    imgs = []
    k = style_imgs.shape[0]
    for i in range(k):
        valid = bool(float(style_valid[i].detach().cpu()) > 0.5) if i < style_valid.shape[0] else False
        if valid:
            im = tensor_to_pil_rgb(style_imgs[i]).resize((tile, tile), Image.Resampling.BICUBIC)
        else:
            im = Image.new("RGB", (tile, tile), (245, 245, 245))
        imgs.append(im)
    if not imgs:
        imgs = [Image.new("RGB", (tile, tile), (245, 245, 245))]
    strip = Image.new("RGB", (tile * len(imgs), tile), "white")
    for i, im in enumerate(imgs):
        strip.paste(im, (i * tile, 0))
    return strip


@torch.no_grad()
def save_training_preview(model, preview_batch, device, args, output_dir: Path, epoch: int) -> Path:
    model.eval()
    x = preview_batch["input"].to(device)
    clean = preview_batch["clean"].to(device)
    damaged = preview_batch["damaged"].to(device)
    mask = preview_batch["mask"].to(device)
    style_imgs = preview_batch["style_imgs"].to(device)
    style_scores = preview_batch["style_scores"].to(device)
    style_valid = preview_batch["style_valid"].to(device)

    out = model(x, mask, style_imgs, style_scores, style_valid)
    final = out["refined"] * mask + damaged * (1 - mask)

    clean_img = tensor_to_pil_rgb(clean[0])
    damaged_img = tensor_to_pil_rgb(damaged[0])
    mask_img = tensor_l_to_pil_rgb(mask[0])
    output_img = tensor_to_pil_rgb(final[0])
    coarse_img = tensor_to_pil_rgb((out["coarse"] * mask + damaged * (1 - mask))[0])
    refs_img = make_style_strip_from_tensor(style_imgs[0].detach().cpu(), style_valid[0].detach().cpu())
    refs_img = refs_img.resize((clean_img.width, clean_img.height), Image.Resampling.BICUBIC)

    w, h = clean_img.size
    header_h = 34
    items = [
        ("clean", clean_img),
        ("damaged", damaged_img),
        ("mask", mask_img),
        ("style_refs", refs_img),
        ("coarse", coarse_img),
        ("epoch_output", output_img),
    ]
    canvas = Image.new("RGB", (w * len(items), h + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (lab, im) in enumerate(items):
        canvas.paste(im.resize((w, h)), (i * w, header_h))
        draw.text((i * w + 5, 8), lab, fill=(0, 0, 0))
    sid = preview_batch.get("sample_id", ["sample"])[0]
    draw.text((6, h + header_h - 18), f"epoch={epoch} | sample={sid}", fill=(0, 0, 0))

    preview_dir = output_dir / "training_previews"
    ensure_dir(preview_dir)
    out_path = preview_dir / f"epoch_{epoch:03d}_preview.jpg"
    canvas.save(out_path, quality=92)
    return out_path




                              
                   
                              

def train(args, df, output_dir, device):
    train_df = df[df["split"].astype(str) == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"].astype(str) == "val"].copy().reset_index(drop=True)
    if args.max_train_samples > 0:
        train_df = train_df.head(args.max_train_samples)
    if args.max_val_samples > 0:
        val_df = val_df.head(args.max_val_samples)
    if train_df.empty:
        raise ValueError("No train samples found. Check package_tag / target_roi_types.")
    if val_df.empty:
        print("[WARN] No val samples found; using train subset.")
        val_df = train_df.head(min(32, len(train_df))).copy()

    train_loader = DataLoader(StyleContextROIDataset(train_df, args, train=True), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(StyleContextROIDataset(val_df, args, train=False), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"))

                                                                  
    preview_batch = None
    try:
        preview_ds = StyleContextROIDataset(val_df.head(1).copy(), args, train=False)
        preview_loader = DataLoader(preview_ds, batch_size=1, shuffle=False, num_workers=0)
        preview_batch = next(iter(preview_loader))
    except Exception as e:
        print(f"[WARN] Could not create training preview batch: {e}")

    style_grids = parse_grids(args.style_token_grids)
    model = StyleReferenceEncoderV2UNet(
        in_ch=6,
        base_ch=args.base_ch,
        num_heads=args.num_heads,
        style_grids=style_grids,
        style_strength=args.style_strength,
        attention_strength=args.attention_strength,
        vgg_pretrained=not args.vgg_no_pretrained,
    ).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)

    discriminator = PatchDiscriminator(in_ch=3, base=max(32, args.base_ch)).to(device) if args.lambda_gan > 0 else None
    opt_g = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=args.d_lr, weight_decay=args.weight_decay) if discriminator is not None else None

    sch_g = None
    sch_d = None
    if not args.disable_lr_scheduler:
        sch_g = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt_g,
            T_0=int(args.cosine_t0),
            T_mult=int(args.cosine_t_mult),
            eta_min=float(args.lr) * float(args.eta_min_ratio),
        )
        if opt_d is not None:
            sch_d = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt_d,
                T_0=int(args.cosine_t0),
                T_mult=int(args.cosine_t_mult),
                eta_min=float(args.d_lr) * float(args.eta_min_ratio),
            )

    scaler_g = torch.cuda.amp.GradScaler(enabled=(args.amp and device == "cuda"))
    scaler_d = torch.cuda.amp.GradScaler(enabled=(args.amp and device == "cuda"))

                               
    vgg_loss_net = VGGFeatureExtractor(pretrained=not args.vgg_no_pretrained).to(device).eval() if (args.lambda_vgg > 0 or args.lambda_style_gram > 0) else None
    dino_loss_net = DinoFeatureLoss(args.dino_hub_repo, args.dino_model, device) if args.lambda_dino > 0 else None

    lpips_val = None
    if args.val_lpips_every and args.val_lpips_every > 0:
        import lpips
        lpips_val = lpips.LPIPS(net=args.lpips_net).to(device).eval()

    ckpt_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ensure_dir(ckpt_dir); ensure_dir(log_dir)
    best = -1e9
    best_path = ckpt_dir / "best_checkpoint.pt"
    logs = []

    print(f"[INFO] Train={len(train_df)} Val={len(val_df)} epochs={args.epochs} stage1={args.stage1_epochs} batch={args.batch_size}")
    print(f"[INFO] VGG multi-scale style tokens. grids={style_grids}")
    print(f"[INFO] Stage 1: L1 + perceptual/style. Stage 2: edge/lap + GAN after epoch {args.gan_start_epoch}.")
    if args.disable_lr_scheduler:
        print("[INFO] LR scheduler disabled.")
    else:
        print(f"[INFO] LR schedule: warmup={args.warmup_epochs}, cosine T_0={args.cosine_t0}, T_mult={args.cosine_t_mult}, eta_min_ratio={args.eta_min_ratio}")
    if args.lambda_gan > 0:
        print(f"[INFO] Masked GAN: enabled={not args.disable_masked_gan}, lambda_gan={args.lambda_gan}, d_update_every={args.d_update_every}, gan_mask_dilate_kernel={args.gan_mask_dilate_kernel}")
    print(f"[INFO] Training preview: every {args.preview_every} epochs; periodic checkpoint: every {args.save_every} epochs.")
    if dino_loss_net is not None:
        print(f"[INFO] DINO loss enabled: {getattr(dino_loss_net, 'enabled', False)}")

    global_g_step = 0

    for ep in range(1, args.epochs + 1):
        lr_state = update_learning_rates(ep, args, opt_g, sch_g, opt_d, sch_d)
        model.train()
        if discriminator is not None:
            discriminator.train()
        stage_w = get_stage_weights(ep, args)
        ep_logs = []
        for batch in tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}", leave=False):
            global_g_step += 1
            x = batch["input"].to(device, non_blocking=True)
            clean = batch["clean"].to(device, non_blocking=True)
            damaged = batch["damaged"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            visible = batch["visible_edge"].to(device, non_blocking=True)
            style_imgs = batch["style_imgs"].to(device, non_blocking=True)
            style_scores = batch["style_scores"].to(device, non_blocking=True)
            style_valid = batch["style_valid"].to(device, non_blocking=True)

            opt_g.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(args.amp and device == "cuda")):
                out = model(x, mask, style_imgs, style_scores, style_valid)
                g_loss, final, g_logs = compute_g_loss_core(
                    out, clean, damaged, mask, visible, style_imgs, style_valid, args, stage_w,
                    vgg_loss_net=vgg_loss_net,
                    dino_loss_net=dino_loss_net,
                )

            d_loss_val = 0.0
            g_gan_val = 0.0
            gan_weight_coverage = 0.0
            d_updated = 0.0
            if discriminator is not None and stage_w["gan"] > 0:
                                                                                    
                                                                                      
                real_comp = clean * mask + damaged * (1.0 - mask)

                                                         
                do_d_update = (global_g_step % max(1, int(args.d_update_every)) == 0)
                if do_d_update:
                    set_requires_grad(discriminator, True)
                    opt_d.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=(args.amp and device == "cuda")):
                        real_logits = discriminator(real_comp)
                        fake_logits = discriminator(final.detach())
                        d_weight = gan_weight_from_mask(mask, real_logits, args)
                        d_loss = masked_hinge_d_loss(real_logits, fake_logits, d_weight)
                    scaler_d.scale(d_loss).backward()
                    scaler_d.step(opt_d)
                    scaler_d.update()
                    d_loss_val = float(d_loss.detach().cpu())
                    gan_weight_coverage = float(d_weight.detach().mean().cpu())
                    d_updated = 1.0

                                                                                                                
                set_requires_grad(discriminator, False)
                with torch.cuda.amp.autocast(enabled=(args.amp and device == "cuda")):
                    fake_logits_for_g = discriminator(final)
                    g_weight = gan_weight_from_mask(mask, fake_logits_for_g, args)
                    g_adv = masked_hinge_g_loss(fake_logits_for_g, g_weight)
                    g_loss = g_loss + float(stage_w["gan"]) * g_adv
                set_requires_grad(discriminator, True)

                g_gan_val = float(g_adv.detach().cpu())
                if gan_weight_coverage == 0.0:
                    gan_weight_coverage = float(g_weight.detach().mean().cpu())

            scaler_g.scale(g_loss).backward()
            scaler_g.step(opt_g)
            scaler_g.update()

            g_logs.update({
                "loss": float(g_loss.detach().cpu()),
                "d_loss": d_loss_val,
                "g_gan": g_gan_val,
                "gan_mask_coverage": gan_weight_coverage,
                "d_updated": d_updated,
            })
            ep_logs.append(g_logs)

        do_lpips = lpips_val is not None and (ep % int(args.val_lpips_every) == 0 or ep == args.epochs)
        val = validate(model, val_loader, device, args, lpips_model=lpips_val if do_lpips else None, lpips_limit=args.val_lpips_max)

        avg = {k: float(np.mean([r.get(k, 0.0) for r in ep_logs])) for k in ep_logs[0].keys()} if ep_logs else {}
        row = {"epoch": ep, "lr_g": lr_state["lr_g"], "lr_d": lr_state["lr_d"], **{f"train_{k}": v for k, v in avg.items()}, **val}
        logs.append(row)
        pd.DataFrame(logs).to_csv(log_dir / "train_log.csv", index=False, encoding="utf-8-sig")

        msg = (
            f"[EPOCH {ep:03d}] loss={row.get('train_loss', 0):.5f} "
            f"mask_psnr={val['val_mask_psnr']:.3f} full_psnr={val['val_full_psnr']:.3f} "
            f"style_gate={val['val_style_gate']:.4f} lr={lr_state['lr_g']:.2e}"
        )
        if "val_full_lpips" in val:
            msg += f" lpips={val['val_full_lpips']:.4f}"
        if stage_w["gan"] > 0:
            msg += (
                f" gan={avg.get('g_gan', 0.0):.4f} d={avg.get('d_loss', 0.0):.4f}"
                f" d_upd={avg.get('d_updated', 0.0):.2f} gan_cov={avg.get('gan_mask_coverage', 0.0):.4f}"
            )
        print(msg)

        score = val["val_mask_psnr"] + 0.10 * val["val_full_psnr"]
        if "val_full_lpips" in val and not math.isnan(val["val_full_lpips"]):
            score = score - 3.0 * val["val_full_lpips"]
        score += 0.20 * min(val["val_style_gate"], 0.10)
        ckpt_payload = {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": ep,
            "best_score": best,
            "current_score": score,
            "val_metrics": val,
        }

        if score > best:
            best = score
            ckpt_payload["best_score"] = best
            torch.save(ckpt_payload, best_path)

                                                                                         
        latest_path = ckpt_dir / "latest_checkpoint.pt"
        torch.save(ckpt_payload, latest_path)

        if args.save_every and args.save_every > 0 and (ep % int(args.save_every) == 0 or ep == args.epochs):
            epoch_path = ckpt_dir / f"checkpoint_{ep:03d}.pt"
            torch.save(ckpt_payload, epoch_path)

        if preview_batch is not None and args.preview_every and args.preview_every > 0 and (ep % int(args.preview_every) == 0 or ep == args.epochs):
            try:
                preview_path = save_training_preview(model, preview_batch, device, args, output_dir, ep)
                print(f"[PREVIEW] {preview_path}")
            except Exception as e:
                print(f"[WARN] Failed to save training preview at epoch {ep}: {e}")

    print(f"[DONE] Best checkpoint: {best_path} | best_score={best:.4f}")
    return best_path


def make_style_strip(paths: List[str], size=128) -> Image.Image:
    ims = []
    for p in paths:
        if p and Path(p).exists():
            try:
                ims.append(load_rgb(p, (size, size)))
                continue
            except Exception:
                pass
        ims.append(Image.new("RGB", (size, size), (245,245,245)))
    if not ims:
        return Image.new("RGB", (size, size), "white")
    strip = Image.new("RGB", (size * len(ims), size), "white")
    for i, im in enumerate(ims):
        strip.paste(im, (i * size, 0))
    return strip


def make_comparison(clean, damaged, mask, visible, style_refs, out, title):
    w, h = clean.size
    header_h = 34
    refs_vis = style_refs.resize((w, h), Image.Resampling.BICUBIC)
    items = [
        ("clean", clean),
        ("damaged", damaged),
        ("mask", mask.convert("RGB")),
        ("visible", visible.convert("RGB")),
        ("style_refs", refs_vis),
        ("rsptnet", out),
    ]
    canvas = Image.new("RGB", (w * len(items), h + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (lab, im) in enumerate(items):
        canvas.paste(im.resize((w, h)), (i * w, header_h))
        draw.text((i * w + 5, 8), lab, fill=(0, 0, 0))
    draw.text((6, h + header_h - 18), title[:180], fill=(0, 0, 0))
    return canvas


def make_preview_sheet(paths: List[Path], out_path: Path, max_rows: int):
    paths = paths[:max_rows]
    if not paths:
        return
    imgs = [Image.open(p).convert("RGB") for p in paths]
    max_w = max(im.width for im in imgs)
    total_h = sum(im.height for im in imgs)
    sheet = Image.new("RGB", (max_w, total_h), "white")
    y = 0
    for im in imgs:
        sheet.paste(im, (0, y))
        y += im.height
    sheet.save(out_path, quality=95)


@torch.no_grad()
def infer(args, df, output_dir, device, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    margs = ckpt.get("args", {})
    model = StyleReferenceEncoderV2UNet(
        in_ch=6,
        base_ch=int(margs.get("base_ch", args.base_ch)),
        num_heads=int(margs.get("num_heads", args.num_heads)),
        style_grids=parse_grids(margs.get("style_token_grids", args.style_token_grids)),
        style_strength=float(margs.get("style_strength", args.style_strength)),
        attention_strength=float(margs.get("attention_strength", args.attention_strength)),
        vgg_pretrained=not bool(margs.get("vgg_no_pretrained", args.vgg_no_pretrained)),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    splits = [s.strip() for s in args.infer_splits.split(",") if s.strip()]
    if "all" in splits:
        infer_df = df.copy().reset_index(drop=True)
    else:
        infer_df = df[df["split"].astype(str).isin(splits)].copy().reset_index(drop=True)
    if args.max_infer_samples > 0:
        infer_df = infer_df.head(args.max_infer_samples).copy()
    if infer_df.empty:
        raise ValueError("No inference samples selected.")

    ds = StyleContextROIDataset(infer_df, args, train=False)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    out_dir = output_dir / "outputs"
    cmp_dir = output_dir / "comparisons"
    coarse_dir = output_dir / "coarse"
    ensure_dir(out_dir); ensure_dir(cmp_dir); ensure_dir(coarse_dir)

    rows, cmp_paths = [], []
    for batch in tqdm(loader, desc="SRE-v2 inference"):
        x = batch["input"].to(device)
        damaged = batch["damaged"].to(device)
        mask = batch["mask"].to(device)
        style_imgs = batch["style_imgs"].to(device)
        style_scores = batch["style_scores"].to(device)
        style_valid = batch["style_valid"].to(device)
        sid = safe_name(batch["sample_id"][0])
        split = str(batch["split"][0])
        row = json.loads(batch["row_json"][0])

        ensure_dir(out_dir / split); ensure_dir(cmp_dir / split); ensure_dir(coarse_dir / split)
        out_path = out_dir / split / f"{sid}_rsptnet.png"
        cmp_path = cmp_dir / split / f"{sid}_rsptnet_comparison.jpg"
        coarse_path = coarse_dir / split / f"{sid}_coarse.png"

        if out_path.exists() and cmp_path.exists() and not args.overwrite_outputs:
            rec = dict(row)
            rec.update({"rsptnet_output_path": str(out_path), "comparison_path": str(cmp_path), "status": "exists"})
            rows.append(rec); cmp_paths.append(cmp_path); continue

        out = model(x, mask, style_imgs, style_scores, style_valid)
        final_t = out["refined"] * mask + damaged * (1 - mask)
        coarse_t = out["coarse"] * mask + damaged * (1 - mask)
        out_img = tensor_to_pil_rgb(final_t[0])
        coarse_img = tensor_to_pil_rgb(coarse_t[0])
        out_img.save(out_path)
        coarse_img.save(coarse_path)

        clean_img = load_rgb(row["clean_roi_path"], (args.image_size, args.image_size))
        damaged_img = load_rgb(row["damaged_roi_path"], (args.image_size, args.image_size))
        mask_img = load_l(row["mask_roi_path"], (args.image_size, args.image_size))
        visible_img = load_l(row["visible_edge_roi_path"], (args.image_size, args.image_size)) if norm_text(row.get("visible_edge_roi_path", "")) and Path(norm_text(row.get("visible_edge_roi_path", ""))).exists() else Image.new("L", (args.image_size, args.image_size), 0)
        style_ref_paths = [norm_text(row.get(f"style_ref_{i:02d}_ref_letterbox_path", "")) for i in range(1, min(args.style_ref_n, 5) + 1)]
        style_strip = make_style_strip(style_ref_paths)

        cmp = make_comparison(clean_img, damaged_img, mask_img, visible_img, style_strip, out_img, f"{sid} | {row.get('roi_type','')}")
        cmp.save(cmp_path, quality=92)
        cmp_paths.append(cmp_path)

        clean_np = np.asarray(clean_img, dtype=np.float32) / 255.0
        out_np = np.asarray(out_img, dtype=np.float32) / 255.0
        mask_np = np.asarray(mask_img, dtype=np.float32) / 255.0

        rec = dict(row)
        rec.update({
            "clean_path": row["clean_roi_path"],
            "mask_path": row["mask_roi_path"],
            "rsptnet_output_path": str(out_path),
            "coarse_output_path": str(coarse_path),
            "comparison_path": str(cmp_path),
            "status": "done",
            "checkpoint_path": str(ckpt_path),
            "full_psnr_quick": psnr_np(clean_np, out_np, 1.0),
            "masked_psnr_quick": masked_psnr_np(clean_np, out_np, mask_np, 1.0),
        })
        rows.append(rec)

    res = pd.DataFrame(rows)
    results_csv = output_dir / "baseline_results.csv"
    res.to_csv(results_csv, index=False, encoding="utf-8-sig")
    make_preview_sheet(cmp_paths, output_dir / "preview_sheet.jpg", args.preview_rows)
    print(f"[DONE] Results CSV: {results_csv}")
    print(f"[DONE] Preview: {output_dir / 'preview_sheet.jpg'}")
    return results_csv


def main():
    args = parse_args()
    set_seed(args.seed)

    root = Path(args.project_root)
    output_dir = Path(args.output_dir) if args.output_dir else (
        root / "restoration_benchmark" / "roi_restoration" / f"rsptnet_{safe_name(args.package_tag)}"
    )
    ensure_dir(output_dir)

    df = load_metadata(args)
    required = ["clean_roi_path", "damaged_roi_path", "mask_roi_path", "split", "sample_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    print(f"[INFO] Samples after ROI filtering: {len(df)}")
    print(f"[INFO] Output: {output_dir}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Target ROI types: {args.target_roi_types}")
    print("[INFO] SRE-v2: VGG multi-scale style tokens + staged training + optional DINO/GAN.")

    ckpt_path = Path(args.resume) if args.resume else None
    if not args.infer_only:
        ckpt_path = train(args, df, output_dir, device)
    if args.train_only:
        return
    if ckpt_path is None:
        ckpt_path = output_dir / "checkpoints" / "best_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    results_csv = infer(args, df, output_dir, device, ckpt_path)

    cfg = vars(args).copy()
    cfg.update({
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "checkpoint_path": str(ckpt_path),
        "results_csv": str(results_csv),
    })
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("\n[DONE] RSPT-Net SRE-v2 finished.")


if __name__ == "__main__":
    main()
