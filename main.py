import argparse
import contextlib
import copy
import io
import json
import logging
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import nms
from tqdm import tqdm


# ===================== Args =====================
def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--train_img_dir", default="/home/yunching/dl_vision/nycu-hw2-data/train")
    p.add_argument("--train_ann", default="/home/yunching/dl_vision/nycu-hw2-data/train.json")
    p.add_argument("--copypaste_img_dir", default="/home/yunching/dl_vision/nycu-hw2-data/train_copypaste")
    p.add_argument("--copypaste_ann", default="/home/yunching/dl_vision/nycu-hw2-data/train_copypaste.json")
    p.add_argument("--val_img_dir", default="/home/yunching/dl_vision/nycu-hw2-data/valid")
    p.add_argument("--val_ann", default="/home/yunching/dl_vision/nycu-hw2-data/valid.json")
    p.add_argument("--test_img_dir", default="/home/yunching/dl_vision/nycu-hw2-data/test")
    p.add_argument("--output_dir", default="./output")
    p.add_argument("--pred_file", default="pred.json")

    # run mode
    p.add_argument("--do_train", action="store_true")
    p.add_argument("--do_infer", action="store_true")
    p.add_argument("--resume", type=str, default=None)

    # dataset / classes
    p.add_argument("--num_classes", type=int, default=10)
    p.add_argument("--img_size", type=int, default=800)

    # model
    p.add_argument("--num_queries", type=int, default=75)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--enc_layers", type=int, default=6)
    p.add_argument("--dec_layers", type=int, default=6)
    p.add_argument("--dim_feedforward", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--n_points", type=int, default=4)

    # DN
    p.add_argument("--use_dn", action="store_true", default=True,
                   help="Denoising training for faster convergence")
    p.add_argument("--dn_number", type=int, default=5)
    p.add_argument("--label_noise_ratio", type=float, default=0.2)
    p.add_argument("--box_noise_scale", type=float, default=0.4)
    p.add_argument("--dn_loss_coef", type=float, default=1.0)

    # training
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--backbone_lr_decay", type=float, default=0.7,
                   help="Layer-wise LR decay for backbone. layer4=lr_backbone, layer3*decay, ...")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--clip_max_norm", type=float, default=0.1)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr_ratio", type=float, default=0.05)
    p.add_argument("--ema_decay", type=float, default=0.9997)

    # loss / matcher
    p.add_argument("--cost_class", type=float, default=2.0)
    p.add_argument("--cost_bbox", type=float, default=5.0)
    p.add_argument("--cost_giou", type=float, default=2.0)
    p.add_argument("--loss_ce", type=float, default=1.0)
    p.add_argument("--loss_bbox", type=float, default=5.0)
    p.add_argument("--loss_giou", type=float, default=2.0)
    p.add_argument("--eos_coef", type=float, default=0.1)
    p.add_argument("--aux_loss_decay", type=float, default=0.8,
                   help="Aux loss weight decay per layer from top. Final=1.0, L-1=decay, L-2=decay^2 ...")
    p.add_argument("--use_focal", action="store_true", default=True,
                   help="Use Focal Loss for classification")
    p.add_argument("--focal_alpha", type=float, default=0.25)
    p.add_argument("--focal_gamma", type=float, default=2.0)

    # eval / infer
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--val_score_thresh", type=float, default=0.05)
    p.add_argument("--score_thresh", type=float, default=0.3)
    p.add_argument("--nms_thresh", type=float, default=0.5)
    p.add_argument("--use_soft_nms", action="store_true", default=True,
                   help="Use Soft-NMS instead of hard NMS for postprocessing")

    # mosaic augmentation
    p.add_argument("--use_mosaic", action="store_true", default=True,
                   help="Mosaic augmentation: stitch 4 images into 1")
    p.add_argument("--mosaic_prob", type=float, default=0.5,
                   help="Probability of applying mosaic per sample")

    # TTA
    p.add_argument("--use_tta", action="store_true", default=False,
                   help="Multi-scale TTA during inference")
    p.add_argument("--tta_scales", type=int, nargs="+", default=[512, 640, 800],
                   help="Image sizes for TTA")

    # misc
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:2")
    # early stopping
    p.add_argument("--early_stop_patience", type=int, default=10,
                   help="Stop training if val mAP does not improve for this many eval rounds. 0=disabled.")

    return p.parse_args()


# ===================== Utilities =====================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_autocast(device: torch.device, enabled: bool = True):
    if enabled and device.type == "cuda":
        return torch.amp.autocast(device_type="cuda")
    return contextlib.nullcontext()


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


# ===================== EMA =====================
class ModelEMA:
    """Exponential Moving Average of model parameters for stable evaluation."""

    def __init__(self, model: nn.Module, decay: float = 0.9997):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
        # Also update buffers (e.g. BatchNorm running stats if any are trainable)
        for ema_b, model_b in zip(self.ema.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)


# ===================== Resize with Padding =====================
def resize_with_pad(img: Image.Image, target_size: int):
    w, h = img.size
    scale = target_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    padded = Image.new("RGB", (target_size, target_size), (128, 128, 128))
    padded.paste(img, (pad_left, pad_top))
    return padded, scale, pad_left, pad_top


# ===================== Dataset =====================
class CocoDigitDataset(Dataset):
    def __init__(self, img_dir, ann_file, img_size=512, is_train=True):
        with open(ann_file, "r", encoding="utf-8") as f:
            coco = json.load(f)
        self.img_dir = img_dir
        self.img_size = img_size
        self.is_train = is_train
        self.images = sorted(coco["images"], key=lambda x: x["id"])
        self.img_id_to_anns = defaultdict(list)
        if "annotations" in coco:
            for ann in coco["annotations"]:
                self.img_id_to_anns[ann["image_id"]].append(ann)
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # ---- Albumentations photometric pipeline (no flip / rotation) ----
        self.albu = A.Compose([
            # Noise / Blur: pick one — 加入 MedianBlur、調整觸發率
            A.OneOf([
                A.GaussNoise(std_range=(0.01, 0.08)),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.MotionBlur(blur_limit=(3, 9)),
                A.MedianBlur(blur_limit=(3, 5)),
            ], p=0.35),
            # Brightness / Contrast：縮小範圍避免過度失真
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.45),
            # Hue / Saturation / Value：hue 縮小、sat/val 微調
            A.HueSaturationValue(
                hue_shift_limit=8, sat_shift_limit=25, val_shift_limit=25, p=0.35),
            # CLAHE：提升低對比度場景的數字可見度
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        ]) if is_train else None

    def __len__(self):
        return len(self.images)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _random_crop(self, img, raw_boxes, raw_labels):
        """
        Random crop (prob 0.4) keeping ≥75% of original image area.
        Only keep boxes whose center falls inside the crop region.
        The crop is resized back to the original WxH.
        raw_boxes: list of [x, y, bw, bh] in absolute pixel coords.
        """
        if random.random() > 0.4 or not raw_boxes:
            return img, raw_boxes, raw_labels

        w, h = img.size
        # Each dim scaled ≥ sqrt(0.75) ≈ 0.866 → area ≥ 75%
        min_ratio = math.sqrt(0.75)
        crop_w = int(w * random.uniform(min_ratio, 1.0))
        crop_h = int(h * random.uniform(min_ratio, 1.0))
        x1 = random.randint(0, max(0, w - crop_w))
        y1 = random.randint(0, max(0, h - crop_h))
        x2, y2 = x1 + crop_w, y1 + crop_h

        # Crop and resize back to original size
        img = img.crop((x1, y1, x2, y2)).resize((w, h), Image.BILINEAR)

        # Adjust boxes: keep only those whose center is inside crop
        new_boxes, new_labels = [], []
        for (bx, by, bw, bh), lbl in zip(raw_boxes, raw_labels):
            cx_abs = bx + bw / 2.0
            cy_abs = by + bh / 2.0
            if x1 <= cx_abs <= x2 and y1 <= cy_abs <= y2:
                # Map to crop coords, then scale back to original size
                nx  = (bx - x1) * w / crop_w
                ny  = (by - y1) * h / crop_h
                nbw = bw * w / crop_w
                nbh = bh * h / crop_h
                new_boxes.append([nx, ny, nbw, nbh])
                new_labels.append(lbl)

        if not new_boxes:          # fallback: keep original if all boxes lost
            return img, raw_boxes, raw_labels
        return img, new_boxes, new_labels

    def _translate(self, img, raw_boxes, max_ratio=0.05):
        """
        Random translation ±5% of image size (shift canvas, fill gray).
        raw_boxes: list of [x, y, bw, bh] in absolute pixel coords.
        """
        w, h = img.size
        dx = int(random.uniform(-max_ratio, max_ratio) * w)
        dy = int(random.uniform(-max_ratio, max_ratio) * h)
        if dx == 0 and dy == 0:
            return img, raw_boxes
        canvas = Image.new("RGB", (w, h), (128, 128, 128))
        canvas.paste(img, (dx, dy))
        new_boxes = [[bx + dx, by + dy, bw, bh] for (bx, by, bw, bh) in raw_boxes]
        return canvas, new_boxes

    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        img_info = self.images[idx]
        img = Image.open(
            os.path.join(self.img_dir, img_info["file_name"])).convert("RGB")
        orig_w, orig_h = img.size

        # Collect raw annotation boxes [x, y, bw, bh] + labels
        anns = self.img_id_to_anns[img_info["id"]]
        raw_boxes  = [[a["bbox"][0], a["bbox"][1], a["bbox"][2], a["bbox"][3]] for a in anns]
        raw_labels = [int(a["category_id"]) for a in anns]

        if self.is_train:
            # 1. Random Crop (50% prob, ≥70% area) — NO flip / rotation
            img, raw_boxes, raw_labels = self._random_crop(img, raw_boxes, raw_labels)
            orig_w, orig_h = img.size   # size unchanged (resized back)

            # 2. Scale jitter 0.8×~1.2× — 縮小極端值降低小物件被裁掉機率
            sf = random.uniform(0.8, 1.2)
            new_w = max(1, int(round(orig_w * sf)))
            new_h = max(1, int(round(orig_h * sf)))
            img = img.resize((new_w, new_h), Image.BILINEAR)
            raw_boxes = [[bx*sf, by*sf, bw*sf, bh*sf] for (bx, by, bw, bh) in raw_boxes]
            orig_w, orig_h = new_w, new_h

            # 3. Translation ±4%（略小於參考版，減少 box 被推出邊界的機率）
            img, raw_boxes = self._translate(img, raw_boxes, max_ratio=0.04)

        # 4. Letterbox resize + pad
        img, scale, pad_left, pad_top = resize_with_pad(img, self.img_size)

        # 5. Convert raw boxes → normalized [cx, cy, nw, nh]
        boxes, labels = [], []
        for (bx, by, bw, bh), lbl in zip(raw_boxes, raw_labels):
            x_pad  = bx * scale + pad_left
            y_pad  = by * scale + pad_top
            bw_pad = bw * scale
            bh_pad = bh * scale
            cx = (x_pad + bw_pad / 2.0) / self.img_size
            cy = (y_pad + bh_pad / 2.0) / self.img_size
            nw = bw_pad / self.img_size
            nh = bh_pad / self.img_size
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            if nw > 1e-3 and nh > 1e-3:
                boxes.append([cx, cy, nw, nh])
                labels.append(lbl)

        # 6. Albumentations photometric augmentation (numpy HWC uint8)
        if self.albu is not None:
            img_np = np.array(img)
            img_np = self.albu(image=img_np)["image"]
            img = Image.fromarray(img_np)

        # 7. ToTensor + Normalize
        img = self.normalize(img)

        return img, {
            "boxes":     torch.tensor(boxes,  dtype=torch.float32) if boxes  else torch.zeros((0, 4),  dtype=torch.float32),
            "labels":    torch.tensor(labels, dtype=torch.long)    if labels else torch.zeros((0,),    dtype=torch.long),
            "image_id":  int(img_info["id"]),
            "orig_size": torch.tensor([orig_h, orig_w], dtype=torch.long),
            "scale":     torch.tensor(scale,    dtype=torch.float32),
            "pad":       torch.tensor([pad_left, pad_top], dtype=torch.float32),
        }


class TestDataset(Dataset):
    def __init__(self, img_dir, img_size=512):
        self.img_dir = img_dir
        self.img_size = img_size
        self.filenames = sorted(os.listdir(img_dir))
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        orig_w, orig_h = img.size
        img, scale, pad_left, pad_top = resize_with_pad(img, self.img_size)
        img = self.normalize(img)
        img_id = int(os.path.splitext(fname)[0])
        return img, img_id, orig_h, orig_w, scale, pad_left, pad_top


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, dim=0), list(targets)


def collate_fn_test(batch):
    imgs, ids, hs, ws, scales, pls, pts = zip(*batch)
    return torch.stack(imgs, dim=0), list(ids), list(hs), list(ws), list(scales), list(pls), list(pts)


# ===================== Box Utils =====================
def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(2)
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-7)
    lt_enc = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_enc = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    area_enc = (rb_enc - lt_enc).clamp(min=0).prod(2)
    return iou - (area_enc - union) / (area_enc + 1e-7)


def convert_to_orig_coords(boxes_cxcywh, img_size, scale, pad_left, pad_top, orig_w, orig_h):
    cx = boxes_cxcywh[:, 0] * img_size
    cy = boxes_cxcywh[:, 1] * img_size
    bw = boxes_cxcywh[:, 2] * img_size
    bh = boxes_cxcywh[:, 3] * img_size
    cx = (cx - pad_left) / scale
    cy = (cy - pad_top) / scale
    bw = bw / scale
    bh = bh / scale
    x1 = (cx - bw / 2.0).clamp(min=0)
    y1 = (cy - bh / 2.0).clamp(min=0)
    x2 = (cx + bw / 2.0).clamp(max=orig_w)
    y2 = (cy + bh / 2.0).clamp(max=orig_h)
    return torch.stack([x1, y1, (x2 - x1).clamp(min=0), (y2 - y1).clamp(min=0)], dim=-1)


# ===================== Soft NMS =====================
def soft_nms(boxes_xyxy: torch.Tensor, scores: torch.Tensor,
             sigma: float = 0.5, score_thresh: float = 0.001) -> torch.Tensor:
    """
    Soft-NMS (Gaussian penalty): decay overlapping box scores instead of removing.
    Better than hard NMS for closely spaced digits.
    """
    boxes = boxes_xyxy.clone()
    sc    = scores.clone()
    keep  = []
    idxs  = sc.argsort(descending=True).tolist()
    while idxs:
        i = idxs[0]; keep.append(i); idxs = idxs[1:]
        if not idxs:
            break
        rest = torch.tensor(idxs, dtype=torch.long)
        from torchvision.ops import box_iou
        iou  = box_iou(boxes[i:i+1], boxes[rest])[0]
        sc[rest] *= torch.exp(-(iou ** 2) / sigma)
        idxs = [j for j, r in zip(idxs, range(len(idxs))) if sc[idxs[r]] > score_thresh]
    return torch.tensor(keep, dtype=torch.long)


# ===================== Mosaic Augmentation =====================
def mosaic_collate(batch, dataset, img_size: int, prob: float = 0.5):
    """
    With probability `prob`, stitch 4 images into one mosaic canvas.
    Gives the model 4× more digit instances per step and forces small-object detection.
    No flip/rotation applied inside mosaic (digit identity preserved).
    """
    from torch.nn import functional as _F
    imgs_out, targets_out = [], []
    for img, tgt in batch:
        if random.random() > prob or len(dataset) < 4:
            imgs_out.append(img); targets_out.append(tgt); continue

        idxs  = random.sample(range(len(dataset)), 3)
        four  = [(img, tgt)] + [dataset[j] for j in idxs]

        cx = random.uniform(0.3, 0.7)
        cy = random.uniform(0.3, 0.7)

        mean   = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std    = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        canvas = (torch.full((3, img_size, img_size), 0.5) - mean) / std

        quads = [
            (0,              0,              int(cx*img_size), int(cy*img_size)),
            (int(cx*img_size), 0,            img_size,          int(cy*img_size)),
            (0,              int(cy*img_size), int(cx*img_size), img_size),
            (int(cx*img_size), int(cy*img_size), img_size,       img_size),
        ]
        all_boxes, all_labels = [], []
        for k, (im, tg) in enumerate(four):
            qx1, qy1, qx2, qy2 = quads[k]
            qw = max(qx2-qx1, 1); qh = max(qy2-qy1, 1)
            src = _F.interpolate(im.unsqueeze(0), size=(qh, qw),
                                 mode="bilinear", align_corners=False).squeeze(0)
            canvas[:, qy1:qy2, qx1:qx2] = src
            if len(tg["boxes"]) == 0: continue
            bx = tg["boxes"].clone() * img_size
            bx[:, 0] = bx[:, 0] * qw / img_size + qx1
            bx[:, 1] = bx[:, 1] * qh / img_size + qy1
            bx[:, 2] = bx[:, 2] * qw / img_size
            bx[:, 3] = bx[:, 3] * qh / img_size
            x1c = (bx[:,0]-bx[:,2]/2).clamp(qx1, qx2)
            y1c = (bx[:,1]-bx[:,3]/2).clamp(qy1, qy2)
            x2c = (bx[:,0]+bx[:,2]/2).clamp(qx1, qx2)
            y2c = (bx[:,1]+bx[:,3]/2).clamp(qy1, qy2)
            bw_ = (x2c-x1c).clamp(0); bh_ = (y2c-y1c).clamp(0)
            valid = (bw_ > 2) & (bh_ > 2)
            if not valid.any(): continue
            cx_ = ((x1c+x2c)/2/img_size)[valid]
            cy_ = ((y1c+y2c)/2/img_size)[valid]
            nw_ = (bw_/img_size)[valid]; nh_ = (bh_/img_size)[valid]
            all_boxes.append(torch.stack([cx_, cy_, nw_, nh_], 1))
            all_labels.append(tg["labels"][valid])

        new_boxes  = torch.cat(all_boxes,  0) if all_boxes  else torch.zeros((0,4))
        new_labels = torch.cat(all_labels, 0) if all_labels else torch.zeros((0,), dtype=torch.long)
        new_tgt = dict(tgt); new_tgt["boxes"] = new_boxes; new_tgt["labels"] = new_labels
        imgs_out.append(canvas); targets_out.append(new_tgt)

    return torch.stack(imgs_out), targets_out


# ===================== Positional Encoding =====================
def make_pos_embed_2d(h, w, dim, device):
    if dim % 4 != 0:
        raise ValueError(f"hidden_dim must be divisible by 4, got {dim}")
    grid_y, grid_x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32, device=device),
        torch.arange(w, dtype=torch.float32, device=device),
        indexing="ij",
    )
    grid_x = (grid_x + 0.5) / max(w, 1)
    grid_y = (grid_y + 0.5) / max(h, 1)
    omega = torch.arange(dim // 4, dtype=torch.float32, device=device)
    omega = 1.0 / (10000 ** (omega / max(dim // 4, 1)))
    out_x = grid_x.flatten()[:, None] * omega[None, :] * 2.0 * math.pi
    out_y = grid_y.flatten()[:, None] * omega[None, :] * 2.0 * math.pi
    pos = torch.cat([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=1)
    return pos


# ===================== Multi-Scale Deformable Attention =====================
class MultiScaleDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=4, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = d_model // n_heads
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= (i + 1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.reshape(-1))
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query, reference_points, value, spatial_shapes):
        B, Nq, _ = query.shape
        Nv = value.shape[1]
        H = self.n_heads; D = self.head_dim
        L = self.n_levels; P = self.n_points

        # Project value
        v_proj = self.value_proj(value).view(B, Nv, H, D)

        # Compute sampling offsets and normalize to [0,1] coords
        raw_offsets = self.sampling_offsets(query).view(B, Nq, H, L, P, 2)
        wh_tensor   = torch.tensor([[w, h] for h, w in spatial_shapes],
                                   dtype=torch.float32, device=query.device)
        ref_pts   = reference_points[:, :, None, :, None, :]          # (B,Nq,1,L,1,2)
        norm_wh   = wh_tensor[None, None, None, :, None, :]           # (1,1,1,L,1,2)
        samp_locs = ref_pts + raw_offsets / norm_wh                    # (B,Nq,H,L,P,2)
        samp_grid = 2.0 * samp_locs - 1.0                             # remap to [-1,1]

        # Attention weights over all L*P points (softmax)
        attn_w = self.attention_weights(query).view(B, Nq, H, L * P)
        attn_w = F.softmax(attn_w, dim=-1).view(B, Nq, H, L, P)      # (B,Nq,H,L,P)

        # Sample each level then weighted-sum
        level_sizes  = [h * w for h, w in spatial_shapes]
        v_per_level  = v_proj.split(level_sizes, dim=1)
        sampled_all  = []
        for lid, (lh, lw) in enumerate(spatial_shapes):
            feat_l  = v_per_level[lid].permute(0, 2, 3, 1).reshape(B * H, D, lh, lw)
            grid_l  = samp_grid[:, :, :, lid, :, :].permute(0, 2, 1, 3, 4).reshape(B * H, Nq, P, 2)
            samp_l  = F.grid_sample(feat_l, grid_l, mode="bilinear",
                                    padding_mode="zeros", align_corners=False)  # (B*H,D,Nq,P)
            sampled_all.append(samp_l)

        # Concat along last dim → (B*H, D, Nq, L*P)
        sampled_cat = torch.cat(sampled_all, dim=-1)
        attn_flat   = attn_w.view(B, Nq, H, L * P).permute(0, 2, 1, 3).reshape(B * H, 1, Nq, L * P)
        out = (sampled_cat * attn_flat).sum(dim=-1)                    # (B*H, D, Nq)
        out = out.view(B, H, D, Nq).permute(0, 3, 1, 2).reshape(B, Nq, self.d_model)
        return self.output_proj(out)


# ===================== Encoder / Decoder =====================
class DefEncLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=4, n_points=4, d_ffn=1024, dropout=0.1):
        super().__init__()
        self.self_attn = MultiScaleDeformAttn(d_model, n_heads, n_levels, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_ffn, d_model))
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, pos, reference_points, spatial_shapes):
        src2 = self.self_attn(src + pos, reference_points, src, spatial_shapes)
        src = self.norm1(src + self.dropout1(src2))
        src2 = self.ffn(src)
        src = self.norm2(src + self.dropout2(src2))
        return src


class DefDecLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, n_levels=4, n_points=4, d_ffn=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiScaleDeformAttn(d_model, n_heads, n_levels, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(d_ffn, d_model))
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, query_pos, memory, reference_points, spatial_shapes, attn_mask=None):
        q = k = tgt + query_pos
        tgt2, _ = self.self_attn(q, k, tgt, attn_mask=attn_mask)
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.cross_attn(tgt + query_pos, reference_points, memory, spatial_shapes)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


# ===================== DN helpers =====================
def build_dn_queries(targets, num_classes, dn_number, label_noise_ratio, box_noise_scale, device):
    if dn_number <= 0:
        return None
    batch_size = len(targets)
    gt_counts = [int(t["labels"].numel()) for t in targets]
    max_gt = max(gt_counts) if gt_counts else 0
    if max_gt == 0:
        return None

    pad_size = max_gt * dn_number
    dn_labels = torch.zeros((batch_size, pad_size), dtype=torch.long, device=device)
    dn_boxes = torch.zeros((batch_size, pad_size, 4), dtype=torch.float32, device=device)
    dn_mask = torch.zeros((batch_size, pad_size), dtype=torch.bool, device=device)
    dn_positive_idx: List[List[Tuple[int, int]]] = []

    for b, target in enumerate(targets):
        labels = target["labels"].to(device)
        boxes = target["boxes"].to(device)
        num_gt = labels.numel()
        sample_pairs: List[Tuple[int, int]] = []
        if num_gt == 0:
            dn_positive_idx.append(sample_pairs)
            continue
        for rep in range(dn_number):
            start = rep * max_gt
            end = start + num_gt
            noisy_labels = labels.clone()
            if label_noise_ratio > 0:
                noise_mask = torch.rand(num_gt, device=device) < label_noise_ratio
                random_labels = torch.randint(1, num_classes + 1, (num_gt,), device=device)
                noisy_labels = torch.where(noise_mask, random_labels, noisy_labels)
            noisy_boxes = boxes.clone()
            if box_noise_scale > 0:
                rand_sign = torch.randint(0, 2, noisy_boxes.shape, device=device).float() * 2.0 - 1.0
                rand_mag = torch.rand_like(noisy_boxes)
                box_wh = boxes[:, [2, 3, 2, 3]].clamp(min=1e-3)
                noisy_boxes = noisy_boxes + rand_sign * rand_mag * box_wh * box_noise_scale
                noisy_boxes = noisy_boxes.clamp(0.0, 1.0)
                noisy_boxes[:, 2:] = noisy_boxes[:, 2:].clamp(min=1e-3)
            dn_labels[b, start:end] = noisy_labels
            dn_boxes[b, start:end] = noisy_boxes
            dn_mask[b, start:end] = True
            sample_pairs.extend([(start + i, i) for i in range(num_gt)])
        dn_positive_idx.append(sample_pairs)

    attn_mask = torch.zeros((pad_size, pad_size), dtype=torch.bool, device=device)
    for rep in range(dn_number):
        s = rep * max_gt
        e = s + max_gt
        attn_mask[s:e, :s] = True
        attn_mask[s:e, e:pad_size] = True

    return dn_labels, dn_boxes, {
        "pad_size": pad_size, "max_gt": max_gt,
        "dn_positive_idx": dn_positive_idx, "dn_mask": dn_mask, "attn_mask": attn_mask,
    }


# ===================== Deformable DETR =====================
class DigitDETR(nn.Module):
    def __init__(
        self, num_classes=10, num_queries=75, hidden_dim=256, nheads=8,
        enc_layers=6, dec_layers=6, dim_feedforward=1024, dropout=0.1,
        n_points=4, n_levels=4,
        use_dn=False, dn_number=5, label_noise_ratio=0.2, box_noise_scale=0.4,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_classes_with_bg = num_classes + 1
        self.n_levels = n_levels
        self.use_dn = use_dn
        self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # ---- Backbone ----
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.layer1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool, backbone.layer1)
        self.layer2 = backbone.layer2   # C3, 512ch, stride 8
        self.layer3 = backbone.layer3   # C4, 1024ch, stride 16
        self.layer4 = backbone.layer4   # C5, 2048ch, stride 32

        for module in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

        # ---- Feature projection: C3, C4, C5 via 1x1; C6 via 3x3 stride-2 from C5 ----
        self.input_proj = nn.ModuleList([
            nn.Sequential(nn.Conv2d(512, hidden_dim, 1), nn.GroupNorm(32, hidden_dim)),    # C3
            nn.Sequential(nn.Conv2d(1024, hidden_dim, 1), nn.GroupNorm(32, hidden_dim)),   # C4
            nn.Sequential(nn.Conv2d(2048, hidden_dim, 1), nn.GroupNorm(32, hidden_dim)),   # C5
            nn.Sequential(nn.Conv2d(2048, hidden_dim, 3, stride=2, padding=1), nn.GroupNorm(32, hidden_dim)),  # C6
        ])
        self.level_embed = nn.Parameter(torch.randn(n_levels, hidden_dim))

        # ---- Encoder ----
        self.encoder_layers = nn.ModuleList([
            DefEncLayer(hidden_dim, nheads, n_levels, n_points, dim_feedforward, dropout)
            for _ in range(enc_layers)
        ])

        # ---- Decoder ----
        self.decoder_layers = nn.ModuleList([
            DefDecLayer(hidden_dim, nheads, n_levels, n_points, dim_feedforward, dropout)
            for _ in range(dec_layers)
        ])
        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        self.reference_points_head = nn.Linear(hidden_dim, 4)

        # ---- DN embeddings ----
        self.label_enc = nn.Embedding(self.num_classes_with_bg, hidden_dim)
        self.box_enc = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim))
        self.dn_query_pos = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim))

        # ---- Prediction heads ----
        self.class_heads = nn.ModuleList([nn.Linear(hidden_dim, self.num_classes_with_bg) for _ in range(dec_layers)])
        self.bbox_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 4),
            ) for _ in range(dec_layers)
        ])

    def _get_encoder_reference_points(self, spatial_shapes, device):
        refs = []
        for h, w in spatial_shapes:
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, h, device=device) / h,
                torch.linspace(0.5, w - 0.5, w, device=device) / w,
                indexing="ij",
            )
            ref = torch.stack([ref_x.reshape(-1), ref_y.reshape(-1)], dim=-1)
            refs.append(ref)
        reference_points = torch.cat(refs, dim=0)
        reference_points = reference_points[:, None, :].repeat(1, len(spatial_shapes), 1)
        return reference_points

    def _build_dn_inputs(self, targets, device):
        if not self.training or not self.use_dn or targets is None:
            return None, None, None, None
        out = build_dn_queries(targets, self.num_classes, self.dn_number,
                                     self.label_noise_ratio, self.box_noise_scale, device)
        if out is None:
            return None, None, None, None
        dn_labels, dn_boxes, dn_meta = out
        dn_tgt = self.label_enc(dn_labels) + self.box_enc(dn_boxes)
        dn_qpos = self.dn_query_pos(dn_boxes)
        return dn_tgt, dn_qpos, dn_boxes, dn_meta

    def forward(self, x: torch.Tensor, targets=None):
        bsz = x.size(0)
        device = x.device

        # ---- Backbone ----
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # ---- Multi-scale: C3, C4, C5, C6 ----
        raw_features = [c3, c4, c5, c5]  # C6 produced by stride-2 conv in input_proj[3]
        srcs, poss, spatial_shapes = [], [], []
        for lid, feat in enumerate(raw_features):
            src = self.input_proj[lid](feat)
            _, _, h, w = src.shape
            spatial_shapes.append((h, w))
            pos = make_pos_embed_2d(h, w, self.hidden_dim, device)
            pos = pos.unsqueeze(0).expand(bsz, -1, -1) + self.level_embed[lid].view(1, 1, -1)
            srcs.append(src.flatten(2).permute(0, 2, 1))
            poss.append(pos)

        src_flat = torch.cat(srcs, dim=1)
        pos_flat = torch.cat(poss, dim=1)

        # ---- Encoder ----
        enc_ref = self._get_encoder_reference_points(spatial_shapes, device)
        enc_ref = enc_ref.unsqueeze(0).expand(bsz, -1, -1, -1)
        memory = src_flat
        for layer in self.encoder_layers:
            memory = layer(memory, pos_flat, enc_ref, spatial_shapes)

        # ---- Decoder queries ----
        query_embed = self.query_embed.weight
        query_pos, query_content = query_embed.split(self.hidden_dim, dim=-1)
        query_pos = query_pos.unsqueeze(0).expand(bsz, -1, -1)
        tgt = query_content.unsqueeze(0).expand(bsz, -1, -1)
        reference_points = self.reference_points_head(query_pos).sigmoid()

        # ---- DN queries (training only) ----
        dn_tgt, dn_qpos, dn_ref, dn_meta = self._build_dn_inputs(targets, device)
        total_attn_mask = None
        if dn_tgt is not None:
            tgt = torch.cat([dn_tgt, tgt], dim=1)
            query_pos = torch.cat([dn_qpos, query_pos], dim=1)
            reference_points = torch.cat([dn_ref, reference_points], dim=1)
            total_len = tgt.size(1)
            pad_size = dn_meta["pad_size"]
            total_attn_mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)
            total_attn_mask[:pad_size, :pad_size] = dn_meta["attn_mask"]
            total_attn_mask[pad_size:, :pad_size] = True

        # ---- Decoder ----
        outputs_classes, outputs_coords = [], []
        output = tgt
        for lid, layer in enumerate(self.decoder_layers):
            ref_for_attn = reference_points[:, :, None, :2].repeat(1, 1, self.n_levels, 1)
            output = layer(output, query_pos, memory, ref_for_attn, spatial_shapes, attn_mask=total_attn_mask)
            output_norm = self.decoder_norm(output)
            logits = self.class_heads[lid](output_norm)
            box_delta = self.bbox_heads[lid](output_norm)
            pred_boxes = (box_delta + inverse_sigmoid(reference_points)).sigmoid()
            outputs_classes.append(logits)
            outputs_coords.append(pred_boxes)
            reference_points = pred_boxes.detach()

        out = {
            "pred_logits": outputs_classes[-1],
            "pred_boxes": outputs_coords[-1],
            "aux_outputs": [
                {"pred_logits": outputs_classes[i], "pred_boxes": outputs_coords[i]}
                for i in range(len(outputs_classes) - 1)
            ],
        }
        if dn_meta is not None:
            out["dn_meta"] = dn_meta
        return out

    def train(self, mode=True):
        super().train(mode)
        for module in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self


# ===================== Matcher =====================
class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets, use_focal=False):
        bsz = outputs["pred_logits"].shape[0]
        indices = []
        for b in range(bsz):
            tgt_labels = targets[b]["labels"]
            tgt_boxes = targets[b]["boxes"]
            if tgt_labels.numel() == 0:
                indices.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue
            if use_focal:
                pred_prob = outputs["pred_logits"][b].sigmoid()
            else:
                pred_prob = outputs["pred_logits"][b].softmax(-1)
            pred_box = outputs["pred_boxes"][b]
            cost_class = -pred_prob[:, tgt_labels]
            cost_bbox = torch.cdist(pred_box, tgt_boxes, p=1)
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(pred_box), box_cxcywh_to_xyxy(tgt_boxes))
            cost = self.cost_class * cost_class + self.cost_bbox * cost_bbox + self.cost_giou * cost_giou
            row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())
            indices.append((torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)))
        return indices


# ===================== Focal Loss =====================
def sigmoid_focal_loss(logits, targets_onehot, alpha=0.25, gamma=2.0, reduction="mean"):
    prob = logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets_onehot, reduction="none")
    p_t = prob * targets_onehot + (1 - prob) * (1 - targets_onehot)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets_onehot + (1 - alpha) * (1 - targets_onehot)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    return loss.sum() if reduction == "sum" else loss


# ===================== Loss =====================
class SetCriterion(nn.Module):
    def __init__(self, num_classes_with_bg, matcher, w_ce=1.0, w_bbox=5.0, w_giou=2.0, eos_coef=0.1,
                 dn_loss_coef=1.0, use_focal=False, focal_alpha=0.25, focal_gamma=2.0,
                 aux_loss_decay=0.8):
        super().__init__()
        self.num_classes_with_bg = num_classes_with_bg
        self.matcher = matcher
        self.w_ce = w_ce
        self.w_bbox = w_bbox
        self.w_giou = w_giou
        self.dn_loss_coef = dn_loss_coef
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.aux_loss_decay = aux_loss_decay
        empty_weight = torch.ones(num_classes_with_bg)
        empty_weight[0] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _loss_labels(self, pred_logits, target_classes):
        if not self.use_focal:
            return F.cross_entropy(pred_logits.permute(0, 2, 1), target_classes, weight=self.empty_weight)
        bsz, nq, nc = pred_logits.shape
        target_onehot = torch.zeros((bsz, nq, nc), dtype=pred_logits.dtype, device=pred_logits.device)
        target_onehot.scatter_(2, target_classes.unsqueeze(-1), 1.0)
        return sigmoid_focal_loss(pred_logits, target_onehot, alpha=self.focal_alpha, gamma=self.focal_gamma)

    def _compute_loss(self, outputs, targets, indices):
        device = outputs["pred_logits"].device
        target_classes = torch.zeros(outputs["pred_logits"].shape[:2], dtype=torch.long, device=device)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[b, src_idx] = targets[b]["labels"][tgt_idx]
        loss_ce = self._loss_labels(outputs["pred_logits"], target_classes)

        src_boxes, tgt_boxes = [], []
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                src_boxes.append(outputs["pred_boxes"][b][src_idx])
                tgt_boxes.append(targets[b]["boxes"][tgt_idx])
        if src_boxes:
            src_boxes = torch.cat(src_boxes, dim=0)
            tgt_boxes = torch.cat(tgt_boxes, dim=0)
            loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="mean")
            loss_giou = 1.0 - generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes)).diag().mean()
        else:
            loss_bbox = torch.tensor(0.0, device=device)
            loss_giou = torch.tensor(0.0, device=device)
        total = self.w_ce * loss_ce + self.w_bbox * loss_bbox + self.w_giou * loss_giou
        return total, float(loss_ce.item()), float(loss_bbox.item()), float(loss_giou.item())

    def _compute_dn_loss(self, outputs, targets, dn_meta):
        device = outputs["pred_logits"].device
        pad_size = int(dn_meta["pad_size"])
        if pad_size <= 0:
            return torch.tensor(0.0, device=device), 0.0
        pred_logits = outputs["pred_logits"][:, :pad_size]
        pred_boxes = outputs["pred_boxes"][:, :pad_size]

        src_logits_list, tgt_labels_list, src_boxes_list, tgt_boxes_list = [], [], [], []
        for b, pairs in enumerate(dn_meta["dn_positive_idx"]):
            if not pairs:
                continue
            src_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=device)
            tgt_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=device)
            src_logits_list.append(pred_logits[b, src_idx])
            tgt_labels_list.append(targets[b]["labels"][tgt_idx].to(device))
            src_boxes_list.append(pred_boxes[b, src_idx])
            tgt_boxes_list.append(targets[b]["boxes"][tgt_idx].to(device))
        if not src_logits_list:
            return torch.tensor(0.0, device=device), 0.0

        src_logits = torch.cat(src_logits_list, dim=0)
        tgt_labels = torch.cat(tgt_labels_list, dim=0)
        src_boxes = torch.cat(src_boxes_list, dim=0)
        tgt_boxes = torch.cat(tgt_boxes_list, dim=0)

        if not self.use_focal:
            loss_ce = F.cross_entropy(src_logits, tgt_labels, weight=self.empty_weight)
        else:
            onehot = torch.zeros_like(src_logits)
            onehot.scatter_(1, tgt_labels.unsqueeze(1), 1.0)
            loss_ce = sigmoid_focal_loss(src_logits, onehot, alpha=self.focal_alpha, gamma=self.focal_gamma)

        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction="mean")
        loss_giou = 1.0 - generalized_box_iou(box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes)).diag().mean()
        total_dn = (self.w_ce * loss_ce + self.w_bbox * loss_bbox + self.w_giou * loss_giou) * self.dn_loss_coef
        return total_dn, float(total_dn.item())

    def _split_match_outputs(self, outputs):
        dn_meta = outputs.get("dn_meta")
        if dn_meta is None:
            return outputs, None
        pad_size = int(dn_meta["pad_size"])
        match_outputs = {
            "pred_logits": outputs["pred_logits"][:, pad_size:],
            "pred_boxes": outputs["pred_boxes"][:, pad_size:],
        }
        if "aux_outputs" in outputs:
            match_outputs["aux_outputs"] = [
                {"pred_logits": aux["pred_logits"][:, pad_size:], "pred_boxes": aux["pred_boxes"][:, pad_size:]}
                for aux in outputs["aux_outputs"]
            ]
        return match_outputs, dn_meta

    def forward(self, outputs, targets):
        match_outputs, dn_meta = self._split_match_outputs(outputs)
        indices = self.matcher(match_outputs, targets, use_focal=self.use_focal)
        loss_main, ce, bbox, giou = self._compute_loss(match_outputs, targets, indices)

        aux_total = torch.tensor(0.0, device=loss_main.device)
        if "aux_outputs" in match_outputs:
            aux_list = match_outputs["aux_outputs"]
            for k, aux in enumerate(reversed(aux_list)):   # k=0 is layer closest to final
                weight = self.aux_loss_decay ** (k + 1)
                aux_indices = self.matcher(aux, targets, use_focal=self.use_focal)
                aux_loss, _, _, _ = self._compute_loss(aux, targets, aux_indices)
                aux_total = aux_total + weight * aux_loss

        dn_total = torch.tensor(0.0, device=loss_main.device)
        dn_val = 0.0
        if dn_meta is not None:
            dn_total, dn_val = self._compute_dn_loss(outputs, targets, dn_meta)

        total_loss = loss_main + aux_total + dn_total
        loss_dict = {
            "loss": float(total_loss.item()),
            "loss_main": float(loss_main.item()),
            "loss_aux": float(aux_total.item()),
            "loss_ce": ce, "loss_bbox": bbox, "loss_giou": giou,
            "loss_dn": dn_val,
        }
        return total_loss, loss_dict, indices


# ===================== Postprocess =====================
def postprocess_single_image_predictions(
    logits, boxes, img_size, scale, pad_left, pad_top,
    orig_w, orig_h, score_thresh, nms_thresh, use_focal=False, use_soft_nms=True,
):
    if use_focal:
        probs = logits.sigmoid()
    else:
        probs = logits.softmax(-1)
    scores, cls_ids = probs[:, 1:].max(dim=-1)
    cls_ids = cls_ids + 1

    keep = scores > score_thresh
    scores = scores[keep]
    cls_ids = cls_ids[keep]
    boxes = boxes[keep]
    if scores.numel() == 0:
        return []

    boxes_xywh = convert_to_orig_coords(boxes.cpu(), img_size, scale, pad_left, pad_top, orig_w, orig_h)
    boxes_xyxy = boxes_xywh.clone()
    boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + boxes_xyxy[:, 2]
    boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes_xyxy[:, 3]

    keep_indices = []
    for cls in cls_ids.unique():
        cls_mask = cls_ids == cls
        cls_idx  = cls_mask.nonzero(as_tuple=True)[0]
        if use_soft_nms:
            ki = soft_nms(boxes_xyxy[cls_idx].float(), scores[cls_idx].float(),
                          sigma=0.5, score_thresh=score_thresh * 0.1)
        else:
            ki = nms(boxes_xyxy[cls_idx].float(), scores[cls_idx].float(), nms_thresh)
        keep_indices.append(cls_idx[ki])
    if not keep_indices:
        return []

    keep_indices = torch.cat(keep_indices, dim=0)
    results = []
    for idx in keep_indices.tolist():
        results.append({
            "bbox": boxes_xywh[idx].tolist(),
            "score": float(scores[idx].item()),
            "category_id": int(cls_ids[idx].item()),
        })
    return results


# ===================== Evaluation =====================
@torch.no_grad()
def evaluate(model, loader, criterion, device, img_size, output_dir, epoch, num_classes, val_score_thresh, nms_thresh, use_focal, use_soft_nms=True):
    model.eval()
    total_loss = defaultdict(float)
    n_batches = 0
    all_preds_coco = []
    all_gt_coco = {"images": [], "annotations": [], "categories": [{"id": i} for i in range(1, num_classes + 1)]}
    gt_ann_id = 0
    all_pred_cls, all_gt_cls = [], []
    use_amp = device.type == "cuda"

    for imgs, targets in tqdm(loader, desc="Eval", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with get_autocast(device, enabled=use_amp):
            outputs = model(imgs)
            loss, loss_dict, indices = criterion(outputs, targets)

        match_logits = outputs["pred_logits"]
        if "dn_meta" in outputs:
            pad = int(outputs["dn_meta"]["pad_size"])
            match_logits = match_logits[:, pad:]

        for k, v in loss_dict.items():
            total_loss[k] += float(v)
        n_batches += 1

        pred_cls = match_logits.argmax(dim=-1)
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                all_pred_cls.extend(pred_cls[b][src_idx].detach().cpu().tolist())
                all_gt_cls.extend(targets[b]["labels"][tgt_idx].detach().cpu().tolist())

        pred_boxes_for_eval = outputs["pred_boxes"]
        if "dn_meta" in outputs:
            pad = int(outputs["dn_meta"]["pad_size"])
            pred_boxes_for_eval = pred_boxes_for_eval[:, pad:]

        for b in range(imgs.size(0)):
            img_id = int(targets[b]["image_id"])
            orig_h, orig_w = targets[b]["orig_size"].tolist()
            scale = float(targets[b]["scale"].item())
            pad_left, pad_top = targets[b]["pad"].tolist()
            all_gt_coco["images"].append({"id": img_id, "width": int(orig_w), "height": int(orig_h)})
            for j in range(len(targets[b]["labels"])):
                box_abs = convert_to_orig_coords(
                    targets[b]["boxes"][j:j + 1].detach().cpu(), img_size, scale, pad_left, pad_top, orig_w, orig_h)
                bx = box_abs[0].tolist()
                all_gt_coco["annotations"].append({
                    "id": gt_ann_id, "image_id": img_id,
                    "category_id": int(targets[b]["labels"][j].item()),
                    "bbox": bx, "area": float(bx[2] * bx[3]), "iscrowd": 0,
                })
                gt_ann_id += 1
            preds = postprocess_single_image_predictions(
                match_logits[b].detach().cpu(), pred_boxes_for_eval[b].detach().cpu(),
                img_size, scale, pad_left, pad_top, orig_w, orig_h,
                val_score_thresh, nms_thresh, use_focal=use_focal, use_soft_nms=use_soft_nms,
            )
            for pred in preds:
                all_preds_coco.append({
                    "image_id": img_id, "category_id": pred["category_id"],
                    "bbox": pred["bbox"], "score": pred["score"],
                })

    avg_loss = {k: v / max(n_batches, 1) for k, v in total_loss.items()}
    mAP = 0.0
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        coco_gt = COCO()
        coco_gt.dataset = all_gt_coco
        coco_gt.createIndex()
        if all_preds_coco:
            coco_dt = coco_gt.loadRes(all_preds_coco)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            with contextlib.redirect_stdout(io.StringIO()):
                coco_eval.evaluate()
                coco_eval.accumulate()
            coco_eval.summarize()
            mAP = float(coco_eval.stats[0])
    except Exception as e:
        print(f"mAP eval failed: {e}")

    if all_pred_cls:
        plot_confusion_matrix(all_pred_cls, all_gt_cls, output_dir, epoch, num_classes)
        matched_acc = sum(int(p == g) for p, g in zip(all_pred_cls, all_gt_cls)) / len(all_pred_cls)
    else:
        matched_acc = 0.0
    return avg_loss, matched_acc, mAP


# ===================== Plotting =====================
def plot_confusion_matrix(preds, gts, output_dir, epoch, num_classes):
    try:
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
        labels = list(range(1, num_classes + 1))
        cm = confusion_matrix(gts, preds, labels=labels)
        disp = ConfusionMatrixDisplay(cm, display_labels=[str(i) for i in labels])
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"Confusion Matrix (epoch {epoch + 1})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_ep{epoch + 1}.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass


def plot_curves(history, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(history["train_loss"], label="Train Loss")
    axes[0, 0].set_title("Train Loss"); axes[0, 0].grid(True); axes[0, 0].legend()
    ep = [i for i, v in enumerate(history["val_loss"]) if v is not None]
    vl = [v for v in history["val_loss"] if v is not None]
    if vl: axes[0, 1].plot(ep, vl, label="Val Loss", marker="o", markersize=3)
    axes[0, 1].set_title("Val Loss"); axes[0, 1].grid(True); axes[0, 1].legend()
    ea = [i for i, v in enumerate(history["val_acc"]) if v is not None]
    va = [v for v in history["val_acc"] if v is not None]
    if va: axes[1, 0].plot(ea, va, label="Val Matched Acc", marker="o", markersize=3)
    axes[1, 0].set_title("Val Matched Accuracy"); axes[1, 0].grid(True); axes[1, 0].legend()
    em = [i for i, v in enumerate(history["val_mAP"]) if v is not None]
    vm = [v for v in history["val_mAP"] if v is not None]
    if vm: axes[1, 1].plot(em, vm, label="Val mAP", marker="o", markersize=3)
    axes[1, 1].set_title("Val mAP @[.5:.95]"); axes[1, 1].grid(True); axes[1, 1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curves.png"), dpi=150)
    plt.close(fig)


# ===================== Layer-wise Backbone LR =====================
def build_param_groups(model, lr, lr_backbone, backbone_lr_decay):
    """
    Layer-wise LR decay for backbone:
      layer4 → lr_backbone
      layer3 → lr_backbone * decay
      layer2 → lr_backbone * decay^2
      layer1 → lr_backbone * decay^3
    Shallower layers stay closer to ImageNet weights.
    """
    backbone_layers = {
        "layer4": lr_backbone,
        "layer3": lr_backbone * backbone_lr_decay,
        "layer2": lr_backbone * backbone_lr_decay ** 2,
        "layer1": lr_backbone * backbone_lr_decay ** 3,
    }
    param_groups = []
    for layer_name, layer_lr in backbone_layers.items():
        params = [p for n, p in model.named_parameters()
                  if n.startswith(layer_name) and p.requires_grad]
        if params:
            param_groups.append({"params": params, "lr": layer_lr,
                                 "name": f"backbone_{layer_name}"})
    backbone_names = tuple(backbone_layers.keys())
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith(backbone_names) and p.requires_grad]
    param_groups.append({"params": other_params, "lr": lr, "name": "transformer"})
    return param_groups


# ===================== Scheduler =====================
def build_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.05):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ===================== Inference =====================
@torch.no_grad()
def inference(model, test_dir, img_size, device, score_thresh, nms_thresh,
              output_path, use_focal, use_soft_nms=True):
    """
    Run detection on the test set.
    Collects (img_id, logits, boxes, meta) for all images first,
    then postprocesses in a second pass to keep the loop body clean.
    """
    model.eval()
    use_amp = device.type == "cuda"
    ds      = TestDataset(test_dir, img_size)
    dl      = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                         collate_fn=collate_fn_test, pin_memory=use_amp)

    # --- Pass 1: forward, collect raw outputs ---
    raw_records = []
    for imgs, img_ids, hs, ws, scales, pls, pts in tqdm(dl, desc="Inference"):
        imgs = imgs.to(device, non_blocking=True)
        with get_autocast(device, enabled=use_amp):
            out = model(imgs)
        raw_records.append({
            "img_id":  int(img_ids[0]),
            "logits":  out["pred_logits"][0].detach().cpu(),
            "boxes":   out["pred_boxes"][0].detach().cpu(),
            "orig_h":  hs[0], "orig_w": ws[0],
            "scale":   scales[0], "pl": pls[0], "pt": pts[0],
        })

    # --- Pass 2: postprocess and build submission list ---
    submission = []
    for rec in raw_records:
        preds = postprocess_single_image_predictions(
            rec["logits"], rec["boxes"],
            img_size, rec["scale"], rec["pl"], rec["pt"],
            rec["orig_w"], rec["orig_h"],
            score_thresh, nms_thresh,
            use_focal=use_focal, use_soft_nms=use_soft_nms,
        )
        for p in preds:
            submission.append({
                "image_id":    rec["img_id"],
                "category_id": p["category_id"],
                "bbox":        [round(v, 2) for v in p["bbox"]],
                "score":       round(p["score"], 6),
            })

    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(submission, fp)
    logging.getLogger().info(f"Inference done: {len(submission)} predictions -> {output_path}")


# ===================== TTA Inference =====================
@torch.no_grad()
def inference_tta(model, args, device):
    """
    Multi-scale Test-Time Augmentation: run at each scale in tta_scales,
    collect all predictions, then merge per-image per-class with Soft NMS.
    Typically adds +1~2 mAP over single-scale inference.
    """
    model.eval()
    use_amp   = device.type == "cuda"
    all_by_id = defaultdict(list)   # img_id → list of raw preds

    for img_size in args.tta_scales:
        ds = TestDataset(args.test_img_dir, img_size)
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                        collate_fn=collate_fn_test, pin_memory=(device.type == "cuda"))
        for imgs, img_ids, hs, ws, scales, pls, pts in tqdm(dl, desc=f"TTA scale={img_size}"):
            imgs = imgs.to(device, non_blocking=True)
            with get_autocast(device, enabled=use_amp):
                outputs = model(imgs)
            orig_h, orig_w = hs[0], ws[0]
            sc, pl, pt = scales[0], pls[0], pts[0]

            logits = outputs["pred_logits"][0].detach().cpu()
            boxes  = outputs["pred_boxes"][0].detach().cpu()
            probs  = logits.sigmoid() if args.use_focal else logits.softmax(-1)
            scores_, cls_ids_ = probs[:, 1:].max(-1); cls_ids_ = cls_ids_ + 1
            keep = scores_ > args.score_thresh * 0.5   # lower thresh before merging
            if not keep.any(): continue
            scores_  = scores_[keep]; cls_ids_ = cls_ids_[keep]; boxes = boxes[keep]

            boxes_xywh = convert_to_orig_coords(boxes, img_size, sc, pl, pt, orig_w, orig_h)
            boxes_xyxy = boxes_xywh.clone()
            boxes_xyxy[:, 2] += boxes_xyxy[:, 0]
            boxes_xyxy[:, 3] += boxes_xyxy[:, 1]

            img_id = int(img_ids[0])
            for q in range(len(scores_)):
                all_by_id[img_id].append({
                    "category_id": int(cls_ids_[q]),
                    "score":       float(scores_[q]),
                    "bbox":        boxes_xywh[q].tolist(),
                    "bbox_xyxy":   boxes_xyxy[q].tolist(),
                })

    # Per-image per-class Soft NMS to merge all scales
    results = []
    for img_id, preds in tqdm(all_by_id.items(), desc="Merging TTA"):
        for c in set(p["category_id"] for p in preds):
            cp = [p for p in preds if p["category_id"] == c]
            bx = torch.tensor([p["bbox_xyxy"] for p in cp])
            sc = torch.tensor([p["score"]     for p in cp])
            ki = soft_nms(bx, sc, sigma=0.5, score_thresh=args.score_thresh)
            for k in ki.tolist():
                results.append({
                    "image_id":    int(img_id),
                    "bbox":        [round(v, 2) for v in cp[k]["bbox"]],
                    "score":       round(float(sc[k]), 6),
                    "category_id": c,
                })

    out_path = os.path.join(args.output_dir, args.pred_file)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    logging.getLogger().info(
        f"TTA inference done ({len(args.tta_scales)} scales): "
        f"{len(results)} predictions → {out_path}"
    )


# ===================== Setup Helpers =====================
def _build_model(args, device):
    """Instantiate DigitDETR + EMA and move to device."""
    net = DigitDETR(
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        n_points=args.n_points,
        use_dn=args.use_dn,
        dn_number=args.dn_number,
        label_noise_ratio=args.label_noise_ratio,
        box_noise_scale=args.box_noise_scale,
    ).to(device)
    shadow = ModelEMA(net, decay=args.ema_decay)
    return net, shadow


def _restore_checkpoint(path, net, shadow, device):
    """Load a saved checkpoint; return (saved_dict, first_epoch, top_mAP, history)."""
    saved = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(saved["model"])
    if "ema" in saved:
        shadow.ema.load_state_dict(saved["ema"])
    first_epoch = int(saved.get("epoch", -1)) + 1
    top_mAP     = float(saved.get("best_mAP", 0.0))
    run_history = saved.get("history",
                            {"train_loss": [], "val_loss": [], "val_acc": [], "val_mAP": []})
    return saved, first_epoch, top_mAP, run_history


def _make_dataloaders(args, train_ds, val_ds, device):
    """Build train / val DataLoaders."""
    use_pin = device.type == "cuda"
    tr_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
        drop_last=True, pin_memory=use_pin,
    )
    va_dl = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
        pin_memory=use_pin,
    )
    return tr_dl, va_dl


# ===================== Main =====================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    # ---- Logger ----
    log_path = os.path.join(args.output_dir, "train.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="a"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    logger = logging.getLogger()
    logger.info(f"Args: {vars(args)}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"device={device} | img_size={args.img_size} | "
                f"use_dn={args.use_dn} | use_focal={args.use_focal}")

    # ---- Model + EMA ----
    net, shadow = _build_model(args, device)
    logger.info(f"Model params: {sum(p.numel() for p in net.parameters()):,}")

    first_epoch = 0
    top_mAP     = 0.0
    run_history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_mAP": []}
    saved       = None

    if args.resume:
        saved, first_epoch, top_mAP, run_history = _restore_checkpoint(
            args.resume, net, shadow, device)
        logger.info(f"Resumed {args.resume} -> epoch {first_epoch}, best mAP {top_mAP:.4f}")

    if args.do_train:
        # ---- Datasets ----
        train_ds = CocoDigitDataset(args.train_img_dir, args.train_ann,
                                    args.img_size, is_train=True)
        if os.path.isdir(args.copypaste_img_dir) and os.path.isfile(args.copypaste_ann):
            from torch.utils.data import ConcatDataset
            cp_ds    = CocoDigitDataset(args.copypaste_img_dir, args.copypaste_ann,
                                        args.img_size, is_train=True)
            train_ds = ConcatDataset([train_ds, cp_ds])
            logger.info(f"train_copypaste merged -> total train: {len(train_ds)}")
        else:
            logger.info(f"train_copypaste not found -> train size: {len(train_ds)}")

        val_ds = CocoDigitDataset(args.val_img_dir, args.val_ann,
                                  args.img_size, is_train=False)
        tr_dl, va_dl = _make_dataloaders(args, train_ds, val_ds, device)
        logger.info(f"train={len(train_ds)}  val={len(val_ds)}")

        # ---- Optimizer / Scheduler ----
        param_groups = build_param_groups(net, args.lr, args.lr_backbone, args.backbone_lr_decay)
        optim     = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        scheduler = build_warmup_cosine_scheduler(optim, args.warmup_epochs, args.epochs, args.min_lr_ratio)
        use_amp   = device.type == "cuda"
        scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)

        if saved is not None:
            if "optimizer" in saved: optim.load_state_dict(saved["optimizer"])
            if "scheduler" in saved: scheduler.load_state_dict(saved["scheduler"])

        # ---- Criterion ----
        matcher   = HungarianMatcher(args.cost_class, args.cost_bbox, args.cost_giou)
        criterion = SetCriterion(
            args.num_classes + 1, matcher,
            args.loss_ce, args.loss_bbox, args.loss_giou, args.eos_coef,
            dn_loss_coef=args.dn_loss_coef,
            use_focal=args.use_focal,
            focal_alpha=args.focal_alpha,
            focal_gamma=args.focal_gamma,
            aux_loss_decay=args.aux_loss_decay,
        ).to(device)

        # ---- Early stopping ----
        es_patience = args.early_stop_patience
        es_counter  = 0

        for epoch in range(first_epoch, args.epochs):
            net.train()
            running_loss = 0.0
            n_steps      = 0
            pbar = tqdm(tr_dl, desc=f"Epoch {epoch + 1}/{args.epochs}")
            for imgs, targets in pbar:
                if args.use_mosaic:
                    imgs, targets = mosaic_collate(
                        list(zip(imgs, targets)), train_ds,
                        args.img_size, prob=args.mosaic_prob)
                    imgs = torch.stack(imgs) if not isinstance(imgs, torch.Tensor) else imgs
                imgs    = imgs.to(device, non_blocking=True)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in t.items()} for t in targets]
                optim.zero_grad(set_to_none=True)
                with get_autocast(device, enabled=use_amp):
                    out         = net(imgs, targets if args.use_dn else None)
                    loss, ld, _ = criterion(out, targets)
                scaler.scale(loss).backward()
                if args.clip_max_norm > 0:
                    scaler.unscale_(optim)
                    nn.utils.clip_grad_norm_(net.parameters(), args.clip_max_norm)
                scaler.step(optim); scaler.update()
                shadow.update(net)

                running_loss += ld["loss"]; n_steps += 1
                pbar.set_postfix(
                    loss=f"{ld['loss']:.4f}", ce=f"{ld['loss_ce']:.3f}",
                    bbox=f"{ld['loss_bbox']:.3f}", giou=f"{ld['loss_giou']:.3f}",
                    dn=f"{ld['loss_dn']:.3f}",
                )

            scheduler.step()
            avg_loss = running_loss / max(n_steps, 1)
            run_history["train_loss"].append(avg_loss)

            if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
                val_loss, val_acc, val_mAP = evaluate(
                    shadow.ema, va_dl, criterion, device, args.img_size,
                    args.output_dir, epoch, args.num_classes,
                    args.val_score_thresh, args.nms_thresh,
                    args.use_focal, use_soft_nms=args.use_soft_nms,
                )
                run_history["val_loss"].append(val_loss.get("loss", 0.0))
                run_history["val_acc"].append(val_acc)
                run_history["val_mAP"].append(val_mAP)
                logger.info(
                    f"[Epoch {epoch + 1}] Train: {avg_loss:.4f} | "
                    f"Val: {val_loss.get('loss', 0.0):.4f} | "
                    f"Acc: {val_acc:.4f} | mAP(EMA): {val_mAP:.4f}"
                )

                def _save_ckpt(path):
                    torch.save({
                        "epoch": epoch, "model": net.state_dict(),
                        "ema": shadow.ema.state_dict(),
                        "optimizer": optim.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_mAP": top_mAP, "history": run_history,
                    }, path)

                if val_mAP > top_mAP:
                    top_mAP    = val_mAP
                    es_counter = 0
                    _save_ckpt(os.path.join(args.output_dir, "best.pth"))
                    logger.info(f"  ✓ New best mAP: {top_mAP:.4f} -> best.pth saved")
                else:
                    es_counter += 1
                    logger.info(f"  [EarlyStopping] {es_counter}/{es_patience} rounds without improvement.")
                    if es_patience > 0 and es_counter >= es_patience:
                        logger.info(f"  [EarlyStopping] Triggered at epoch {epoch + 1}.")
                        _save_ckpt(os.path.join(args.output_dir, "latest.pth"))
                        plot_curves(run_history, args.output_dir)
                        break
            else:
                run_history["val_loss"].append(None)
                run_history["val_acc"].append(None)
                run_history["val_mAP"].append(None)
                logger.info(f"[Epoch {epoch + 1}] Train: {avg_loss:.4f}")

            torch.save(
                {"epoch": epoch, "model": net.state_dict(),
                 "ema": shadow.ema.state_dict(),
                 "optimizer": optim.state_dict(),
                 "scheduler": scheduler.state_dict(),
                 "best_mAP": top_mAP, "history": run_history},
                os.path.join(args.output_dir, "latest.pth"),
            )
            if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
                plot_curves(run_history, args.output_dir)

        plot_curves(run_history, args.output_dir)
        logger.info(f"Training done! Best mAP: {top_mAP:.4f}")

    if args.do_infer:
        if not args.do_train and args.resume is None:
            raise ValueError("--do_infer requires --resume")
        if args.use_tta:
            inference_tta(shadow.ema, args, device)
        else:
            inference(shadow.ema, args.test_img_dir, args.img_size, device,
                      args.score_thresh, args.nms_thresh,
                      os.path.join(args.output_dir, args.pred_file),
                      args.use_focal, use_soft_nms=args.use_soft_nms)


if __name__ == "__main__":
    main()