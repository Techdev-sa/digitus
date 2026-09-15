"""CreaseNet — our own 4-keypoint detector for 2D:4D measurement.

Regresses (index_base, index_tip, ring_base, ring_tip) directly from the
image; no MediaPipe involved. MobileNetV3-small backbone (ImageNet weights)
with a small regression head, 384px letterboxed input, heavy augmentation so
it trains usefully from tens of annotated images and keeps improving as the
annotation set grows.

A trained checkpoint is NOT used for pre-placement automatically. It only
becomes "ready" once it beats the MediaPipe + fixed-offset heuristic on a
held-out validation split, both scored in the same units (pixel error /
hand size), with enough validation samples (>= MIN_VAL_FOR_READY) that the
comparison isn't noise. Below that bar the annotation server keeps using
MediaPipe for pre-placement no matter how many epochs CreaseNet has run —
"trained" and "good enough to trust" are different things.

Checkpoint: models/crease_net.pt
    {state_dict, trained_on, val_err_norm, val_err_px, point_names, input_size,
     val_count, model_err_hand, heuristic_err_hand, ready}

CLI:  python scripts/crease_net.py --epochs 80          # train from annotations/
      python scripts/crease_net.py --min-samples 3      # allow tiny datasets
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import mp_hands

REPO_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_DIR = REPO_ROOT / "annotations"
CHECKPOINT = REPO_ROOT / "models" / "crease_net.pt"
MIN_VAL_FOR_READY = 5  # below this many val samples, a "win" over MediaPipe is noise

POINT_NAMES = ["index_base", "index_tip", "ring_base", "ring_tip"]
INPUT_SIZE = 384
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pick_device(prefer: str = "4060") -> torch.device:
    """Use the GPU whose name matches `prefer` (default: the RTX 4060 Ti),
    else the first CUDA device, else CPU."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            if prefer.lower() in torch.cuda.get_device_name(i).lower():
                return torch.device(f"cuda:{i}")
        return torch.device("cuda:0")
    return torch.device("cpu")


def release_memory() -> None:
    """Return unused CPU/GPU blocks after training or a predictor swap."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def letterbox(bgr: np.ndarray, size: int = INPUT_SIZE):
    """Resize keeping aspect, pad to size x size. Returns (image, scale, dx, dy)
    with  point_lb = point_orig * scale + (dx, dy)."""
    h, w = bgr.shape[:2]
    s = size / max(h, w)
    nw, nh = round(w * s), round(h * s)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas = np.zeros((size, size, 3), dtype=resized.dtype)
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, s, dx, dy


def to_tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1))


class CreaseNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = mobilenet_v3_small(weights=weights)
        self.features = m.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 256), nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, len(POINT_NAMES) * 2),
            nn.Sigmoid(),  # coords normalised to [0, 1] in letterbox frame
        )

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def load_samples(ann_dir: Path = ANNOTATION_DIR) -> list[dict]:
    """Accepted annotations -> image_path, ground-truth points, and enough of the
    original MediaPipe context (hand size, offset_frac, detection flag) to
    reconstruct the exact heuristic prediction for a fair comparison later."""
    samples = []
    for p in sorted(ann_dir.glob("*.json")):
        try:
            a = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if a.get("status") != "accepted":
            continue

        img_path = REPO_ROOT / a["filtered_path"]
        if not img_path.exists():
            continue
        pts = np.array([[a["points"][n]["x"], a["points"][n]["y"]] for n in POINT_NAMES],
                       dtype=np.float32)
        mp = a.get("mediapipe") or {}
        samples.append({
            "image_path": img_path,
            "points": pts,
            # Set aside from the review page. Carried, not filtered: fixed_split
            # seeds a shuffle over len(samples), so dropping rows here renumbers
            # the whole split and silently moves the validation set. Measured on
            # 13 exclusions that put 68% of the new val set into the previous
            # run's training data, which would have made any before/after
            # comparison meaningless. The train side drops these after the split.
            "excluded": bool(a.get("excluded")),
            "offset_frac": a.get("offset_frac", 0.15),
            "hand_size_px": mp.get("hand_size_px"),
            "mp_detected": bool(mp.get("detected")),
        })
    return samples


def _heuristic_points(px: np.ndarray, offset_frac: float) -> np.ndarray:
    """The same MediaPipe + fixed-offset prediction app/server.py's bootstrap
    mode would have made, as a (4, 2) array in POINT_NAMES order."""
    return np.array([
        mp_hands.crease_estimate(px, mp_hands.INDEX_MCP, mp_hands.INDEX_PIP, offset_frac),
        px[mp_hands.INDEX_TIP],
        mp_hands.crease_estimate(px, mp_hands.RING_MCP, mp_hands.RING_PIP, offset_frac),
        px[mp_hands.RING_TIP],
    ])


def _eval_against_heuristic(model: nn.Module, val_samples: list[dict],
                            device: torch.device) -> dict:
    """Score CreaseNet and the MediaPipe heuristic on the same held-out images,
    both as mean point error normalised by hand size — the units a human
    annotator's drag distance is already measured in, so the comparison means
    something. Re-runs MediaPipe rather than trusting stored landmarks, since
    only 4 of its 21 points were ever saved."""
    model.eval()
    detector = mp_hands.HandDetector(num_hands=1)
    model_errs, heuristic_errs = [], []
    try:
        for s in val_samples:
            if not s["hand_size_px"] or not s["mp_detected"]:
                continue
            bgr = mp_hands.imread_unicode(s["image_path"])
            if bgr is None:
                continue
            hand_size = s["hand_size_px"]
            gt = s["points"]

            lb, sc, dx, dy = letterbox(bgr)
            with torch.no_grad():
                x = to_tensor(lb)[None].to(device)
                pred = model(x)[0].detach().cpu().numpy().reshape(4, 2)
                del x
            pred_full = (pred * INPUT_SIZE - np.array([dx, dy])) / sc
            model_errs.append(float(np.mean(np.linalg.norm(pred_full - gt, axis=1))) / hand_size)

            det = detector.detect(bgr)
            if det is not None:
                hpts = _heuristic_points(det["landmarks_px"], s["offset_frac"])
                heuristic_errs.append(float(np.mean(np.linalg.norm(hpts - gt, axis=1))) / hand_size)
            del bgr
    finally:
        detector.close()

    model_err = float(np.mean(model_errs)) if model_errs else None
    heuristic_err = float(np.mean(heuristic_errs)) if heuristic_errs else None
    ready = (model_err is not None and heuristic_err is not None
             and len(heuristic_errs) >= MIN_VAL_FOR_READY and model_err < heuristic_err)
    return {"model_err_hand": model_err, "heuristic_err_hand": heuristic_err,
            "val_count": len(val_samples), "val_compared": len(heuristic_errs), "ready": ready}


class CreaseDataset(Dataset):
    """Lazy dataset: decode + letterbox per sample, never keep full-res images."""

    def __init__(self, samples: list[dict], augment: bool):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        bgr = mp_hands.imread_unicode(self.samples[i]["image_path"])
        if bgr is None:
            bgr = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        pts = self.samples[i]["points"].copy()
        lb, s, dx, dy = letterbox(bgr)
        del bgr
        pts = pts * s + np.array([dx, dy], dtype=np.float32)

        if self.augment:
            # affine: rotation, scale, translation about the image centre
            ang = random.uniform(-20, 20)
            sc = random.uniform(0.85, 1.15)
            tx = random.uniform(-0.08, 0.08) * INPUT_SIZE
            ty = random.uniform(-0.08, 0.08) * INPUT_SIZE
            M = cv2.getRotationMatrix2D((INPUT_SIZE / 2, INPUT_SIZE / 2), ang, sc)
            M[:, 2] += (tx, ty)
            lb = cv2.warpAffine(lb, M, (INPUT_SIZE, INPUT_SIZE), flags=cv2.INTER_LINEAR)
            pts = (np.hstack([pts, np.ones((4, 1), np.float32)]) @ M.T).astype(np.float32)
            if random.random() < 0.5:  # mirror = other-hand chirality
                lb = lb[:, ::-1].copy()
                pts[:, 0] = INPUT_SIZE - 1 - pts[:, 0]
            # photometric jitter
            lb = lb.astype(np.float32)
            lb *= random.uniform(0.7, 1.3)                    # brightness
            lb = (lb - 128) * random.uniform(0.8, 1.2) + 128  # contrast
            lb = np.clip(lb, 0, 255).astype(np.uint8)

        target = torch.from_numpy((pts / INPUT_SIZE).clip(0, 1).astype(np.float32).ravel())
        return to_tensor(lb), target


def train_model(epochs: int = 80, min_samples: int = 10, batch_size: int = 16,
                lr: float = 3e-4, device: torch.device | None = None,
                checkpoint: Path = CHECKPOINT, log=print) -> dict | None:
    """Train from scratch on all accepted annotations; returns checkpoint meta."""
    samples = load_samples()
    if len(samples) < min_samples:
        log(f"crease_net: only {len(samples)} samples (< {min_samples}); not training")
        return None
    device = device or pick_device()

    random.seed(0)
    idx = list(range(len(samples)))
    random.shuffle(idx)
    n_val = max(1, len(samples) // 7) if len(samples) >= 20 else 0
    val_samples = [samples[i] for i in idx[:n_val]]
    train_samples = [samples[i] for i in idx[n_val:]]

    train_ds = CreaseDataset(train_samples, augment=True)
    val_ds = CreaseDataset(val_samples, augment=False) if val_samples else None
    dl_kw = dict(num_workers=0, pin_memory=False)
    train_dl = DataLoader(train_ds, batch_size=min(batch_size, len(train_samples)),
                          shuffle=True, **dl_kw)
    val_dl = DataLoader(val_ds, batch_size=batch_size, **dl_kw) if val_ds else None

    model = CreaseNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss(beta=0.01)

    best = math.inf
    best_state = None
    eval_model = None
    try:
        for ep in range(epochs):
            model.train()
            tr_loss = 0.0
            for x, y in train_dl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                tr_loss += loss.item() * len(x)
                del x, y, loss
            sched.step()
            tr_loss /= len(train_dl.dataset)

            if val_dl:
                model.eval()
                errs = []
                with torch.no_grad():
                    for x, y in val_dl:
                        pred = model(x.to(device)).cpu()
                        # mean per-point distance in normalised letterbox units
                        e = (pred.view(-1, 4, 2) - y.view(-1, 4, 2)).norm(dim=2).mean(dim=1)
                        errs.extend(e.tolist())
                        del x, y, pred, e
                metric = float(np.mean(errs))
            else:
                metric = tr_loss
            if metric < best:
                best = metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if (ep + 1) % 10 == 0:
                log(f"crease_net: epoch {ep + 1}/{epochs} train_loss={tr_loss:.5f} metric={best:.5f}")

        val_err_norm = best if val_dl else float("nan")

        if val_samples:
            eval_model = CreaseNet(pretrained=False).to(device)
            eval_model.load_state_dict(best_state)
            comparison = _eval_against_heuristic(eval_model, val_samples, device)
        else:
            comparison = {"model_err_hand": None, "heuristic_err_hand": None,
                         "val_count": 0, "val_compared": 0, "ready": False}

        meta = {
            "state_dict": best_state,
            "trained_on": len(samples),
            "val_err_norm": val_err_norm,
            "val_err_px_at_1600": round(val_err_norm * 1600, 1) if val_dl else None,
            "point_names": POINT_NAMES,
            "input_size": INPUT_SIZE,
            **comparison,
        }
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        tmp = checkpoint.with_suffix(".tmp")
        torch.save(meta, tmp)
        tmp.replace(checkpoint)  # atomic swap so a live server never reads a partial file
        verdict = "READY (beats MediaPipe)" if comparison["ready"] else "not ready yet"
        log(f"crease_net: saved {checkpoint} (trained_on={len(samples)}, {verdict}, "
            f"model_err={comparison['model_err_hand']}, heuristic_err={comparison['heuristic_err_hand']}, "
            f"val_compared={comparison['val_compared']})")
        # caller only needs scalars; do not hand back the weight tensors
        return {k: v for k, v in meta.items() if k != "state_dict"}
    finally:
        del train_dl, val_dl, train_ds, val_ds, model, opt, sched, eval_model, best_state
        release_memory()


class CreasePredictor:
    """Inference wrapper the annotation server uses for pre-placement."""

    def __init__(self, checkpoint: Path = CHECKPOINT, device: torch.device | None = None):
        self.device = device or pick_device()
        meta = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = CreaseNet(pretrained=False)
        self.model.load_state_dict(meta["state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.trained_on = meta["trained_on"]
        self.val_err_norm = meta.get("val_err_norm")
        self.ready = bool(meta.get("ready", False))
        self.model_err_hand = meta.get("model_err_hand")
        self.heuristic_err_hand = meta.get("heuristic_err_hand")
        self.val_count = meta.get("val_count", 0)
        self.val_compared = meta.get("val_compared", 0)
        del meta

    def close(self) -> None:
        self.model = None
        release_memory()

    @torch.no_grad()
    def predict(self, bgr: np.ndarray) -> dict[str, tuple[float, float]]:
        lb, s, dx, dy = letterbox(bgr)
        x = to_tensor(lb)[None].to(self.device)
        pred = self.model(x)[0].detach().cpu().numpy()
        del x
        pts = pred.reshape(4, 2) * INPUT_SIZE
        pts = (pts - np.array([dx, dy])) / s
        h, w = bgr.shape[:2]
        pts[:, 0] = pts[:, 0].clip(0, w - 1)
        pts[:, 1] = pts[:, 1].clip(0, h - 1)
        return {n: (float(p[0]), float(p[1])) for n, p in zip(POINT_NAMES, pts)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    args = ap.parse_args()
    train_model(epochs=args.epochs, min_samples=args.min_samples,
                batch_size=args.batch_size, checkpoint=args.checkpoint)