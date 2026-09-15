"""Overnight synthetic palmar data + heatmap CreaseNet.

Allowed outputs:
  data/synth_crease/
  models/synth_heatmap*.pt
  logs/synth_*.log
  logs/synth_results.jsonl
  %TEMP%/datahunt/overnight_summary.md

Does not touch models/crease_net.pt. Does not touch port 5000.
"""
from __future__ import annotations

import atexit
import json
import math
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import crease_net as cn  # noqa: E402

POINT_NAMES = getattr(cn, "POINT_NAMES", getattr(cn, "POINT_NAMES"))
INPUT_SIZE = int(getattr(cn, "INPUT_SIZE", getattr(cn, "INPUT_SIZE", 384)))
letterbox = getattr(cn, "letterbox", getattr(cn, "letterbox"))
to_tensor = getattr(cn, "to_tensor", getattr(cn, "to_tensor"))
pick_device = getattr(cn, "pick_device", getattr(cn, "pick_device"))
CreaseDataset = getattr(cn, "CreaseDataset", getattr(cn, "CreaseDataset"))
release_memory = getattr(cn, "release_memory", lambda: None)
load_samples = getattr(cn, "load_samples", getattr(cn, "load_samples"))
imread = cn.mp_hands.imread_unicode

SYNTH_DIR = REPO / "data" / "synth_crease"
IMG_DIR = SYNTH_DIR / "images"
PREV_DIR = SYNTH_DIR / "preview"
LABELS = SYNTH_DIR / "labels.jsonl"
MANIFEST = SYNTH_DIR / "manifest.json"
MODELS = REPO / "models"
LOGS = REPO / "logs"
RESULTS = LOGS / "synth_results.jsonl"
GEN_LOG = LOGS / "synth_gen.log"
TRAIN_LOG = LOGS / "synth_train.log"
SUMMARY = Path(os.environ.get("TEMP", str(REPO))) / "datahunt" / "overnight_summary.md"
PROD_CKPT = Path(getattr(cn, "CHECKPOINT", REPO / "models" / "crease_net.pt"))
LOCK_DIR = LOGS / "ablate_gpu.lockdir"

HEATMAP_POINT = 0.017745004892690642
HEATMAP_RATIO = 0.023620177849050558
SPLIT_SEED = 0
MAX_TRAIN = 1200
MAX_VAL = 200
N_ROUND = 10_000
EPOCHS = 25
BATCH = 16
LR = 3e-4
REAL_REPEAT = 8


def log_to(path: Path, msg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def imwrite_u(path: Path, bgr: np.ndarray, q: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError(f"encode failed {path}")
    buf.tofile(str(path))


class CreaseHeatmapNet(nn.Module):
    def __init__(self, pretrained: bool = True, upsample: int = 4):
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = mobilenet_v3_small(weights=weights)
        self.features = m.features
        self.head = nn.Sequential(
            nn.Conv2d(576, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.Hardswish(inplace=True),
            nn.Conv2d(128, 4, 1),
        )
        self.upsample = upsample

    def forward(self, x):
        heat = self.head(self.features(x))
        if self.upsample > 1:
            heat = nn.functional.interpolate(
                heat, scale_factor=self.upsample, mode="bilinear", align_corners=False
            )
        return soft_argmax_2d(heat)


def soft_argmax_2d(heatmaps: torch.Tensor) -> torch.Tensor:
    b, k, h, w = heatmaps.shape
    prob = torch.softmax(heatmaps.view(b, k, -1), dim=-1).view(b, k, h, w)
    xs = (torch.arange(w, device=heatmaps.device, dtype=heatmaps.dtype) + 0.5) / w
    ys = (torch.arange(h, device=heatmaps.device, dtype=heatmaps.dtype) + 0.5) / h
    x = (prob * xs.view(1, 1, 1, w)).sum(dim=(2, 3))
    y = (prob * ys.view(1, 1, h, 1)).sum(dim=(2, 3))
    return torch.stack([x, y], dim=-1).reshape(b, k * 2)


def add_dataset(samples: list[dict]) -> None:
    for s in samples:
        if "image_path" not in s:
            s["image_path"] = s.get("image_path") or s.get("image_path")
        if s.get("hand_size_px") is None:
            s["hand_size_px"] = s.get("hand_size_px") or s.get("hand_size_px")
        p = str(s["image_path"]).lower()
        if "mendeley" in p or "2d4d" in p:
            s["dataset"] = "mendeley_2d4d"
        else:
            s["dataset"] = "11k_hands"


def fixed_split(samples: list[dict]):
    rng = random.Random(SPLIT_SEED)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    n_val = max(1, len(samples) // 7) if len(samples) >= 20 else 0
    val_s = [samples[i] for i in idx[:n_val][:MAX_VAL]]
    train_s = [samples[i] for i in idx[n_val:][:MAX_TRAIN]]
    info = {
        "n_accepted": len(samples),
        "n_val_full": n_val,
        "n_train_full": len(samples) - n_val,
        "n_train_used": len(train_s),
        "n_val_used": len(val_s),
        "max_train": MAX_TRAIN,
        "max_val": MAX_VAL,
        "split_seed": SPLIT_SEED,
        "train_mendeley": sum(1 for s in train_s if "mendeley" in s.get("dataset", "")),
        "val_mendeley": sum(1 for s in val_s if "mendeley" in s.get("dataset", "")),
    }
    return train_s, val_s, info


@torch.no_grad()
def evaluate(model: nn.Module, val_samples: list[dict], device: torch.device) -> dict:
    model.eval()
    ratio_aes, point_hand = [], []
    for s in val_samples:
        bgr = imread(s["image_path"])
        if bgr is None:
            continue
        lb, sc, dx, dy = letterbox(bgr)
        x = to_tensor(lb)[None].to(device)
        pred = model(x)[0].detach().cpu().numpy().reshape(4, 2)
        del x
        pred_full = (pred * INPUT_SIZE - np.array([dx, dy], dtype=np.float32)) / sc
        gt = s["points"]
        l2 = float(np.linalg.norm(pred_full[1] - pred_full[0]))
        l4 = float(np.linalg.norm(pred_full[3] - pred_full[2]))
        l2t = float(np.linalg.norm(gt[1] - gt[0]))
        l4t = float(np.linalg.norm(gt[3] - gt[2]))
        if l4 > 1e-6 and l4t > 1e-6:
            ratio_aes.append(abs(l2 / l4 - l2t / l4t))
        hs = s.get("hand_size_px") or s.get("hand_size_px")
        if hs:
            pe = float(np.mean(np.linalg.norm(pred_full - gt, axis=1))) / float(hs)
            point_hand.append(pe)
        del bgr
    aes = np.array(ratio_aes, dtype=np.float64) if ratio_aes else np.array([np.nan])
    return {
        "n_ratio": int(len(ratio_aes)),
        "ratio_mae": float(np.mean(aes)),
        "ratio_median_ae": float(np.median(aes)),
        "pct_le_0.008": float(np.mean(aes <= 0.008) * 100.0),
        "pct_le_0.02": float(np.mean(aes <= 0.02) * 100.0),
        "n_point": int(len(point_hand)),
        "point_err_hand": float(np.mean(point_hand)) if point_hand else float("nan"),
        "point_err_hand": float(np.mean(point_hand)) if point_hand else float("nan"),
    }


def other_train_pids() -> list[int]:
    me = os.getpid()
    pids = []
    try:
        import psutil
        for p in psutil.process_iter(["pid", "cmdline"]):
            if p.info["pid"] == me:
                continue
            cmd = " ".join(p.info.get("cmdline") or [])
            if "ablate_train.py" in cmd or "run_overnight.py" in cmd:
                pids.append(p.info["pid"])
    except Exception:
        pass
    return pids


def release_gpu_lock() -> None:
    try:
        pid_file = LOCK_DIR / "pid.txt"
        if pid_file.exists() and pid_file.read_text(encoding="utf-8").strip() == str(os.getpid()):
            shutil.rmtree(LOCK_DIR, ignore_errors=True)
    except Exception:
        pass


def acquire_gpu_lock(log) -> None:
    while True:
        others = other_train_pids()
        if others:
            log(f"gpu lock: waiting for pid={others}")
            time.sleep(20)
            continue
        try:
            LOCK_DIR.mkdir()
            (LOCK_DIR / "pid.txt").write_text(str(os.getpid()), encoding="utf-8")
            atexit.register(release_gpu_lock)
            log(f"gpu lock: acquired pid={os.getpid()}")
            return
        except FileExistsError:
            old = None
            try:
                old = int((LOCK_DIR / "pid.txt").read_text(encoding="utf-8").strip())
            except Exception:
                pass
            alive = False
            if old is not None:
                try:
                    import psutil
                    alive = psutil.pid_exists(old)
                except Exception:
                    alive = True
            if not alive:
                log(f"gpu lock: stale pid={old}, removing")
                shutil.rmtree(LOCK_DIR, ignore_errors=True)
                continue
            log(f"gpu lock: held by pid={old}")
            time.sleep(15)


def rot_xyz(ax, ay, az):
    ca, sa = math.cos(ax), math.sin(ax)
    cb, sb = math.cos(ay), math.sin(ay)
    cc, sc = math.cos(az), math.sin(az)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], np.float32)
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], np.float32)
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]], np.float32)
    return Rz @ Ry @ Rx


def make_hand_3d(rng: np.random.Generator):
    """Palmar +Z, fingers +Y. Proximal digital crease is distal to MCP on palmar skin."""
    palm_w, palm_h, thick = 1.05, 1.15, 0.22
    r24 = float(rng.uniform(0.90, 1.02))
    ring_len = float(rng.uniform(0.92, 1.08))
    index_len = ring_len * r24
    mid_len = ring_len * float(rng.uniform(1.02, 1.10))
    pinky_len = ring_len * float(rng.uniform(0.72, 0.85))
    thumb_len = ring_len * float(rng.uniform(0.62, 0.75))
    fingers = [
        ("thumb", -0.62, 0.22, thumb_len),
        ("index", -0.36, 0.18, index_len),
        ("middle", -0.08, 0.19, mid_len),
        ("ring", 0.18, 0.175, ring_len),
        ("pinky", 0.42, 0.15, pinky_len),
    ]
    crease_frac = float(rng.uniform(0.16, 0.22))
    verts, faces = [], []

    def add_box(cx, cy, cz, sx, sy, sz):
        base = len(verts)
        for dz in (-sz / 2, sz / 2):
            for dy in (-sy / 2, sy / 2):
                for dx in (-sx / 2, sx / 2):
                    verts.append([cx + dx, cy + dy, cz + dz])
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        for a, b, c, d in quads:
            faces.append((base + a, base + b, base + c))
            faces.append((base + a, base + c, base + d))

    add_box(0.0, 0.0, 0.0, palm_w, palm_h, thick)
    add_box(-0.38, -0.15, 0.04, 0.42, 0.55, thick * 0.9)
    landmarks = {}
    for name, x, w, length in fingers:
        y0 = 0.05 if name == "thumb" else palm_h / 2 - 0.02
        if name == "thumb":
            add_box(x, y0 + length * 0.35, 0.02, w, length, thick * 0.75)
            continue
        add_box(x, y0 + length / 2, 0.0, w, length, thick * 0.72)
        # Palmar skin faces -Z so a +Z-looking camera sees the palm, not the dorsum.
        mcp = np.array([x, y0, -thick / 2], np.float32)
        tip = np.array([x, y0 + length, -thick / 2], np.float32)
        crease = mcp + (tip - mcp) * crease_frac
        if name == "index":
            landmarks["index_base"] = crease
            landmarks["index_tip"] = tip
        elif name == "ring":
            landmarks["ring_base"] = crease
            landmarks["ring_tip"] = tip
    return np.array(verts, np.float32), np.array(faces, np.int32), landmarks, r24


def project_points(X, R, t, f, cx, cy):
    Xc = (R @ X.T).T + t
    z = np.clip(Xc[:, 2], 1e-4, None)
    u = f * Xc[:, 0] / z + cx
    v = f * Xc[:, 1] / z + cy
    return np.stack([u, v], 1), Xc[:, 2]


def render_hand_3d(rng: np.random.Generator, w=960, h=1280):
    V, faces, lm, r24 = make_hand_3d(rng)
    R = rot_xyz(float(rng.uniform(-0.35, 0.35)),
                float(rng.uniform(-0.40, 0.40)),
                float(rng.uniform(-0.55, 0.55)))
    dist = float(rng.uniform(3.4, 4.6))
    t = np.array([float(rng.uniform(-0.15, 0.15)),
                  float(rng.uniform(-0.20, 0.10)), dist], np.float32)
    f = float(rng.uniform(900, 1300))
    cx, cy = w / 2.0, h / 2.0
    uv, z = project_points(V, R, t, f, cx, cy)
    names = ["index_base", "index_tip", "ring_base", "ring_tip"]
    lm_uv, _ = project_points(np.stack([lm[n] for n in names], 0), R, t, f, cx, cy)
    if not (np.all(lm_uv[:, 0] > 8) and np.all(lm_uv[:, 0] < w - 8)
            and np.all(lm_uv[:, 1] > 8) and np.all(lm_uv[:, 1] < h - 8)):
        return None

    bg = np.array([int(rng.integers(190, 250)),
                   int(rng.integers(190, 250)),
                   int(rng.integers(190, 250))], np.uint8)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bg
    img = np.clip(img.astype(np.float32) + rng.normal(0, 4.0, (h, w, 1)), 0, 255).astype(np.uint8)
    base = np.array([float(rng.uniform(70, 130)),
                     float(rng.uniform(110, 180)),
                     float(rng.uniform(160, 230))], np.float32)
    order = np.argsort(z[faces].mean(axis=1))[::-1]
    for fi in order:
        tri = faces[int(fi)]
        pts = np.round(uv[tri]).astype(np.int32)
        n3 = np.cross(V[tri[1]] - V[tri[0]], V[tri[2]] - V[tri[0]])
        n3 = n3 / (np.linalg.norm(n3) + 1e-8)
        lambert = float(np.clip(0.45 + 0.55 * max(0.0, (R @ n3)[2]), 0.25, 1.15))
        col = tuple(int(c) for c in np.clip(base * lambert, 0, 255))
        cv2.fillConvexPoly(img, pts, col)

    palm_uv, _ = project_points(
        np.array([[0.0, -0.1, -0.12], [0.25, 0.2, -0.12], [-0.3, 0.15, -0.12]], np.float32),
        R, t, f, cx, cy)
    crease_col = tuple(int(c) for c in np.clip(base * 0.55, 0, 255))
    cv2.line(img, tuple(palm_uv[0].astype(int)), tuple(palm_uv[1].astype(int)), crease_col, 2, cv2.LINE_AA)
    cv2.line(img, tuple(palm_uv[0].astype(int)), tuple(palm_uv[2].astype(int)), crease_col, 2, cv2.LINE_AA)
    for b_i, t_i in ((0, 1), (2, 3)):
        b = lm_uv[b_i]
        tp = lm_uv[t_i]
        v = tp - b
        n = np.array([-v[1], v[0]], np.float32)
        n = n / (np.linalg.norm(n) + 1e-6) * 14.0
        cv2.line(img, tuple((b - n).astype(int)), tuple((b + n).astype(int)),
                 tuple(int(c) for c in np.clip(base * 0.42, 0, 90)), 3, cv2.LINE_AA)
        cv2.circle(img, tuple(lm_uv[t_i].astype(int)), 6,
                   tuple(int(c) for c in np.clip(base * 1.15, 0, 255)), -1, cv2.LINE_AA)

    yy, xx = np.mgrid[0:h, 0:w]
    lx, ly = int(rng.integers(w // 5, 4 * w // 5)), int(rng.integers(h // 5, 4 * h // 5))
    att = 0.78 + 0.22 * np.clip(1 - np.sqrt((xx - lx) ** 2 + (yy - ly) ** 2) / (0.85 * max(w, h)), 0, 1)
    img = np.clip(img.astype(np.float32) * att[..., None], 0, 255).astype(np.uint8)
    hs = float(np.linalg.norm(lm_uv[1] - lm_uv[0]) + np.linalg.norm(lm_uv[3] - lm_uv[2])) * 0.65
    return img, lm_uv.astype(np.float32), hs, r24


def tps_maps(h, w, src, dst, grid=36):
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    n = len(src)

    def U(r2):
        r2 = np.maximum(r2, 1e-12)
        return r2 * np.log(r2)

    K = U(((src[:, None, :] - src[None, :, :]) ** 2).sum(-1))
    P = np.hstack([np.ones((n, 1)), src])
    L = np.zeros((n + 3, n + 3), np.float64)
    L[:n, :n] = K
    L[:n, n:] = P
    L[n:, :n] = P.T
    wx = np.linalg.lstsq(L, np.concatenate([dst[:, 0], np.zeros(3)]), rcond=None)[0]
    wy = np.linalg.lstsq(L, np.concatenate([dst[:, 1], np.zeros(3)]), rcond=None)[0]
    xs = np.linspace(0, w - 1, grid)
    ys = np.linspace(0, h - 1, grid)
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel()], 1)
    r2 = ((pts[:, None, :] - src[None, :, :]) ** 2).sum(-1)
    u = U(r2)
    Px = np.hstack([np.ones((len(pts), 1)), pts])
    mapx = cv2.resize((u @ wx[:n] + Px @ wx[n:]).reshape(grid, grid).astype(np.float32), (w, h))
    mapy = cv2.resize((u @ wy[:n] + Px @ wy[n:]).reshape(grid, grid).astype(np.float32), (w, h))
    return mapx, mapy


def warp_real(bgr, pts, rng: np.random.Generator):
    h, w = bgr.shape[:2]
    ib, it, rb, rt = [pts[i].astype(np.float64) for i in range(4)]
    s2 = float(rng.uniform(0.80, 0.88))
    s4 = float(rng.uniform(0.80, 0.88))
    it2, rt2 = ib + (it - ib) * s2, rb + (rt - rb) * s4
    mid_i, mid_r = ib + 0.5 * (it - ib), rb + 0.5 * (rt - rb)
    mid_i2, mid_r2 = ib + 0.5 * (it2 - ib), rb + 0.5 * (rt2 - rb)
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)
    palm = (ib + rb) / 2
    src = np.vstack([ib, it, rb, rt, mid_i, mid_r, palm, corners])
    dst = np.vstack([ib, it2, rb, rt2, mid_i2, mid_r2, palm, corners])
    mapx, mapy = tps_maps(h, w, dst, src)
    img = cv2.remap(bgr, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    new_pts = np.stack([ib, it2, rb, rt2], 0).astype(np.float32)
    img = img.astype(np.float32) * float(rng.uniform(0.75, 1.25))
    img = np.clip((img - 128) * float(rng.uniform(0.85, 1.2)) + 128, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    if rng.random() < 0.35:
        img = cv2.flip(img, 1)
        new_pts[:, 0] = w - 1 - new_pts[:, 0]
    ang = float(rng.uniform(-18, 18))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, float(rng.uniform(0.92, 1.08)))
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    new_pts = (np.hstack([new_pts, np.ones((4, 1), np.float32)]) @ M.T).astype(np.float32)
    if not (np.all(new_pts[:, 0] > 2) and np.all(new_pts[:, 0] < w - 2)
            and np.all(new_pts[:, 1] > 2) and np.all(new_pts[:, 1] < h - 2)):
        return None
    hs = float(np.linalg.norm(new_pts[1] - new_pts[0]) + np.linalg.norm(new_pts[3] - new_pts[2])) * 0.65
    return img, new_pts, hs


def draw_preview(bgr, pts, path: Path):
    vis = bgr.copy()
    cols = [(0, 90, 255), (0, 220, 255), (40, 200, 40), (255, 80, 80)]
    for i, p in enumerate(pts):
        cv2.circle(vis, (int(p[0]), int(p[1])), 8, cols[i], 2, cv2.LINE_AA)
    cv2.line(vis, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)), (0, 180, 255), 1)
    cv2.line(vis, tuple(pts[2].astype(int)), tuple(pts[3].astype(int)), (40, 180, 40), 1)
    imwrite_u(path, vis, 85)


def generate_batch(train_real: list[dict], n_total: int, start_i: int, log) -> int:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    PREV_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1000 + start_i)
    n_3d = n_total * 4 // 10
    n_warp = n_total - n_3d
    written = 0
    labels = []
    log(f"generate: start_i={start_i} n_3d={n_3d} n_warp={n_warp}")

    got = 0
    tries = 0
    while got < n_3d and tries < n_3d * 8:
        tries += 1
        out = render_hand_3d(rng)
        if out is None:
            continue
        img, pts, hs, r24 = out
        i = start_i + written
        rel = f"images/3d_{i:06d}.jpg"
        imwrite_u(IMG_DIR / f"3d_{i:06d}.jpg", img)
        labels.append({"rel": rel, "kind": "3d", "r24": r24, "hand_size_px": hs, "points": pts.tolist()})
        if got < 8:
            draw_preview(img, pts, PREV_DIR / f"3d_{i:06d}.jpg")
        written += 1
        got += 1
        if got % 500 == 0:
            log(f"generate 3d {got}/{n_3d}")
    log(f"generate: 3d done {got} tries={tries}")

    real_ok = train_real
    gotw = 0
    tries = 0
    while gotw < n_warp and tries < n_warp * 6:
        tries += 1
        s = real_ok[int(rng.integers(0, len(real_ok)))]
        bgr = imread(s["image_path"])
        if bgr is None:
            continue
        out = warp_real(bgr, s["points"], rng)
        if out is None:
            continue
        img, pts, hs = out
        i = start_i + written
        rel = f"images/warp_{i:06d}.jpg"
        imwrite_u(IMG_DIR / f"warp_{i:06d}.jpg", img)
        labels.append({"rel": rel, "kind": "warp", "src": Path(s["image_path"]).name,
                       "hand_size_px": hs, "points": pts.tolist()})
        if gotw < 8:
            draw_preview(img, pts, PREV_DIR / f"warp_{i:06d}.jpg")
        written += 1
        gotw += 1
        if gotw % 500 == 0:
            log(f"generate warp {gotw}/{n_warp}")
    log(f"generate: warp done {gotw} tries={tries}")

    with LABELS.open("a", encoding="utf-8") as f:
        for rec in labels:
            f.write(json.dumps(rec) + "\n")
    old = {}
    if MANIFEST.exists():
        try:
            old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            old = {}
    old.setdefault("rounds", []).append({"n_written": written, "start_i": start_i, "n_3d": got, "n_warp": gotw})
    old["n_images"] = old.get("n_images", 0) + written
    MANIFEST.write_text(json.dumps(old, indent=2), encoding="utf-8")
    log(f"generate: wrote {written} (running total {old['n_images']})")
    return written


def load_synth_samples() -> list[dict]:
    files = []
    if LABELS.exists():
        files.append(LABELS)
    files.extend(sorted(SYNTH_DIR.glob("labels_*.jsonl")))
    seen = set()
    out = []
    for fp in files:
        try:
            lines = fp.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            rec = json.loads(line)
            rel = rec.get("rel") or rec.get("rel") or rec.get("path")
            if not rel or rel in seen:
                continue
            p = SYNTH_DIR / rel
            if not p.exists():
                continue
            seen.add(rel)
            hs = rec.get("hand_size_px")
            if hs is None:
                hs = rec.get("hand_size_px")
            out.append({
                "image_path": p,
                "points": np.array(rec["points"], np.float32),
                "hand_size_px": hs,
                "dataset": "synth",
                "offset_frac": 0.15,
                "mp_detected": True,
            })
    return out


def train_heatmap(train_real, synth, val_real, ckpt: Path, log) -> dict:
    if ckpt.resolve() == PROD_CKPT.resolve() or "crease_net.pt" in ckpt.name:
        raise RuntimeError("refusing to write prod checkpoint")
    if not ckpt.name.startswith("synth_heatmap"):
        raise RuntimeError(f"unexpected ckpt name {ckpt}")

    device = pick_device()
    log(f"train: device={device} real={len(train_real)} synth={len(synth)} val={len(val_real)}")
    mixed = list(train_real) * REAL_REPEAT + list(synth)
    random.Random(SPLIT_SEED).shuffle(mixed)
    ds = CreaseDataset(mixed, True)
    dl = DataLoader(ds, batch_size=min(BATCH, len(mixed)), shuffle=True, num_workers=0)
    model = CreaseHeatmapNet(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    l1 = nn.L1Loss()
    best, best_ep, best_state = math.inf, -1, None
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for ep in range(EPOCHS):
        model.train()
        tr, nseen = 0.0, 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = l1(pred, y)
            loss.backward()
            opt.step()
            tr += loss.item() * len(x)
            nseen += len(x)
            del x, y, pred, loss
        sched.step()
        tr /= max(nseen, 1)
        metrics = evaluate(model, val_real, device)
        sel = metrics["point_err_hand"]
        star = ""
        if sel < best:
            best = sel
            best_ep = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            star = " *"
        log(f"heatmap: epoch {ep+1}/{EPOCHS} train_loss={tr:.5f} "
            f"val_point={metrics['point_err_hand']:.5f} val_ratio={metrics['ratio_mae']:.5f}{star}")

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_ep = EPOCHS
    eval_m = CreaseHeatmapNet(pretrained=False).to(device)
    eval_m.load_state_dict(best_state)
    metrics = evaluate(eval_m, val_real, device)
    minutes = (time.perf_counter() - t0) / 60.0
    vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else None
    meta = {
        "state_dict": best_state, "head": "heatmap",
        "n_real": len(train_real), "n_synth": len(synth), "n_val": len(val_real),
        "epochs": EPOCHS, "best_epoch": best_ep, "minutes": minutes, "peak_vram_mb": vram,
        **metrics,
    }
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    tmp = ckpt.with_suffix(".tmp")
    torch.save(meta, tmp)
    tmp.replace(ckpt)
    log(f"saved {ckpt} best_epoch={best_ep} point/hand={metrics['point_err_hand']:.6f} "
        f"ratio_mae={metrics['ratio_mae']:.6f} minutes={minutes:.2f}")
    del model, eval_m, opt, dl, ds, best_state
    release_memory()
    return {k: v for k, v in meta.items() if k != "state_dict"}


def append_result(row: dict) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def write_summary(rows: list[dict], extra: str) -> None:
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# overnight synth heatmap",
        "",
        f"written {datetime.now(timezone.utc).isoformat()}",
        "",
        "No `hands3d_*.md` in %TEMP%\\datahunt. MANO/HTML are gated. "
        "Used a palmar 3D mesh with proximal-digital-crease vertices (distal to MCP) "
        "plus TPS finger-length warps of the real 1200-train photos. "
        "Eval is the same 200 real val images as the heatmap ablation.",
        "",
        f"heatmap ablation bar: point/hand={HEATMAP_POINT:.6f} ratio_mae={HEATMAP_RATIO:.6f}",
        "",
    ]
    for r in rows:
        lines.append(
            f"- {r.get('run')}: n_synth={r.get('n_synth')} "
            f"point/hand={r.get('point_err_hand'):.6f} "
            f"ratio_mae={r.get('ratio_mae'):.6f} minutes={r.get('minutes')}"
        )
    lines += ["", extra, ""]
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")


def count_synth_files() -> int:
    return sum(1 for _ in IMG_DIR.glob("*.jpg")) if IMG_DIR.exists() else 0


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    train_only = len(sys.argv) > 1 and sys.argv[1] in {"train", "train-only"}

    def glog(m): log_to(GEN_LOG, m)
    def tlog(m): log_to(TRAIN_LOG, m)

    glog("=== overnight synth start ===")
    glog(f"prod checkpoint left alone: {PROD_CKPT}")
    samples = load_samples()
    add_dataset(samples)
    train_real, val_real, split = fixed_split(samples)
    glog(f"split {json.dumps(split)}")
    val_ids = {str(s["image_path"]) for s in val_real}

    rows = []
    prev_point = HEATMAP_POINT
    round_i = 0
    extra_note = ""

    while True:
        round_i += 1
        have = 0 if train_only else count_synth_files()
        want = N_ROUND * round_i
        if not train_only and have < want:
            glog(f"round {round_i}: generating {want - have} (have {have})")
            generate_batch(train_real, want - have, start_i=have, log=glog)
        synth = [s for s in load_synth_samples() if str(s["image_path"]) not in val_ids]
        glog(f"round {round_i}: synth usable {len(synth)}")
        if len(synth) < 10000 and round_i == 1:
            extra_note = f"STOP: only {len(synth)} synth images"
            glog(extra_note)
            write_summary(rows, extra_note)
            return

        ckpt = MODELS / f"synth_heatmap_r{round_i}.pt"
        tlog(f"=== train round {round_i} n_synth={len(synth)} ===")
        acquire_gpu_lock(tlog)
        try:
            row = train_heatmap(train_real, synth, val_real, ckpt, tlog)
        finally:
            release_gpu_lock()
        row["run"] = f"synth_r{round_i}"
        row["split"] = split
        row["vs_heatmap_point"] = HEATMAP_POINT
        row["vs_heatmap_ratio"] = HEATMAP_RATIO
        row["point_gain_vs_prev"] = (prev_point - row["point_err_hand"]) / prev_point
        append_result(row)
        rows.append(row)
        write_summary(rows, "in progress")

        pt = row["point_err_hand"]
        gain = (prev_point - pt) / prev_point
        tlog(f"round {round_i}: point={pt:.6f} prev={prev_point:.6f} gain={gain:.4f}")
        if pt >= prev_point:
            extra_note = (
                f"STOP: not better. point/hand {pt:.6f} vs bar {prev_point:.6f} "
                f"(heatmap {HEATMAP_POINT:.6f}). ratio_mae={row['ratio_mae']:.6f}."
            )
            tlog(extra_note)
            write_summary(rows, extra_note)
            return
        if gain < 0.02:
            extra_note = (
                f"STOP: better but gain {gain*100:.2f}% < 2%. "
                f"point/hand {pt:.6f} (heatmap {HEATMAP_POINT:.6f}). "
                f"ratio_mae {row['ratio_mae']:.6f} (heatmap {HEATMAP_RATIO:.6f})."
            )
            tlog(extra_note)
            write_summary(rows, extra_note)
            return
        extra_note = f"round {round_i} gained {gain*100:.2f}% point/hand; generating more."
        tlog(extra_note)
        prev_point = pt
        if round_i >= 6:
            extra_note = "STOP: 6 rounds cap."
            write_summary(rows, extra_note)
            return


# Names used by split_run.py (same objects).
IMG_DIR = IMG_DIR
PREV_DIR = PREV_DIR
LABELS = LABELS
MANIFEST = MANIFEST
GEN_LOG = GEN_LOG
TRAIN_LOG = TRAIN_LOG
HEATMAP_POINT = HEATMAP_POINT
HEATMAP_RATIO = HEATMAP_RATIO
log_to = log_to
imread = imread
imwrite_u = imwrite_u
load_samples = load_samples
add_dataset = add_dataset
fixed_split = fixed_split
warp_real = warp_real
train_heatmap = train_heatmap
pick_device = pick_device
acquire_gpu_lock = acquire_gpu_lock
release_gpu_lock = release_gpu_lock
load_synth_samples = load_synth_samples
append_result = append_result
write_summary = write_summary
render_hand_3d = render_hand_3d
draw_preview = draw_preview
LOGS = LOGS
MODELS = MODELS


if __name__ == "__main__":
    main()
