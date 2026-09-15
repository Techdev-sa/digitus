"""Two-site capture bench for the crease model.

  /        user-facing: live camera, red record button, 5s clip. The clip is
           split into many frames of one hand, each scored by the model. Moving
           the hand during the clip is the point -- one recording yields dozens
           of distinct viewpoints, which is how we harvest real-world training
           data fast.
  /admin   review + hand-correct the four points on any frame, and export the
           corrected set as annotations.

Frames are stored per session under CAPTURES/sessions/<id>/ so a recording stays
one unit: raw frames, model predictions, and any human corrections travel
together and can be exported or discarded as a group.

Storage sits outside the repo and is hard-capped -- Margin shares one volume
with pgvector and SeaweedFS, and an upload endpoint must not be able to fill it.

Run:
  python3 app/capture_site.py --ckpt models/synth_heatmap_robust768.pt \
      --input-size 768 --port 5055
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import hashlib
import re
import shutil
import subprocess
import threading
from urllib.parse import quote
import hmac
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, request, send_file, send_from_directory, make_response, redirect

REPO = Path(__file__).resolve().parents[1]
SYNTH = REPO / "data" / "synth_crease"
CAPTURES = Path("/root/hand_captures")
SESSIONS = CAPTURES / "sessions"
# Study captures are kept apart from the development ones on purpose. Once
# children are being photographed under an ethics approval, that data must not
# be interleaved with adult test recordings in the same directory -- separate
# retention rules, separate access, and no chance of a stray test session
# reaching an analysis.
STUDY_SESSIONS = CAPTURES / "study" / "sessions"


def _parse_extra(form) -> tuple[dict, str | None]:
    """Pull the schema-defined extras off a multipart form.

    Returns (values, error). An out-of-range number is an error rather than a
    silent drop: a research record that quietly loses a field is worse than one
    that refuses to be written.
    """
    out: dict = {}
    for key, pat in SUBJECT_TEXT.items():
        v = (form.get(key) or "").strip()
        if not v:
            continue
        if not re.fullmatch(pat, v, re.S):
            return {}, f"{key}: not in the accepted format"
        out[key] = v
    for key, (lo, hi) in list(SUBJECT_NUM.items()) + list(GEOMETRY_NUM.items()):
        raw = (form.get(key) or "").strip()
        if not raw:
            continue
        try:
            f = float(raw)
        except ValueError:
            return {}, f"{key}: must be a number"
        if not (lo <= f <= hi):
            return {}, f"{key}: must be between {lo} and {hi}"
        out[key] = f
    return out, None


def _parse_json_blob(form, name: str, max_bytes: int = 64_000):
    """Device and AR payloads are stored verbatim.

    Deliberately not schema-checked: the app will learn to record things this
    server has never heard of, and a strict schema would silently discard
    exactly the novel measurement that turns out to explain something.
    """
    raw = form.get(name)
    if not raw or len(raw) > max_bytes:
        return None
    try:
        d = json.loads(raw)
    except Exception:
        return None
    return d if isinstance(d, (dict, list)) else None


def _store_dir(store: str):
    return STUDY_SESSIONS if store == "study" else SESSIONS
MAX_MB = 6000              # retained for the CLI flag; no longer prunes
DISK_MIN_GB = 50           # the ONLY reason an upload is ever refused
DISK_WARN_GB = 120         # warn well before that, so it never arrives unseen
BACKUP = Path("/root/hand_backup")
sys.path.insert(0, str(REPO / "scripts"))
NAMES = ["index_base", "index_tip", "ring_base", "ring_tip"]
COLS = [(60, 90, 240), (60, 200, 250), (70, 200, 70), (240, 140, 60)]


def load_mod(n, p):
    spec = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[n] = m
    spec.loader.exec_module(m)
    return m


ro = load_mod("run_overnight", SYNTH / "run_overnight.py")
app = Flask(__name__, static_folder=None)
MODEL = None
DEVICE = None
CKPT_INFO: dict = {}


def set_size(size: int):
    import crease_net as cn
    cn.INPUT_SIZE = size
    ro.INPUT_SIZE = size
    cn.letterbox.__defaults__ = (size,)   # bound at def-time; must patch too


def load_model(ckpt: Path, input_size: int | None):
    global MODEL, DEVICE, CKPT_INFO
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    meta = torch.load(ckpt, map_location="cpu", weights_only=False)
    size = int(input_size or meta.get("input_size") or 384)
    set_size(size)
    MODEL = ro.CreaseHeatmapNet(pretrained=False).to(DEVICE)
    MODEL.load_state_dict(meta["state_dict"])
    MODEL.eval()
    CKPT_INFO = {"checkpoint": ckpt.name, "input_size": size, "device": str(DEVICE),
                 "tta": TTA, "val_ratio_mae": meta.get("ratio_mae")}
    print(f"loaded {ckpt.name} @ {size}px on {DEVICE}", flush=True)


TTA = True


def predict_once(bgr: np.ndarray) -> np.ndarray:
    lb, sc, dx, dy = ro.letterbox(bgr)
    with torch.no_grad():
        p = MODEL(ro.to_tensor(lb)[None].to(DEVICE))[0].cpu().numpy().reshape(4, 2)
    return (p * ro.INPUT_SIZE - np.array([dx, dy], np.float32)) / sc


def _heat(x):
    h = MODEL.head(MODEL.features(x))
    if MODEL.upsample > 1:
        h = torch.nn.functional.interpolate(h, scale_factor=MODEL.upsample,
                                            mode="bilinear", align_corners=False)
    return h


def _decode(h):
    b, k, H, W = h.shape
    p = torch.softmax(h.reshape(b, k, -1), -1).reshape(b, k, H, W)
    gx = ((torch.arange(W, device=h.device) + 0.5) / W).view(1, 1, 1, W)
    gy = ((torch.arange(H, device=h.device) + 0.5) / H).view(1, 1, H, 1)
    return torch.stack([(p * gx).sum((2, 3)), (p * gy).sum((2, 3))], -1)


def _decode_spread(h):
    """Soft-argmax plus the heatmap distribution's own spatial spread.

    The integral-regression decode already builds a probability distribution
    over the image and takes its expectation; the second moment of that same
    distribution is the model's uncertainty about where the point is, and was
    simply being discarded. Returned in normalised [0,1] units of the
    letterboxed image, so the caller scales it the same way it scales the
    coordinates.

    Also returns the peak probability, which separates 'broad but unimodal'
    from 'two competing candidates' -- a diffuse heatmap and a bimodal one can
    share a spread but mean different things.
    """
    b, k, H, W = h.shape
    p = torch.softmax(h.reshape(b, k, -1), -1).reshape(b, k, H, W)
    gx = ((torch.arange(W, device=h.device) + 0.5) / W).view(1, 1, 1, W)
    gy = ((torch.arange(H, device=h.device) + 0.5) / H).view(1, 1, H, 1)
    mx = (p * gx).sum((2, 3))
    my = (p * gy).sum((2, 3))
    vx = (p * gx * gx).sum((2, 3)) - mx * mx
    vy = (p * gy * gy).sum((2, 3)) - my * my
    sd = torch.sqrt(torch.clamp(vx + vy, min=0.0))          # radial, normalised
    peak = p.amax(dim=(2, 3))
    return torch.stack([mx, my], -1), sd, peak


def predict(bgr: np.ndarray, with_conf: bool = False):
    """8-view TTA: 4 rotations x {identity, mirror}, heatmaps averaged, decode once.

    Measured on 120 held-out images (ratio MAE):
        single pass          0.01255
        point-median x4      0.01218   <- what was deployed
        point-mean   x4      0.01195
        heatmap-avg  x4      0.01195
        heatmap-avg  x8      0.01174   <- this, -6.5% vs single, -3.6% vs median

    Two things that mattered. Median was the wrong reducer: with only four
    samples it discards information that the mean keeps. And the mirror is where
    the gain is -- training uses mirror augmentation, so the model really is
    equivariant to it, whereas heatmap-averaging on its own only ties the mean
    (these heatmaps are already sharp, so there is little soft evidence to pool).

    Only 90-degree rotations and mirrors are used: both are exact pixel
    permutations. Arbitrary-angle rotation was measured 3.5x WORSE because
    resampling and border padding cost more than the alignment gains.
    """
    lb, sc, dx, dy = ro.letterbox(bgr)
    to_img = lambda q: (q * ro.INPUT_SIZE - np.array([dx, dy], np.float32)) / sc

    if not TTA:
        with torch.no_grad():
            h = _heat(ro.to_tensor(lb)[None].to(DEVICE))
            q, sd, peak = _decode_spread(h)
        pts = to_img(q[0].cpu().numpy())
        if not with_conf:
            return pts
        return pts, _conf(sd[0].cpu().numpy(), peak[0].cpu().numpy(), None, sc)

    x = ro.to_tensor(lb)[None].to(DEVICE)
    acc, views = None, []
    with torch.no_grad():
        for k in range(4):
            xr = torch.rot90(x, k, dims=(2, 3))
            for mirror in (False, True):
                xi = torch.flip(xr, dims=(3,)) if mirror else xr
                h = _heat(xi)
                if mirror:
                    h = torch.flip(h, dims=(3,))
                h = torch.rot90(h, -k, dims=(2, 3))
                acc = h if acc is None else acc + h
                if with_conf:
                    views.append(_decode(h)[0].cpu().numpy())
        q, sd, peak = _decode_spread(acc / 8.0)
        p = q[0].cpu().numpy()
    pts = to_img(p)
    if not with_conf:
        return pts
    # Disagreement BETWEEN the eight views, in the same units as the points.
    # This catches a different failure from heatmap spread: every view can be
    # individually confident and still disagree about which blob is the tip.
    v = np.stack(views)                                   # (8, 4, 2) normalised
    tta_sd = np.sqrt(((v - v.mean(0)) ** 2).sum(-1).mean(0)) * ro.INPUT_SIZE / sc
    return pts, _conf(sd[0].cpu().numpy(), peak[0].cpu().numpy(), tta_sd, sc)


def _conf(sd_norm, peak, tta_sd, sc):
    """Package the three signals in image pixels."""
    out = {
        "spread_px": (sd_norm * ro.INPUT_SIZE / sc).astype(float).tolist(),
        "peak": peak.astype(float).tolist(),
    }
    out["tta_px"] = None if tta_sd is None else tta_sd.astype(float).tolist()
    return out


# Measured over 250 training and 200 val images: the hand spans 0.255 of the
# long side (p10 0.217, p90 0.308). Phone captures sit at 0.192, i.e. 1.33x
# small, and the model is letterboxed to a fixed 1024 so that FRACTION -- not
# the pixel count -- is what sets the hand's size in model space. Cropping aims
# at the training value rather than as tight as possible: a hand filling 90% of
# the input is as unfamiliar as one filling 19%.
CROP_TARGET_FRAC = 0.255
CROP_MIN_GAIN = 1.08       # not worth a second pass below this


REMOTE_PREDICT = None      # set by --remote-predict, e.g. http://<inference-host>:5056


def predict_remote(bgr: np.ndarray, timeout: float = 25.0):
    """Ask another machine for the keypoints.

    The GTX 1660 box runs the same checkpoint and returns bit-identical points,
    at ~1.7x the 5060 Ti's latency -- which does not matter for a single image
    in the review editor, and does mean a failure here never has to surface as
    'no prediction'. Returns None on any problem so the caller can degrade.
    """
    if not REMOTE_PREDICT:
        return None
    try:
        import urllib.request
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            return None
        req = urllib.request.Request(
            REMOTE_PREDICT.rstrip("/") + "/predict", data=buf.tobytes(),
            headers={"Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        pts = d.get("points")
        if not pts or len(pts) != 4:
            return None
        return np.array(pts, np.float32)
    except Exception as e:
        print(f"remote predict failed: {type(e).__name__}: {e}", flush=True)
        return None


def predict_cropped(bgr: np.ndarray):
    """Locate the hand, re-frame it to the training scale, predict again.

    Returns (points_in_original_coords, box_or_None). The first pass is kept as
    the answer whenever the second cannot be trusted -- an implausible first
    pass (which is what a cluttered background produces) would otherwise crop
    to the wrong place and the second pass would confidently refine noise.
    """
    p1 = predict(bgr)
    H, W = bgr.shape[:2]
    hs = (float(np.linalg.norm(p1[1] - p1[0])) +
          float(np.linalg.norm(p1[3] - p1[2]))) / 2
    long_side = float(max(H, W))
    if hs <= 1e-6 or long_side <= 1e-6:
        return p1, None
    frac = hs / long_side
    # Sanity gate on the first pass. Real hands land near 0.19-0.30 of the long
    # side; far outside that the detection is not a hand and must not drive a crop.
    if not (0.05 < frac < 0.60):
        return p1, None
    gain = CROP_TARGET_FRAC / frac
    if gain < CROP_MIN_GAIN:
        return p1, None                      # already framed like training

    target_long = long_side / gain
    # Preserve the original aspect so letterbox padding behaves as it does on
    # a full frame; only the field of view shrinks.
    if W >= H:
        cw, ch = target_long, target_long * H / W
    else:
        ch, cw = target_long, target_long * W / H
    cx = float(np.mean(p1[:, 0])); cy = float(np.mean(p1[:, 1]))
    x0 = int(round(min(max(cx - cw / 2, 0), max(W - cw, 0))))
    y0 = int(round(min(max(cy - ch / 2, 0), max(H - ch, 0))))
    x1 = int(round(min(x0 + cw, W))); y1 = int(round(min(y0 + ch, H)))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return p1, None
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return p1, None

    p2 = predict(crop) + np.array([x0, y0], np.float32)
    # Reject a second pass that disagrees wildly -- it means the crop cut the
    # hand or landed off it, and the first pass saw more of the picture.
    hs2 = (float(np.linalg.norm(p2[1] - p2[0])) +
           float(np.linalg.norm(p2[3] - p2[2]))) / 2
    if hs2 <= 1e-6 or not (0.6 < hs2 / hs < 1.7):
        return p1, None
    return p2, [x0, y0, x1, y1]


def draw(bgr, pts, thick=None, ghost=None):
    """Overlay the four points. `ghost` draws a second, hollow white set --
    used on the review page to show the model's own prediction beside the saved
    label, so a reviewer can see WHICH of the two is off rather than only that
    they differ."""
    vis = bgr.copy()
    if ghost is not None and len(ghost) == 4:
        g = np.asarray(ghost, float)
        s_ = max(vis.shape[:2]) / 700.0
        r_ = max(4, int(7 * s_))
        for k in range(4):
            q = tuple(g[k].astype(int))
            cv2.circle(vis, q, r_, (0, 0, 0), max(3, int(4 * s_)), cv2.LINE_AA)
            cv2.circle(vis, q, r_, (255, 255, 255), max(1, int(2 * s_)), cv2.LINE_AA)
    s = max(vis.shape[:2]) / 700.0
    t = thick or max(2, int(3 * s))
    cv2.line(vis, tuple(pts[0].astype(int)), tuple(pts[1].astype(int)), (60, 170, 250), t, cv2.LINE_AA)
    cv2.line(vis, tuple(pts[2].astype(int)), tuple(pts[3].astype(int)), (70, 200, 70), t, cv2.LINE_AA)
    for k in range(4):
        q = tuple(pts[k].astype(int))
        cv2.circle(vis, q, max(5, int(8 * s)), COLS[k], -1, cv2.LINE_AA)
        cv2.circle(vis, q, max(5, int(8 * s)), (255, 255, 255), max(1, int(1.5 * s)), cv2.LINE_AA)
    return vis


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def prune():
    """Deliberately does nothing. Kept so old call sites stay honest.

    This used to delete the oldest sessions once the store passed MAX_MB.
    Captures are not reproducible -- a school visit under IRB cannot be run
    again to recover a frame -- so nothing is ever deleted now. Space is
    protected at the door instead: `space_check()` refuses new uploads while
    the disk is low, which fails a recording rather than destroying earlier
    ones.
    """
    return


def space_check():
    """Free-space gate. Returns (ok, message).

    Disk is the one and only reason a capture is ever turned away, and only
    below DISK_MIN_GB. Refusing loses one recording that can be retaken;
    pruning loses recordings nobody can retake, and Margin runs production on
    this same volume, so the floor protects the machine rather than this app.
    The warning threshold sits far above the floor so it is never a surprise.
    """
    free = shutil.disk_usage(CAPTURES).free
    gb = free / 2**30
    if gb < DISK_MIN_GB:
        return False, (f"server low on disk ({gb:.1f} GB free, need "
                       f"{DISK_MIN_GB} GB) - nothing was deleted, free space first")
    if gb < DISK_WARN_GB:
        return True, f"disk getting low: {gb:.1f} GB free"
    return True, ""


def mirror_session(sid: str):
    """Second copy under BACKUP. Same filesystem, so this guards against
    accidental deletion and bad code -- not against disk failure. A real
    backup lives on another machine.
    """
    if BACKUP is None:
        return None
    try:
        src = SESSIONS / sid
        dst = BACKUP / "sessions" / sid
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return str(dst)
    except Exception as e:                      # never fail a capture on backup
        print(f"backup failed for {sid}: {e}", flush=True)
        return None


# Frames are letterboxed to the model's 1024 input, so extra pixels do not make
# the hand bigger to the model -- that depends on the fraction of frame it fills.
# They matter for cropping to the hand, where real sensor detail is the
# difference between resolving a crease and interpolating one. 2048 keeps files
# to roughly 600 KB at q92, so a 32-frame session is ~19 MB.
FRAME_MAX_PX = 2048

BLUR_FLOOR = 60.0          # absolute Laplacian-variance floor
BLUR_REL = 0.42            # ...and at least this fraction of the clip's best


BG_NORM = 640              # normalise before measuring: Laplacian magnitude
                           # scales with resolution, so a 1600px training image
                           # and a 1280px phone frame are not comparable raw


def background_check(bgr):
    """How close is this background to what the model was trained on?

    Measured over a 10% border ring after normalising the long side to BG_NORM
    (the hand is centred, so the ring is nearly all background). Calibrated on
    150 random 11k_hands images -- studio shots on near-white paper:

        stat          train median   train p95   plain wall   concrete
        brightness        248.9         252.8      110-138       147
        saturation          8.1          44.5        30-50        64
        texture             0.34          1.43     1.64-1.93      6.93

    Texture is the discriminator and it predicted the failures: the concrete
    shot at 6.93 was the one where every keypoint collapsed into the palm,
    while the ~1.7 plain-wall shots worked. Brightness matters far less -- the
    plain-wall shots are half as bright as training and were fine.
    """
    h, w = bgr.shape[:2]
    sc = BG_NORM / max(h, w)
    if sc < 1:
        bgr = cv2.resize(bgr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    t = max(4, int(0.10 * min(h, w)))
    m = np.zeros((h, w), bool)
    m[:t, :] = m[-t:, :] = True
    m[:, :t] = m[:, -t:] = True
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_32F)
    V = float(hsv[..., 2][m].mean())
    S = float(hsv[..., 1][m].mean())
    tex = float(np.abs(lap)[m].mean())
    if tex <= 1.45 and V >= 200 and S <= 45:
        verdict, note = "ideal", "Plain light background - matches training data."
    elif tex <= 2.5:
        verdict, note = "ok", "Plain background - usable, even if dark or tinted."
    elif tex <= 5.0:
        verdict, note = "busy", "Busy background - accuracy will suffer. Try a plain wall or sheet of paper."
    else:
        verdict, note = "bad", "Very cluttered background - use a plain surface."
    return {"brightness": round(V, 1), "saturation": round(S, 1),
            "texture": round(tex, 2), "verdict": verdict, "note": note}


KEEP_ALL_FRAMES = True     # store every frame; blur is recorded, never deleted


def blur_threshold(sharps):
    """Absolute floor combined with one relative to this clip's own best.

    A fixed threshold does not travel: Laplacian variance scales with contrast
    and lighting, so a sharp frame in dim light can score below a blurry one in
    bright light. Comparing within a recording is fair -- same hand, same
    camera, same scene, seconds apart.
    """
    a = np.asarray(sharps, float)
    if a.size == 0:
        return BLUR_FLOOR
    return max(BLUR_FLOOR, BLUR_REL * float(np.percentile(a, 90)))


def mark_blurry(cand):
    """Flag soft frames without discarding them.

    This used to delete them, with keep_min=8 as a floor for an entirely soft
    clip -- which is why real captures kept landing on exactly 8 frames: almost
    nothing cleared the threshold and the floor was doing all the work. Frames
    are expensive to collect (a school visit is not repeatable) and cheap to
    store, so they are all kept and labelled; anything downstream can filter on
    `blurry`, and the admin button moves them to set_aside/ on request.
    """
    if not cand:
        return [], 0.0
    thr = blur_threshold([c[1] for c in cand])
    return [(im, sh, sh < thr) for im, sh in cand], thr


def drop_blurry(cand, floor=BLUR_FLOOR, rel=BLUR_REL, keep_min=8):
    """Filter a session's frames down to the in-focus ones.

    A fixed threshold does not travel: Laplacian variance scales with contrast
    and lighting, so a sharp frame in dim light can score below a blurry one in
    bright light. We combine an absolute floor with a relative one -- a frame
    must be sharp in its own right AND not far below the best frame in the same
    recording, which is a fair comparison because it is the same hand, camera
    and scene seconds apart.

    keep_min guards the pathological case of an entirely soft clip: rather than
    return nothing, keep the sharpest few so the user sees a result and the
    warning, instead of an opaque failure.
    """
    if not cand:
        return []
    sharps = np.array([c[1] for c in cand], float)
    thr = max(floor, rel * float(np.percentile(sharps, 90)))
    keep = [c for c in cand if c[1] >= thr]
    if len(keep) < min(keep_min, len(cand)):
        order = sorted(cand, key=lambda c: -c[1])
        keep = order[:min(keep_min, len(cand))]
    return keep


def extract_frames(video_path: Path, max_frames: int = 90):
    """Decode the clip to frames, keeping sharp and visually distinct ones.

    A 5s clip is ~150 frames but consecutive ones are near-identical, so we
    score each on Laplacian variance (blur) and drop any frame too similar to
    the last kept one -- what we want from a recording is many *different*
    viewpoints, not 150 copies of the same pose.
    """
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():                     # browser WebM cv2 sometimes rejects
        conv = video_path.with_suffix(".conv.mp4")
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path),
                            "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(conv)],
                           check=True, timeout=120)
            cap = cv2.VideoCapture(str(conv))
        except Exception as e:
            print(f"ffmpeg fallback failed: {e}", flush=True)
            return []
    pool = []                                   # every in-focus frame
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        h, w = fr.shape[:2]
        if max(h, w) > 1280:                    # keep files sane; plenty for 768px input
            s = 1280 / max(h, w)
            fr = cv2.resize(fr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharp < 25:                          # motion blur
            continue
        pool.append((fr, sharp, cv2.resize(gray, (160, 120))))
    cap.release()
    if not pool:
        return []

    # Greedy dedup: keep a frame only if it differs enough from the last keeper.
    # The threshold has to adapt -- a hand waved across the frame produces huge
    # inter-frame differences, a hand held fairly still produces tiny ones, and a
    # fixed threshold either keeps everything or (as first written, at 6.0)
    # collapses a 150-frame clip to a single image.
    diffs = []
    for i in range(1, len(pool)):
        diffs.append(float(np.abs(pool[i][2].astype(np.int16)
                                  - pool[i - 1][2].astype(np.int16)).mean()))
    thr = float(np.percentile(diffs, 55)) if diffs else 0.0
    kept, prev = [], None
    for fr, sharp, small in pool:
        if prev is not None:
            d = float(np.abs(small.astype(np.int16) - prev.astype(np.int16)).mean())
            if d < thr:
                continue
        prev = small
        kept.append((fr, sharp))
        if len(kept) >= max_frames:
            break

    # Guarantee coverage: if dedup was still too aggressive (a mostly-static
    # clip), fall back to uniform temporal sampling of the sharp pool.
    if len(kept) < min(24, len(pool)):
        idx = np.linspace(0, len(pool) - 1, min(max_frames, len(pool))).astype(int)
        kept = [(pool[i][0], pool[i][1]) for i in sorted(set(idx.tolist()))]
    return kept


# --------------------------------------------------------------------- pages
USER_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#126c74"><link rel="stylesheet" href="/static/console.css"><script src="/static/console.js"></script><title>Capture | Hand Study</title></head><body><div id="app"></div><script>
const H = HandConsole;
function field(name, type, attrs = '') { return `<label><span class="label">${H.esc(name.replaceAll('_',' '))}</span><input class="field" name="${H.esc(name)}" type="${type}" ${attrs}></label>`; }
function renderForm(schema) {
  const tagFields = Object.entries(schema.tags || {}).map(([name, values]) => `<label><span class="label">${H.esc(name.replaceAll('_',' '))}</span><select class="field" name="${H.esc(name)}"><option value="">Select…</option>${values.map(value => `<option value="${H.esc(value)}">${H.esc(value)}</option>`).join('')}</select></label>`).join('');
  const numberFields = Object.entries(schema.subject_numeric || {}).map(([name, rules]) => field(name, 'number', `min="${rules.min}" max="${rules.max}" step="any"`)).join('');
  const textFields = Object.keys(schema.subject_text || {}).map(name => name === 'notes' ? `<label><span class="label">notes</span><textarea class="field" name="notes" maxlength="240"></textarea></label>` : field(name, 'text')).join('');
  H.byId('schema-form').innerHTML = `<div class="form-grid">${tagFields}${numberFields}${textFields}</div><p class="card-note" style="margin:12px 0 0">Form fields and accepted values are supplied by the server schema. Values outside numeric limits are rejected rather than altered.</p>`;
}
function page() {
  H.byId('app').innerHTML = H.shell({active:'/capture', eyebrow:'Desktop capture', title:'Capture a palm measurement', subtitle:'Record a short multi-frame sequence; review the quality verdict before relying on a reading.', actions:'<a class="btn" href="/method">Measurement method</a>', content:`
    <div class="camera-layout"><section><div class="camera-stage"><video id="camera" autoplay muted playsinline aria-label="Live camera preview"></video><div class="camera-overlay"><span class="camera-guide" id="camera-status">Preparing camera preview…</span><span class="camera-guide">Use a plain background. The final verdict is calculated by the server.</span></div><div class="camera-controls"><button id="record" class="record" type="button" aria-label="Record measurement frames">REC</button></div></div><div id="result" class="result-panel" hidden></div></section>
    <aside class="card card-pad"><h2>Capture record</h2><p class="card-note">These fields describe the capture. They do not produce a diagnostic or screening result.</p><form id="schema-form" class="stack" style="margin-top:16px">${H.loading(5)}</form><div id="capture-error" class="instrument-note error-note" hidden style="margin-top:16px"></div></aside></div>
  `}); H.bindPreferences();
}
async function bootCamera() {
  const video = H.byId('camera'), status = H.byId('camera-status');
  try { video.srcObject = await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}},audio:false}); status.textContent='Camera ready. Record a short sequence.'; }
  catch (error) { status.textContent='Camera unavailable.'; H.byId('capture-error').hidden=false; H.byId('capture-error').textContent='Camera permission is required for browser capture. Check browser permissions and try again.'; H.byId('record').disabled=true; }
}
function showResult(data) {
  const target = H.byId('result'); target.hidden=false;
  const thumbs = Array.isArray(data.thumbs) && data.thumbs.length ? `<div class="thumb-grid">${data.thumbs.map(url=>`<img alt="Landmark overlay" src="${H.esc(url)}">`).join('')}</div>` : H.state('empty','No overlays returned','This response did not include overlay thumbnails.');
  target.innerHTML = `${H.reading(data,{label:'Measurement result'})}<section class="card card-pad" style="margin-top:16px"><div class="inline justify"><div><h2>Capture evidence</h2><p class="card-note">The result includes only server-reported evidence.</p></div><button class="btn" type="button" id="recapture">Record again</button></div><div class="grid grid-3" style="margin-top:14px"><div><span class="label">Hand found</span><strong class="num">${H.n(data.n_valid)} of ${H.n(data.n_frames)} frames</strong></div><div><span class="label">Blur threshold</span><strong class="num">${H.n(data.blur_threshold,1)}</strong></div><div><span class="label">Server processing</span><strong class="num">${H.n(data.seconds,2)} seconds</strong></div></div>${thumbs}</section>`;
  H.byId('recapture').onclick=()=>{target.hidden=true; H.byId('record').focus();}; target.scrollIntoView({behavior:'smooth',block:'start'});
}
async function record() {
  const video=H.byId('camera'), button=H.byId('record'), error=H.byId('capture-error');
  error.hidden=true; if (!video.videoWidth) { error.hidden=false; error.textContent='The camera is not ready yet.'; return; }
  button.disabled=true; button.classList.add('recording'); button.textContent='…';
  try {
    const canvas=document.createElement('canvas'); canvas.width=video.videoWidth; canvas.height=video.videoHeight; const ctx=canvas.getContext('2d'); const form=new FormData(H.byId('schema-form'));
    for(let i=0;i<8;i+=1) { ctx.drawImage(video,0,0); const blob=await new Promise(resolve=>canvas.toBlob(resolve,'image/jpeg',.92)); if(blob) form.append('frames',blob,`f${String(i).padStart(4,'0')}.jpg`); await new Promise(resolve=>setTimeout(resolve,160)); }
    form.append('capture_res',`${video.videoWidth}x${video.videoHeight}`); const response=await fetch('/api/session',{method:'POST',body:form}); const data=await response.json().catch(()=>null); if(!response.ok||data?.error) throw new Error(data?.error||'The measurement request failed.'); showResult(data);
  } catch (err) { error.hidden=false; error.textContent=err.message; }
  finally { button.disabled=false; button.classList.remove('recording'); button.textContent='REC'; }
}
page(); H.byId('record').onclick=record; H.api('/api/schema').then(renderForm).catch(error=>{H.byId('schema-form').innerHTML=H.state('error','Could not load the capture schema',error.message);H.byId('schema-form').querySelector('[data-retry]')?.addEventListener('click',()=>location.reload());}); bootCamera();
</script></body></html>"""

ADMIN_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#006b63"><link rel="stylesheet" href="/static/console.css"><script src="/static/console.js"></script><title>Administration | Hand Study</title></head><body><div id="app"></div><script>
const H=HandConsole;
function result(row){return {...row,quality:{level:row?.quality||'unknown',reasons:Array.isArray(row?.quality_reasons)?row.quality_reasons:[]}};}
function measurementCell(row){const fail=row.quality==='fail'||!H.numeric(row.median_ratio);const value=fail?'Not reported':H.ratio(row.median_ratio);return `<div class="measurement-cell"><strong class="mono">${value}</strong><span class="mono muted">SD ${H.n(row.sd_ratio,4)}</span>${H.quality(result(row).quality)}</div>`;}
function page(){H.byId('app').innerHTML=H.shell({active:'/admin',eyebrow:'Study operations',title:'Administration',subtitle:'Session inventory, capacity and retention controls. Measurements retain their quality evidence.',actions:'<a class="btn" href="/review">Open review</a>',content:`
<section class="grid grid-3"><article class="card metric"><span class="metric-label">Sessions returned</span><strong id="count" class="metric-value">—</strong><span class="metric-sub">Matches the active study-session filters.</span></article><article class="card metric"><span class="metric-label">Storage capacity</span><strong id="storage-value" class="metric-value">—</strong><span id="storage-sub" class="metric-sub">Loading current capacity.</span></article><article class="card metric"><span class="metric-label">Upload guard</span><strong id="guard-value" class="metric-value">—</strong><span id="guard-sub" class="metric-sub">Loading threshold status.</span></article></section>
<section class="card card-pad" style="margin-top:16px"><div class="inline justify"><div><h2>Session inventory</h2><p class="card-note">Each capture retains its date, subject, measurement uncertainty and quality verdict.</p></div><div class="inline"><button class="btn" id="export" type="button">Export corrected frames</button><button class="btn btn-danger" id="cleanup" type="button">Set aside soft frames</button></div></div><form id="filters" class="form-grid" style="margin-top:16px" aria-label="Study-session filters"><label><span class="label">Store</span><select class="field" id="store"><option value="">All stores</option><option value="study">Study</option><option value="development">Development</option></select></label><label><span class="label">Subject</span><input class="field" id="subject" placeholder="e.g. 014"></label><label><span class="label">Quality</span><select class="field" id="quality"><option value="">All quality levels</option><option value="ok">Acceptable</option><option value="poor">Review advised</option><option value="fail">Not reliable</option><option value="unknown">Unavailable</option></select></label><div class="form-actions" style="grid-column:1 / -1"><button class="btn" type="button" id="clear-filters">Clear filters</button><button class="btn btn-primary" type="submit">Apply filters</button></div></form><div id="inventory" style="margin-top:16px">${H.loading(5,'Loading study sessions')}</div></section><div id="admin-status" class="instrument-note" hidden style="margin-top:16px"></div>`});H.bindPreferences();}
let sessions=[];
function render(){H.byId('count').textContent=H.n(sessions.length);const rows=sessions.map(row=>`<tr><td class="mono">${H.esc(row.id)}</td><td><a href="/subject?subject=${encodeURIComponent(row.subject)}" class="mono">${H.esc(row.subject)}</a></td><td>${H.esc(H.iso(row.utc))}</td><td>${H.esc(row.hand||'—')}</td><td>${measurementCell(row)}</td><td class="num">${H.n(row.n_frames)}</td><td>${H.esc(row.store||'—')}</td><td><a href="/review?session=${encodeURIComponent(row.id)}" class="btn">Review frames</a></td></tr>`).join('');H.byId('inventory').innerHTML=rows?`<div class="table-wrap"><table><thead><tr><th>Session</th><th>Subject</th><th>Captured</th><th>Hand</th><th>Measurement</th><th>Frames</th><th>Store</th><th>Actions</th></tr></thead><tbody>${rows}</tbody></table></div>`:H.state('empty','No sessions match these filters','Change a documented store, subject or quality filter, then try again.');}
function notice(message,error=false){const el=H.byId('admin-status');el.hidden=false;el.className=`instrument-note ${error?'error-note':''}`;el.textContent=message;}
function storageState(data){const free=Number(data.free_gb),warn=Number(data.warn_below_gb),refuse=Number(data.refuse_below_gb);const blocked=Number.isFinite(free)&&Number.isFinite(refuse)&&free<refuse;const warning=!data.ok||Number.isFinite(free)&&Number.isFinite(warn)&&free<warn;H.byId('storage-value').textContent=Number.isFinite(free)?`${H.n(free,1)} GB`:'—';H.byId('storage-sub').textContent=Number.isFinite(data.used_gb)&&Number.isFinite(data.total_gb)?`${H.n(data.used_gb,1)} GB used of ${H.n(data.total_gb,1)} GB total.`:'Capacity totals unavailable.';H.byId('guard-value').textContent=blocked?'Uploads refused':warning?'Storage warning':'Capacity OK';H.byId('guard-sub').textContent=blocked?`Below the ${H.n(refuse,0)} GB refusal threshold.`:warning?`Below the ${H.n(warn,0)} GB warning threshold; uploads stop below ${H.n(refuse,0)} GB.`:`Warning below ${H.n(warn,0)} GB; uploads stop below ${H.n(refuse,0)} GB.`;}
async function loadInventory(){const params=new URLSearchParams();for(const key of ['store','subject','quality']){const value=H.byId(key).value.trim();if(value)params.set(key,value);}params.set('limit','100');H.byId('inventory').innerHTML=H.loading(5,'Loading study sessions');try{const data=await H.api(`/api/study/sessions?${params}`);sessions=Array.isArray(data)?data:[];render();}catch(error){sessions=[];H.byId('count').textContent='—';H.byId('inventory').innerHTML=H.state('error','Could not load study sessions',error.message);}}
async function cleanup(){const current=sessions[0];if(!current){notice('No returned session is available to clean up.',true);return;}const preview=await H.api(`/api/admin/cleanup?session=${encodeURIComponent(current.id)}&dry=1`,{method:'POST'});if(!preview.removed){notice('No eligible soft frames are available to set aside.');return;}H.confirmDialog({title:'Set aside soft frames?',detail:`${preview.removed} eligible frame(s) in ${current.id} are below the reported threshold of ${H.n(preview.threshold,1)}.`,consequence:'The current API moves eligible files to set_aside; it reports deleted: 0. Edited frames are retained.',confirmLabel:'Set aside frames',onConfirm:async()=>{const done=await H.api(`/api/admin/cleanup?session=${encodeURIComponent(current.id)}`,{method:'POST'});notice(`${done.removed} frames set aside; ${done.kept} kept. The response reports ${done.deleted} deleted.`);}});}
async function exportFrames(){const current=sessions[0];if(!current){notice('No returned session is available to export.',true);return;}try{const data=await H.api(`/api/admin/export?session=${encodeURIComponent(current.id)}`,{method:'POST'});notice(`Exported ${H.n(data.n)} corrected frames to ${data.path}.`);}catch(error){notice(error.message,true);}}
page();H.byId('filters').onsubmit=event=>{event.preventDefault();loadInventory();};H.byId('clear-filters').onclick=()=>{H.byId('filters').reset();loadInventory();};H.byId('cleanup').onclick=()=>cleanup().catch(error=>notice(error.message,true));H.byId('export').onclick=exportFrames;loadInventory();H.api('/api/study/storage').then(storageState).catch(error=>{H.byId('storage-value').textContent='Unavailable';H.byId('storage-sub').textContent='Current free capacity could not be loaded.';H.byId('guard-value').textContent='Check storage';H.byId('guard-sub').textContent=error.message;});
</script></body></html>"""


REVIEW_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#126c74"><link rel="stylesheet" href="/static/console.css"><script src="/static/console.js"></script><title>Review | Hand Study</title></head><body><div id="app"></div><div id="editor" hidden></div><script>
const H=HandConsole; let items=[],selected=null,editor=null;
function page(){H.byId('app').innerHTML=H.shell({active:'/review',eyebrow:'Scientific review',title:'Landmark review',subtitle:'Model predictions remain visible while a reviewer checks or corrects the saved points.',actions:'<a class="btn" href="/admin">Administration</a>',content:`
<div class="instrument-note required-note">The documented review list is frame-level and does not return session ratio, SD, quality, subject, or session date. Those required session-list and live-recomputation fields need a new session-review endpoint; they are not inferred here.</div><div class="review-layout" style="margin-top:16px"><aside class="card card-pad session-list"><h2>Available review frames</h2><p class="card-note">Loading, empty and error states are explicit below.</p><div id="frames" class="stack" style="margin-top:12px">${H.loading(4)}</div></aside><section class="card card-pad"><div class="inline justify"><div><h2 id="frame-title">Select a frame</h2><p id="frame-meta" class="card-note">The frame record supplies sharpness and blur status.</p></div><span id="edited-status"></span></div><div id="frame-detail" style="margin-top:16px">${H.state('empty','No frame selected','Choose a frame from the available review list.')}</div></section><aside class="card card-pad"><h2>Review protocol</h2><div class="stack" style="margin-top:14px"><div><span class="label">Model output</span><p class="card-note">Hollow white markers in the editor retain the model points.</p></div><div><span class="label">Current saved points</span><p class="card-note">Coloured markers are the points that will be saved after review.</p></div><div><span class="label">Frame exclusion</span><p class="card-note">Excluding a frame is reversible. The API does not delete source data.</p></div></div></aside></div>`});H.bindPreferences();}
function renderItems(){const root=H.byId('frames');root.innerHTML=items.length?items.map(item=>`<button class="session-item ${selected?.id===item.id?'active':''}" data-id="${H.esc(item.id)}"><strong class="mono">${H.esc(item.id)}</strong><small>Frame ratio ${H.ratio(item.ratio)} · session uncertainty unavailable</small><span class="inline">${item.edited?'<span class="quality ok">Edited</span>':''}${item.blurry?'<span class="quality poor">Soft frame</span>':''}</span></button>`).join(''):H.state('empty','No review frames are available','No frames were returned by the review list.');root.querySelectorAll('button[data-id]').forEach(button=>button.onclick=()=>select(button.dataset.id));}
async function select(id){selected=items.find(item=>item.id===id);renderItems();H.byId('frame-title').textContent=selected.id;H.byId('frame-meta').textContent=`Frame ratio ${H.ratio(selected.ratio)}; session SD and quality are not supplied by this endpoint.`;H.byId('edited-status').innerHTML=selected.edited?'<span class="quality ok">Human-edited</span>':'<span class="quality unknown">Not edited</span>';H.byId('frame-detail').innerHTML=H.loading(4);try{const data=await H.api(`/api/review/item/${encodeURIComponent(id)}`);H.byId('frame-detail').innerHTML=`<div class="grid grid-3"><div><span class="label">Sharpness</span><strong class="num">${H.n(selected.sharpness,1)}</strong></div><div><span class="label">Blur flag</span><strong>${selected.blurry?'Flagged soft':'Not flagged'}</strong></div><div><span class="label">Point source</span><strong>${data.pred?'Model + saved points':'Saved points only'}</strong></div></div><div class="inline" style="margin-top:16px"><button class="btn btn-primary" id="open-editor">Open landmark editor</button><button class="btn" id="exclude" type="button">${selected.excluded?'Restore frame':'Set aside frame'}</button></div>`;H.byId('open-editor').onclick=()=>openEditor(data);H.byId('exclude').onclick=()=>excludeFrame();}catch(error){H.byId('frame-detail').innerHTML=H.state('error','Could not load this review frame',error.message);}}
function openEditor(data){const host=H.byId('editor');host.hidden=false;host.className='editor-modal';host.innerHTML=`<header class="editor-bar"><button class="btn" id="close-editor">Close</button><strong class="mono">${H.esc(data.id)}</strong><span class="editor-legend">Hollow white: model prediction. Coloured: current saved point.</span><button class="btn" id="reset-points">Reset to model</button><button class="btn btn-primary" id="save-points">Save corrected points</button></header><div class="editor-stage"><canvas id="review-canvas" aria-label="Drag coloured landmarks to correct the saved points"></canvas></div>`;editor=H.canvasReview(H.byId('review-canvas'),`/api/review/image/${encodeURIComponent(data.id)}`,{points:data.points,model_points:data.pred||[]});H.byId('close-editor').onclick=()=>{host.hidden=true;};H.byId('reset-points').onclick=()=>editor.reset();H.byId('save-points').onclick=async()=>{try{await H.api('/api/review/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:data.id,points:editor.points})});selected.edited=true;host.hidden=true;renderItems();H.byId('edited-status').innerHTML='<span class="quality ok">Human-edited</span>';}catch(error){alert(`Could not save: ${error.message}`);}};}
function excludeFrame(){if(!selected)return;H.confirmDialog({title:selected.excluded?'Restore this frame?':'Set aside this frame?',detail:selected.excluded?'This removes the set-aside flag.':'This marks the selected frame as excluded from the measurement workflow.',consequence:'The documented endpoint is reversible and retains the frame, its image and landmark data.',confirmLabel:selected.excluded?'Restore frame':'Set aside frame',onConfirm:async()=>{const data=await H.api('/api/review/exclude',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:selected.id,excluded:!selected.excluded,reason:'review decision'})});selected.excluded=data.excluded;await select(selected.id);}});}
page();H.api('/api/review/list?src=captures&sort=seq&page=0&per=60').then(data=>{items=data.items||[];renderItems();if(items[0])select(items[0].id);}).catch(error=>{H.byId('frames').innerHTML=H.state('error','Could not load the review list',error.message);});
</script></body></html>"""


DASHBOARD_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#006b63"><link rel="stylesheet" href="/static/console.css"><script src="/static/console.js"></script><title>Study dashboard | Hand Study</title></head><body><div id="app"></div><script>
const H=HandConsole;
function refBand(bands,key,fallback){const raw=bands?.[key];if(Array.isArray(raw)&&raw.length>1)return {mean:Number(raw[0]),sd:Number(raw[1])};if(raw&&Number.isFinite(Number(raw.mean))&&Number.isFinite(Number(raw.sd)))return {mean:Number(raw.mean),sd:Number(raw.sd)};return fallback;}
function histogram(summary){const edges=summary?.ratio_histogram?.edges||[],counts=summary?.ratio_histogram?.counts||[];if(edges.length!==counts.length+1||!counts.length)return H.state('empty','No ratio distribution returned','The aggregate response did not include usable histogram bins.');const W=680,Ht=252,L=42,R=12,T=18,B=34,w=W-L-R,h=Ht-T-B,max=Math.max(...counts,1),min=Number(edges[0]),maxEdge=Number(edges.at(-1));const x=value=>L+(Number(value)-min)/(maxEdge-min)*w;const y=value=>T+h-value/max*h;const grid=[0,.25,.5,.75,1].map(part=>`<line class="chart-gridline" x1="${L}" x2="${W-R}" y1="${T+h-part*h}" y2="${T+h-part*h}"/>`).join('');const bars=counts.map((count,index)=>{const left=x(edges[index])+1.5,right=x(edges[index+1])-1.5;return `<rect class="chart-bar" x="${left.toFixed(1)}" y="${y(count).toFixed(1)}" width="${Math.max(1,right-left).toFixed(1)}" height="${(T+h-y(count)).toFixed(1)}" rx="2"><title>${edges[index].toFixed(2)}–${edges[index+1].toFixed(2)}: ${count}</title></rect>`;}).join('');const male=refBand(summary.reference_bands,'male',{mean:.964,sd:.030}),female=refBand(summary.reference_bands,'female',{mean:.975,sd:.028});const band=(pair,cls)=>`<rect class="${cls}" x="${x(pair.mean-pair.sd).toFixed(1)}" y="${T}" width="${Math.max(2,x(pair.mean+pair.sd)-x(pair.mean-pair.sd)).toFixed(1)}" height="${h}"/>`;const labels=[min,.90,.95,1.00,1.05,maxEdge].filter((v,i,a)=>v>=min&&v<=maxEdge&&a.indexOf(v)===i).map(v=>`<text class="chart-label" x="${x(v)}" y="${Ht-10}" text-anchor="middle">${Number(v).toFixed(2)}</text>`).join('');return `<div class="chart"><svg viewBox="0 0 ${W} ${Ht}" role="img" aria-label="Histogram of session median ratio; n equals ${H.n(summary.ratio?.n)}"><title>Session median-ratio distribution</title>${grid}${band(male,'chart-ref-male')}${band(female,'chart-ref-female')}<line class="chart-axis" x1="${L}" y1="${T+h}" x2="${W-R}" y2="${T+h}"/>${bars}${labels}<text class="chart-label" x="${L}" y="${T+10}">sessions</text></svg><div class="chart-legend"><span><i class="legend-key male"></i> Male reference</span><span><i class="legend-key female"></i> Female reference</span></div></div>`;}
function qualityMix(mix){const entries=[['ok','Acceptable'],['poor','Review advised'],['fail','Not reliable'],['unknown','Unavailable']];const total=entries.reduce((sum,[key])=>sum+Number(mix?.[key]||0),0);if(!total)return H.state('empty','No quality mix returned','The aggregate response contains no quality counts.');const segments=entries.filter(([key])=>Number(mix?.[key]||0)>0).map(([key,label])=>`<span class="segment ${key==='unknown'?'unknown':key}" style="flex:${Number(mix[key])}" title="${H.esc(label)}: ${H.n(mix[key])}">${H.n(mix[key])}</span>`).join('');return `<div class="stacked-bar" role="img" aria-label="Quality mix across ${total} sessions">${segments}</div><div class="chart-legend">${entries.map(([key,label])=>`<span><i class="legend-key ${key}"></i>${H.esc(label)}: ${H.n(mix?.[key])}</span>`).join('')}</div>`;}
function sdComparison(summary){const sd=summary?.sd||{},rows=[['Median session SD',Number(sd.median),'chart-bar'],['Model repeatability',Number(summary.model_repeatability),'chart-line-repeat'],['Human inter-observer',Number(summary.human_interobserver),'chart-line-human']].filter(([,value])=>Number.isFinite(value));if(!rows.length)return H.state('empty','No spread context returned','The aggregate response contains no session-spread values.');const W=680,Ht=190,L=172,R=56,T=16,B=12,w=W-L-R,max=Math.max(.08,...rows.map(([,value])=>value));const x=value=>L+value/max*w;return `<div class="chart"><svg viewBox="0 0 ${W} ${Ht}" role="img" aria-label="Session spread compared with model repeatability and human inter-observer context">${rows.map((row,index)=>{const [label,value,cls]=row,y=T+index*54+22;return `<text class="chart-label" x="${L-10}" y="${y+4}" text-anchor="end">${label}</text><line class="chart-gridline" x1="${L}" x2="${W-R}" y1="${y}" y2="${y}"/><line class="${cls}" x1="${L}" x2="${x(value)}" y1="${y}" y2="${y}"/><circle cx="${x(value)}" cy="${y}" r="4" fill="var(--teal)"><title>${label}: ${value.toFixed(4)}</title></circle><text class="chart-label" x="${W-R+6}" y="${y+4}">${value.toFixed(4)}</text>`;}).join('')}</svg></div><div class="chart-legend"><span><i class="legend-key repeat"></i>${H.n(sd.above_poor)} above poor threshold</span><span><i class="legend-key human"></i>${H.n(sd.above_fail)} above fail threshold</span></div>`;}
function groupTable(title,groups,minGroup){const rows=Object.entries(groups||[]);if(!rows.length)return H.state('empty','No subgroup summary returned','The aggregate response did not return this comparison.');return `<article class="card card-pad"><h3>${H.esc(title)}</h3><p class="card-note">Each row shows n, median, mean and SD where the server permits reporting. Groups below n = ${H.n(minGroup)} are withheld.</p><div class="table-wrap" style="margin-top:12px"><table style="min-width:0"><thead><tr><th>Group</th><th>n</th><th>Summary</th></tr></thead><tbody>${rows.map(([label,value])=>value?.withheld?`<tr><td>${H.esc(label)}</td><td class="num">${H.n(value.n)}</td><td><span class="quality unknown">Withheld</span><div class="card-note">${H.esc(value.note||`Fewer than ${H.n(minGroup)} sessions in this subgroup.`)}</div></td></tr>`:`<tr><td>${H.esc(label)}</td><td class="num">${H.n(value?.n)}</td><td class="mono">Median ${H.ratio(value?.median)} · mean ${H.ratio(value?.mean)} · SD ${H.n(value?.sd,4)}</td></tr>`).join('')}</tbody></table></div></article>`;}
function page(){H.byId('app').innerHTML=H.shell({active:'/dashboard',eyebrow:'For study review',title:'Study dashboard',subtitle:'Session-level study summary with explicit sample sizes, measurement spread and withheld small subgroups.',actions:'<a class="btn" href="/method">Read method</a>',content:`<div id="dashboard-notice" class="instrument-note">Loading study aggregates.</div><section id="metrics" class="grid grid-3" style="margin-top:16px">${H.loading(3,'Loading study summary')}</section><section class="grid grid-2" style="margin-top:16px"><article class="card"><div class="card-head"><div><h2>Distribution of session median ratio</h2><p id="ratio-note" class="card-note">Loading histogram.</p></div></div><div class="card-body" id="ratio-chart">${H.loading(4)}</div></article><article class="card"><div class="card-head"><div><h2>Quality mix</h2><p id="quality-note" class="card-note">Loading quality counts.</p></div></div><div class="card-body" id="quality-chart">${H.loading(3)}</div></article><article class="card"><div class="card-head"><div><h2>Repeatability context</h2><p id="sd-note" class="card-note">Loading session spread.</p></div></div><div class="card-body" id="sd-chart">${H.loading(3)}</div></article><article class="card"><div class="card-head"><div><h2>Captures over time</h2><p id="time-note" class="card-note">Loading dated capture counts.</p></div></div><div class="card-body" id="time-chart">${H.loading(3)}</div></article></section><section class="stack" style="margin-top:16px"><div><h2>Covariate and hand summaries</h2><p id="subgroup-note" class="card-note">Subgroups are only reported when the server does not withhold them.</p></div><div id="subgroups" class="grid grid-2">${H.loading(4)}</div><article class="card card-pad"><h3>Age-band distribution</h3><div class="empty-chart">Age is available per capture, but the aggregate response does not include an age-band summary. The console does not derive one client-side because it would bypass the server’s small-subgroup withholding rule.</div></article></section>`});H.bindPreferences();}
function render(summary){H.byId('dashboard-notice').className='instrument-note';H.byId('dashboard-notice').textContent=`${H.n(summary.n_sessions)} capture sessions; ${H.n(summary.n_measured)} measured sessions; ${H.n(summary.n_subjects)} subjects. Subgroups below n = ${H.n(summary.min_group)} are shown as withheld.`;H.byId('metrics').innerHTML=`<article class="card metric"><span class="metric-label">Total subjects</span><strong class="metric-value num">${H.n(summary.n_subjects)}</strong><span class="metric-sub">n = ${H.n(summary.n_subjects)} subjects represented in the summary.</span></article><article class="card metric"><span class="metric-label">Total captures</span><strong class="metric-value num">${H.n(summary.n_sessions)}</strong><span class="metric-sub">n = ${H.n(summary.n_sessions)} session records.</span></article><article class="card metric"><span class="metric-label">Measured sessions</span><strong class="metric-value num">${H.n(summary.n_measured)}</strong><span class="metric-sub">n = ${H.n(summary.n_measured)} with a reported measurement.</span></article>`;H.byId('ratio-note').textContent=`n = ${H.n(summary.ratio?.n)} measured sessions; reference bands are contextual, not individual verdicts.`;H.byId('ratio-chart').innerHTML=histogram(summary);H.byId('quality-note').textContent=`n = ${H.n(summary.n_sessions)} capture sessions, including unavailable-quality sessions.`;H.byId('quality-chart').innerHTML=qualityMix(summary.quality_mix);H.byId('sd-note').textContent=`n = ${H.n(summary.sd?.n)} sessions with session SD; median ${H.n(summary.sd?.median,4)}.`;H.byId('sd-chart').innerHTML=sdComparison(summary);H.byId('time-note').textContent=`n = ${H.n(summary.n_sessions)} sessions; each point is a dated capture count.`;H.byId('time-chart').innerHTML=H.lineChart((summary.captures_over_time||[]).map(row=>({label:row.date,value:row.n})),{label:'Captures over time'});H.byId('subgroup-note').textContent=`The server withholds statistics below n = ${H.n(summary.min_group)}. Withheld is not a zero and is not missing data.`;H.byId('subgroups').innerHTML=[groupTable('Nationality',summary.by_nationality,summary.min_group),groupTable('Ethnicity',summary.by_ethnicity,summary.min_group),groupTable('Sex',summary.by_sex,summary.min_group),groupTable('Autistic',summary.by_autistic,summary.min_group),groupTable('Hand',summary.by_hand,summary.min_group)].join('');}
page();H.api('/api/study/summary').then(render).catch(error=>{H.byId('dashboard-notice').className='instrument-note error-note';H.byId('dashboard-notice').textContent=`Could not load study aggregates: ${error.message}`;H.byId('metrics').innerHTML=H.state('error','Could not load the study summary',error.message);for(const id of ['ratio-chart','quality-chart','sd-chart','time-chart','subgroups'])H.byId(id).innerHTML=H.state('error','Aggregate data unavailable',error.message);});
</script></body></html>"""


SUBJECTS_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#006b63"><link rel="stylesheet" href="/static/console.css"><script src="/static/console.js"></script><title>Subjects | Hand Study</title></head><body><div id="app"></div><script>
const H=HandConsole;
function quality(row){return {level:row?.quality||'unknown',reasons:Array.isArray(row?.quality_reasons)?row.quality_reasons:[]};}
function latest(rows){return [...rows].sort((a,b)=>String(b.utc||'').localeCompare(String(a.utc||'')))[0];}
function cell(row){const fail=row.quality==='fail'||!H.numeric(row.median_ratio);return `<div class="measurement-cell"><strong class="mono">${fail?'Not reported':H.ratio(row.median_ratio)}</strong><span class="mono muted">SD ${H.n(row.sd_ratio,4)}</span>${H.quality(quality(row))}</div>`;}
function page(){H.byId('app').innerHTML=H.shell({active:'/subjects',eyebrow:'Study record',title:'Subject records',subtitle:'Every subject record preserves individual captures and their quality evidence; no cross-capture average is substituted.',actions:'<a class="btn" href="/dashboard">Study dashboard</a>',content:`<section class="card card-pad" style="max-width:800px"><h2>Find a subject</h2><p class="card-note">Open a server-issued subject ID. The record is populated from the documented subject-filtered study-session response.</p><form id="subject-search" class="inline" style="margin-top:16px"><label style="flex:1"><span class="label">Server-issued subject ID</span><input class="field" name="subject" required placeholder="e.g. 014"></label><button class="btn btn-primary" type="submit">Open record</button></form></section><section class="card card-pad" style="margin-top:16px"><div class="inline justify"><div><h2>Subjects in current response</h2><p class="card-note">Each row is built from returned capture rows; the latest capture is shown with its SD and quality verdict.</p></div></div><div id="subject-list" style="margin-top:14px">${H.loading(5,'Loading study subjects')}</div></section>`});H.bindPreferences();}
function render(rows){const grouped=new Map();for(const row of rows||[]){if(!row.subject)continue;const list=grouped.get(row.subject)||[];list.push(row);grouped.set(row.subject,list);}const subjects=[...grouped.entries()].sort(([a],[b])=>a.localeCompare(b));H.byId('subject-list').innerHTML=subjects.length?`<div class="table-wrap"><table><thead><tr><th>Subject</th><th>Captures</th><th>Latest capture</th><th>Latest measurement</th><th>Action</th></tr></thead><tbody>${subjects.map(([subject,rows])=>{const row=latest(rows);return `<tr><td class="mono">${H.esc(subject)}</td><td class="num">${H.n(rows.length)}</td><td>${H.esc(H.iso(row.utc))}</td><td>${cell(row)}</td><td><a class="btn" href="/subject?subject=${encodeURIComponent(subject)}">Open record</a></td></tr>`;}).join('')}</tbody></table></div>`:H.state('empty','No subject records returned','The study-session response did not return any subject-associated captures.');}
page();H.byId('subject-search').onsubmit=event=>{event.preventDefault();const subject=new FormData(event.currentTarget).get('subject');location.href=`/subject?subject=${encodeURIComponent(subject)}`;};H.api('/api/study/sessions').then(render).catch(error=>{H.byId('subject-list').innerHTML=H.state('error','Could not load subject records',error.message);});
</script></body></html>"""


SUBJECT_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#006b63"><link rel="stylesheet" href="/static/console.css"><script src="/static/console.js"></script><title>Subject detail | Hand Study</title></head><body><div id="app"></div><script>
const H=HandConsole;const subject=new URLSearchParams(location.search).get('subject')||'';
function result(row){return {...row,quality:{level:row?.quality||'unknown',reasons:Array.isArray(row?.quality_reasons)?row.quality_reasons:[]}};}
function values(rows,key){const items=[...new Set(rows.map(row=>row?.[key]).filter(value=>value!==undefined&&value!==null&&value!==''))];return items.length?items.map(H.esc).join(' · '):'—';}
function captureCard(row){return `<article class="card card-pad"><div class="inline justify"><div><h3>${H.esc(H.iso(row.utc))}</h3><p class="card-note">${H.esc(row.hand||'Hand unavailable')} hand · ${H.esc(row.store||'Store unavailable')} · ${H.esc(row.source||'Source unavailable')}</p></div><a href="/review?session=${encodeURIComponent(row.id)}" class="btn">Review frames</a></div><div style="margin-top:14px">${H.reading(result(row),{label:'2D:4D measurement'})}</div><div class="grid grid-3" style="margin-top:14px"><div><span class="label">Frames</span><strong class="num">${H.n(row.n_valid)} valid of ${H.n(row.n_frames)}</strong></div><div><span class="label">Frame subset</span><strong class="num">${H.n(row.n_sharp)} sharp · ${H.n(row.n_blurry)} soft</strong></div><div><span class="label">Method</span><strong class="mono">${H.esc(row.model_checkpoint||'—')}</strong></div></div></article>`;}
function page(){H.byId('app').innerHTML=H.shell({active:'/subjects',eyebrow:'Subject detail',title:subject?`Subject ${subject}`:'Subject detail',subtitle:'Each capture is preserved separately with its ratio, SD and quality verdict. No average is substituted for the capture record.',actions:'<a class="btn" href="/subjects">All subjects</a>',content:`<div class="instrument-note">The record is populated only from the filtered study-session response. Quality failures are retained as evidence and do not show a ratio as a result.</div><section class="grid grid-2" style="margin-top:16px"><article class="card card-pad"><h2>Capture history</h2><p class="card-note">Repeated captures are listed individually; hand is shown on each record.</p><div id="history" class="stack" style="margin-top:14px">${H.loading(4,'Loading subject captures')}</div></article><article class="card card-pad"><h2>Recorded covariates</h2><p class="card-note">Values are shown only when returned with this subject’s capture rows.</p><div id="covariates" class="stack" style="margin-top:14px">${H.loading(4)}</div></article></section>`});H.bindPreferences();}
function render(rows){const sorted=[...(rows||[])].sort((a,b)=>String(b.utc||'').localeCompare(String(a.utc||'')));H.byId('history').innerHTML=sorted.length?sorted.map(captureCard).join(''):H.state('empty','No captures returned for this subject','No study-session rows matched this server-issued subject ID.');H.byId('covariates').innerHTML=sorted.length?`<div class="grid grid-2"><div><span class="label">Age</span><strong>${values(sorted,'age')}</strong></div><div><span class="label">Sex</span><strong>${values(sorted,'sex')}</strong></div><div><span class="label">Autistic</span><strong>${values(sorted,'autistic')}</strong></div><div><span class="label">Nationality</span><strong>${values(sorted,'nationality')}</strong></div><div><span class="label">Ethnicity</span><strong>${values(sorted,'ethnicity')}</strong></div><div><span class="label">Dominant hand</span><strong>${values(sorted,'dominant_hand')}</strong></div><div><span class="label">Nail overhang</span><strong>${values(sorted,'nail_overhang')}</strong></div><div><span class="label">Protocol</span><strong class="mono">${values(sorted,'protocol')}</strong></div></div><div class="notice"><strong>Operator values:</strong> ${values(sorted,'operator')}</div>`:H.state('empty','No covariates returned','Covariates are not inferred when the filtered response is empty.');}
page();if(!subject){H.byId('history').innerHTML=H.state('empty','No subject selected','Return to subject records and enter a server-issued subject ID.');H.byId('covariates').innerHTML='';}else{H.api(`/api/study/sessions?subject=${encodeURIComponent(subject)}`).then(render).catch(error=>{H.byId('history').innerHTML=H.state('error','Could not request subject sessions',error.message);H.byId('covariates').innerHTML=H.state('error','Covariates unavailable',error.message);});}
</script></body></html>"""


METHOD_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#126c74"><link rel="stylesheet" href="/static/console.css"><script src="/static/console.js"></script><title>Method | Hand Study</title></head><body><div id="app"></div><script>
const H=HandConsole;
function page(){H.byId('app').innerHTML=H.shell({active:'/method',eyebrow:'Research instrument',title:'Method and limitations',subtitle:'This system measures a 2D:4D ratio from palm photographs. It does not screen for, diagnose or predict any condition.',actions:'<a class="btn" href="/dashboard">Study dashboard</a>',content:`<section class="card card-pad"><h2>What is measured</h2><p class="card-note" style="font-size:14px;margin-top:8px">The 2D:4D ratio is the measured index-finger length divided by the measured ring-finger length. The system identifies the base and tip landmark for each of those fingers in a palm photograph, measures the two pixel distances, then reports their ratio only when the capture passes quality gates.</p><div class="method-flow" style="margin-top:16px"><article class="flow-step"><span class="quality unknown">1</span><strong>Photograph</strong><p>A short multi-frame palm capture is collected.</p></article><article class="flow-step"><span class="quality unknown">2</span><strong>Locate landmarks</strong><p>The model estimates four landmarks: index base/tip and ring base/tip.</p></article><article class="flow-step"><span class="quality unknown">3</span><strong>Measure</strong><p>Index length ÷ ring length is calculated from the selected frames.</p></article><article class="flow-step"><span class="quality unknown">4</span><strong>Qualify</strong><p>Spread and capture evidence determine whether a result is reported.</p></article></div></section><section class="grid grid-2" style="margin-top:16px"><article class="card card-pad"><h2>Measurement settings</h2><div id="settings" style="margin-top:14px">${H.loading(4)}</div></article><article class="card card-pad"><h2>Quality thresholds</h2><table style="min-width:0"><thead><tr><th>Condition</th><th>Verdict</th><th>Reason</th></tr></thead><tbody><tr><td>SD ratio &gt; 0.08</td><td>${H.quality({level:'fail'})}</td><td>Readings disagree wildly.</td></tr><tr><td>SD ratio &gt; 0.04</td><td>${H.quality({level:'poor'})}</td><td>Readings vary substantially.</td></tr><tr><td>Fewer than 3 valid frames</td><td>${H.quality({level:'fail'})}</td><td>Too little frame evidence for a reliable measurement.</td></tr></tbody></table><p class="card-note" style="margin-top:12px">Model repeatability is 0.016 ratio units; the human inter-observer benchmark is approximately 0.008. Blur is assessed against a 60.0 Laplacian-variance floor.</p></article></section><section class="card card-pad" style="margin-top:16px"><h2>Limitations</h2><div class="grid grid-3" style="margin-top:14px"><div><h3>Image geometry</h3><p class="card-note">A photograph measures projected finger length. Tilt is a prominent source of error and does not necessarily cancel between fingers.</p></div><div><h3>Capture quality</h3><p class="card-note">Blur, poor framing and a cluttered background can prevent reliable landmark placement. A low-quality result should be recaptured or reviewed, not treated as a confident value.</p></div><div><h3>Interpretation</h3><p class="card-note">Reference bands provide context only. They are not a verdict about an individual and do not support diagnostic, screening or predictive claims.</p></div></div></section>`});H.bindPreferences();}
function settings(schema,info){H.byId('settings').innerHTML=`<div class="stack"><div><span class="label">Crease convention</span><strong class="mono">${H.esc(schema.crease_convention||'Unavailable')}</strong></div><div><span class="label">Model checkpoint</span><strong class="mono">${H.esc(info.checkpoint||schema.model_checkpoint||'Unavailable')}</strong></div><div><span class="label">Validation ratio MAE</span><strong class="num">${H.n(info.val_ratio_mae,6)}</strong></div><div><span class="label">Inference configuration</span><strong>${H.esc(info.device||'Unavailable')} · input ${H.n(info.input_size)} · TTA ${info.tta?'on':'off'}</strong></div></div>`;}
page();Promise.all([H.api('/api/schema'),H.api('/api/info')]).then(([schema,info])=>settings(schema,info)).catch(error=>{H.byId('settings').innerHTML=H.state('error','Could not load method settings',error.message);});
</script></body></html>"""


DEVICES_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#126c74"><link rel="stylesheet" href="/static/console.css"><script src="/static/console.js"></script><title>Devices | Hand Study</title></head><body><div id="app"></div><script>
const H=HandConsole;let devices=[];
function page(){H.byId('app').innerHTML=H.shell({active:'/devices',eyebrow:'Capture infrastructure',title:'Enrolled devices',subtitle:'Activation makes an enrolled device eligible for the app-install workflow.',actions:'<a class="btn btn-primary" href="/enroll">Enrol device</a><a class="btn" href="/install">Install page</a>',content:`<section class="card card-pad"><div class="inline justify"><div><h2>Device activation</h2><p class="card-note">Device information is returned by the documented /api/devices response.</p></div><span id="slot-summary" class="notice">Loading slots…</span></div><div id="device-list" style="margin-top:16px">${H.loading(4)}</div></section>`});H.bindPreferences();}
function render(data){devices=data.devices||[];H.byId('slot-summary').textContent=`${H.n(data.activated)} active of ${H.n(data.slots)} slots`;H.byId('device-list').innerHTML=devices.length?`<div class="table-wrap"><table><thead><tr><th>Device</th><th>UDID</th><th>Last seen</th><th>Activation</th><th>Action</th></tr></thead><tbody>${devices.map(device=>`<tr><td>${H.esc(device.name||device.product||'Unnamed device')}<br><span class="muted">${H.esc(device.product||'—')} · iOS ${H.esc(device.version||'—')}</span></td><td class="mono">${H.esc(device.udid)}</td><td>${H.iso(device.last_seen)}</td><td>${device.activated?'<span class="quality ok">Activated</span>':'<span class="quality unknown">Not activated</span>'}</td><td><button class="btn ${device.activated?'btn-danger':'btn-primary'}" data-udid="${H.esc(device.udid)}" data-on="${device.activated?'0':'1'}">${device.activated?'Deactivate':'Activate'}</button></td></tr>`).join('')}</tbody></table></div>`:H.state('empty','No devices are enrolled','Open the enrolment flow on the iPhone that will run the capture app.');H.byId('device-list').querySelectorAll('button[data-udid]').forEach(button=>button.onclick=()=>toggle(button.dataset.udid,button.dataset.on==='1'));}
function toggle(udid,on){const title=on?'Activate this device?':'Deactivate this device?';const consequence=on?'The device becomes eligible for the server installation workflow.':'The device will no longer be active for the capture-app installation workflow until reactivated.';H.confirmDialog({title,detail:`Device ${udid}`,consequence,confirmLabel:on?'Activate device':'Deactivate device',destructive:!on,onConfirm:async()=>{await H.api(`/api/devices/${encodeURIComponent(udid)}/activate`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({activated:on})});const data=await H.api('/api/devices');render(data);}});}
page();H.api('/api/devices').then(render).catch(error=>{H.byId('device-list').innerHTML=H.state('error','Could not load enrolled devices',error.message);});
</script></body></html>"""


@app.after_request
def _no_cache_api(resp):
    """Stop the browser caching API responses.

    The pages were given no-store but the JSON endpoints were not, and a plain
    GET with no cache headers is fair game for heuristic caching. A response
    fetched during the re-scan -- when predictions were briefly unavailable --
    then kept being replayed from disk with no request reaching the server at
    all, which is why this looked like a server bug that testing could not
    reproduce.

    Thumbnails are exempt: their urls already carry a version stamp, so they
    are safe to cache hard and expensive to re-render.
    """
    pth = request.path or ""
    if pth.startswith("/api/"):
        if pth.startswith("/api/review/thumb") or pth.startswith("/api/review/image"):
            resp.headers.setdefault("Cache-Control", "public, max-age=604800, immutable")
        else:
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
    return resp


def _page(html: str) -> Response:
    """Serve a page uncached.

    These pages carry their own JavaScript inline, so a browser holding a
    cached copy silently runs an old build -- which looked exactly like a
    server-side bug ("model could not read this image" long after the fix
    shipped). No headers meant heuristic caching, which is the worst case:
    invisible and inconsistent between machines.
    """
    r = Response(html, mimetype="text/html")
    r.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r


@app.get("/capture")
def capture_page():
    return _page(USER_PAGE)


@app.get("/")
def root():
    """The public project page -- and the console for anyone already signed in.

    A printed QR points at this bare hostname, so an anonymous visitor must
    never meet a login wall here. Staff carrying a session are sent on to the
    dashboard, so an existing bookmark does not change meaning.
    """
    if auth_enabled() and _cookie_ok(request.cookies.get("hs_session", "")):
        return redirect("/dashboard")
    return send_from_directory(REPO / "app" / "static", "project.html")


COMPARE_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#126c74"><link rel="stylesheet" href="/static/console.css"><script src="/static/console.js"></script><title>Compare overlays | Hand Study</title></head><body><div id="app"></div><script>
const H=HandConsole;let pool=[],index=0,votes={};
function page(){H.byId('app').innerHTML=H.shell({active:'/compare',eyebrow:'Blind A/B review',title:'Compare model overlays',subtitle:'Use the keyboard for repeated judgement: 1 for left, 2 for right, 3 for unable to tell.',actions:'<span id="progress" class="muted num"></span>',content:`<div id="comparison">${H.loading(4)}</div><section class="card card-pad" style="margin-top:16px"><div class="vote-actions"><button class="btn btn-primary" data-vote="a">Left is better <span class="key">1</span></button><button class="btn btn-primary" data-vote="b">Right is better <span class="key">2</span></button><button class="btn" data-vote="tie">Cannot tell <span class="key">3</span></button></div><div class="inline justify" style="margin-top:14px"><p id="compare-note" class="card-note">The assignment of model variant to side is not exposed by the pool response.</p><button class="btn" id="tally">Show running tally</button></div><div id="tally-result" class="notice" hidden style="margin-top:12px"></div></section>`});H.bindPreferences();}
function render(){const row=pool[index];if(!row){H.byId('comparison').innerHTML=H.state('empty','No comparison pool is available','Build a pool before collecting blinded visual judgements.');return;}H.byId('progress').textContent=`${index+1} of ${pool.length}`;H.byId('comparison').innerHTML=`<div class="compare-images"><article class="compare-image"><header><span>Overlay A</span><span class="muted">${H.n(row.disagree_px,1)} px difference</span></header><img src="/api/compare/img/${index}?which=a&v=${encodeURIComponent(row.id)}" alt="Blind comparison overlay A"></article><article class="compare-image"><header><span>Overlay B</span><span class="muted">Max ${H.n(row.max_pt_px,1)} px</span></header><img src="/api/compare/img/${index}?which=b&v=${encodeURIComponent(row.id)}" alt="Blind comparison overlay B"></article></div>`;}
async function vote(pick){const row=pool[index];if(!row)return;try{await H.api('/api/compare/vote',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:row.id,pick})});votes[row.id]=pick;if(index<pool.length-1){index+=1;render();}else H.byId('compare-note').textContent='All currently available comparisons have a recorded vote.';}catch(error){H.byId('compare-note').textContent=`Could not record vote: ${error.message}`;}}
page();document.addEventListener('keydown',event=>{if(event.key==='1')vote('a');if(event.key==='2')vote('b');if(event.key==='3')vote('tie');if(event.key==='ArrowLeft'){index=Math.max(0,index-1);render();}if(event.key==='ArrowRight'){index=Math.min(pool.length-1,index+1);render();}});document.querySelectorAll('[data-vote]').forEach(button=>button.onclick=()=>vote(button.dataset.vote));H.byId('tally').onclick=async()=>{try{const data=await H.api('/api/compare/tally');const el=H.byId('tally-result');el.hidden=false;el.textContent=`Votes cast: ${H.n(data.n)}. Drift: ${H.n(data.drift)}; aniso: ${H.n(data.aniso)}; unable to tell: ${H.n(data.tie)}. ${data.note}`;}catch(error){H.byId('compare-note').textContent=error.message;}};H.api('/api/compare/pool').then(data=>{pool=data.items||[];votes=data.votes||{};render();}).catch(error=>{H.byId('comparison').innerHTML=H.state('error','Could not load the comparison pool',error.message);});
</script></body></html>"""


# --- model A/B comparison -----------------------------------------------------
# The reviewed labels cannot referee these two models: 133 of 153 sit within 1 px
# of driftfull's own output, because the reviewer accepted the AI's points rather
# than placing them independently. Scoring against them measures authorship, not
# accuracy. So the pool is drawn from held-out val images the reviewer has never
# touched, and the verdict comes from a blind human vote.
COMPARE_POOL = None
COMPARE_VOTES = REPO / "agents" / "compare_votes.json"


def _compare_pool():
    global COMPARE_POOL
    if COMPARE_POOL is None:
        f = REPO / "agents" / "compare_pool.json"
        COMPARE_POOL = json.loads(f.read_text(encoding="utf-8")) if f.is_file() else []
    return COMPARE_POOL


def _swapped(ident):
    """Stable per-image blind assignment: whether 'A' is anisofull."""
    return hashlib.sha1(ident.encode()).digest()[0] % 2 == 1


def _read_votes():
    if COMPARE_VOTES.is_file():
        try:
            return json.loads(COMPARE_VOTES.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


@app.get("/compare")
def compare():
    return _page(COMPARE_PAGE)


@app.get("/api/compare/pool")
def compare_pool():
    rows = _compare_pool()
    items = [{"id": r["id"], "disagree_px": r["disagree_px"],
              "max_pt_px": r["max_pt_px"]} for r in rows]
    return jsonify({"items": items, "votes": _read_votes()})


@app.get("/api/compare/img/<int:i>")
def compare_img(i):
    rows = _compare_pool()
    if not (0 <= i < len(rows)):
        return jsonify({"error": "out of range"}), 404
    r = rows[i]
    im = cv2.imread(str(REPO / r["path"]))
    if im is None:
        return jsonify({"error": "unreadable"}), 404
    which = request.args.get("which", "a")
    sw = _swapped(r["id"])
    key = ("aniso" if sw else "drift") if which == "a" else ("drift" if sw else "aniso")
    P = np.asarray(r[key], float)
    dr = np.asarray(r["drift"], float)
    an = np.asarray(r["aniso"], float)
    both = np.vstack([dr, an])

    if request.args.get("tight") == "1":
        c = P[int(np.argmax(np.linalg.norm(dr - an, axis=1)))]
        half = 95
        x0, y0 = int(c[0] - half), int(c[1] - half)
        x1, y1 = x0 + 2 * half, y0 + 2 * half
    else:
        pad = 0.18 * max(np.ptp(both[:, 0]), np.ptp(both[:, 1]), 40)
        x0, y0 = int(both[:, 0].min() - pad), int(both[:, 1].min() - pad)
        x1, y1 = int(both[:, 0].max() + pad), int(both[:, 1].max() + pad)
    H, W = im.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return jsonify({"error": "degenerate crop"}), 404

    crop = im[y0:y1, x0:x1].copy()
    sc = min(6.0, max(1.0, 1100.0 / max(crop.shape[1], 1)))
    if sc > 1.01:
        crop = cv2.resize(crop, None, fx=sc, fy=sc,
                          interpolation=cv2.INTER_NEAREST if sc >= 3 else cv2.INTER_CUBIC)

    # Markers are drawn AFTER upscaling, deliberately. A dot sized in source
    # pixels becomes a blob at 4-6x zoom and hides the very pixels being judged --
    # the whole point here is to see whether the landmark sits on the fingertip.
    Q = (P - np.array([x0, y0], float)) * sc
    col = [(80, 160, 255), (60, 220, 255), (120, 230, 80), (255, 180, 60)]
    ipt = lambda q: (int(round(q[0])), int(round(q[1])))
    cv2.line(crop, ipt(Q[0]), ipt(Q[1]), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(crop, ipt(Q[2]), ipt(Q[3]), (255, 255, 255), 1, cv2.LINE_AA)
    for k, q in enumerate(Q):
        cx, cy = ipt(q)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):      # gapped crosshair
            cv2.line(crop, (cx + dx * 3, cy + dy * 3), (cx + dx * 10, cy + dy * 10),
                     (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(crop, (cx, cy), 6, col[k], 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return jsonify({"error": "encode failed"}), 500
    resp = make_response(buf.tobytes())
    resp.headers["Content-Type"] = "image/jpeg"
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.post("/api/compare/vote")
def compare_vote():
    d = request.get_json(force=True, silent=True) or {}
    ident, pick = d.get("id"), d.get("pick")
    if not ident or pick not in ("a", "b", "tie"):
        return jsonify({"error": "bad vote"}), 400
    v = _read_votes()
    v[ident] = pick
    COMPARE_VOTES.parent.mkdir(parents=True, exist_ok=True)
    COMPARE_VOTES.write_text(json.dumps(v, indent=1), encoding="utf-8")
    return jsonify({"ok": True, "n": len(v)})


@app.get("/api/compare/tally")
def compare_tally():
    v = _read_votes()
    drift = aniso = tie = 0
    for ident, pick in v.items():
        if pick == "tie":
            tie += 1
        elif (pick == "a") != _swapped(ident):
            drift += 1
        else:
            aniso += 1
    n = drift + aniso
    if n == 0:
        note = "No decisive votes yet."
    else:
        note = (str(max(drift, aniso)) + "/" + str(n) + " decisive votes for the "
                "leader. The two models place points 1.9 px apart on average, so a "
                "near-even split is the expected result and would mean neither is "
                "visibly better.")
    return jsonify({"drift": drift, "aniso": aniso, "tie": tie,
                    "n": len(v), "note": note})


ENROLL_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#126c74"><link rel="stylesheet" href="/static/console.css"><script src="/static/console.js"></script><title>Enrol device | Hand Study</title></head><body><div id="app"></div><script>
const H=HandConsole;H.byId('app').innerHTML=H.shell({active:'/devices',eyebrow:'Device enrolment',title:'Register an iPhone',subtitle:'This registration flow is for the capture app. Console sign-in uses a browser password; the app uses its own device token.',actions:'<a class="btn" href="/devices">Device list</a>',content:`<section class="card card-pad" style="max-width:720px"><div class="steps"><article class="step"><div><h2>Download the registration profile</h2><p>Open this page on the device being registered, then request the profile.</p><p><a class="btn btn-primary" href="/enroll/profile.mobileconfig">Register this iPhone</a></p></div></article><article class="step"><div><h2>Install the profile in Settings</h2><p>iOS displays the installed profile and returns its device identifier to the registration endpoint.</p></div></article><article class="step"><div><h2>Await activation</h2><p>A device becomes installable only after an operator activates it. Registration does not itself activate the device.</p></div></article><article class="step"><div><h2>Install the app build</h2><p>When activated, return to the installation link provided by the device workflow.</p></div></article></div></section>`});H.bindPreferences();
</script></body></html>"""


INSTALL_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#126c74"><link rel="stylesheet" href="/static/console.css"><script src="/static/console.js"></script><title>Install | Hand Study</title></head><body><div id="app"></div><script>
const H=HandConsole;H.byId('app').innerHTML=H.shell({active:'/devices',eyebrow:'Device installation',title:'Hand Study capture app',subtitle:'The server determines whether this registered device can install the build.',actions:'<a class="btn" href="/enroll">Enrol device</a>',content:`<section class="card card-pad" style="max-width:650px"><div class="install-body">{{BODY}}</div><p class="card-note" style="margin-top:20px">The iOS app authenticates with a device token. This browser installation page does not provide console access.</p></section>`});H.bindPreferences();
</script></body></html>"""


# ---------------------------------------------------------- subject ids ----
# Subject numbers are issued by the server, not typed on the phone. With five
# devices in the field two operators would otherwise both decide the next child
# is "12". The counter lives with the study data and is bumped under a lock.
SUBJECT_FILE = CAPTURES / "study" / "subjects.json"
_SUBJECT_LOCK = threading.Lock()


def _subjects() -> dict:
    if SUBJECT_FILE.is_file():
        try:
            return json.loads(SUBJECT_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"next": 1, "issued": []}


@app.post("/api/subject/next")
def subject_next():
    """Hand out the next subject number and record who took it.

    Deliberately a POST with a record of issue rather than a bare counter: if a
    device asks for a number and then loses it (app killed before the first
    capture), the gap is visible in the issued list rather than being silently
    reused for a different child.
    """
    body = request.get_json(silent=True) or {}
    dev = str(body.get("device") or "")[:64]
    note = str(body.get("note") or "")[:120]
    with _SUBJECT_LOCK:
        st = _subjects()
        n = int(st.get("next", 1))
        st["next"] = n + 1
        st.setdefault("issued", []).append({
            "subject": str(n), "device": dev, "note": note,
            "utc": datetime.now(timezone.utc).isoformat(),
        })
        SUBJECT_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = SUBJECT_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1), encoding="utf-8")
        tmp.replace(SUBJECT_FILE)
    return jsonify({"subject": str(n)})


@app.get("/api/subject/list")
def subject_list():
    st = _subjects()
    return jsonify({"next": st.get("next", 1), "issued": st.get("issued", [])})


# ------------------------------------------------------ device enrolment ----
# Ad-hoc distribution needs each device's UDID registered in the provisioning
# profile before it can install. iOS will report its own UDID if it is sent a
# "Profile Service" configuration profile, which is what /enroll serves: the
# device posts its identifiers back here, we approve it, re-sign, and only then
# does the install link work.
DEVICES_FILE = REPO / "agents" / "devices.json"
_DEVICE_LOCK = threading.Lock()


def _devices() -> dict:
    if DEVICES_FILE.is_file():
        try:
            return json.loads(DEVICES_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_devices(d: dict) -> None:
    DEVICES_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEVICES_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
    tmp.replace(DEVICES_FILE)


def _base_url() -> str:
    return request.url_root.rstrip("/").replace("http://", "https://")


@app.get("/enroll")
def enroll():
    return _page(ENROLL_PAGE)


@app.get("/enroll/profile.mobileconfig")
def enroll_profile():
    """The Profile Service payload. Installing it makes iOS POST its UDID back.

    Unsigned, so iOS shows it as 'Unverified'. That is cosmetic but alarming to a
    non-technical user, and signing it with the Apple developer certificate is a
    build step worth doing before this goes to anyone outside the team.
    """
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>PayloadContent</key><dict>
    <key>URL</key><string>{_base_url()}/api/enroll/udid</string>
    <key>DeviceAttributes</key>
    <array><string>UDID</string><string>PRODUCT</string>
      <string>VERSION</string><string>DEVICE_NAME</string></array>
  </dict>
  <key>PayloadType</key><string>Profile Service</string>
  <key>PayloadIdentifier</key><string>sa.tdev.hand.enroll</string>
  <key>PayloadUUID</key><string>{uuid.uuid4()}</string>
  <key>PayloadVersion</key><integer>1</integer>
  <key>PayloadDisplayName</key><string>Hand Study — register this iPhone</string>
  <key>PayloadDescription</key><string>Reports this device identifier so the app can be installed on it.</string>
  <key>PayloadOrganization</key><string>Hand Study</string>
</dict></plist>"""
    resp = make_response(body)
    resp.headers["Content-Type"] = "application/x-apple-aspen-config"
    resp.headers["Content-Disposition"] = 'attachment; filename="enroll.mobileconfig"'
    return resp


@app.post("/api/enroll/udid")
def enroll_udid():
    """iOS posts a PKCS#7-signed plist here. The UDID is pulled out of it.

    The signature is not verified: this endpoint grants nothing on its own -- a
    device still has to be approved by hand and the build re-signed before it can
    install anything -- so a forged post costs an unwanted row in a list, not
    access.
    """
    raw = request.get_data() or b""
    txt = raw.decode("utf-8", "ignore")

    def field(name):
        m = re.search(r"<key>" + name + r"</key>\s*<string>([^<]*)</string>", txt)
        return m.group(1).strip() if m else None

    udid = field("UDID")
    if not udid or not re.fullmatch(r"[A-Za-z0-9-]{20,64}", udid):
        return make_response("could not read device id", 400)
    with _DEVICE_LOCK:
        d = _devices()
        rec = d.get(udid) or {"activated": False,
                              "first_seen": datetime.now(timezone.utc).isoformat()}
        rec.update({"product": field("PRODUCT"), "version": field("VERSION"),
                    "name": field("DEVICE_NAME"),
                    "last_seen": datetime.now(timezone.utc).isoformat()})
        d[udid] = rec
        _save_devices(d)
    # iOS follows this redirect in the profile-install browser sheet.
    resp = make_response("", 302)
    resp.headers["Location"] = f"{_base_url()}/install?udid={udid}"
    return resp


@app.get("/install")
def install():
    udid = (request.args.get("udid") or "").strip()
    d = _devices()
    rec = d.get(udid)
    if rec and rec.get("activated"):
        link = ("itms-services://?action=download-manifest&url="
                + _base_url() + "/install/manifest.plist")
        body = ("<h2>Ready to install</h2><p>" + (rec.get("name") or "This iPhone")
                + "</p><p><a class=b href=\"" + link + "\">Install the app</a></p>")
    elif rec:
        body = ("<h2>Registered</h2><p>This iPhone has been registered and is "
                "waiting to be approved. You will be told when it is ready — then "
                "open this page again.</p><p class=m>" + udid + "</p>")
    else:
        body = ("<h2>Not registered yet</h2><p>Install the registration profile "
                "first.</p><p><a class=b href=\"/enroll\">Go back</a></p>")
    return _page(INSTALL_PAGE.replace("{{BODY}}", body))


@app.get("/install/manifest.plist")
def install_manifest():
    """OTA manifest. IPA_URL must point at an https-hosted build."""
    ipa = _base_url() + "/install/app.ipa"
    body = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>items</key><array><dict>
  <key>assets</key><array><dict>
    <key>kind</key><string>software-package</string>
    <key>url</key><string>{ipa}</string>
  </dict></array>
  <key>metadata</key><dict>
    <key>bundle-identifier</key><string>sa.tdev.hand.capture</string>
    <key>bundle-version</key><string>1.0</string>
    <key>kind</key><string>software</string>
    <key>title</key><string>Hand Study</string>
  </dict>
</dict></array></dict></plist>"""
    resp = make_response(body)
    resp.headers["Content-Type"] = "application/xml"
    return resp


@app.get("/install/app.ipa")
def install_ipa():
    f = REPO / "dist" / "app.ipa"
    if not f.is_file():
        return jsonify({"error": "no build uploaded yet"}), 404
    return send_file(f, mimetype="application/octet-stream",
                     as_attachment=True, download_name="app.ipa")


@app.get("/api/devices")
def devices_list():
    d = _devices()
    return jsonify({"devices": [dict(v, udid=k) for k, v in sorted(d.items())],
                    "activated": sum(1 for v in d.values() if v.get("activated")),
                    "slots": 5})


@app.post("/api/devices/<udid>/activate")
def device_activate(udid: str):
    on = bool((request.get_json(silent=True) or {}).get("activated", True))
    with _DEVICE_LOCK:
        d = _devices()
        if udid not in d:
            return jsonify({"error": "unknown device"}), 404
        live = sum(1 for k, v in d.items() if v.get("activated") and k != udid)
        if on and live >= 5:
            return jsonify({"error": "all 5 device slots are in use"}), 409
        d[udid]["activated"] = on
        d[udid]["activated_utc"] = (datetime.now(timezone.utc).isoformat()
                                    if on else None)
        _save_devices(d)
    return jsonify({"ok": True, "udid": udid, "activated": on})


LOGIN_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="theme-color" content="#126c74"><link rel="stylesheet" href="/static/console.css"><script src="/static/console.js"></script><title>Sign in | Hand Study</title></head><body class="auth-page"><form class="auth-card" id="login-form"><h1>Hand Study</h1><label><span class="label">Password</span><input class="field" id="password" type="password" autocomplete="current-password" autofocus required></label><button class="btn btn-primary" style="width:100%;margin-top:12px" type="submit">Sign in</button><p class="err" id="login-error" aria-live="polite"></p><p class="minor">The iOS capture app uses a device token, not this page.</p><div class="minor" style="text-align:center;margin:16px 0 8px">or</div><button class="btn" style="width:100%" type="button" id="demo-login">Try the demo console</button><p class="minor" style="text-align:center">Sample data only — no real participants.</p></form><script>
const H=HandConsole;H.bindPreferences();H.byId('login-form').onsubmit=async event=>{event.preventDefault();const error=H.byId('login-error');error.textContent='';try{await H.api('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:H.byId('password').value})});const next=new URLSearchParams(location.search).get('next')||'/dashboard';location.href=next.startsWith('/')?next:'/dashboard';}catch(err){error.textContent=err.message;}};H.byId('demo-login').onclick=async()=>{const error=H.byId('login-error');error.textContent='';try{await H.api('/api/demo-login',{method:'POST'});location.href='/dashboard';}catch(err){error.textContent=err.message;}};
</script></body></html>"""


# ------------------------------------------------------------------ auth ----
# Everything that can show a captured image is behind a login. Until now the
# review and admin pages were world-readable, which was tolerable for adult
# volunteer hands and is not once a child has been photographed under an ethics
# approval.
#
# Two ways in, because two different things need access:
#   - a password, for a person using /review or /admin in a browser
#   - a device token, for the capture app posting frames
# The capture page itself stays open: it holds no data, and a phone in a school
# should not be blocked by a forgotten password. What it POSTS is what matters,
# and that needs the token.
AUTH_FILE = REPO / "agents" / "auth.json"
_SESSION_TTL = 14 * 24 * 3600


def _auth_conf() -> dict:
    if AUTH_FILE.is_file():
        try:
            return json.loads(AUTH_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _pw_hash(pw: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 200_000).hex()


def _sign(value: str) -> str:
    conf = _auth_conf()
    secret = conf.get("secret") or ""
    return hmac.new(secret.encode(), value.encode(), hashlib.sha256).hexdigest()[:32]


def _issue_cookie() -> str:
    exp = str(int(time.time()) + _SESSION_TTL)
    return f"{exp}.{_sign(exp)}"


def _cookie_ok(raw: str) -> bool:
    try:
        exp, sig = (raw or "").split(".", 1)
    except ValueError:
        return False
    if not hmac.compare_digest(sig, _sign(exp)):
        return False
    try:
        return int(exp) > time.time()
    except ValueError:
        return False


def _token_ok(tok: str) -> bool:
    conf = _auth_conf()
    for t in conf.get("device_tokens", []):
        if hmac.compare_digest(str(t), tok or ""):
            return True
    return False


def auth_enabled() -> bool:
    c = _auth_conf()
    return bool(c.get("password_hash") and c.get("secret"))


# Open to anyone: the capture page and the assets it needs, the enrolment and
# install flow (a phone has no cookie before it has the app), and the health
# check. Everything else requires a session or a device token.
# The capture page is NOT public. It was, on the reasoning that it holds no data
# -- but its uploads need a credential, so leaving the page open just produced a
# page that 401s when you press record. Requiring a login is both more secure and
# less confusing. The app carries a device token instead and never sees this page.
_PUBLIC_EXACT = {
    "/login", "/api/login", "/api/info", "/api/schema",
    "/enroll", "/enroll/profile.mobileconfig", "/api/enroll/udid",
    "/install", "/install/manifest.plist", "/install/app.ipa",
    "/project", "/", "/api/demo-login",
}


# --- public demo mode ---------------------------------------------------------
# "Try the demo console" gives a visitor the real console pages driven by the
# synthetic fixtures in fixtures/ -- no participant, no photograph, no study
# record. The demo is an ALLOWLIST answered inside the auth gate: a demo request
# never reaches a real handler, and anything not listed below is a 404,
# including routes added later. A missed route therefore fails closed instead of
# leaking study data. The demo cookie is signed over a different message from
# the staff cookie and lives under a different name, so neither can stand in for
# the other; a real staff session always takes precedence.
DEMO_FIXTURES = REPO / "fixtures"
_DEMO_TTL = 2 * 3600
_DEMO_BANNER = (
    '<div role="note" style="position:sticky;top:0;z-index:9999;background:#57BDB3;'
    'color:#0B1412;font:600 14px/1.4 system-ui,-apple-system,sans-serif;padding:8px 16px;'
    'text-align:center">Demo console · sample data only — no real participants</div>'
)
_DEMO_PAGES = {
    "/capture": "USER_PAGE", "/dashboard": "DASHBOARD_PAGE", "/subjects": "SUBJECTS_PAGE",
    "/subject": "SUBJECT_PAGE", "/review": "REVIEW_PAGE", "/admin": "ADMIN_PAGE",
    "/devices": "DEVICES_PAGE", "/compare": "COMPARE_PAGE", "/method": "METHOD_PAGE",
}
_DEMO_JSON = {
    "/api/admin/sessions": "admin_sessions.json", "/api/study/summary": "study_summary.json",
    "/api/study/storage": "study_storage.json", "/api/admin/frames": "admin_frames.json",
    "/api/review/list": "review_list.json", "/api/devices": "devices.json",
    "/api/compare/pool": "compare_pool.json", "/api/compare/tally": "compare_tally.json",
}


def _issue_demo_cookie() -> str:
    exp = str(int(time.time()) + _DEMO_TTL)
    return f"demo.{exp}.{_sign('demo:' + exp)}"


def _demo_ok(raw: str) -> bool:
    try:
        tag, exp, sig = (raw or "").split(".", 2)
    except ValueError:
        return False
    if tag != "demo" or not hmac.compare_digest(sig, _sign("demo:" + exp)):
        return False
    try:
        return int(exp) > time.time()
    except ValueError:
        return False


def _demo_fixture(name: str):
    return json.loads((DEMO_FIXTURES / name).read_text(encoding="utf-8"))


def _demo_placeholder(label: str) -> Response:
    safe = label.replace("&", "and").replace("<", "").replace(">", "")[:42]
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="768" viewBox="0 0 1024 768">'
           '<rect width="1024" height="768" fill="#17272d"/>'
           '<path d="M300 700c-55-154-62-319-12-423 31-65 79-54 79 13v148h32V156c0-81 61-80 61 0v272h33V112c0-80 62-80 62 0v316h32V177c0-76 60-76 60 0v257c77-90 145-29 79 54l-112 141c-57 73-159 105-314 71z" fill="#c99176"/>'
           '<g fill="none" stroke="#fff" stroke-width="4"><path stroke-dasharray="9 7" d="M440 470L490 150M585 479L586 170"/>'
           '<circle cx="440" cy="470" r="11"/><circle cx="490" cy="150" r="11"/><circle cx="585" cy="479" r="11"/><circle cx="586" cy="170" r="11"/></g>'
           f'<text x="42" y="60" fill="#d7e5e8" font-family="system-ui,sans-serif" font-size="28">Sample image — {safe}</text></svg>')
    return Response(svg, mimetype="image/svg+xml")


def _demo_dispatch(path: str):
    if path == "/api/logout" and request.method == "POST":
        # Logout only clears cookies; a demo visitor must be able to leave.
        return api_logout()
    if request.method == "GET":
        if path in _DEMO_PAGES:
            html = globals()[_DEMO_PAGES[path]]
            return _page(re.sub(r"(<body[^>]*>)", lambda m: m.group(1) + _DEMO_BANNER, html, count=1))
        if path == "/api/study/sessions":
            rows = list(_demo_fixture("study_sessions.json"))
            for key in ("subject", "store", "quality"):
                value = request.args.get(key, "")
                if value:
                    rows = [r for r in rows if str(r.get(key, "")) == value]
            limit = request.args.get("limit", "")
            if limit.isdigit():
                rows = rows[:int(limit)]
            return jsonify(rows)
        if path in _DEMO_JSON:
            return jsonify(_demo_fixture(_DEMO_JSON[path]))
        if path.startswith("/api/review/item/"):
            return jsonify(_demo_fixture("review_item.json"))
        if path.startswith(("/api/review/thumb/", "/api/review/image/", "/api/compare/img/", "/f/")):
            return _demo_placeholder(path.rsplit("/", 1)[-1])
    elif request.method == "POST":
        body = request.get_json(silent=True) or {}
        if path == "/api/review/save":
            return jsonify({"ok": True, "demo": True, "id": body.get("id", "")})
        if path == "/api/review/exclude":
            return jsonify({"ok": True, "demo": True, "excluded": bool(body.get("excluded", True)), "total_excluded": 1})
        if path.startswith("/api/devices/") and path.endswith("/activate"):
            return jsonify({"ok": True, "demo": True, "udid": path.split("/")[-2], "activated": bool(body.get("activated", True))})
        if path == "/api/compare/vote":
            return jsonify({"ok": True, "demo": True, "n": 2})
        if path == "/api/admin/cleanup":
            return jsonify({"removed": 0, "kept": 19, "files_moved": 0, "moved_to": "set_aside",
                            "deleted": 0, "threshold": 60.0, "dry": True, "demo": True})
        if path == "/api/admin/export":
            return jsonify({"n": 0, "path": "", "demo": True})
    if path.startswith("/api/"):
        return jsonify({"error": "not available in the demo"}), 404
    return redirect("/dashboard")


@app.before_request
def _require_auth():
    if not auth_enabled():
        return None                     # not configured yet: behave as before
    path = request.path
    if path in _PUBLIC_EXACT or path.startswith("/static/"):
        return None
    # Uploads come from the app and from the capture page. A device token is the
    # app's credential; a browser session covers someone testing from the page.
    # Paths the capture app must reach with only a device token. It cannot hold
    # a browser session, and without a server-issued subject number it would
    # have to invent one -- which collides the moment a second device is in the
    # field, and is unrecoverable afterwards.
    if (path in ("/api/session", "/api/subject/next")
            or path.startswith("/api/v2/capture")
            or (path.startswith("/api/session/") and path.endswith("/tags"))):
        tok = (request.headers.get("X-Device-Token")
               or request.form.get("device_token") or "")
        if _token_ok(tok) or _cookie_ok(request.cookies.get("hs_session", "")):
            return None
        return jsonify({"error": "unauthorised"}), 401
    if _cookie_ok(request.cookies.get("hs_session", "")):
        return None
    if _demo_ok(request.cookies.get("hs_demo", "")):
        return _demo_dispatch(path)
    if path.startswith("/api/"):
        return jsonify({"error": "unauthorised"}), 401
    return redirect("/login?next=" + quote(path, safe=""))


@app.get("/login")
def login_page():
    return _page(LOGIN_PAGE)


@app.post("/api/login")
def api_login():
    conf = _auth_conf()
    if not auth_enabled():
        return jsonify({"error": "auth is not configured"}), 400
    pw = (request.get_json(silent=True) or {}).get("password") or ""
    # Constant-time compare, and a deliberate delay so the endpoint cannot be
    # used to guess at speed.
    ok = hmac.compare_digest(_pw_hash(pw, conf.get("salt", "")),
                             conf.get("password_hash", ""))
    time.sleep(0.4)
    if not ok:
        return jsonify({"error": "wrong password"}), 401
    resp = make_response(jsonify({"ok": True}))
    resp.set_cookie("hs_session", _issue_cookie(), max_age=_SESSION_TTL,
                    httponly=True, samesite="Lax", secure=True)
    return resp


@app.post("/api/logout")
def api_logout():
    resp = make_response(jsonify({"ok": True}))
    resp.set_cookie("hs_session", "", max_age=0)
    resp.set_cookie("hs_demo", "", max_age=0)
    return resp


@app.post("/api/demo-login")
def api_demo_login():
    """Start a demo session: the console with sample data, never the study data."""
    if not auth_enabled():
        return jsonify({"error": "auth is not configured"}), 400
    resp = make_response(jsonify({"ok": True, "demo": True}))
    resp.set_cookie("hs_demo", _issue_demo_cookie(), max_age=_DEMO_TTL,
                    httponly=True, samesite="Lax", secure=True)
    return resp


@app.get("/review")
def review():
    return _page(REVIEW_PAGE)


@app.get("/admin")
def admin():
    return _page(ADMIN_PAGE)


@app.get("/static/<path:name>")
def console_static(name: str):
    """Serve the console-only static asset collection."""
    asset_root = REPO / "app" / "static"
    asset = asset_root / name
    if not asset.is_file() or asset.parent != asset_root:
        return jsonify({"error": "not found"}), 404
    return send_from_directory(asset_root, name)


@app.get("/project")
def project_page():
    """The public project page linked from the poster QR.

    Read-only and deliberately outside the console: it serves one static file
    and touches no study data, so a printed QR can never land a stranger on a
    login wall -- or anywhere near a capture.
    """
    return redirect("/", code=302)


@app.get("/dashboard")
def dashboard():
    return _page(DASHBOARD_PAGE)


@app.get("/subjects")
def subjects():
    return _page(SUBJECTS_PAGE)


@app.get("/subject")
def subject_detail():
    return _page(SUBJECT_PAGE)


@app.get("/method")
def method_page():
    return _page(METHOD_PAGE)


@app.get("/devices")
def device_page():
    return _page(DEVICES_PAGE)


@app.get("/api/info")
def info():
    return jsonify(CKPT_INFO)


@app.get("/api/schema")
def api_schema():
    """What a capture may carry. The app reads this rather than hard-coding a
    field list, so adding a field here does not require shipping a new build."""
    return jsonify({
        "tags": {k: list(v) for k, v in TAG_FIELDS.items()},
        "subject_text": {k: v for k, v in SUBJECT_TEXT.items()},
        "subject_numeric": {k: {"min": a, "max": b} for k, (a, b) in SUBJECT_NUM.items()},
        "geometry_numeric": {k: {"min": a, "max": b} for k, (a, b) in GEOMETRY_NUM.items()},
        "json_blobs": ["ar", "device_info"],
        "identifiers": ["subject", "source", "store", "app_version"],
        "crease_convention": CREASE_CONVENTION,
        "model_checkpoint": CKPT_INFO.get("checkpoint"),
    })


@app.get("/f/<session>/<path:name>")
def frame_file(session, name):
    return send_from_directory(SESSIONS / session, name)


@app.post("/api/session")
def api_session():
    ok_space, space_msg = space_check()
    if not ok_space:
        return jsonify({"error": space_msg}), 507
    uploaded = request.files.getlist("frames")
    if not uploaded:
        return jsonify({"error": "nothing uploaded"}), 400

    # Tags are applied from the results modal (see /api/session/<sid>/tags),
    # not here -- asking before the recording made people tag and then lose it
    # to a failed capture.
    # In study mode every tag is chosen BEFORE recording and posted with the
    # frames, because the upload is backgrounded and there is no results modal
    # to tag afterwards. The dev flow still tags from the modal, so all of these
    # stay optional here.
    tags = {}
    for key, allowed in TAG_FIELDS.items():
        v = (request.form.get(key) or "").strip().lower()
        if not v:
            tags[key] = None          # simply not supplied
            continue
        if v not in allowed:
            # Refused rather than dropped. A silently blanked field is
            # indistinguishable from one that was never sent, so a typo in an
            # operator's form would erase a covariate without anyone noticing.
            return jsonify({"error": f"{key}: must be one of "
                                     f"{'/'.join(allowed)}"}), 400
        tags[key] = v
    sex = tags.get("sex")
    age = None
    _a = (request.form.get("age") or "").strip()
    if _a:
        try:
            _av = float(_a)
            age = _av if 0 < _av < 25 else None
        except ValueError:
            age = None
    extra, extra_err = _parse_extra(request.form)
    if extra_err:
        return jsonify({"error": extra_err}), 400
    # Verbatim blobs: ARKit depth/pose per frame, and whatever the device knows
    # about itself. Both are evidence about the measurement, not about the child.
    ar = _parse_json_blob(request.form, "ar")
    device_info = _parse_json_blob(request.form, "device_info")

    subject = (request.form.get("subject") or "").strip()[:32]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", subject or ""):
        subject = None

    # What the camera said it was delivering, before any server-side resize.
    # Free text from the client, so it is pattern-matched rather than trusted.
    cres = (request.form.get("capture_res") or "").strip()[:16]
    if not re.fullmatch(r"\d{1,5}x\d{1,5}", cres or ""):
        cres = None

    # The iOS LiDAR app posts its per-frame 3D detail alongside the frames:
    # the 2D landmark, the depth sampled there, and the unprojected 3D point.
    # Stored verbatim so the jitter can be decomposed -- Vision moving the
    # landmark and LiDAR depth noise under a still landmark look identical in
    # the ratio but need different fixes.
    source = (request.form.get("source") or "web").strip()[:32]
    if not re.fullmatch(r"[a-z0-9_-]{1,32}", source):
        source = "web"
    ios_raw = request.form.get("ios_frames")
    ios_frames = None
    if ios_raw:
        try:
            d = json.loads(ios_raw)
            if isinstance(d, list) and len(d) <= 500:
                ios_frames = d
        except Exception as e:
            print(f"bad ios_frames payload: {e}", flush=True)

    store = (request.form.get("store") or "").strip().lower()
    store = "study" if store == "study" else "dev"
    sid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
    out = _store_dir(store) / sid
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    if uploaded:
        # Preferred path: the browser grabbed frames off a canvas, so there is
        # no container to decode and nothing browser-specific to go wrong.
        cand = []
        raw_dir = out / "full"
        raw_dir.mkdir(exist_ok=True)
        for fi, f in enumerate(uploaded):
            data = f.read()
            # Exactly what the phone sent, before any resize. The processed
            # frame is derived and reproducible; this is not.
            (raw_dir / f"f{fi:04d}_full.jpg").write_bytes(data)
            buf = np.frombuffer(data, np.uint8)
            im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if im is None:
                continue
            h, w = im.shape[:2]
            if max(h, w) > FRAME_MAX_PX:
                sc_ = FRAME_MAX_PX / max(h, w)
                im = cv2.resize(im, (int(w * sc_), int(h * sc_)), interpolation=cv2.INTER_AREA)
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            cand.append((im, float(cv2.Laplacian(g, cv2.CV_64F).var())))
        n_in = len(cand)
        if KEEP_ALL_FRAMES:
            marked, blur_thr = mark_blurry(cand)
            frames = [(im, sh) for im, sh, _ in marked]
            blurry_flags = [b for _, _, b in marked]
        else:
            frames = drop_blurry(cand)
            blurry_flags = [False] * len(frames)
    else:
        tmp = out / ("clip" + (Path(clip.filename or "clip.webm").suffix or ".webm"))
        if not tmp.exists():
            clip.save(str(tmp))
        frames = extract_frames(tmp)
        n_in = len(frames)
        blur_thr = blur_threshold([f[1] for f in frames]) if frames else BLUR_FLOOR
        blurry_flags = [f[1] < blur_thr for f in frames]
        if not frames:
            # keep the clip: a failed decode is only diagnosable with the file
            # Keep everything: the clip stays where it is and the session
            # directory survives, so the failure is inspectable later.
            (out / "session.json").write_text(json.dumps({
                "session": sid, "error": "clip could not be decoded",
                "utc": datetime.now(timezone.utc).isoformat()}, indent=2),
                encoding="utf-8")
            mirror_session(sid)
            return jsonify({"error": "could not read that clip - kept for diagnosis"}), 400

    if not frames:
        (out / "session.json").write_text(json.dumps({
            "session": sid, "error": "no usable frames",
            "utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
        mirror_session(sid)
        return jsonify({"error": "no usable frames - kept for diagnosis"}), 400

    ratios, recs, thumbs = [], [], []
    sharp_ratios = []
    for i, (fr, sharp) in enumerate(frames):
        pts_full = predict(fr)
        pts, crop_box = predict_cropped(fr)
        li = float(np.linalg.norm(pts[1] - pts[0]))
        lr = float(np.linalg.norm(pts[3] - pts[2]))
        name = f"f{i:04d}"
        blurry = bool(blurry_flags[i]) if i < len(blurry_flags) else False
        cv2.imwrite(str(out / f"{name}.jpg"), fr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cv2.imwrite(str(out / f"{name}_over.jpg"), draw(fr, pts), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        rec = {"name": name, "model_points": pts.tolist(), "points": pts.tolist(),
               "points_uncropped": pts_full.tolist(), "crop_box": crop_box,
               "edited": False, "sharpness": sharp, "blurry": blurry,
               "index_len_px": li, "ring_len_px": lr,
               "ratio": (li / lr) if lr > 1e-6 else None}
        (out / f"{name}.json").write_text(json.dumps(rec), encoding="utf-8")
        recs.append(rec)
        if rec["ratio"]:
            ratios.append(rec["ratio"])
            if not blurry:
                sharp_ratios.append(rec["ratio"])
        if len(thumbs) < 9:
            thumbs.append(f"/f/{sid}/{name}_over.jpg")

    # The raw clip and any transcode are kept. They used to be deleted as junk
    # once frames were extracted; they are the only record of what the camera
    # actually produced, and a decode failure is undiagnosable without them.
    # Every frame is stored. The headline number still comes from the in-focus
    # subset when there are enough of them -- keeping a frame and trusting its
    # measurement are different decisions.
    r_all = np.array(ratios) if ratios else np.array([0.0])
    use_sharp = len(sharp_ratios) >= 3
    r = np.array(sharp_ratios) if use_sharp else r_all
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    bg = background_check(frames[0][0])

    # Ratio of mean lengths, not mean of ratios. Frames of one hand differ
    # mostly by apparent scale s, and index_i = s_i*I, ring_i = s_i*R, so
    # mean(index)/mean(ring) = I/R exactly while the mean of per-frame ratios
    # does not have that property. It is also implicitly weighted toward the
    # frames where the hand is largest, which are the ones with the most pixels
    # on the fingers. Measured +0.0055 against median-of-ratios on real clips.
    sel = [rec for rec in recs if rec.get("ratio")
           and (not use_sharp or not rec.get("blurry"))]
    if sel:
        mi = float(np.mean([rec["index_len_px"] for rec in sel]))
        mr = float(np.mean([rec["ring_len_px"] for rec in sel]))
        ratio_of_means = (mi / mr) if mr > 1e-6 else None
    else:
        ratio_of_means = None

    # Frames of ONE hand should agree closely; the repeatability study put the
    # model's own spread at 0.016. A clip well above that is not a noisy
    # measurement, it is the model failing to find the hand -- which is what a
    # cluttered background does to it. Refuse to show a number in that case,
    # because a plausible-looking median next to SD 0.45 is worse than no
    # answer: the reader has no way to tell it is meaningless.
    reasons = []
    level = "ok"
    if len(ratios) < 3:
        level = "fail"; reasons.append("hand found in too few frames")
    if sd > 0.08:
        level = "fail"; reasons.append(f"readings disagree wildly (SD {sd:.3f})")
    elif sd > 0.04:
        level = "poor" if level == "ok" else level
        reasons.append(f"readings vary a lot (SD {sd:.3f})")
    if bg.get("verdict") in ("bad", "busy"):
        if level == "ok":
            level = "poor"
        reasons.append(bg.get("note") or "cluttered background")
    quality = {"level": level, "reasons": reasons}

    meta = {"session": sid,
            # every declared tag, so a new one in TAG_FIELDS is stored without
            # a second edit here -- dominant_hand and nail_overhang were both
            # silently dropped when this was a hand-written list
            **{k: tags.get(k) for k in TAG_FIELDS},
            "subject": subject, "age": age, "store": store,
            **{k: v for k, v in extra.items() if k != "age"},
            "ar": ar, "device_info": device_info,
            # Which measurement rules produced this record. If the crease
            # convention or the model changes, records taken under the old one
            # stay identifiable instead of quietly mixing.
            "crease_convention": CREASE_CONVENTION,
            "model_checkpoint": CKPT_INFO.get("checkpoint"),
            "capture_res": cres, "frame_max_px": FRAME_MAX_PX,
            "source": source,
            "n_ios_frames": len(ios_frames) if ios_frames else 0,
            "quality": quality,
            "n_frames": len(recs), "n_valid": len(ratios),
            "n_blurry": int(sum(bool(b) for b in blurry_flags)),
            "n_dropped_blurry": 0,
            "blur_threshold": round(float(blur_thr), 1),
            "ratio_from": "sharp" if use_sharp else "all",
            "median_ratio_all": float(np.median(r_all)),
            "n_sharp": len(sharp_ratios),
            "background": bg,
            "median_ratio": float(ratio_of_means) if ratio_of_means
                            else float(np.median(r)),
            "ratio_of_means": ratio_of_means,
            "median_of_ratios": float(np.median(r)),
            "mean_index_px": float(np.mean([x["index_len_px"] for x in sel])) if sel else None,
            "mean_ring_px": float(np.mean([x["ring_len_px"] for x in sel])) if sel else None,
            "sd_ratio": sd,
            "seconds": time.perf_counter() - t0,
            "utc": datetime.now(timezone.utc).isoformat(), **CKPT_INFO}
    if ios_frames is not None:
        (out / "ios_frames.json").write_text(json.dumps(ios_frames), encoding="utf-8")
    (out / "session.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    meta["backup"] = mirror_session(sid)
    if space_msg:
        meta.setdefault("warnings", []).append(space_msg)
    meta["thumbs"] = thumbs
    return jsonify(meta)


# Whitelisted tag vocabularies. Al-Zaid 2015 measures the RIGHT hand only, so
# which hand a clip shows is a covariate, not a detail -- a left-hand reading
# cannot be compared against her 0.91/0.96 figures.
# Everything below is recorded AT CAPTURE because none of it can be recovered
# afterwards. As a measurement instrument rather than a screening tool, the
# conditions a reading was taken under are not overhead -- they are the part that
# lets anyone else reproduce or discount the number.
#
# Grouped by what they explain:
#   subject     -- who; the covariates 2D:4D is known to vary with
#   geometry    -- how the hand sat relative to the lens; the projection error
#   device      -- what took the picture and what measured it
#   protocol    -- which rules were in force, so a later change is detectable
#
# Population is captured because it explains MORE variance in 2D:4D than sex
# does (partial eta-squared 0.073 vs 0.056), and age because the ratio rises
# through childhood -- both make a single fixed reference range indefensible.

# Free-text-ish subject fields: validated shape, no fixed vocabulary, because
# imposing one on nationality or ethnicity across a Saudi cohort would lose more
# than it standardises.
SUBJECT_TEXT = {
    "nationality": r"[A-Za-z \-]{1,40}",
    "ethnicity": r"[A-Za-z \-]{1,40}",
    "operator": r"[A-Za-z0-9_\-]{1,32}",
    "protocol": r"[A-Za-z0-9_\-./]{1,48}",     # IRB / consent reference
    "notes": r".{0,240}",
}

# Numeric fields with a plausible range. Out-of-range is rejected rather than
# clamped: a silently clamped value is indistinguishable from a real one.
SUBJECT_NUM = {
    "age": (0.0, 25.0),
    "height_cm": (30.0, 220.0),
    "weight_kg": (2.0, 200.0),
}

# Per-capture geometry, mostly from ARKit. Tilt is the one that matters most:
# a photograph measures PROJECTED finger length, so a hand tilted away from the
# lens reads short, and the error does not cancel between index and ring.
GEOMETRY_NUM = {
    "distance_mm": (50.0, 2000.0),      # lens to palm
    "tilt_deg": (0.0, 90.0),            # palm plane vs image plane
    "roll_deg": (-180.0, 180.0),
    "hand_fraction": (0.0, 1.0),        # of the long side; target ~0.255
    "depth_confidence": (0.0, 1.0),
}


TAG_FIELDS = {
    "sex": ("male", "female"),
    "autistic": ("yes", "no"),
    "hand": ("right", "left"),
    # Marking a recording as study data MOVES it to the study store. Kept as a
    # tag rather than a separate capture page so the operator has one flow and
    # one set of buttons; the separation happens on the server.
    "study": ("yes", "no"),
    # Which hand the subject WRITES with -- not which hand was photographed.
    # 2D:4D asymmetry is reported to differ by handedness, and the two are
    # routinely conflated.
    "dominant_hand": ("right", "left", "ambidextrous"),
    # Whether the nail projected past the flesh, judged at capture. Detected
    # server-side too, but the operator can see the hand and the server cannot.
    "nail_overhang": ("yes", "no"),
}


@app.post("/api/session/<sid>/tags")
def api_session_tags(sid: str):
    """Attach demographic tags to an existing session.

    2D:4D is sex-dimorphic, and the autism association is the reason this
    dataset is worth collecting at all, so both are the covariates every
    reading is interpreted against. Values are whitelisted rather than stored
    as sent -- they land in session.json and render in the admin list.
    """
    # sid comes from the URL and is used to build a path, so it is matched
    # against the directory listing rather than trusted or sanitised.
    # A session may already have been moved to the study store by an earlier
    # tag call, so both roots are searched.
    d = None
    for root in (SESSIONS, STUDY_SESSIONS):
        cand = root / sid
        if cand.is_dir() and cand.parent == root:
            d = cand
            break
    if d is None:
        return jsonify({"error": "unknown session"}), 404
    sj = d / "session.json"
    if not sj.exists():
        return jsonify({"error": "session has no metadata"}), 404

    body = request.get_json(silent=True) or {}
    meta = json.loads(sj.read_text(encoding="utf-8"))
    changed = []

    # The subject id is what links a child's right hand to their left, and one
    # visit to the next. It is free text rather than an enum, so it is validated
    # separately from TAG_FIELDS -- and deliberately restricted, since it ends up
    # in filenames and listings.
    # Al-Zaid's cohort is boys aged 3-8 and 2D:4D changes with growth, so age is
    # needed to check a cohort matches hers and to control for it. It cannot be
    # recovered once the child has gone, which is why it is captured rather than
    # inferred later.
    if "age" in body:
        v = str(body.get("age") or "").strip()[:5]
        if v:
            try:
                a_ = float(v)
                if not (0 < a_ < 25):
                    return jsonify({"error": "age must be between 0 and 25"}), 400
            except ValueError:
                return jsonify({"error": "age must be a number"}), 400
            meta["age"] = a_
        else:
            meta["age"] = None
        changed.append("age")

    for key in list(SUBJECT_TEXT) + list(SUBJECT_NUM) + list(GEOMETRY_NUM):
        if key == "age" or key not in body:
            continue
        v = body.get(key)
        if v in (None, ""):
            meta.pop(key, None); changed.append(key); continue
        if key in SUBJECT_TEXT:
            sv = str(v).strip()
            if not re.fullmatch(SUBJECT_TEXT[key], sv, re.S):
                return jsonify({"error": f"{key}: not in the accepted format"}), 400
            meta[key] = sv
        else:
            lo, hi = (SUBJECT_NUM | GEOMETRY_NUM)[key]
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return jsonify({"error": f"{key}: must be a number"}), 400
            if not (lo <= fv <= hi):
                return jsonify({"error": f"{key}: must be between {lo} and {hi}"}), 400
            meta[key] = fv
        changed.append(key)

    if "subject" in body:
        v = (body.get("subject") or "").strip()[:32]
        if v and not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", v):
            return jsonify({"error": "subject: letters, digits, - and _ only"}), 400
        meta["subject"] = v or None
        changed.append("subject")
    for key, allowed in TAG_FIELDS.items():
        if key not in body:
            continue
        v = body.get(key)
        v = (v or "").strip().lower() if isinstance(v, str) else None
        if v not in allowed:
            return jsonify({"error": f"{key} must be one of {'/'.join(allowed)}"}), 400
        meta[key] = v
        changed.append(key)
    if not changed:
        return jsonify({"error": "nothing to set"}), 400

    tmp = sj.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp.replace(sj)

    # Marking it study data moves the whole session directory into the study
    # store, so child recordings never share a directory with test ones. The
    # move happens after the metadata write, so a failure here leaves a
    # correctly-tagged session in the old place rather than an untagged one in
    # the new place.
    if "study" in changed:
        want = STUDY_SESSIONS if meta.get("study") == "yes" else SESSIONS
        if d.parent != want:
            want.mkdir(parents=True, exist_ok=True)
            dest = want / sid
            if not dest.exists():
                try:
                    shutil.move(str(d), str(dest))
                    d = dest
                except Exception as e:
                    print(f"study move failed for {sid}: {e}", flush=True)
    return jsonify({"ok": True, "store": ("study" if d.parent == STUDY_SESSIONS
                                          else "dev"),
                    "subject": meta.get("subject"), "age": meta.get("age"),
                    **{k: meta.get(k) for k in TAG_FIELDS}})


# ---------------------------------------------------------------- review ----
THUMBS = CAPTURES / "_thumbs"
_PRED_CACHE = {"mtime": None, "map": {}}


PRED_FILE = REPO / "agents" / "disagreement.json"


def _pred_version():
    try:
        return int(PRED_FILE.stat().st_mtime)
    except Exception:
        return 0


def _predictions():
    """id -> model-predicted points, from the disagreement scan.

    Reloaded when the file changes so a re-scan takes effect without a restart.
    A parse failure keeps the PREVIOUS map rather than blanking it: the scan
    rewrites this file in place, and a read landing mid-write was briefly
    surfacing 'no prediction' in the editor for every image.
    """
    if not PRED_FILE.exists():
        return _PRED_CACHE["map"]
    mt = PRED_FILE.stat().st_mtime
    if _PRED_CACHE["mtime"] != mt:
        try:
            rows = json.loads(PRED_FILE.read_text(encoding="utf-8"))
            got = {r["id"]: r["pred"] for r in rows if r.get("pred")}
            if got:
                _PRED_CACHE["map"] = got
                _PRED_CACHE["mtime"] = mt
        except Exception:
            pass                      # keep serving the last good map
    return _PRED_CACHE["map"]
# Stamped into every capture. The literature settles the DIRECTION -- six
# independent papers state "where there was a band of creases at the base of the
# digit, the most proximal crease was used", one of them with Manning as a
# co-author -- but not the operational detail, so the rest is ours and is
# versioned rather than assumed.
CREASE_CONVENTION = "most-proximal-v1"

POINT_ORDER = ("index_base", "index_tip", "ring_base", "ring_tip")


def _seg_cross(a, b, c, d) -> bool:
    """Do segments ab and cd properly intersect?"""
    def o(p, q, r):
        v = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        return 0 if abs(v) < 1e-9 else (1 if v > 0 else 2)
    o1, o2, o3, o4 = o(a, b, c), o(a, b, d), o(c, d, a), o(c, d, b)
    return o1 != o2 and o3 != o4


def _anomalies(q, w=None, h=None):
    """Raw geometric measures for one label, plus any unambiguous fault.

    Deliberately returns measurements rather than a verdict. The corpus turned
    out to have no anatomically impossible labels at all -- axis angle tops out
    at 52 degrees, length asymmetry at 0.18, and the widest-splayed hand is a
    correctly labelled one. A fixed threshold therefore only flags the top of a
    normal distribution, so the caller z-scores these against the live corpus
    and ranks by how unusual a label is rather than by whether it trips a line
    someone guessed.

    `hard` holds the faults that are wrong at any scale: crossed segments, a
    point off the image, a zero-length finger. Those never need a distribution.
    """
    P = np.asarray(q, dtype=float)
    if P.shape != (4, 2) or not np.isfinite(P).all():
        return {"hard": ["malformed"]}
    ib, it_, rb, rt = P
    vi, vr = it_ - ib, rt - rb
    Li, Lr = float(np.linalg.norm(vi)), float(np.linalg.norm(vr))
    if Li < 1e-6 or Lr < 1e-6:
        return {"hard": ["zero-length finger"]}
    m = (Li + Lr) / 2.0
    hard = []
    if _seg_cross(ib, it_, rb, rt):
        hard.append("segments cross")
    if w and h:
        e = 2.0
        if (P[:, 0] < -e).any() or (P[:, 0] > w + e).any() or \
           (P[:, 1] < -e).any() or (P[:, 1] > h + e).any():
            hard.append("off-image")
    return {
        "ang": float(np.degrees(np.arccos(
            np.clip(float(vi @ vr) / (Li * Lr), -1.0, 1.0)))),
        "asym": abs(Li - Lr) / m,
        "bsep": float(np.linalg.norm(rb - ib)) / m,
        "tsep": float(np.linalg.norm(rt - it_)) / m,
        "hard": hard,
    }


# Human-readable name for each measure, used in the tile caption.
_GEO_LABEL = {"ang": "splay", "asym": "length gap",
              "bsep": "base spacing", "tsep": "tip spacing"}


def _score_anomalies(items):
    """Rank labels by how far their geometry sits from the corpus norm.

    Median and MAD rather than mean and SD: the outliers being looked for would
    otherwise inflate the very spread they are measured against. 1.4826 rescales
    MAD to be comparable with a standard deviation on normal data, so the
    numbers read like the sigmas used elsewhere on this page.
    """
    keys = ("ang", "asym", "bsep", "tsep")
    stats = {}
    for k in keys:
        v = np.array([x["geo"][k] for x in items
                      if x.get("geo") and k in x["geo"]], dtype=float)
        if len(v) < 8:
            continue
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * 1.4826
        stats[k] = (med, mad if mad > 1e-9 else 1.0)
    for x in items:
        g = x.get("geo") or {}
        hard = g.get("hard") or []
        score, why = 10.0 * len(hard), list(hard)
        best = None
        for k, (med, mad) in stats.items():
            if k not in g:
                continue
            z = abs(g[k] - med) / mad
            if best is None or z > best[1]:
                best = (k, z)
        if best and best[1] >= 2.5:
            score += best[1]
            why.append(f"{_GEO_LABEL[best[0]]} {best[1]:.1f}\u03c3")
        x["anom"] = score
        x["anom_why"] = why


def _review_index():
    """Every image we have points for, from both sources.

    annotations/  -- the 3,390 hand-placed labels the model trains on. These
                     are the ones worth eyeballing: a wrong label is invisible
                     in any metric that is itself computed from the labels.
    sessions/     -- captured frames, showing the model's own predictions
                     (or the hand-corrected points once edited).
    """
    items = []
    for j in sorted((REPO / "annotations").glob("*.json")):
        try:
            a = json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            continue
        if a.get("status") != "accepted":
            continue
        pts = a.get("points") or {}
        if not all(k in pts for k in POINT_ORDER):
            continue
        q = [[pts[k]["x"], pts[k]["y"]] for k in POINT_ORDER]
        li = float(np.hypot(q[1][0] - q[0][0], q[1][1] - q[0][1]))
        lr = float(np.hypot(q[3][0] - q[2][0], q[3][1] - q[2][1]))
        geo = _anomalies(q, a.get("width"), a.get("height"))
        items.append({
            "id": "ann/" + j.stem, "src": "annotations",
            "ver": int(j.stat().st_mtime),
            "dataset": a.get("dataset") or "?",
            "ratio": (li / lr) if lr > 1e-6 else None,
            "time": a.get("timestamp_utc") or "",
            "geo": geo, "pts": q, "excluded": bool(a.get("excluded")),
        })
    for d in sorted(SESSIONS.glob("*")):
        if not d.is_dir():
            continue
        for j in sorted(d.glob("f*.json")):
            try:
                r = json.loads(j.read_text(encoding="utf-8"))
            except Exception:
                continue
            cq = r.get("points")
            if isinstance(cq, dict):
                cq = ([[cq[k]["x"], cq[k]["y"]] for k in POINT_ORDER]
                      if all(k in cq for k in POINT_ORDER) else None)
            cgeo = _anomalies(cq) if cq else None
            items.append({
                "id": f"cap/{d.name}/{j.stem}", "src": "captures",
                "ver": int(j.stat().st_mtime),
                "dataset": d.name,
                "ratio": r.get("ratio"),
                "edited": bool(r.get("edited")),
                "blurry": bool(r.get("blurry")),
                "geo": cgeo, "pts": cq,
                "time": "",
            })
    return items


def _review_source(ident: str):
    """The file that changes when an item is edited -- its mtime versions the
    thumbnail url so a save busts the browser cache as well as ours."""
    parts = ident.split("/")
    if parts[0] == "ann" and len(parts) == 2:
        j = REPO / "annotations" / (parts[1] + ".json")
        return j if j.is_file() and j.parent == REPO / "annotations" else None
    if parts[0] == "cap" and len(parts) == 3:
        d = SESSIONS / parts[1]
        j = d / (parts[2] + ".json")
        return j if d.is_dir() and d.parent == SESSIONS and j.is_file() else None
    return None


def _review_locate(ident: str):
    """Resolve a review id to (image_path, points). Ids come from the URL, so
    each part is matched against the on-disk listing rather than trusted."""
    parts = ident.split("/")
    if parts[0] == "ann" and len(parts) == 2:
        j = REPO / "annotations" / (parts[1] + ".json")
        if not j.is_file() or j.parent != REPO / "annotations":
            return None, None
        a = json.loads(j.read_text(encoding="utf-8"))
        img = REPO / a.get("filtered_path", "")
        pts = a.get("points") or {}
        if not img.is_file() or not all(k in pts for k in POINT_ORDER):
            return None, None
        # float64: these round-trip through the editor and back into the label
        # file, and float32 quantises the stored coordinate on every save.
        return img, np.array([[pts[k]["x"], pts[k]["y"]] for k in POINT_ORDER], np.float64)
    if parts[0] == "cap" and len(parts) == 3:
        d = SESSIONS / parts[1]
        j = d / (parts[2] + ".json")
        if not d.is_dir() or d.parent != SESSIONS or not j.is_file():
            return None, None
        r = json.loads(j.read_text(encoding="utf-8"))
        img = d / (parts[2] + ".jpg")
        if not img.is_file():
            return None, None
        return img, np.array(r.get("points") or [], np.float32)
    return None, None


@app.get("/api/review/list")
def api_review_list():
    src = request.args.get("src", "annotations")
    sort = request.args.get("sort", "seq")
    page = max(0, int(request.args.get("page", 0) or 0))
    per = min(200, max(12, int(request.args.get("per", 60) or 60)))

    items = [x for x in _review_index() if src == "all" or x["src"] == src]
    # Sorts demote unscorable rows rather than dropping them, so the tail of
    # every ranked queue is N/A with nothing on screen to mark where the real
    # hits stopped. These two counts draw that line: how many rows could be
    # scored at all, and how many cleared the threshold worth looking at.
    n_scored = n_strong = None
    # Which field the active sort ranks on, so one filter step can serve all of
    # them. Set inside each branch; left None for "seq", which has no score.
    score_key = None
    try:
        minv = float(request.args["min"]) if request.args.get("min") else None
    except (TypeError, ValueError):
        minv = None
    rs = [x["ratio"] for x in items if x.get("ratio")]
    med = float(np.median(rs)) if rs else 1.0
    # Real 2D:4D spread in this dataset is SD 0.046, so a fixed 0.05 cutoff sits
    # at ~1.1 SD and flags a quarter of all hands -- normal variation, not
    # errors. Three SD flags ~0.7%, which is the right order for "look at this
    # one". Sent to the client so the threshold follows the data rather than
    # being hard-coded in two places.
    sd = float(np.std(rs, ddof=1)) if len(rs) > 2 else 0.05
    for x in items:
        x["dev"] = abs(x["ratio"] - med) if x.get("ratio") else 0.0
    if sort in ("disagree", "worstpt"):
        # Ranked by how far the model's own prediction sits from the human
        # label. Ratio outliers only catch labels that look wrong in isolation;
        # a trained model disagrees wherever a label contradicts what it learned
        # from the other 3,389, which also catches errors at an ordinary ratio.
        dis = {}
        f = REPO / "agents" / "disagreement.json"
        if f.exists():
            try:
                for r in json.loads(f.read_text(encoding="utf-8")):
                    dis[r["id"]] = r
            except Exception:
                dis = {}
        for x in items:
            r = dis.get(x["id"])
            x["disagree"] = float(r["err"]) if r else None
            x["worst_dev"] = None
            if r:
                x["worst_point"] = r.get("worst_point")
                x["model_ratio"] = r.get("model_ratio")
                pp = r.get("per_point") or {}
                if pp:
                    x["worst_dev"] = float(max(pp.values()))
        # "disagree" sorts on the MEAN of the four point distances; "worstpt"
        # on the largest single one. They find different faults: one point badly
        # misplaced is divided by four in the mean and lands mid-queue, scoring
        # the same as four points each drifting slightly -- which is a normal
        # label, not a mistake.
        key = "worst_dev" if sort == "worstpt" else "disagree"
        score_key = key
        n_scored = sum(1 for x in items if x.get(key) is not None)
        # worst_dev is a MAX over four points and disagree is their mean,
        # so they do not share a scale: 0.02 shortlists 104 rows on the
        # mean and 1544 on the max.
        cut = 0.05 if sort == "worstpt" else 0.02
        n_strong = sum(1 for x in items if (x.get(key) or 0) > cut)
        items.sort(key=lambda x: -(x[key] if x.get(key) is not None else -1))
    elif sort == "nails":
        # Measured nail overhang: the gap between the human label (which marks
        # the flesh) and where the finger silhouette actually ends. No model in
        # the loop, so model error is not a confound.
        #
        # Two things had to be excluded to make this trustworthy. Stopping at
        # "close to the background colour" walks straight through the shadow
        # under a hand and reports ~50px of nail where there is none, so the
        # test is a drop in saturation relative to this finger's own flesh --
        # paper and shadow are both desaturated, flesh and nail are not. And a
        # finger pressed against a scanner fades out over tens of pixels with no
        # nail at all, so a gradual transition is rejected rather than measured.
        nl = {}
        f = REPO / "agents" / "nails.json"
        if f.exists():
            try:
                for r in json.loads(f.read_text(encoding="utf-8")):
                    nl["ann/" + r["id"]] = r
            except Exception:
                nl = {}
        for x in items:
            r = nl.get(x["id"])
            x["nail"] = float(r["max"]) if r else None
            x["nail_why"] = ([f"index {r['index']:.0f} / ring {r['ring']:.0f}px"]
                             if r else [])
            if r:
                x["nail_dratio"] = float(r["dratio"])
        score_key = "nail"
        n_scored = sum(1 for x in items if x.get("nail") is not None)
        n_strong = sum(1 for x in items if (x.get("nail") or 0) > 8)
        items.sort(key=lambda x: -(x["nail"] if x.get("nail") is not None
                                   else -1e9))
    elif sort == "tipover":
        # How far past the human tip the model pushes, measured ALONG the finger
        # and signed. This is the nail signature: annotators mark the flesh, the
        # model follows the silhouette, and a nail that overhangs the flesh sits
        # between the two. Unsigned distance (the "disagree" sorts) mixes those
        # cases in with every other kind of error and buries them.
        dis = {}
        f = REPO / "agents" / "disagreement.json"
        if f.exists():
            try:
                for r in json.loads(f.read_text(encoding="utf-8")):
                    dis[r["id"]] = r
            except Exception:
                dis = {}
        for x in items:
            x["tipover"] = None
            x["tipover_why"] = []
            r = dis.get(x["id"])
            q = x.get("pts")
            if not r or not q or not r.get("pred"):
                continue
            L = np.asarray(q, dtype=float)
            M = np.asarray(r["pred"], dtype=float)
            if L.shape != (4, 2) or M.shape != (4, 2):
                continue
            over = {}
            for b, t, nm in ((0, 1, "index"), (2, 3, "ring")):
                ax = L[t] - L[b]
                n = float(np.linalg.norm(ax))
                if n < 1e-6:
                    break
                over[nm] = float((M[t] - L[t]) @ (ax / n))
            if len(over) != 2:
                continue
            x["tipover"] = max(over.values())
            x["tipover_why"] = [f"index {over['index']:+.1f}",
                                f"ring {over['ring']:+.1f}"]
            if r.get("model_ratio") and r.get("human_ratio"):
                x["tipover_dratio"] = float(r["model_ratio"] - r["human_ratio"])
        score_key = "tipover"
        n_scored = sum(1 for x in items if x.get("tipover") is not None)
        n_strong = sum(1 for x in items if (x.get("tipover") or 0) > 5)
        items.sort(key=lambda x: -(x["tipover"] if x.get("tipover") is not None
                                   else -1e9))
    elif sort == "anomaly":
        # Geometry only, scored against the corpus it belongs to. Distinct from
        # "odd", which keys on distance from the median RATIO: a label can be
        # geometrically strange and still divide to an ordinary ratio, so it
        # never surfaces there. Measured overlap between the two queues is zero.
        _score_anomalies(items)
        score_key = "anom"
        n_scored = sum(1 for x in items if x.get("geo"))
        # 2.5 sigma is the bar for appearing in the ranking at all; 4 is
        # the bar for being worth opening. At 2.5 this counted 443 rows,
        # an eighth of the corpus, which is a distribution tail rather
        # than a shortlist.
        n_strong = sum(1 for x in items if (x.get("anom") or 0) >= 4.0)
        items.sort(key=lambda x: -(x.get("anom") or 0.0))
    elif sort == "odd":
        # Largest deviation from the population median first. A mislabelled
        # point almost always shows up as an implausible ratio, so this puts
        # the likely mistakes on page one instead of making you scan 3,390.
        score_key = "dev"
        n_scored = sum(1 for x in items if x.get("ratio"))
        n_strong = sum(1 for x in items if x.get("ratio") and x["dev"] > 3 * sd)
        items.sort(key=lambda x: -x["dev"])
    # Hide everything below the threshold rather than ranking it to the tail.
    # Rows with no score at all are dropped too: an unmeasurable image is not
    # the same as a clean one, but it certainly is not above the line.
    if minv is not None and score_key:
        items = [x for x in items
                 if x.get(score_key) is not None and x[score_key] >= minv]

    total = len(items)
    # geo holds the raw measures the anomaly ranking is computed from; the
    # browser only needs the score and the reason, so drop it from the payload.
    for x in items[page * per:(page + 1) * per]:
        x.pop("geo", None)
        x.pop("pts", None)
    return jsonify({"total": total, "page": page, "per": per, "median": med,
                    "sd": sd, "flag_at": 3 * sd,
                    "scored": n_scored, "strong": n_strong,
                    "min": minv, "score_key": score_key,
                    "items": items[page * per:(page + 1) * per]})


@app.get("/api/review/thumb/<path:ident>")
def api_review_thumb(ident: str):
    w = min(1600, max(160, int(request.args.get("w", 320) or 320)))
    ghost_on = request.args.get("ghost", "1") != "0"
    THUMBS.mkdir(parents=True, exist_ok=True)
    # The label file's mtime is part of the key, so an edit produces a NEW
    # cache entry and a NEW url -- clearing the server cache alone was not
    # enough, because the browser kept serving its own copy of an unchanged url.
    src = _review_source(ident)
    ver = int(src.stat().st_mtime) if src and src.exists() else 0
    # The ghost rings come from the scan file, so re-running the scan with a
    # different model must invalidate thumbnails too. Without this the grid kept
    # drawing the previous model's predictions while the editor drew the new
    # ones -- the same image showing two different answers.
    pv = _pred_version()
    key = hashlib.sha1(f"{ident}|{w}|{ghost_on}|{ver}|{pv}".encode()).hexdigest()
    cached = THUMBS / f"{key}.jpg"
    if cached.exists():
        return send_file(cached, mimetype="image/jpeg")
    img, pts = _review_locate(ident)
    if img is None or pts is None or pts.shape != (4, 2):
        return jsonify({"error": "not found"}), 404
    bgr = cv2.imread(str(img))
    if bgr is None:
        return jsonify({"error": "unreadable"}), 404
    gh = _predictions().get(ident) if ghost_on else None
    vis = draw(bgr, pts, ghost=gh)
    h, wd = vis.shape[:2]
    sc = w / max(1, wd)
    vis = cv2.resize(vis, (int(wd * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(cached), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return send_file(cached, mimetype="image/jpeg")


def _rescore(ident: str, pts):
    """Update this image's disagreement score after an edit.

    The model's prediction depends on the image, not the label, so a correction
    cannot change it -- only the distance to it. That makes rescoring pure
    arithmetic with no inference, and it means an image you just fixed drops out
    of "model disagrees most" immediately instead of staying pinned at the top
    with a stale score.

    Written atomically. Rewriting this file in place is what previously let a
    reader catch it half-written.
    """
    pred = _predictions().get(ident)
    if not pred or not PRED_FILE.exists():
        return
    try:
        a = np.asarray(pts, float)
        b = np.asarray(pred, float)
        hs = (float(np.linalg.norm(a[1] - a[0])) +
              float(np.linalg.norm(a[3] - a[2]))) / 2
        if hs < 1e-6:
            return
        d = np.linalg.norm(b - a, axis=1) / hs
        rows = json.loads(PRED_FILE.read_text(encoding="utf-8"))
        names = list(POINT_ORDER)
        hit = False
        for r in rows:
            if r.get("id") == ident:
                r["err"] = float(d.mean())
                r["per_point"] = {names[k]: float(d[k]) for k in range(4)}
                r["worst_point"] = names[int(np.argmax(d))]
                ai = np.linalg.norm(a[1] - a[0]); ar = np.linalg.norm(a[3] - a[2])
                r["human_ratio"] = float(ai / ar) if ar > 1e-6 else None
                hit = True
                break
        if not hit:
            return
        tmp = PRED_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows), encoding="utf-8")
        tmp.replace(PRED_FILE)
        _PRED_CACHE["mtime"] = None          # force a reload on next read
    except Exception as e:
        print(f"rescore failed for {ident}: {e}", flush=True)


@app.get("/api/review/image/<path:ident>")
def api_review_image(ident: str):
    """The source image with no overlay -- the editor draws its own points."""
    img, _ = _review_locate(ident)
    if img is None:
        return jsonify({"error": "not found"}), 404
    return send_file(img)


@app.get("/api/review/item/<path:ident>")
def api_review_item(ident: str):
    img, pts = _review_locate(ident)
    if img is None or pts is None:
        return jsonify({"error": "not found"}), 404
    bgr = cv2.imread(str(img))
    if bgr is None:
        return jsonify({"error": "unreadable"}), 404
    h, w = bgr.shape[:2]
    extra = {}
    if ident.startswith("ann/"):
        j = REPO / "annotations" / (ident.split("/", 1)[1] + ".json")
        if j.is_file():
            a = json.loads(j.read_text(encoding="utf-8"))
            extra = {"dataset": a.get("dataset"),
                     "preplace": a.get("preplace_source"),
                     "timestamp": a.get("timestamp_utc"),
                     "reviewed": bool(a.get("review_edited"))}
    # Prefer the batch scan, but fall back to predicting this one image live.
    # A single forward pass is ~60ms and the model is already resident, so the
    # editor should never be blocked on whether a background job has run.
    pred = _predictions().get(ident)
    pred_src = "scan"
    if pred is None:
        try:
            pred = predict(bgr).tolist()
            pred_src = "live"
        except Exception as e:
            print(f"live predict failed for {ident}: {e}", flush=True)
            pred, pred_src = None, "unavailable"
    if pred is None:
        # Last resort: the GTX 1660 node. Same checkpoint, same answer, and it
        # is idle -- so a busy or wedged local GPU is not a reason to show the
        # reviewer nothing.
        rp = predict_remote(bgr)
        if rp is not None:
            pred, pred_src = rp.tolist(), "remote"
    return jsonify({"id": ident, "width": w, "height": h,
                    "points": pts.tolist(), "pred": pred, "pred_src": pred_src,
                    "names": list(POINT_ORDER), **extra})


@app.post("/api/review/exclude")
def api_review_exclude():
    """Set a label aside from training, or put it back.

    Not a delete. The file, the points and the image all stay exactly where they
    are; a single flag tells the training loader to skip this one. That matters
    because the labels being set aside here are CORRECT -- on a long-nailed hand
    the annotator marked the flesh and the model followed the nail, so the image
    is hard, not wrong. Being able to reverse the decision cheaply is the point.
    """
    body = request.get_json(silent=True) or {}
    ident = str(body.get("id") or "")
    on = bool(body.get("excluded", True))
    reason = str(body.get("reason") or "")[:120]
    if not ident.startswith("ann/"):
        return jsonify({"error": "only annotations can be set aside"}), 400
    j = REPO / "annotations" / (ident.split("/", 1)[1] + ".json")
    if not j.is_file() or j.parent != REPO / "annotations":
        return jsonify({"error": "unknown annotation"}), 404
    a = json.loads(j.read_text(encoding="utf-8"))
    if on:
        a["excluded"] = True
        a["excluded_reason"] = reason
        a["excluded_utc"] = datetime.now(timezone.utc).isoformat()
    else:
        a.pop("excluded", None)
        a.pop("excluded_reason", None)
        a.pop("excluded_utc", None)
    j.write_text(json.dumps(a, indent=2), encoding="utf-8")
    n = sum(1 for f in (REPO / "annotations").glob("*.json")
            if b'"excluded": true' in f.read_bytes())
    return jsonify({"ok": True, "excluded": on, "total_excluded": n})


@app.post("/api/review/save")
def api_review_save():
    """Write corrected points back, keeping the original.

    These are training labels, so the pre-edit values are preserved under
    points_original the FIRST time an image is edited -- re-editing must not
    overwrite the true original with a previous correction. Nothing is deleted
    and the thumbnail cache entry for this image is dropped so the grid shows
    the new position.
    """
    body = request.get_json(silent=True) or {}
    ident = str(body.get("id") or "")
    pts = body.get("points")
    if not isinstance(pts, list) or len(pts) != 4:
        return jsonify({"error": "need 4 points"}), 400
    try:
        arr = [[float(q[0]), float(q[1])] for q in pts]
    except Exception:
        return jsonify({"error": "bad points"}), 400

    img, _ = _review_locate(ident)
    if img is None:
        return jsonify({"error": "unknown id"}), 404

    if ident.startswith("ann/"):
        j = REPO / "annotations" / (ident.split("/", 1)[1] + ".json")
        if not j.is_file() or j.parent != REPO / "annotations":
            return jsonify({"error": "unknown annotation"}), 404
        a = json.loads(j.read_text(encoding="utf-8"))
        if "points_original" not in a:
            a["points_original"] = a.get("points")
        a["points"] = {n: {"x": arr[i][0], "y": arr[i][1]}
                       for i, n in enumerate(POINT_ORDER)}
        a["review_edited"] = True
        a["review_edited_utc"] = datetime.now(timezone.utc).isoformat()
        tmp = j.with_suffix(".tmp")
        tmp.write_text(json.dumps(a, indent=2), encoding="utf-8")
        tmp.replace(j)
    elif ident.startswith("cap/"):
        _, sess, frame = ident.split("/", 2)
        j = SESSIONS / sess / (frame + ".json")
        if not j.is_file():
            return jsonify({"error": "unknown frame"}), 404
        r = json.loads(j.read_text(encoding="utf-8"))
        r.setdefault("points_original", r.get("points"))
        r["points"] = arr
        r["edited"] = True
        r["edited_utc"] = datetime.now(timezone.utc).isoformat()
        li = float(np.hypot(arr[1][0]-arr[0][0], arr[1][1]-arr[0][1]))
        lr = float(np.hypot(arr[3][0]-arr[2][0], arr[3][1]-arr[2][1]))
        r["index_len_px"], r["ring_len_px"] = li, lr
        r["ratio"] = (li / lr) if lr > 1e-6 else None
        j.write_text(json.dumps(r), encoding="utf-8")
        mirror_session(sess)
    else:
        return jsonify({"error": "unknown id"}), 404

    _rescore(ident, arr)

    # drop cached thumbs for this image so the grid reflects the edit
    if THUMBS.exists():
        for wdt in (160, 190, 320, 640, 900, 1400):
            f = THUMBS / (hashlib.sha1(f"{ident}|{wdt}".encode()).hexdigest() + ".jpg")
            f.unlink(missing_ok=True)
    return jsonify({"ok": True, "id": ident})


@app.get("/api/admin/clip/<session>")
def admin_clip(session: str):
    """Stream a session's recorded video. Saved since the retention change but
    never viewable -- the admin page only ever listed frames."""
    d = SESSIONS / session
    if not d.is_dir() or d.parent != SESSIONS:
        return jsonify({"error": "no such session"}), 404
    for c in sorted(d.glob("clip.*")):
        if c.suffix.lower() in (".mp4", ".webm", ".mov", ".m4v"):
            return send_file(c, mimetype="video/mp4" if c.suffix.lower() != ".webm"
                             else "video/webm", conditional=True)
    return jsonify({"error": "no clip for this session"}), 404


@app.get("/api/admin/full/<session>/<name>")
def admin_full(session: str, name: str):
    """The untouched full-resolution frame, as the phone sent it."""
    d = SESSIONS / session / "full"
    if not d.is_dir() or d.parent.parent != SESSIONS:
        return jsonify({"error": "no such session"}), 404
    f = d / (name + "_full.jpg")
    if not f.is_file() or f.parent != d:
        return jsonify({"error": "not found"}), 404
    return send_file(f, mimetype="image/jpeg")


@app.get("/api/coreml/model")
def coreml_model():
    """The exported Core ML model, zipped, for the iOS build to fetch."""
    f = REPO / "models" / "CreaseNet.mlpackage.zip"
    if not f.is_file():
        return jsonify({"error": "not exported yet"}), 404
    return send_file(f, mimetype="application/zip", as_attachment=True,
                     download_name="CreaseNet.mlpackage.zip")


@app.post("/api/coreml/test")
def coreml_test():
    """Verify on-device Core ML against the server's PyTorch model.

    The app posts each frame plus the points its Core ML model produced. The
    server runs the SAME checkpoint in PyTorch on the SAME pixels and reports
    the disagreement. That isolates the one thing an export can silently break:
    float16 on the Neural Engine is a different implementation from float32 on
    CUDA, and a quiet numeric drift would look like model error later.

    multipart form:
      frames      one or more JPEGs
      points      JSON list, one [[x,y] x4] per frame, NORMALISED 0-1,
                  in the model's own order: index_base, index_tip,
                  ring_base, ring_tip
      device      free text, e.g. "iPhone 17 Pro / iOS 26.1"
      ms          JSON list of per-frame inference milliseconds (optional)
    """
    ok_space, space_msg = space_check()
    if not ok_space:
        return jsonify({"error": space_msg}), 507

    uploaded = request.files.getlist("frames")
    if not uploaded:
        return jsonify({"error": "no frames"}), 400
    try:
        device_pts = json.loads(request.form.get("points") or "[]")
    except Exception:
        return jsonify({"error": "points must be JSON"}), 400
    try:
        ms = json.loads(request.form.get("ms") or "[]")
    except Exception:
        ms = []
    device = (request.form.get("device") or "unknown")[:64]

    # Optional 3D payload from the combined app: LiDAR depth sampled at each
    # Core ML keypoint, the unprojected camera-space point, and the tilt. The
    # question this exists to answer is narrow -- does the 3D ratio stay flat
    # through a pose change that makes the 2D ratio swing?
    def _opt(name):
        try:
            v = json.loads(request.form.get(name) or "null")
            return v if isinstance(v, list) else None
        except Exception:
            return None
    depth_m = _opt("depth_m")
    pts3d = _opt("pts3d")
    tilt = _opt("tilt")

    sid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_coreml"
    out = SESSIONS / sid
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, f in enumerate(uploaded):
        raw = f.read()
        name = f"f{i:04d}"
        (out / f"{name}.jpg").write_bytes(raw)
        im = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if im is None:
            continue
        h, w = im.shape[:2]
        server = predict(im)                       # PyTorch, full 8-view TTA
        row = {"frame": i, "w": w, "h": h,
               "server_px": server.tolist(),
               "server_ratio": _ratio(server)}

        # Write the same per-frame record and overlay a normal capture gets, so
        # these sessions are viewable and correctable in /admin and /review
        # rather than showing up as "0 frames".
        li = float(np.linalg.norm(server[1] - server[0]))
        lr = float(np.linalg.norm(server[3] - server[2]))
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(out / f"{name}_over.jpg"), draw(im, server),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        (out / f"{name}.json").write_text(json.dumps({
            "name": name, "model_points": server.tolist(), "points": server.tolist(),
            "edited": False, "blurry": False,
            "sharpness": float(cv2.Laplacian(g, cv2.CV_64F).var()),
            "index_len_px": li, "ring_len_px": lr,
            "ratio": (li / lr) if lr > 1e-6 else None,
        }), encoding="utf-8")
        if i < len(device_pts) and device_pts[i]:
            try:
                # the app sends normalised coords; the model is letterboxed, so
                # undo the same letterbox to land back in original pixels
                dp = np.array(device_pts[i], np.float32)
                lb, sc, dx, dy = ro.letterbox(im)
                dev_px = (dp * ro.INPUT_SIZE - np.array([dx, dy], np.float32)) / sc
                hs = (float(np.linalg.norm(server[1] - server[0])) +
                      float(np.linalg.norm(server[3] - server[2]))) / 2
                d = np.linalg.norm(dev_px - server, axis=1)
                row.update({"device_px": dev_px.tolist(),
                            "device_ratio": _ratio(dev_px),
                            "px_diff": d.tolist(),
                            "px_diff_mean": float(d.mean()),
                            "norm_diff_mean": float(d.mean() / hs) if hs > 0 else None})
            except Exception as e:
                row["device_error"] = str(e)[:120]
        if i < len(ms):
            row["device_ms"] = ms[i]
        if depth_m and i < len(depth_m):
            row["depth_m"] = depth_m[i]
        if tilt and i < len(tilt):
            row["tilt_deg"] = tilt[i]
        if pts3d and i < len(pts3d) and pts3d[i]:
            try:
                q = np.asarray(pts3d[i], float)
                a = float(np.linalg.norm(q[1] - q[0]))
                b = float(np.linalg.norm(q[3] - q[2]))
                row["pts3d"] = q.tolist()
                row["index_mm"] = a * 1000
                row["ring_mm"] = b * 1000
                row["ratio3d"] = (a / b) if b > 1e-9 else None
            except Exception as e:
                row["pts3d_error"] = str(e)[:120]
        rows.append(row)

    have = [r for r in rows if "px_diff_mean" in r]
    summary = {"session": sid, "device": device, "n_frames": len(rows),
               "n_compared": len(have)}
    if have:
        pd_ = np.array([r["px_diff_mean"] for r in have])
        dr = np.array([r["device_ratio"] for r in have if r.get("device_ratio")])
        sr = np.array([r["server_ratio"] for r in have if r.get("server_ratio")])
        summary.update({
            "px_diff_mean": float(pd_.mean()), "px_diff_max": float(pd_.max()),
            "norm_diff_mean": float(np.mean([r["norm_diff_mean"] for r in have
                                             if r.get("norm_diff_mean") is not None])),
            "device_ratio_mean": float(dr.mean()) if len(dr) else None,
            "server_ratio_mean": float(sr.mean()) if len(sr) else None,
            "ratio_diff_mean": float(np.mean(np.abs(dr - sr))) if len(dr) == len(sr) and len(dr) else None,
            "device_ratio_sd": float(dr.std(ddof=1)) if len(dr) > 1 else None,
            "server_ratio_sd": float(sr.std(ddof=1)) if len(sr) > 1 else None,
        })
    if ms:
        summary["device_ms_mean"] = float(np.mean(ms))

    # The comparison the whole exercise is for: is the 3D ratio steadier than
    # the 2D one across the same frames?
    r3 = np.array([r["ratio3d"] for r in rows if r.get("ratio3d")], float)
    r2 = np.array([r["device_ratio"] for r in rows
                   if r.get("ratio3d") and r.get("device_ratio")], float)
    if len(r3) > 1:
        summary["ratio3d_mean"] = float(r3.mean())
        summary["ratio3d_sd"] = float(r3.std(ddof=1))
        summary["ratio2d_sd_same_frames"] = float(r2.std(ddof=1)) if len(r2) > 1 else None
        if len(r2) > 1 and r2.std(ddof=1) > 0:
            summary["sd_ratio_3d_over_2d"] = float(r3.std(ddof=1) / r2.std(ddof=1))
        tl = np.array([r["tilt_deg"] for r in rows if r.get("tilt_deg") is not None], float)
        if len(tl):
            summary["tilt_mean"] = float(tl.mean())
            summary["tilt_min"] = float(tl.min())
            summary["tilt_max"] = float(tl.max())
    (out / "coreml_test.json").write_text(json.dumps({"summary": summary, "frames": rows}, indent=2),
                                          encoding="utf-8")
    ratios = [r["server_ratio"] for r in rows if r.get("server_ratio")]
    sess = dict(summary)
    sess.update({
        "source": "coreml-test",
        "n_valid": len(ratios),
        "median_ratio": float(np.median(ratios)) if ratios else 0.0,
        "sd_ratio": float(np.std(ratios, ddof=1)) if len(ratios) > 1 else 0.0,
        "utc": datetime.now(timezone.utc).isoformat(),
        "sex": None, "autistic": None, "hand": None,
        "quality": {"level": "ok", "reasons": []},
        **CKPT_INFO,
    })
    (out / "session.json").write_text(json.dumps(sess, indent=2), encoding="utf-8")
    mirror_session(sid)
    return jsonify(summary)


def _ratio(p):
    p = np.asarray(p, float)
    a = float(np.linalg.norm(p[1] - p[0])); b = float(np.linalg.norm(p[3] - p[2]))
    return (a / b) if b > 1e-6 else None


@app.get("/api/admin/sessions")
def admin_sessions():
    items = []
    if SESSIONS.exists():
        for d in sorted(SESSIONS.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            js = sorted(d.glob("f*.json"))
            edited = 0
            for j in js:
                try:
                    if json.loads(j.read_text(encoding="utf-8")).get("edited"):
                        edited += 1
                except Exception:
                    pass
            tags = {k: "" for k in TAG_FIELDS}
            sj = d / "session.json"
            if sj.exists():
                try:
                    m = json.loads(sj.read_text(encoding="utf-8"))
                    tags = {k: (m.get(k) or "") for k in TAG_FIELDS}
                except Exception:
                    pass
            clip = next((c.name for c in sorted(d.glob("clip.*"))
                         if c.suffix.lower() in (".mp4", ".webm", ".mov", ".m4v")), None)
            items.append({"id": d.name, "n": len(js), "edited": edited,
                          "clip": clip, "has_full": (d / "full").is_dir(),
                          **tags})
    return jsonify({"sessions": items})


@app.get("/api/admin/frames")
def admin_frames():
    sid = request.args.get("session", "")
    d = SESSIONS / sid
    if not d.is_dir():
        return jsonify({"frames": []})
    frames = []
    for j in sorted(d.glob("f*.json")):
        try:
            r = json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            continue
        frames.append({"name": r["name"], "url": f"/f/{sid}/{r['name']}.jpg",
                       "points": r["points"], "model_points": r["model_points"],
                       "sharpness": round(r.get("sharpness", 0.0), 1),
                       "edited": r.get("edited", False)})
    return jsonify({"frames": frames})


@app.post("/api/admin/annotate")
def admin_annotate():
    b = request.get_json(force=True)
    d = SESSIONS / b["session"]
    j = d / f"{b['name']}.json"
    if not j.exists():
        return jsonify({"error": "no such frame"}), 404
    r = json.loads(j.read_text(encoding="utf-8"))
    pts = np.array(b["points"], float)
    r["points"] = pts.tolist()
    r["edited"] = True
    r["edited_utc"] = datetime.now(timezone.utc).isoformat()
    li = float(np.linalg.norm(pts[1] - pts[0]))
    lr = float(np.linalg.norm(pts[3] - pts[2]))
    r["index_len_px"], r["ring_len_px"] = li, lr
    r["ratio"] = (li / lr) if lr > 1e-6 else None
    j.write_text(json.dumps(r), encoding="utf-8")
    img = cv2.imread(str(d / f"{b['name']}.jpg"))
    if img is not None:
        cv2.imwrite(str(d / f"{b['name']}_over.jpg"), draw(img, pts),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return jsonify({"ok": True, "ratio": r["ratio"]})


@app.post("/api/admin/cleanup")
def admin_cleanup():
    """Move blurry frames aside. Nothing is deleted.

    This used to unlink them. Frames are not reproducible -- a school visit
    under IRB cannot be re-run to recover one -- so soft frames are moved into
    <session>/set_aside/ instead, where they stay recoverable and still get
    mirrored to the backup. An edited frame is never touched: it represents
    human effort and may have been corrected precisely *because* it was
    marginal.
    """
    sid = request.args.get("session", "")
    dry = request.args.get("dry", "0") == "1"
    d = SESSIONS / sid
    if not d.is_dir():
        return jsonify({"error": "no such session"}), 404
    recs = []
    for j in sorted(d.glob("f*.json")):
        try:
            recs.append((j, json.loads(j.read_text(encoding="utf-8"))))
        except Exception:
            continue
    if not recs:
        return jsonify({"removed": 0, "kept": 0})
    sharps = np.array([r.get("sharpness", 0.0) for _, r in recs], float)
    thr = max(BLUR_FLOOR, BLUR_REL * float(np.percentile(sharps, 90)))
    doomed = [(j, r) for j, r in recs
              if r.get("sharpness", 0.0) < thr and not r.get("edited")]
    if len(recs) - len(doomed) < 8:            # never strip a session bare
        doomed = sorted(doomed, key=lambda t: t[1].get("sharpness", 0.0))[:max(0, len(recs) - 8)]
    moved = 0
    if not dry and doomed:
        aside = d / "set_aside"
        aside.mkdir(exist_ok=True)
        for j, r in doomed:
            for f in d.glob(r["name"] + "*"):
                if f.is_file():
                    shutil.move(str(f), str(aside / f.name))
                    moved += 1
        mirror_session(sid)
    return jsonify({"removed": len(doomed), "kept": len(recs) - len(doomed),
                    "files_moved": moved, "moved_to": "set_aside",
                    "deleted": 0,
                    "threshold": round(thr, 1), "dry": dry})


@app.post("/api/admin/export")
def admin_export():
    """Write hand-corrected frames out in the annotator's JSON shape."""
    sid = request.args.get("session", "")
    d = SESSIONS / sid
    outdir = CAPTURES / "exported" / sid
    outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for j in sorted(d.glob("f*.json")):
        r = json.loads(j.read_text(encoding="utf-8"))
        if not r.get("edited"):
            continue
        img_id = f"phone__{sid}__{r['name']}.jpg"
        shutil.copy2(d / f"{r['name']}.jpg", outdir / img_id)
        pts = r["points"]
        (outdir / f"{img_id}.json").write_text(json.dumps({
            "image_id": img_id, "dataset": "phone_capture",
            "filtered_path": f"data/phone/{img_id}",
            "status": "accepted", "source_session": sid,
            "timestamp_utc": r.get("edited_utc"),
            "points": {NAMES[i]: {"x": pts[i][0], "y": pts[i][1],
                                  "preplaced_x": r["model_points"][i][0],
                                  "preplaced_y": r["model_points"][i][1]}
                       for i in range(4)},
        }, indent=2), encoding="utf-8")
        n += 1
    return jsonify({"n": n, "path": str(outdir)})



# ---------------------------------------------------------------------------
# v2 upload: one frame per request.
#
# /api/session takes a whole capture as a single multipart body. On school wifi
# that is the wrong shape: a 25 MB body that fails at 90% costs the entire
# capture and every frame in it, and the phone cannot tell the operator which
# part failed. Here each frame is its own request, so a dropped connection costs
# one frame, the phone retries only that frame, and a capture survives being
# carried between rooms.
#
# Staging only. The measurement is NOT reimplemented here: commit re-enters
# /api/session internally with the frames it collected, so there is exactly one
# copy of the analysis. Two copies would drift, and an instrument that measures
# differently depending on which endpoint the phone used is worse than a slow one.
V2_STAGING = CAPTURES / "staging"
_V2_CID = re.compile(r"[0-9]{8}_[0-9]{6}_[0-9a-f]{6}")
V2_MAX_FRAMES = 400


def _v2_dir(cid: str, make: bool = False):
    """Staging directory for a capture id, or None if the id is not one of ours."""
    if not _V2_CID.fullmatch(cid or ""):
        return None
    d = V2_STAGING / cid
    if make:
        (d / "frames").mkdir(parents=True, exist_ok=True)
    elif not d.is_dir():
        return None
    return d


def _v2_received(d):
    return [int(p.stem[1:]) for p in sorted((d / "frames").glob("f[0-9][0-9][0-9][0-9].jpg"))]


@app.post("/api/v2/capture/begin")
def api_v2_begin():
    ok_space, space_msg = space_check()
    if not ok_space:
        return jsonify({"error": space_msg}), 507
    cid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    d = _v2_dir(cid, make=True)
    # Every field the phone would have posted to /api/session, kept so commit can
    # replay it. The device token is deliberately not stored: it is a credential,
    # it is resent on every request, and a token on disk outlives its capture.
    form = {k: v for k, v in request.form.items() if k != "device_token"}
    (d / "form.json").write_text(json.dumps(form), encoding="utf-8")
    return jsonify({"capture": cid})


@app.post("/api/v2/capture/<cid>/frame")
def api_v2_frame(cid):
    d = _v2_dir(cid)
    if d is None:
        return jsonify({"error": "unknown capture"}), 404
    f = request.files.get("frame")
    if f is None:
        return jsonify({"error": "no frame"}), 400
    try:
        idx = int(request.form.get("idx", ""))
    except ValueError:
        return jsonify({"error": "idx must be an integer"}), 400
    if not 0 <= idx < V2_MAX_FRAMES:
        return jsonify({"error": f"idx out of range 0..{V2_MAX_FRAMES - 1}"}), 400

    data = f.read()
    if not data:
        return jsonify({"error": "empty frame"}), 400
    # Written under a temp name and renamed, so a connection that dies mid-write
    # cannot leave a truncated JPEG that later looks like a delivered frame.
    dst = d / "frames" / f"f{idx:04d}.jpg"
    tmp = dst.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(dst)

    # The phone's own score for this frame: why it thought this one worth sending
    # first. Kept so a bad selection heuristic can be diagnosed later against what
    # the server independently measures.
    score = request.form.get("score")
    if score:
        try:
            json.loads(score)
            (d / "frames" / f"f{idx:04d}.json").write_text(score[:4000], encoding="utf-8")
        except Exception:
            pass
    return jsonify({"ok": True, "idx": idx, "received": len(_v2_received(d))})


@app.get("/api/v2/capture/<cid>/status")
def api_v2_status(cid):
    d = _v2_dir(cid)
    if d is None:
        return jsonify({"error": "unknown capture"}), 404
    got = _v2_received(d)
    return jsonify({"capture": cid, "received": got, "count": len(got)})


@app.post("/api/v2/capture/<cid>/commit")
def api_v2_commit(cid):
    d = _v2_dir(cid)
    if d is None:
        return jsonify({"error": "unknown capture"}), 404
    got = _v2_received(d)
    if not got:
        return jsonify({"error": "no frames were received"}), 400

    try:
        form = json.loads((d / "form.json").read_text(encoding="utf-8"))
    except Exception:
        form = {}
    # Late-arriving fields (a tag chosen after the frames went up) win.
    for k, v in request.form.items():
        if k not in ("device_token", "idx", "score"):
            form[k] = v

    data = dict(form)
    tok = request.headers.get("X-Device-Token") or request.form.get("device_token") or ""
    if tok:
        data["device_token"] = tok
    data["frames"] = [
        (io.BytesIO((d / "frames" / f"f{i:04d}.jpg").read_bytes()), f"f{i:04d}.jpg")
        for i in got
    ]

    with app.test_client() as c:
        resp = c.post("/api/session", data=data,
                      content_type="multipart/form-data",
                      headers={"X-Device-Token": tok} if tok else None)
    body = resp.get_json(silent=True) or {"error": "session did not return JSON"}
    if isinstance(body, dict):
        body["frames_committed"] = len(got)
        body["capture"] = cid

    if 200 <= resp.status_code < 300:
        # Carry the phone's own per-frame scores into the session before the
        # staging copy goes. They were being discarded here, which defeated the
        # reason for collecting them: without them there is no way to check the
        # phone's selection against what the server independently measured, and
        # that comparison is exactly what caught the sharpness-scale bug.
        sid = body.get("session") if isinstance(body, dict) else None
        if sid:
            store = (form.get("store") or "dev").strip().lower()
            store = "study" if store == "study" else "dev"
            dest = _store_dir(store) / sid
            if dest.is_dir():
                try:
                    (dest / "client_scores").mkdir(exist_ok=True)
                    for j in sorted((d / "frames").glob("f[0-9][0-9][0-9][0-9].json")):
                        shutil.copy2(j, dest / "client_scores" / j.name)
                except Exception as e:
                    print(f"could not keep client scores for {sid}: {e}", flush=True)
        shutil.rmtree(d, ignore_errors=True)
    return jsonify(body), resp.status_code


# ---------------------------------------------------------------------------
# Study aggregates.
#
# /api/admin/sessions answers "what files exist" -- frame counts, edit counts,
# tags. It cannot answer "what did we measure", because it never opens the
# measurement. Everything below reads session.json, which api_session already
# writes with the full result, so these are views over existing records rather
# than a second source of truth. Nothing here recomputes a ratio.
REF_BANDS = {"male": {"mean": 0.964, "sd": 0.030},
             "female": {"mean": 0.975, "sd": 0.028}}
MODEL_REPEATABILITY = 0.016     # the model's own spread across frames of one hand
HUMAN_INTEROBSERVER = 0.008     # expert agreement; the bar an automated pipeline beats


def _study_dirs():
    for store, root in (("dev", SESSIONS), ("study", STUDY_SESSIONS)):
        if root.exists():
            for d in sorted(root.iterdir(), reverse=True):
                if d.is_dir():
                    yield store, d


def _session_row(store, d):
    sj = d / "session.json"
    if not sj.exists():
        return None
    try:
        m = json.loads(sj.read_text(encoding="utf-8"))
    except Exception:
        return None
    q = m.get("quality") or {}
    row = {"id": d.name, "store": store,
           "subject": m.get("subject"), "utc": m.get("utc"),
           "median_ratio": m.get("median_ratio"),
           "ratio_of_means": m.get("ratio_of_means"),
           "sd_ratio": m.get("sd_ratio"),
           "quality": q.get("level"), "quality_reasons": q.get("reasons") or [],
           "ratio_from": m.get("ratio_from"),
           "n_frames": m.get("n_frames"), "n_valid": m.get("n_valid"),
           "n_sharp": m.get("n_sharp"), "n_blurry": m.get("n_blurry"),
           "age": m.get("age"), "source": m.get("source"),
           "model_checkpoint": m.get("model_checkpoint"),
           "crease_convention": m.get("crease_convention")}
    for k in TAG_FIELDS:
        row[k] = m.get(k)
    for k in ("nationality", "ethnicity", "operator", "protocol"):
        row[k] = m.get(k)
    return row


def _study_rows(subject=None, store=None, quality=None):
    out = []
    for st, d in _study_dirs():
        if store and st != store:
            continue
        r = _session_row(st, d)
        if r is None:
            continue
        if subject and (r.get("subject") or "") != subject:
            continue
        if quality and (r.get("quality") or "") != quality:
            continue
        out.append(r)
    return out


@app.get("/api/study/sessions")
def study_sessions():
    """Session-level measurements. The dashboard, subject detail and review list
    all need ratio + SD + quality per session, which the inventory route does
    not carry."""
    rows = _study_rows(subject=(request.args.get("subject") or "").strip() or None,
                       store=(request.args.get("store") or "").strip() or None,
                       quality=(request.args.get("quality") or "").strip() or None)
    try:
        limit = max(1, min(2000, int(request.args.get("limit", "500"))))
    except ValueError:
        limit = 500
    return jsonify({"sessions": rows[:limit], "n": len(rows), "returned": min(len(rows), limit)})


@app.get("/api/study/summary")
def study_summary():
    """Aggregates for the dashboard. Every figure reports the n it came from;
    a subgroup smaller than MIN_GROUP is counted but its statistics are withheld,
    because a median over two children invites a conclusion it cannot support."""
    MIN_GROUP = 5
    rows = _study_rows(store=(request.args.get("store") or "").strip() or None)
    measured = [r for r in rows if isinstance(r.get("median_ratio"), (int, float))
                and r.get("quality") != "fail"]

    def stats(vals):
        if not vals:
            return {"n": 0}
        a = np.array(vals, dtype=float)
        return {"n": int(a.size),
                "median": float(np.median(a)),
                "mean": float(a.mean()),
                "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0}

    def group_by(key):
        buckets = {}
        for r in measured:
            k = r.get(key)
            if k in (None, ""):
                k = "unrecorded"
            buckets.setdefault(str(k), []).append(r["median_ratio"])
        out = {}
        for k, v in buckets.items():
            if len(v) < MIN_GROUP:
                out[k] = {"n": len(v), "withheld": True,
                          "note": f"fewer than {MIN_GROUP} sessions"}
            else:
                out[k] = stats(v)
        return out

    # Age is banded on the server, not in the browser. Grouping client-side
    # would let a page assemble a subgroup the withholding rule was meant to
    # suppress. The bands follow the childhood range the study recruits from;
    # 2D:4D rises through childhood, so a single pooled figure hides the trend
    # that matters.
    def age_bands():
        edges = [(0, 6), (6, 9), (9, 12), (12, 15), (15, 25)]
        out = {}
        for lo, hi in edges:
            label = f"{lo}-{hi}"
            vals = [r["median_ratio"] for r in measured
                    if isinstance(r.get("age"), (int, float)) and lo <= r["age"] < hi]
            if not vals:
                out[label] = {"n": 0}
            elif len(vals) < MIN_GROUP:
                out[label] = {"n": len(vals), "withheld": True,
                              "note": f"fewer than {MIN_GROUP} sessions"}
            else:
                out[label] = stats(vals)
        unrecorded = [r for r in measured if not isinstance(r.get("age"), (int, float))]
        out["unrecorded"] = {"n": len(unrecorded)}
        return out

    ratios = [r["median_ratio"] for r in measured]
    edges = [round(0.85 + i * 0.01, 2) for i in range(26)]      # 0.85 .. 1.10
    hist = [0] * (len(edges) - 1)
    for v in ratios:
        for i in range(len(hist)):
            if edges[i] <= v < edges[i + 1]:
                hist[i] += 1
                break

    sds = [r["sd_ratio"] for r in rows if isinstance(r.get("sd_ratio"), (int, float))]
    by_day = {}
    for r in rows:
        u = (r.get("utc") or "")[:10]
        if u:
            by_day[u] = by_day.get(u, 0) + 1

    quality_mix = {"ok": 0, "poor": 0, "fail": 0, "unknown": 0}
    for r in rows:
        quality_mix[r.get("quality") or "unknown"] = quality_mix.get(r.get("quality") or "unknown", 0) + 1

    subjects = {r.get("subject") for r in rows if r.get("subject")}
    return jsonify({
        "n_sessions": len(rows),
        "n_measured": len(measured),
        "n_subjects": len(subjects),
        "quality_mix": quality_mix,
        "ratio": stats(ratios),
        "ratio_histogram": {"edges": edges, "counts": hist},
        "sd": {**stats(sds),
               "above_poor": int(sum(1 for v in sds if v > 0.04)),
               "above_fail": int(sum(1 for v in sds if v > 0.08))},
        "by_age_band": age_bands(),
        "by_sex": group_by("sex"),
        "by_autistic": group_by("autistic"),
        "by_hand": group_by("hand"),
        "by_nationality": group_by("nationality"),
        "by_ethnicity": group_by("ethnicity"),
        "captures_over_time": [{"date": k, "n": by_day[k]} for k in sorted(by_day)],
        "reference_bands": REF_BANDS,
        "model_repeatability": MODEL_REPEATABILITY,
        "human_interobserver": HUMAN_INTEROBSERVER,
        "min_group": MIN_GROUP,
    })


@app.get("/api/study/storage")
def study_storage():
    """Free space, so the admin page can show the real figure instead of
    inventing one. The same numbers the upload gate uses."""
    u = shutil.disk_usage(CAPTURES)
    return jsonify({"free_gb": round(u.free / 2**30, 1),
                    "total_gb": round(u.total / 2**30, 1),
                    "used_gb": round(u.used / 2**30, 1),
                    "warn_below_gb": DISK_WARN_GB,
                    "refuse_below_gb": DISK_MIN_GB,
                    "ok": u.free / 2**30 >= DISK_MIN_GB})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/synth_heatmap_robust768.pt")
    ap.add_argument("--input-size", type=int, default=None)
    ap.add_argument("--port", type=int, default=5055)
    ap.add_argument("--captures", default=None)
    ap.add_argument("--remote-predict", default=None,
                    help="fallback inference node, e.g. http://<inference-host>:5056 "
                         "(the GTX 1660). Used only when local prediction fails.")
    ap.add_argument("--max-mb", type=int, default=None)
    ap.add_argument("--no-tta", action="store_true",
                    help="single pass instead of 4-way rotation TTA (4x faster, 4.3%% worse)")
    a = ap.parse_args()
    if a.captures:
        CAPTURES = Path(a.captures)
        SESSIONS = CAPTURES / "sessions"
    if a.max_mb:
        MAX_MB = a.max_mb
    if a.no_tta:
        TTA = False
    load_model(REPO / a.ckpt, a.input_size)
    SESSIONS.mkdir(parents=True, exist_ok=True)
    if a.remote_predict:
        globals()["REMOTE_PREDICT"] = a.remote_predict.rstrip("/")
        print(f"remote predict fallback -> {REMOTE_PREDICT}", flush=True)
    print(f"sessions -> {SESSIONS}  (cap {MAX_MB} MB)", flush=True)
    app.run(host="0.0.0.0", port=a.port, threaded=True)
