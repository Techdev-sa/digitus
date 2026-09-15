/* 2D:4D crease annotator — vanilla JS canvas client.
 *
 * World coords = image pixels. View transform: screen = world * scale + pan.
 * Keyboard: Enter accept+next, Backspace previous, S skip, F fit view.
 */
"use strict";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const el = {
  progress: document.getElementById("progress"),
  imageLabel: document.getElementById("imageLabel"),
  status: document.getElementById("status"),
  model: document.getElementById("model"),
  modelGuessLegend: document.getElementById("modelGuessLegend"),
  lastMatch: document.getElementById("lastMatch"),
  datasetPick: document.getElementById("datasetPick"),
  measure: document.getElementById("measure"),
};

// Expert measurements from the Mendeley study (same pixel scale as our split
// scans). Loaded once; used for the live closest-expert-match readout.
let mendeleyRef = [];
fetch("/api/reference/mendeley").then(r => r.json()).then(rows => { mendeleyRef = rows; });

function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

// expert row whose scanner lengths (this hand, averaged over both observers)
// sit closest to the given measurement; null when not a Mendeley image
function bestExpertMatch(d2, d4) {
  if (!meta || meta.dataset !== "mendeley_2d4d" || !mendeleyRef.length) return null;
  const hand = (meta.manifest_handedness || "").toLowerCase() === "left" ? "l" : "r";
  let best = null, bestCost = Infinity;
  for (const row of mendeleyRef) {
    const ref2 = (parseFloat(row[`o1_scan_${hand}2d`]) + parseFloat(row[`o2_scan_${hand}2d`])) / 2;
    const ref4 = (parseFloat(row[`o1_scan_${hand}4d`]) + parseFloat(row[`o2_scan_${hand}4d`])) / 2;
    if (!isFinite(ref2) || !isFinite(ref4)) continue;
    const cost = Math.abs(d2 - ref2) + Math.abs(d4 - ref4);
    if (cost < bestCost) { bestCost = cost; best = { subject: row.subject, hand, ref2, ref4 }; }
  }
  return best;
}

function updateMeasure() {
  if (!meta) { el.measure.textContent = ""; return; }
  const d2 = dist(points.index_base, points.index_tip);
  const d4 = dist(points.ring_base, points.ring_tip);
  const ratio = d4 > 1 ? (d2 / d4) : 0;
  const locked = (lockLen.index || lockLen.ring) ? "🔒 " : "";
  el.measure.innerHTML = "";
  el.measure.append(`${locked}2D ${d2.toFixed(0)}px · 4D ${d4.toFixed(0)}px · ratio ${ratio.toFixed(4)}`);

  const best = bestExpertMatch(d2, d4);
  if (best) {
    const span = document.createElement("span");
    span.className = "expert";
    span.textContent = `  |  expert ${best.subject} (${best.hand.toUpperCase()}): ` +
      `2D ${best.ref2.toFixed(0)} · 4D ${best.ref4.toFixed(0)} · ratio ${(best.ref2 / best.ref4).toFixed(4)} ` +
      `(Δ2D ${(d2 - best.ref2).toFixed(0)}px, Δ4D ${(d4 - best.ref4).toFixed(0)}px)`;
    el.measure.append(span);
  }
}

// ?dataset=mendeley_2d4d -> annotate that dataset only (worklist, progress,
// and resume all scope to it); no param = all datasets
const DATASET_FILTER = new URLSearchParams(location.search).get("dataset") || "";

let statusPoll = null;

function acceptedCount() {
  return worklist.filter(w => w.status === "accepted").length;
}

function fmtErr(x) { return x == null ? "?" : x.toFixed(3); }

function showModel(m) {
  if (!m) return;
  const n = acceptedCount();
  el.model.title = "";

  if (m.training) {
    el.model.textContent = `🧠 training… (n=${n})`;
    el.model.className = "training";
    startStatusPoll();
    return;
  }

  if (m.ready) {
    // ready means it was measured beating MediaPipe on held-out images
    const pct = (m.model_err_hand != null && m.heuristic_err_hand)
      ? Math.round((1 - m.model_err_hand / m.heuristic_err_hand) * 100) : null;
    el.model.textContent = `🧠 our model` +
      (pct !== null ? ` · ${pct}% better than MediaPipe` : "") +
      ` · n=${n} · err ${fmtErr(m.model_err_hand)} vs mp ${fmtErr(m.heuristic_err_hand)}`;
    el.model.title = `validated on ${m.val_compared} held-out images`;
    el.model.className = "ours";
  } else if (m.trained_on > 0) {
    // a checkpoint exists but hasn't beaten MediaPipe yet (or not enough
    // validation images to trust the comparison) -- MediaPipe still places
    const untilRetrain = Math.max(0, m.retrain_every - (n - m.trained_on));
    const proof = (m.model_err_hand != null && m.heuristic_err_hand != null)
      ? `err ${fmtErr(m.model_err_hand)} vs mp ${fmtErr(m.heuristic_err_hand)}`
      : `val ${m.val_compared}/${m.min_val_for_ready}`;
    el.model.textContent = `MediaPipe (in use) · our net warming up: n=${m.trained_on}, ${proof}` +
      (untilRetrain > 0 ? ` · retrains in ${untilRetrain}` : "");
    el.model.title = `checkpoint trained on ${m.trained_on} · needs ${m.min_val_for_ready}+ ` +
      `held-out images beating MediaPipe to take over`;
    el.model.className = "warming";
  } else {
    const remain = Math.max(0, m.min_train - n);
    el.model.textContent = `MediaPipe (bootstrap) · n=${n}` +
      (remain > 0 ? ` · ${remain} more to first training` : "");
    el.model.className = "";
  }
}

// Training runs in a background thread server-side; if you sit on one image
// while it finishes, nothing else would refresh the badge. Poll until done.
function startStatusPoll() {
  if (statusPoll) return;
  statusPoll = setInterval(async () => {
    const r = await fetch("/api/status");
    if (!r.ok) return;
    const { model } = await r.json();
    if (!model.training) {
      clearInterval(statusPoll);
      statusPoll = null;
    }
    showModel(model);
  }, 3000);
}

const POINT_STYLE = {
  index_base: { color: "#4cd964", label: "2D base" },
  index_tip:  { color: "#a6e88f", label: "2D tip" },
  ring_base:  { color: "#4da3ff", label: "4D base" },
  ring_tip:   { color: "#9fd0ff", label: "4D tip" },
};
const POINT_ORDER = ["index_base", "index_tip", "ring_base", "ring_tip"];
const LINES = { index: ["index_base", "index_tip"], ring: ["ring_base", "ring_tip"] };
const HIT_RADIUS = 12; // screen px

let worklist = [];        // [{image_id, dataset, status}]
let cursor = -1;          // index into worklist
let img = new Image();
let meta = null;          // /api/preplace response for current image
let points = {};          // name -> {x, y} in image px (current, draggable)
let prePoints = {};       // name -> {x, y} pre-placed positions (frozen)
let view = { scale: 1, tx: 0, ty: 0 };
let drag = null;          // {type:"point"|"line"|"pan", ...}
let spaceHeld = false;
let saving = false;
// length lock per finger: when set, dragging the line translates it rigidly
// and dragging an endpoint ROTATES about the opposite endpoint instead of
// stretching — the length stays exactly as locked (E = expert lengths).
let lockLen = { index: null, ring: null };

// ---------- view helpers ----------

function toScreen(p) { return { x: p.x * view.scale + view.tx, y: p.y * view.scale + view.ty }; }
function toWorld(sx, sy) { return { x: (sx - view.tx) / view.scale, y: (sy - view.ty) / view.scale }; }

function fitView() {
  if (!meta) return;
  const s = Math.min(canvas.width / meta.width, canvas.height / meta.height) * 0.97;
  view.scale = s;
  view.tx = (canvas.width - meta.width * s) / 2;
  view.ty = (canvas.height - meta.height * s) / 2;
  draw();
}

function zoomToPoints() {
  // open zoomed on the annotation region: bbox of the 4 points + margin
  if (!meta) return;
  const xs = POINT_ORDER.map(n => points[n].x);
  const ys = POINT_ORDER.map(n => points[n].y);
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  let y0 = Math.min(...ys), y1 = Math.max(...ys);
  const padX = Math.max((x1 - x0) * 0.35, meta.width * 0.04);
  const padY = Math.max((y1 - y0) * 0.20, meta.height * 0.04);
  x0 -= padX; x1 += padX; y0 -= padY; y1 += padY;
  const s = Math.min(canvas.width / (x1 - x0), canvas.height / (y1 - y0),
                     4 * (window.devicePixelRatio || 1)); // don't over-magnify
  view.scale = s;
  view.tx = (canvas.width - (x0 + x1) * s) / 2;
  view.ty = (canvas.height - (y0 + y1) * s) / 2;
  draw();
}

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(r.width * dpr);
  canvas.height = Math.round(r.height * dpr);
  canvas.style.width = r.width + "px";
  canvas.style.height = r.height + "px";
  fitView();
}
window.addEventListener("resize", resize);

// Mouse events arrive in CSS px; the canvas backing store is DPR-scaled.
function eventPos(e) {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  return { x: (e.clientX - r.left) * dpr, y: (e.clientY - r.top) * dpr };
}

// ---------- drawing ----------

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!meta || !img.complete) return;
  ctx.save();
  ctx.translate(view.tx, view.ty);
  ctx.scale(view.scale, view.scale);
  ctx.imageSmoothingEnabled = view.scale < 2; // crisp pixels when zoomed in
  ctx.drawImage(img, 0, 0);
  ctx.restore();

  // finger measurement lines (base -> tip), drawn in screen space
  for (const [a, b] of [["index_base", "index_tip"], ["ring_base", "ring_tip"]]) {
    const pa = toScreen(points[a]), pb = toScreen(points[b]);
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.setLineDash([6, 5]);
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.stroke();
    ctx.setLineDash([]);
  }

  // our model's current guess, shown as a faint ring even while MediaPipe is
  // the one actually placing the crosshairs -- lets you watch it improve
  // image by image instead of only reading aggregate stats. Skipped when the
  // model IS the pre-placement source (crosshair already shows this exact
  // point, a ghost on top of it would be redundant).
  if (meta.model_prediction && !meta.preplace_source.startsWith("crease_net")) {
    for (const name of POINT_ORDER) {
      const mp = meta.model_prediction[name];
      if (!mp) continue;
      const p = toScreen(mp);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(255,255,255,0.55)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(p.x, p.y, 1.5, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.85)";
      ctx.fill();
    }
  }

  for (const name of POINT_ORDER) {
    const p = toScreen(points[name]);
    const st = POINT_STYLE[name];
    // crosshair for pixel-precise placement
    ctx.strokeStyle = st.color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(p.x - 14, p.y); ctx.lineTo(p.x - 4, p.y);
    ctx.moveTo(p.x + 4, p.y);  ctx.lineTo(p.x + 14, p.y);
    ctx.moveTo(p.x, p.y - 14); ctx.lineTo(p.x, p.y - 4);
    ctx.moveTo(p.x, p.y + 4);  ctx.lineTo(p.x, p.y + 14);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
    ctx.fillStyle = st.color;
    ctx.fill();
    ctx.font = "12px system-ui";
    ctx.fillStyle = st.color;
    ctx.fillText(st.label, p.x + 16, p.y - 8);
  }

  updateMeasure();
}

// ---------- data flow ----------

async function loadWorklist() {
  const all = await (await fetch("/api/images")).json();

  // dataset switcher: "all" + each dataset with its image count
  const counts = {};
  for (const w of all) counts[w.dataset] = (counts[w.dataset] || 0) + 1;
  el.datasetPick.innerHTML =
    `<option value="">all datasets (${all.length})</option>` +
    Object.entries(counts).map(([d, n]) =>
      `<option value="${d}"${d === DATASET_FILTER ? " selected" : ""}>${d} (${n})</option>`
    ).join("");
  el.datasetPick.onchange = () => {
    location.search = el.datasetPick.value ? `?dataset=${encodeURIComponent(el.datasetPick.value)}` : "";
  };

  worklist = DATASET_FILTER ? all.filter(w => w.dataset === DATASET_FILTER) : all;
  if (!worklist.length) {
    el.progress.textContent = `no images for dataset "${DATASET_FILTER}"`;
    return;
  }
  const firstOpen = worklist.findIndex(w => w.status === null);
  cursor = firstOpen === -1 ? worklist.length - 1 : firstOpen;
  await loadImage(cursor);
}

function updateProgress() {
  const done = acceptedCount();
  const skipped = worklist.filter(w => w.status === "skipped").length;
  const pct = worklist.length ? Math.round((done / worklist.length) * 1000) / 10 : 0;
  el.progress.textContent = `${done} / ${worklist.length} annotated (${pct}%)` +
    (skipped ? ` · ${skipped} skipped` : "");
  const w = worklist[cursor];
  el.imageLabel.textContent = w ? `#${cursor + 1}: ${w.image_id}` : "";
}

async function loadImage(i) {
  if (i < 0 || i >= worklist.length) return;
  cursor = i;
  updateProgress();
  el.status.textContent = "loading…";
  el.status.className = "";
  const id = worklist[i].image_id;
  meta = await (await fetch(`/api/preplace/${encodeURIComponent(id)}`)).json();

  prePoints = {};
  points = {};
  for (const name of POINT_ORDER) {
    prePoints[name] = { ...meta.points[name] };
    // revisiting an accepted image: show the corrected points, keep original pre-placement
    const saved = meta.existing && meta.existing.points && meta.existing.points[name];
    if (saved) {
      points[name] = { x: saved.x, y: saved.y };
      prePoints[name] = { x: saved.preplaced_x, y: saved.preplaced_y };
    } else {
      points[name] = { ...meta.points[name] };
    }
  }

  lockLen = { index: null, ring: null };
  showModel(meta.model);
  el.modelGuessLegend.style.display =
    (meta.model_prediction && !meta.preplace_source.startsWith("crease_net")) ? "inline-block" : "none";
  img = new Image();
  img.onload = () => {
    if (meta.preplace_source === "grid_fallback") {
      fitView();
      el.status.textContent = "no detection — points parked in a grid";
    } else {
      zoomToPoints();
      el.status.textContent = "";
    }
  };
  img.src = `/api/image/${encodeURIComponent(id)}`;
}

async function save(status) {
  if (saving || !meta) return;
  saving = true;
  const payload = {
    status,
    width: meta.width,
    height: meta.height,
    preplace_source: meta.preplace_source,
    model_prediction: meta.model_prediction,
    mediapipe: meta.mediapipe,
    points: {},
  };
  for (const name of POINT_ORDER) {
    payload.points[name] = {
      x: points[name].x, y: points[name].y,
      pre_x: prePoints[name].x, pre_y: prePoints[name].y,
    };
  }
  const id = worklist[cursor].image_id;
  const r = await fetch(`/api/save/${encodeURIComponent(id)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  saving = false;
  if (!r.ok) { el.status.textContent = "SAVE FAILED"; return; }
  const resp = await r.json();
  showModel(resp.model);
  // persistent (survives the next loadImage's "loading…", unlike #status) so
  // you actually get to read how close the model's guess was this time
  if (resp.model_similarity_pct) {
    el.lastMatch.textContent = `last match: ${resp.model_similarity_pct.avg}%`;
  }
  worklist[cursor].status = status;
  el.status.textContent = status === "accepted" ? "saved ✓" : "skipped";
  el.status.className = "saved";
  // advance to the next unannotated image after this one, else next in order
  let next = worklist.findIndex((w, j) => j > cursor && w.status === null);
  if (next === -1) next = cursor + 1;
  if (next < worklist.length) await loadImage(next);
  else { updateProgress(); el.status.textContent = "all images done 🎉"; }
}

// ---------- interaction ----------

function distToSegment(p, a, b) {
  const abx = b.x - a.x, aby = b.y - a.y;
  const len2 = abx * abx + aby * aby;
  const t = len2 ? Math.max(0, Math.min(1, ((p.x - a.x) * abx + (p.y - a.y) * aby) / len2)) : 0;
  return Math.hypot(p.x - (a.x + t * abx), p.y - (a.y + t * aby));
}

canvas.addEventListener("mousedown", (e) => {
  const pos = eventPos(e);
  const dpr = window.devicePixelRatio || 1;
  if (!spaceHeld) {
    for (const name of POINT_ORDER) {
      const p = toScreen(points[name]);
      if (Math.hypot(p.x - pos.x, p.y - pos.y) <= HIT_RADIUS * dpr) {
        drag = { type: "point", name };
        canvas.classList.add("dragging-point");
        return;
      }
    }
    // grab the line body: moves both endpoints rigidly (length preserved)
    for (const [finger, [a, b]] of Object.entries(LINES)) {
      if (distToSegment(pos, toScreen(points[a]), toScreen(points[b])) <= 8 * dpr) {
        drag = { type: "line", a, b, lastX: pos.x, lastY: pos.y };
        canvas.classList.add("dragging-point");
        return;
      }
    }
  }
  drag = { type: "pan", startX: pos.x, startY: pos.y, tx0: view.tx, ty0: view.ty };
  canvas.classList.add("panning");
});

window.addEventListener("mousemove", (e) => {
  if (!drag) return;
  const pos = eventPos(e);
  if (drag.type === "point") {
    const finger = drag.name.startsWith("index") ? "index" : "ring";
    const locked = lockLen[finger];
    if (locked) {
      // length-locked: rotate about the opposite endpoint at fixed length
      const pivotName = LINES[finger].find(n => n !== drag.name);
      const pivot = points[pivotName];
      const w = toWorld(pos.x, pos.y);
      const d = Math.hypot(w.x - pivot.x, w.y - pivot.y);
      if (d > 1) {
        points[drag.name] = {
          x: pivot.x + (w.x - pivot.x) / d * locked,
          y: pivot.y + (w.y - pivot.y) / d * locked,
        };
      }
    } else {
      points[drag.name] = toWorld(pos.x, pos.y);
    }
  } else if (drag.type === "line") {
    const dx = (pos.x - drag.lastX) / view.scale;
    const dy = (pos.y - drag.lastY) / view.scale;
    for (const n of [drag.a, drag.b]) {
      points[n] = { x: points[n].x + dx, y: points[n].y + dy };
    }
    drag.lastX = pos.x;
    drag.lastY = pos.y;
  } else {
    view.tx = drag.tx0 + (pos.x - drag.startX);
    view.ty = drag.ty0 + (pos.y - drag.startY);
  }
  draw();
});

window.addEventListener("mouseup", () => {
  drag = null;
  canvas.classList.remove("dragging-point", "panning");
});

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const pos = eventPos(e);
  const factor = Math.exp(-e.deltaY * 0.0015);
  const w = toWorld(pos.x, pos.y);
  view.scale = Math.min(40, Math.max(0.02, view.scale * factor));
  view.tx = pos.x - w.x * view.scale;
  view.ty = pos.y - w.y * view.scale;
  draw();
}, { passive: false });

window.addEventListener("keydown", (e) => {
  if (e.code === "Space") { spaceHeld = true; e.preventDefault(); return; }
  if (e.key === "Enter") { e.preventDefault(); save("accepted"); }
  else if (e.key === "Backspace") { e.preventDefault(); if (cursor > 0) loadImage(cursor - 1); }
  else if (e.key === "s" || e.key === "S") { e.preventDefault(); save("skipped"); }
  else if (e.key === "f" || e.key === "F") { fitView(); }
  else if (e.key === "z" || e.key === "Z") { zoomToPoints(); }
  else if (e.key === "e" || e.key === "E") { snapToExpertLengths(); }
  else if (e.key === "l" || e.key === "L") { toggleLengthLock(); }
});
window.addEventListener("keyup", (e) => { if (e.code === "Space") spaceHeld = false; });

// E (Mendeley images): set each line's length to the matched expert's CSV
// value — base stays, tip slides along the current direction — then lock, so
// you fit a known-length line onto the finger instead of re-measuring.
function snapToExpertLengths() {
  if (!meta) return;
  const best = bestExpertMatch(dist(points.index_base, points.index_tip),
                               dist(points.ring_base, points.ring_tip));
  if (!best) {
    el.status.textContent = "expert lengths only exist for mendeley_2d4d images";
    el.status.className = "";
    return;
  }
  for (const [finger, refLen] of [["index", best.ref2], ["ring", best.ref4]]) {
    const [baseN, tipN] = LINES[finger];
    const base = points[baseN], tip = points[tipN];
    const d = dist(base, tip);
    if (d > 1) {
      points[tipN] = { x: base.x + (tip.x - base.x) / d * refLen,
                       y: base.y + (tip.y - base.y) / d * refLen };
    }
    lockLen[finger] = refLen;
  }
  el.status.textContent = `locked to expert ${best.subject} lengths — drag line to move, ` +
    `endpoint to rotate, L to unlock`;
  el.status.className = "";
  draw();
}

// L: lock/unlock current lengths (any dataset) without changing them
function toggleLengthLock() {
  if (lockLen.index || lockLen.ring) {
    lockLen = { index: null, ring: null };
    el.status.textContent = "length lock off";
  } else {
    lockLen = { index: dist(points.index_base, points.index_tip),
                ring: dist(points.ring_base, points.ring_tip) };
    el.status.textContent = "lengths locked — drag line to move, endpoint to rotate";
  }
  el.status.className = "";
  draw();
}

resize();
loadWorklist();
