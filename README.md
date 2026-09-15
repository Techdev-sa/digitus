# Digitus

A 2D:4D measurement instrument. Digitus measures the ratio between the index
and ring finger from a phone photograph of an open palm, and reports it only
with the spread it was measured to.

It measures a ratio. It does not screen for, diagnose or predict any condition.

Project page: https://hand.tdev.sa · KSCDR AI Hackathon for People with
Disabilities, team KSCDR_Hackathon_039.

## What is here

```
ios/      The capture app (Swift, SwiftUI, Core ML). TestFlight name: Hand Study.
server/   The measurement server and research console (Python, Flask, PyTorch).
```

### ios/

- Takes a 24-frame burst of the open palm with live guidance: framing, focus,
  and on devices with LiDAR, tilt and distance.
- `CreaseNet` (MobileNetV3-small heatmap model, 1024 × 1024, float16) places four
  landmarks on the phone's Neural Engine: base and tip of the index and ring
  fingers, bases on the most-proximal crease.
- Every frame is saved on the device before upload; the five best frames upload
  first, the rest follow one request at a time and resume after interruption.
- Shows the ratio, its spread and a quality status together, never the number alone.

Open `ios/HandStudy.xcodeproj` in Xcode. To talk to a server, set
`deviceToken` in `ios/HandStudy/Engine.swift` to a token that server accepts —
it is a placeholder in this repository. Simulator builds render in software and
can replay a folder of JPEG frames as the camera: launch with
`SIMCTL_CHILD_HANDSTUDY_SIM_FRAMES=/path/to/frames`.

### server/

`server/app/capture_site.py` is the whole server: the upload API, the
measurement, and the research console (dashboard, subjects, landmark review,
blind model comparison, administration, method). It re-runs the same model in
PyTorch on the exact pixels it receives; the phone proposes, the server decides.

- Quality gates: no number when SD > 0.08 or fewer than 3 frames are usable;
  flagged when SD > 0.04; subgroups under n = 5 are withheld.
- `/` is the public project page; `/capture` is the browser capture page.
- Demo mode ("Try the demo console" on the login page) serves the console with
  the synthetic records in `server/fixtures/` only.

Run it:

```bash
cd server
pip install -r requirements.txt
python3 app/capture_site.py --ckpt /path/to/checkpoint.pt --input-size 1024 \
  --port 5055 --captures /path/to/captures
```

It needs, and this repository deliberately does not contain:

- the PyTorch checkpoint (`synth_heatmap_anisofixed.pt` in production),
- `agents/auth.json`, holding the console password hash, the signing secret and
  the accepted device tokens (see the auth section of `capture_site.py`),
- any captures.

## What is deliberately not here

- Study data. Captures are photographs of children and never leave the study server.
- Credentials, tokens and deployment configuration.
- Training data, the server checkpoint and the research scripts used to build them.

## Status

Built and working: ratio measurement, end to end, on iPhone and server.
Not yet validated: agreement with callipers on the same hands, and any
screening use.
