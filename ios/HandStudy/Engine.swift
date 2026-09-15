//  Engine.swift
//  Config, the server schema, capture geometry, frame scoring and the
//  visit state machine.
//
//  A VISIT is one child. A CAPTURE is one hand. The operator is told what to do
//  next at every point rather than having to remember the protocol, because the
//  second hand cannot be collected once the child has left the room.
//
//  This is a measurement instrument, not a screening tool. It reports a ratio
//  and the landmarks it came from. There is deliberately no threshold, no flag
//  and no verdict anywhere in this file.

import CoreVideo
import Foundation
import os
import simd
import SwiftUI
import UIKit
import Vision

enum Config {
    static let base = URL(string: "https://hand.tdev.sa")!
    static var sessionEndpoint: URL { base.appendingPathComponent("api/session") }
    static var subjectEndpoint: URL { base.appendingPathComponent("api/subject/next") }
    static var schemaEndpoint: URL { base.appendingPathComponent("api/schema") }

    // v2 upload: one frame per request. A 25 MB body that fails at 90% on school
    // wifi costs the whole capture; a frame that fails costs a retry.
    static var captureBeginEndpoint: URL { base.appendingPathComponent("api/v2/capture/begin") }
    static func frameEndpoint(_ cid: String) -> URL {
        base.appendingPathComponent("api/v2/capture/\(cid)/frame")
    }
    static func commitEndpoint(_ cid: String) -> URL {
        base.appendingPathComponent("api/v2/capture/\(cid)/commit")
    }
    static func statusEndpoint(_ cid: String) -> URL {
        base.appendingPathComponent("api/v2/capture/\(cid)/status")
    }

    static let framesPerCapture = 24
    /// How many of those go up first. The rest follow in the background, so the
    /// operator sees a result in seconds without any frame being discarded.
    static let priorityFrames = 5
    /// Read from the bundle rather than kept in a second place. A version
    /// string that disagrees with the build is worse than none: it ends up in
    /// the capture metadata, where it is the only record of what took the photo.
    static let appVersion: String = {
        let v = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "?"
        let b = Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?"
        return "\(v) (\(b))"
    }()
    /// Fraction of the long side the hand should occupy. Measured: framing
    /// alone moves ratio error by about 15%, and the training set sits here.
    static let targetHandFraction: Float = 0.255

    /// Ad-hoc deployment, so the token ships in the binary. That is weak, and
    /// the mitigation is that it is never written to a log or a crash report.
    private static let deviceToken = "REPLACE_WITH_DEVICE_TOKEN"

    static func authorize(_ req: inout URLRequest) {
        req.setValue(deviceToken, forHTTPHeaderField: "X-Device-Token")
    }
}

// MARK: - Server schema

struct FieldRange: Codable { let min: Double; let max: Double }

/// The server's own description of what a capture may carry. Read at launch so
/// a new field can be added server-side without shipping a new build.
struct Schema: Codable {
    var crease_convention: String?
    var model_checkpoint: String?
    var identifiers: [String]?
    var json_blobs: [String]?
    var tags: [String: [String]]?
    var subject_text: [String: String]?
    var subject_numeric: [String: FieldRange]?
    var geometry_numeric: [String: FieldRange]?

    /// Used only when the server cannot be reached at launch, so that a morning
    /// offline is still a morning of usable captures.
    static let fallback = Schema(
        crease_convention: nil, model_checkpoint: nil,
        identifiers: ["subject", "source", "store", "app_version"],
        json_blobs: ["ar", "device_info"],
        tags: ["sex": ["male", "female"], "autistic": ["yes", "no"],
               "hand": ["right", "left"], "study": ["yes", "no"],
               "dominant_hand": ["right", "left", "ambidextrous"],
               "nail_overhang": ["yes", "no"]],
        subject_text: [:], subject_numeric: [:], geometry_numeric: [:])

    /// Tags the operator sets. `hand` is driven by the capture flow and `study`
    /// has its own toggle, so neither belongs in the generated entry form.
    var entryTags: [(String, [String])] {
        (tags ?? [:]).filter { $0.key != "hand" && $0.key != "study" }
            .sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
    }
    var entryText: [(String, String)] {
        (subject_text ?? [:]).sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
    }
    var entryNumeric: [(String, FieldRange)] {
        (subject_numeric ?? [:]).sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
    }

    /// Empty string means "not set" and is simply not sent.
    func validate(_ key: String, _ value: String) -> String? {
        if value.isEmpty { return nil }
        if let opts = tags?[key], !opts.contains(value) {
            return "\(key) must be one of \(opts.joined(separator: ", "))"
        }
        if let r = subject_numeric?[key] {
            guard let d = Double(value) else { return "\(key) must be a number" }
            if d < r.min || d > r.max {
                return "\(key) must be between \(fmt(r.min)) and \(fmt(r.max))"
            }
        }
        if let pattern = subject_text?[key] {
            guard let re = try? NSRegularExpression(pattern: "^(?:\(pattern))$") else { return nil }
            let range = NSRange(value.startIndex..<value.endIndex, in: value)
            if re.firstMatch(in: value, range: range) == nil {
                return "\(key) has characters the server will not accept"
            }
        }
        return nil
    }

    private func fmt(_ d: Double) -> String {
        d == d.rounded() ? String(Int(d)) : String(d)
    }
}

enum Step: Equatable {
    case needSubject
    case capture(hand: String, repeatIndex: Int)
    case visitDone
}

/// print() writes to stdout, which the unified log does not capture and which
/// `simctl launch` only forwards with a pty. Routing the self-test through
/// os.Logger makes it readable with `log show` after the fact, which is what a
/// headless run actually needs. .notice rather than .info because .info is not
/// persisted by default.
@MainActor
func slog(_ line: String) {
    Logger(subsystem: "sa.tdev.handstudy", category: "selftest")
        .notice("\(line, privacy: .public)")
    // Also to stdout, so `devicectl process launch --console` can read it
    // without root on the Mac. Reading the unified log off a device needs
    // sudo; a field diagnostic nobody can retrieve is not a diagnostic.
    print("HANDSTUDY \(line)")
    fflush(stdout)
}

// MARK: - Geometry

/// Per-capture geometry. Tilt is the reason this exists: a photograph measures
/// PROJECTED finger length, so a hand tilted away from the sensor reads short,
/// and the error does not cancel between index and ring. Recording the tilt
/// makes that confound measurable instead of invisible.
struct GeometryResult {
    var send: [String: Double] = [:]      // validated, in range
    var dropped: [String: Double] = [:]   // computed but refused locally
    var note: String = ""
}

enum Geometry {

    /// `pts` are the four landmarks normalised to the upright portrait frame,
    /// in the order index_base, index_tip, ring_base, ring_tip.
    static func compute(pts: [[Float]], depth: DepthSample?,
                        schema: Schema?) -> GeometryResult {
        var raw: [String: Double] = [:]
        guard pts.count == 4 else { return GeometryResult() }

        // 2D, always available -- no depth needed.
        let xs = pts.map { Double($0[0]) }, ys = pts.map { Double($0[1]) }
        raw["hand_fraction"] = max((xs.max() ?? 0) - (xs.min() ?? 0),
                                   (ys.max() ?? 0) - (ys.min() ?? 0))

        // Roll is measured in the upright frame the operator sees: 0 when the
        // fingers point straight up, positive when the hand leans right.
        let baseMid = (x: (xs[0] + xs[2]) / 2, y: (ys[0] + ys[2]) / 2)
        let tipMid  = (x: (xs[1] + xs[3]) / 2, y: (ys[1] + ys[3]) / 2)
        let ax = tipMid.x - baseMid.x, ay = tipMid.y - baseMid.y
        if ax != 0 || ay != 0 {
            raw["roll_deg"] = atan2(ax, -ay) * 180 / .pi
        }

        // 3D, only with a depth map.
        var note = "2d-only"
        if let d = depth {
            note = "lidar"
            var p3: [SIMD3<Double>] = []
            var confs: [Double] = []
            for p in pts {
                // The colour buffer handed to the model is the upright portrait
                // rotation of ARKit's landscape frame; the depth map and the
                // intrinsics are still in that landscape frame, so the point
                // has to be rotated back before it is looked up.
                let lx = Double(p[1])
                let ly = 1 - Double(p[0])
                guard let z = d.sampleDepth(nx: lx, ny: ly), z > 0.05, z < 2.5 else {
                    p3 = []; break
                }
                confs.append(d.sampleConfidence(nx: lx, ny: ly))
                let px = lx * Double(d.colorWidth)
                let py = ly * Double(d.colorHeight)
                p3.append(SIMD3<Double>((px - d.cx) * z / d.fx,
                                        (py - d.cy) * z / d.fy, z))
            }
            if p3.count == 4 {
                let v1 = p3[1] - p3[0], v2 = p3[2] - p3[0]
                let n = simd_cross(v1, v2)
                if simd_length(n) > 0 {
                    let nz = abs(simd_normalize(n).z)
                    raw["tilt_deg"] = acos(Swift.min(1, nz)) * 180 / .pi
                }
                raw["distance_mm"] = (p3.map { $0.z }.reduce(0, +) / 4) * 1000
                if !confs.isEmpty {
                    raw["depth_confidence"] = confs.reduce(0, +) / Double(confs.count)
                }
            } else {
                note = "lidar-no-return"
            }
        }

        // Range-check locally. The server rejects an out-of-range value outright
        // and would take the whole capture down with it, so a bad number is held
        // back -- but recorded in the `ar` blob, never silently discarded.
        var out = GeometryResult()
        out.note = note
        let ranges = schema?.geometry_numeric ?? [:]
        for (k, v) in raw {
            guard v.isFinite else { out.dropped[k] = v; continue }
            if let r = ranges[k], v < r.min || v > r.max { out.dropped[k] = v; continue }
            out.send[k] = v
        }
        return out
    }
}


// MARK: - Frame scoring

/// One captured frame and everything known about it at capture time.
struct ScoredFrame {
    var index: Int
    var points: [[Float]]        // normalised, exactly as the model emitted
    var inferMS: Double
    var metrics: ImageMetrics
    var geometry: [String: Double]

    // Sub-scores, each 0-1, filled in once the whole burst is known.
    var sharpScore: Double = 0
    var stabilityScore: Double = 0
    var framingScore: Double = 0
    var exposureScore: Double = 0
    var total: Double = 0

    var ratio: Float {
        let i = Predictor.dist(points[1], points[0])
        let r = Predictor.dist(points[3], points[2])
        return r > 0 ? i / r : 0
    }
}

/// Picks which frames are worth uploading first.
///
/// Four properties decide it, and they are deliberately not collapsed into one
/// number until the end, because they fail in different ways and a frame can be
/// perfect on three and useless on the fourth:
///
///   sharpness  - a blurred crease has no edge to measure. Scored against the
///                same Laplacian floor the server uses, so the phone and the
///                server agree about what blurry means.
///   stability  - how far this frame's landmarks sit from the burst median. A
///                frame where the model wobbled is not a measurement of the
///                hand, it is a measurement of the model.
///   framing    - hand size against the 0.255 target, and no landmark near an
///                edge. More pixels across the creases, nothing clipped.
///   exposure   - blown highlights and crushed shadows both destroy the crease
///                contrast the model is looking for.
///
/// Selection decides UPLOAD ORDER, never what is kept. Every frame is written
/// to disk and every frame is eventually sent: a heuristic that discarded a
/// frame would be discarding something no one can photograph again.
enum FrameSelector {

    static func score(_ frames: [ScoredFrame]) -> [ScoredFrame] {
        guard !frames.isEmpty else { return [] }
        var out = frames

        // Median landmark position across the burst, per point. Median rather
        // than mean so one lost-hand frame cannot drag the reference toward
        // itself and make the good frames look unstable.
        var medians: [[Float]] = []
        for p in 0..<4 {
            let xs = frames.map { $0.points[p][0] }.sorted()
            let ys = frames.map { $0.points[p][1] }.sorted()
            medians.append([xs[xs.count / 2], ys[ys.count / 2]])
        }
        let deviations = frames.map { f in
            (0..<4).reduce(0.0) { $0 + Double(Predictor.dist(f.points[$1], medians[$1])) } / 4
        }
        let worstDeviation = max(deviations.max() ?? 0, 0.0001)

        for i in out.indices {
            let f = out[i]

            // Sharpness: the floor is the zero point, and 6x the floor is
            // treated as saturated -- past that, more variance is texture and
            // noise rather than a better measurement.
            let s = (f.metrics.sharpness - Predictor.blurFloor) / (Predictor.blurFloor * 5)
            out[i].sharpScore = min(1, max(0, s))

            // Stability: best frame in the burst scores 1.
            out[i].stabilityScore = 1 - (deviations[i] / worstDeviation)

            // Framing: distance from the target fraction, plus a hard penalty
            // for any landmark within 4% of an edge, where the finger may
            // already be cut off.
            let frac = f.geometry["hand_fraction"] ?? 0
            let closeness = 1 - min(1, abs(frac - Double(Config.targetHandFraction)) / 0.2)
            let nearEdge = f.points.contains { p in
                p[0] < 0.04 || p[0] > 0.96 || p[1] < 0.04 || p[1] > 0.96
            }
            out[i].framingScore = nearEdge ? closeness * 0.25 : closeness

            // Exposure: mid-grey is ideal, and clipped pixels are punished
            // separately because a frame can have a fine mean and a blown palm.
            let lumaOff = abs(f.metrics.meanLuma - 128) / 128
            let clip = min(1, f.metrics.clippedFraction * 12)
            out[i].exposureScore = max(0, (1 - lumaOff) * (1 - clip))

            // Sharpness and stability carry the most weight because they are
            // the two that make a measurement wrong rather than merely worse.
            out[i].total = 0.35 * out[i].sharpScore
                + 0.30 * out[i].stabilityScore
                + 0.20 * out[i].framingScore
                + 0.15 * out[i].exposureScore
        }
        return out
    }

    /// Upload order: best first. Ties broken by frame index so the order is
    /// reproducible for a given burst.
    static func uploadOrder(_ scored: [ScoredFrame]) -> [ScoredFrame] {
        scored.sorted {
            $0.total == $1.total ? $0.index < $1.index : $0.total > $1.total
        }
    }
}

// MARK: - Live capture guidance

/// What the operator is told, right now, about how the hand is sitting.
/// Each case carries the fix, not just the fault -- "move closer" rather than
/// "distance out of range" -- because the person holding the phone is also
/// holding a child's hand and has no attention to spare for interpretation.
enum GuidanceLevel: Int, Comparable {
    case good = 0, warn = 1, bad = 2
    static func < (a: GuidanceLevel, b: GuidanceLevel) -> Bool { a.rawValue < b.rawValue }
}

struct Guidance: Identifiable {
    var id: String
    var label: String        // what it is measuring
    var advice: String       // what to do about it
    var level: GuidanceLevel
    var detail: String       // the number, for anyone who wants it
}

enum GuidanceEngine {

    static func evaluate(tilt: Double?, distanceMM: Double?,
                         handFraction: Double?, sharpness: Double?) -> [Guidance] {
        var out: [Guidance] = []

        // Tilt first: it is the dominant error source, because a photograph
        // measures projected length and the shortening does not cancel between
        // index and ring.
        if let t = tilt {
            let lvl: GuidanceLevel = t <= 12 ? .good : (t <= 22 ? .warn : .bad)
            out.append(Guidance(id: "tilt", label: "Tilt",
                                advice: lvl == .good ? "Hand is flat"
                                    : (lvl == .warn ? "Flatten the hand a little" : "Lay the hand flat to the lens"),
                                level: lvl, detail: String(format: "%.0f°", t)))
        }
        if let d = distanceMM {
            let lvl: GuidanceLevel = (d >= 180 && d <= 400) ? .good
                : ((d >= 140 && d < 180) || (d > 400 && d <= 500) ? .warn : .bad)
            let advice = d < 180 ? "Move back a little" : (d > 400 ? "Move closer" : "Distance is good")
            out.append(Guidance(id: "distance", label: "Distance",
                                advice: lvl == .good ? "Distance is good" : advice,
                                level: lvl, detail: String(format: "%.0f mm", d)))
        }
        if let f = handFraction {
            let target = Double(Config.targetHandFraction)
            let off = abs(f - target)
            let lvl: GuidanceLevel = off <= 0.05 ? .good : (off <= 0.10 ? .warn : .bad)
            out.append(Guidance(id: "framing", label: "Framing",
                                advice: lvl == .good ? "Hand fills the guide"
                                    : (f < target ? "Bring the hand closer to the guide" : "Pull back, the hand overfills"),
                                level: lvl, detail: String(format: "%.0f%%", f * 100)))
        }
        if let s = sharpness {
            // The live figure is on the 256-square scale, so it is judged
            // against liveBlurFloor, never blurFloor: they are different
            // measures, and comparing this one to the server's floor calls a
            // frame sharp that the server then marks blurry. liveBlurFloor is
            // calibrated to real captures of steady hands (see Predictor).
            let lvl: GuidanceLevel = s >= Predictor.liveBlurFloor * 2 ? .good
                : (s >= Predictor.liveBlurFloor ? .warn : .bad)
            out.append(Guidance(id: "focus", label: "Focus",
                                advice: lvl == .good ? "Sharp"
                                    : (lvl == .warn ? "Hold steadier" : "Too blurred to measure"),
                                level: lvl, detail: String(format: "%.0f", s)))
        }
        return out
    }

    /// Capture is refused while anything is `.bad`. A capture taken through a
    /// red state produces a number nobody should trust, and the child would
    /// have to be photographed again to replace it.
    static func blocking(_ g: [Guidance]) -> [Guidance] { g.filter { $0.level == .bad } }
}

// MARK: - App model

enum Screen: Hashable { case home, subject, capture, result, queue, settings }

@MainActor
final class AppModel: ObservableObject {

    // Navigation
    @Published var screen: Screen = .home

    // Server-described form
    @Published var schema = Schema.fallback
    @Published var schemaLive = false
    @Published var fields: [String: String] = [:]
    @Published var subject: String = ""
    @Published var hand: String = "right"

    // Live capture state
    @Published var points: [[Float]] = []
    @Published var guidance: [Guidance] = []
    @Published var inferMS: Double = 0
    @Published var frameW = 0
    @Published var frameH = 0
    /// Downscaled copy of the current frame for the preview. Rendered on the
    /// capture queue: handing a full 1080p buffer to SwiftUI 30 times a second
    /// is what makes a camera UI stutter.
    @Published var preview: CGImage?
    @Published var capturing = false
    @Published var captured = 0
    @Published var liveRatio: Float = 0

    // Result of the last capture
    @Published var lastResult: CaptureOutcome?
    @Published var status: String = ""

    // Upload
    let queue = UploadQueue()

    private var predictor: Predictor?
    private var source: FrameSource
    private let lock = NSLock()
    private var burst: [ScoredFrame] = []
    private var wantCapture = false
    private var captureDir: URL?
    private var pendingJPEG: Data?
    private var frameCounter = 0
    private var lastDepth: DepthSample?
    private var lastLiveSharpness: Double?
    #if targetEnvironment(simulator)
    // Software rendering on the simulator: its Metal host crashes on these
    // IOSurface renders (see Camera.swift). Device builds are unchanged.
    private let jpegContext = CIContext(options: [.cacheIntermediates: false, .useSoftwareRenderer: true])
    private let previewContext = CIContext(options: [.cacheIntermediates: false, .useSoftwareRenderer: true])
    #else
    private let jpegContext = CIContext(options: [.cacheIntermediates: false])
    private let previewContext = CIContext(options: [.cacheIntermediates: false])
    #endif
    private let log = Logger(subsystem: "sa.tdev.handstudy", category: "engine")

    var isSimulated: Bool { source.isSimulated }

    init() {
        // Camera.swift already knows how to choose: LiDAR where the hardware
        // has it, plain capture otherwise, a still frame in the simulator.
        source = makeFrameSource()
        do { predictor = try Predictor() }
        catch { status = "Model failed to load: \(error.localizedDescription)" }
        source.onFrame = { [weak self] pb, depth in self?.handle(pb, depth) }
        source.configure()
        Task { await loadSchema() }
    }

    func start() { source.start() }
    func stop() { source.stop() }

    // MARK: Frame path

    /// Runs on the capture queue. Everything expensive happens here; only the
    /// small published values hop to the main actor, because a 30 fps hop with
    /// a JPEG attached is what makes a capture UI stutter.
    private func handle(_ pb: CVPixelBuffer, _ depth: DepthSample?) {
        guard let predictor else { return }
        frameCounter &+= 1
        lastDepth = depth

        guard let p = try? predictor.run(on: pb) else { return }

        // Image metrics are only needed for guidance and for scoring a captured
        // frame. At 30 fps the guidance does not need every frame, so it is
        // sampled -- this is the difference between a warm phone and a hot one.
        var shouldStore = false
        lock.lock()
        if wantCapture && burst.count < Config.framesPerCapture { shouldStore = true }
        lock.unlock()

        // Image metrics are needed for guidance and for scoring a captured
        // frame. At 30 fps the guidance does not need every frame, so it is
        // sampled -- the difference between a warm phone and a hot one.
        //
        // A frame that is being KEPT is scored the way the server will score it
        // (long side capped at 2048), so selection and the server agree. The
        // live figure stays on the cheap 256 square and is judged against
        // liveBlurFloor, calibrated separately to real captures on that scale.
        let wantMetrics = shouldStore || frameCounter % 4 == 0
        let m = wantMetrics ? predictor.metrics(for: pb, serverScale: shouldStore) : nil
        // A frame being kept is scored at SERVER scale, which is not the live
        // figure the focus row is judged on. Showing it against liveBlurFloor
        // turned the row red mid-burst ("too blurred", 26) on a hand that had
        // just passed. During the burst the row keeps the last live reading.
        if let m, !shouldStore { lastLiveSharpness = m.sharpness }
        let focusSharpness = shouldStore ? lastLiveSharpness : m?.sharpness
        let geo = Geometry.compute(pts: p.onFrame, depth: depth, schema: schema)

        var stored: ScoredFrame?
        if shouldStore, let m {
            // The original camera frame at maximum quality. The server re-runs
            // on these exact pixels, so compression loss would masquerade as
            // model disagreement.
            if let jpeg = jpegContext.jpegRepresentation(
                of: CIImage(cvPixelBuffer: pb),
                colorSpace: CGColorSpaceCreateDeviceRGB(),
                options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 1.0]) {
                stored = ScoredFrame(index: 0, points: p.raw,
                                     inferMS: p.inferMS, metrics: m, geometry: geo.send)
                pendingJPEG = jpeg
            }
        }

        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)

        // Preview at a fraction of the sensor size; the operator cannot see
        // 1080p on a 6-inch screen and the conversion is the expensive part.
        var previewImage: CGImage?
        if frameCounter % 2 == 0 {
            let target: CGFloat = 640
            let k = target / CGFloat(max(w, h))
            let small = CIImage(cvPixelBuffer: pb)
                .transformed(by: CGAffineTransform(scaleX: k, y: k))
            previewImage = previewContext.createCGImage(small, from: small.extent)
        }
        var finished: [ScoredFrame]?
        if var s = stored {
            lock.lock()
            s.index = burst.count
            // On disk now, not at the end of the burst.
            if let dir = captureDir, let jpeg = pendingJPEG {
                CaptureStore.append(dir, index: s.index, jpeg: jpeg)
            }
            pendingJPEG = nil
            burst.append(s)
            if burst.count >= Config.framesPerCapture {
                wantCapture = false
                finished = burst
            }
            let n = burst.count
            lock.unlock()
            Task { @MainActor in self.captured = n }
        }

        Task { @MainActor in
            self.points = p.onFrame
            self.inferMS = p.inferMS
            self.liveRatio = p.ratio
            self.frameW = w; self.frameH = h
            if let previewImage { self.preview = previewImage }
            if let m {
                self.guidance = GuidanceEngine.evaluate(
                    tilt: geo.send["tilt_deg"] ?? geo.dropped["tilt_deg"],
                    distanceMM: geo.send["distance_mm"] ?? geo.dropped["distance_mm"],
                    handFraction: geo.send["hand_fraction"],
                    sharpness: focusSharpness)
            }
            if let finished { await self.finishCapture(finished) }
        }
    }

    // MARK: Capture

    var canCapture: Bool {
        !capturing && !subject.isEmpty && GuidanceEngine.blocking(guidance).isEmpty
    }

    func beginCapture() {
        guard canCapture else { return }
        guard let dir = CaptureStore.begin() else {
            status = "No room on this device to start a capture"
            return
        }
        lock.lock()
        captureDir = dir
        burst.removeAll(keepingCapacity: true)
        wantCapture = true
        lock.unlock()
        captured = 0
        capturing = true
        status = "Hold still"
        source.lockExposureAndFocus()
    }

    private func finishCapture(_ frames: [ScoredFrame]) async {
        capturing = false
        source.unlock()
        let scored = FrameSelector.score(frames)
        let ordered = FrameSelector.uploadOrder(scored)

        let ratios = scored.map { Double($0.ratio) }.sorted()
        let median = ratios.isEmpty ? 0 : ratios[ratios.count / 2]
        let sd: Double = {
            guard ratios.count > 1 else { return 0 }
            let m = ratios.reduce(0, +) / Double(ratios.count)
            return (ratios.reduce(0) { $0 + ($1 - m) * ($1 - m) } / Double(ratios.count - 1)).squareRoot()
        }()

        let meta = CaptureMeta(
            subject: subject, hand: hand,
            capturedAt: Date(), appVersion: Config.appVersion,
            deviceName: UIDevice.current.model,
            frameCount: scored.count,
            medianRatio: Float(median), sdRatio: Float(sd),
            fields: fields,
            geometry: scored.first?.geometry ?? [:],
            uploadOrder: ordered.map { $0.index },
            scores: ordered.reduce(into: [:]) { acc, f in
                acc["\(f.index)"] = ["sharp": f.sharpScore, "stability": f.stabilityScore,
                                     "framing": f.framingScore, "exposure": f.exposureScore,
                                     "total": f.total]
            })

        // The frames are already on disk; this seals the capture.
        guard let dir = captureDir, CaptureStore.finish(dir, meta: meta) != nil else {
            status = "Could not save the capture to this device"
            captureDir = nil
            return
        }
        captureDir = nil
        if true {
            lastResult = CaptureOutcome(median: median, sd: sd, frames: scored.count,
                                        best: Array(ordered.prefix(Config.priorityFrames)))
            screen = .result
            status = ""
            queue.kick()
        } else {
            status = "Could not save the capture to this device"
        }
    }

    // MARK: Server

    func loadSchema() async {
        var req = URLRequest(url: Config.schemaEndpoint)
        req.timeoutInterval = 20
        if let (d, _) = try? await URLSession.shared.data(for: req),
           let s = try? JSONDecoder().decode(Schema.self, from: d) {
            schema = s
            schemaLive = true
        }
    }

    /// The subject number comes from the server. Inventing one locally collides
    /// the moment a second device is in the field, and is unrecoverable after.
    func fetchSubject() async {
        var req = URLRequest(url: Config.subjectEndpoint)
        req.httpMethod = "POST"
        req.timeoutInterval = 20
        Config.authorize(&req)
        guard let (d, resp) = try? await URLSession.shared.data(for: req),
              (resp as? HTTPURLResponse)?.statusCode == 200,
              let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
              let s = j["subject"] as? String else {
            status = "Could not get a subject number from the server"
            return
        }
        subject = s
        status = ""
    }

    /// Populates a plausible capture and opens one screen. Demo only: it never
    /// writes to disk and never uploads, so it cannot contaminate the study.
    func demoState(_ name: String) {
        subject = "S-0148"
        hand = "right"
        fields = ["sex": "female", "autistic": "no", "age": "9"]
        lastResult = CaptureOutcome(
            median: 0.968, sd: 0.011, frames: Config.framesPerCapture, best: [])
        guidance = GuidanceEngine.evaluate(tilt: 6, distanceMM: 260,
                                           handFraction: 0.251, sharpness: 180)
        switch name {
        case "subject": screen = .subject
        case "capture": screen = .capture
        case "result": screen = .result
        case "queue": screen = .queue
        case "settings": screen = .settings
        default: screen = .home
        }
    }

    func validateAll() -> String? {
        for (k, v) in fields where !v.isEmpty {
            if let e = schema.validate(k, v) { return "\(k): \(e)" }
        }
        return nil
    }
}

/// What the operator is shown after a capture. Deliberately carries the spread
/// and the frame count alongside the ratio: a number without them is a number
/// nobody can weigh.
struct CaptureOutcome {
    var median: Double
    var sd: Double
    var frames: Int
    var best: [ScoredFrame]

    /// Mirrors the server's own gates so the phone and the console agree.
    var quality: (level: String, reason: String) {
        if frames < 3 { return ("fail", "hand found in too few frames") }
        if sd > 0.08 { return ("fail", String(format: "readings disagree wildly (SD %.3f)", sd)) }
        if sd > 0.04 { return ("poor", String(format: "readings vary a lot (SD %.3f)", sd)) }
        return ("ok", "")
    }
}
