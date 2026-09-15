//  HandStudyApp.swift
//  Hand Study — a capture client for a 2D:4D field study.
//
//  Measurement, review and admin live on the server. This app exists to take
//  photographs and get them there intact, under conditions the server will
//  never see: a corridor, one hand, poor wifi, and a child who will not be
//  photographed twice.

import os
import SwiftUI

@main
struct HandStudyApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            RootView(model: model)
                .task {
                    let args = ProcessInfo.processInfo.arguments
                    if args.contains("-selftest") { await SelfTest.run(model) }
                    // Jump straight to a screen, with a representative capture
                    // behind it. Used for screenshots and for walking someone
                    // through the app without a child in front of the lens.
                    if let i = args.firstIndex(of: "-screen"), i + 1 < args.count {
                        model.demoState(args[i + 1])
                    }
                }
        }
    }
}

/// A launch-argument smoke test, so the capture path can be exercised on a
/// device without a person tapping through it. It asserts rather than narrates:
/// a self-test that cannot fail is decoration.
enum SelfTest {
    @MainActor
    static func run(_ model: AppModel) async {
        func say(_ s: String) { slog("SELFTEST \(s)") }
        try? await Task.sleep(nanoseconds: 2_000_000_000)

        say("begin simulated=\(model.isSimulated) version=\(Config.appVersion)")
        await model.loadSchema()
        say("schema_live=\(model.schemaLive) tags=\(model.schema.entryTags.map { $0.0 }.joined(separator: ","))")

        // Scoring must actually order frames, or selection is decoration.
        let fake = (0..<6).map { i -> ScoredFrame in
            ScoredFrame(index: i,
                        points: [[0.4, 0.5], [0.4, 0.3], [0.6, 0.5], [0.6, 0.28]],
                        inferMS: 5,
                        metrics: ImageMetrics(sharpness: Double(40 + i * 60),
                                              meanLuma: 130, clippedFraction: 0.001),
                        geometry: ["hand_fraction": 0.255])
        }
        let scored = FrameSelector.score(fake)
        let order = FrameSelector.uploadOrder(scored)
        if order.first?.index == 5 && order.last?.index == 0 {
            say("selection ok best=\(order.first!.index) worst=\(order.last!.index)")
        } else {
            say("FAIL selection put \(order.map { $0.index }) in that order")
        }

        // The guidance engine must refuse a bad hand, not merely colour it.
        let bad = GuidanceEngine.evaluate(tilt: 45, distanceMM: 250,
                                          handFraction: 0.25, sharpness: 300)
        if GuidanceEngine.blocking(bad).isEmpty {
            say("FAIL guidance allowed a 45 degree tilt")
        } else {
            say("guidance blocks tilt ok")
        }

        let good = GuidanceEngine.evaluate(tilt: 5, distanceMM: 250,
                                           handFraction: 0.255,
                                           sharpness: Predictor.liveBlurFloor * 3)
        say(GuidanceEngine.blocking(good).isEmpty
            ? "guidance passes a good hand ok"
            : "FAIL guidance blocked a good hand")

        // The sharpness scale has to stay comparable to the server's
        // cv2.Laplacian variance, because the 60.0 floor is shared. A flat field
        // must read ~0 and a hard checkerboard must read far above the floor;
        // if the vectorised path ever drifts, this is what catches it.
        if let p = try? Predictor() {
            let tw = 1440, th = 1920      // a real capture frame size
            func buffer(checker: Bool) -> CVPixelBuffer? {
                var pb: CVPixelBuffer?
                CVPixelBufferCreate(nil, tw, th, kCVPixelFormatType_32BGRA,
                                    [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &pb)
                guard let pb else { return nil }
                CVPixelBufferLockBaseAddress(pb, [])
                if let base = CVPixelBufferGetBaseAddress(pb) {
                    let stride = CVPixelBufferGetBytesPerRow(pb)
                    let px = base.assumingMemoryBound(to: UInt8.self)
                    for y in 0..<th {
                        for x in 0..<tw {
                            let v: UInt8 = checker ? (((x / 8) + (y / 8)) % 2 == 0 ? 0 : 255) : 128
                            let o = y * stride + x * 4
                            px[o] = v; px[o + 1] = v; px[o + 2] = v; px[o + 3] = 255
                        }
                    }
                }
                CVPixelBufferUnlockBaseAddress(pb, [])
                return pb
            }
            if let flat = buffer(checker: false), let edges = buffer(checker: true) {
                let mf = p.metrics(for: flat), me = p.metrics(for: edges)
                say(String(format: "sharpness live flat=%.1f edges=%.1f floor=%.0f",
                           mf.sharpness, me.sharpness, Predictor.liveBlurFloor))
                if mf.sharpness < Predictor.liveBlurFloor && me.sharpness > Predictor.liveBlurFloor * 4 {
                    say("live sharpness scale ok")
                } else {
                    say("FAIL live sharpness scale moved")
                }

                // The bug this exists to catch: the two measures are on
                // DIFFERENT scales, and the first real capture read 184 on the
                // live scale against 23.6 at server scale -- sharp by one
                // measure, blurry by the other. They must stay far apart, and
                // each must be judged against its own floor.
                let sf = p.metrics(for: edges, serverScale: true)
                say(String(format: "sharpness server-scale edges=%.1f floor=%.0f",
                           sf.sharpness, Predictor.blurFloor))
                let ratio = me.sharpness / max(sf.sharpness, 0.001)
                say(String(format: "live/server scale ratio=%.1f (liveBlurFloor/blurFloor=%.1f)",
                           ratio, Predictor.liveBlurFloor / Predictor.blurFloor))
                if sf.sharpness <= Predictor.blurFloor {
                    say("FAIL a hard checkerboard scored blurry at server scale")
                } else if ratio < 2 {
                    say(String(format: "FAIL the two scales collapsed (ratio %.1f); the live floor is no longer calibrated", ratio))
                } else {
                    say("server-scale sharpness ok, scales distinct")
                }
            }
        }

        say("pending_on_disk=\(CaptureStore.pending().count) incomplete=\(CaptureStore.incomplete().count)")
        // Full path, unattended: fetch a subject, run a real capture through
        // the model on the simulator's palm frame, score it, write it to disk
        // and upload it frame by frame. Nothing below is mocked, so a break
        // anywhere between the camera and the server shows up here.
        if ProcessInfo.processInfo.arguments.contains("-e2e") {
            model.start()
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            await model.fetchSubject()
            guard !model.subject.isEmpty else { say("FAIL no subject issued"); return }
            say("subject=\(model.subject)")
            // Test data, not study data: the server keeps these stores apart.
            model.fields["study"] = "no"

            var waited = 0
            while !model.canCapture && waited < 40 {
                try? await Task.sleep(nanoseconds: 250_000_000); waited += 1
            }
            say("guidance=\(model.guidance.map { "\($0.id):\($0.level)" }.joined(separator: " "))")
            guard model.canCapture else { say("FAIL capture never became available"); return }

            model.beginCapture()
            waited = 0
            while model.capturing && waited < 200 {
                try? await Task.sleep(nanoseconds: 100_000_000); waited += 1
            }
            guard let r = model.lastResult else { say("FAIL capture did not complete"); return }
            say(String(format: "captured frames=%d ratio=%.4f sd=%.4f quality=%@",
                       r.frames, r.median, r.sd, r.quality.level))

            waited = 0
            while (model.queue.pendingCount > 0 || model.queue.uploading) && waited < 600 {
                try? await Task.sleep(nanoseconds: 500_000_000); waited += 1
            }
            if model.queue.pendingCount == 0 {
                say("upload ok, queue empty")
                if let j = model.queue.lastResultJSON, j.contains("median_ratio") {
                    say("server measured it and returned a session")
                } else {
                    say("FAIL server reply had no measurement")
                }
            } else {
                say("FAIL \(model.queue.pendingCount) left in the queue: \(model.queue.lastError ?? "no error given")")
            }
            model.stop()
        }

        say("done")
    }
}
