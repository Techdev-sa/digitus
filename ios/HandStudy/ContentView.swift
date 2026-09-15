//  ContentView.swift
//  The operator-facing surface: a design system, then the screens built from it.
//
//  Two rules run through all of it. A ratio is never shown without its spread
//  and its quality verdict beside it, at the same weight -- a lone number
//  invites a confidence the measurement has not earned. And every state that
//  blocks the operator says what to DO, not what is wrong, because the person
//  holding the phone is also holding a child's hand.

import SwiftUI
import UIKit

// MARK: - Design system

enum HS {
    /// Tokens from docs/APP_UI_SPEC.md, so the app and the web console are one
    /// system rather than two things that merely share a colour. Every token is
    /// defined for both appearances: a corridor by a window and a darkened room
    /// are both real conditions this is used in.
    private static func dyn(_ light: UInt32, _ dark: UInt32) -> Color {
        Color(UIColor { $0.userInterfaceStyle == .dark ? UIColor(hex: dark) : UIColor(hex: light) })
    }

    static let bg        = dyn(0xF5F7F6, 0x111816)
    static let surface   = dyn(0xFFFFFF, 0x18221F)
    static let raised    = dyn(0xEDF2F0, 0x202C28)
    static let inset     = dyn(0xE7EEEB, 0x111A17)
    static let ink       = dyn(0x182522, 0xEDF4F0)
    static let inkSoft   = dyn(0x40514D, 0xC5D1CC)
    static let muted     = dyn(0x667671, 0x9FAEA7)
    static let line      = dyn(0xD4DED9, 0x33423D)
    static let lineStrong = dyn(0xB8C8C2, 0x506058)
    static let accent    = dyn(0x006B63, 0x57BDB3)
    static let accentSoft = dyn(0xD9EEEA, 0x133C37)
    static let info      = dyn(0x23679D, 0x81BCE7)
    static let good      = dyn(0x167153, 0x78CEA6)
    static let goodSoft  = dyn(0xDCEFE6, 0x143B2D)
    static let warn      = dyn(0x9B5D08, 0xE7BD70)
    static let warnSoft  = dyn(0xF7EBD5, 0x493614)
    static let bad       = dyn(0xA1363A, 0xF09294)
    static let badSoft   = dyn(0xF7E0E1, 0x482126)
    /// The camera well stays dark in both appearances: a bright chrome around a
    /// live preview washes out the hand it is there to show.
    static let captureWell = dyn(0x101A18, 0x09100E)

    static func colour(_ l: GuidanceLevel) -> Color {
        switch l { case .good: return good; case .warn: return warn; case .bad: return bad }
    }
    static func soft(_ l: GuidanceLevel) -> Color {
        switch l { case .good: return goodSoft; case .warn: return warnSoft; case .bad: return badSoft }
    }
    /// Never colour alone: each state also carries a glyph and a word, for
    /// bright light, for colour blindness, and for a screenshot in a report.
    static func glyph(_ l: GuidanceLevel) -> String {
        switch l {
        case .good: return "checkmark.circle.fill"
        case .warn: return "exclamationmark.triangle.fill"
        case .bad: return "xmark.octagon.fill"
        }
    }

    static let radius: CGFloat = 14
    static let pad: CGFloat = 16
}

extension UIColor {
    convenience init(hex: UInt32) {
        self.init(red: CGFloat((hex >> 16) & 0xFF) / 255,
                  green: CGFloat((hex >> 8) & 0xFF) / 255,
                  blue: CGFloat(hex & 0xFF) / 255, alpha: 1)
    }
}

extension Text {
    func hsTitle() -> Text { font(.system(size: 26, weight: .semibold, design: .default)) }
    func hsLabel() -> Text { font(.system(size: 11, weight: .semibold)).kerning(0.6) }
    func hsBody() -> Text { font(.system(size: 15)) }
    func hsMono(_ size: CGFloat = 15) -> Text {
        font(.system(size: size, weight: .medium, design: .monospaced))
    }
}

struct Card<Content: View>: View {
    @ViewBuilder var content: () -> Content
    var body: some View {
        VStack(alignment: .leading, spacing: 12, content: content)
            .padding(HS.pad)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(HS.surface)
            .overlay(RoundedRectangle(cornerRadius: HS.radius).stroke(HS.line, lineWidth: 1))
            .clipShape(RoundedRectangle(cornerRadius: HS.radius))
    }
}

struct PrimaryButton: View {
    var title: String
    var systemImage: String?
    var enabled: Bool = true
    var action: () -> Void
    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if let systemImage { Image(systemName: systemImage) }
                Text(title).font(.system(size: 17, weight: .semibold))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 16)
            .background(enabled ? HS.accent : HS.raised)
            .foregroundStyle(enabled ? Color.black : HS.muted)
            .clipShape(RoundedRectangle(cornerRadius: HS.radius))
        }
        .disabled(!enabled)
    }
}

/// The quality verdict. Text and glyph carry it; colour only reinforces.
struct QualityChip: View {
    var level: String
    var body: some View {
        let (c, g, t): (Color, String, String) = {
            switch level {
            case "ok": return (HS.good, "checkmark.circle.fill", "Reliable")
            case "poor": return (HS.warn, "exclamationmark.triangle.fill", "Review advised")
            default: return (HS.bad, "xmark.octagon.fill", "Not reliable")
            }
        }()
        HStack(spacing: 6) {
            Image(systemName: g).font(.system(size: 12, weight: .bold))
            Text(t).font(.system(size: 13, weight: .semibold))
        }
        .padding(.horizontal, 10).padding(.vertical, 6)
        .background(c.opacity(0.14))
        .foregroundStyle(c)
        .overlay(Capsule().stroke(c.opacity(0.4), lineWidth: 1))
        .clipShape(Capsule())
    }
}

/// A ratio, its spread and its verdict as ONE component. They are not separable
/// by construction, so no screen can accidentally show the number alone.
struct RatioReadout: View {
    var ratio: Double
    var sd: Double
    var quality: (level: String, reason: String)
    var frames: Int
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Text(String(format: "%.3f", ratio))
                    .font(.system(size: 46, weight: .semibold, design: .monospaced))
                    .foregroundStyle(HS.ink)
                VStack(alignment: .leading, spacing: 2) {
                    Text("SD \(String(format: "%.3f", sd))").hsMono(14).foregroundStyle(HS.muted)
                    Text("\(frames) frames").hsBody().foregroundStyle(HS.muted)
                }
                Spacer()
            }
            QualityChip(level: quality.level)
            if !quality.reason.isEmpty {
                Text(quality.reason).hsBody().foregroundStyle(HS.muted)
            }
            Text("Model repeatability is 0.016. A spread far above that is the model losing the hand, not the hand changing.")
                .font(.system(size: 12)).foregroundStyle(HS.muted.opacity(0.8))
        }
    }
}

struct GuidanceRow: View {
    var g: Guidance
    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: HS.glyph(g.level))
                .font(.system(size: 14, weight: .bold))
                .foregroundStyle(HS.colour(g.level))
                .frame(width: 20)
            VStack(alignment: .leading, spacing: 1) {
                Text(g.advice).font(.system(size: 14, weight: .medium)).foregroundStyle(HS.ink)
                Text(g.label).font(.system(size: 11)).foregroundStyle(HS.muted)
            }
            Spacer()
            Text(g.detail).hsMono(13).foregroundStyle(HS.muted)
        }
        .padding(.horizontal, 12).padding(.vertical, 9)
        .background(HS.soft(g.level))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }
}

// MARK: - Root

struct RootView: View {
    @ObservedObject var model: AppModel
    var body: some View {
        ZStack {
            HS.bg.ignoresSafeArea()
            switch model.screen {
            case .home: HomeScreen(model: model)
            case .subject: SubjectScreen(model: model)
            case .capture: CaptureScreen(model: model)
            case .result: ResultScreen(model: model)
            case .queue: QueueScreen(model: model)
            case .settings: SettingsScreen(model: model)
            }
        }
        .foregroundStyle(HS.ink)
        .animation(.easeInOut(duration: 0.22), value: model.screen)
    }
}

// MARK: - Home

struct HomeScreen: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var queue: UploadQueue
    init(model: AppModel) { self.model = model; self.queue = model.queue }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("HAND STUDY").hsLabel().foregroundStyle(HS.accent)
                    Text("Capture").hsTitle()
                    Text("2D:4D measurement. One visit is one child; one capture is one hand.")
                        .hsBody().foregroundStyle(HS.muted)
                }
                .padding(.top, 8)

                PrimaryButton(title: "New capture", systemImage: "hand.raised.fill") {
                    Task { await model.fetchSubject(); model.screen = .subject }
                }

                Card {
                    HStack {
                        Text("Waiting to upload").hsLabel().foregroundStyle(HS.muted)
                        Spacer()
                        Text("\(queue.pendingCount)").hsMono(20)
                    }
                    if queue.uploading {
                        ProgressView(value: Double(queue.progress.sent),
                                     total: Double(max(queue.progress.total, 1)))
                            .tint(HS.accent)
                        Text("Sending frame \(queue.progress.sent) of \(queue.progress.total)")
                            .font(.system(size: 12)).foregroundStyle(HS.muted)
                    } else if queue.pendingCount > 0 {
                        Text("Held on this device. They will upload when a network is available.")
                            .font(.system(size: 12)).foregroundStyle(HS.muted)
                    } else {
                        Text("Everything has reached the server.")
                            .font(.system(size: 12)).foregroundStyle(HS.muted)
                    }
                    if queue.rejectedCount > 0 {
                        Text("\(queue.rejectedCount) refused by the server — kept, not deleted")
                            .font(.system(size: 12)).foregroundStyle(HS.warn)
                    }
                    Button("Open queue") { model.screen = .queue }
                        .font(.system(size: 14, weight: .medium)).foregroundStyle(HS.accent)
                }

                if !model.status.isEmpty {
                    Text(model.status).hsBody().foregroundStyle(HS.warn)
                }

                Card {
                    HStack {
                        Text("Server schema").hsLabel().foregroundStyle(HS.muted)
                        Spacer()
                        Image(systemName: model.schemaLive ? "checkmark.circle.fill" : "wifi.slash")
                            .foregroundStyle(model.schemaLive ? HS.good : HS.warn)
                        Text(model.schemaLive ? "Live" : "Using built-in copy")
                            .font(.system(size: 13)).foregroundStyle(HS.muted)
                    }
                    Button("Settings and diagnostics") { model.screen = .settings }
                        .font(.system(size: 14, weight: .medium)).foregroundStyle(HS.accent)
                }
                Spacer(minLength: 24)
            }
            .padding(HS.pad)
        }
    }
}

// MARK: - Subject

struct SubjectScreen: View {
    @ObservedObject var model: AppModel
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Header(title: "This child", subtitle: "Recorded now because none of it can be recovered after the child leaves.") {
                    model.screen = .home
                }

                Card {
                    Text("Subject number").hsLabel().foregroundStyle(HS.muted)
                    HStack {
                        Text(model.subject.isEmpty ? "—" : model.subject).hsMono(22)
                        Spacer()
                        if model.subject.isEmpty {
                            Button("Get number") { Task { await model.fetchSubject() } }
                                .foregroundStyle(HS.accent)
                        }
                    }
                    Text("Issued by the server. The app never invents one — two devices in the field would collide, and it could not be undone.")
                        .font(.system(size: 12)).foregroundStyle(HS.muted)
                }

                Card {
                    Text("Hand being photographed").hsLabel().foregroundStyle(HS.muted)
                    Picker("", selection: $model.hand) {
                        Text("Right").tag("right"); Text("Left").tag("left")
                    }
                    .pickerStyle(.segmented)
                    Text("The published reference figures are right-hand only, so which hand this is changes what the number can be compared against.")
                        .font(.system(size: 12)).foregroundStyle(HS.muted)
                }

                ForEach(model.schema.entryTags, id: \.0) { key, options in
                    Card {
                        Text(key.replacingOccurrences(of: "_", with: " ")).hsLabel().foregroundStyle(HS.muted)
                        Picker("", selection: Binding(
                            get: { model.fields[key] ?? "" },
                            set: { model.fields[key] = $0 })) {
                                Text("—").tag("")
                                ForEach(options, id: \.self) { Text($0).tag($0) }
                            }
                            .pickerStyle(.segmented)
                    }
                }

                ForEach(model.schema.entryNumeric, id: \.0) { key, range in
                    Card {
                        Text("\(key.replacingOccurrences(of: "_", with: " "))  (\(Int(range.min))–\(Int(range.max)))")
                            .hsLabel().foregroundStyle(HS.muted)
                        TextField("", text: Binding(
                            get: { model.fields[key] ?? "" },
                            set: { model.fields[key] = $0 }))
                            .keyboardType(.decimalPad)
                            .textFieldStyle(.plain)
                            .padding(10).background(HS.raised)
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                }

                if let e = model.validateAll() {
                    Text(e).hsBody().foregroundStyle(HS.bad)
                }

                PrimaryButton(title: "Start capture", systemImage: "camera.fill",
                              enabled: !model.subject.isEmpty && model.validateAll() == nil) {
                    model.screen = .capture
                }
                Spacer(minLength: 24)
            }
            .padding(HS.pad)
        }
    }
}

// MARK: - Capture

struct CaptureScreen: View {
    @ObservedObject var model: AppModel

    var body: some View {
        ZStack {
            CameraPreview(model: model).ignoresSafeArea()
            LandmarkOverlay(model: model).ignoresSafeArea()

            VStack {
                HStack {
                    Button { model.stop(); model.screen = .subject } label: {
                        Image(systemName: "chevron.left").padding(10)
                            .background(HS.captureWell.opacity(0.82)).clipShape(Circle())
                    }
                    Spacer()
                    VStack(alignment: .trailing, spacing: 2) {
                        Text(model.subject).hsMono(14)
                        Text(model.hand).font(.system(size: 11)).foregroundStyle(HS.muted)
                    }
                    .padding(.horizontal, 10).padding(.vertical, 6)
                    .background(HS.captureWell.opacity(0.82)).clipShape(RoundedRectangle(cornerRadius: 8))
                }
                .padding(.horizontal, HS.pad)

                Spacer()

                VStack(spacing: 8) {
                    ForEach(model.guidance) { GuidanceRow(g: $0) }
                }
                .padding(10)
                .background(HS.captureWell.opacity(0.86))
                .clipShape(RoundedRectangle(cornerRadius: HS.radius))
                .padding(.horizontal, HS.pad)

                HStack(spacing: 12) {
                    Label(String(format: "%.0f ms", model.inferMS), systemImage: "cpu")
                    if model.liveRatio > 0 {
                        Label(String(format: "%.3f", model.liveRatio), systemImage: "ruler")
                    }
                }
                .font(.system(size: 12, design: .monospaced))
                .foregroundStyle(HS.muted)
                .padding(.top, 6)

                if model.capturing {
                    VStack(spacing: 6) {
                        ProgressView(value: Double(model.captured),
                                     total: Double(Config.framesPerCapture)).tint(HS.accent)
                        Text("Hold still — \(model.captured) of \(Config.framesPerCapture)")
                            .font(.system(size: 13, weight: .medium))
                    }
                    .padding(.horizontal, HS.pad).padding(.top, 8)
                } else {
                    let blocked = GuidanceEngine.blocking(model.guidance)
                    PrimaryButton(title: blocked.isEmpty ? "Capture" : blocked[0].advice,
                                  systemImage: blocked.isEmpty ? "camera.fill" : "hand.raised.slash",
                                  enabled: model.canCapture) {
                        model.beginCapture()
                    }
                    .padding(.horizontal, HS.pad).padding(.top, 8)
                }
            }
            .padding(.bottom, 20)
        }
        .onAppear { model.start() }
        .onDisappear { model.stop() }
    }
}

/// Landmarks drawn over the preview. The preview letterboxes the frame inside
/// the view and the points are normalised to the frame, so the same fit has to
/// be recomputed here or the dots drift away from the fingers.
struct LandmarkOverlay: View {
    @ObservedObject var model: AppModel
    private let colours: [Color] = [HS.accent, .cyan, .green, .orange]

    var body: some View {
        GeometryReader { geo in
            let vw = geo.size.width, vh = geo.size.height
            let fw = CGFloat(max(model.frameW, 1)), fh = CGFloat(max(model.frameH, 1))
            let s = min(vw / fw, vh / fh)          // preview is fitted, so scale is min
            let dw = fw * s, dh = fh * s
            let ox = (vw - dw) / 2, oy = (vh - dh) / 2
            ZStack {
                ForEach(Array(model.points.enumerated()), id: \.offset) { i, p in
                    Circle()
                        .strokeBorder(colours[min(i, 3)], lineWidth: 2)
                        .background(Circle().fill(colours[min(i, 3)].opacity(0.25)))
                        .frame(width: 16, height: 16)
                        .position(x: ox + CGFloat(p[0]) * dw, y: oy + CGFloat(p[1]) * dh)
                }
            }
        }
        .allowsHitTesting(false)
    }
}

// MARK: - Result

struct ResultScreen: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var queue: UploadQueue
    init(model: AppModel) { self.model = model; self.queue = model.queue }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Header(title: "Capture recorded", subtitle: "Saved to this device before anything else.") {
                    model.screen = .home
                }
                if let r = model.lastResult {
                    Card {
                        RatioReadout(ratio: r.median, sd: r.sd,
                                     quality: r.quality, frames: r.frames)
                    }
                    Card {
                        Text("Uploading").hsLabel().foregroundStyle(HS.muted)
                        if queue.uploading {
                            ProgressView(value: Double(queue.progress.sent),
                                         total: Double(max(queue.progress.total, 1))).tint(HS.accent)
                            Text("Frame \(queue.progress.sent) of \(queue.progress.total)")
                                .font(.system(size: 12)).foregroundStyle(HS.muted)
                        }
                        Text("The \(Config.priorityFrames) best-scoring frames go first, so the server has a measurable capture within seconds. The remaining \(max(0, r.frames - Config.priorityFrames)) follow in the background — nothing is discarded.")
                            .font(.system(size: 12)).foregroundStyle(HS.muted)
                    }
                    if r.quality.level != "ok" {
                        Card {
                            Text("Recapture recommended").hsLabel().foregroundStyle(HS.warn)
                            Text(r.quality.reason).hsBody().foregroundStyle(HS.muted)
                            PrimaryButton(title: "Capture this hand again", systemImage: "arrow.clockwise") {
                                model.screen = .capture
                            }
                        }
                    }
                }
                PrimaryButton(title: "Next child", systemImage: "person.fill.badge.plus") {
                    model.subject = ""; model.fields = [:]; model.lastResult = nil
                    Task { await model.fetchSubject(); model.screen = .subject }
                }
                Spacer(minLength: 24)
            }
            .padding(HS.pad)
        }
    }
}

// MARK: - Queue

struct QueueScreen: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var queue: UploadQueue
    init(model: AppModel) { self.model = model; self.queue = model.queue }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Header(title: "Upload queue", subtitle: "Captures live on this device until the server confirms them.") {
                    model.screen = .home
                }
                Card {
                    HStack {
                        Text("Waiting").hsLabel().foregroundStyle(HS.muted)
                        Spacer()
                        Text("\(queue.pendingCount)").hsMono(20)
                    }
                    if queue.pendingCount == 0 {
                        Text("Nothing waiting.").hsBody().foregroundStyle(HS.muted)
                    }
                    if let e = queue.lastError {
                        Text(e).font(.system(size: 12)).foregroundStyle(HS.warn)
                    }
                    PrimaryButton(title: "Retry now", systemImage: "arrow.clockwise",
                                  enabled: queue.pendingCount > 0 && !queue.uploading) {
                        queue.kick()
                    }
                }
                if queue.incompleteCount > 0 {
                    Card {
                        Text("Interrupted mid-capture").hsLabel().foregroundStyle(HS.warn)
                        Text("\(queue.incompleteCount) capture\(queue.incompleteCount == 1 ? "" : "s") stopped before all frames were taken — the app closed or the phone ran out of memory. The frames that were taken are kept on the device, but they are not a complete capture and are not uploaded as one. Photograph those children again if they are still available.")
                            .font(.system(size: 12)).foregroundStyle(HS.muted)
                    }
                }
                if queue.rejectedCount > 0 {
                    Card {
                        Text("Refused by the server").hsLabel().foregroundStyle(HS.warn)
                        Text("\(queue.rejectedCount) capture\(queue.rejectedCount == 1 ? "" : "s") the server would not accept. They are kept on the device, not deleted — the frames cannot be retaken.")
                            .font(.system(size: 12)).foregroundStyle(HS.muted)
                    }
                }
                Spacer(minLength: 24)
            }
            .padding(HS.pad)
        }
    }
}

// MARK: - Settings

struct SettingsScreen: View {
    @ObservedObject var model: AppModel
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Header(title: "Settings", subtitle: "What this build is and what it is talking to.") {
                    model.screen = .home
                }
                Card {
                    Row("Server", Config.base.host ?? "—")
                    Row("App version", Config.appVersion)
                    Row("Frames per capture", "\(Config.framesPerCapture)")
                    Row("Uploaded first", "\(Config.priorityFrames) best")
                    Row("Camera", model.isSimulated ? "simulated" : "device")
                    Row("Schema", model.schemaLive ? "live" : "built-in")
                }
                Card {
                    Text("Measurement").hsLabel().foregroundStyle(HS.muted)
                    Row("Crease convention", model.schema.crease_convention ?? "—")
                    Row("Model checkpoint", model.schema.model_checkpoint ?? "—")
                    Row("Target hand fraction", String(format: "%.3f", Config.targetHandFraction))
                    Row("Blur floor", String(format: "%.0f", Predictor.blurFloor))
                    Text("This app measures a ratio. It does not screen for, diagnose or predict any condition.")
                        .font(.system(size: 12)).foregroundStyle(HS.muted)
                }
                Card {
                    Text("On this device").hsLabel().foregroundStyle(HS.muted)
                    Row("Captures waiting", "\(CaptureStore.pending().count)")
                    Row("Refused, kept", "\(CaptureStore.rejected().count)")
                }
                Spacer(minLength: 24)
            }
            .padding(HS.pad)
        }
    }

    @ViewBuilder private func Row(_ k: String, _ v: String) -> some View {
        HStack {
            Text(k).hsBody().foregroundStyle(HS.muted)
            Spacer()
            Text(v).hsMono(13)
        }
    }
}

// MARK: - Shared

struct Header: View {
    var title: String
    var subtitle: String
    var back: () -> Void
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Button(action: back) {
                HStack(spacing: 4) {
                    Image(systemName: "chevron.left").font(.system(size: 13, weight: .semibold))
                    Text("Back").font(.system(size: 15))
                }
                .foregroundStyle(HS.accent)
            }
            Text(title).hsTitle()
            Text(subtitle).hsBody().foregroundStyle(HS.muted)
        }
        .padding(.top, 4)
    }
}

/// The live frame. One representation for all three sources -- AVFoundation,
/// ARKit and the simulator -- because the operator should not be able to tell
/// which one is running, and neither should the rest of this file.
struct CameraPreview: View {
    @ObservedObject var model: AppModel
    var body: some View {
        GeometryReader { geo in
            ZStack {
                Color.black
                if let cg = model.preview {
                    Image(decorative: cg, scale: 1, orientation: .up)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(width: geo.size.width, height: geo.size.height)
                } else {
                    VStack(spacing: 10) {
                        ProgressView().tint(HS.muted)
                        Text("Starting the camera…").hsBody().foregroundStyle(HS.muted)
                    }
                }
            }
        }
    }
}
