//  Camera.swift
//  The frame source. Two implementations behind one protocol: the rear camera
//  on a device, and a still image replayed on a timer in the simulator.
//
//  The simulator has no camera at all. Without a substitute the whole pipeline
//  below the preview -- model, overlay, capture, queue, upload -- cannot be
//  exercised without a phone in hand, which is exactly the part that most needs
//  testing before it goes to a school.

import ARKit
import AVFoundation
import CoreImage
import CoreVideo
import Foundation
import UIKit

/// A depth map plus the intrinsics needed to unproject through it. Still in
/// ARKit's LANDSCAPE frame -- the colour buffer handed downstream has been
/// rotated upright, so a landmark must be rotated back before it is looked up.
struct DepthSample {
    let depth: CVPixelBuffer
    let confidence: CVPixelBuffer?
    let fx: Double, fy: Double, cx: Double, cy: Double
    let colorWidth: Int, colorHeight: Int

    /// Median of a small window: a single depth pixel at a fingertip often
    /// lands on the background behind the finger.
    func sampleDepth(nx: Double, ny: Double) -> Double? {
        CVPixelBufferLockBaseAddress(depth, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depth, .readOnly) }
        let w = CVPixelBufferGetWidth(depth), h = CVPixelBufferGetHeight(depth)
        guard let base = CVPixelBufferGetBaseAddress(depth) else { return nil }
        let rb = CVPixelBufferGetBytesPerRow(depth)
        let cx = Int(nx * Double(w)), cy = Int(ny * Double(h))
        var vals: [Double] = []
        for oy in -2...2 {
            for ox in -2...2 {
                let x = cx + ox, y = cy + oy
                guard x >= 0, x < w, y >= 0, y < h else { continue }
                let v = base.advanced(by: y * rb)
                    .assumingMemoryBound(to: Float32.self)[x]
                if v.isFinite && v > 0 { vals.append(Double(v)) }
            }
        }
        guard !vals.isEmpty else { return nil }
        vals.sort()
        return vals[vals.count / 2]
    }

    /// ARKit reports 0/1/2 (low/medium/high); normalised to 0-1.
    func sampleConfidence(nx: Double, ny: Double) -> Double {
        guard let c = confidence else { return 0 }
        CVPixelBufferLockBaseAddress(c, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(c, .readOnly) }
        let w = CVPixelBufferGetWidth(c), h = CVPixelBufferGetHeight(c)
        guard let base = CVPixelBufferGetBaseAddress(c) else { return 0 }
        let rb = CVPixelBufferGetBytesPerRow(c)
        let x = Swift.min(Swift.max(Int(nx * Double(w)), 0), w - 1)
        let y = Swift.min(Swift.max(Int(ny * Double(h)), 0), h - 1)
        let v = base.advanced(by: y * rb).assumingMemoryBound(to: UInt8.self)[x]
        return Swift.min(1.0, Double(v) / 2.0)
    }
}

protocol FrameSource: AnyObject {
    /// Called off the main thread. The buffers are valid only for this call.
    var onFrame: ((CVPixelBuffer, DepthSample?) -> Void)? { get set }
    /// Passed to Vision explicitly, never defaulted: chirality flips with
    /// orientation and a wrong value is confidently wrong.
    var visionOrientation: CGImagePropertyOrientation { get }
    var kind: String { get }
    func configure()
    func start()
    func stop()
    /// Focus and exposure are pinned for the duration of one child's captures,
    /// so their hands are comparable to each other rather than to the room.
    func lockExposureAndFocus()
    func unlock()
    var isSimulated: Bool { get }
}

// MARK: - Device

final class CameraSource: NSObject, FrameSource, AVCaptureVideoDataOutputSampleBufferDelegate {

    let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let frameQueue = DispatchQueue(label: "handstudy.frames")
    private let sessionQueue = DispatchQueue(label: "handstudy.session")
    private var device: AVCaptureDevice?

    var onFrame: ((CVPixelBuffer, DepthSample?) -> Void)?
    var isSimulated: Bool { false }
    var kind: String { "avfoundation" }
    /// The connection is rotated to portrait below, so the buffer is upright.
    var visionOrientation: CGImagePropertyOrientation { .up }

    func configure() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.session.beginConfiguration()
            self.session.sessionPreset = .hd1920x1080
            if let dev = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                 for: .video, position: .back),
               let input = try? AVCaptureDeviceInput(device: dev),
               self.session.canAddInput(input) {
                self.session.addInput(input)
                self.device = dev
            }
            self.output.videoSettings =
                [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            self.output.alwaysDiscardsLateVideoFrames = true
            self.output.setSampleBufferDelegate(self, queue: self.frameQueue)
            if self.session.canAddOutput(self.output) { self.session.addOutput(self.output) }
            if let c = self.output.connection(with: .video) {
                if #available(iOS 17.0, *),
                   c.isVideoRotationAngleSupported(90) { c.videoRotationAngle = 90 }
            }
            self.session.commitConfiguration()
        }
    }

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] ok in
            guard ok, let self else { return }
            self.sessionQueue.async {
                if !self.session.isRunning { self.session.startRunning() }
            }
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
        }
    }

    func lockExposureAndFocus() {
        sessionQueue.async { [weak self] in
            guard let dev = self?.device, (try? dev.lockForConfiguration()) != nil else { return }
            if dev.isFocusModeSupported(.locked) { dev.focusMode = .locked }
            if dev.isExposureModeSupported(.locked) { dev.exposureMode = .locked }
            dev.unlockForConfiguration()
        }
    }

    func unlock() {
        sessionQueue.async { [weak self] in
            guard let dev = self?.device, (try? dev.lockForConfiguration()) != nil else { return }
            if dev.isFocusModeSupported(.continuousAutoFocus) { dev.focusMode = .continuousAutoFocus }
            if dev.isExposureModeSupported(.continuousAutoExposure) {
                dev.exposureMode = .continuousAutoExposure
            }
            dev.unlockForConfiguration()
        }
    }

    func captureOutput(_ o: AVCaptureOutput, didOutput sb: CMSampleBuffer,
                       from c: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sb) else { return }
        onFrame?(pb, nil)
    }
}

// MARK: - Simulator

final class SimulatedSource: FrameSource {

    var onFrame: ((CVPixelBuffer, DepthSample?) -> Void)?
    var isSimulated: Bool { true }
    var kind: String { "simulated" }
    /// The embedded JPEG is stored upright.
    var visionOrientation: CGImagePropertyOrientation { .up }

    private var timer: DispatchSourceTimer?
    private let queue = DispatchQueue(label: "handstudy.simframes")
    private var buffers: [CVPixelBuffer] = []
    private var tick = 0

    func configure() {
        // A directory of real frames, passed at launch as HANDSTUDY_SIM_FRAMES
        // (SIMCTL_CHILD_HANDSTUDY_SIM_FRAMES via simctl), plays back as the camera
        // in order, so a recorded burst of a real hand drives the whole capture
        // path -- guidance, burst, result and upload -- with real frame-to-frame
        // variation. Without it, the single embedded still is used as before.
        if let dir = ProcessInfo.processInfo.environment["HANDSTUDY_SIM_FRAMES"],
           let names = try? FileManager.default.contentsOfDirectory(atPath: dir) {
            buffers = names.filter { $0.lowercased().hasSuffix(".jpg") }.sorted().compactMap { name in
                guard let img = UIImage(contentsOfFile: (dir as NSString).appendingPathComponent(name)),
                      let cg = img.cgImage else { return nil }
                return SimulatedSource.pixelBuffer(from: cg)
            }
        }
        if buffers.isEmpty,
           let data = Data(base64Encoded: simulatedHandJPEGBase64.replacingOccurrences(of: "\n", with: "")),
           let img = UIImage(data: data), let cg = img.cgImage,
           let b = SimulatedSource.pixelBuffer(from: cg) {
            buffers = [b]
        }
    }

    func start() {
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now(), repeating: .milliseconds(100))
        t.setEventHandler { [weak self] in
            guard let self, !self.buffers.isEmpty else { return }
            let b = self.buffers[self.tick % self.buffers.count]
            self.tick += 1
            self.onFrame?(b, nil)
        }
        t.resume()
        timer = t
    }

    func stop() { timer?.cancel(); timer = nil }
    func lockExposureAndFocus() {}
    func unlock() {}

    private static func pixelBuffer(from cg: CGImage) -> CVPixelBuffer? {
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()
        ]
        var pb: CVPixelBuffer?
        guard CVPixelBufferCreate(nil, cg.width, cg.height, kCVPixelFormatType_32BGRA,
                                  attrs as CFDictionary, &pb) == kCVReturnSuccess,
              let out = pb else { return nil }
        CVPixelBufferLockBaseAddress(out, [])
        defer { CVPixelBufferUnlockBaseAddress(out, []) }
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(out),
                                  width: cg.width, height: cg.height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(out),
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
                                            | CGBitmapInfo.byteOrder32Little.rawValue)
        else { return nil }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: cg.width, height: cg.height))
        return out
    }
}

// MARK: - Device with LiDAR

/// ARKit rather than AVFoundation, purely to get `sceneDepth`. Tilt is the
/// point: a photograph measures PROJECTED finger length, so a hand tilted away
/// from the sensor reads short, and the error does not cancel between index and
/// ring. Depth is what turns that confound from invisible into recorded.
final class ARSource: NSObject, FrameSource, ARSessionDelegate {

    private let session = ARSession()
    private var cfg = ARWorldTrackingConfiguration()
    private var pool: CVPixelBufferPool?
    private var poolSize = CGSize.zero
    #if targetEnvironment(simulator)
    // The simulator's Metal host (SimMetalHost) segfaults when CoreImage renders
    // into IOSurface-backed buffers, and takes the app down with it. Software
    // rendering is slower, but this path only ever feeds the simulated source.
    private let ciContext = CIContext(options: [.cacheIntermediates: false, .useSoftwareRenderer: true])
    #else
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    #endif

    var onFrame: ((CVPixelBuffer, DepthSample?) -> Void)?
    var isSimulated: Bool { false }
    var kind: String { "arkit" }
    /// capturedImage arrives landscape; it is rotated upright below before it
    /// goes downstream, so by the time Vision sees it, it is .up -- the same as
    /// the AVFoundation path.
    var visionOrientation: CGImagePropertyOrientation { .up }

    func configure() {
        cfg = ARWorldTrackingConfiguration()
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            cfg.frameSemantics.insert(.sceneDepth)
        }
        session.delegate = self
    }

    func start() { session.run(cfg, options: [.resetTracking, .removeExistingAnchors]) }
    func stop() { session.pause() }

    /// ARKit owns the capture session, so exposure cannot be pinned the way it
    /// can through AVCaptureDevice. Focus can, and is; re-running without reset
    /// options changes the configuration without dropping tracking.
    func lockExposureAndFocus() {
        cfg.isAutoFocusEnabled = false
        session.run(cfg)
    }

    func unlock() {
        cfg.isAutoFocusEnabled = true
        session.run(cfg)
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let src = frame.capturedImage
        guard let upright = rotateUpright(src) else { return }

        var sample: DepthSample?
        if let d = frame.sceneDepth {
            let k = frame.camera.intrinsics
            sample = DepthSample(
                depth: d.depthMap, confidence: d.confidenceMap,
                fx: Double(k[0][0]), fy: Double(k[1][1]),
                cx: Double(k[2][0]), cy: Double(k[2][1]),
                colorWidth: CVPixelBufferGetWidth(src),
                colorHeight: CVPixelBufferGetHeight(src))
        }
        onFrame?(upright, sample)
    }

    /// 90 degrees clockwise, into a pooled BGRA buffer.
    private func rotateUpright(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        let w = CVPixelBufferGetWidth(src), h = CVPixelBufferGetHeight(src)
        let want = CGSize(width: h, height: w)
        if pool == nil || poolSize != want {
            let attrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: Int(want.width),
                kCVPixelBufferHeightKey as String: Int(want.height),
                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()
            ]
            var p: CVPixelBufferPool?
            guard CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &p) == kCVReturnSuccess
            else { return nil }
            pool = p; poolSize = want
        }
        guard let pool else { return nil }
        var made: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &made) == kCVReturnSuccess,
              let dst = made else { return nil }
        ciContext.render(CIImage(cvPixelBuffer: src).oriented(.right), to: dst)
        return dst
    }
}

func makeFrameSource() -> FrameSource {
    #if targetEnvironment(simulator)
    return SimulatedSource()
    #else
    // Depth is what makes tilt measurable, so it is preferred wherever the
    // hardware has it; without LiDAR the app still captures, just without the
    // 3D geometry fields.
    if ARWorldTrackingConfiguration.isSupported,
       ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
        return ARSource()
    }
    return CameraSource()
    #endif
}
/// A real palm photograph, embedded so the simulator has something to see.
/// The simulator has no camera; without a real hand in the pipeline the
/// model, the overlay and the capture path cannot be exercised at all.
let simulatedHandJPEGBase64 =
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAkGBwgHBgkIBwgKCgkLDRYPDQwMDRsUFRAWIB0iIiAdHx8kKDQsJCYxJx8fLT0tMTU3" +
    "Ojo6Iys/RD84QzQ5Ojf/2wBDAQoKCg0MDRoPDxo3JR8lNzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3" +
    "Nzc3Nzc3Nzf/wAARCAIwATsDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUF" +
    "BAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVW" +
    "V1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi" +
    "4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAEC" +
    "AxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVm" +
    "Z2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq" +
    "8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD1qiiisigooooAKWkprMScIPxpgPoFMCngsxJ/IU40ALRRRQAUhozSGgBpFNIp5ppoAYel" +
    "MNSGmkUgGGm4p5FNI5pgRsoPWoJIFbqBVkjNNI9aAM2W0BzxVOWxX+7W2Vz06UxkB6ipcUO5gmx9qYbL2reMQ9KaYR3pciHzGA1m" +
    "R2phtTW+YR6UxrcelHKHMYX2dvSnC3PpWwbcelHkDNHKFzKFvTvs9aXkYHFHk+1HKK5nCCneV7Ve8n2o8r2p2C5TEftQI/arnl0e" +
    "XRYCr5dAjq1so2U7CK+yjZVjZQUosBX2e1JsPpVjbRtosB01FB6UCgBaKSk+99P50ABycgcD1pRgDHpRR3zTAWjNJRmgBaM0lJQA" +
    "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"

