//  Predictor.swift
//  Letterbox to 1024x1024 BGRA, run CreaseNet on the Neural Engine, and measure
//  the two image properties that decide whether a frame is worth uploading.
//
//  The model bakes ImageNet normalisation into the graph, so pixels go in as
//  plain 0-255 RGB. The only preprocessing that matters is the letterbox: the
//  long side is scaled to 1024 and the remainder padded, ASPECT PRESERVED. A
//  stretched image gives wrong points with no error, so the scale and pad are
//  recorded rather than implicit.

import Accelerate
import CoreImage
import CoreML
import CoreVideo
import Foundation

struct Prediction {
    /// Exactly as the model emitted: normalised 0-1 in the 1024 letterboxed frame.
    /// This is what gets uploaded -- the server compares against its own
    /// preprocessing, so anything we map or round here would be a divergence.
    var raw: [[Float]]
    /// The same points on the original camera frame, normalised 0-1, for drawing.
    var onFrame: [[Float]]
    var inferMS: Double
    var totalMS: Double
    var ratio: Float
}

/// What one frame looks like as an image, independent of what the model found.
struct ImageMetrics {
    /// Variance of the Laplacian. The server uses the same measure with a floor
    /// of 60.0, so the phone and the server agree about what "blurry" means.
    var sharpness: Double
    var meanLuma: Double        // 0-255
    var clippedFraction: Double // share of pixels at 0 or 255
}

final class Predictor {

    static let side = 1024
    /// The server's BLUR_FLOOR, and it only means the same thing if the pixels
    /// are the same. The server computes Laplacian variance on the frame with
    /// its long side capped at 2048 -- for a 1440x1920 capture that is native
    /// resolution. Scoring a captured frame therefore uses `serverScale: true`.
    static let blurFloor: Double = 60.0
    /// The live-guidance figure is computed on a fixed 256 square, which is far
    /// cheaper but NOT the same measure as the server's: downscaling
    /// concentrates edges and inflates the variance (one real frame read 184 at
    /// 256px against 23.6 at native). So it gets its own floor.
    ///
    /// That floor was first set to the server's floor times that ratio, 470.
    /// It is now calibrated to real captures instead: six real bursts of an
    /// adult hand, steady to +-0.002 and measured well by the model, read
    /// 112-318 on an offline replica of this figure, and 470 refused every one
    /// of them. The replica reads low on full-resolution frames (one burst it
    /// put at 229 reads about 465 in the app), so real frames clear 150 with
    /// room. The soft embedded simulator still reads 128 in the app and is
    /// still refused.
    static let liveBlurFloor: Double = 150.0
    /// The server's FRAME_MAX_PX.
    static let serverMaxPx: CGFloat = 2048

    private let model: MLModel
    private let ciContext: CIContext
    private var pool: CVPixelBufferPool?
    private let black: CIImage
    /// Reused so a 256x256 grey buffer is not allocated 30 times a second.
    private var greyBuf = [UInt8](repeating: 0, count: 256 * 256)
    private var lapBuf = [Float](repeating: 0, count: 256 * 256)

    private(set) var lastScale: CGFloat = 1
    private(set) var lastPadX: CGFloat = 0
    private(set) var lastPadY: CGFloat = 0

    init() throws {
        guard let url = Bundle.main.url(forResource: "CreaseNet", withExtension: "mlmodelc") else {
            throw NSError(domain: "HandStudy", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "CreaseNet.mlmodelc not in bundle"])
        }
        let cfg = MLModelConfiguration()
        #if targetEnvironment(simulator)
        cfg.computeUnits = .cpuOnly          // the simulator's Metal host is unreliable
        #else
        cfg.computeUnits = .all              // let it pick the ANE
        #endif
        model = try MLModel(contentsOf: url, configuration: cfg)
        // Intermediates are never reused across frames, so caching them only
        // grows memory on a device that is also holding 20 full-res JPEGs.
        #if targetEnvironment(simulator)
        ciContext = CIContext(options: [.cacheIntermediates: false,
                                        .workingColorSpace: NSNull(),
                                        .useSoftwareRenderer: true])
        #else
        ciContext = CIContext(options: [.cacheIntermediates: false,
                                        .workingColorSpace: NSNull()])
        #endif
        black = CIImage(color: .black)
            .cropped(to: CGRect(x: 0, y: 0, width: Predictor.side, height: Predictor.side))

        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: Predictor.side,
            kCVPixelBufferHeightKey as String: Predictor.side,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any]()
        ]
        CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &pool)
    }

    /// Scale the long side to 1024, centre it, pad the rest black. Centre
    /// padding is the training convention and is invariant to CoreImage's
    /// bottom-left origin, which a corner pad would not be.
    private func letterbox(_ src: CVPixelBuffer) -> CVPixelBuffer? {
        guard let pool else { return nil }
        var made: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &made) == kCVReturnSuccess,
              let dst = made else { return nil }

        let w = CGFloat(CVPixelBufferGetWidth(src))
        let h = CGFloat(CVPixelBufferGetHeight(src))
        let side = CGFloat(Predictor.side)
        let scale = side / max(w, h)
        let padX = (side - w * scale) / 2
        let padY = (side - h * scale) / 2
        lastScale = scale; lastPadX = padX; lastPadY = padY

        let img = CIImage(cvPixelBuffer: src)
            .transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            .transformed(by: CGAffineTransform(translationX: padX, y: padY))
        ciContext.render(img.composited(over: black), to: dst)
        return dst
    }

    func run(on src: CVPixelBuffer) throws -> Prediction {
        let t0 = CFAbsoluteTimeGetCurrent()
        guard let boxed = letterbox(src) else {
            throw NSError(domain: "HandStudy", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "letterbox failed"])
        }
        let input = try MLDictionaryFeatureProvider(
            dictionary: ["image": MLFeatureValue(pixelBuffer: boxed)])

        let t1 = CFAbsoluteTimeGetCurrent()
        let out = try model.prediction(from: input)
        let t2 = CFAbsoluteTimeGetCurrent()

        guard let arr = out.featureValue(for: "points")?.multiArrayValue, arr.count >= 8 else {
            throw NSError(domain: "HandStudy", code: 3,
                          userInfo: [NSLocalizedDescriptionKey: "no points output"])
        }

        // float16, shape (1,8); flat indexing walks x,y per point. Read through
        // a typed pointer rather than 8 NSNumber boxes per frame.
        var raw: [[Float]] = []
        raw.reserveCapacity(4)
        if arr.dataType == .float32 {
            let p = arr.dataPointer.bindMemory(to: Float.self, capacity: 8)
            for i in 0..<4 { raw.append([p[i * 2], p[i * 2 + 1]]) }
        } else {
            for i in 0..<4 { raw.append([arr[i * 2].floatValue, arr[i * 2 + 1].floatValue]) }
        }

        let side = CGFloat(Predictor.side)
        let w = CGFloat(CVPixelBufferGetWidth(src))
        let h = CGFloat(CVPixelBufferGetHeight(src))
        let onFrame: [[Float]] = raw.map { p in
            let px = (CGFloat(p[0]) * side - lastPadX) / lastScale
            let py = (CGFloat(p[1]) * side - lastPadY) / lastScale
            return [Float(px / w), Float(py / h)]
        }

        // Measured in the letterboxed square, where x and y share a scale.
        // Doing it on the normalised camera frame would stretch one axis.
        let index = Predictor.dist(raw[1], raw[0])
        let ring = Predictor.dist(raw[3], raw[2])

        return Prediction(raw: raw, onFrame: onFrame,
                          inferMS: (t2 - t1) * 1000,
                          totalMS: (t2 - t0) * 1000,
                          ratio: ring > 0 ? index / ring : 0)
    }

    static func dist(_ a: [Float], _ b: [Float]) -> Float {
        let dx = a[0] - b[0], dy = a[1] - b[1]
        return (dx * dx + dy * dy).squareRoot()
    }

    // MARK: - Image metrics

    /// Sharpness and exposure from a 256x256 grey copy.
    ///
    /// Downsampling first is deliberate: Laplacian variance on a full 1080p
    /// frame is dominated by sensor noise, and at 30 fps it is also too slow.
    /// A fixed 256 square makes the number comparable between frames and
    /// between devices, which is the only way a threshold can mean anything.
    /// - Parameter serverScale: reproduce the server's preprocessing (long side
    ///   capped at 2048) so the number is directly comparable to BLUR_FLOOR.
    ///   Used when scoring captured frames. The cheap fixed-256 path is for live
    ///   guidance only and is compared against `liveBlurFloor` instead.
    func metrics(for src: CVPixelBuffer, serverScale: Bool = false) -> ImageMetrics {
        let srcW = CGFloat(CVPixelBufferGetWidth(src)), srcH = CGFloat(CVPixelBufferGetHeight(src))
        let n: Int = serverScale
            ? Int(min(1, Predictor.serverMaxPx / max(srcW, srcH)) * max(srcW, srcH))
            : 256
        let scaled = CIImage(cvPixelBuffer: src)
        let w = CGFloat(CVPixelBufferGetWidth(src)), h = CGFloat(CVPixelBufferGetHeight(src))
        // A forced square is fine for the cheap live figure, but at server scale
        // the aspect must be preserved or the resampling itself changes the
        // variance and the comparison is meaningless again.
        let kx = serverScale ? CGFloat(n) / max(w, h) : CGFloat(n) / w
        let ky = serverScale ? CGFloat(n) / max(w, h) : CGFloat(n) / h
        let img = scaled
            .transformed(by: CGAffineTransform(scaleX: kx, y: ky))
            .applyingFilter("CIPhotoEffectMono")

        let wpx = serverScale ? Int((srcW * kx).rounded()) : n
        let hpx = serverScale ? Int((srcH * ky).rounded()) : n
        let need = wpx * hpx
        if greyBuf.count < need { greyBuf = [UInt8](repeating: 0, count: need) }
        if lapBuf.count < need { lapBuf = [Float](repeating: 0, count: need) }
        greyBuf.withUnsafeMutableBytes { rawPtr in
            guard let base = rawPtr.baseAddress else { return }
            ciContext.render(img,
                             toBitmap: base,
                             rowBytes: wpx,
                             bounds: CGRect(x: 0, y: 0, width: wpx, height: hpx),
                             format: .L8,
                             colorSpace: nil)
        }

        // Mean and clipping straight off the bytes; a tight UInt8 pass is
        // cheaper than promoting the whole plane to Float for two scalars.
        var clipped = 0
        var sum: UInt32 = 0
        greyBuf.withUnsafeBufferPointer { g in
            for i in 0..<need {
                let v = g[i]
                sum &+= UInt32(v)
                if v == 0 || v == 255 { clipped &+= 1 }
            }
        }
        let mean = Double(sum) / Double(need)

        // Laplacian variance, vectorised. The scalar version was ~64k
        // double-precision iterations per call, several times a second, on a
        // phone that is also running the Neural Engine and encoding JPEGs.
        var variance = 0.0
        greyBuf.withUnsafeMutableBufferPointer { g in
            lapBuf.withUnsafeMutableBufferPointer { lap in
                guard let gp = g.baseAddress, let lp = lap.baseAddress else { return }
                vDSP_vfltu8(gp, 1, lp, 1, vDSP_Length(need))

                var kernel: [Float] = [0, 1, 0,
                                       1, -4, 1,
                                       0, 1, 0]
                var src = vImage_Buffer(data: lp, height: vImagePixelCount(hpx),
                                        width: vImagePixelCount(wpx), rowBytes: wpx * 4)
                var out = [Float](repeating: 0, count: need)
                out.withUnsafeMutableBufferPointer { ob in
                    guard let op = ob.baseAddress else { return }
                    var dst = vImage_Buffer(data: op, height: vImagePixelCount(hpx),
                                            width: vImagePixelCount(wpx), rowBytes: wpx * 4)
                    _ = vImageConvolve_PlanarF(&src, &dst, nil, 0, 0, &kernel, 3, 3, 0,
                                               vImage_Flags(kvImageEdgeExtend))
                    // Variance is E[x^2] - E[x]^2; the Laplacian's mean is ~0 but
                    // it is subtracted rather than assumed.
                    var m: Float = 0, ms: Float = 0
                    vDSP_meanv(op, 1, &m, vDSP_Length(need))
                    vDSP_measqv(op, 1, &ms, vDSP_Length(need))
                    variance = max(0, Double(ms) - Double(m) * Double(m))
                }
            }
        }

        return ImageMetrics(sharpness: variance,
                            meanLuma: mean,
                            clippedFraction: Double(clipped) / Double(need))
    }
}
