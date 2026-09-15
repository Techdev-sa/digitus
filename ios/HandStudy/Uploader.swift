//  Uploader.swift
//  A capture store that survives, and a queue that drains it one frame at a time.
//
//  Everything is written to disk the moment it is captured and deleted only once
//  the server has confirmed it. School wifi is expected to be poor or absent, so
//  the app must be able to record a whole morning offline and upload later.
//
//  Uploads go frame by frame against /api/v2/capture. A single 25 MB body that
//  fails at 90% costs the whole capture and tells the operator nothing about
//  which part failed; a frame that fails costs one retry. The best-scoring
//  frames go first, so a result comes back while the rest are still climbing.

import Foundation
import os
import UIKit

struct CaptureMeta: Codable {
    var subject: String
    var hand: String
    var capturedAt: Date
    var appVersion: String
    var deviceName: String
    var frameCount: Int
    var medianRatio: Float
    var sdRatio: Float
    var fields: [String: String]
    var geometry: [String: Double]
    /// Frame indices, best first. The queue uploads in this order.
    var uploadOrder: [Int]
    /// Why each frame scored as it did, kept so a bad heuristic can be
    /// diagnosed later against what the server independently measures.
    var scores: [String: [String: Double]]
}

/// One capture on disk: a folder of JPEGs plus its metadata.
enum CaptureStore {

    static let log = Logger(subsystem: "sa.tdev.handstudy", category: "store")

    private static func dir(_ name: String) -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory,
                                            in: .userDomainMask)[0]
            .appendingPathComponent(name, isDirectory: true)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        // Images of children must not be readable while the phone is locked.
        try? FileManager.default.setAttributes(
            [.protectionKey: FileProtectionType.complete], ofItemAtPath: base.path)
        return base
    }

    static var root: URL = { dir("pending") }()
    /// Captures the server refused for a reason retrying cannot fix. Kept, not
    /// deleted -- the frames are unrepeatable -- but out of the queue's way so
    /// one bad capture cannot block the morning's uploads behind it.
    static var rejectedRoot: URL = { dir("rejected") }()

    /// Opens a capture directory before the first frame exists.
    ///
    /// Frames are written as they arrive rather than buffered and saved at the
    /// end. Holding 24 full-quality JPEGs is about 30 MB of live memory, and a
    /// capture that exists only in memory is exactly what a low-memory kill or a
    /// phone call destroys -- which is the one thing this app promises cannot
    /// happen, because the child will not be photographed twice.
    static func begin() -> URL? {
        let d = root.appendingPathComponent(UUID().uuidString, isDirectory: true)
        do {
            try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
            return d
        } catch {
            log.error("could not open a capture: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }

    static func append(_ d: URL, index: Int, jpeg: Data) {
        do {
            try jpeg.write(to: d.appendingPathComponent(String(format: "f%04d.jpg", index)),
                           options: .completeFileProtection)
        } catch {
            log.error("frame write failed: \(error.localizedDescription, privacy: .public)")
        }
    }

    /// Writes the metadata last. Its presence is what marks a capture complete,
    /// so a directory interrupted mid-burst is recognisable rather than being
    /// uploaded as if it were whole.
    @discardableResult
    static func finish(_ d: URL, meta: CaptureMeta) -> URL? {
        do {
            let enc = JSONEncoder()
            enc.dateEncodingStrategy = .iso8601
            try enc.encode(meta).write(to: d.appendingPathComponent("meta.json"),
                                       options: .completeFileProtection)
            return d
        } catch {
            log.error("capture meta write failed: \(error.localizedDescription, privacy: .public)")
            return nil
        }
    }

    /// A directory with frames but no meta.json: the app died mid-capture.
    /// The frames are real and are kept, but they are not a complete capture and
    /// must not be uploaded as one.
    static func incomplete() -> [URL] {
        pending().filter { !FileManager.default.fileExists(atPath: $0.appendingPathComponent("meta.json").path) }
    }

    static func pending() -> [URL] {
        let c = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        return c.filter { $0.hasDirectoryPath }.sorted { $0.path < $1.path }
    }

    static func rejected() -> [URL] {
        let c = (try? FileManager.default.contentsOfDirectory(at: rejectedRoot, includingPropertiesForKeys: nil)) ?? []
        return c.filter { $0.hasDirectoryPath }.sorted { $0.path < $1.path }
    }

    static func reject(_ d: URL, reason: String) {
        let dest = rejectedRoot.appendingPathComponent(d.lastPathComponent, isDirectory: true)
        try? FileManager.default.removeItem(at: dest)
        do {
            try FileManager.default.moveItem(at: d, to: dest)
            try? Data(reason.utf8).write(to: dest.appendingPathComponent("error.txt"),
                                         options: .completeFileProtection)
        } catch {
            log.error("could not quarantine capture: \(error.localizedDescription, privacy: .public)")
        }
    }

    static func remove(_ d: URL) { try? FileManager.default.removeItem(at: d) }

    static func meta(_ d: URL) -> CaptureMeta? {
        guard let data = FileManager.default.contents(atPath: d.appendingPathComponent("meta.json").path)
        else { return nil }
        let dec = JSONDecoder()
        dec.dateDecodingStrategy = .iso8601
        return try? dec.decode(CaptureMeta.self, from: data)
    }

    static func bytes(_ d: URL) -> Int64 {
        let files = (try? FileManager.default.contentsOfDirectory(at: d, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        return files.reduce(0) { $0 + Int64((try? $1.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) }
    }
}

struct UploadError: Error {
    var message: String
    var retryable: Bool
}

/// What the queue is doing, for the UI to show honestly.
struct QueueProgress {
    var capture: String = ""
    var sent: Int = 0
    var total: Int = 0
}

@MainActor
final class UploadQueue: ObservableObject {

    @Published private(set) var pendingCount = 0
    @Published private(set) var rejectedCount = 0
    /// Captures with frames but no meta.json: the app died mid-burst. Real
    /// photographs, but not a complete capture, so they are never uploaded as
    /// one. Surfaced rather than left to accumulate invisibly.
    @Published private(set) var incompleteCount = 0
    @Published private(set) var uploading = false
    @Published private(set) var progress = QueueProgress()
    @Published private(set) var lastError: String?
    @Published private(set) var lastResultJSON: String?

    private var draining = false
    private var attempts: [String: Int] = [:]
    private let log = Logger(subsystem: "sa.tdev.handstudy", category: "upload")

    init() { refresh() }

    func refresh() {
        let incomplete = CaptureStore.incomplete()
        incompleteCount = incomplete.count
        pendingCount = CaptureStore.pending().count - incomplete.count
        rejectedCount = CaptureStore.rejected().count
    }

    func kick() {
        refresh()
        guard !draining else { return }
        Task { await drain() }
    }

    private func drain() async {
        draining = true
        defer { draining = false; uploading = false; progress = QueueProgress(); refresh() }

        while let d = CaptureStore.pending().first(where: {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("meta.json").path)
        }) {
            let key = d.lastPathComponent
            uploading = true
            do {
                try await upload(d)
                CaptureStore.remove(d)
                attempts[key] = nil
                lastError = nil
                refresh()
            } catch let e as UploadError {
                // A rejection is a fact about the payload, not the network.
                // Retrying cannot fix it and would block every later capture
                // behind it, so it is surfaced and set aside.
                if !e.retryable {
                    log.error("capture rejected: \(e.message, privacy: .public)")
                    CaptureStore.reject(d, reason: e.message)
                    lastError = e.message
                    attempts[key] = nil
                    refresh()
                    continue
                }
                if await backOff(key, e.message) { return }
            } catch {
                if await backOff(key, error.localizedDescription) { return }
            }
        }
    }

    private func backOff(_ key: String, _ message: String) async -> Bool {
        let n = (attempts[key] ?? 0) + 1
        attempts[key] = n
        lastError = message
        log.error("upload failed (\(n, privacy: .public)): \(message, privacy: .public)")
        // Backing off rather than spinning, and never discarding: an upload that
        // cannot succeed now may succeed on the way home.
        let delay = min(60.0, pow(2.0, Double(min(n, 6))))
        try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
        return n >= 3
    }

    // MARK: One capture, frame by frame

    private func upload(_ d: URL) async throws {
        guard let meta = CaptureStore.meta(d) else {
            throw UploadError(message: "capture folder is incomplete", retryable: false)
        }
        let fm = FileManager.default
        let files = ((try? fm.contentsOfDirectory(at: d, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "jpg" }
        guard !files.isEmpty else {
            throw UploadError(message: "capture has no frames", retryable: false)
        }
        var byIndex: [Int: URL] = [:]
        for f in files {
            if let n = Int(f.deletingPathExtension().lastPathComponent.dropFirst()) { byIndex[n] = f }
        }
        // Best frames first; anything not scored follows in natural order.
        var order = meta.uploadOrder.filter { byIndex[$0] != nil }
        order += byIndex.keys.filter { !order.contains($0) }.sorted()

        // Resume: a capture interrupted mid-upload keeps its id, so the frames
        // already delivered are not sent twice.
        let cid = try await begin(meta: meta, resuming: d)
        let already = Set(try await status(cid))

        progress = QueueProgress(capture: meta.subject, sent: already.count, total: order.count)
        for idx in order where !already.contains(idx) {
            guard let url = byIndex[idx], let data = fm.contents(atPath: url.path) else { continue }
            try await sendFrame(cid: cid, idx: idx, jpeg: data,
                                score: meta.scores["\(idx)"])
            progress.sent += 1
        }
        let body = try await commit(cid: cid, meta: meta)
        lastResultJSON = body
        try? fm.removeItem(at: d.appendingPathComponent("cid.txt"))
    }

    private func begin(meta: CaptureMeta, resuming d: URL) async throws -> String {
        // A capture id already on disk means this upload was interrupted;
        // reusing it is what makes the retry cheap.
        let marker = d.appendingPathComponent("cid.txt")
        if let existing = try? String(contentsOf: marker, encoding: .utf8),
           !existing.isEmpty { return existing }

        // The operator marks a capture as study data; the server moves it to the
        // study store on that tag. Defaulting everything to "study" would put
        // every test capture and every practice run into the research corpus.
        let isStudy = (meta.fields["study"] ?? "").lowercased() == "yes"
        var form: [String: String] = [
            "subject": meta.subject, "hand": meta.hand,
            "store": isStudy ? "study" : "dev",
            "source": "ios", "app_version": meta.appVersion,
            "expected": String(meta.frameCount)
        ]
        for (k, v) in meta.fields where !v.isEmpty { form[k] = v }
        if let g = try? JSONSerialization.data(withJSONObject: meta.geometry),
           let s = String(data: g, encoding: .utf8) { form["ar"] = s }

        let (data, code) = try await post(Config.captureBeginEndpoint, fields: form)
        guard code == 200,
              let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let cid = j["capture"] as? String else {
            throw UploadError(message: Self.serverMessage(data, code) ?? "could not start the upload",
                              retryable: code == 0 || code >= 500 || code == 507)
        }
        try? Data(cid.utf8).write(to: marker, options: .completeFileProtection)
        return cid
    }

    private func status(_ cid: String) async throws -> [Int] {
        var req = URLRequest(url: Config.statusEndpoint(cid))
        req.timeoutInterval = 30
        Config.authorize(&req)
        guard let (d, r) = try? await URLSession.shared.data(for: req),
              (r as? HTTPURLResponse)?.statusCode == 200,
              let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
              let got = j["received"] as? [Int] else { return [] }
        return got
    }

    private func sendFrame(cid: String, idx: Int, jpeg: Data,
                           score: [String: Double]?) async throws {
        let boundary = "B-\(UUID().uuidString)"
        var req = URLRequest(url: Config.frameEndpoint(cid))
        req.httpMethod = "POST"
        req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        // One frame is small, so a stalled connection is worth abandoning early
        // rather than holding the queue for minutes.
        req.timeoutInterval = 90
        Config.authorize(&req)

        var body = Data()
        func raw(_ s: String) { body.append(s.data(using: .utf8)!) }
        raw("--\(boundary)\r\nContent-Disposition: form-data; name=\"idx\"\r\n\r\n\(idx)\r\n")
        if let score, let sd = try? JSONSerialization.data(withJSONObject: score),
           let ss = String(data: sd, encoding: .utf8) {
            raw("--\(boundary)\r\nContent-Disposition: form-data; name=\"score\"\r\n\r\n\(ss)\r\n")
        }
        raw("--\(boundary)\r\nContent-Disposition: form-data; name=\"frame\"; filename=\"f\(idx).jpg\"\r\n")
        raw("Content-Type: image/jpeg\r\n\r\n")
        body.append(jpeg)
        raw("\r\n--\(boundary)--\r\n")
        req.httpBody = body

        let (data, resp) = try await URLSession.shared.data(for: req)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(code) else {
            throw UploadError(message: Self.serverMessage(data, code) ?? "frame \(idx) refused",
                              retryable: code == 0 || code >= 500 || code == 408 || code == 429)
        }
    }

    private func commit(cid: String, meta: CaptureMeta) async throws -> String {
        var form: [String: String] = ["subject": meta.subject, "hand": meta.hand]
        for (k, v) in meta.fields where !v.isEmpty { form[k] = v }
        let (data, code) = try await post(Config.commitEndpoint(cid), fields: form, timeout: 300)
        guard (200..<300).contains(code) else {
            throw UploadError(message: Self.serverMessage(data, code) ?? "the server would not accept the capture",
                              retryable: code == 0 || code >= 500)
        }
        return String(data: data, encoding: .utf8) ?? ""
    }

    private func post(_ url: URL, fields: [String: String],
                      timeout: TimeInterval = 60) async throws -> (Data, Int) {
        let boundary = "B-\(UUID().uuidString)"
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        req.timeoutInterval = timeout
        Config.authorize(&req)
        var body = Data()
        for (k, v) in fields where !v.isEmpty {
            body.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(k)\"\r\n\r\n\(v)\r\n".data(using: .utf8)!)
        }
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        req.httpBody = body
        do {
            let (d, r) = try await URLSession.shared.data(for: req)
            return (d, (r as? HTTPURLResponse)?.statusCode ?? 0)
        } catch {
            throw UploadError(message: error.localizedDescription, retryable: true)
        }
    }

    /// The server explains its refusals in an `error` field. Showing that text
    /// beats showing a status code to someone standing in a corridor.
    private static func serverMessage(_ d: Data, _ code: Int) -> String? {
        if let j = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
           let e = j["error"] as? String { return e }
        return code > 0 ? "server returned \(code)" : nil
    }
}
