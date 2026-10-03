import Foundation

/// The long-audio windows of one recording that is still growing, decoded as soon as
/// each is complete (`AsrManager.transcribeAhead`), so that `transcribe` at the end
/// decodes only the windows that are left.
///
/// A window is reused only for exactly the same samples and settings, so the text is
/// the same as decoding every window at the end. A window still being decoded when
/// `transcribe` reaches it is awaited rather than decoded twice.
public actor DecodedWindows {
    typealias Tokens = [ChunkProcessor.TokenWindow]

    struct Key: Hashable {
        /// The first sample read, including any warm-up or context before the window.
        let firstSample: Int
        let sampleCount: Int
        let contextSamples: Int
        let chunkStart: Int
        let isLastChunk: Bool
        let emitTokensAfterFrame: Int?
        let language: String?
    }

    private var windows: [Key: (samples: [Float], decoding: Task<Tokens, Error>)] = [:]

    public init() {}

    /// The window's tokens: decoded earlier from the same samples, or by `decode` now.
    func tokens(
        for key: Key, samples: [Float], decode: @escaping @Sendable () async throws -> Tokens
    ) async throws -> Tokens {
        if let window = windows[key], window.samples == samples {
            return try await window.decoding.value
        }
        let decoding = Task { try await decode() }
        windows[key] = (samples, decoding)
        do {
            return try await decoding.value
        } catch {
            // A failed window is decoded again next time, not remembered as failed.
            if windows[key]?.decoding == decoding { windows[key] = nil }
            throw error
        }
    }
}
