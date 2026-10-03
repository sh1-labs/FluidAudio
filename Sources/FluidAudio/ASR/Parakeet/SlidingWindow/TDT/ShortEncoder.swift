@preconcurrency import CoreML
import CryptoKit
import Foundation

/// A copy of Parakeet's encoder for short audio.
///
/// The encoder reads 15 s of mel frames however short the audio is, and takes as long for
/// 2 s as for 15. A short encoder is the same compiled program with its frame counts
/// lowered, sharing the encoder's weights. Ultra's takes 17 ms for 1,016 mel frames
/// (10.15 s) where the encoder takes 30 ms for 1,501 (M4 Mac mini, 2026-10-03). From 129
/// encoder frames, 1,025 mel frames, it takes 26 ms or more. At 1,024 mel frames it was
/// faster still, but the Neural Engine computes that size differently, and the output was
/// no longer the encoder's.
///
/// For audio that fits, its output on the audio's frames is the encoder's, bit for bit.
/// Frames past the audio are masked out of attention and of the convolution modules. The
/// subsampling's three stride-2 convolutions aren't masked, but with `melFrames` a multiple
/// of 8 they read past the audio only into frames the copy computes too. Each layer's
/// relative positions are the middle of the encoder's table.
///
/// A malformed program can crash Core ML rather than fail, so a copy is made only of an
/// encoder program it was checked on (`checkedPrograms`), and
/// `AsrManager.loadShortEncoder` compares its output with the encoder's before using it.
struct ShortEncoder: Sendable {
    let model: MLModel
    /// The mel frames it reads: audio of up to `melFrames - 1` hops of 10 ms.
    let melFrames: Int

    /// SHA-256 of the encoder programs (`model.mil`) a short encoder was checked on:
    /// Parakeet TDT 0.6B Ultra as FluidAudio 0.17.4 downloads it.
    static let checkedPrograms: Set<String> = [
        "f5d601568a4171d99a314c0fe3f6bc67715da2623732a3fb566e050ea83e5848"
    ]

    /// The encoder input `input` cut to its first `melFrames` mel frames, or nil if its
    /// audio is longer.
    static func cut(_ input: MLFeatureProvider, to melFrames: Int) throws -> MLFeatureProvider? {
        guard let mel = input.featureValue(for: "mel")?.multiArrayValue,
            let length = input.featureValue(for: "mel_length")?.multiArrayValue,
            mel.dataType == .float32, mel.shape.count == 3, mel.shape[2].intValue >= melFrames,
            length[0].intValue <= melFrames
        else { return nil }
        let bins = mel.shape[1].intValue
        let cut = try MLMultiArray(shape: [1, mel.shape[1], NSNumber(value: melFrames)], dataType: .float32)
        let sourceStrides = mel.strides.map(\.intValue)
        mel.withUnsafeBufferPointer(ofType: Float.self) { source in
            cut.withUnsafeMutableBufferPointer(ofType: Float.self) { target, targetStrides in
                for bin in 0..<bins {
                    for frame in 0..<melFrames {
                        target[bin * targetStrides[1] + frame * targetStrides[2]] =
                            source[bin * sourceStrides[1] + frame * sourceStrides[2]]
                    }
                }
            }
        }
        return try MLDictionaryFeatureProvider(dictionary: ["mel": cut, "mel_length": length])
    }

    /// Throws unless this encoder's output is `encoder`'s on the audio's frames, for audio
    /// as long as it reads and for a third of that.
    func check(against encoder: MLModel) async throws {
        guard let shape = encoder.modelDescription.inputDescriptionsByName["mel"]?.multiArrayConstraint?.shape,
            shape.count == 3
        else { throw ASRError.processingFailed("The encoder has no mel input") }
        let (bins, frames) = (shape[1].intValue, shape[2].intValue)
        for length in [melFrames, melFrames / 3] {
            let mel = try MLMultiArray(shape: shape, dataType: .float32)
            mel.withUnsafeMutableBufferPointer(ofType: Float.self) { values, strides in
                // Made-up values in a normalized mel's range, and zeros past the audio as
                // the preprocessor leaves them.
                for bin in 0..<bins {
                    for frame in 0..<frames {
                        values[bin * strides[1] + frame * strides[2]] =
                            frame < length ? 2 * Float(sin(Double(bin * 31 + frame * 7))) : 0
                    }
                }
            }
            let melLength = try MLMultiArray(shape: [1], dataType: .int32)
            melLength[0] = NSNumber(value: length)
            let input = try MLDictionaryFeatureProvider(dictionary: ["mel": mel, "mel_length": melLength])
            guard let cut = try Self.cut(input, to: melFrames) else {
                throw ASRError.processingFailed("The short encoder can't read \(length) mel frames")
            }
            let expected = try await encoder.prediction(from: input)
            let actual = try await model.prediction(from: cut)
            guard Self.sameOutput(expected, actual) else {
                throw ASRError.processingFailed("The short encoder's output differs from the encoder's")
            }
        }
    }

    /// Whether two encoder outputs have the same length and, on those frames, the same values.
    static func sameOutput(_ first: MLFeatureProvider, _ second: MLFeatureProvider) -> Bool {
        guard let a = first.featureValue(for: "encoder")?.multiArrayValue,
            let b = second.featureValue(for: "encoder")?.multiArrayValue,
            let length = first.featureValue(for: "encoder_length")?.multiArrayValue?[0].intValue,
            second.featureValue(for: "encoder_length")?.multiArrayValue?[0].intValue == length,
            a.dataType == .float32, b.dataType == .float32, a.shape.count == 3, b.shape.count == 3,
            a.shape[1] == b.shape[1], a.shape[2].intValue >= length, b.shape[2].intValue >= length
        else { return false }
        let hidden = a.shape[1].intValue
        let (aStrides, bStrides) = (a.strides.map(\.intValue), b.strides.map(\.intValue))
        return a.withUnsafeBufferPointer(ofType: Float.self) { a in
            b.withUnsafeBufferPointer(ofType: Float.self) { b in
                for unit in 0..<hidden {
                    for frame in 0..<length
                    where a[unit * aStrides[1] + frame * aStrides[2]] != b[unit * bStrides[1] + frame * bStrides[2]] {
                        return false
                    }
                }
                return true
            }
        }
    }
}

extension ShortEncoder {

    /// Writes a short encoder for `melFrames` mel frames into `directory`, made from the
    /// compiled encoder at `source`, and returns its URL. Files already there and the same
    /// are left alone, so Core ML's compiled copy of them stays valid.
    static func write(from source: URL, melFrames: Int, into directory: URL) throws -> URL {
        let program = try Data(contentsOf: source.appendingPathComponent("model.mil"))
        let digest = SHA256.hash(data: program).map { String(format: "%02x", $0) }.joined()
        guard checkedPrograms.contains(digest) else {
            throw ASRError.processingFailed("No short encoder has been checked on this encoder")
        }
        let text = String(decoding: program, as: UTF8.self)
        let files = [
            "model.mil": Data(try Self.program(text, melFrames: melFrames).utf8),
            "coremldata.bin": try Self.description(
                Data(contentsOf: source.appendingPathComponent("coremldata.bin")), of: text, melFrames: melFrames),
        ]
        let weights = source.appendingPathComponent("weights")
        let url = directory.appendingPathComponent("Encoder-\(melFrames).mlmodelc")
        let fileManager = FileManager.default
        if files.allSatisfy({ (try? Data(contentsOf: url.appendingPathComponent($0.key))) == $0.value }),
            (try? fileManager.destinationOfSymbolicLink(atPath: url.appendingPathComponent("weights").path))
                == weights.path
        {
            return url
        }
        // Put together beside it and then moved in, so nothing loads half a model.
        let assembling = directory.appendingPathComponent(UUID().uuidString)
        try fileManager.createDirectory(at: assembling, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: assembling) }
        for (name, data) in files {
            try data.write(to: assembling.appendingPathComponent(name))
        }
        try fileManager.createSymbolicLink(
            at: assembling.appendingPathComponent("weights"), withDestinationURL: weights)
        try? fileManager.removeItem(at: url)
        try fileManager.moveItem(at: assembling, to: url)
        return url
    }

    /// Frame counts through the subsampling's three stride-2 convolutions, mel frames first.
    static func frameCounts(_ melFrames: Int) -> [Int] {
        (0..<3).reduce(into: [melFrames]) { counts, _ in counts.append((counts[counts.count - 1] - 1) / 2 + 1) }
    }

    /// The mel input's shape and the encoder output's, as `program` declares them.
    static func shapes(of program: String) throws -> (mel: [Int], encoder: [Int]) {
        func shape(after marker: String) -> [Int]? {
            guard let end = program.range(of: marker)?.lowerBound,
                let start = program[..<end].range(of: "[", options: .backwards)?.upperBound
            else { return nil }
            let shape = program[start..<end].dropLast().split(separator: ", ").compactMap { Int($0) }
            return shape.count == 3 ? shape : nil
        }
        guard let mel = shape(after: "> mel,"), let encoder = shape(after: "> encoder = ") else {
            throw ASRError.processingFailed("The encoder program declares no mel input or encoder output")
        }
        return (mel, encoder)
    }

    /// The start of a declaration: indent, type, dimensions, name and operation.
    private static let declaration = try! NSRegularExpression(
        pattern: #"^(\s*)tensor<(\w+), \[([0-9, ]*)\]> (\w+) = (\w+)\("#)
    /// The dimensions in a type, and the values of an int32 list.
    private static let lists = [
        try! NSRegularExpression(pattern: #"tensor<\w+, \[([0-9, ]*)\]"#),
        try! NSRegularExpression(pattern: #"val = tensor<int32, \[\d+\]>\(\[([-0-9, ]+)\]\)"#),
    ]

    /// The encoder program `text`, reading `melFrames` mel frames.
    static func program(_ text: String, melFrames: Int) throws -> String {
        let full = frameCounts(try shapes(of: text).mel[2])
        let short = frameCounts(melFrames)
        let (fullFrames, frames) = (full[3], short[3])
        guard melFrames % 8 == 0, melFrames < full[0] else {
            throw ASRError.processingFailed("A short encoder reads fewer mel frames, a multiple of 8")
        }
        let subsampling = Dictionary(uniqueKeysWithValues: zip(full, short))
        // After it: the frames; the frames with 4 of padding either side, for the
        // convolution modules' 9-wide kernels; and the relative positions, 2T - 1, padded
        // to 2T to shift them.
        let layers = [
            fullFrames: frames, fullFrames + 8: frames + 8,
            2 * fullFrames - 1: 2 * frames - 1, 2 * fullFrames: 2 * frames,
        ]
        var (inLayers, positions, ranges, masks) = (false, 0, 0, 0)
        var lines: [String] = []
        for line in text.components(separatedBy: "\n") {
            if line.contains("]> encoder_length = cast(") { inLayers = true }
            let range = NSRange(location: 0, length: (line as NSString).length)
            guard let match = declaration.firstMatch(in: line, range: range) else {
                lines.append(remapped(line, inLayers ? layers : subsampling))
                continue
            }
            let part = { (index: Int) in (line as NSString).substring(with: match.range(at: index)) }
            let (indent, type, name, operation) = (part(1), part(2), part(4), part(5))
            let shape = part(3).split(separator: ", ").compactMap { Int($0) }
            if type == "fp16", operation == "constexpr_affine_dequantize", shape.count == 4,
                shape[3] == 2 * fullFrames - 1
            {
                // A layer's relative positions: the middle 2T - 1 of the table.
                let (heads, size) = (shape[1], shape[2])
                lines.append(line.replacingOccurrences(of: "> \(name) = ", with: "> \(name)_full = "))
                lines.append(
                    "\(indent)tensor<int32, [4]> \(name)_b = const()[name = tensor<string, []>(\"\(name)_b\"), "
                        + "val = tensor<int32, [4]>([0, 0, 0, \(fullFrames - frames)])];")
                lines.append(
                    "\(indent)tensor<int32, [4]> \(name)_e = const()[name = tensor<string, []>(\"\(name)_e\"), "
                        + "val = tensor<int32, [4]>([1, \(heads), \(size), \(fullFrames - 1 + frames)])];")
                lines.append(
                    "\(indent)tensor<fp16, [1, \(heads), \(size), \(2 * frames - 1)]> \(name) = slice_by_index("
                        + "begin = \(name)_b, end = \(name)_e, x = \(name)_full)"
                        + "[name = tensor<string, []>(\"\(name)_middle\")];")
                positions += 1
            } else if type == "int32", operation == "const", shape == [1, fullFrames] {
                // The frame indices the padding mask compares with the length.
                let indices = (0..<frames).map(String.init).joined(separator: ", ")
                lines.append(
                    "\(indent)tensor<int32, [1, \(frames)]> \(name) = const()[name = tensor<string, []>(\"\(name)\"), "
                        + "val = tensor<int32, [1, \(frames)]>([[\(indices)]])];")
                ranges += 1
            } else if type == "bool", operation == "const", shape == [1, fullFrames, fullFrames] {
                // The attention mask before the padding is taken out of it: all true.
                let row = "[" + Array(repeating: "true", count: frames).joined(separator: ", ") + "]"
                let rows = Array(repeating: row, count: frames).joined(separator: ", ")
                let type = "tensor<bool, [1, \(frames), \(frames)]>"
                lines.append(
                    "\(indent)\(type) \(name) = const()[name = tensor<string, []>(\"\(name)\"), "
                        + "val = \(type)([[\(rows)]])];")
                masks += 1
            } else {
                lines.append(remapped(line, inLayers ? layers : subsampling))
            }
        }
        guard inLayers, positions > 0, ranges == 1, masks == 1 else {
            throw ASRError.processingFailed("The encoder program isn't laid out as expected")
        }
        return lines.joined(separator: "\n")
    }

    /// `line` with the dimensions in its types and the values in its int32 lists mapped by
    /// `table`.
    static func remapped(_ line: String, _ table: [Int: Int]) -> String {
        lists.reduce(line) { line, list in
            let text = line as NSString
            var (result, end) = ("", 0)
            for match in list.matches(in: line, range: NSRange(location: 0, length: text.length)) {
                let numbers = match.range(at: 1)
                result += text.substring(with: NSRange(location: end, length: numbers.location - end))
                result += text.substring(with: numbers).split(separator: ", ")
                    .map { Int($0).map { String(table[$0] ?? $0) } ?? String($0) }.joined(separator: ", ")
                end = numbers.location + numbers.length
            }
            return result + text.substring(from: end)
        }
    }

    /// The model description `data` of the encoder program `program`, with the mel input's
    /// and the encoder output's frame counts for `melFrames`. Each count is written as a
    /// two-byte varint, so no length in the description changes.
    static func description(_ data: Data, of program: String, melFrames: Int) throws -> Data {
        let shapes = try shapes(of: program)
        var data = data
        for (shape, frames) in [(shapes.mel, melFrames), (shapes.encoder, frameCounts(melFrames)[3])] {
            guard frames < 1 << 14 else { throw ASRError.processingFailed("Too many frames for a short encoder") }
            let original = packed(shape.map(varint))
            let changed = packed(shape.dropLast().map(varint) + [[UInt8(frames & 0x7f) | 0x80, UInt8(frames >> 7)]])
            guard let range = data.range(of: original), data[range.upperBound...].range(of: original) == nil,
                original.count == changed.count
            else { throw ASRError.processingFailed("The encoder's description isn't laid out as expected") }
            data.replaceSubrange(range, with: changed)
        }
        return data
    }

    /// A shape as the description stores it, a packed repeated field: tag, byte count, then
    /// each dimension.
    private static func packed(_ dimensions: [[UInt8]]) -> Data {
        let bytes = dimensions.flatMap { $0 }
        return Data([0x0a, UInt8(bytes.count)] + bytes)
    }

    private static func varint(_ value: Int) -> [UInt8] {
        var (value, bytes) = (value, [UInt8]())
        repeat {
            bytes.append(UInt8(value & 0x7f) | (value >> 7 > 0 ? 0x80 : 0))
            value >>= 7
        } while value > 0
        return bytes
    }
}

extension AsrManager {

    /// Makes a short encoder (`ShortEncoder`) of the compiled encoder at `source`, in
    /// `directory`, and from then on uses it for audio of up to `melFrames` mel frames
    /// (10.15 s by default), once its output has matched the encoder's.
    ///
    /// The first time, Core ML compiles it for the Neural Engine, which can take 15 s;
    /// transcriptions meanwhile use the encoder.
    public func loadShortEncoder(from source: URL, into directory: URL, melFrames: Int = 1016) async throws {
        guard let encoder = encoderModel else { throw ASRError.notInitialized }
        // Written off this actor, so transcriptions don't wait for it.
        let url = try await Task.detached {
            try ShortEncoder.write(from: source, melFrames: melFrames, into: directory)
        }
        .value
        let model = try await MLModel.load(contentsOf: url, configuration: encoder.configuration)
        let shortEncoder = ShortEncoder(model: model, melFrames: melFrames)
        try await shortEncoder.check(against: encoder)
        // Models loaded meanwhile need their own.
        guard encoderModel === encoder else { return }
        self.shortEncoder = shortEncoder
        logger.info("Short encoder in use for up to \(melFrames) mel frames")
    }
}
