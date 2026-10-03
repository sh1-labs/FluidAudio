import CoreML
import Foundation
import XCTest

@testable import FluidAudio

final class ShortEncoderTests: XCTestCase {

    /// The kinds of line in Parakeet's compiled encoder that a short encoder changes, in
    /// its order, without the attributes the rewrite doesn't read.
    private let program = """
        program(1.0)
        {
            func main<ios17>(tensor<fp32, [1, 128, 1501]> mel, tensor<int32, [1]> mel_length) {
                tensor<int32, [3]> sub_shape = const()[val = tensor<int32, [3]>([1, 751, 376])];
                tensor<fp16, [1, 256, 751, 64]> sub = conv(x = mel);
                tensor<int32, [1, 188]> expand_dims_0 = const()[val = tensor<int32, [1, 188]>([[0, 1, 2]])];
                tensor<int32, [1]> encoder_length = cast(dtype = int32, x = lengths);
                tensor<bool, [1, 188, 188]> const_7 = const()[val = tensor<bool, [1, 188, 188]>([[[true]]])];
                tensor<fp16, [1, 8, 128, 375]> pos = constexpr_affine_dequantize()[axis = tensor<int32, []>(3)];
                tensor<int32, [4]> shift = const()[val = tensor<int32, [4]>([1, 8, -1, 188])];
                tensor<fp16, [1, 1024, 196]> padded = pad(x = x);
                tensor<fp16, [1, 8, 188, 376]> shifted = pad(x = scores);
                tensor<fp32, [1, 1024, 188]> encoder = cast(x = y);
            } -> (encoder, encoder_length);
        }
        """

    func testFrameCountsFollowTheThreeStrideTwoConvolutions() {
        XCTAssertEqual(ShortEncoder.frameCounts(1501), [1501, 751, 376, 188])
        XCTAssertEqual(ShortEncoder.frameCounts(1016), [1016, 508, 254, 127])
    }

    func testProgramLowersTheFrameCountsThroughout() throws {
        let lines = try ShortEncoder.program(program, melFrames: 1016).components(separatedBy: "\n")
        func line(_ name: String) -> String { lines.first { $0.contains("> \(name) = ") } ?? "" }

        XCTAssertTrue(lines[2].contains("tensor<fp32, [1, 128, 1016]> mel,"))
        XCTAssertTrue(line("sub_shape").contains("([1, 508, 254])"))
        XCTAssertTrue(line("sub").hasPrefix("        tensor<fp16, [1, 256, 508, 64]> sub = conv("))
        XCTAssertTrue(line("expand_dims_0").contains("val = tensor<int32, [1, 127]>([[0, 1, 2, "))
        XCTAssertTrue(line("expand_dims_0").hasSuffix(", 125, 126]])];"))
        XCTAssertTrue(line("encoder_length").contains("tensor<int32, [1]> encoder_length = cast("))
        XCTAssertTrue(line("const_7").contains("tensor<bool, [1, 127, 127]> const_7 = const()"))
        XCTAssertEqual(line("const_7").components(separatedBy: "true").count - 1, 127 * 127)
        XCTAssertTrue(line("shift").contains("([1, 8, -1, 127])"))
        XCTAssertTrue(line("padded").contains("tensor<fp16, [1, 1024, 135]> padded"))
        XCTAssertTrue(line("shifted").contains("tensor<fp16, [1, 8, 127, 254]> shifted"))
        XCTAssertTrue(line("encoder").contains("tensor<fp32, [1, 1024, 127]> encoder = cast("))
    }

    func testProgramTakesTheMiddleOfEachLayersRelativePositions() throws {
        let lines = try ShortEncoder.program(program, melFrames: 1016).components(separatedBy: "\n")
        func line(_ name: String) -> String { lines.first { $0.contains("> \(name) = ") } ?? "" }

        // The full table, unchanged, then its middle 2 × 127 - 1 positions around 187.
        XCTAssertTrue(line("pos_full").contains("tensor<fp16, [1, 8, 128, 375]> pos_full = constexpr_affine"))
        XCTAssertTrue(line("pos_b").contains("([0, 0, 0, 61])"))
        XCTAssertTrue(line("pos_e").contains("([1, 8, 128, 314])"))
        XCTAssertTrue(
            line("pos").contains("tensor<fp16, [1, 8, 128, 253]> pos = slice_by_index(begin = pos_b, end = pos_e"))
    }

    func testProgramRefusesFrameCountsThatWouldChangeTheOutput() {
        // Not a multiple of 8: the subsampling would read padding the encoder computes.
        XCTAssertThrowsError(try ShortEncoder.program(program, melFrames: 1001))
        XCTAssertThrowsError(try ShortEncoder.program(program, melFrames: 1504))
    }

    func testProgramRefusesAProgramLaidOutDifferently() {
        let withoutMask = program.components(separatedBy: "\n").filter { !$0.contains("const_7") }
            .joined(separator: "\n")
        XCTAssertThrowsError(try ShortEncoder.program(withoutMask, melFrames: 1016))
    }

    func testDescriptionChangesOnlyTheFrameCounts() throws {
        let mel: [UInt8] = [0x0a, 0x05, 0x01, 0x80, 0x01, 0xdd, 0x0b]
        let encoder: [UInt8] = [0x0a, 0x05, 0x01, 0x80, 0x08, 0xbc, 0x01]
        let data = Data([0x12, 0x03] + mel + [0x22, 0x01] + encoder + [0x2a])

        let changed = try ShortEncoder.description(data, of: program, melFrames: 1016)

        // 1,016 and 127 as two-byte varints, so no length changes.
        let expected = Data(
            [
                0x12, 0x03, 0x0a, 0x05, 0x01, 0x80, 0x01, 0xf8, 0x07, 0x22, 0x01, 0x0a, 0x05, 0x01, 0x80, 0x08, 0xff,
                0x00, 0x2a,
            ])
        XCTAssertEqual(changed, expected)
    }

    func testDescriptionRefusesAShapeItCannotFindOnce() {
        let mel: [UInt8] = [0x0a, 0x05, 0x01, 0x80, 0x01, 0xdd, 0x0b]
        XCTAssertThrowsError(try ShortEncoder.description(Data(mel), of: program, melFrames: 1016))
    }

    func testWriteRefusesAnEncoderItWasNotCheckedOn() throws {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: folder) }
        let source = folder.appendingPathComponent("Encoder.mlmodelc")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try Data(program.utf8).write(to: source.appendingPathComponent("model.mil"))

        XCTAssertThrowsError(
            try ShortEncoder.write(from: source, melFrames: 1016, into: folder.appendingPathComponent("Short")))
        XCTAssertFalse(FileManager.default.fileExists(atPath: folder.appendingPathComponent("Short").path))
    }

    func testCutKeepsTheFirstFramesOfAudioThatFits() throws {
        let mel = try MLMultiArray(shape: [1, 2, 6], dataType: .float32)
        for index in 0..<12 { mel[index] = NSNumber(value: Float(index)) }
        let length = try MLMultiArray(shape: [1], dataType: .int32)
        length[0] = 3
        let input = try MLDictionaryFeatureProvider(dictionary: ["mel": mel, "mel_length": length])

        let cut = try XCTUnwrap(try ShortEncoder.cut(input, to: 4))

        let cutMel = try XCTUnwrap(cut.featureValue(for: "mel")?.multiArrayValue)
        XCTAssertEqual(cutMel.shape, [1, 2, 4])
        XCTAssertEqual((0..<8).map { cutMel[[0, $0 / 4, $0 % 4] as [NSNumber]].floatValue }, [0, 1, 2, 3, 6, 7, 8, 9])
        XCTAssertEqual(cut.featureValue(for: "mel_length")?.multiArrayValue?[0], 3)
    }

    func testCutLeavesLongerAudioToTheEncoder() throws {
        let mel = try MLMultiArray(shape: [1, 2, 6], dataType: .float32)
        let length = try MLMultiArray(shape: [1], dataType: .int32)
        length[0] = 5
        let input = try MLDictionaryFeatureProvider(dictionary: ["mel": mel, "mel_length": length])

        XCTAssertNil(try ShortEncoder.cut(input, to: 4))
    }

    func testSameOutputComparesOnlyTheAudiosFrames() throws {
        func output(_ values: [Float], length: Int) throws -> MLFeatureProvider {
            let encoder = try MLMultiArray(shape: [1, 2, NSNumber(value: values.count / 2)], dataType: .float32)
            for (index, value) in values.enumerated() { encoder[index] = NSNumber(value: value) }
            let encoderLength = try MLMultiArray(shape: [1], dataType: .int32)
            encoderLength[0] = NSNumber(value: length)
            return try MLDictionaryFeatureProvider(dictionary: ["encoder": encoder, "encoder_length": encoderLength])
        }
        let reference = try output([1, 2, 9, 3, 4, 9], length: 2)

        XCTAssertTrue(ShortEncoder.sameOutput(reference, try output([1, 2, 3, 4], length: 2)))
        XCTAssertFalse(ShortEncoder.sameOutput(reference, try output([1, 2, 3, 5], length: 2)))
        XCTAssertFalse(ShortEncoder.sameOutput(reference, try output([1, 2, 3, 4], length: 1)))
    }
}
