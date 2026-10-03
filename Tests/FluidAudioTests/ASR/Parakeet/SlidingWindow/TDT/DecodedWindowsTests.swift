import Foundation
import XCTest

@testable import FluidAudio

final class DecodedWindowsTests: XCTestCase {

    private actor Calls {
        private(set) var count = 0
        func add() { count += 1 }
    }

    private struct Failure: Error {}

    private func key(start: Int = 0, isLastChunk: Bool = false) -> DecodedWindows.Key {
        DecodedWindows.Key(
            firstSample: start, sampleCount: 3, contextSamples: 0, chunkStart: start, isLastChunk: isLastChunk,
            emitTokensAfterFrame: nil, language: nil)
    }

    private func decoder(token: Int, calls: Calls) -> @Sendable () async throws -> DecodedWindows.Tokens {
        {
            await calls.add()
            return [(token: token, timestamp: 0, confidence: 1, duration: 1)]
        }
    }

    func testReusesAWindowWithTheSameSamplesAndSettings() async throws {
        let windows = DecodedWindows()
        let calls = Calls()
        _ = try await windows.tokens(for: key(), samples: [0, 1, 2], decode: decoder(token: 1, calls: calls))
        let again = try await windows.tokens(for: key(), samples: [0, 1, 2], decode: decoder(token: 2, calls: calls))

        XCTAssertEqual(again.map(\.token), [1])
        let count = await calls.count
        XCTAssertEqual(count, 1)
    }

    func testDecodesAgainWhenTheSamplesDiffer() async throws {
        let windows = DecodedWindows()
        let calls = Calls()
        _ = try await windows.tokens(for: key(), samples: [0, 1, 2], decode: decoder(token: 1, calls: calls))
        let changed = try await windows.tokens(for: key(), samples: [0, 1, 3], decode: decoder(token: 2, calls: calls))

        XCTAssertEqual(changed.map(\.token), [2])
        let count = await calls.count
        XCTAssertEqual(count, 2)
    }

    func testDecodesAgainWhenTheSettingsDiffer() async throws {
        let windows = DecodedWindows()
        let calls = Calls()
        _ = try await windows.tokens(for: key(), samples: [0, 1, 2], decode: decoder(token: 1, calls: calls))
        let last = try await windows.tokens(
            for: key(isLastChunk: true), samples: [0, 1, 2], decode: decoder(token: 2, calls: calls))

        XCTAssertEqual(last.map(\.token), [2])
        let count = await calls.count
        XCTAssertEqual(count, 2)
    }

    func testForgetsAWindowThatFailed() async throws {
        let windows = DecodedWindows()
        let calls = Calls()
        do {
            _ = try await windows.tokens(for: key(), samples: [0, 1, 2]) { throw Failure() }
            XCTFail("Expected the failure")
        } catch is Failure {}
        let retried = try await windows.tokens(for: key(), samples: [0, 1, 2], decode: decoder(token: 1, calls: calls))

        XCTAssertEqual(retried.map(\.token), [1])
        let count = await calls.count
        XCTAssertEqual(count, 1)
    }
}
