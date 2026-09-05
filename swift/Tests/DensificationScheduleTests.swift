import XCTest
@testable import Msplat

final class DensificationScheduleTests: XCTestCase {
    func testDefaultScheduleCameraBoundary() throws {
        let config = TrainingConfig()
        let available = try config.validateDensificationSchedule(trainingCameraCount: 2_799)
        XCTAssertEqual(available.firstStep, 2_900)
        XCTAssertEqual(available.eventCount, 5)
        let unavailable = try config.densificationSchedule(trainingCameraCount: 2_800)
        XCTAssertTrue(unavailable.isEnabled)
        XCTAssertNil(unavailable.firstStep)
        XCTAssertEqual(unavailable.eventCount, 0)
        XCTAssertThrowsError(try config.validateDensificationSchedule(trainingCameraCount: 2_800))
    }

    func testShortRunCameraBoundaryAndExplicitCutoff() throws {
        var config = TrainingConfig()
        config.iterations = 2_000
        let schedule = try config.validateDensificationSchedule(trainingCameraCount: 799)
        XCTAssertEqual(schedule.firstStep, 900)
        XCTAssertEqual(schedule.eventCount, 1)
        XCTAssertThrowsError(try config.validateDensificationSchedule(trainingCameraCount: 800))
        config.stopDensifyAt = 1_001
        XCTAssertEqual(try config.validateDensificationSchedule(trainingCameraCount: 800).firstStep, 1_000)
        config.iterations = 900
        config.stopDensifyAt = 900
        XCTAssertThrowsError(try config.validateDensificationSchedule(trainingCameraCount: 799))
        config.stopDensifyAt = 901
        XCTAssertEqual(try config.validateDensificationSchedule(trainingCameraCount: 799).eventCount, 1)
    }

    func testExplicitFixedPopulationAndWarmup() throws {
        var config = TrainingConfig()
        config.iterations = 2_000
        for cutoff: Int32 in [0, 1] {
            config.stopDensifyAt = cutoff
            let schedule = try config.validateDensificationSchedule(trainingCameraCount: 800)
            XCTAssertFalse(schedule.isEnabled)
            XCTAssertNil(schedule.firstStep)
            XCTAssertEqual(schedule.eventCount, 0)
        }
        config.stopDensifyAt = 1_000
        config.warmupLength = 1_000
        XCTAssertThrowsError(try config.validateDensificationSchedule(trainingCameraCount: 100))
    }

    func testInvalidAndLargeCameraCounts() throws {
        var config = TrainingConfig()
        XCTAssertThrowsError(try config.densificationSchedule(trainingCameraCount: 0))
        XCTAssertThrowsError(try config.densificationSchedule(trainingCameraCount: -1))
        XCTAssertThrowsError(try config.densificationSchedule(trainingCameraCount: Int.max))
        XCTAssertEqual(try config.densificationSchedule(trainingCameraCount: Int(Int32.max)).eventCount, 0)
        config.refineEvery = 0
        XCTAssertThrowsError(try config.densificationSchedule(trainingCameraCount: 1))
    }
}
