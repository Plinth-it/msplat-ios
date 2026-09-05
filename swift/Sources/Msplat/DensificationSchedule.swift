/// Opportunities to refine topology, before gradient and population limits apply.
public struct DensificationSchedule: Sendable, Equatable {
    /// False when the effective cutoff explicitly excludes all training steps.
    public let isEnabled: Bool
    public let firstStep: Int?
    public let eventCount: Int
}

extension TrainingConfig {
    /// Resolve the schedule using the actual training-camera count after any
    /// evaluation split. This matches the native Model's refinement predicate.
    public func densificationSchedule(trainingCameraCount: Int) throws -> DensificationSchedule {
        try validate()
        guard (1...Int(Int32.max)).contains(trainingCameraCount) else {
            throw MsplatError.invalidArgument("Training camera count must be in 1...2147483647")
        }
        let stopStep = stopDensifyAt >= 0 ? Int(stopDensifyAt) : Int(iterations) / 2
        let lastStep = min(Int(iterations), stopStep - 1)
        let refine = Int(refineEvery)
        let resetInterval = Int(resetAlphaEvery) * refine
        var firstStep: Int?
        var eventCount = 0
        if lastStep >= refine {
            for step in stride(from: refine, through: lastStep, by: refine) {
                if step > Int(warmupLength) &&
                    step % resetInterval > trainingCameraCount + refine {
                    if firstStep == nil { firstStep = step }
                    eventCount += 1
                }
            }
        }
        return DensificationSchedule(
            isEnabled: stopStep > 1,
            firstStep: firstStep,
            eventCount: eventCount
        )
    }

    /// Reject accidentally growth-free plans while allowing an explicit cutoff
    /// of zero or one for fixed-population training.
    @discardableResult
    public func validateDensificationSchedule(trainingCameraCount: Int) throws -> DensificationSchedule {
        let schedule = try densificationSchedule(trainingCameraCount: trainingCameraCount)
        guard !schedule.isEnabled || schedule.eventCount > 0 else {
            throw MsplatError.invalidArgument(
                "No Gaussian growth opportunities are scheduled for \(trainingCameraCount) " +
                "training cameras in \(iterations) iterations. Increase the iteration budget " +
                "or adjust the refinement/reset schedule and densification cutoff. " +
                "Set stopDensifyAt to 0 for intentional fixed-population training."
            )
        }
        return schedule
    }
}
