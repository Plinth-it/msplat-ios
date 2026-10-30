import UIKit

/// Disables auto-lock while at least one long-running activity holds an
/// assertion. Training, benchmarks, ARKit capture, and RealityKit alignment
/// can each run for minutes without a touch, and auto-lock would background
/// the app mid-run. Assertions are tracked individually, so one activity
/// ending cannot re-enable auto-lock while another is still running.
@MainActor
final class ScreenAwakeController {
    static let shared = ScreenAwakeController { isIdleTimerDisabled in
        UIApplication.shared.isIdleTimerDisabled = isIdleTimerDisabled
    }

    private let setIdleTimerDisabled: @MainActor (Bool) -> Void
    private var activeAssertionIDs: Set<UInt64> = []
    private var nextAssertionID: UInt64 = 0

    init(setIdleTimerDisabled: @escaping @MainActor (Bool) -> Void) {
        self.setIdleTimerDisabled = setIdleTimerDisabled
    }

    var keepsScreenAwake: Bool { !activeAssertionIDs.isEmpty }

    func makeAssertion() -> ScreenAwakeAssertion {
        nextAssertionID &+= 1
        return ScreenAwakeAssertion(controller: self, id: nextAssertionID)
    }

    fileprivate func setAssertion(_ id: UInt64, isActive: Bool) {
        let wasKeepingScreenAwake = keepsScreenAwake
        if isActive {
            activeAssertionIDs.insert(id)
        } else {
            activeAssertionIDs.remove(id)
        }
        if keepsScreenAwake != wasKeepingScreenAwake {
            setIdleTimerDisabled(keepsScreenAwake)
        }
    }
}

/// One activity's hold on ``ScreenAwakeController``. Setting `isActive` is
/// idempotent, so an owner can mirror its busy state on every phase change.
/// Owners deactivate explicitly; a released assertion does not end itself.
@MainActor
final class ScreenAwakeAssertion {
    private let controller: ScreenAwakeController
    private let id: UInt64

    var isActive = false {
        didSet {
            guard isActive != oldValue else { return }
            controller.setAssertion(id, isActive: isActive)
        }
    }

    fileprivate init(controller: ScreenAwakeController, id: UInt64) {
        self.controller = controller
        self.id = id
    }
}
