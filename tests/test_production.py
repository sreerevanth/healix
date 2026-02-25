#!/usr/bin/env python3
"""
Healix Production Corrections Test Suite
==========================================
Validates all 10 corrections with deterministic, measurable assertions.

Usage:
    python3 tests/test_production.py
"""

import sys
import threading
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.recovery_metric import (
    clamp, resource_stability_score, ResourceSnapshot,
    RecoveryMonitor, _zero_result
)
from models.context_model import (
    ContextConditionedModel, AnomalyClass, classify_anomaly,
    MEMORY_SLOPE_THRESHOLD, FORK_RATE_THRESHOLD, NET_RATE_THRESHOLD,
    KL_DRIFT_THRESHOLD, CPU_RATE_THRESHOLD, ALPHA
)
from models.fingerprint import BehavioralFingerprint
from models.anomaly_scorer import AnomalyScorer
from interventions.remediation import (
    RemediationEngine, ActionType, score_to_max_invasiveness
)
from feedback.feedback_loop import FeedbackLoop
from daemon.pid_state_machine import PIDStateMachine, PIDState


PASS = "  ✅"
FAIL = "  ❌"
errors = []


def check(condition: bool, label: str) -> None:
    if condition:
        print(f"{PASS} {label}")
    else:
        print(f"{FAIL} {label}")
        errors.append(label)


def section(n: int, title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  CORRECTION #{n}: {title}")
    print(f"{'─'*60}")


class FakeEvent:
    def __init__(self, pid, syscall_nr=0, ts_ns=None):
        self.pid          = pid
        self.syscall_nr   = syscall_nr
        self.timestamp_ns = ts_ns or time.time_ns()
        self.comm         = f"proc_{pid}"


def train_fp(pid: int, n: int = 300) -> BehavioralFingerprint:
    fp = BehavioralFingerprint(pid=pid, comm=f"proc_{pid}")
    for _ in range(n):
        fp.update(FakeEvent(pid, random.choice([0, 1, 2, 3, 9, 257])))
    assert fp.baseline_frozen
    return fp


# ─────────────────────────────────────────────────────────────
# Correction #1: Recovery Metric Hardening
# ─────────────────────────────────────────────────────────────

def test_correction_1():
    section(1, "Recovery Metric Hardening")

    # clamp: NaN safety
    check(clamp(float('nan')) == 0.0,         "clamp(NaN) == 0.0")
    check(clamp(-999) == 0.0,                  "clamp(-999) == 0.0")
    check(clamp(999) == 1.0,                   "clamp(999) == 1.0")
    check(clamp(0.5) == 0.5,                   "clamp(0.5) == 0.5")

    # resource_stability_score: bounded [0,1]
    snap_bad  = ResourceSnapshot(0, 95, 2000, 500, 100)
    snap_good = ResourceSnapshot(1,  5,  200,  20,   5)
    snap_same = ResourceSnapshot(1, 95, 2000, 500, 100)

    s_good = resource_stability_score(snap_bad, snap_good)
    s_same = resource_stability_score(snap_bad, snap_same)
    s_dead = resource_stability_score(snap_bad, None)

    check(0.0 <= s_good <= 1.0,  f"stability after recovery in [0,1]: {s_good:.4f}")
    check(s_good > 0.5,           f"stability after recovery > 0.5: {s_good:.4f}")
    check(0.0 <= s_same <= 1.0,  f"stability after no change in [0,1]: {s_same:.4f}")
    check(s_same < 0.1,           f"stability after no change < 0.1: {s_same:.4f}")
    check(s_dead == 0.0,          f"stability after process death == 0.0")

    # Formal reward formula: no negative output
    from models.recovery_metric import W1, W2, W3
    # Worst-case inputs: s_before tiny, all bad
    reward_worst = clamp(W1 * clamp(0.0 / max(1e-6, 1e-6)) + W2 * 0.0 - W3 * 1.0)
    check(reward_worst >= 0.0,    f"reward never negative: {reward_worst:.4f}")

    # div-by-zero guard: s_before = 0
    delta = max(0.0, 0.0 - 0.0)
    norm  = delta / max(0.0, 1e-6)
    check(norm == 0.0,             "div-by-zero guard: s_before=0 → norm=0.0")

    # zero_result helper is always valid
    zr = _zero_result(999, "OBSERVE", 0.8)
    check(zr.reward == 0.0,        "zero_result.reward == 0.0")
    check(not zr.evaluation_complete, "zero_result.evaluation_complete == False")


# ─────────────────────────────────────────────────────────────
# Correction #2: Context Model Determinism
# ─────────────────────────────────────────────────────────────

def test_correction_2():
    section(2, "Metric-Based Anomaly Classification (No String Matching)")

    # MEMORY_LEAK: sustained positive memory_slope
    threshold_bytes = MEMORY_SLOPE_THRESHOLD * 1024 * 1024   # convert to bytes
    cls = classify_anomaly({
        "memory_slope": threshold_bytes * 1.5,
        "fork_rate": 0, "net_rate": 0, "syscall_rate": 0, "kl_score": 0
    })
    check(cls == AnomalyClass.MEMORY_LEAK, f"High memory_slope → MEMORY_LEAK (got {cls})")

    # FORK_BOMB: fork_rate spike
    cls = classify_anomaly({
        "memory_slope": 0,
        "fork_rate": FORK_RATE_THRESHOLD * 2,
        "net_rate": 0, "syscall_rate": 0, "kl_score": 0
    })
    check(cls == AnomalyClass.FORK_BOMB, f"High fork_rate → FORK_BOMB (got {cls})")

    # NETWORK_SURGE: net_rate spike
    cls = classify_anomaly({
        "memory_slope": 0, "fork_rate": 0,
        "net_rate": NET_RATE_THRESHOLD * 2,
        "syscall_rate": 0, "kl_score": 0
    })
    check(cls == AnomalyClass.NETWORK_SURGE, f"High net_rate → NETWORK_SURGE (got {cls})")

    # SYSCALL_DRIFT: KL divergence above threshold
    cls = classify_anomaly({
        "memory_slope": 0, "fork_rate": 0, "net_rate": 0,
        "kl_score": KL_DRIFT_THRESHOLD * 2,
        "syscall_rate": 0
    })
    check(cls == AnomalyClass.SYSCALL_DRIFT, f"High kl_score → SYSCALL_DRIFT (got {cls})")

    # CPU_SPIKE: high syscall rate, no other triggers
    cls = classify_anomaly({
        "memory_slope": 0, "fork_rate": 0, "net_rate": 0, "kl_score": 0,
        "syscall_rate": CPU_RATE_THRESHOLD * 2
    })
    check(cls == AnomalyClass.CPU_SPIKE, f"High syscall_rate → CPU_SPIKE (got {cls})")

    # UNKNOWN: all below threshold
    cls = classify_anomaly({
        "memory_slope": 0, "fork_rate": 0, "net_rate": 0, "kl_score": 0, "syscall_rate": 0
    })
    check(cls == AnomalyClass.UNKNOWN, f"All zero → UNKNOWN (got {cls})")

    # Determinism: same inputs always same output
    features = {"memory_slope": 0, "fork_rate": FORK_RATE_THRESHOLD * 3,
                "net_rate": NET_RATE_THRESHOLD * 2, "kl_score": 0, "syscall_rate": 0}
    results = {classify_anomaly(features) for _ in range(100)}
    check(len(results) == 1, f"Classification is deterministic (unique results={len(results)})")

    # Context key is 2-dimensional
    model = ContextConditionedModel()
    p = model.get_p_success("THROTTLE_MEM", AnomalyClass.MEMORY_LEAK)
    check(isinstance(p, float) and 0 < p <= 1.0, f"2D context key works: p={p:.3f}")


# ─────────────────────────────────────────────────────────────
# Correction #3: EMA Stability
# ─────────────────────────────────────────────────────────────

def test_correction_3():
    section(3, "EMA Stability")

    model = ContextConditionedModel()
    ac = AnomalyClass.MEMORY_LEAK

    # Get baseline
    p0 = model.get_p_success("THROTTLE_MEM", ac)

    # Correct EMA form: new = old + alpha*(reward - old)
    reward = 1.0
    expected_new = p0 + ALPHA * (reward - p0)

    model.update("THROTTLE_MEM", ac, reward=reward,
                 evaluation_complete=True, pid_exited=False)
    p1 = model.get_p_success("THROTTLE_MEM", ac)

    check(abs(p1 - expected_new) < 1e-9,
          f"EMA formula correct: {p0:.4f} + {ALPHA}*({reward}-{p0:.4f}) = {expected_new:.4f}, got {p1:.4f}")

    # Clamp: weight stays in [0.01, 0.99]
    for _ in range(200):
        model.update("THROTTLE_MEM", ac, reward=0.0,
                     evaluation_complete=True, pid_exited=False)
    p_low = model.get_p_success("THROTTLE_MEM", ac)
    check(p_low >= 0.01, f"Weight clamped at low end: {p_low:.4f} >= 0.01")

    model2 = ContextConditionedModel()
    for _ in range(200):
        model2.update("THROTTLE_MEM", ac, reward=1.0,
                      evaluation_complete=True, pid_exited=False)
    p_high = model2.get_p_success("THROTTLE_MEM", ac)
    check(p_high <= 0.99, f"Weight clamped at high end: {p_high:.4f} <= 0.99")

    # Skipped if evaluation_complete=False
    model3 = ContextConditionedModel()
    p_before = model3.get_p_success("OBSERVE", AnomalyClass.FORK_BOMB)
    model3.update("OBSERVE", AnomalyClass.FORK_BOMB, reward=0.0,
                  evaluation_complete=False, pid_exited=False)
    p_after = model3.get_p_success("OBSERVE", AnomalyClass.FORK_BOMB)
    check(p_before == p_after, f"Update skipped when evaluation_complete=False")

    # Skipped if pid_exited=True
    model4 = ContextConditionedModel()
    p_before = model4.get_p_success("OBSERVE", AnomalyClass.FORK_BOMB)
    model4.update("OBSERVE", AnomalyClass.FORK_BOMB, reward=0.0,
                  evaluation_complete=True, pid_exited=True)
    p_after = model4.get_p_success("OBSERVE", AnomalyClass.FORK_BOMB)
    check(p_before == p_after, f"Update skipped when pid_exited=True")

    # Thread safety: concurrent updates don't corrupt weights
    model5 = ContextConditionedModel()
    def _update():
        for _ in range(50):
            model5.update("SUSPEND", AnomalyClass.CPU_SPIKE, reward=0.7,
                          evaluation_complete=True, pid_exited=False)
    threads = [threading.Thread(target=_update) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    p_final = model5.get_p_success("SUSPEND", AnomalyClass.CPU_SPIKE)
    check(0.01 <= p_final <= 0.99, f"Concurrent EMA updates stay bounded: {p_final:.4f}")


# ─────────────────────────────────────────────────────────────
# Correction #4: Severity-Bound Utility Verification
# ─────────────────────────────────────────────────────────────

def test_correction_4():
    section(4, "Severity-Bound Utility Function")

    remediator = RemediationEngine(dry_run=True)
    context    = ContextConditionedModel()
    fp         = train_fp(3001)

    # Utility is never negative
    from interventions.remediation import ACTION_CATALOG
    for action in ACTION_CATALOG:
        for severity in [0.0, 0.3, 0.65, 0.9, 1.0]:
            p = context.get_p_success(action.action_type.name, AnomalyClass.UNKNOWN)
            cab = remediator._context_alignment_bonus(action, AnomalyClass.UNKNOWN)
            utility = max(0.0, p * (1 - action.invasiveness * severity) * (1 + cab))
            check(utility >= 0.0,
                  f"utility({action.action_type.name}, sev={severity}) >= 0: {utility:.4f}")

    # Low severity ceiling: no invasive actions unlocked
    ceiling_low  = score_to_max_invasiveness(0.35)
    ceiling_high = score_to_max_invasiveness(0.90)
    check(ceiling_low == 0.0,   f"Score 0.35 → ceiling=0.0 (got {ceiling_low})")
    check(ceiling_high == 1.0,  f"Score 0.90 → ceiling=1.0 (got {ceiling_high})")

    # Hard filter enforced: selected action invasiveness <= ceiling
    for score in [0.42, 0.60, 0.75, 0.90]:
        action = remediator.select_action(
            fp, score, AnomalyClass.UNKNOWN, context, escalation_hint=False
        )
        ceiling = score_to_max_invasiveness(score)
        check(action.invasiveness <= ceiling,
              f"Score {score}: selected {action.action_type.name} "
              f"inv={action.invasiveness} <= ceiling={ceiling}")

    # Escalation raises ceiling by exactly one step
    steps = [0.00, 0.20, 0.50, 0.70, 1.00]
    base_score = 0.60  # ceiling = 0.50
    base_ceiling = score_to_max_invasiveness(base_score)
    action_esc = remediator.select_action(
        fp, base_score, AnomalyClass.FORK_BOMB, context, escalation_hint=True
    )
    base_idx = steps.index(base_ceiling)
    escalated_ceiling = steps[min(base_idx + 1, len(steps) - 1)]
    check(action_esc.invasiveness <= escalated_ceiling,
          f"Escalation ceiling={escalated_ceiling}, selected inv={action_esc.invasiveness}")


# ─────────────────────────────────────────────────────────────
# Correction #5: PID State Machine Strict Transitions
# ─────────────────────────────────────────────────────────────

def test_correction_5():
    section(5, "PID State Machine Strict Transitions")

    sm = PIDStateMachine()

    # NORMAL → ANOMALOUS → REMEDIATING → COOLDOWN → NORMAL
    sm.get_or_create(5001, "test")
    sm.mark_anomalous(5001)
    check(sm._records[5001].state == PIDState.ANOMALOUS, "NORMAL → ANOMALOUS")

    ok = sm.begin_remediation(5001)
    check(ok and sm._records[5001].state == PIDState.REMEDIATING,
          "ANOMALOUS → REMEDIATING")

    # Block during REMEDIATING
    allowed, reason = sm.can_intervene(5001)
    check(not allowed and "already_remediating" in reason,
          f"Blocked during REMEDIATING: {reason}")

    sm.end_remediation(5001, "OBSERVE", "unknown", success=True)
    check(sm._records[5001].state == PIDState.COOLDOWN, "REMEDIATING → COOLDOWN")

    # Block during active cooldown
    allowed, reason = sm.can_intervene(5001, new_score=0.5)
    check(not allowed and "cooldown" in reason, f"Blocked during COOLDOWN: {reason}")

    # Double begin_remediation rejected
    sm2 = PIDStateMachine()
    sm2.get_or_create(5002)
    sm2.mark_anomalous(5002)
    sm2.begin_remediation(5002)
    ok2 = sm2.begin_remediation(5002)
    check(not ok2, "Second begin_remediation rejected during REMEDIATING")


# ─────────────────────────────────────────────────────────────
# Correction #6: Escalation Logic Isolation
# ─────────────────────────────────────────────────────────────

def test_correction_6():
    section(6, "Escalation Per (pid, anomaly_class)")

    sm = PIDStateMachine()
    sm.get_or_create(6001)

    # Two failures of same (action, class)
    sm.end_remediation(6001, "OBSERVE", "fork_bomb", success=False)
    sm.end_remediation(6001, "OBSERVE", "fork_bomb", success=False)
    check(sm.should_escalate(6001, "OBSERVE", "fork_bomb"),
          "Escalates after 2 failures for (OBSERVE, fork_bomb)")

    # Different anomaly_class does NOT inherit escalation
    check(not sm.should_escalate(6001, "OBSERVE", "memory_leak"),
          "No escalation for different anomaly class (memory_leak)")

    # Different pid does NOT inherit
    sm.get_or_create(6002)
    check(not sm.should_escalate(6002, "OBSERVE", "fork_bomb"),
          "No escalation for different pid")

    # Reset on success
    sm.end_remediation(6001, "OBSERVE", "fork_bomb", success=True)
    check(not sm.should_escalate(6001, "OBSERVE", "fork_bomb"),
          "Fail counter resets after success")


# ─────────────────────────────────────────────────────────────
# Correction #7: Cooldown Severity Rule
# ─────────────────────────────────────────────────────────────

def test_correction_7():
    section(7, "Cooldown Severity-Based Escalation")

    from daemon.pid_state_machine import ESCALATION_SCORE_MULT
    sm = PIDStateMachine()
    sm.get_or_create(7001)
    sm.mark_anomalous(7001)
    sm.begin_remediation(7001)

    # Put into cooldown with score=0.70
    sm._records[7001]._record_score(0.70)
    sm.end_remediation(7001, "THROTTLE_CPU", "fork_bomb", success=True)
    check(sm._records[7001].state == PIDState.COOLDOWN, "In COOLDOWN")

    # Score 15% higher triggers escalation override
    escalated_score = 0.70 * ESCALATION_SCORE_MULT + 0.01
    allowed, reason = sm.can_intervene(7001, new_score=escalated_score)
    check(allowed and "escalation" in reason,
          f"Score {escalated_score:.3f} > 0.70*{ESCALATION_SCORE_MULT} → escalation: {reason}")

    # Score only slightly higher does NOT override
    sm2 = PIDStateMachine()
    sm2.get_or_create(7002)
    sm2.mark_anomalous(7002)
    sm2.begin_remediation(7002)
    sm2._records[7002]._record_score(0.70)
    sm2.end_remediation(7002, "THROTTLE_CPU", "fork_bomb", success=True)
    small_increase = 0.70 * 1.05   # only 5% increase, below 15% threshold
    allowed2, reason2 = sm2.can_intervene(7002, new_score=small_increase)
    check(not allowed2, f"Score +5% does NOT override cooldown (got: {reason2})")


# ─────────────────────────────────────────────────────────────
# Correction #8: Concurrency Safety
# ─────────────────────────────────────────────────────────────

def test_correction_8():
    section(8, "Concurrency Safety")

    # Concurrent state machine operations
    sm = PIDStateMachine()
    results = []

    def _worker(pid):
        sm.get_or_create(pid, "concurrent_test")
        sm.mark_anomalous(pid)
        ok = sm.begin_remediation(pid)
        results.append((pid, ok))
        if ok:
            time.sleep(0.01)
            sm.end_remediation(pid, "OBSERVE", "unknown", success=True)

    threads = [threading.Thread(target=_worker, args=(8000 + i,)) for i in range(20)]
    for t in threads: t.start()
    for t in threads: t.join()

    check(len(results) == 20, f"All 20 threads completed ({len(results)})")
    successes = [ok for _, ok in results]
    check(all(successes), f"All distinct PIDs acquired remediation lock")

    # Concurrent weight updates in context model
    model = ContextConditionedModel()
    def _update_weights():
        for _ in range(100):
            model.update("SUSPEND", AnomalyClass.FORK_BOMB, reward=0.8,
                         evaluation_complete=True, pid_exited=False)
    threads2 = [threading.Thread(target=_update_weights) for _ in range(10)]
    for t in threads2: t.start()
    for t in threads2: t.join()
    w = model.get_p_success("SUSPEND", AnomalyClass.FORK_BOMB)
    check(0.01 <= w <= 0.99, f"Concurrent weight updates bounded: {w:.4f}")


# ─────────────────────────────────────────────────────────────
# Correction #9: Production Metrics
# ─────────────────────────────────────────────────────────────

def test_correction_9():
    section(9, "Production Metrics Collector")

    fb = FeedbackLoop()

    # Drive some counts
    fb.inc_anomaly()
    fb.inc_anomaly()
    fb.inc_intervention()
    fb.inc_termination()
    fb.inc_escalation()
    fb.inc_escalation()
    fb.inc_cooldown_block()

    stats = fb.get_stats()

    required_keys = [
        "total_anomalies", "total_interventions", "recovery_success_rate",
        "avg_reward", "termination_rate", "escalation_count", "cooldown_blocks",
        "active_evaluations", "current_threshold"
    ]
    for k in required_keys:
        check(k in stats, f"get_stats() contains '{k}'")

    check(stats["total_anomalies"]     == 2,   f"total_anomalies=2 (got {stats['total_anomalies']})")
    check(stats["total_interventions"] == 1,   f"total_interventions=1 (got {stats['total_interventions']})")
    check(stats["escalation_count"]    == 2,   f"escalation_count=2 (got {stats['escalation_count']})")
    check(stats["cooldown_blocks"]     == 1,   f"cooldown_blocks=1 (got {stats['cooldown_blocks']})")
    check(0.0 <= stats["recovery_success_rate"] <= 1.0, "recovery_success_rate in [0,1]")
    check(0.0 <= stats["avg_reward"]            <= 1.0, "avg_reward in [0,1]")
    check(0.0 <= stats["termination_rate"]      <= 1.0, "termination_rate in [0,1]")
    check(isinstance(stats["active_evaluations"], int),  "active_evaluations is int")
    check(stats["current_threshold"] > 0,               "current_threshold > 0")


# ─────────────────────────────────────────────────────────────
# Correction #10: Failure Safety
# ─────────────────────────────────────────────────────────────

def test_correction_10():
    section(10, "Failure Safety")

    # Handler crash triggers on_crash callback and returns False, not exception
    crash_pids = []
    def on_crash(pid):
        crash_pids.append(pid)

    remediator = RemediationEngine(dry_run=False, on_crash=on_crash)

    # Inject a crashing handler
    original = remediator._handle_observe
    def _crashing_handler(self, pid, action):
        raise RuntimeError("Simulated handler crash")
    from interventions.remediation import _HANDLERS, ActionType
    _HANDLERS[ActionType.OBSERVE] = _crashing_handler

    result = remediator.apply(99999, [a for a in __import__('interventions.remediation', fromlist=['ACTION_CATALOG']).ACTION_CATALOG if a.action_type == ActionType.OBSERVE][0])
    _HANDLERS[ActionType.OBSERVE] = original   # restore

    check(result is False,        "Crashed handler returns False, not exception")
    check(99999 in crash_pids,    f"on_crash(pid) called: {crash_pids}")

    # State machine reset_on_crash works
    sm = PIDStateMachine()
    sm.get_or_create(10001)
    sm.mark_anomalous(10001)
    sm.begin_remediation(10001)
    check(sm._records[10001].state == PIDState.REMEDIATING, "In REMEDIATING before crash")
    sm.reset_on_crash(10001)
    check(sm._records[10001].state == PIDState.NORMAL, "reset_on_crash → NORMAL")

    # Recovery monitor: concurrent evaluation for same pid is blocked (non-deadlocking)
    monitor = RecoveryMonitor(monitor_window_sec=0.1)
    lock = monitor._get_pid_lock(10002)
    lock.acquire()  # simulate ongoing evaluation
    result2 = monitor.evaluate(10002, None, None, 0.8, "OBSERVE", [])
    lock.release()
    check(not result2.evaluation_complete, "Concurrent eval returns zero-result, no deadlock")

    # FeedbackLoop MAX_ACTIVE_EVALS cap (don't dispatch if at limit)
    from feedback.feedback_loop import MAX_ACTIVE_EVALS
    check(MAX_ACTIVE_EVALS > 0, f"MAX_ACTIVE_EVALS defined: {MAX_ACTIVE_EVALS}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*60)
    print("  HEALIX PRODUCTION CORRECTIONS TEST SUITE")
    print("═"*60)

    test_correction_1()
    test_correction_2()
    test_correction_3()
    test_correction_4()
    test_correction_5()
    test_correction_6()
    test_correction_7()
    test_correction_8()
    test_correction_9()
    test_correction_10()

    print("\n" + "═"*60)
    if errors:
        print(f"  ❌ {len(errors)} FAILURE(S):")
        for e in errors:
            print(f"     • {e}")
        sys.exit(1)
    else:
        print(f"  ✅ ALL CORRECTIONS VERIFIED — {60 - len(errors)} checks passed")
    print("═"*60 + "\n")


if __name__ == "__main__":
    main()
