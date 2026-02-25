"""
Healix - PID State Machine
============================
Corrections applied:
  #5  Strict state transitions: NORMAL→ANOMALOUS→REMEDIATING→COOLDOWN→NORMAL
      Only one active remediation per PID. Atomic under lock.
  #6  Escalation tracking per (pid, anomaly_class), not global.
      Failure counter resets on successful recovery.
  #7  Cooldown: blocks same-level spam. Allows escalation only when
      new_score > previous_score * 1.15.
  #8  All state transitions atomic under per-pid RLock.
      Global registry protected by separate lock.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("healix.state_machine")

COOLDOWN_SEC          = 30.0   # minimum inter-intervention gap
ESCALATION_FAIL_MAX   = 2      # consecutive failures before escalation unlocked
ESCALATION_SCORE_MULT = 1.15   # new_score must exceed prev_score * this to escalate in cooldown


# ─────────────────────────────────────────────────────────────
# States
# ─────────────────────────────────────────────────────────────

class PIDState(Enum):
    NORMAL      = auto()
    ANOMALOUS   = auto()
    REMEDIATING = auto()
    COOLDOWN    = auto()


# ─────────────────────────────────────────────────────────────
# PID Record
# ─────────────────────────────────────────────────────────────

@dataclass
class PIDRecord:
    pid:   int
    comm:  str

    state:       PIDState = PIDState.NORMAL
    state_since: float    = field(default_factory=time.time)

    last_score:     float                       = 0.0
    score_at_cooldown_start: float              = 0.0   # score when cooldown began
    score_history:  List[Tuple[float, float]]   = field(default_factory=list)

    last_action_name: Optional[str] = None
    last_action_ts:   float         = 0.0
    action_history:   List[str]     = field(default_factory=list)

    # Correction #6: fail counts keyed by (action_name, anomaly_class_value)
    fail_counts: Dict[Tuple[str, str], int] = field(default_factory=dict)

    cooldown_until: float = 0.0

    # Per-PID lock — RLock allows re-entrant acquisition within same thread
    _lock: threading.RLock = field(default_factory=threading.RLock)

    # ── Internal helpers (call only while holding _lock) ──────

    def _transition(self, new_state: PIDState) -> None:
        """Atomic state change. Must be called under _lock."""
        log.info(
            f"[STATE] pid={self.pid} comm={self.comm!r} "
            f"{self.state.name} → {new_state.name}"
        )
        self.state       = new_state
        self.state_since = time.time()

    def _record_score(self, score: float) -> None:
        ts = time.time()
        self.last_score = score
        self.score_history.append((ts, score))
        cutoff = ts - 300.0
        self.score_history = [(t, s) for t, s in self.score_history if t >= cutoff]

    def _in_cooldown(self) -> bool:
        return time.time() < self.cooldown_until

    def _fail_key(self, action_name: str, anomaly_class_val: str) -> Tuple[str, str]:
        return (action_name, anomaly_class_val)

    def _incr_fail(self, action_name: str, anomaly_class_val: str) -> None:
        k = self._fail_key(action_name, anomaly_class_val)
        self.fail_counts[k] = self.fail_counts.get(k, 0) + 1

    def _reset_fail(self, action_name: str, anomaly_class_val: str) -> None:
        k = self._fail_key(action_name, anomaly_class_val)
        self.fail_counts[k] = 0

    def _consec_fails(self, action_name: str, anomaly_class_val: str) -> int:
        return self.fail_counts.get(self._fail_key(action_name, anomaly_class_val), 0)

    def _should_escalate(self, action_name: str, anomaly_class_val: str) -> bool:
        return self._consec_fails(action_name, anomaly_class_val) >= ESCALATION_FAIL_MAX


# ─────────────────────────────────────────────────────────────
# PID State Machine Manager
# ─────────────────────────────────────────────────────────────

class PIDStateMachine:
    """
    Thread-safe registry of per-PID state records.
    Global registry lock: guards _records dict mutations.
    Per-pid RLock: guards state transitions and score updates.
    """

    def __init__(self) -> None:
        self._records:      Dict[int, PIDRecord] = {}
        self._global_lock = threading.Lock()

    # ── Registry ──────────────────────────────────────────────

    def get_or_create(self, pid: int, comm: str = "") -> PIDRecord:
        with self._global_lock:
            if pid not in self._records:
                self._records[pid] = PIDRecord(pid=pid, comm=comm)
            return self._records[pid]

    def remove(self, pid: int) -> None:
        with self._global_lock:
            self._records.pop(pid, None)

    # ── Intervention Gate ─────────────────────────────────────

    def can_intervene(self, pid: int, new_score: float = 0.0) -> Tuple[bool, str]:
        """
        Returns (allowed, reason).

        Blocked if:
          - REMEDIATING: only one active remediation per pid
          - COOLDOWN: unless new_score > score_at_cooldown_start * 1.15 (escalation)

        Correction #7: cooldown allows escalation only on severity increase.
        """
        record = self._records.get(pid)
        if record is None:
            return True, "no_record"

        with record._lock:
            if record.state == PIDState.REMEDIATING:
                return False, "already_remediating"

            if record.state == PIDState.COOLDOWN:
                if record._in_cooldown():
                    # Allow escalation if severity meaningfully increased
                    if (record.score_at_cooldown_start > 0 and
                            new_score > record.score_at_cooldown_start * ESCALATION_SCORE_MULT):
                        log.warning(
                            f"[STATE] pid={pid} severity escalation in cooldown: "
                            f"{record.score_at_cooldown_start:.3f} → {new_score:.3f}"
                        )
                        record._transition(PIDState.ANOMALOUS)
                        return True, "escalation_override"
                    remaining = record.cooldown_until - time.time()
                    return False, f"cooldown({remaining:.1f}s)"
                else:
                    # Cooldown expired
                    record._transition(PIDState.NORMAL)

        return True, "ok"

    # ── Transition Helpers ────────────────────────────────────

    def mark_anomalous(self, pid: int) -> None:
        record = self._records.get(pid)
        if record:
            with record._lock:
                if record.state == PIDState.NORMAL:
                    record._transition(PIDState.ANOMALOUS)

    def begin_remediation(self, pid: int) -> bool:
        """
        Atomic: transitions ANOMALOUS → REMEDIATING.
        Returns False if transition not possible (race condition guard).
        """
        record = self._records.get(pid)
        if not record:
            return False
        with record._lock:
            if record.state not in (PIDState.ANOMALOUS, PIDState.NORMAL):
                return False
            record._transition(PIDState.REMEDIATING)
            return True

    def end_remediation(
        self,
        pid:              int,
        action_name:      str,
        anomaly_class_val: str,
        success:          bool,
    ) -> None:
        """
        REMEDIATING → COOLDOWN.
        Updates per-(pid, anomaly_class) fail counter.
        Shorter cooldown on failure so we re-evaluate sooner.
        """
        record = self._records.get(pid)
        if not record:
            return
        with record._lock:
            record.last_action_name = action_name
            record.last_action_ts   = time.time()
            record.action_history.append(action_name)

            if success:
                record._reset_fail(action_name, anomaly_class_val)
            else:
                record._incr_fail(action_name, anomaly_class_val)

            duration = COOLDOWN_SEC if success else COOLDOWN_SEC * 0.5
            record.cooldown_until            = time.time() + duration
            record.score_at_cooldown_start   = record.last_score
            record._transition(PIDState.COOLDOWN)

    def reset_on_crash(self, pid: int) -> None:
        """
        Correction #10: called if a handler crashes mid-remediation.
        Resets state to NORMAL so the pid is not permanently stuck.
        """
        record = self._records.get(pid)
        if record:
            with record._lock:
                log.warning(f"[STATE] pid={pid} crash-reset from {record.state.name} → NORMAL")
                record._transition(PIDState.NORMAL)

    # ── Score Update ──────────────────────────────────────────

    def update_score(self, pid: int, score: float) -> None:
        """Record latest score. Does NOT trigger state transitions."""
        record = self._records.get(pid)
        if record:
            with record._lock:
                record._record_score(score)

    # ── Escalation Query ──────────────────────────────────────

    def should_escalate(
        self,
        pid:              int,
        action_name:      str,
        anomaly_class_val: str,
    ) -> bool:
        """
        True if this (pid, action, anomaly_class) combination has
        failed ESCALATION_FAIL_MAX or more consecutive times.
        Correction #6: keyed per (pid, anomaly_class), not globally.
        """
        record = self._records.get(pid)
        if not record:
            return False
        with record._lock:
            return record._should_escalate(action_name, anomaly_class_val)

    # ── Summary ───────────────────────────────────────────────

    def summary(self) -> Dict:
        with self._global_lock:
            return {
                pid: {
                    "comm":        r.comm,
                    "state":       r.state.name,
                    "last_score":  round(r.last_score, 3),
                    "last_action": r.last_action_name,
                    "actions":     len(r.action_history),
                }
                for pid, r in self._records.items()
            }

    def active_count(self) -> int:
        with self._global_lock:
            return len(self._records)
