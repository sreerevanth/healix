"""
Healix - Formal Recovery Metric R(t)
======================================
Corrections applied:
  #1  Hardened reward: max(0,ΔS), div-by-zero guard, strict [0,1] clamps
  #8  Concurrency: per-pid lock prevents concurrent evaluations
  #10 Failure safety: crashes return zero-reward result, never propagate
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import psutil

log = logging.getLogger("healix.recovery")

W1 = 0.50   # score-improvement weight
W2 = 0.35   # resource-stability weight
W3 = 0.15   # recurrence-penalty weight

MONITOR_WINDOW_SEC    = 30.0
RECURRENCE_WINDOW_SEC = 120.0
RECURRENCE_THRESHOLD  = 0.55


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Strict clamp. Guards against NaN."""
    if x != x:
        return lo
    return max(lo, min(hi, x))


# ─────────────────────────────────────────────────────────────
# Resource Snapshot
# ─────────────────────────────────────────────────────────────

@dataclass
class ResourceSnapshot:
    timestamp:   float
    cpu_pct:     float
    rss_mb:      float
    num_fds:     int
    num_threads: int


def capture_snapshot(pid: int) -> Optional[ResourceSnapshot]:
    try:
        p  = psutil.Process(pid)
        mi = p.memory_info()
        return ResourceSnapshot(
            timestamp   = time.time(),
            cpu_pct     = p.cpu_percent(interval=0.1),
            rss_mb      = mi.rss / 1024 / 1024,
            num_fds     = p.num_fds(),
            num_threads = p.num_threads(),
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
        return None


# ─────────────────────────────────────────────────────────────
# Resource Stability Score — bounded [0, 1]
# ─────────────────────────────────────────────────────────────

def resource_stability_score(
    before: ResourceSnapshot,
    after:  Optional[ResourceSnapshot],
) -> float:
    """
    Returns value in [0, 1]. Each dimension independently clamped.
    Returns 0.0 if process died — reward formula handles the interpretation.
    """
    if after is None:
        return 0.0

    def _dim(b: float, a: float, scale: float) -> float:
        return clamp((b - a) / max(scale, 1e-6))

    cpu_stab = _dim(before.cpu_pct,     after.cpu_pct,     100.0)
    mem_stab = _dim(before.rss_mb,      after.rss_mb,      max(before.rss_mb * 0.5, 1.0))
    fd_stab  = _dim(before.num_fds,     after.num_fds,     100.0)
    thr_stab = _dim(before.num_threads, after.num_threads, 20.0)

    return clamp(0.35 * cpu_stab + 0.35 * mem_stab + 0.15 * fd_stab + 0.15 * thr_stab)


# ─────────────────────────────────────────────────────────────
# Recovery Result
# ─────────────────────────────────────────────────────────────

@dataclass
class RecoveryResult:
    pid:                  int
    action_name:          str
    s_before:             float
    s_after:              float
    delta_s:              float        # max(0, s_before - s_after)
    normalized_delta:     float        # delta_s / max(s_before, 1e-6)
    resource_stability:   float        # [0, 1]
    recurrence_penalty:   float        # [0, 1]
    reward:               float        # [0, 1], never negative
    process_survived:     bool
    monitor_duration_sec: float
    evaluation_complete:  bool


# ─────────────────────────────────────────────────────────────
# Recovery Monitor
# ─────────────────────────────────────────────────────────────

class RecoveryMonitor:
    """
    Observes a process after intervention and computes R(t).

    Thread safety: per-pid locks prevent concurrent evaluations for the same pid.
    Failure safety: any exception returns a zero-reward result, never raises.
    """

    def __init__(self, monitor_window_sec: float = MONITOR_WINDOW_SEC):
        self.window        = monitor_window_sec
        self._pid_locks:   dict             = {}
        self._registry_lock = threading.Lock()

    def _get_pid_lock(self, pid: int) -> threading.Lock:
        with self._registry_lock:
            if pid not in self._pid_locks:
                self._pid_locks[pid] = threading.Lock()
            return self._pid_locks[pid]

    def evaluate(
        self,
        pid:             int,
        scorer,
        fingerprint,
        s_before:        float,
        action_name:     str,
        anomaly_history: List[Tuple[float, float]],
    ) -> RecoveryResult:
        """
        Safe entry point. Returns zero-result if already evaluating this pid
        or if any internal exception occurs.
        """
        pid_lock = self._get_pid_lock(pid)
        if not pid_lock.acquire(blocking=False):
            log.debug(f"[RECOVERY] pid={pid} evaluation already running, skipping")
            return _zero_result(pid, action_name, s_before)

        try:
            return self._run(pid, scorer, fingerprint, s_before, action_name, anomaly_history)
        except Exception as exc:
            log.error(f"[RECOVERY] pid={pid} evaluation crashed: {exc}", exc_info=True)
            return _zero_result(pid, action_name, s_before)
        finally:
            pid_lock.release()

    def _run(
        self,
        pid:             int,
        scorer,
        fingerprint,
        s_before:        float,
        action_name:     str,
        anomaly_history: List[Tuple[float, float]],
    ) -> RecoveryResult:
        t_start     = time.time()
        snap_before = capture_snapshot(pid)

        time.sleep(self.window)

        t_end      = time.time()
        snap_after = capture_snapshot(pid)
        survived   = snap_after is not None

        if survived:
            try:
                s_after, _ = scorer.score(fingerprint, _FakeEvent(pid))
                s_after = clamp(s_after)
            except Exception:
                s_after = s_before
        else:
            s_after = 0.0

        # ── Hardened reward formula (Correction #1) ───────────

        delta_s          = max(0.0, s_before - s_after)
        normalized_delta = clamp(delta_s / max(s_before, 1e-6))
        stab             = clamp(resource_stability_score(snap_before, snap_after) if snap_before else 0.0)
        recurrence       = clamp(self._recurrence_penalty(anomaly_history, t_start))
        reward           = clamp(W1 * normalized_delta + W2 * stab - W3 * recurrence)

        result = RecoveryResult(
            pid                  = pid,
            action_name          = action_name,
            s_before             = s_before,
            s_after              = s_after,
            delta_s              = delta_s,
            normalized_delta     = normalized_delta,
            resource_stability   = stab,
            recurrence_penalty   = recurrence,
            reward               = reward,
            process_survived     = survived,
            monitor_duration_sec = t_end - t_start,
            evaluation_complete  = True,
        )
        log.info(
            f"[RECOVERY] pid={pid} ΔS={delta_s:.3f} norm={normalized_delta:.3f} "
            f"stab={stab:.3f} rec={recurrence:.3f} reward={reward:.4f}"
        )
        return result

    def _recurrence_penalty(
        self,
        history: List[Tuple[float, float]],
        since:   float,
    ) -> float:
        post = [s for ts, s in history if ts >= since]
        if not post:
            return 0.0
        return clamp(sum(1 for s in post if s > RECURRENCE_THRESHOLD) / len(post))


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _zero_result(pid: int, action_name: str, s_before: float) -> RecoveryResult:
    return RecoveryResult(
        pid=pid, action_name=action_name,
        s_before=s_before, s_after=s_before,
        delta_s=0.0, normalized_delta=0.0,
        resource_stability=0.0, recurrence_penalty=0.0,
        reward=0.0, process_survived=False,
        monitor_duration_sec=0.0, evaluation_complete=False,
    )


class _FakeEvent:
    __slots__ = ("pid", "syscall_nr", "timestamp_ns", "comm")

    def __init__(self, pid: int) -> None:
        self.pid          = pid
        self.syscall_nr   = 0
        self.timestamp_ns = time.time_ns()
        self.comm         = ""
