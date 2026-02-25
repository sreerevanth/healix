"""
Healix - Context-Conditioned Success Model
============================================
Corrections applied:
  #2  Deterministic anomaly classification from measurable metric thresholds only.
      No string matching. No heuristic flags. Strict numeric conditions.
  #2  Context key is (action_type, anomaly_class) — no process_profile dimension.
  #3  EMA: new_weight = old + ALPHA*(reward - old). Clamped [0.01, 0.99].
      Updates ignored if evaluation_complete=False (monitor window incomplete)
      or pid_exited=True.
  #8  Weight dict protected by threading.RLock.
"""

import logging
import threading
from enum import Enum
from typing import Dict, Optional, Tuple

log = logging.getLogger("healix.context_model")

ALPHA = 0.15   # EMA learning rate

# ── Anomaly classification thresholds ─────────────────────────
MEMORY_SLOPE_THRESHOLD  = 1.0      # MB/sec sustained growth → MEMORY_LEAK
FORK_RATE_THRESHOLD     = 10.0     # forks/sec → FORK_BOMB
KL_DRIFT_THRESHOLD      = 0.50     # KL divergence → SYSCALL_DRIFT
NET_RATE_THRESHOLD      = 50.0     # network syscalls/sec → NETWORK_SURGE
CPU_RATE_THRESHOLD      = 500.0    # syscalls/sec without fork/net → CPU_SPIKE


# ─────────────────────────────────────────────────────────────
# Anomaly Classification — metric-based, deterministic
# ─────────────────────────────────────────────────────────────

class AnomalyClass(str, Enum):
    MEMORY_LEAK   = "memory_leak"
    FORK_BOMB     = "fork_bomb"
    SYSCALL_DRIFT = "syscall_drift"
    NETWORK_SURGE = "network_surge"
    CPU_SPIKE     = "cpu_spike"
    UNKNOWN       = "unknown"


def classify_anomaly(features: Dict) -> AnomalyClass:
    """
    Deterministic classification from measured feature values only.
    No string matching. No reason parsing. Strict numeric thresholds.

    Priority order resolves ambiguity when multiple conditions true:
        MEMORY_LEAK → FORK_BOMB → NETWORK_SURGE → SYSCALL_DRIFT → CPU_SPIKE → UNKNOWN
    """
    memory_slope = features.get("memory_slope", 0.0)   # bytes/sec; convert to MB
    fork_rate    = features.get("fork_rate",    0.0)
    net_rate     = features.get("net_rate",     0.0)
    syscall_rate = features.get("syscall_rate", 0.0)
    kl_score     = features.get("kl_score",     0.0)   # populated by scorer

    memory_slope_mb = memory_slope / (1024 * 1024)  # bytes/sec → MB/sec

    if memory_slope_mb > MEMORY_SLOPE_THRESHOLD:
        return AnomalyClass.MEMORY_LEAK
    if fork_rate > FORK_RATE_THRESHOLD:
        return AnomalyClass.FORK_BOMB
    if net_rate > NET_RATE_THRESHOLD:
        return AnomalyClass.NETWORK_SURGE
    if kl_score > KL_DRIFT_THRESHOLD:
        return AnomalyClass.SYSCALL_DRIFT
    if syscall_rate > CPU_RATE_THRESHOLD:
        return AnomalyClass.CPU_SPIKE
    return AnomalyClass.UNKNOWN


# ─────────────────────────────────────────────────────────────
# Heuristic Prior Table
# P(success | action_name, anomaly_class)
# ─────────────────────────────────────────────────────────────

HEURISTIC_PRIORS: Dict[Tuple[str, AnomalyClass], float] = {
    ("OBSERVE",       AnomalyClass.MEMORY_LEAK):   1.0,
    ("OBSERVE",       AnomalyClass.FORK_BOMB):     1.0,
    ("OBSERVE",       AnomalyClass.SYSCALL_DRIFT): 1.0,
    ("OBSERVE",       AnomalyClass.NETWORK_SURGE): 1.0,
    ("OBSERVE",       AnomalyClass.CPU_SPIKE):     1.0,
    ("OBSERVE",       AnomalyClass.UNKNOWN):       1.0,

    ("THROTTLE_CPU",  AnomalyClass.FORK_BOMB):     0.85,
    ("THROTTLE_CPU",  AnomalyClass.CPU_SPIKE):     0.80,
    ("THROTTLE_CPU",  AnomalyClass.MEMORY_LEAK):   0.35,
    ("THROTTLE_CPU",  AnomalyClass.SYSCALL_DRIFT): 0.40,
    ("THROTTLE_CPU",  AnomalyClass.NETWORK_SURGE): 0.30,
    ("THROTTLE_CPU",  AnomalyClass.UNKNOWN):       0.50,

    ("THROTTLE_MEM",  AnomalyClass.MEMORY_LEAK):   0.88,
    ("THROTTLE_MEM",  AnomalyClass.FORK_BOMB):     0.40,
    ("THROTTLE_MEM",  AnomalyClass.CPU_SPIKE):     0.20,
    ("THROTTLE_MEM",  AnomalyClass.SYSCALL_DRIFT): 0.30,
    ("THROTTLE_MEM",  AnomalyClass.NETWORK_SURGE): 0.25,
    ("THROTTLE_MEM",  AnomalyClass.UNKNOWN):       0.45,

    ("SYSCALL_BLOCK", AnomalyClass.SYSCALL_DRIFT): 0.82,
    ("SYSCALL_BLOCK", AnomalyClass.NETWORK_SURGE): 0.78,
    ("SYSCALL_BLOCK", AnomalyClass.FORK_BOMB):     0.60,
    ("SYSCALL_BLOCK", AnomalyClass.MEMORY_LEAK):   0.40,
    ("SYSCALL_BLOCK", AnomalyClass.CPU_SPIKE):     0.35,
    ("SYSCALL_BLOCK", AnomalyClass.UNKNOWN):       0.50,

    ("CHECKPOINT",    AnomalyClass.MEMORY_LEAK):   0.60,
    ("CHECKPOINT",    AnomalyClass.FORK_BOMB):     0.55,
    ("CHECKPOINT",    AnomalyClass.SYSCALL_DRIFT): 0.50,
    ("CHECKPOINT",    AnomalyClass.NETWORK_SURGE): 0.50,
    ("CHECKPOINT",    AnomalyClass.CPU_SPIKE):     0.50,
    ("CHECKPOINT",    AnomalyClass.UNKNOWN):       0.50,

    ("SUSPEND",       AnomalyClass.FORK_BOMB):     0.88,
    ("SUSPEND",       AnomalyClass.MEMORY_LEAK):   0.70,
    ("SUSPEND",       AnomalyClass.CPU_SPIKE):     0.85,
    ("SUSPEND",       AnomalyClass.SYSCALL_DRIFT): 0.65,
    ("SUSPEND",       AnomalyClass.NETWORK_SURGE): 0.72,
    ("SUSPEND",       AnomalyClass.UNKNOWN):       0.75,

    ("ISOLATE_NET",   AnomalyClass.NETWORK_SURGE): 0.90,
    ("ISOLATE_NET",   AnomalyClass.SYSCALL_DRIFT): 0.55,
    ("ISOLATE_NET",   AnomalyClass.FORK_BOMB):     0.30,
    ("ISOLATE_NET",   AnomalyClass.MEMORY_LEAK):   0.20,
    ("ISOLATE_NET",   AnomalyClass.CPU_SPIKE):     0.15,
    ("ISOLATE_NET",   AnomalyClass.UNKNOWN):       0.40,

    ("TERMINATE",     AnomalyClass.MEMORY_LEAK):   0.95,
    ("TERMINATE",     AnomalyClass.FORK_BOMB):     0.95,
    ("TERMINATE",     AnomalyClass.SYSCALL_DRIFT): 0.95,
    ("TERMINATE",     AnomalyClass.NETWORK_SURGE): 0.95,
    ("TERMINATE",     AnomalyClass.CPU_SPIKE):     0.95,
    ("TERMINATE",     AnomalyClass.UNKNOWN):       0.95,
}


# ─────────────────────────────────────────────────────────────
# Context-Conditioned Model
# ─────────────────────────────────────────────────────────────

class ContextConditionedModel:
    """
    Maintains P(success | action_name, anomaly_class).
    Key: (action_name, anomaly_class) — two-dimensional, deterministic.

    EMA correction (#3):
        new_weight = old_weight + ALPHA * (reward - old_weight)
        Equivalent to: (1-ALPHA)*old + ALPHA*reward
        Clamped to [0.01, 0.99].

    Updates are ignored if:
        - evaluation_complete is False (monitor window did not finish)
        - pid_exited is True before evaluation window completed
    """

    def __init__(self) -> None:
        # (action_name, AnomalyClass) → float in [0.01, 0.99]
        self._weights: Dict[Tuple[str, AnomalyClass], float] = {}
        self._lock = threading.RLock()

    def get_p_success(self, action_name: str, anomaly_class: AnomalyClass) -> float:
        """
        Returns P(success | action, anomaly_class).
        Falls back to heuristic prior, then 0.50.
        """
        with self._lock:
            key = (action_name, anomaly_class)
            if key in self._weights:
                return self._weights[key]
        return HEURISTIC_PRIORS.get((action_name, anomaly_class), 0.50)

    def update(
        self,
        action_name:         str,
        anomaly_class:       AnomalyClass,
        reward:              float,
        evaluation_complete: bool,
        pid_exited:          bool,
    ) -> None:
        """
        EMA update. Silently ignored if evaluation is incomplete or pid exited early.
        """
        if not evaluation_complete:
            log.debug(f"[CONTEXT_MODEL] Skipping update: evaluation incomplete")
            return
        if pid_exited:
            log.debug(f"[CONTEXT_MODEL] Skipping update: pid exited before window")
            return

        key = (action_name, anomaly_class)
        with self._lock:
            old_w = self._weights.get(key, HEURISTIC_PRIORS.get(key, 0.50))
            # Correction #3: standard EMA formula
            new_w = old_w + ALPHA * (reward - old_w)
            # Clamp to prevent oscillation extremes
            new_w = max(0.01, min(0.99, new_w))
            self._weights[key] = new_w

        log.debug(
            f"[CONTEXT_MODEL] ({action_name}, {anomaly_class.value}) "
            f"{old_w:.4f} → {new_w:.4f} (reward={reward:.4f})"
        )

    def serialize(self) -> Dict:
        """Export weights for persistence."""
        with self._lock:
            return {f"{a}|{c.value}": w for (a, c), w in self._weights.items()}

    def __repr__(self) -> str:
        return f"<ContextConditionedModel entries={len(self._weights)}>"
