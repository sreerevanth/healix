"""
Healix - Layer 3: Real-Time Anomaly Scoring Engine
====================================================
Computes a composite anomaly score for each process event.

Rule-based in Phase 1. ML hooks (autoencoder/LSTM) stubbed for Phase 2.

Score = weighted sum of:
    1. Syscall rate deviation       (vs. baseline)
    2. Group distribution drift     (KL divergence)
    3. Memory slope spike           (absolute threshold + relative)
    4. Fork/exec burst              (rate vs. baseline)
    5. Network call surge           (rate vs. baseline)
    6. Thread explosion             (delta threshold)

Each sub-score is normalized 0→1.
Final composite score is 0→1. Score > threshold → trigger remediation.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("healix.scorer")

# ─────────────────────────────────────────────────────────────
# Thresholds (tunable - will be learned via feedback in Phase 4)
# ─────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 0.65   # initial trigger threshold

# How many std-devs above baseline rate before we flag
RATE_DEVIATION_FACTOR    = 5.0    # 5x baseline rate
MEMORY_SLOPE_THRESHOLD   = 10 * 1024 * 1024   # 10 MB/sec growth
FORK_RATE_THRESHOLD      = 20.0   # forks/sec
NET_SURGE_FACTOR         = 8.0    # 8x baseline net rate
THREAD_DELTA_THRESHOLD   = 10     # absolute new threads

# Per-feature weights (must sum to 1.0)
WEIGHTS = {
    "rate_deviation":   0.30,
    "dist_drift":       0.20,
    "memory_slope":     0.20,
    "fork_burst":       0.15,
    "net_surge":        0.10,
    "thread_explosion": 0.05,
}


# ─────────────────────────────────────────────────────────────
# KL Divergence helper
# ─────────────────────────────────────────────────────────────

def kl_divergence(p: Dict, q: Dict) -> float:
    """KL(P || Q) for two normalized syscall group distributions."""
    all_keys = set(p) | set(q)
    result = 0.0
    for k in all_keys:
        pk = p.get(k, 1e-10)
        qk = q.get(k, 1e-10)
        result += pk * math.log(pk / qk)
    return max(0.0, result)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


# ─────────────────────────────────────────────────────────────
# Anomaly Scorer
# ─────────────────────────────────────────────────────────────

class AnomalyScorer:
    """
    Stateless scorer - operates on a BehavioralFingerprint snapshot.
    The threshold adapts over time via feedback loop recommendations.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold

    def score(
        self,
        fingerprint,   # BehavioralFingerprint
        event,         # SyscallEvent
    ) -> Tuple[float, List[str]]:
        """
        Returns (composite_score: float, reasons: List[str]).
        composite_score is 0.0 → 1.0.
        reasons is a human-readable list of triggered sub-rules.
        """
        features = fingerprint.feature_vector()
        reasons  = []
        sub_scores = {}

        # ── 1. Syscall Rate Deviation ─────────────────────────
        current_rate  = features["syscall_rate"]
        baseline_rate = fingerprint.baseline_rate or 1.0
        if baseline_rate < 1.0:
            baseline_rate = 1.0

        rate_ratio = current_rate / baseline_rate
        rate_score = min(1.0, (rate_ratio - 1.0) / RATE_DEVIATION_FACTOR)
        rate_score = max(0.0, rate_score)
        sub_scores["rate_deviation"] = rate_score

        if rate_ratio > RATE_DEVIATION_FACTOR:
            reasons.append(
                f"syscall_rate_spike({current_rate:.0f}/s vs baseline {baseline_rate:.0f}/s)"
            )

        # ── 2. Group Distribution Drift (KL divergence) ───────
        kl = 0.0
        if fingerprint.baseline_frozen and fingerprint.baseline_syscall_dist:
            current_dist  = features["group_dist"]
            baseline_dist = fingerprint.baseline_syscall_dist
            kl = kl_divergence(current_dist, baseline_dist)
            dist_score = min(1.0, kl / 2.0)
            sub_scores["dist_drift"] = dist_score
            if kl > 0.5:
                reasons.append(f"syscall_pattern_drift(KL={kl:.3f})")
        else:
            sub_scores["dist_drift"] = 0.0
        features["kl_score"] = kl  # expose for metric-based classify_anomaly()

        # ── 3. Memory Slope Spike ─────────────────────────────
        slope = features["memory_slope"]
        if slope > 0:
            mem_score = min(1.0, slope / MEMORY_SLOPE_THRESHOLD)
        else:
            mem_score = 0.0
        sub_scores["memory_slope"] = mem_score

        if slope > MEMORY_SLOPE_THRESHOLD:
            reasons.append(f"memory_leak({slope/1024/1024:.1f} MB/s growth)")

        # ── 4. Fork / Exec Burst ──────────────────────────────
        fork_rate  = features["fork_rate"]
        fork_score = min(1.0, fork_rate / FORK_RATE_THRESHOLD)
        sub_scores["fork_burst"] = fork_score

        if fork_rate > FORK_RATE_THRESHOLD:
            reasons.append(f"fork_bomb_pattern({fork_rate:.1f} forks/s)")

        # ── 5. Network Surge ──────────────────────────────────
        net_rate = features["net_rate"]
        baseline_net = (fingerprint.baseline_rate or 10.0) * 0.05  # rough proxy
        net_ratio    = net_rate / max(1.0, baseline_net)
        net_score    = min(1.0, (net_ratio - 1.0) / NET_SURGE_FACTOR)
        net_score    = max(0.0, net_score)
        sub_scores["net_surge"] = net_score

        if net_ratio > NET_SURGE_FACTOR:
            reasons.append(f"network_surge({net_rate:.1f} net_calls/s)")

        # ── 6. Thread Explosion ───────────────────────────────
        t_delta = features["thread_delta"]
        t_score = min(1.0, abs(t_delta) / THREAD_DELTA_THRESHOLD)
        sub_scores["thread_explosion"] = t_score

        if abs(t_delta) > THREAD_DELTA_THRESHOLD:
            reasons.append(f"thread_explosion(delta={t_delta:+d})")

        # ── Composite Score ───────────────────────────────────
        composite = sum(WEIGHTS[k] * sub_scores.get(k, 0.0) for k in WEIGHTS)

        # Boost score if baseline not yet frozen (we're less certain)
        if not fingerprint.baseline_frozen:
            composite *= 0.5  # dampen alerts during warmup

        log.debug(
            f"pid={fingerprint.pid} composite={composite:.3f} "
            f"sub={sub_scores} reasons={reasons}"
        )

        return composite, reasons

    # ── ML stub (Phase 2) ────────────────────────────────────

    def score_ml(self, fingerprint, event) -> Optional[float]:
        """
        Phase 2: Replace rule-based scoring with trained model.

        Options:
          - Autoencoder: train on normal fingerprints, flag high reconstruction error
          - LSTM:        sequence model over syscall number stream
          - IsolationForest: unsupervised anomaly detection on feature vector

        Stub returns None → falls back to rule-based score.
        """
        return None
