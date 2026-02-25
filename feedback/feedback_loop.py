"""
Healix - Layer 5: Feedback Learning Loop + Production Metrics Collector
=========================================================================
Corrections applied:
  #3  EMA: new_weight = old + ALPHA*(reward - old). Clamped [0.01, 0.99].
      Updates skipped if evaluation incomplete or pid exited.
  #8  pending_evaluations and weight updates protected by threading.Lock.
  #9  Production metrics: total_anomalies, total_interventions,
      recovery_success_rate, avg_reward, termination_rate,
      escalation_count, cooldown_blocks — exposed via get_stats().
  #10 No infinite remediation loops: max_active_evaluations limit.
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("healix.feedback")

RECOVERY_THRESHOLD   = 0.30
ALPHA                = 0.15
HISTORY_FILE         = Path(__file__).parent.parent / "logs" / "repair_history.jsonl"
MAX_ACTIVE_EVALS     = 64      # Correction #10: cap on concurrent monitor threads


@dataclass
class RepairRecord:
    timestamp:    float
    pid:          int
    comm:         str
    score_before: float
    score_after:  float
    action_name:  str
    anomaly_class: str
    reward:       float
    success:      bool
    duration_sec: float


class FeedbackLoop:
    def __init__(self) -> None:
        # Per-action EMA weights
        self._weights:      Dict[str, float] = {}
        self._weights_lock  = threading.Lock()

        # Threshold adaptation
        self._threshold_history: deque = deque(maxlen=200)
        self._threshold_history.append(0.65)

        # Repair history
        self._history: List[RepairRecord] = []

        # Correction #8: pending evaluations protected by lock
        self._pending_evals: Dict[int, threading.Thread] = {}
        self._pending_lock  = threading.Lock()

        # Correction #9: production metrics
        self._m_anomalies         = 0
        self._m_interventions     = 0
        self._m_rewards:    deque = deque(maxlen=500)
        self._m_terminates        = 0
        self._m_escalations       = 0
        self._m_cooldown_blocks   = 0
        self._m_lock              = threading.Lock()

        self._load_history()

    # ── Metrics Increment Helpers (Correction #9) ─────────────

    def inc_anomaly(self)          -> None:
        with self._m_lock: self._m_anomalies += 1

    def inc_intervention(self)     -> None:
        with self._m_lock: self._m_interventions += 1

    def inc_termination(self)      -> None:
        with self._m_lock: self._m_terminates += 1

    def inc_escalation(self)       -> None:
        with self._m_lock: self._m_escalations += 1

    def inc_cooldown_block(self)   -> None:
        with self._m_lock: self._m_cooldown_blocks += 1

    def get_stats(self) -> Dict:
        """
        Correction #9: structured dict of production metrics.
        No dashboard. Pure data output.
        """
        with self._m_lock:
            rewards = list(self._m_rewards)
            total   = self._m_interventions or 1
            successes = sum(1 for r in rewards if r >= RECOVERY_THRESHOLD)
            return {
                "total_anomalies":        self._m_anomalies,
                "total_interventions":    self._m_interventions,
                "recovery_success_rate":  round(successes / max(len(rewards), 1), 4),
                "avg_reward":             round(sum(rewards) / max(len(rewards), 1), 4),
                "termination_rate":       round(self._m_terminates / total, 4),
                "escalation_count":       self._m_escalations,
                "cooldown_blocks":        self._m_cooldown_blocks,
                "active_evaluations":     self._active_eval_count(),
                "current_threshold":      round(self._threshold_history[-1], 4),
            }

    def _active_eval_count(self) -> int:
        with self._pending_lock:
            return sum(1 for t in self._pending_evals.values() if t.is_alive())

    # ── Record Outcome ────────────────────────────────────────

    def record(
        self,
        pid:                 int,
        score_before:        float,
        action,
        anomaly_class,
        context_model,
        reward:              float,
        evaluation_complete: bool,
        pid_exited:          bool,
        score_after:         float = 0.0,
        comm:                str   = "",
    ) -> None:
        """
        Records a completed intervention outcome.
        Updates context model and EMA weights.
        All weight updates under lock. Updates skipped if evaluation incomplete.
        """
        action_name = action.action_type.name
        success     = reward >= RECOVERY_THRESHOLD

        # Correction #8: weight update under lock
        with self._weights_lock:
            if evaluation_complete and not pid_exited:
                old_w = self._weights.get(action_name, action.prior_success)
                # Correction #3: correct EMA form
                new_w = old_w + ALPHA * (reward - old_w)
                new_w = max(0.01, min(0.99, new_w))
                self._weights[action_name] = new_w
                log.debug(
                    f"[EMA] {action_name} {old_w:.4f} → {new_w:.4f} "
                    f"(reward={reward:.4f})"
                )

        # Update context-conditioned model (has its own internal lock)
        context_model.update(
            action_name          = action_name,
            anomaly_class        = anomaly_class,
            reward               = reward,
            evaluation_complete  = evaluation_complete,
            pid_exited           = pid_exited,
        )

        # Metrics
        with self._m_lock:
            self._m_rewards.append(reward)
            if action_name == "TERMINATE":
                self._m_terminates += 1

        rec = RepairRecord(
            timestamp    = time.time(),
            pid          = pid,
            comm         = comm,
            score_before = score_before,
            score_after  = score_after,
            action_name  = action_name,
            anomaly_class = anomaly_class.value if hasattr(anomaly_class, "value") else str(anomaly_class),
            reward       = reward,
            success      = success,
            duration_sec = 0.0,
        )
        self._history.append(rec)
        self._append_to_file(rec)

        log.info(
            f"[FEEDBACK] pid={pid} action={action_name} "
            f"reward={reward:.4f} success={success} "
            f"complete={evaluation_complete}"
        )

    # ── Threshold Adaptation ──────────────────────────────────

    def recommended_threshold(self) -> float:
        current = self._threshold_history[-1]
        with self._m_lock:
            cooldown_blocks = self._m_cooldown_blocks
            terminates      = self._m_terminates

        # Too sensitive: many cooldown blocks with no escalations
        if cooldown_blocks > 30 and terminates < 2:
            new_t = min(0.85, current + 0.02)
        elif terminates > 5:
            new_t = max(0.40, current - 0.03)
        else:
            new_t = current

        if new_t != current:
            log.info(f"[THRESHOLD] {current:.3f} → {new_t:.3f}")

        self._threshold_history.append(new_t)
        return new_t

    # ── Async Evaluation Dispatch ─────────────────────────────

    def dispatch_evaluation(
        self,
        monitor,          # RecoveryMonitor
        pid:              int,
        scorer,
        fingerprint,
        s_before:         float,
        action,
        anomaly_class,
        context_model,
        score_history:    list,
        comm:             str = "",
    ) -> None:
        """
        Launches background evaluation thread. Respects MAX_ACTIVE_EVALS cap.
        Correction #10: cap prevents infinite remediation loops.
        """
        with self._pending_lock:
            # Clean up finished threads
            self._pending_evals = {
                p: t for p, t in self._pending_evals.items() if t.is_alive()
            }
            if len(self._pending_evals) >= MAX_ACTIVE_EVALS:
                log.warning(
                    f"[FEEDBACK] Max active evaluations ({MAX_ACTIVE_EVALS}) reached, "
                    f"skipping async eval for pid={pid}"
                )
                return
            if pid in self._pending_evals:
                log.debug(f"[FEEDBACK] Evaluation already pending for pid={pid}, skipping")
                return

        def _run():
            result = monitor.evaluate(
                pid=pid, scorer=scorer, fingerprint=fingerprint,
                s_before=s_before, action_name=action.action_type.name,
                anomaly_history=score_history,
            )
            pid_exited = not result.process_survived and not result.evaluation_complete
            self.record(
                pid                  = pid,
                score_before         = s_before,
                action               = action,
                anomaly_class        = anomaly_class,
                context_model        = context_model,
                reward               = result.reward,
                evaluation_complete  = result.evaluation_complete,
                pid_exited           = pid_exited,
                score_after          = result.s_after,
                comm                 = comm,
            )

        t = threading.Thread(target=_run, daemon=True, name=f"eval-{pid}")
        with self._pending_lock:
            self._pending_evals[pid] = t
        t.start()

    # ── Persistence ───────────────────────────────────────────

    def _append_to_file(self, record: RepairRecord) -> None:
        try:
            HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(HISTORY_FILE, "a") as f:
                f.write(json.dumps(asdict(record)) + "\n")
        except Exception as e:
            log.warning(f"Failed to write repair history: {e}")

    def _load_history(self) -> None:
        if not HISTORY_FILE.exists():
            return
        try:
            with open(HISTORY_FILE) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        rec  = RepairRecord(**data)
                        self._history.append(rec)
                        with self._weights_lock:
                            old_w = self._weights.get(rec.action_name, 0.50)
                            new_w = old_w + ALPHA * (rec.reward - old_w)
                            self._weights[rec.action_name] = max(0.01, min(0.99, new_w))
                    except Exception:
                        continue
            log.info(f"Loaded {len(self._history)} historical records")
        except Exception as e:
            log.warning(f"Could not load repair history: {e}")

    def save(self) -> None:
        weights_file = HISTORY_FILE.parent / "learned_weights.json"
        try:
            with self._weights_lock:
                snapshot = dict(self._weights)
            with open(weights_file, "w") as f:
                json.dump(snapshot, f, indent=2)
            log.info(f"Weights saved ({len(snapshot)} entries)")
        except Exception as e:
            log.warning(f"Could not save weights: {e}")
