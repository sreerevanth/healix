"""
Healix - Layer 4: Adaptive Remediation Engine
===============================================
Corrections applied:
  #4  Severity-bound utility formula:
          utility = P_success * (1 - invasiveness * severity) * (1 + cab)
      Requirements enforced:
        - severity clamped to [0, 1] before use
        - invasiveness in [0, 1] (catalog constants)
        - utility floored at 0.0 (never negative)
        - invasiveness ceiling applied FIRST as hard filter
        - Does NOT auto-prefer most invasive at high severity
  #8  success_weights dict protected by threading.Lock
  #10 Failure safety: handler exceptions caught, state reset triggered via callback
"""

import logging
import os
import signal
import subprocess
import threading
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

log = logging.getLogger("healix.remediator")


# ─────────────────────────────────────────────────────────────
# Action Definitions
# ─────────────────────────────────────────────────────────────

class ActionType(IntEnum):
    OBSERVE       = 0
    THROTTLE_CPU  = 1
    THROTTLE_MEM  = 2
    SYSCALL_BLOCK = 3
    CHECKPOINT    = 4
    SUSPEND       = 5
    ISOLATE_NET   = 6
    TERMINATE     = 7


@dataclass
class InterventionAction:
    action_type:   ActionType
    invasiveness:  float           # constant in [0.0, 1.0]
    prior_success: float           # initial P(success) heuristic
    description:   str
    params:        Dict = field(default_factory=dict)


ACTION_CATALOG: List[InterventionAction] = [
    InterventionAction(ActionType.OBSERVE,       0.00, 1.00, "Alert only"),
    InterventionAction(ActionType.THROTTLE_CPU,  0.15, 0.70, "cgroup CPU quota", {"cpu_quota_pct": 50}),
    InterventionAction(ActionType.THROTTLE_MEM,  0.20, 0.65, "cgroup memory limit", {"headroom_factor": 1.2}),
    InterventionAction(ActionType.SYSCALL_BLOCK, 0.35, 0.75, "seccomp syscall block"),
    InterventionAction(ActionType.CHECKPOINT,    0.40, 0.55, "CRIU checkpoint"),
    InterventionAction(ActionType.SUSPEND,       0.50, 0.80, "SIGSTOP + auto-resume", {"auto_resume_sec": 30}),
    InterventionAction(ActionType.ISOLATE_NET,   0.65, 0.60, "network namespace isolation"),
    InterventionAction(ActionType.TERMINATE,     1.00, 0.95, "SIGTERM → SIGKILL"),
]


def score_to_max_invasiveness(score: float) -> float:
    """Maps anomaly score [0,1] to maximum permitted invasiveness [0,1]."""
    if score < 0.40: return 0.00
    if score < 0.55: return 0.20
    if score < 0.70: return 0.50
    if score < 0.85: return 0.70
    return 1.00


# ─────────────────────────────────────────────────────────────
# Remediation Engine
# ─────────────────────────────────────────────────────────────

class RemediationEngine:
    def __init__(
        self,
        dry_run:       bool = False,
        on_crash:      Optional[Callable[[int], None]] = None,   # crash callback(pid)
    ) -> None:
        self.dry_run  = dry_run
        self.on_crash = on_crash   # called with pid if handler crashes (Correction #10)

        self._weights_lock = threading.Lock()
        # action_type → float (separate from context model weights)
        self._success_weights: Dict[ActionType, float] = {
            a.action_type: a.prior_success for a in ACTION_CATALOG
        }
        self._cgroup_root = Path("/sys/fs/cgroup/healix")

    # ── Action Selection ──────────────────────────────────────

    def select_action(
        self,
        fingerprint,
        score:          float,
        anomaly_class,              # AnomalyClass
        context_model,              # ContextConditionedModel
        escalation_hint: bool = False,
    ) -> InterventionAction:
        """
        Correction #4: Severity-Bound Utility Selector.

        Algorithm:
          1. Clamp severity to [0, 1].
          2. Compute invasiveness ceiling from score.
          3. Raise ceiling by one step if escalation_hint.
          4. FILTER catalog to actions within ceiling (hard gate).
          5. For each candidate, compute:
               utility = max(0, P_success * (1 - inv*severity) * (1 + cab))
          6. Select argmax(utility). OBSERVE is always in candidate set.
        """
        severity = max(0.0, min(1.0, score))   # clamp
        max_inv  = score_to_max_invasiveness(severity)

        if escalation_hint:
            # Raise ceiling by exactly one ladder step, capped at 1.0
            steps = [0.00, 0.20, 0.50, 0.70, 1.00]
            idx   = next((i for i, s in enumerate(steps) if s >= max_inv), len(steps) - 1)
            max_inv = steps[min(idx + 1, len(steps) - 1)]
            log.info(f"[ESCALATION] Invasiveness ceiling raised to {max_inv:.2f}")

        best_action  = ACTION_CATALOG[0]   # OBSERVE — always safe fallback
        best_utility = 0.0

        for action in ACTION_CATALOG:
            # Hard ceiling filter (Correction #4)
            if action.invasiveness > max_inv:
                continue

            p_success = context_model.get_p_success(
                action.action_type.name, anomaly_class
            )

            cab = self._context_alignment_bonus(action, anomaly_class)

            # Utility formula — floor at 0.0
            raw_utility = (
                p_success
                * (1.0 - action.invasiveness * severity)
                * (1.0 + cab)
            )
            utility = max(0.0, raw_utility)

            if utility > best_utility:
                best_utility = utility
                best_action  = action

        log.info(
            f"[SELECT] score={score:.3f} sev={severity:.3f} "
            f"max_inv={max_inv:.2f} action={best_action.action_type.name} "
            f"utility={best_utility:.4f} anomaly={anomaly_class}"
        )
        return best_action

    def _context_alignment_bonus(self, action: InterventionAction, anomaly_class) -> float:
        """
        Domain-matched bonus in [0, 0.35]. Raises utility of best-fit actions.
        Does not penalise others — only additive.
        """
        from models.context_model import AnomalyClass
        ac = anomaly_class
        at = action.action_type
        if ac == AnomalyClass.FORK_BOMB     and at == ActionType.THROTTLE_CPU:  return 0.30
        if ac == AnomalyClass.MEMORY_LEAK   and at == ActionType.THROTTLE_MEM:  return 0.35
        if ac == AnomalyClass.NETWORK_SURGE and at == ActionType.ISOLATE_NET:   return 0.25
        if ac == AnomalyClass.SYSCALL_DRIFT and at == ActionType.SYSCALL_BLOCK: return 0.28
        if ac == AnomalyClass.FORK_BOMB     and at == ActionType.SUSPEND:       return 0.20
        return 0.0

    # ── Action Application ────────────────────────────────────

    def apply(self, pid: int, action: InterventionAction) -> bool:
        """
        Apply intervention. Returns True on success.
        Correction #10: handler exceptions are caught. on_crash(pid) is called
        so state machine can reset. Never raises.
        """
        tag = "[DRY-RUN] " if self.dry_run else ""
        log.info(f"{tag}Applying {action.action_type.name} to pid={pid}")

        if self.dry_run:
            return True

        handler = _HANDLERS.get(action.action_type)
        if handler is None:
            log.warning(f"No handler for {action.action_type.name}")
            return False

        try:
            return handler(self, pid, action)
        except Exception as exc:
            log.error(
                f"[HANDLER CRASH] {action.action_type.name} pid={pid}: {exc}",
                exc_info=True,
            )
            if self.on_crash:
                try:
                    self.on_crash(pid)
                except Exception:
                    pass
            return False

    # ── Weight Update (called by FeedbackLoop) ────────────────

    def update_success_weight(self, action_type: ActionType, new_weight: float) -> None:
        with self._weights_lock:
            self._success_weights[action_type] = max(0.01, min(0.99, new_weight))

    # ── Handlers ─────────────────────────────────────────────

    def _handle_observe(self, pid: int, action: InterventionAction) -> bool:
        log.warning(f"[ALERT] pid={pid} flagged for enhanced monitoring")
        return True

    def _handle_throttle_cpu(self, pid: int, action: InterventionAction) -> bool:
        cg = self._cgroup_root / f"pid_{pid}"
        cg.mkdir(parents=True, exist_ok=True)
        quota_pct = action.params.get("cpu_quota_pct", 50)
        period_us = 100_000
        quota_us  = int(period_us * quota_pct / 100)
        try:
            (cg / "cpu.max").write_text(f"{quota_us} {period_us}\n")
            (cg / "cgroup.procs").write_text(str(pid))
            return True
        except PermissionError:
            log.error("cgroup write failed — need root + cgroup v2")
            return False

    def _handle_throttle_mem(self, pid: int, action: InterventionAction) -> bool:
        import psutil
        cg = self._cgroup_root / f"pid_{pid}"
        cg.mkdir(parents=True, exist_ok=True)
        try:
            rss    = psutil.Process(pid).memory_info().rss
            factor = action.params.get("headroom_factor", 1.2)
            limit  = int(rss * factor)
            (cg / "memory.max").write_text(str(limit))
            (cg / "cgroup.procs").write_text(str(pid))
            return True
        except Exception as e:
            log.error(f"Memory throttle failed: {e}")
            return False

    def _handle_syscall_block(self, pid: int, action: InterventionAction) -> bool:
        log.warning(f"SYSCALL_BLOCK stub — seccomp injection pending (Phase 2) pid={pid}")
        return False

    def _handle_checkpoint(self, pid: int, action: InterventionAction) -> bool:
        ckpt_dir = Path(f"/tmp/healix_ckpt_{pid}")
        ckpt_dir.mkdir(exist_ok=True)
        try:
            r = subprocess.run(
                ["criu", "dump", "-t", str(pid), "-D", str(ckpt_dir), "--shell-job"],
                capture_output=True, timeout=30,
            )
            return r.returncode == 0
        except FileNotFoundError:
            log.warning("criu not installed")
            return False

    def _handle_suspend(self, pid: int, action: InterventionAction) -> bool:
        try:
            os.kill(pid, signal.SIGSTOP)
            resume_sec = action.params.get("auto_resume_sec", 30)

            def _resume():
                import time
                time.sleep(resume_sec)
                try:
                    os.kill(pid, signal.SIGCONT)
                    log.info(f"Auto-resumed pid={pid}")
                except ProcessLookupError:
                    pass

            threading.Thread(target=_resume, daemon=True).start()
            return True
        except ProcessLookupError:
            return False

    def _handle_isolate_net(self, pid: int, action: InterventionAction) -> bool:
        log.warning(f"ISOLATE_NET stub — nsenter integration pending pid={pid}")
        return False

    def _handle_terminate(self, pid: int, action: InterventionAction) -> bool:
        import time
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(3)
            os.kill(pid, signal.SIGKILL)
            return True
        except ProcessLookupError:
            return True


_HANDLERS = {
    ActionType.OBSERVE:       RemediationEngine._handle_observe,
    ActionType.THROTTLE_CPU:  RemediationEngine._handle_throttle_cpu,
    ActionType.THROTTLE_MEM:  RemediationEngine._handle_throttle_mem,
    ActionType.SYSCALL_BLOCK: RemediationEngine._handle_syscall_block,
    ActionType.CHECKPOINT:    RemediationEngine._handle_checkpoint,
    ActionType.SUSPEND:       RemediationEngine._handle_suspend,
    ActionType.ISOLATE_NET:   RemediationEngine._handle_isolate_net,
    ActionType.TERMINATE:     RemediationEngine._handle_terminate,
}
