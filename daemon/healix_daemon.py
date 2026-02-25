#!/usr/bin/env python3
"""
Healix Daemon - Userspace Controller
=====================================
Layer 1 → 5 orchestrator.

Loads eBPF probe, reads syscall events from ring buffer,
feeds them through the behavioral fingerprint engine,
anomaly scorer, remediation selector, and feedback loop.

Requirements:
    pip install bcc psutil rich

Run as root:
    sudo python3 healix_daemon.py [--dry-run] [--log-level DEBUG]
"""

import argparse
import logging
import os
import signal
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import psutil

try:
    from bcc import BPF, PerfType, PerfSWConfig
    BCC_AVAILABLE = True
except ImportError:
    BCC_AVAILABLE = False
    print("[WARN] bcc not found. Running in SIMULATION mode.")

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    console = Console()
except ImportError:
    console = None

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

EBPF_SRC = Path(__file__).parent.parent / "ebpf" / "syscall_probe.c"

LOG_DIR  = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "healix.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("healix.daemon")

# ─────────────────────────────────────────────────────────────
# Import internal layers
# ─────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fingerprint import BehavioralFingerprint, ProcessState
from models.anomaly_scorer import AnomalyScorer
from models.context_model import ContextConditionedModel, AnomalyClass, classify_anomaly
from models.recovery_metric import RecoveryMonitor
from interventions.remediation import RemediationEngine
from feedback.feedback_loop import FeedbackLoop
from daemon.pid_state_machine import PIDStateMachine

# ─────────────────────────────────────────────────────────────
# Event dataclass (mirrors C struct syscall_event)
# ─────────────────────────────────────────────────────────────

@dataclass
class SyscallEvent:
    timestamp_ns: int
    pid: int
    ppid: int
    uid: int
    syscall_nr: int
    arg0: int
    arg1: int
    arg2: int
    comm: str

# ─────────────────────────────────────────────────────────────
# Daemon
# ─────────────────────────────────────────────────────────────

class HealixDaemon:
    def __init__(self, dry_run: bool = False):
        self.dry_run   = dry_run
        self.running   = False
        self.bpf       = None

        # Layer 2: per-process behavioral fingerprints
        self.fingerprints: Dict[int, BehavioralFingerprint] = {}

        # Layer 3: anomaly scorer
        self.scorer = AnomalyScorer()

        # Layer 4: remediation engine
        self.remediator = RemediationEngine(dry_run=dry_run)

        # Layer 5: feedback loop
        self.feedback = FeedbackLoop()

        # Production components
        self.state_machine    = PIDStateMachine()
        self.context_model    = ContextConditionedModel()
        self.recovery_monitor = RecoveryMonitor(monitor_window_sec=30.0)
        self.remediator.on_crash = self.state_machine.reset_on_crash  # Correction #10

        # Tracking
        self.event_count    = 0
        self.anomaly_count  = 0
        self.repair_count   = 0

        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    # ── eBPF Setup ────────────────────────────────────────────

    def load_ebpf(self):
        if not BCC_AVAILABLE:
            log.warning("BCC unavailable – using simulated event stream")
            return

        if not EBPF_SRC.exists():
            raise FileNotFoundError(f"eBPF source not found: {EBPF_SRC}")

        log.info(f"Loading eBPF probe from {EBPF_SRC}")
        self.bpf = BPF(src_file=str(EBPF_SRC))
        log.info("eBPF probe loaded successfully")

    # ── Event Callback ────────────────────────────────────────

    def handle_event(self, cpu, data, size):
        """Called for each syscall event from ring buffer."""
        if not self.bpf:
            return

        raw = self.bpf["events"].event(data)

        event = SyscallEvent(
            timestamp_ns = raw.timestamp_ns,
            pid          = raw.pid,
            ppid         = self._get_ppid(raw.pid),
            uid          = raw.uid,
            syscall_nr   = raw.syscall_nr,
            arg0         = raw.arg0,
            arg1         = raw.arg1,
            arg2         = raw.arg2,
            comm         = raw.comm.decode("utf-8", errors="replace").rstrip("\x00"),
        )

        self._process_event(event)

    def _process_event(self, event: SyscallEvent):
        """Core pipeline: fingerprint → score → state gate → remediate → async feedback."""
        self.event_count += 1

        # ── Layer 2: Behavioral fingerprint ──────────────────
        if event.pid not in self.fingerprints:
            self.fingerprints[event.pid] = BehavioralFingerprint(
                pid=event.pid, comm=event.comm
            )
            self.state_machine.get_or_create(event.pid, event.comm)

        fp = self.fingerprints[event.pid]
        fp.update(event)

        # ── Layer 3: Anomaly scoring ──────────────────────────
        score, reasons = self.scorer.score(fp, event)
        self.state_machine.update_score(event.pid, score)

        if score <= self.scorer.threshold:
            return

        # ── Metrics ───────────────────────────────────────────
        self.anomaly_count += 1
        self.feedback.inc_anomaly()
        log.warning(
            f"[ANOMALY] pid={event.pid} comm={event.comm} "
            f"score={score:.3f}"
        )

        # ── State Gate (Correction #5) ────────────────────────
        allowed, gate_reason = self.state_machine.can_intervene(event.pid, score)
        if not allowed:
            if "cooldown" in gate_reason:
                self.feedback.inc_cooldown_block()
            log.debug(f"[GATE] pid={event.pid} blocked: {gate_reason}")
            return

        # ── Context Classification (Correction #2: metric-based) ──
        features      = fp.feature_vector()
        anomaly_class = classify_anomaly(features)

        # ── Escalation check (Correction #6: per pid+class) ───
        record = self.state_machine.get_or_create(event.pid, event.comm)
        escalation_hint = (
            record.last_action_name is not None
            and self.state_machine.should_escalate(
                event.pid,
                record.last_action_name,
                anomaly_class.value,
            )
        )
        if escalation_hint:
            self.feedback.inc_escalation()

        # ── Layer 4: Remediation ──────────────────────────────
        if not self.state_machine.begin_remediation(event.pid):
            log.debug(f"[GATE] pid={event.pid} begin_remediation rejected")
            return

        action = self.remediator.select_action(
            fp, score, anomaly_class, self.context_model, escalation_hint
        )

        success = self.remediator.apply(event.pid, action)
        self.feedback.inc_intervention()
        if success:
            self.repair_count += 1
        if action.action_type.name == "TERMINATE":
            self.feedback.inc_termination()

        self.state_machine.end_remediation(
            event.pid,
            action.action_type.name,
            anomaly_class.value,
            success,
        )

        # ── Layer 5: Async feedback evaluation ───────────────
        self.feedback.dispatch_evaluation(
            monitor       = self.recovery_monitor,
            pid           = event.pid,
            scorer        = self.scorer,
            fingerprint   = fp,
            s_before      = score,
            action        = action,
            anomaly_class = anomaly_class,
            context_model = self.context_model,
            score_history = list(record.score_history),
            comm          = event.comm,
        )

        self.scorer.threshold = self.feedback.recommended_threshold()

    # ── Simulation Mode ───────────────────────────────────────

    def _simulate_events(self):
        """Generate synthetic events when bcc is not available."""
        import random
        SYSCALLS = [0, 1, 2, 3, 9, 56, 59, 105, 293]  # read,write,open,mmap,fork,execve,etc

        pid_pool = [1000 + i for i in range(5)]
        comms    = ["nginx", "postgres", "python3", "node", "redis"]

        while self.running:
            pid  = random.choice(pid_pool)
            idx  = pid_pool.index(pid)
            comm = comms[idx]

            # Occasionally inject anomalous burst
            burst = random.random() < 0.02  # 2% chance
            count = random.randint(50, 200) if burst else 1

            for _ in range(count):
                event = SyscallEvent(
                    timestamp_ns = time.time_ns(),
                    pid          = pid,
                    ppid         = 1,
                    uid          = 1000,
                    syscall_nr   = random.choice(SYSCALLS),
                    arg0         = random.randint(0, 0xFFFF),
                    arg1         = random.randint(0, 0xFFFF),
                    arg2         = 0,
                    comm         = comm,
                )
                self._process_event(event)

            time.sleep(0.005)

    # ── Main Loop ─────────────────────────────────────────────

    def run(self):
        self.running = True
        log.info("Healix daemon starting...")

        self.load_ebpf()

        if self.bpf:
            self.bpf["events"].open_ring_buffer(self.handle_event)
            log.info("Ring buffer open. Monitoring syscalls...")
            while self.running:
                self.bpf.ring_buffer_poll(timeout=100)
                self._print_stats()
        else:
            log.info("Simulation mode active.")
            self._simulate_events()

    def _print_stats(self):
        if self.event_count % 10000 == 0 and self.event_count > 0:
            log.info(
                f"Stats → events={self.event_count} "
                f"anomalies={self.anomaly_count} "
                f"repairs={self.repair_count} "
                f"processes_tracked={len(self.fingerprints)}"
            )

    def _get_ppid(self, pid: int) -> int:
        try:
            return psutil.Process(pid).ppid()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    def _shutdown(self, signum, frame):
        log.info("Shutting down Healix daemon...")
        self.running = False
        self.feedback.save()
        sys.exit(0)


# ─────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Healix Self-Healing OS Daemon")
    parser.add_argument("--dry-run",    action="store_true", help="Detect only, no interventions")
    parser.add_argument("--log-level",  default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    if not BCC_AVAILABLE and os.geteuid() != 0:
        log.warning("Not running as root AND bcc unavailable → simulation mode")

    daemon = HealixDaemon(dry_run=args.dry_run)
    daemon.run()
