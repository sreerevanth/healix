"""
Healix - Layer 2: Behavioral Fingerprint Model
================================================
Builds and maintains a "Process Behavioral DNA" for each running process.

Rule-based baseline. ML hooks are stubbed for Phase 2.

Each process gets a BehavioralFingerprint that tracks:
  - Syscall frequency distribution
  - Syscall sequence (rolling window)
  - Memory growth slope (via /proc)
  - Thread spawn density
  - File access entropy (unique fd opens)
  - Network call timing patterns
  - Fork/exec rate
"""

import math
import os
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import psutil


# ─────────────────────────────────────────────────────────────
# Syscall Classification (Linux x86-64)
# ─────────────────────────────────────────────────────────────

# Grouped by behavior type for feature engineering
SYSCALL_GROUPS = {
    "file_read":    {0, 17, 18, 19},                    # read, pread64, preadv, preadv2
    "file_write":   {1, 18, 20, 21},                    # write, pwrite64, writev, pwritev
    "file_open":    {2, 257, 304},                      # open, openat, openat2
    "memory":       {9, 10, 11, 12, 28},                # mmap, mprotect, munmap, brk, madvise
    "process":      {56, 57, 58, 59, 60, 61},           # clone, fork, vfork, execve, exit, wait4
    "network":      {41, 42, 43, 44, 45, 46, 47, 49},  # socket, connect, accept, send/recv, bind
    "ipc":          {62, 200, 202, 203, 204},           # kill, tkill, futex, etc
}

# Inverted: syscall_nr → group name
NR_TO_GROUP: Dict[int, str] = {}
for grp, nrs in SYSCALL_GROUPS.items():
    for nr in nrs:
        NR_TO_GROUP[nr] = grp


# ─────────────────────────────────────────────────────────────
# Process State Snapshot
# ─────────────────────────────────────────────────────────────

@dataclass
class ProcessState:
    """Point-in-time OS-level snapshot of a process."""
    timestamp:    float = 0.0
    rss_bytes:    int   = 0     # resident set size
    vms_bytes:    int   = 0     # virtual memory
    cpu_percent:  float = 0.0
    num_threads:  int   = 0
    num_fds:      int   = 0
    status:       str   = "unknown"

    @classmethod
    def capture(cls, pid: int) -> Optional["ProcessState"]:
        try:
            p   = psutil.Process(pid)
            mi  = p.memory_info()
            return cls(
                timestamp   = time.time(),
                rss_bytes   = mi.rss,
                vms_bytes   = mi.vms,
                cpu_percent = p.cpu_percent(interval=None),
                num_threads = p.num_threads(),
                num_fds     = p.num_fds(),
                status      = p.status(),
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None


# ─────────────────────────────────────────────────────────────
# Behavioral Fingerprint
# ─────────────────────────────────────────────────────────────

class BehavioralFingerprint:
    """
    Maintains rolling behavioral statistics for a single process.

    Feature vector (used by AnomalyScorer):
        - syscall_dist:         normalized Counter of syscall numbers
        - group_dist:           normalized Counter of syscall groups
        - recent_sequence:      deque of last N syscall numbers
        - syscall_rate:         calls per second (rolling)
        - fork_rate:            fork/clone/exec calls per second
        - net_rate:             network syscalls per second
        - memory_slope:         linear slope of RSS growth (bytes/sec)
        - thread_delta:         change in thread count
        - fd_entropy:           Shannon entropy of file descriptor opens
        - baseline_frozen:      True once enough samples collected
    """

    SEQUENCE_WINDOW    = 64     # syscall sequence length to keep
    RATE_WINDOW_SEC    = 10.0   # rolling window for rate calc
    MIN_SAMPLES        = 200    # samples before baseline is "trusted"
    STATE_INTERVAL_SEC = 2.0    # how often to refresh /proc state

    def __init__(self, pid: int, comm: str):
        self.pid   = pid
        self.comm  = comm
        self.born  = time.time()

        # Syscall distribution tracking
        self.syscall_counter:   Counter = Counter()
        self.group_counter:     Counter = Counter()
        self.total_calls:       int     = 0

        # Sequence tracking
        self.recent_sequence: deque = deque(maxlen=self.SEQUENCE_WINDOW)

        # Time-windowed rate tracking: list of (timestamp, syscall_nr)
        self._raw_events: deque = deque()  # (ts, syscall_nr)

        # Baseline statistics (computed once MIN_SAMPLES reached)
        self.baseline_syscall_dist: Optional[Dict] = None
        self.baseline_rate:         Optional[float] = None
        self.baseline_frozen:       bool            = False

        # Resource state history (sampled periodically)
        self._state_history: List[ProcessState]  = []
        self._last_state_ts: float               = 0.0
        self.latest_state:   Optional[ProcessState] = None

        # Derived metrics cache
        self._cached_metrics: Dict = {}
        self._metrics_dirty:  bool = True

    # ── Update ────────────────────────────────────────────────

    def update(self, event) -> None:
        """Ingest a new SyscallEvent."""
        nr  = event.syscall_nr
        ts  = event.timestamp_ns * 1e-9  # convert ns → sec

        self.total_calls += 1
        self.syscall_counter[nr] += 1
        grp = NR_TO_GROUP.get(nr, "other")
        self.group_counter[grp] += 1
        self.recent_sequence.append(nr)
        self._raw_events.append((ts, nr))

        # Prune old events outside rate window
        cutoff = ts - self.RATE_WINDOW_SEC
        while self._raw_events and self._raw_events[0][0] < cutoff:
            self._raw_events.popleft()

        # Periodically sample /proc
        now = time.time()
        if now - self._last_state_ts > self.STATE_INTERVAL_SEC:
            state = ProcessState.capture(self.pid)
            if state:
                self._state_history.append(state)
                self.latest_state   = state
                self._last_state_ts = now
                if len(self._state_history) > 60:  # keep ~2 min of history
                    self._state_history.pop(0)

        # Freeze baseline after enough samples
        if not self.baseline_frozen and self.total_calls >= self.MIN_SAMPLES:
            self._freeze_baseline()

        self._metrics_dirty = True

    # ── Baseline ──────────────────────────────────────────────

    def _freeze_baseline(self) -> None:
        """Snapshot current distribution as the "normal" baseline."""
        self.baseline_syscall_dist = dict(self._normalized_dist(self.syscall_counter))
        self.baseline_rate         = self.syscall_rate
        self.baseline_frozen       = True

    def _normalized_dist(self, counter: Counter) -> Dict[int, float]:
        total = sum(counter.values()) or 1
        return {k: v / total for k, v in counter.items()}

    # ── Computed Features ────────────────────────────────────

    @property
    def syscall_rate(self) -> float:
        """Syscalls per second in the current rolling window."""
        if not self._raw_events:
            return 0.0
        span = self._raw_events[-1][0] - self._raw_events[0][0]
        if span < 0.001:
            return float(len(self._raw_events))
        return len(self._raw_events) / span

    @property
    def fork_rate(self) -> float:
        fork_nrs = SYSCALL_GROUPS["process"]
        count = sum(1 for _, nr in self._raw_events if nr in fork_nrs)
        span  = max(1e-3, self._raw_events[-1][0] - self._raw_events[0][0]) if self._raw_events else 1
        return count / span

    @property
    def net_rate(self) -> float:
        net_nrs = SYSCALL_GROUPS["network"]
        count = sum(1 for _, nr in self._raw_events if nr in net_nrs)
        span  = max(1e-3, self._raw_events[-1][0] - self._raw_events[0][0]) if self._raw_events else 1
        return count / span

    @property
    def memory_slope(self) -> float:
        """RSS growth slope in bytes/sec. Positive = growing."""
        hist = self._state_history
        if len(hist) < 2:
            return 0.0
        # Simple linear regression (last 10 samples)
        window = hist[-10:]
        n = len(window)
        xs = [s.timestamp for s in window]
        ys = [s.rss_bytes  for s in window]
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = sum((x - x_mean) ** 2 for x in xs)
        return num / den if den > 1e-9 else 0.0

    @property
    def fd_entropy(self) -> float:
        """Shannon entropy of file syscall args (proxy for file access diversity)."""
        file_nrs = SYSCALL_GROUPS["file_open"]
        args = [arg0 for ts, nr in self._raw_events
                if nr in file_nrs
                for _, arg0, *_ in [("", 0)]]  # placeholder; real impl reads arg0
        if not args:
            return 0.0
        c = Counter(args)
        total = sum(c.values())
        return -sum((v / total) * math.log2(v / total + 1e-12) for v in c.values())

    @property
    def thread_delta(self) -> int:
        """Change in thread count vs. oldest available sample."""
        if len(self._state_history) < 2:
            return 0
        return self._state_history[-1].num_threads - self._state_history[0].num_threads

    # ── Feature Vector ────────────────────────────────────────

    def feature_vector(self) -> Dict:
        """Returns current behavioral feature dict for anomaly scorer."""
        return {
            "syscall_rate":    self.syscall_rate,
            "fork_rate":       self.fork_rate,
            "net_rate":        self.net_rate,
            "memory_slope":    self.memory_slope,
            "fd_entropy":      self.fd_entropy,
            "thread_delta":    self.thread_delta,
            "total_calls":     self.total_calls,
            "group_dist":      dict(self._normalized_dist(self.group_counter)),
        }

    def __repr__(self) -> str:
        return (
            f"<BehavioralFingerprint pid={self.pid} comm={self.comm!r} "
            f"calls={self.total_calls} frozen={self.baseline_frozen}>"
        )
