![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-production--hardened-brightgreen)
![Platform](https://img.shields.io/badge/platform-linux-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-60%20passed-success)
# Healix

> Adaptive Runtime Process Remediation Engine for Linux

Healix is a self-healing runtime controller that detects anomalous process behavior and applies the least invasive corrective action using a bounded, context-aware, reward-based policy.

Instead of blindly terminating unstable processes, Healix detects, evaluates, intervenes, and learns.

---

## 🚀 What Problem Does It Solve?

Traditional systems either:
- Ignore abnormal behavior  
- Or terminate processes aggressively  

Healix introduces a structured remediation ladder with adaptive learning:

Detect → Select safest viable action → Measure recovery → Update policy

---

## 🧠 Core Architecture
healix/
├── ebpf/ # Syscall observation layer
├── daemon/ # Orchestrator + PID state machine
├── models/ # Behavioral fingerprint + anomaly scoring
├── interventions/ # Remediation engine (bounded ladder)
├── feedback/ # Contextual reward-based learning
├── tests/ # Production hardening test suite
├── logs/ # Runtime logs
└── requirements.txt

---

## 🧩 How It Works

### 1️⃣ Behavioral Fingerprint
Per-process behavioral features:
- syscall_rate
- memory_slope
- fork_rate
- network_rate
- syscall distribution drift (KL divergence)

### 2️⃣ Anomaly Scoring
Composite anomaly score ∈ [0,1].

### 3️⃣ Context Classification
Metric-based deterministic classification:
- MEMORY_LEAK
- FORK_BOMB
- NETWORK_SURGE
- SYSCALL_DRIFT
- CPU_SPIKE
- UNKNOWN

### 4️⃣ Severity-Bound Utility Selection

Utility Function:
utility = P(success | action, anomaly_class)
× (1 - invasiveness × severity)
× (1 + context_alignment_bonus)

Actions are filtered by a severity-based invasiveness ceiling.

### 5️⃣ Reward-Based Adaptive Learning

After intervention:

ΔS = max(0, S_before - S_after)

reward = clamp(
W1 * (ΔS / max(S_before, ε))

W2 * stability_score

W3 * recurrence_penalty,
0.0,
1.0
)


Weights updated via bounded EMA:

new_weight = old_weight + α * (reward - old_weight)

Context-conditioned per (action, anomaly_class).

---

## 🛡 Safety & Stability

- Strict PID state machine  
  NORMAL → ANOMALOUS → REMEDIATING → COOLDOWN  
- Escalation scoped per (pid, anomaly_class)  
- Cooldown override only on 15% severity increase  
- Crash-safe handlers  
- Thread-safe weight updates  
- MAX_ACTIVE_EVALS cap  
- 60-check production test suite  

---

## 📊 Metrics Exposed

`feedback.get_stats()` returns:

- total_anomalies
- total_interventions
- recovery_success_rate
- avg_reward
- termination_rate
- escalation_count
- cooldown_blocks
- active_evaluations
- current_threshold

---

## 🧪 Running Tests

From project root:

```bash
python -m tests.test_production

All 60 checks must pass.
⚙ Requirements
pip install -r requirements.txt




									🎯 Why Healix?

Healix is not just monitoring.

It is a bounded, contextual, adaptive control system operating over runtime process behavior.

It detects.
It intervenes.
It evaluates.
It learns.
