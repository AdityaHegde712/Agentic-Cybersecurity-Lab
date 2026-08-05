## What the Research Found

### The Problem Space

You asked us to investigate building an **agentic cybersecurity lab** that simulates, detects, and refines responses to **sensor-level data corruption attacks** on time-series data. The lab's purpose is to let attack and defense agents co-evolve — getting better at attacking and defending over time.

We surveyed **58 papers** across three domains:

| Domain                | Papers | What We Searched                                             |
| --------------------- | ------ | ------------------------------------------------------------ |
| Agentic Orchestration | 25     | How multi-agent systems coordinate for security tasks        |
| Anomaly Detection     | 18     | How ML/DL models detect anomalies in time-series sensor data |
| Attack Simulation     | 15     | How attacks on sensor data are generated and evaluated       |

---

### Finding 1: Nobody Has Built What You're Describing

The single most important finding: **no existing paper provides a unified framework combining offensive and defensive multi-agent systems for sensor data attacks.**

- Offensive agents (red teams) exist — BreachSeek, VulnBot — but they target **network/penetration testing**, not sensor data.
- Defensive agents (blue teams) exist — SOC architectures, MARL IDS — but they're **detection-only**, not co-evolving with attackers.
- Cyber ranges exist — CyberBattleSim, MITRE CALDERA — but they focus on **network/systems attacks**, not sensor-layer data corruption.

Your lab would fill a genuine gap: a sensor-specific, integrated attack-defense framework with co-evolution.

---

### Finding 2: The Sensor Data Angle Is Distinct

Sensor data attacks are fundamentally different from network attacks:

| Characteristic       | Network Attacks       | Sensor Data Attacks                                  |
| -------------------- | --------------------- | ---------------------------------------------------- |
| Data type            | Packets, flows        | Time-series (pressure, temperature, flow, vibration) |
| Attack surface       | Protocol exploitation | Data corruption, injection, drift, replay            |
| Spatial structure    | Network topology      | Inter-sensor physical correlations                   |
| Temporal dynamics    | Burst/flow patterns   | Slow-and-low drift, seasonal patterns                |
| Physical constraints | None                  | Attacks must respect (or exploit) physics            |

This matters because **detection models and attack strategies designed for network data don't transfer directly to sensor data.** The taxonomy maps out which detection methods work against which attack types on sensor data specifically.

---

### Finding 3: The Detection Landscape Has Clear Winners

For time-series sensor data anomaly detection, three model families stand out:

| Model                       | What It Catches                                    | What It Misses              | Source Quality   |
| --------------------------- | -------------------------------------------------- | --------------------------- | ---------------- |
| **Anomaly Transformer**     | Long-range temporal patterns, slow-and-low attacks | Coordinated spatial attacks | 🟢 ICLR 2022     |
| **GNN (learned graph)**     | Coordinated injection across correlated sensors    | Sequential temporal attacks | 🟢 AAAI 2021     |
| **Variational Transformer** | Novel/ambiguous patterns (via uncertainty)         | Computationally expensive   | 🟢 Peer-reviewed |

The research recommends **combining all three** into an ensemble. However — and the adversary flagged this — **this specific combination has never been validated.** Each model is evaluated independently in the literature. We've built an ablation study into Phase 2 to test whether the ensemble actually outperforms individual models.

**Unsupervised learning dominates** — most sensor data lacks labeled attack examples, so models trained on "normal" data only are the practical baseline.

---

### Finding 4: The Agent Architecture Has Converged on Hierarchical Hybrid

Three coordination topologies exist:

```
Hierarchical          Flat/Modular           Hybrid (Dynamic)
    ┌───┐              A ←→ B ←→ C          ┌───┐ ←→ ┌───┐
    │ O │             ↕     ↕     ↕          │ O │    │ O │
   ╱│││╲             D ←→ E ←→ F          └───┘    └───┘
  A B C D                                    ↕         ↕
                                        Fixed roles  Evolving roles
```

**Recommendation: Hierarchical Hybrid.** Why:

- Hierarchical = clear command chain, deterministic execution, audit trails (critical for security governance)
- Hybrid = agent roles can shift based on threat level (e.g., a triage agent becomes an investigation agent under heavy attack)
- Flat is simpler but harder to coordinate complex attack-response cycles
- Pure MARL is powerful but convergence is an open challenge

---

### Finding 5: The Orchestration Framework Choice Is Narrow

| Framework     | Status                            | Security Governance    | Cyclic Execution | Verdict              |
| ------------- | --------------------------------- | ---------------------- | ---------------- | -------------------- |
| **LangGraph** | Active                            | Strong (deterministic) | Native           | **Recommended**      |
| **CrewAI**    | Active                            | Weak (no audit)        | Limited          | Good for prototyping |
| **AutoGen**   | **Maintenance mode** (April 2026) | Weak                   | Native           | **Do not use**       |
| **MAF v1.0**  | New (April 2026)                  | Unknown                | Unknown          | Monitor              |
| **AG2**       | Fork                              | Unknown                | Unknown          | Monitor              |

LangGraph wins on **deterministic execution** (audit trails), **state persistence** (long-running attack-defense cycles), and **human-in-the-loop** (response approval gates). AutoGen is dead — Microsoft redirected to MAF.

**Caveat**: Token overhead numbers (9% LangGraph vs 31% AutoGen) came from a predatory journal source. Don't trust those numbers. The framework choice is justified by architectural features, not benchmarks.

---

### Finding 6: MARL Self-Play Is the Most Promising (and Riskiest) Component

The attack-defense co-evolution mechanism uses **Multi-Agent Reinforcement Learning** with alternating game structure:

```
Round t:
  Red Agent → selects attack strategy
  Environment → applies attack to sensor data
  Blue Agent → observes corruption, selects detection response
  Both → receive rewards (attack success vs. detection accuracy)
  Both → update policies
```

**Why promising**: Can discover attack/defense strategies not in any fixed library. Agents co-adapt over time.

**Why risky**: MARL convergence in multi-agent adversarial settings is an **open research challenge** (Landolt et al., 2025). There's no guarantee the red and blue agents will converge to meaningful strategies.

**Mitigation**: The architecture includes four fallback mechanisms — fixed-strategy freeze, curriculum learning, reward shaping, and population-based training. The lab treats convergence failure as a research finding, not a system failure.

---

### Finding 7: The LLM's Role Should Be Minimal

This was one of the adversary's critical findings. The original architecture had the LLM **planning sensor-specific attacks**. But:

- No surveyed paper demonstrates LLM-based planning for sensor data attacks (FDIA, calibration drift, replay on IoT protocols)
- LLM attack papers (Papers #2, #6, #8) focus on **phishing, prompt injection, and browser agent attacks** — not sensor data
- LLMs don't have the domain knowledge to reason about sensor topology, bad data detection bypass, or gradient-based perturbation

**Revised role**: LLM is auxiliary only — generating human-readable reports, post-hoc analysis, lessons learned. Attack selection is **rule-based + MARL adaptive**, not LLM-planned.

---

### Finding 8: The Training Data Problem Is Real

The original architecture referenced CIC-IDS-2017 for training. The adversary correctly flagged: **that's network traffic data, not sensor data.**

**Corrected to**:

- **SWaT** (Secure Water Treatment) — real ICS sensor time-series with documented attack injections
- **WADI** (Water Distribution) — water network sensor correlations
- **BATADAL** — purpose-built for attack detection evaluation

These are the only public datasets with real sensor time-series + documented attacks that match the lab's focus.

---

### Summary: What Exists vs. What You're Building

```
EXISTS                          YOUR LAB
─────────────────────          ─────────────────────────
Network penetration agents     Sensor data attack agents ✓ (NEW)
Detection-only pipelines       Co-evolving attack+defense ✓ (NEW)
Fixed attack libraries         MARL adaptive attacks ✓ (NEW)
Single-model detection         3-model ensemble ✓ (NEW)
No adversarial robustness      Adversarial robustness benchmark ✓ (NEW)
Network-focused cyber ranges   Sensor-specific cyber range ✓ (NEW)
```

The novelty claim is narrow and defensible: **"Sensor-specific integrated attack-defense framework with MARL co-evolution."**

---

Ready for section 2 (what we learned)?

---
