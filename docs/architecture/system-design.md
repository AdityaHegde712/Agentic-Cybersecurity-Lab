# System Design: Agentic Cyber-Defense for Sensor Data Attacks

**Version**: 1.0
**Date**: 2026-07-30
**Source synthesis**: `architecture-recommendation-v2.md`, `claude-agentic-cyber-defense-blueprint.md`, `equation-discovery-findings/UNIFIED_FINDINGS.md`, `taxonomy.md`, `docs\findings.md`, `_internal/scratchpad.md`

---

## 1. Problem & Scope

### 1.1 The Attack Model

Sensor data attacks corrupt the readings that cyber-physical systems use to make decisions. The canonical model:

```
y(t) = x(t) + delta_x(t)
```

Where:
- `x(t)` = true sensor value (unknown to the defender)
- `delta_x(t)` = adversarial perturbation (what the attacker adds)
- `y(t)` = observed sensor reading (what the defender sees)

**Three attack families in scope**:

| Attack | delta_x(t) behavior | Detection difficulty | Physical constraint |
|--------|---------------------|---------------------|---------------------|
| **FDIA** | Uses system physics (measurement matrix H) to compute delta_x that evades standard bad-data detection | High — looks physically valid | Must satisfy H * delta_x ≈ 0 |
| **Bias injection** | Constant offset, ramp, or periodic function added to sensor | Low-Medium — statistical tests catch step changes | Must not push readings outside physical bounds |
| **Replay** | Legitimate past sensor data retransmitted in place of current data | Medium — timing analysis reveals it | Replayed data was physically valid at capture time |

**Key insight from research**: All stealthy additive attacks are constrained by the system's physics. An attacker cannot push a pressure sensor to report 500 bar in a pipe rated for 10 bar without immediate detection. This means `delta_x` lives in a constrained subspace, and physics-aware models can learn those constraints.

Reference: `equation-discovery-findings/UNIFIED_FINDINGS.md` Section 1, `additive-attacks/ANALYSIS.md`

---

### 1.2 What This System Does

Six coordinated agents detect, model, and respond to sensor data corruption:

```
                         ┌─────────────────────────────┐
                         │   Sensor / Time-Series Feed   │
                         │  (HAI / SWaT / WADI / Synth)  │
                         └───────────────┬───────────────┘
                                         │
                                         ▼
                         ┌─────────────────────────────┐
                         │   1. Anomaly Detection Agent │
                         │   (LLM wrapper + ensemble    │
                         │    detector tool)            │
                         └───────────────┬───────────────┘
                                         │ anomaly scores
                                         ▼
                         ┌─────────────────────────────┐
              ┌─────────│   2. Attack Modeling Agent    │◄────────┐
              │         │   (RAG over attack taxonomy;   │         │
              │         │   proposes ranked hypotheses)  │         │
              │         └───────────────┬───────────────┘         │
              │                         │ hypotheses              │
              │                         ▼                        │
              │         ┌─────────────────────────────┐          │
              │         │   3. Simulation Agent        │          │
              │         │  (rule-based physical check  │          │
              │         │   OR perturbation replay)    │          │
              │         └───────────────┬───────────────┘          │
              │                         │ consistency verdict      │
              │          revise hyp.    ▼                          │
              └────────►(React loop 1-2x)──────────────────────────┘
                                         │
                                         ▼
                         ┌─────────────────────────────┐
                         │   4. Memory Agent             │
                         │  (FalkorDB episode store;     │◄──┐
                         │   retrieves top-k similar)    │   │ write
                         └───────────────┬───────────────┘   │ episode
                                         │ retrieved context  │ outcome
                                         ▼                    │
                         ┌─────────────────────────────┐      │
                         │   5. Planning / Response     │──────┘
                         │   Agent (fixed safe-action   │
                         │   taxonomy; human gate)      │
                         └───────────────┬───────────────┘
                                         │ recommended action
                                         ▼
                         ┌─────────────────────────────┐
                         │   Human Analyst              │
                         │   approves / overrides       │
                         └───────────────┬───────────────┘
                                         │ optional retraining batch
                                         ▼
                         ┌─────────────────────────────┐
                         │   6. Retraining-Gate Agent   │
                         │  (statistical poisoning      │
                         │   check before update)       │
                         └─────────────────────────────┘
```

Reference: `claude-agentic-cyber-defense-blueprint.md` Section 5-6

---

### 1.3 What's Not in Scope (Phase 1)

| Out of scope | Why | Future |
|-------------|-----|--------|
| Full physics digital twin | Too expensive for PoC. Use rule-based checks + perturbation replay | Phase 2 |
| Continuous adaptive retraining | Research-grade; demonstrate single offline cycle | Phase 2 |
| Physical sensor hardware | Digital simulation only | Phase 3 |
| LLM-based attack planning | No evidence LLMs reason about sensor data attacks well | Removed per C3 |
| MARL self-play | Convergence is open research challenge | Phase 2 |
| Real-time (<1s) deployment guarantees | Measure latency, optimize later | Phase 2 |

---

## 2. Detection Pipeline

### 2.1 Layered Architecture

Three-tier detection with increasing sophistication and latency:

```
Sensor Data
    │
    ▼
┌──────────────────────────────┐
│  Layer 1: Statistical        │  ~50ms (edge)
│  Screening (CUSUM / BOCPD)   │  Catches obvious anomalies
└──────────┬───────────────────┘
           │ confidence score
           ▼
┌──────────────────────────────┐
│  Confidence Gate             │
│  >0.85 → auto-classify      │  ~10ms
│  0.4-0.85 → ensemble        │
│  <0.4 → human/agent debate  │
└──────┬───────────┬───────────┘
       │           │
       ▼           ▼
┌──────────┐  ┌──────────────────┐
│ Auto-    │  │ Layer 2:         │  ~200ms (cloud)
│ classify │  │ Ensemble Scoring │
│ ~10ms    │  │ LSTM-AE + CUSUM  │
└──────────┘  │ + BOCPD          │
              └──────┬───────────┘
                     │ still uncertain?
                     ▼
              ┌──────────────────┐
              │ Layer 3: Agent   │  ~6s (background)
              │ Debate (3-4      │  3 specialist agents
              │ rounds)          │  argue classification
              └──────┬───────────┘
                     │ verdict
                     ▼
              ┌──────────────────┐
              │   Final Action   │
              └──────────────────┘
```

**Key latency insight**: Ensemble scoring finishes in ~200ms. Agent debate runs in **background** as a spawned sub-process. Initial response is based on ensemble alone (~250ms total). Debate refines classification if it completes within a configurable deadline. If debate takes too long, the ensemble verdict stands.

Reference: `_internal/scratchpad.md` (latency analysis), `architecture-recommendation-v2.md` Section 3.2

### 2.2 Statistical Screening Layer

**Purpose**: Fast, cheap anomaly flagging at the edge.

| Method | Complexity | Catches | Misses | Best for |
|--------|-----------|---------|--------|----------|
| **CUSUM** | O(1) | Step changes, sudden bias injection | Gradual drift, coordinated attacks | Baseline, edge deployment |
| **BOCPD** | O(n) | Gradual and sudden changes; provides uncertainty | Very slow drift, non-Gaussian data | Cloud analysis, UQ-needed |
| **COPOD** | O(nd) | Multi-sensor correlated attacks | Single-sensor subtle shifts | Correlated sensor networks |
| **FOCuS-cpt** | O(log^d n) | Exact GLR for multivariate changepoints | Resource-constrained devices | High-frequency sensors |

**Tier 1 recommendation**: Start with **CUSUM** (simplest), then add **BOCPD** (for UQ). Use **FOCuS-cpt** if multivariate detection needed at low latency.

Reference: `online-learning/FINAL_REPORT.md`, `online-learning/RECOMMENDATIONS.md`

### 2.3 Deep Detection Ensemble (Cloud)

Three-model ensemble providing complementary coverage:

| Model | Target attacks | Mechanism | Training |
|-------|---------------|-----------|----------|
| **LSTM-Autoencoder** | General anomalies, novel patterns | Reconstruction error | Unsupervised (normal data only) |
| **Anomaly Transformer** | Temporal pattern manipulation, slow-and-low | Association discrepancy | Unsupervised (normal data only) |
| **GNN (learned graph)** | Coordinated multi-sensor attacks | Graph attention on sensor correlations | Unsupervised (normal data only) |

**Ensemble combination**: Weighted average of anomaly scores. Weights learned via a simple meta-learner on held-out validation data.

**Caveat**: The AT+GNN+VT combination has never been validated together in the literature. Each component is validated independently. An ablation study in Build Phase 4 will test whether the ensemble provides ≥5% F1 improvement over the best single detector. If not, fall back to the best single model (likely LSTM-AE or Anomaly Transformer).

Reference: `taxonomy.md` Section 2, `architecture-recommendation-v2.md` Section 3, `survey-anomaly-detection.md`

### 2.4 Confidence-Based Routing

```
Anomaly score → confidence = 1 - uncertainty

    >0.85: AUTO-CLASSIFY
            Response: immediate automated action (alert, filter)
            Latency: ~250ms total

    0.4-0.85: ENSEMBLE RE-SCORING
            Response: run full ensemble, re-evaluate
            Then: if confidence > 0.85, auto-classify
                  Else: pass to agent debate
            Latency: ~200ms additional

    <0.4: DEFER TO HUMAN / AGENT DEBATE
            Response: spawn debate sub-process (3 specialist agents)
            Initial response based on best guess
            Debate refines classification (3-4 rounds, ~6s)
            Final verdict: majority vote or weighted confidence
```

Reference: `architecture-recommendation-v2.md` Section 2.4, `scratchpad.md`

---

## 3. Agent Architecture

### 3.1 Coordination Model: Hierarchical Hybrid

```
                     ┌────────────────────┐
                     │    ORCHESTRATOR     │
                     │  (LangGraph state   │
                     │   graph)            │
                     └──┬──────┬──────┬────┘
                        │      │      │
              ┌─────────┘      │      └─────────┐
              ▼                ▼                ▼
      ┌────────────┐   ┌────────────┐   ┌────────────┐
      │  RED TEAM  │   │ BLUE TEAM  │   │ GREEN TEAM │
      │  (Attack)  │   │ (Defense)  │   │ (Evaluate) │
      └────────────┘   └────────────┘   └────────────┘
```

**Hierarchical**: Orchestrator manages all inter-agent communication through a shared state graph. Deterministic execution = audit trails.

**Hybrid**: Agent roles can shift based on threat level. E.g., under heavy attack, the triage agent becomes an investigation agent.

**Why not flat**: Flat topologies are harder to coordinate for complex attack-response cycles. **Why not pure MARL**: Convergence is an open challenge — treat as future work.

Reference: `architecture-recommendation-v2.md` Section 2, `taxonomy.md` Section 1.1

### 3.2 Agent Roles (Sorted by Build Order)

| # | Agent | Team | Build | LLM Use | Automation Level |
|---|-------|------|-------|---------|-----------------|
| 1 | **Anomaly Detection Agent** | Blue | B5 | Wraps classical detector as tool; verbalizes results | Fully autonomous |
| 2 | **Attack Modeling Agent** | Blue | B5 | RAG over attack taxonomy; proposes ranked hypotheses | Semi-autonomous |
| 3 | **Simulation Agent** | Blue | B5 | Runs rule-based checks; no LLM | Fully automated (code) |
| 4 | **Memory Agent** | Blue | B5 | FalkorDB vector store; retrieval only | Fully automated (code) |
| 5 | **Response Agent** | Blue | B5 | LLM recommends from safe taxonomy; human gate | Human-in-the-loop |
| 6 | **Retraining-Gate Agent** | Green | Future | Statistical checks; no LLM needed | Semi-autonomous |

**Design principle**: Every agent calls deterministic tools for anything numeric. The LLM's job is hypothesis generation, explanation, comparison, and planning — never arithmetic over raw sensor values. This follows the AnomaMind pattern and avoids numeric hallucination.

Reference: `claude-agentic-cyber-defense-blueprint.md` Section 5

### 3.3 Orchestration Framework: LangGraph

| Feature | Why It Matters |
|---------|---------------|
| **Stateful graph execution** | Resumable workflows across long-running attack-defense cycles |
| **Typed state schemas** | Compile-time type checking for agent state |
| **Deterministic execution** | Audit trails — each agent decision is logged |
| **Cyclic graph support** | Simulation → revise hypothesis loop |
| **Human-in-the-loop interrupts** | Safety gate before high-impact actions |
| **Parallel branches** | Multi-sensor detection, parallel agent debate |

**State schema**:
```python
class LabState(TypedDict):
    cycle_id: int
    sensor_window: np.ndarray
    anomaly_scores: dict
    attack_hypotheses: list
    simulation_verdicts: list
    memory_context: list
    recommended_action: str
    human_approved: bool
    timestamp: datetime
```

Reference: `architecture-recommendation-v2.md` Section 5

---

## 4. Data Flow

### 4.1 Processing Cycle

```
1. INGEST: Sensor window received from data pipeline
2. DETECT: Statistical screening (CUSUM, ~50ms)
3. ROUTE: Confidence gate
   │
   ├─ HIGH (>0.85): Auto-classify, log, respond
   │
   └─ LOW-MEDIUM:
      ├─ Ensemble re-scoring (~200ms)
      │   ├─ HIGH now? → Auto-classify
      │   └─ Still LOW? → Spawn debate
      └─ Agent debate (background, ~6s)
           └─ Verdict → update classification
4. MODEL: Attack Modeling Agent proposes hypotheses
5. SIMULATE: Simulation Agent validates each hypothesis
6. RETRIEVE: Memory Agent fetches similar past episodes
7. RECOMMEND: Response Agent selects action from safe taxonomy
8. APPROVE: Human gate (if needed)
9. LOG: Full cycle recorded in episode store
```

### 4.2 Data Model

```
Episode {
    id: UUID
    timestamp: datetime
    sensor_data: {
        window: np.ndarray[T x N_sensors]
        metadata: {sensor_names, sampling_rate, ...}
    }
    detection: {
        method: str  # "cusum" | "bocpd" | "ensemble"
        anomaly_score: float
        confidence: float
        flagged_sensors: List[int]
    }
    attack_hypothesis: {
        type: str  # "fdia" | "bias" | "replay" | "unknown"
        confidence: float
        supporting_evidence: str
    }
    simulation_verdict: {
        consistent: bool
        details: str
    }
    memory_context: List[Episode]  # similar past episodes
    recommended_action: str  # "alert" | "isolate" | "flag" | "no_action"
    human_approved: bool
    human_override: Optional[str]
    outcome: {
        ground_truth: str
        correct: bool
        response_latency_ms: float
    }
}
```

---

## 5. Attack Simulation (Red Team)

### 5.1 Attack Generation Stack

```
┌────────────────────────────────────────────┐
│         Attack Simulator (Build B3)         │
│                                            │
│  Rule-Based Attack Library:                │
│  ├─ FDIA Generator                         │
│  │   └─ Stealth subspace: H * delta_x ≈ 0  │
│  ├─ Bias Injection                         │
│  │   └─ Step, ramp, periodic               │
│  ├─ Replay Attack                          │
│  │   └─ Capture → retransmit               │
│  └─ Coordinated Multi-Sensor               │
│      └─ Inject delta_x across N correlated │
│                                            │
│  Attack Parameters:                        │
│  - Magnitude, duration, start_time         │
│  - Stealth level (bypass CUSUM?)          │
│  - Target sensors                          │
└────────────────────────────────────────────┘
```

Reference: `additive-attacks/ANALYSIS.md`, `additive-attacks/KEY-EQUATIONS.md`

### 5.2 Attack Evaluation Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Attack success | Evaded detection by baseline detector | Varies |
| Stealth score | Statistical distance from normal (lower = stealthier) | < 0.5 |
| Impact score | System state deviation caused by attack | Documented |
| Parameter sensitivity | How attack magnitude affects detection probability | Ablation study |

---

## 6. Memory & Learning

### 6.1 Episode Store (FalkorDB)

Each full detection-to-response cycle is stored as an episode.

```
Episode → (embedding) → FalkorDB graph
  ↑
Query new incident → top-k similar past episodes
  → Retrieved context enriches Response Agent's recommendation
```

Schema:
```python
Episode {
    embedding: vector[384]  # sentence-transformer embedding
    window_hash: str        # for exact dedup
    anomaly_score: float
    attack_type: str
    simulation_verdict: str
    action_taken: str
    outcome: str            # correct / incorrect / unknown
}
```

Reference: `claude-agentic-cyber-defense-blueprint.md` Section 5 (Memory Agent)

### 6.2 Retraining Gate (Future)

Single offline retraining cycle with poisoning checks:
- Distributional distance from known-clean baseline (MMD test)
- Task-specific performance check (not just aggregate accuracy)
- If poisoned batch detected → reject, log, alert

Reference: `claude-agentic-cyber-defense-blueprint.md` Section 5 (Retraining-Gate Agent), Section 9.3 (PACOL threat model)

---

## 7. Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Language** | Python 3.10+ | ML ecosystem, research standard |
| **ML framework** | PyTorch 2.x | Industry standard for TSAD research |
| **Orchestration** | LangGraph | Deterministic, stateful, audit trails |
| **Episode store** | FalkorDB | Already proficient from personal assistant project |
| **LLM API** | Claude / Gemini (TBD) | Triage tier: cheap model. Analysis tier: strong model |
| **Data processing** | NumPy + SciPy | 10-line CUSUM, no framework needed |
| **Detection baseline** | LSTM-AE | Matches PyTorch familiarity, standard ICS-AD baseline |
| **Dataset loading** | `pwwl/ics-anomaly-detection` | Pre-built SWaT/WADI/BATADAL loaders |

Reference: `claude-agentic-cyber-defense-blueprint.md` Section 7-8

---

## 8. Datasets

| Dataset | Type | Sensors | Attack Scenarios | Access | Status |
|---------|------|---------|-----------------|--------|--------|
| **HAI** | Hydro/steam turbine | 59 | 38 labeled | Direct download | Use first |
| **SWaT** | Water treatment | 51 | 36 scenarios | iTrust request (~3 days) | Submit request Wk 1 |
| **WADI** | Water distribution | 123 | 15 scenarios | iTrust request (~3 days) | Submit request Wk 1 |
| **BATADAL** | Water network | Variable | Benchmark | iTrust request | When needed |
| **Synthetic** | Custom | Configurable | Fully labeled | Generated in B0 | Always available |

**WADI note (2026-07-30)**: WADI is **not openly accessible** — requires iTrust data request form submission (same gate as SWaT). Versions: A1 (Oct 2017), A2 (Nov 2019 — cleaned), A3 (Dec 2023 — 100-hour run). SWaT has a HuggingFace mirror at `THUgewu/SWaT` (CC BY 4.0) for a subset.

Reference: `architecture-recommendation-v2.md` Section 6.2, `graph-neural-networks/ANALYSIS.md` lines 50-68, iTrust website

---

## 9. Build Phases

```
Phase 1 (Week 1-2):  Foundation
    B0: Project scaffold + synthetic data generator + HAI loader
    B1: CUSUM baseline
    Study: A1 (changepoint), A2 (attacks), A8 (datasets)

Phase 2 (Week 3-4):  Level Up
    B2: BOCPD upgrade (adds uncertainty quantification)
    B3: Attack simulator (FDIA, bias, replay)
    Study: A3 (PINN), A4 (ensemble), A7 (Bayesian)

Phase 3 (Week 4-5):  Ensemble
    B4: Detection ensemble (LSTM-AE + CUSUM + BOCPD)
    Ablation: compare ensemble vs individual models
    Study: A5 (debate), A6 (SINDy)

Phase 4 (Week 5-6):  Agents
    B5: LangGraph agent loop (3 agents)
    End-to-end: HAI → Detection → Attack Modeling → Response → Human
```

Detailed build tasks: `.agent-tasks/architect/ORCHESTRATOR_IMPLEMENTATION_PLAN.md`

---

## 10. Research Questions (from Blueprint)

| RQ | Question | Tested By | Build |
|----|---------|-----------|-------|
| RQ1 | Does splitting detection, modeling, and planning into separate agents improve outcomes over a single LLM? | Compare full pipeline vs monolithic LLM | B5 (Phase 4) |
| RQ2 | Does memory of past episodes improve response recommendations? | With vs without Memory Agent | B5 (Phase 4) |
| RQ3 | Does simulation validation reduce hallucinated attack claims? | Track hypothesis acceptance rate with/without simulation | B5 (Phase 4) |

RQ4 (retraining gate) and RQ5 (pipeline robustness) are stretch goals for Phase 2.

Reference: `claude-agentic-cyber-defense-blueprint.md` Section 4

---

## 11. Risk Register

| Risk | P | I | Mitigation |
|------|---|---|------------|
| Ensemble doesn't outperform single detector | M | M | Ablation in B4; fall back to best single model |
| LLM hallucinates attack types | M | M | Simulation Agent validates all hypotheses; track rate |
| Dataset access delays | H | L | Start with HAI + synthetic; request SWaT/WADI early |
| MARL convergence failure | M | H | Not in Phase 1 scope — treat as research Q |
| Simulator scope creep | M | M | Rule-based checks + perturbation replay only |
| LangGraph complexity | H | M | Start with simple sequential graph; add cycles later |

Reference: `architecture-recommendation-v2.md` Section 10, `claude-agentic-cyber-defense-blueprint.md` Section 9

---

*Document generated 2026-07-30. Synthesizes v2 architecture recommendation, blueprint, unified findings, taxonomy, and scratchpad.*
