# Predicting Startup Follow-On Funding via Temporal Graph Neural Networks

> **Research Question:** Can Graph Neural Networks outperform traditional ML methods in predicting startup follow-on funding outcomes by leveraging venture capital network structure?

---

## 1. Problem Statement

Early-stage startup investing is highly uncertain. VCs evaluate startups using incomplete information — founder quality, market timing, investor reputation, and network effects. Many of these relationships are inherently **relational** and naturally representable as a **graph**.

This project constructs a **temporal venture capital investment graph** and evaluates whether GNNs can outperform traditional ML approaches in predicting startup follow-on funding outcomes.

### Deliverables

| # | Deliverable | Type |
|---|-------------|------|
| 1 | Complete graph ML pipeline | Code |
| 2 | Multiple GNN implementations (GCN, GraphSAGE, GAT) | Code |
| 3 | Node2Vec embeddings + similarity search | Code |
| 4 | Evaluation & ablation studies | Analysis |
| 5 | Conference-style research paper | Paper |

---

## 2. Hypothesis

> **The structure of venture capital investment networks contains predictive information beyond traditional startup metadata.**

Specifically, the following graph properties carry predictive signal:

```
┌─────────────────────────────────────────────────────┐
│  Investor relationships      → who backs whom       │
│  Co-investment patterns      → syndicate structure   │
│  Network centrality          → influence topology    │
│  Graph topology              → community structure   │
│  Temporal edge patterns      → investment momentum   │
└─────────────────────────────────────────────────────┘
```

---

## 3. Prediction Task (Locked)

```
┌──────────────────────────────────────────────────────────────────┐
│  TASK: Binary Node Classification                                │
│                                                                   │
│  INPUT:  Startup node immediately after Seed or Series A round   │
│  OUTPUT: Will it raise another round within 24 months?           │
│          → {0: No follow-on, 1: Follow-on raised}                │
└──────────────────────────────────────────────────────────────────┘
```

**Why this task:**
- Avoids extremely sparse IPO/acquisition labels
- Reduces survivorship bias
- Produces sufficient positive examples (~30-40% expected)
- Supports temporal graph modeling
- Feasible within project timeline

---

## 4. Success Criteria

| Level | Criterion | Status |
|-------|-----------|--------|
| ✅ Minimum | End-to-end graph ML pipeline | Required |
| ✅ Minimum | Temporal graph construction without leakage | Required |
| ✅ Minimum | ≥3 GNN models trained and evaluated | Required |
| ✅ Minimum | Node2Vec similarity search working | Required |
| ✅ Minimum | Conference-format paper completed | Required |
| 🎯 Strong | GNNs outperform tabular baseline | Goal |
| 🎯 Strong | Meaningful ablation studies | Goal |
| 🎯 Strong | Clear evidence graph structure improves prediction | Goal |
| 🎯 Strong | Strong visualization + error analysis | Goal |

---

## 5. Scope

### In Scope

| Domain | Components |
|--------|-----------|
| **Data** | Startup funding rounds, investor-startup relationships, temporal investment data, startup metadata |
| **Graph ML** | Temporal graph construction, Node2Vec embeddings, node classification (GCN, GraphSAGE, GAT), similarity search |
| **LLM Component** | Sentence embeddings for startup descriptions, textual similarity for link prediction |
| **Evaluation** | Precision, Recall, F1, ROC-AUC, ablation studies, error analysis |
| **Paper** | Conference-format, visualizations, comparison tables |

### Out of Scope

- ~~Production systems / real-time predictions~~
- ~~Large-scale distributed training~~
- ~~Training LLMs from scratch~~
- ~~HGT / Graph Transformers / deep heterogeneous architectures~~
- ~~Full deployment / web application~~
- ~~IPO/acquisition forecasting~~
- ~~Financial recommendation system claims~~

---

## 6. Dataset Strategy

### Required Data Fields

```yaml
Startup:
  - startup_id          # unique identifier
  - founded_year        # temporal feature
  - category/industry   # sector classification
  - location            # geographic feature
  - description         # text for embeddings
  - total_funding       # financial feature

Funding Round:
  - round_type          # Seed, A, B, C...
  - round_date          # CRITICAL for temporal splits
  - raised_amount       # deal size

Investor:
  - investor_id         # unique identifier
  - investor_name       # for similarity search
  - investment_history  # temporal investment pattern
```

### Dataset Constraints (Hard Requirements)

| Constraint | Reason |
|-----------|--------|
| Must contain **timestamps** | Temporal split validity |
| Must contain **investor↔startup edges** | Graph construction |
| Must allow **temporal splitting** | No future leakage |
| Must have enough **Seed → A/B transitions** | Label generation |

### Available Datasets

| Dataset | Records | Investor Data | Timestamps | Graph-Ready |
|---------|---------|--------------|------------|-------------|
| Global Startup Funding | 500 | ❌ | ❌ | ❌ |
| YC Startups (2005-2026) | 5,785 | ❌ | Partial (batch year) | ⚠️ |
| Techstars Companies | 4,875 | ❌ | Partial (session year) | ⚠️ |
| Unicorn Companies | 1,074 | ✅ (Select Investors) | ✅ (Date Joined) | ⚠️ |
| AI Funding DB (2014-2025) | 38 | ✅ (Lead + Others) | ✅ (deal_date) | ✅ |
| VCSheets Investors | TBD | ✅ | TBD | TBD |
| Startups & Funding (2024-25) | TBD | TBD | TBD | TBD |

> **⚠️ Critical Gap:** None of the currently downloaded datasets fully satisfy our requirements. The AI Funding DB has the right schema but only 38 records. We need a larger Crunchbase-derived dataset with investor-round-startup triples and timestamps. **VCSheets Investors** and **Crunchbase** sources should be prioritized.

---

## 7. Graph Architecture

```
                    ┌──────────────┐
                    │   INVESTOR   │
                    │              │
                    │ • portfolio  │
                    │ • success %  │
                    │ • centrality │
                    └──────┬───────┘
                           │
                    invested_in
                    (round_type,
                     date, amount)
                           │
                           ▼
                    ┌──────────────┐
                    │   STARTUP    │
                    │              │
                    │ • sector     │
                    │ • location   │
                    │ • funding    │
                    │ • round_cnt  │
                    └──────────────┘

        Optional: co_invested_with (Investor ↔ Investor)
```

### Node Features

| Node Type | Features |
|-----------|----------|
| **Startup** | founded_year, sector (one-hot), location, funding_amount, round_count, investor_count, description_embedding |
| **Investor** | total_investments, portfolio_size, historical_success_ratio, degree_centrality, betweenness_centrality |

### Edge Attributes

| Edge Type | Attributes |
|-----------|-----------|
| `invested_in` | round_type, investment_date, amount_raised |
| `co_invested_with` (optional) | num_co_investments, first_co_investment_date |

---

## 8. Temporal Integrity (CRITICAL)

> **⚠️ The graph must ONLY contain information available BEFORE the prediction timestamp. Future edges MUST NOT leak into training.**

```
Timeline:  ──────────────────────────────────────────────────►
                    │                         │
              Seed Round                 Series A
              (observed)                 (PREDICT THIS)
                    │                         │
                    ▼                         ▼
           ┌───────────────┐        ┌───────────────┐
           │ Graph at t₀   │        │ Label at t₁   │
           │ • All edges   │        │ • Did they    │
           │   before t₀   │        │   raise?      │
           │ • Node feats  │        │ • Within 24mo │
           │   at t₀       │        │               │
           └───────────────┘        └───────────────┘

  ❌ BAD:  Using Series B investors to predict Series B success
  ✅ GOOD: Only using information available immediately after Seed
```

**This is a central methodological contribution of the work.**

---

## 9. Modeling Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        MODELING PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │  BASELINES  │    │  EMBEDDINGS  │    │    GNN MODELS     │  │
│  ├─────────────┤    ├──────────────┤    ├───────────────────┤  │
│  │ • XGBoost   │    │ • Node2Vec   │    │ • GCN (classical) │  │
│  │ • RF        │    │ • Sentence   │    │ • GraphSAGE       │  │
│  │ (tabular    │    │   Embeddings │    │   (inductive)     │  │
│  │  features)  │    │   (HF model) │    │ • GAT (attention) │  │
│  └──────┬──────┘    └──────┬───────┘    └────────┬──────────┘  │
│         │                  │                      │              │
│         └──────────────────┼──────────────────────┘              │
│                            ▼                                     │
│                   ┌─────────────────┐                           │
│                   │   EVALUATION    │                           │
│                   ├─────────────────┤                           │
│                   │ • F1 / ROC-AUC  │                           │
│                   │ • Ablations     │                           │
│                   │ • Error Analysis│                           │
│                   └─────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### Model Details

| Model | Purpose | Key Advantage |
|-------|---------|---------------|
| **XGBoost** | Non-graph baseline | Proves graph value-add |
| **Node2Vec** | Embedding baseline + similarity search | Unsupervised structure learning |
| **GCN** | Classical graph convolution | Foundational GNN benchmark |
| **GraphSAGE** | Inductive learning | Scalable neighborhood sampling |
| **GAT** | Attention mechanism | Learns investor importance weights |

### Models Explicitly Deferred

- HGT (Heterogeneous Graph Transformer)
- Deep heterogeneous architectures
- Large-scale temporal GNNs (TGN, TGAT)

---

## 10. Node2Vec Similarity Search

**Goal:** Demonstrate learned graph embeddings qualitatively.

| Query Type | Example |
|-----------|---------|
| Startup similarity | "Find startups similar to Stripe after Seed" |
| Investor similarity | "Find investors similar to Sequoia" |
| Cross-embedding | "Find startups embedded near successful Series B companies" |

This satisfies the assignment's Node2Vec requirement while providing interpretable qualitative analysis for the paper.

---

## 11. LLM / Link Prediction Component

**Scope:** Lightweight — satisfies assignment requirement without scope explosion.

```
Startup Description → Sentence Transformer (HuggingFace)
                            ↓
                    384/768-dim embedding
                            ↓
              ┌─────────────┴─────────────┐
              │                           │
     Textual similarity          Link prediction assist
     (startup ↔ startup)        (investor → startup fit)
```

**Model:** `all-MiniLM-L6-v2` or `all-mpnet-base-v2` (open-source, no API costs)

---

## 12. Evaluation Plan

### Metrics

| Metric | Purpose |
|--------|---------|
| **Precision** | Avoid false positives (don't recommend bad bets) |
| **Recall** | Capture actual successes |
| **F1-Score** | Balanced measure (primary for imbalanced data) |
| **ROC-AUC** | Threshold-independent ranking quality |

### Ablation Studies

| Ablation | What it tests |
|----------|--------------|
| Without graph structure | Is the graph helping at all? |
| Without investor nodes | Are investor features driving performance? |
| Without node features | Is topology alone predictive? |
| Without temporal constraints | How much does leakage inflate results? |

### Error Analysis Focus

- **Cold-start startups** — few/no graph edges
- **Sparse sectors** — underrepresented industries
- **Early-stage** — minimal funding history

---

## 13. Central Research Insight

> **Does investor network quality matter more than startup metadata?**

Expected finding: graph topology contains predictive signals **not visible** in tabular startup features alone.

This forms the core **Discussion** section of the paper — comparing:
- Feature importance from XGBoost (tabular)
- Attention weights from GAT (graph)
- Embedding neighborhoods from Node2Vec (topology)

---

## 14. Visualizations

| Category | Tools | Outputs |
|----------|-------|---------|
| **Graph** | NetworkX, PyVis | Investment network topology, ego graphs |
| **Embeddings** | t-SNE, UMAP | 2D projections colored by outcome |
| **Evaluation** | Matplotlib, Seaborn | ROC curves, confusion matrices, training curves |
| **Ablation** | Matplotlib | Bar charts comparing model variants |

---

## 15. Timeline

```
WEEK 1 — Data & Graph Foundation
├── Dataset exploration & quality assessment
├── Label generation (follow-on within 24mo)
├── Temporal train/val/test split
├── Graph construction (NetworkX → PyG)
├── Network visualization
└── XGBoost baseline model

WEEK 2 — Representation Learning & GNNs
├── Node2Vec training + similarity search
├── Sentence embeddings (startup descriptions)
├── GCN implementation + training
├── GraphSAGE implementation + training
├── GAT implementation + training
└── Evaluation metrics computation

WEEK 3 — Analysis & Paper
├── Comparison tables (all models)
├── Ablation studies
├── Error analysis (cold-start, sector, stage)
├── All visualizations
├── Paper writing (conference format)
└── Final polish & submission
```

---

## 16. Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| **Temporal leakage** | Invalid results | Medium | Strict temporal cutoff, audit pipeline |
| **Dataset gaps** | Weak labels, sparse graph | High | Combine multiple sources, prioritize Crunchbase |
| **Class imbalance** | Inflated accuracy | High | F1/ROC-AUC as primary metrics, SMOTE/class weights |
| **Cold-start nodes** | Sparse neighborhoods | Medium | Strong node features as fallback |
| **Overengineering** | Missed deadline | Medium | Keep graph schema bipartite + minimal |
| **GNN ≈ Baseline** | Weak paper story | Medium | Frame as "when does graph help?" — still publishable |
| **LLM scope creep** | Timeline risk | Low | Minimal sentence-embedding approach only |

---

## 17. Open Questions (Resolve in Week 1)

### Dataset
- [ ] Does combined dataset have reliable per-round timestamps?
- [ ] Are investor↔startup relationships dense enough for message passing?
- [ ] Can we construct ≥5,000 labeled startup nodes?

### Methodology
- [ ] Optimal prediction horizon: 12 vs 24 months?
- [ ] Should co-investment edges be explicit or inferred?
- [ ] Bipartite graph vs. homogeneous projection?

### Evaluation
- [ ] Expected class imbalance ratio?
- [ ] Minimum graph density for GNN to outperform tabular?

---

## 18. Guiding Principles

| # | Principle | Implication |
|---|-----------|-------------|
| 1 | **Simplicity wins** | Smaller, cleaner, rigorous > large, unfinished |
| 2 | **Temporal integrity is sacred** | No leakage > adding another model |
| 3 | **Baselines matter** | Without XGBoost, you can't prove GNN value |
| 4 | **The story matters** | "Graph structure improves prediction" — not "AI predicts unicorns" |
| 5 | **Finishability is a feature** | Complete + polished >> ambitious + incomplete |

---

## 19. Paper Framing

**Working Title:** *"Temporal Graph Learning for Venture Capital Outcome Prediction"*

**Alternative Titles:**
- "Predicting Startup Follow-On Funding Using Graph Neural Networks"
- "Graph-Based Prediction of Follow-On Startup Funding"
- "Does Network Structure Predict Startup Success? A GNN Approach"

**Venue Targets (by format):**
- Workshop paper: KDD, ICML, NeurIPS (applied ML workshops)
- Full paper: AAAI, IJCAI (AI applications track)
- Domain: Financial ML conferences (ICAIF, FinNLP)

---

## 20. Product Vision (Future — Post-Paper)

> After the academic contribution is established, this work can evolve into a lightweight product.

**Phase 1 (Paper):** Research pipeline → validated methodology → published results

**Phase 2 (Prototype):** REST API serving predictions for new startups via trained model

**Phase 3 (Product):** 
- Investor dashboard with similarity search
- "Should I follow on?" decision support
- Portfolio-level network analysis

*Baby steps: the paper validates the approach. The product is the application.*
