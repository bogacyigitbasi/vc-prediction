# Project Status: Temporal Graph Learning for VC Outcome Prediction

**Last updated:** May 23, 2026

---

## What Is This Project?

We're building a **Graph Neural Network (GNN) pipeline** that predicts whether a startup will raise **follow-on funding within 24 months** of its Seed or Series A round. The core insight: investor network structure (who invests alongside whom, which investors back which startups) carries predictive signal beyond what tabular company features alone can provide.

**Primary goal:** Conference-ready academic paper  
**Secondary goal:** Product feasibility validation (baby steps)

---

## Dataset

**Source:** Crunchbase 2015 relational export ([notpeter/crunchbase-data](https://github.com/notpeter/crunchbase-data) on GitHub)

| File | Rows | Key Columns |
|------|------|-------------|
| `investments.csv` | 168,647 | investor_permalink, company_permalink, funded_at, raised_amount_usd |
| `rounds.csv` | 114,949 | company_permalink, funding_round_type, funded_at, raised_amount_usd |
| `companies.csv` | 66,368 | permalink, name, category_list, country_code, founded_at |
| `acquisitions.csv` | 19,115 | (available but not yet used) |

Stored in: `data/raw/crunchbase_2015/` (immutable -- never modify)

---

## What's Been Built (Steps 1-6 Complete)

### Step 1: Data Exploration
- **Notebook:** `notebooks/01_data_exploration.ipynb`
- Profiled all CSVs, validated joins, plotted distributions
- Confirmed data quality is sufficient (>95% dates parse, >10K eligible startups)

### Step 2: Label Generation
- **Module:** `src/data/label_generator.py`
- **Output:** `data/interim/labels.csv` (23,740 labeled companies)
- Logic: Find each company's first Seed or Series A round ("trigger"), check if they raised again within 24 months
- Observable window constraint: trigger + 24 months must be before dataset end (Dec 2015)
- **Result:** 39.1% positive rate (good class balance)

### Step 3: Temporal Split
- **Module:** `src/data/temporal_split.py`
- **Output:** `data/interim/splits.csv`
- Split by `trigger_funded_at` date (NOT random):

```
TRAIN:  trigger_funded_at < 2012-01-01    --> 12,920 companies (38.3% positive)
VAL:    2012-01-01 <= trigger < 2013-01-01 -->  4,831 companies (40.3% positive)
TEST:   2013-01-01 <= trigger < 2014-01-01 -->  5,989 companies (40.1% positive)
```

- Validated: no company appears in multiple splits, strict temporal ordering
- Note: Originally planned 2013/2014/2014-H1 cutoffs, but shifted one year earlier because the 24-month horizon + dataset end made the test set empty

### Step 4: Graph Construction
- **Module:** `src/data/graph_builder.py`
- **Output:** `data/processed/edges_{train,val,test}.csv`, `company_index.csv`, `investor_index.csv`
- Bipartite graph: Investor <-> Company
- 66,369 company nodes, 30,732 investor nodes
- Edges are **temporally filtered** -- each split's graph only contains edges before its cutoff date:

```
train graph:  73,509 edges (before 2012-01-01)
val graph:    91,573 edges (before 2013-01-01)
test graph:  115,564 edges (before 2014-01-01)
```

- Edge features: round_type_encoded, amount_log, days_since_epoch

### Step 5: Node Feature Engineering
- **Module:** `src/features/node_features.py`
- **Output:** `data/processed/company_features_{split}.npy`, `investor_features_{split}.npy`

**Company features (58 dims):**
| # | Feature | Notes |
|---|---------|-------|
| 0 | founded_year | Normalized |
| 1 | days_since_founded | From cutoff date |
| 2 | log_total_raised | Pre-cutoff only |
| 3 | round_count | Pre-cutoff only |
| 4 | investor_count | Pre-cutoff only |
| 5 | log_max_amount | Largest round pre-cutoff |
| 6 | days_since_last_round | From cutoff date |
| 7 | is_us | Binary (country_code == "USA") |
| 8-57 | Top-50 category one-hot | From category_list |

**Investor features (7 dims):**
total_investments, portfolio_size, category_diversity, log_median_amount, years_active, avg_round_stage, geo_diversity

All features respect temporal cutoffs and are Z-score normalized.

### Step 6 (Ready): XGBoost Baseline + GNN Training
- **Module:** `src/models/baseline_xgboost.py`
- **Module:** `src/evaluation/metrics.py`
- **Notebook:** `notebooks/colab_training.ipynb` -- ready to run, not yet executed

---

## What's Ready But Not Yet Run

### The Colab Training Notebook (`notebooks/colab_training.ipynb`)

This single notebook covers Steps 6-11 of the plan. It's complete but has NOT been executed yet.

**What it will do when you run it:**

| Section | What Happens |
|---------|-------------|
| Step 6: XGBoost Baseline | Train XGBoost on tabular features only. Establishes the performance floor that GNNs must beat. Includes feature importance plot. |
| Step 7: Node2Vec | Train Node2Vec embeddings on the bipartite graph (homogeneous projection). Includes company similarity search demo (e.g., "companies similar to Stripe"). |
| Step 8: Sentence Embeddings | Encode `category_list` text via `all-MiniLM-L6-v2` sentence-transformer. Lightweight LLM component. |
| Step 9: GCN | Train Graph Convolutional Network via PyG's `to_hetero()` wrapper. |
| Step 10: GraphSAGE | Train GraphSAGE (inductive -- handles unseen nodes at inference). |
| Step 11: GAT | Train Graph Attention Network (provides attention weights for interpretability). |
| Evaluation | Model comparison table, ROC curves for all models, training loss/F1 curves. |

**Architecture:** All three GNNs share the same `HeteroGNN` class:
```
company features (58d) --> Linear --> hidden (128d) --|
                                                      |--> 2-layer GNN (message passing) --> classifier --> P(follow-on)
investor features (7d) --> Linear --> hidden (128d) --|
```

**To run it, you need to:**
1. Upload ~65MB of data to Google Colab (see next section)
2. Select GPU runtime in Colab
3. Run all cells

---

## How to Run the Colab Notebook

### Step 1: Create data zip
```bash
cd /Volumes/ThunderDB/ML_Investor_Company_Raise
zip -r ml_investor_data.zip data/processed/ data/interim/
```

### Step 2: Upload to Google Drive
- Upload `ml_investor_data.zip` (~65MB) to your Google Drive root folder

### Step 3: Open Colab
- Upload `notebooks/colab_training.ipynb` to Google Colab
- Set Runtime > Change runtime type > **T4 GPU**

### Step 4: Run
- The notebook mounts Google Drive, extracts the zip, and runs everything
- Expected runtime: ~30-45 minutes total

### What you'll get back:
- `model_results.csv` -- comparison table (F1, AUC, Precision, Recall for all models)
- `predictions_*.npz` -- raw predictions for error analysis
- `*_model.pt` -- saved model weights
- `roc_curves.png`, `training_curves.png` -- paper-ready figures
- `text_embeddings.pt` -- sentence-transformer embeddings

---

## What's NOT Done Yet (Steps 12-16)

These steps require training results from the Colab notebook first:

| Step | What | Status |
|------|------|--------|
| 12 | **Evaluation & Comparison** -- Bootstrap confidence intervals, statistical significance tests | Blocked on training |
| 13 | **Ablation Studies** -- Remove graph / remove investors / topology-only / no temporal / co-investment edges | Blocked on training |
| 14 | **Error Analysis** -- Cold-start companies, sector breakdown, geography, funding amount subgroups | Blocked on training |
| 15 | **Visualizations** -- UMAP embeddings, graph structure, attention heatmaps, ablation bar charts | Blocked on training |
| 16 | **Paper Writing** -- Full paper draft with all figures and tables | Blocked on steps 12-15 |

---

## File Tree (What Exists Right Now)

```
ML_Investor_Company_Raise/
├── CLAUDE.md                          # Project rules & conventions
├── PRD.md                             # Product requirements document
├── requirements.txt                   # Python dependencies
├── links.md                           # Dataset links & references
│
├── data/
│   ├── raw/crunchbase_2015/           # Original CSVs (immutable)
│   │   ├── investments.csv            #   168K investment records
│   │   ├── rounds.csv                 #   115K funding rounds
│   │   ├── companies.csv              #   66K companies
│   │   └── acquisitions.csv           #   19K acquisitions
│   │
│   ├── interim/                       # Intermediate outputs
│   │   ├── labels.csv                 #   23,740 labeled companies
│   │   └── splits.csv                 #   Same + train/val/test split column
│   │
│   └── processed/                     # Graph-ready data (~65MB)
│       ├── edges_{train,val,test}.csv  #   Edge lists per split
│       ├── company_index.csv           #   Permalink -> integer ID (66,369)
│       ├── investor_index.csv          #   Permalink -> integer ID (30,732)
│       ├── company_features_*.npy      #   (66369, 58) per split
│       ├── investor_features_*.npy     #   (30732, 7) per split
│       └── company_texts.csv           #   Category text for sentence embeddings
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Data profiling & quality checks
│   └── colab_training.ipynb           # Steps 6-11 (READY TO RUN)
│
├── src/
│   ├── data/
│   │   ├── label_generator.py         # Binary label creation
│   │   ├── temporal_split.py          # Train/val/test split logic
│   │   └── graph_builder.py           # Edge lists & entity indices
│   ├── features/
│   │   └── node_features.py           # Company (58d) + investor (7d) features
│   ├── models/
│   │   └── baseline_xgboost.py        # XGBoost utilities
│   ├── evaluation/
│   │   └── metrics.py                 # F1, AUC, comparison table
│   └── visualization/                 # (empty -- Step 15)
│
├── tests/                             # (stubs -- to be written)
├── docs/
│   ├── SCRAPER_PLAN.md                # Future: modern data collection plan
│   ├── paper/                         # (empty -- Step 16)
│   └── figures/                       # (empty -- Step 15)
```

---

## Key Design Decisions Made

1. **24-month prediction horizon** -- Balances label density vs. practical relevance
2. **Bipartite graph** (Investor <-> Company) -- Co-investment edges only as ablation
3. **Temporal split by funded_at** -- NOT random split; prevents data leakage
4. **3 static graph snapshots** -- One per split cutoff (not per-node dynamic graphs)
5. **Local data prep, Colab training** -- No PyTorch/PyG installed locally; GPU on Colab
6. **Seed 42 everywhere** -- Reproducibility

---

## Separate Project: VC Data Scraper

A separate project at `/Volumes/ThunderDB/vc-data-scraper/` was initialized for building a modern (2020-2026) dataset from public sources (SEC EDGAR, news APIs, accelerator lists). It has a plan (`docs/SCRAPER_PLAN.md`) and directory structure but no implementation yet. This is deprioritized until the paper is done.

---

## What To Do Next

**Immediate:** Run the Colab notebook to get training results.

After that, the remaining work (Steps 12-16) is analysis and writing -- no new model code needed.
