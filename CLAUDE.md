# Temporal Graph Learning for VC Outcome Prediction

## Project Summary

Academic research project building a temporal GNN pipeline to predict startup follow-on funding using venture capital network structure. Primary output is a conference-ready paper; secondary goal is product feasibility validation.

## Tech Stack

- **Language:** Python 3.11+
- **Graph ML:** PyTorch Geometric (PyG), NetworkX
- **ML:** scikit-learn, XGBoost
- **Embeddings:** Node2Vec (via PyG or node2vec package), sentence-transformers (HuggingFace)
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn, UMAP, t-SNE (sklearn)
- **Notebooks:** Jupyter (exploration), scripts (production pipeline)
- **Environment:** conda or venv with requirements.txt

## Project Structure

```
├── data/
│   ├── raw/            # Original downloaded datasets (never modify)
│   ├── interim/        # Intermediate transformations
│   └── processed/      # Final graph-ready data (PyG Data objects)
├── notebooks/          # Exploration & analysis (numbered: 01_, 02_, ...)
├── src/
│   ├── data/           # Data loading, cleaning, graph construction
│   ├── features/       # Feature engineering, embeddings
│   ├── models/         # GNN models, baselines
│   ├── evaluation/     # Metrics, ablations, error analysis
│   └── visualization/  # Plotting utilities
├── tests/              # Unit tests for data pipeline & temporal splits
├── docs/
│   ├── paper/          # LaTeX or markdown paper drafts
│   └── figures/        # Generated figures for paper
├── PRD.md              # Project requirements document
└── CLAUDE.md           # This file
```

## Critical Rules

### Temporal Integrity (HIGHEST PRIORITY)
- NEVER construct graph edges using information from after the prediction timestamp
- ALWAYS verify temporal splits before any model training
- Test for leakage explicitly in the pipeline
- When in doubt, exclude the edge — false negatives are better than leakage

### Data Handling
- Raw data in `data/raw/` is IMMUTABLE — never overwrite source files
- All transformations must be reproducible from raw → processed
- Document any data cleaning decisions in notebook markdown cells

### Code Style
- Type hints on all function signatures
- Docstrings only when behavior is non-obvious
- Notebooks are for exploration; anything reusable goes into `src/`
- Use `pathlib.Path` for all file paths
- Seed all random operations (default seed: 42)

### Modeling
- Always train XGBoost baseline FIRST before any GNN
- Report F1 and ROC-AUC as primary metrics (accuracy is misleading with imbalance)
- All models must be evaluated on the same temporal test split
- Log hyperparameters and results for reproducibility

### Academic Rigor
- Every claim in the paper must have supporting experiment
- Ablations are not optional — they're core methodology
- Negative results (GNN ≈ baseline) are still publishable if well-analyzed
- Cite prior work properly; don't overclaim novelty

## Common Commands

```bash
# Environment setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run notebooks
jupyter lab notebooks/

# Run tests
pytest tests/ -v

# Train baseline
python src/models/train_baseline.py

# Train GNN
python src/models/train_gnn.py --model graphsage --epochs 200
```

## Key Design Decisions

1. **Bipartite graph** (Investor ↔ Startup) — not homogeneous projection
2. **Node2Vec on bipartite** — handles heterogeneous walks
3. **Temporal split by round_date** — not random split
4. **24-month prediction horizon** — balances label density vs. practical relevance
5. **Sentence embeddings** for descriptions — lightweight LLM component (no fine-tuning)
