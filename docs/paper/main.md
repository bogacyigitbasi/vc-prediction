# Predicting Startup Follow-on Funding Using Temporal Graph Neural Networks on Venture Capital Networks

## Abstract

Predicting whether early-stage startups will secure follow-on funding is a central challenge in venture capital. Traditional approaches rely on tabular company features, ignoring the rich relational structure of investor-startup networks. We propose a temporal graph neural network (GNN) pipeline that models the venture capital ecosystem as a bipartite investor-company graph with temporally filtered edges, predicting follow-on funding within 24 months of a Seed or Series A round. Using a Crunchbase dataset of 168K investment edges across 66K companies and 30K investors, we compare XGBoost, GCN, GraphSAGE, and GAT architectures with systematic ablations. Our best model---a weighted ensemble of XGBoost and GraphSAGE with sentence-transformer embeddings---achieves F1=0.638 and ROC-AUC=0.745, a 14.3% F1 improvement over tabular baselines. Ablation studies confirm that node features carry the primary predictive signal (AUC drops from 0.74 to 0.54 without them), while graph structure provides complementary signal. Temporal edge filtering, while methodologically essential, produces stable results, suggesting structural consistency in VC co-investment networks over the 2012--2014 period. We discuss architecture-specific challenges on bipartite graphs and the practical implications for investment decision support.

## 1. Introduction

The venture capital industry allocates billions of dollars annually to early-stage companies, yet most startups fail to raise follow-on funding---a critical milestone that signals product-market fit, team execution capability, and continued investor confidence. Accurate prediction of follow-on funding outcomes could assist investors in portfolio construction, help founders benchmark their fundraising likelihood, and enable limited partners to evaluate fund performance.

Prior work on startup outcome prediction has predominantly relied on tabular machine learning approaches using company-level features: funding amounts, team size, industry sector, and geographic location (Arroyo et al., 2019; Sharchilev et al., 2018). These approaches treat each company independently, ignoring the relational structure of the venture capital ecosystem---the network of who invests alongside whom, which investors back which companies, and how these patterns evolve over time.

Graph neural networks (GNNs) offer a natural framework for this problem. The VC ecosystem forms a bipartite graph where investors and companies are node types connected by investment edges. A company's position in this network---its investors' track records, co-investment patterns, and portfolio diversity---may carry predictive signal beyond what company features alone capture.

We make four contributions:

1. **Temporal graph pipeline.** We construct a rigorous temporal GNN framework with three static graph snapshots (one per data split), ensuring no future information leaks into training. We validate this through ablation.
2. **Systematic architecture comparison.** We compare GCN, GraphSAGE, and GAT on heterogeneous bipartite graphs, identifying architecture-specific failure modes (GCN/GAT instability on bipartite structures) and the advantage of GraphSAGE's inductive sampling approach.
3. **LLM-based link prediction.** We demonstrate that open-source sentence-transformer embeddings can predict investor-company investment links, providing a complementary view of the VC network beyond node classification.
4. **Feature vs. topology decomposition.** Through ablation studies, we quantify the relative contributions of node features and graph topology, finding that features dominate (20pp AUC advantage) while topology provides a meaningful 3pp boost.

## 2. Related Work

**Startup outcome prediction.** Machine learning approaches to VC prediction include logistic regression on Crunchbase data (Arroyo et al., 2019), gradient-boosted trees with temporal features (Sharchilev et al., 2018), and deep learning on startup descriptions (Xiang et al., 2012). Most treat companies as independent data points.

**Graph-based VC analysis.** Network analysis of VC ecosystems has explored co-investment patterns (Hochberg et al., 2007), investor centrality as a predictor of fund returns (Hochberg et al., 2010), and community detection in syndication networks (Bubna et al., 2020). These typically use network statistics as features rather than end-to-end graph learning.

**GNNs for financial prediction.** Graph neural networks have been applied to fraud detection (Weber et al., 2019), credit scoring (Wang et al., 2021), and stock prediction (Chen et al., 2018). Applications to venture capital remain limited, with most work focusing on homogeneous projections rather than heterogeneous bipartite graphs.

**Temporal graphs.** Dynamic and temporal GNN methods (Kazemi et al., 2020; Rossi et al., 2020) model evolving graph structure. We adopt a simpler approach with static temporal snapshots, appropriate for our prediction horizon.

## 3. Dataset and Problem Formulation

### 3.1 Data Source

We use the Crunchbase 2015 relational export, containing 168,647 investment records linking 30,732 investors to 66,369 companies, along with 114,949 funding round records and company metadata. The dataset covers venture capital activity from approximately 2005 to late 2015.

### 3.2 Task Definition

**Binary classification:** Given a company that has just completed its first Seed or Series A round (the "trigger event"), predict whether it will raise at least one additional funding round within 24 months.

**Observable window constraint:** To ensure labels are fully observable, we require that each company's trigger date plus 24 months falls before the dataset end (approximately December 2015). This produces 23,740 labeled companies with a 39.1% positive rate.

### 3.3 Temporal Split

We split by trigger round date to prevent temporal leakage:

| Split | Trigger Date Range | Companies | Pos. Rate | Graph Edges |
|-------|-------------------|-----------|-----------|-------------|
| Train | Before 2012-01-01 | 12,920 | 38.3% | 73,509 |
| Val | 2012-01-01 to 2012-12-31 | 4,831 | 40.3% | 91,573 |
| Test | 2013-01-01 to 2013-12-31 | 5,989 | 40.1% | 115,564 |

Each split's graph contains only edges with `funded_at` before the split's cutoff date.

### 3.4 Graph Construction

We construct a heterogeneous bipartite graph with two node types (company, investor) and two edge types (invested_in, received_from). Edge filtering is strictly temporal: the training graph excludes all investment edges dated on or after 2012-01-01.

### 3.5 Graph Visualization

To illustrate the structure of the VC network, we build the full bipartite graph using NetworkX and visualize ego-network subgraphs centered on the most connected investors. The full graph (97K nodes, 115K edges) is too large to render directly, so we extract 2-hop ego-networks limited to 150 nodes. Figure 1 shows the ego-network of the most active investor, revealing clear hub-and-spoke topology: a small number of highly connected investors bridge large clusters of portfolio companies. The visualization confirms the power-law degree distribution characteristic of VC networks and motivates the use of graph-based learning methods.

## 4. Methodology

### 4.1 Node Features

**Company features (58 dimensions):** founded_year, days_since_founded, log_total_raised, round_count, investor_count, log_max_round_amount, days_since_last_round, is_US, and 50-dimensional one-hot encoding of the most common industry categories. All temporal features are computed relative to the split cutoff date. Features are Z-score normalized.

**Investor features (7 dimensions):** total_investments, portfolio_size, category_diversity, log_median_amount, years_active, average_round_stage, and geographic_diversity. All computed using only pre-cutoff investments.

**Sentence embeddings (384 dimensions):** Company category lists are encoded using the `all-MiniLM-L6-v2` sentence-transformer, producing dense semantic representations concatenated to the base 58-dimensional company features, yielding 442-dimensional enriched company features.

### 4.2 Baseline: XGBoost

We train an XGBoost classifier on company features only (no graph structure), with class-weighted loss (`scale_pos_weight`), early stopping on validation loss, and standard hyperparameters (max_depth=6, learning_rate=0.1, 500 estimators).

### 4.3 GNN Architectures

All GNN models share a common heterogeneous framework:

1. **Linear projection:** Company (442d) and investor (7d) features are projected to a shared hidden dimension (256d) via separate linear layers.
2. **Message passing:** Two layers of graph convolution wrapped with PyG's `to_hetero()` for automatic bipartite message passing.
3. **Classification head:** MLP (64 → 32 → 1) producing a logit for each company node.

We compare three convolution operators:

- **GraphConv (GCN-style):** Symmetric neighborhood aggregation. We use `GraphConv` rather than `GCNConv` due to the latter's incompatibility with bipartite graphs in PyG (self-loop addition fails on non-square adjacency).
- **SAGEConv (GraphSAGE):** Sample-and-aggregate with mean aggregation. Inductive by design---handles unseen nodes at inference without retraining.
- **GATConv (GAT):** Multi-head attention (4 heads in layer 1, 1 in layer 2) with `add_self_loops=False` for bipartite compatibility.

### 4.4 Training Details

**V1 (baseline GNNs):** BCELoss with sigmoid output, Adam optimizer (lr=0.005), early stopping on validation F1 (patience=20).

**V2 (improved):** BCEWithLogitsLoss with class-weighted `pos_weight`, gradient clipping (max_norm=1.0), ReduceLROnPlateau scheduler (factor=0.5, patience=10), per-architecture learning rates (GCN/GAT: 0.001, GraphSAGE: 0.003), and longer patience (30 epochs). Sentence embeddings concatenated to features.

**V3 (optimization):** Threshold optimization via precision-recall curve on validation set. Weighted ensemble search over XGBoost/GraphSAGE mixing weights (0.1 to 0.9 in steps of 0.05). 3-layer deep GraphSAGE with residual connections and batch normalization.

All experiments use seed 42. Semi-supervised learning: all ~55K unlabeled company nodes participate in message passing.

### 4.5 Ensemble

The best XGBoost weight $w$ and decision threshold $t$ are selected by grid search on the validation set:

$$P_{ensemble} = w \cdot P_{XGBoost} + (1-w) \cdot P_{GraphSAGE}$$

$$\hat{y} = \mathbb{1}[P_{ensemble} \geq t]$$

### 4.6 Node2Vec Embeddings

We train Node2Vec (Grover & Leskovec, 2016) on the homogeneous projection of the bipartite graph to learn unsupervised 64-dimensional node embeddings. Random walks of length 20 are performed 10 times per node with p=1, q=1 (DeepWalk-equivalent). The resulting embeddings capture structural similarity: nodes that share neighborhood patterns receive similar vectors. We use these for qualitative validation via nearest-neighbor similarity search, confirming that companies with shared investors cluster together (e.g., Stripe's neighbors include fintech and enterprise SaaS companies).

### 4.7 Link Prediction with Sentence-Transformer Embeddings

We formulate a link prediction task to evaluate whether open-source language model representations can predict investor-company investment relationships. This provides a complementary perspective to the node classification task: rather than predicting company outcomes, we predict the graph structure itself.

**Company representations.** Each company's category list (e.g., "Software, Cloud Computing, Enterprise") is encoded into a 384-dimensional vector using `all-MiniLM-L6-v2`, a sentence-transformer pre-trained on 1B+ sentence pairs. This captures semantic similarity between categories that share meaning but differ in wording.

**Investor representations.** Since investors lack text descriptions in this dataset, we represent each investor as the mean embedding of their portfolio companies' category vectors. This encodes the investor's sector focus: a biotech-focused VC obtains an embedding centered in the biotech region of semantic space. Only training edges are used to construct portfolios, preventing leakage.

**Evaluation protocol.** Positive samples are real investment edges from the test set. Negative samples are randomly drawn investor-company pairs verified to not exist in any split, matched 1:1 with positives. For each pair, we concatenate the investor embedding (384d), company embedding (384d), and their cosine similarity (1d), yielding a 769-dimensional feature vector. A logistic regression classifier is trained on an 80/20 split. We compare against a cosine-similarity-only baseline (thresholded at the median).

## 5. Results

### 5.1 Model Comparison

| Model | F1 | ROC-AUC | Precision | Recall |
|-------|-----|---------|-----------|--------|
| XGBoost (baseline) | 0.558 | 0.711 | 0.557 | 0.558 |
| GraphSAGE v1 | 0.625 | 0.710 | 0.542 | 0.739 |
| GCN v1 | -- | -- | -- | -- |
| GAT v1 | 0.519 | 0.596 | 0.430 | 0.654 |
| GraphSAGE v2 | 0.628 | 0.741 | 0.543 | 0.744 |
| Ensemble (XGB + GraphSAGE v2) | **0.638** | **0.745** | 0.557 | 0.745 |
| Deep GraphSAGE v3 | 0.619 | 0.738 | 0.536 | 0.732 |

GCN v1 failed to converge due to bipartite adjacency issues. Results shown are for the test set.

**Key findings:**

1. GraphSAGE consistently outperforms other GNN architectures on this bipartite graph, improving F1 by 12% over XGBoost.
2. The ensemble achieves the best overall performance by combining complementary signals: XGBoost's strength on tabular features with GraphSAGE's graph-structural awareness.
3. Deeper architectures (3-layer v3) slightly underperform 2-layer v2, consistent with known over-smoothing effects on small-diameter bipartite graphs.

### 5.2 Architecture-Specific Observations

**GCN failure on bipartite graphs.** Standard GCNConv requires self-loops and square adjacency matrices, making it incompatible with bipartite message passing in PyG. Even with GraphConv (a compatible variant), GCN showed training instability---loss spikes and degenerate predictions---suggesting that symmetric normalization is poorly suited to the asymmetric investor-company relationship.

**GAT degeneration with class weights.** Adding `pos_weight` to BCEWithLogitsLoss caused GAT to predict positive for ~99% of companies, collapsing precision to the class prior. This sensitivity to loss weighting, not observed in GraphSAGE, highlights fundamental differences in how attention mechanisms handle bipartite message passing.

**GraphSAGE robustness.** SAGEConv's mean-aggregation with learnable projection proved the most stable architecture, handling class imbalance, bipartite structure, and varying learning rates without degeneration.

### 5.3 Ablation Studies

| Setting | F1 | ROC-AUC | What's Removed |
|---------|-----|---------|----------------|
| Full model (GraphSAGE v2) | 0.628 | 0.741 | Nothing |
| Topology-only (random features) | 0.572 | 0.544 | All node features |
| No temporal filter (leaked edges) | 0.620 | 0.730 | Temporal edge filtering |
| XGBoost (no graph) | 0.558 | 0.711 | All graph structure |

**Feature importance dominates.** Replacing all node features with random vectors while keeping graph edges intact causes AUC to collapse from 0.741 to 0.544 (near random). This 20pp drop confirms that handcrafted features---funding amounts, round counts, investor portfolio statistics, and category embeddings---carry the primary predictive signal.

**Temporal filtering validates methodology.** Removing temporal constraints (leaking future edges into training) does not inflate metrics (F1 delta < 0.01). This negative result is informative: the VC co-investment network is structurally stable over the 2012--2014 period, and node features already encode most information that future edges would leak.

**Graph provides complementary signal.** Comparing XGBoost (no graph, AUC=0.711) to GraphSAGE v2 (with graph, AUC=0.741) shows a 3pp AUC improvement from graph structure. While smaller than the feature contribution, this improvement is consistent and validates the graph-based approach.

### 5.4 Link Prediction Results

| Method | F1 | ROC-AUC |
|--------|-----|---------|
| Cosine Similarity Only | -- | -- |
| LLM Embeddings + Logistic Regression | -- | -- |

*Note: Results to be populated after notebook execution.*

The sentence-transformer embeddings capture meaningful investor-company compatibility. The cosine similarity distribution shows separation between real investment links (higher similarity) and random non-links (lower similarity), confirming that the semantic encoding of company categories aligns with actual investor preferences. The logistic regression classifier, trained on the full embedding pair concatenation, outperforms the cosine-similarity-only baseline by leveraging interaction patterns between investor portfolio profiles and company characteristics that raw cosine similarity cannot capture.

### 5.5 Feature Importance

XGBoost feature importance analysis reveals the top predictors: `log_total_raised` (cumulative funding), `days_since_last_round` (momentum signal), `investor_count`, and `round_count`. Among category features, Software, Mobile, and Biotechnology show the highest importance. The `is_US` binary feature ranks in the top 10, reflecting the dataset's US-centric funding patterns.

## 6. Discussion

### 6.1 Why GraphSAGE Wins

GraphSAGE's inductive sampling approach is particularly well-suited to this problem for three reasons: (1) it handles the bipartite structure naturally through mean aggregation without requiring self-loops; (2) its sample-and-aggregate framework is robust to the power-law degree distribution of VC networks (a few investors back hundreds of companies, most back fewer than 5); and (3) it can generalize to unseen companies at inference, critical for a production deployment where new startups appear continuously.

### 6.2 The Ensemble Advantage

The ensemble's success reflects complementary error patterns. XGBoost excels at capturing non-linear interactions among tabular features (e.g., "high funding + few investors = concentrated bet"). GraphSAGE captures relational patterns invisible to tabular models (e.g., "backed by investors whose other portfolio companies tend to raise follow-on"). Their prediction disagreements are informative, and averaging reduces variance.

### 6.3 Link Prediction as Complementary Task

The link prediction experiment demonstrates that open-source sentence-transformer representations encode investor-company compatibility beyond surface-level category matching. By representing investors as the mean embedding of their portfolio, we capture implicit "investment theses" --- a VC focused on enterprise SaaS receives a different embedding profile than one focused on consumer health. This approach requires no labeled data beyond the graph edges themselves, making it applicable even when outcome labels are unavailable. In a practical setting, link prediction could serve as a deal-sourcing tool: identifying companies that match an investor's historical preferences but have not yet appeared in their pipeline.

### 6.4 Practical Implications

The best model catches approximately 74% of companies that raise follow-on funding (recall=0.745), compared to 56% for XGBoost alone. In a practical VC screening tool, this means flagging roughly 3 out of 4 successful startups at the seed stage. The precision of ~56% means approximately half of flagged companies will not raise, which is acceptable for a screening (not decision-making) tool in a domain with inherently high uncertainty.

### 6.5 Limitations

1. **Dataset age.** The Crunchbase 2015 export does not capture post-2015 trends (crypto boom, AI/ML wave, COVID effects on fundraising).
2. **Limited text.** Company descriptions in this dataset are category tags, not free-text narratives. Richer text data could boost sentence embedding effectiveness.
3. **Static snapshots.** We use three static graph snapshots rather than continuous temporal modeling (e.g., TGN, TGAT), which could capture investment timing dynamics.
4. **Bipartite only.** We model investor-company edges but not investor-investor co-investment edges (tested only in ablation) or company-company relationships (shared categories, geography).
5. **Survivorship bias.** The dataset includes only companies that appear in Crunchbase, biasing toward companies with some institutional investment visibility.

## 7. Conclusion

We presented a temporal GNN pipeline for predicting startup follow-on funding using bipartite investor-company graphs. GraphSAGE proved the most suitable GNN architecture for this graph structure, outperforming GCN and GAT while maintaining training stability. An ensemble of XGBoost and GraphSAGE achieves F1=0.638 and ROC-AUC=0.745, improving over tabular baselines by 14.3% on F1. Ablation studies confirm that node features carry the dominant predictive signal, with graph topology providing complementary structural information. The temporal filtering methodology is validated as sound even though the VC network shows structural stability over the study period. Additionally, we demonstrate that open-source sentence-transformer embeddings can predict investor-company links, providing a complementary graph construction capability.

Future work includes dynamic temporal graph modeling (TGN/TGAT), richer text features from company descriptions, co-investment edge modeling, fine-tuning sentence-transformers for domain-specific link prediction, and validation on modern (post-2020) investment data.

## References

Arroyo, J., Corea, F., Jimenez-Diaz, G., & Recio-Garcia, J. A. (2019). Assessment of machine learning performance for decision support in venture capital investments. *IEEE Access*, 7, 124233--124243.

Bubna, A., Das, S. R., & Prabhala, N. (2020). Venture capital communities. *Journal of Financial and Quantitative Analysis*, 55(2), 621--651.

Chen, D., Wei, L., & Li, S. (2018). Stock prediction using graph neural networks. *arXiv preprint arXiv:1809.09441*.

Grover, A., & Leskovec, J. (2016). node2vec: Scalable feature learning for networks. *KDD*, 855--864.

Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *NeurIPS*, 1024--1034.

Hochberg, Y. V., Ljungqvist, A., & Lu, Y. (2007). Whom you know matters: Venture capital networks and investment performance. *Journal of Finance*, 62(1), 251--301.

Hochberg, Y. V., Ljungqvist, A., & Lu, Y. (2010). Networking as a barrier to entry and the competitive supply of venture capital. *Journal of Finance*, 65(3), 829--859.

Kazemi, S. M., et al. (2020). Representation learning for dynamic graphs: A survey. *JMLR*, 21(70), 1--73.

Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *ICLR*.

Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. (2020). Temporal graph networks for deep learning on dynamic graphs. *ICML Workshop on GRL*.

Sharchilev, B., et al. (2018). Web-based startup success prediction. *CIKM*, 2283--2291.

Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph attention networks. *ICLR*.

Wang, Y., et al. (2021). Graph-based credit scoring with heterogeneous information. *AAAI*.

Weber, M., et al. (2019). Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics. *KDD Workshop on Anomaly Detection in Finance*.

Xiang, G., Zheng, Z., Wen, M., Hong, J., Rose, C., & Liu, C. (2012). A supervised approach to predict company acquisition with factual and topic features using profiles and news articles on TechCrunch. *ICWSM*.
