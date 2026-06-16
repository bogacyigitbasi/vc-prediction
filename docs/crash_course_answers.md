# Crash Course: Answers to All Questions

---

## 1. Why is tabular work bad for our problem?

It's not that tabular methods are bad in general — XGBoost is excellent at what it does. The problem is what it **cannot see**.

Tabular = each company is one row in a spreadsheet. XGBoost looks at that row and makes a prediction. It sees: "this company raised $500K, has 2 investors, is in fintech, is US-based." That's it.

What it misses:

- **Who** those 2 investors are. Is one of them Sequoia with a 70% success rate, or a random angel who's never invested before? XGBoost doesn't know.
- **What else** those investors have funded. If Sequoia's other 14 portfolio companies all raised follow-on, that's a strong signal. XGBoost can't see this.
- **The network pattern.** Maybe this company shares investors with 5 other companies that all succeeded. That cluster effect is invisible to a spreadsheet.

Think of it this way: if you're evaluating a job candidate, tabular methods only look at their resume (GPA, years of experience, skills listed). Graph methods also look at who recommended them, where those recommenders work, and how successful their other recommendations have been. Both matter, but the network context tells you things the resume can't.

The gap isn't that tabular is "stupid" — it's that tabular is **blind to relationships**. Our hypothesis is that those relationships carry predictive signal, and the results prove it (+3pp AUC from graph structure).

---

## 2. Why XGBoost? How does it learn anything if it's just decision trees?

**XGBoost does learn.** Here's how:

A single decision tree is indeed simple — it just memorizes rules like "if raised > $1M and investors > 3, then yes." You're right that one tree doesn't generalize well.

But XGBoost doesn't use one tree. It uses **500 trees that correct each other's mistakes**. This is called **gradient boosting**. Here's the exact process:

1. **Tree #1** makes predictions on all 12,920 training companies. It gets some right, some wrong.
2. For each company, compute the **error** (how far off the prediction was).
3. **Tree #2** is trained NOT on the original labels, but on the **errors of Tree #1**. It literally learns "where did Tree #1 go wrong?"
4. **Tree #3** is trained on the errors of Tree #1 + Tree #2 combined.
5. After 500 trees, the final prediction is the **sum** of all 500 trees' outputs.

**Why this works on test data (generalization):**

Each tree is deliberately kept small (max_depth=6, so at most 6 levels of if/else). A shallow tree can't memorize the training data — it can only learn broad patterns like "companies with more funding tend to raise again." The boosting process finds **combinations** of these broad patterns that are predictive. This is fundamentally different from a single deep tree that memorizes everything.

It's like asking 500 different people for advice, where each person is specifically trained to address the mistakes of the previous advisors. No single person is an expert, but together they're very accurate.

**Early stopping** prevents overfitting: we monitor performance on the validation set, and stop adding trees when validation performance stops improving. If we kept going to 10,000 trees, it would start memorizing training data. We stop at the sweet spot.

**Why it's our baseline:** XGBoost is the best algorithm for "rows and columns" data. If a GNN can't beat XGBoost, the graph structure isn't worth the complexity. XGBoost is the bar to clear.

---

## 3. Feature Engineering — What are the features, what's the LLM doing, and what's it all for?

**The big picture:** Every node in our graph needs a numerical description (a vector of numbers). The GNN takes these numbers and passes them between connected nodes. Without features, the GNN has nothing to pass.

### Company features (58 dimensions)

These describe what we know about each company before the prediction date:

| Feature | What it tells you | Example |
|---------|------------------|---------|
| founded_year | How old the company is | 2010 |
| days_since_founded | Age in days (more precise) | 730 |
| log_total_raised | How much money they've raised so far | log(500000) = 13.1 |
| round_count | How many funding rounds they've had | 1 |
| investor_count | How many investors have funded them | 2 |
| log_max_round_amount | Largest single round | log(500000) = 13.1 |
| days_since_last_round | Momentum — are they actively raising? | 90 |
| is_US | Geographic signal | 1 |
| 50 category columns | One-hot encoding of industry | [0,0,1,0,...,0] for "fintech" |

All of these are computed using **only data before the cutoff date** (no leakage).

### Investor features (7 dimensions)

These describe the investor's track record:

| Feature | What it tells you |
|---------|------------------|
| total_investments | How active they are |
| portfolio_size | Number of unique companies backed |
| category_diversity | Do they invest in one sector or many? |
| log_median_amount | Typical check size |
| years_active | How long they've been investing |
| avg_round_stage | Do they do seed, or later stages? |
| geo_diversity | Invest in one country or globally? |

### The LLM's role: text embeddings (+384 dimensions)

The 58 company features are all numbers. But companies also have **text descriptions** — their category list like "Machine Learning, AI, Analytics."

The problem: "AI, Machine Learning" and "Artificial Intelligence, Deep Learning" mean almost the same thing, but in a one-hot encoding they'd look completely different (different words = different columns).

**What the LLM does:**

We feed the category text into `all-MiniLM-L6-v2` (a small, pre-trained language model from HuggingFace). It outputs a 384-dimensional vector where **similar meanings produce similar vectors**.

```
"Machine Learning, AI"          -> [0.23, -0.41, 0.67, ...]
"Artificial Intelligence, NLP"  -> [0.21, -0.39, 0.65, ...]  <- very close!
"Biotech, Healthcare"           -> [-0.55, 0.82, 0.11, ...]  <- very different
```

We **concatenate** this 384-dim vector to the 58 numeric features: 58 + 384 = **442 dimensions per company node**.

We do NOT fine-tune the LLM. We just use it as a fixed encoder — feed text in, get vector out.

### What it's all for

These feature vectors are the **input** to every model:

- **XGBoost** gets the 442 company features as a flat row. It ignores investor features (it can't use them without the graph).
- **GNNs** get company features (442d) on company nodes AND investor features (7d) on investor nodes. During message passing, these features flow between connected nodes. That's how a company "learns" about its investors' track records.
- **Node2Vec** ignores features entirely — it only uses graph structure (the edges).
- **Link prediction** uses a separate set of 384d sentence-transformer embeddings to represent both investors and companies for predicting investment links.

---

## 4. Metrics — What are they, what do they represent?

Imagine we predict on 100 test companies. 40 actually raised (positive), 60 didn't (negative).

Our model predicts "yes" for 50 companies. Of those 50:
- 30 actually raised (True Positives, TP)
- 20 didn't raise (False Positives, FP — false alarms)

Of the 50 the model said "no" to:
- 10 actually raised (False Negatives, FN — we missed them)
- 40 didn't raise (True Negatives, TN — correct rejections)

### Precision = 30 / (30 + 20) = 0.60

"When the model says YES, it's right 60% of the time."

This matters if false alarms are expensive. If you're an investor and the model says "invest in this company," you want to know how often that advice is correct.

### Recall = 30 / (30 + 10) = 0.75

"Of all 40 companies that actually raised, the model caught 30 of them (75%)."

This matters if you don't want to miss opportunities. Missing a company that would have raised is a missed deal.

### F1 = 2 * (0.60 * 0.75) / (0.60 + 0.75) = 0.667

"The balance between precision and recall."

You can always cheat on one metric at the expense of the other:
- Say "yes" to everyone → recall = 100% (you catch all positives!) but precision = 40% (most predictions are wrong)
- Say "yes" to only the most obvious case → precision = 95% but recall = 5% (you miss almost everyone)

F1 punishes both extremes. A high F1 means you're good at **both** catching positives and avoiding false alarms.

### ROC-AUC

**ROC** = Receiver Operating Characteristic (historical name from WWII radar — ignore it).
**AUC** = Area Under the Curve.

This one is different — it doesn't depend on a yes/no threshold.

Every model outputs a **probability**, not yes/no. For example:

```
Company A: P = 0.92
Company B: P = 0.71
Company C: P = 0.48
Company D: P = 0.33
Company E: P = 0.15
```

To make a decision, you pick a **threshold**. If threshold = 0.5: A and B are YES, C/D/E are NO. If threshold = 0.3: A, B, C, D are YES, only E is NO. Different threshold = different precision/recall tradeoff.

**The ROC curve** tries every possible threshold from 0.0 to 1.0. At each threshold, it computes:
- **True Positive Rate** (recall): of all companies that raised, what % did we catch?
- **False Positive Rate**: of all companies that didn't raise, what % did we wrongly say "yes" to?

Plot these as a curve:

```
True Positive Rate (Recall)
  1.0 |        .-------*  <- perfect model (area = 1.0)
      |       /
      |      /
      |     /    <- our model's curve
  0.5 |    /
      |   /
      |  /   <- random guesser: diagonal (area = 0.5)
      | /
  0.0 |/_________________________
      0.0        0.5        1.0
         False Positive Rate
```

**AUC = the area under this curve.** Random guesser = 0.5 (diagonal). Perfect model = 1.0.

**Simple interpretation:** Pick one random company that raised (Stripe, P=0.78) and one that didn't (FailedStartup, P=0.31). Did the model rank Stripe higher? Yes. AUC = 0.745 means if you repeat this random pair experiment thousands of times, **74.5% of the time the model ranks the successful company higher.**

**Why AUC is useful alongside F1:** F1 depends on the threshold you pick. AUC doesn't depend on any threshold — it measures overall separation ability regardless of where you draw the line.

### Why not Accuracy?

Accuracy = (30 + 40) / 100 = 70%. Sounds good!

But a model that says "NO" to every single company gets accuracy = 60/100 = 60%. It never even looked at the data and got 60%! With accuracy, the bar is so low that a useless model looks decent.

F1 for the "always say no" model = 0.00 (zero recall → zero F1). That's the truth.

---

## 5. How do we compute metrics on XGBoost? Same as GNNs?

**Yes, exactly the same.** Every model — XGBoost, GCN, GraphSAGE, GAT, ensemble — is evaluated with the same metrics (F1, AUC, precision, recall) on the same test set (5,989 companies from 2013).

The process for any model:

1. Model takes features as input, outputs a **probability** P(follow-on) for each test company.
2. We compute **AUC** directly from these probabilities (no threshold needed).
3. We pick a threshold (default 0.5, or optimized on validation set), convert probabilities to yes/no.
4. We compute **precision, recall, F1** from the yes/no predictions compared to actual labels.

The only difference is what inputs the model gets:
- XGBoost: 442 company features per company (flat row)
- GNNs: 442 company features + 7 investor features + graph edges (message passing)
- Ensemble: takes the probability outputs of both XGBoost and GraphSAGE and averages them

But once each model outputs a probability, the evaluation is identical. This is what makes the comparison fair — same test set, same metrics, same evaluation code.

---

## 6. "Weighted equally by their degree" — what does degree mean in GCN?

**Degree** = the number of edges connected to a node. That's it.

In our graph:
- Sequoia has invested in 200 companies → degree = 200
- A random angel has invested in 2 companies → degree = 2
- Stripe has received investment from 5 investors → degree = 5

**What GCN does with degree:**

The GCN formula normalizes by degree: `D^(-1/2) * A * D^(-1/2) * H * W`

In practice, this means: when Company X averages its 3 investors, it doesn't just do a straight average. It applies a weight of `1 / sqrt(degree_X * degree_investor)` to each edge.

Let's say Company X has degree 3 (3 investors):

- Sequoia (degree 200): weight = 1 / sqrt(3 * 200) = 1/24.5 = 0.041
- YC (degree 1000): weight = 1 / sqrt(3 * 1000) = 1/54.8 = 0.018
- angel (degree 2): weight = 1 / sqrt(3 * 2) = 1/2.45 = 0.408

So actually, the angel gets a **much higher weight** than Sequoia! That's because GCN normalizes by degree — highly connected nodes get their signal diluted. The idea is that Sequoia's signal is spread across 200 companies, so each individual company gets a smaller share. The angel's entire signal goes to just 2 companies.

**The problem:** This isn't really what we want. A signal from Sequoia (experienced, successful) should probably matter more than from a random angel, even if Sequoia's signal is "diluted" across many companies. GCN can't distinguish between "diluted signal from an expert" and "concentrated signal from a nobody."

This is one reason GraphSAGE works better — it just takes the mean and lets the learned W matrix figure out what's important.

**So "weighted equally by degree" on the slide is slightly simplified.** The more precise statement is: "GCN weights each neighbor by the inverse square root of both nodes' degrees." The key point is that the weighting is purely structural (based on how many connections), not based on importance or quality.

---

## 7. GraphSAGE Layer 2 — What exactly happens?

Let me walk through both layers in detail with a concrete example.

**Setup:** Company X has 3 investors: Sequoia, YC, angel.
Sequoia has invested in 4 companies: Company X, Airbnb, Stripe, DoorDash.

### Layer 1: Direct neighbor aggregation

Every node collects features from its immediate neighbors.

**For Company X:**
```
neighbor_mean = Mean(Sequoia_features, YC_features, angel_features)
new_X = ReLU(W1 * CONCAT(X_own_features, neighbor_mean))
```
Now Company X's representation contains information about its investors' features (portfolio size, years active, etc.).

**But simultaneously, the same thing happens for Sequoia:**
```
Sequoia's neighbors are its portfolio companies: X, Airbnb, Stripe, DoorDash
neighbor_mean = Mean(X_features, Airbnb_features, Stripe_features, DoorDash_features)
new_Sequoia = ReLU(W1 * CONCAT(Sequoia_own_features, neighbor_mean))
```
Now Sequoia's representation contains info about its portfolio companies.

**And for Airbnb, Stripe, DoorDash** — same thing. They each absorb info from their investors.

After Layer 1, every node has been updated once.

### Layer 2: This is where it gets powerful

Now we repeat the process, but using the **updated** representations from Layer 1.

**For Company X again:**
```
neighbor_mean = Mean(new_Sequoia, new_YC, new_angel)
final_X = ReLU(W2 * CONCAT(new_X, neighbor_mean))
```

But here's the key: `new_Sequoia` is no longer just Sequoia's original 7 features. After Layer 1, `new_Sequoia` contains information about Airbnb, Stripe, DoorDash (Sequoia's portfolio companies).

So when Company X absorbs `new_Sequoia` in Layer 2, **Company X now indirectly knows about Airbnb, Stripe, and DoorDash** — even though they're 2 hops away in the graph and not directly connected to X.

```
Layer 1: Company X ← [Sequoia, YC, angel]           (1-hop: direct investors)
Layer 2: Company X ← [Sequoia*, YC*, angel*]         (1-hop, but enriched)
                       ↑
                Sequoia* now contains info from:
                [Airbnb, Stripe, DoorDash, X]          (2-hop: portfolio peers)
```

**What Company X effectively "knows" after 2 layers:**
- Its own features (raised $500K, fintech, 2 investors)
- Sequoia's features (200 investments, 12 years active)
- The fact that Sequoia's other companies (Airbnb, Stripe, DoorDash) are well-funded, in tech, and most raised follow-on
- YC's features and portfolio characteristics
- The angel's features

This is the entire value of GNNs: **after 2 layers, every company knows about itself, its investors, and its investors' other portfolio companies.** XGBoost can never see this.

---

## 8. GAT — How are the attention weights computed? Based on what?

The attention weight between two nodes is computed from **their features**, not from any external label or metadata. Here's the exact process:

### Step 1: Project both nodes

Take Company X (442 features) and Sequoia (7 features). Both get projected to the same dimension (256d) through a learned linear layer:

```
h_X = W * X_features       → 256-dimensional vector
h_Seq = W * Seq_features   → 256-dimensional vector
```

### Step 2: Compute raw attention score

GAT has a small learned vector `a` (also 256-dimensional). The attention score is:

```
e(X, Seq) = LeakyReLU( a^T * CONCAT(h_X, h_Seq) )
```

In plain English: concatenate the two projected feature vectors, multiply by a learned weight vector, apply LeakyReLU. This gives a single number — the raw attention score.

The learned vector `a` is what makes GAT data-driven. During training, backpropagation adjusts `a` so that it assigns higher scores to neighbors that are actually useful for the prediction task. The idea: if having a high-portfolio-size investor is predictive of follow-on, `a` will learn to produce high scores when `h_Seq` has a high portfolio-size component.

### Step 3: Normalize with softmax

```
attention(X, Seq) = exp(e(X, Seq)) / [exp(e(X, Seq)) + exp(e(X, YC)) + exp(e(X, angel))]
```

This is just softmax — it makes the weights sum to 1.0 across all neighbors.

So if: e(X, Seq) = 2.1, e(X, YC) = 1.3, e(X, angel) = 0.4

Then: attention(X, Seq) = 0.6, attention(X, YC) = 0.3, attention(X, angel) = 0.1

### Step 4: Multi-head attention

We run 4 separate attention heads, each with its own `a` vector. Each head can learn to attend to different things:
- Head 1 might focus on investor portfolio size
- Head 2 might focus on category diversity
- Head 3 might focus on investment amount
- Head 4 might focus on years active

The outputs of all 4 heads are concatenated (or averaged in the final layer).

### Why it failed on our graph

The attention mechanism has to learn what makes a neighbor "important" from the training data. With ~12,920 training companies, ~24K labeled total, and the complexity of bipartite message passing, there may not be enough signal for 4 attention heads to learn meaningful weights. The model needs to learn: "when Company X has features like [this] and Investor Y has features like [that], the attention should be [this value]." That's a lot of parameters to learn from limited data.

GraphSAGE avoids this by just taking the mean (no learned attention) and letting the W matrix do the work. It's simpler, needs less data, and works better on our graph.

---

## 9. Training Improvements: V1 → V2 → V3

### V1: First attempt (baseline GNNs)

Setup:
- **BCELoss with sigmoid**: BCELoss = Binary Cross-Entropy Loss. Measures how far the predicted probability is from the true label. If true label is 1 and model says 0.9, loss is small. If it says 0.2, loss is big. Sigmoid = a function that squashes any raw number into a probability between 0 and 1.
- **Adam optimizer (lr=0.005)**: Adam = the algorithm that adjusts model weights after each mistake. Learning rate 0.005 means "take small steps." Too big → overshoot. Too small → takes forever. Same lr for all models.
- **Early stopping (patience=20)**: Stop training when validation F1 doesn't improve for 20 epochs. Without this, the model memorizes training data.

Problems in V1:
- GCN: loss spiked to 9.0, failed completely
- GAT: flat loss, never learned (underfitting)
- GraphSAGE: worked, F1 = 0.625

### V2: Fixing the problems

**Change 1: BCEWithLogitsLoss + class weights**

In V1, we applied sigmoid first, then computed loss. Problem: sigmoid can output values very close to 0, and log(0) = infinity → exploding numbers. BCEWithLogitsLoss combines sigmoid + loss in one numerically stable formula. Same result, no crashes.

Class weights (pos_weight = 1.61): Our data is 39% positive, 61% negative. Without weighting, the model learns a lazy shortcut: "just say NO to everything — I'll be right 61% of the time." Class weights say: "a missed positive costs 1.61x more than a false alarm." The 1.61 ≈ 61/39, compensating for the imbalance. Forces the model to actually find positives.

**Change 2: Gradient clipping (max_norm=1.0)**

During training, the model computes gradients — numbers that say "adjust this weight by this much." Sometimes gradients become huge (like 500.0), causing wild jumps and instability (that's what caused GCN's loss spike to 9.0). Gradient clipping caps the maximum step size to 1.0. The model still moves in the right direction, just not wildly.

**Change 3: Per-model learning rates**

In V1, all models used lr=0.005. Different architectures need different step sizes:
- GCN/GAT: lr=0.001 (smaller steps — these models are unstable, be gentle)
- GraphSAGE: lr=0.003 (more stable, can handle bigger steps)

**Change 4: LR scheduler (ReduceLROnPlateau)**

"If the model hasn't improved for 10 epochs, cut the learning rate in half." Early in training, big steps are fine — the model is far from optimal. Later, it's close to optimal and big steps cause overshooting. The scheduler automatically shrinks steps when progress stalls:

```
Epochs 1-30:   lr = 0.003  (exploring)
Epochs 30-40:  no improvement for 10 epochs → lr = 0.0015  (fine-tuning)
Epochs 40-50:  stalled again → lr = 0.00075  (polishing)
```

**Change 5: Text embeddings (+384d)**

Added sentence-transformer vectors to company features. 58 → 442 dimensions. The LLM component from feature engineering.

**V2 Results:**
- GraphSAGE: F1 = 0.628, AUC = 0.741 (improved)
- GCN/GAT: got WORSE — class weights made them predict "positive" for 99% of companies. They couldn't handle the weighted loss. Confirms GraphSAGE is fundamentally more robust.

### V3: Squeezing out more performance

**Change 1: Deep GraphSAGE (3 layers instead of 2)**

If 2 layers lets Company X see 2 hops away, maybe 3 layers (3 hops) would be better? We added:
- Residual connections: output = layer3(x) + x. If the 3rd layer doesn't help, the model can pass input through unchanged. Can never hurt, only help.
- BatchNorm: normalizes values between layers so they don't drift to extreme ranges.

Result: it actually got WORSE. F1 dropped from 0.628 to 0.619. This is over-smoothing — after 3 hops on a bipartite graph, every company has "heard from" almost every investor, so all representations start looking the same. The model can't distinguish between companies anymore. 2 layers is the sweet spot.

**Change 2: Threshold optimization**

By default we say "yes" if P > 0.5. But 0.5 isn't necessarily optimal. We compute the Precision-Recall curve on validation set — try every threshold, pick the one that maximizes F1. Maybe P > 0.43 works better because it catches more positives without too many false alarms.

**Change 3: Ensemble (XGBoost + GraphSAGE)**

Combine both models: P = w * P_XGBoost + (1-w) * P_GraphSAGE. Grid search w on validation (0.1 to 0.9 in steps of 0.05). Best w ≈ 0.4 (40% XGBoost, 60% GraphSAGE).

Why this works: XGBoost and GraphSAGE make different mistakes. XGBoost misses companies with weak features but strong investor networks. GraphSAGE misses companies with strong features but weak networks. Averaging cancels out individual errors.

**V3 Results:**
- Deep GraphSAGE: F1 = 0.619 (worse — over-smoothing)
- Ensemble: F1 = 0.638, AUC = 0.745 (best overall)

### The full journey

```
XGBoost alone:  F1 = 0.558  (starting point)
V1 GraphSAGE:   F1 = 0.625  (graph structure works!)
V2 GraphSAGE:   F1 = 0.628  (stabilized training + text)
V3 Ensemble:    F1 = 0.638  (combined the best of both worlds)
Total gain:     +14.3% F1
```
