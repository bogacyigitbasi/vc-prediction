"""
Node feature engineering for companies and investors.

All features respect temporal cutoffs — only information available
before the cutoff date is used.

Fully vectorized with pandas groupby — no per-entity loops.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def _get_top_categories(companies_df: pd.DataFrame, top_n: int = 50) -> list[str]:
    """Extract top-N categories from pipe-delimited category_list."""
    all_cats = companies_df["category_list"].dropna().str.split("|").explode()
    return all_cats.value_counts().head(top_n).index.tolist()


def compute_company_features(
    companies_df: pd.DataFrame,
    rounds_df: pd.DataFrame,
    investments_df: pd.DataFrame,
    company_to_idx: dict[str, int],
    cutoff_date: str,
) -> np.ndarray:
    """
    Vectorized company feature computation using only pre-cutoff data.

    Features: founded_year, days_since_founded, log_total_raised,
    round_count, investor_count, log_max_amount, days_since_last_round,
    is_us, + top-50 category one-hot (58 total).
    """
    cutoff = pd.Timestamp(cutoff_date)
    n_companies = len(company_to_idx)
    top_cats = _get_top_categories(companies_df)

    # Temporal filtering
    rounds = rounds_df.copy()
    rounds["funded_at"] = pd.to_datetime(rounds["funded_at"], errors="coerce")
    rounds["raised_amount_usd"] = pd.to_numeric(rounds["raised_amount_usd"], errors="coerce")
    rounds_before = rounds[rounds["funded_at"] < cutoff]

    inv = investments_df.copy()
    inv["funded_at"] = pd.to_datetime(inv["funded_at"], errors="coerce")
    inv_before = inv[inv["funded_at"] < cutoff]

    # Vectorized aggregation
    rounds_agg = rounds_before.groupby("company_permalink").agg(
        round_count=("funded_at", "size"),
        total_raised=("raised_amount_usd", "sum"),
        max_amount=("raised_amount_usd", "max"),
        last_round_date=("funded_at", "max"),
    )

    inv_agg = inv_before.groupby("company_permalink")["investor_permalink"].nunique()
    inv_agg.name = "investor_count"

    # Build company info DataFrame indexed by permalink
    comp = companies_df.set_index("permalink")
    comp["founded_at"] = pd.to_datetime(comp["founded_at"], errors="coerce")

    # Join all aggregates
    comp = comp.join(rounds_agg, how="left")
    comp = comp.join(inv_agg, how="left")
    comp = comp.fillna({"round_count": 0, "total_raised": 0, "max_amount": 0, "investor_count": 0})

    # Build feature matrix
    n_features = 8 + len(top_cats)
    features = np.zeros((n_companies, n_features), dtype=np.float32)

    # Map company_to_idx to a Series for vectorized access
    idx_series = pd.Series(company_to_idx)
    common = idx_series.index.intersection(comp.index)
    idxs = idx_series[common].values

    comp_sub = comp.loc[common]

    # Feature 0: founded_year
    founded = comp_sub["founded_at"]
    features[idxs, 0] = np.where(founded.notna(), founded.dt.year, 0).astype(np.float32)

    # Feature 1: days_since_founded
    features[idxs, 1] = np.where(
        founded.notna(), (cutoff - founded).dt.days, 0
    ).astype(np.float32)

    # Feature 2: log total raised
    features[idxs, 2] = np.log1p(comp_sub["total_raised"].values).astype(np.float32)

    # Feature 3: round count
    features[idxs, 3] = comp_sub["round_count"].values.astype(np.float32)

    # Feature 4: investor count
    features[idxs, 4] = comp_sub["investor_count"].values.astype(np.float32)

    # Feature 5: log max amount
    features[idxs, 5] = np.log1p(comp_sub["max_amount"].fillna(0).values).astype(np.float32)

    # Feature 6: days since last round
    last_round = comp_sub["last_round_date"]
    features[idxs, 6] = np.where(
        last_round.notna(), (cutoff - last_round).dt.days, 0
    ).astype(np.float32)

    # Feature 7: is_us
    features[idxs, 7] = (comp_sub["country_code"] == "USA").values.astype(np.float32)

    # Features 8+: category one-hot
    cat_to_col = {cat: 8 + i for i, cat in enumerate(top_cats)}
    for permalink, idx in zip(common, idxs):
        cats = comp_sub.at[permalink, "category_list"]
        if pd.notna(cats):
            for cat in str(cats).split("|"):
                if cat in cat_to_col:
                    features[idx, cat_to_col[cat]] = 1.0

    # Normalize continuous features
    for col in range(7):
        col_data = features[:, col]
        mask = col_data != 0
        if mask.sum() > 0:
            mean, std = col_data[mask].mean(), col_data[mask].std()
            if std > 0:
                features[:, col] = np.where(mask, (col_data - mean) / std, 0)

    return features


def compute_investor_features(
    investments_df: pd.DataFrame,
    investor_to_idx: dict[str, int],
    cutoff_date: str,
) -> np.ndarray:
    """
    Vectorized investor feature computation using only pre-cutoff data.

    Features: total_investments, portfolio_size, category_diversity,
    log_median_amount, years_active, avg_round_stage, geo_diversity (7 total).
    """
    cutoff = pd.Timestamp(cutoff_date)
    n_investors = len(investor_to_idx)

    inv = investments_df.copy()
    inv["funded_at"] = pd.to_datetime(inv["funded_at"], errors="coerce")
    inv = inv[inv["funded_at"] < cutoff]
    inv["amount"] = pd.to_numeric(inv["raised_amount_usd"], errors="coerce")

    stage_map = {"seed": 1, "angel": 1, "A": 2, "B": 3, "C": 4, "D": 5}
    inv["stage_num"] = inv["funding_round_code"].map(stage_map)

    # Vectorized aggregation
    agg = inv.groupby("investor_permalink").agg(
        total_inv=("funded_at", "size"),
        portfolio_size=("company_permalink", "nunique"),
        median_amount=("amount", "median"),
        min_date=("funded_at", "min"),
        max_date=("funded_at", "max"),
        avg_stage=("stage_num", "mean"),
        geo_div=("investor_country_code", "nunique"),
    )

    # Category diversity (needs explode, do separately)
    cat_div = (
        inv.dropna(subset=["company_category_list"])
        .assign(cats=lambda df: df["company_category_list"].str.split("|"))
        .explode("cats")
        .groupby("investor_permalink")["cats"]
        .nunique()
    )
    cat_div.name = "cat_diversity"
    agg = agg.join(cat_div, how="left").fillna({"cat_diversity": 0})

    features = np.zeros((n_investors, 7), dtype=np.float32)

    idx_series = pd.Series(investor_to_idx)
    common = idx_series.index.intersection(agg.index)
    idxs = idx_series[common].values

    agg_sub = agg.loc[common]

    features[idxs, 0] = agg_sub["total_inv"].values.astype(np.float32)
    features[idxs, 1] = agg_sub["portfolio_size"].values.astype(np.float32)
    features[idxs, 2] = agg_sub["cat_diversity"].values.astype(np.float32)
    features[idxs, 3] = np.log1p(agg_sub["median_amount"].fillna(0).values).astype(np.float32)
    features[idxs, 4] = (
        (agg_sub["max_date"] - agg_sub["min_date"]).dt.days / 365.25
    ).fillna(0).values.astype(np.float32)
    features[idxs, 5] = agg_sub["avg_stage"].fillna(0).values.astype(np.float32)
    features[idxs, 6] = agg_sub["geo_div"].values.astype(np.float32)

    for col in range(7):
        col_data = features[:, col]
        mask = col_data != 0
        if mask.sum() > 0:
            mean, std = col_data[mask].mean(), col_data[mask].std()
            if std > 0:
                features[:, col] = np.where(mask, (col_data - mean) / std, 0)

    return features


def save_features(
    output_dir: Path,
    company_features: dict[str, np.ndarray],
    investor_features: dict[str, np.ndarray],
) -> None:
    """Save feature matrices as numpy files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in company_features:
        np.save(output_dir / f"company_features_{split_name}.npy", company_features[split_name])
        np.save(output_dir / f"investor_features_{split_name}.npy", investor_features[split_name])

    print(f"  Saved feature matrices to {output_dir}/")
