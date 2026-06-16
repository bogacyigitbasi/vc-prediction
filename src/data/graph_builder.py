"""
Graph construction from Crunchbase investment data.

Produces edge lists and node index mappings as CSV/numpy files.
PyG HeteroData conversion happens in the Colab training notebook.
"""

from pathlib import Path

import pandas as pd
import numpy as np


def build_entity_indices(
    investments_df: pd.DataFrame,
    companies_df: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Create stable integer ID mappings for all companies and investors.
    Includes ALL entities (not just labeled ones) so graph is dense.
    """
    all_companies = sorted(
        set(investments_df["company_permalink"].unique())
        | set(companies_df["permalink"].unique())
    )
    all_investors = sorted(investments_df["investor_permalink"].dropna().unique())

    company_to_idx = {permalink: i for i, permalink in enumerate(all_companies)}
    investor_to_idx = {permalink: i for i, permalink in enumerate(all_investors)}

    return company_to_idx, investor_to_idx


def build_temporal_edge_list(
    investments_df: pd.DataFrame,
    cutoff_date: str,
    company_to_idx: dict[str, int],
    investor_to_idx: dict[str, int],
) -> pd.DataFrame:
    """
    Build edge list with ONLY edges before cutoff_date.

    Returns DataFrame with columns:
      investor_idx, company_idx, round_type_encoded, amount_log, days_since_epoch
    """
    inv = investments_df.copy()
    inv["funded_at"] = pd.to_datetime(inv["funded_at"], errors="coerce")
    inv = inv[inv["funded_at"].notna()]
    inv = inv[inv["funded_at"] < pd.Timestamp(cutoff_date)]

    inv = inv[inv["investor_permalink"].notna()]
    inv = inv[inv["company_permalink"].isin(company_to_idx)]
    inv = inv[inv["investor_permalink"].isin(investor_to_idx)]

    round_type_map = {
        "seed": 0, "angel": 1, "venture": 2, "undisclosed": 3,
        "convertible_note": 4, "debt_financing": 5, "private_equity": 6,
        "equity_crowdfunding": 7, "grant": 8, "post_ipo_equity": 9,
        "post_ipo_debt": 10, "secondary_market": 11,
        "non_equity_assistance": 12, "product_crowdfunding": 13,
    }

    edges = pd.DataFrame({
        "investor_idx": inv["investor_permalink"].map(investor_to_idx).values,
        "company_idx": inv["company_permalink"].map(company_to_idx).values,
        "round_type_encoded": inv["funding_round_type"].map(round_type_map).fillna(3).astype(int).values,
        "amount_log": np.log1p(pd.to_numeric(inv["raised_amount_usd"], errors="coerce").fillna(0)).values,
        "funded_at": inv["funded_at"].values,
    })

    epoch = pd.Timestamp("2000-01-01")
    edges["days_since_epoch"] = (edges["funded_at"] - epoch).dt.days
    edges = edges.drop(columns=["funded_at"])

    return edges.reset_index(drop=True)


def build_all_edge_lists(
    investments_df: pd.DataFrame,
    companies_df: pd.DataFrame,
    cutoff_dates: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """
    Build edge lists for all temporal splits.

    cutoff_dates: {'train': '2012-01-01', 'val': '2013-01-01', 'test': '2014-01-01'}
    """
    company_to_idx, investor_to_idx = build_entity_indices(investments_df, companies_df)

    edge_lists = {}
    for split_name, cutoff in cutoff_dates.items():
        edges = build_temporal_edge_list(
            investments_df, cutoff, company_to_idx, investor_to_idx
        )
        edge_lists[split_name] = edges
        print(
            f"  {split_name}: {len(edges):>7,} edges | "
            f"{edges['investor_idx'].nunique():>5,} investors | "
            f"{edges['company_idx'].nunique():>5,} companies"
        )

    return edge_lists, company_to_idx, investor_to_idx


def save_graph_data(
    output_dir: Path,
    edge_lists: dict[str, pd.DataFrame],
    company_to_idx: dict[str, int],
    investor_to_idx: dict[str, int],
) -> None:
    """Save all graph data as CSVs for upload to Colab."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, edges in edge_lists.items():
        edges.to_csv(output_dir / f"edges_{split_name}.csv", index=False)

    pd.DataFrame(
        list(company_to_idx.items()), columns=["permalink", "idx"]
    ).to_csv(output_dir / "company_index.csv", index=False)

    pd.DataFrame(
        list(investor_to_idx.items()), columns=["permalink", "idx"]
    ).to_csv(output_dir / "investor_index.csv", index=False)

    print(f"  Saved to {output_dir}/")
