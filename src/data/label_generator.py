from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


def identify_trigger_rounds(rounds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find the FIRST Seed or Series A round for each company.

    Seed: funding_round_type == 'seed'
    Series A: funding_round_type == 'venture' AND funding_round_code == 'A'

    Returns one row per company: the earliest qualifying round.
    """
    is_seed = rounds_df["funding_round_type"] == "seed"
    is_series_a = (rounds_df["funding_round_type"] == "venture") & (
        rounds_df["funding_round_code"] == "A"
    )
    triggers = rounds_df[is_seed | is_series_a].copy()
    triggers = triggers.sort_values("funded_at")
    triggers = triggers.drop_duplicates(subset="company_permalink", keep="first")
    return triggers.reset_index(drop=True)


def generate_followon_labels(
    rounds_df: pd.DataFrame,
    trigger_rounds: pd.DataFrame,
    horizon_months: int = 24,
) -> pd.DataFrame:
    """
    For each trigger round, check if the company has ANY subsequent round
    within `horizon_months` of the trigger date.

    Label = 1 if follow-on exists within horizon, 0 otherwise.
    """
    horizon_delta = pd.DateOffset(months=horizon_months)

    all_rounds_sorted = rounds_df.sort_values(["company_permalink", "funded_at"])

    labels = []
    for _, trigger in trigger_rounds.iterrows():
        company = trigger["company_permalink"]
        trigger_date = trigger["funded_at"]
        horizon_end = trigger_date + horizon_delta

        company_rounds = all_rounds_sorted[
            all_rounds_sorted["company_permalink"] == company
        ]
        followon_rounds = company_rounds[
            company_rounds["funded_at"] > trigger_date
        ]
        within_horizon = followon_rounds[followon_rounds["funded_at"] <= horizon_end]

        if len(within_horizon) > 0:
            first_followon = within_horizon.iloc[0]
            labels.append(
                {
                    "company_permalink": company,
                    "company_name": trigger["company_name"],
                    "trigger_round_type": trigger["funding_round_type"],
                    "trigger_round_code": trigger["funding_round_code"],
                    "trigger_funded_at": trigger_date,
                    "trigger_amount_usd": trigger.get("raised_amount_usd"),
                    "label": 1,
                    "followon_funded_at": first_followon["funded_at"],
                    "followon_round_type": first_followon["funding_round_type"],
                    "followon_round_code": first_followon.get("funding_round_code"),
                }
            )
        else:
            labels.append(
                {
                    "company_permalink": company,
                    "company_name": trigger["company_name"],
                    "trigger_round_type": trigger["funding_round_type"],
                    "trigger_round_code": trigger["funding_round_code"],
                    "trigger_funded_at": trigger_date,
                    "trigger_amount_usd": trigger.get("raised_amount_usd"),
                    "label": 0,
                    "followon_funded_at": pd.NaT,
                    "followon_round_type": None,
                    "followon_round_code": None,
                }
            )

    return pd.DataFrame(labels)


def filter_observable_labels(
    labeled_df: pd.DataFrame,
    dataset_end_date: str = "2015-12-05",
    horizon_months: int = 24,
) -> pd.DataFrame:
    """
    Remove samples where we cannot observe the full prediction window.

    A label is observable only if trigger_funded_at + horizon_months <= dataset_end_date.
    For label=0, we need to confirm nothing happened in the full window.
    """
    end_date = pd.Timestamp(dataset_end_date)
    horizon_delta = pd.DateOffset(months=horizon_months)

    observable = labeled_df[
        labeled_df["trigger_funded_at"] + horizon_delta <= end_date
    ].copy()

    return observable.reset_index(drop=True)


def generate_labels_fast(
    rounds_df: pd.DataFrame,
    horizon_months: int = 24,
    dataset_end_date: str = "2015-12-05",
) -> pd.DataFrame:
    """
    Optimized label generation using vectorized groupby operations.
    Combines identify_trigger_rounds + generate_followon_labels + filter_observable.
    """
    rounds_sorted = rounds_df.sort_values(["company_permalink", "funded_at"]).copy()
    rounds_sorted["funded_at"] = pd.to_datetime(rounds_sorted["funded_at"])

    is_seed = rounds_sorted["funding_round_type"] == "seed"
    is_series_a = (rounds_sorted["funding_round_type"] == "venture") & (
        rounds_sorted["funding_round_code"] == "A"
    )
    trigger_mask = is_seed | is_series_a
    triggers = rounds_sorted[trigger_mask].drop_duplicates(
        subset="company_permalink", keep="first"
    )

    horizon_delta = pd.DateOffset(months=horizon_months)
    end_date = pd.Timestamp(dataset_end_date)

    # Filter to observable window
    triggers = triggers[triggers["funded_at"] + horizon_delta <= end_date].copy()

    # For each trigger, find if follow-on exists within horizon
    results = []
    company_groups = rounds_sorted.groupby("company_permalink")

    for _, trigger in triggers.iterrows():
        company = trigger["company_permalink"]
        trigger_date = trigger["funded_at"]
        horizon_end = trigger_date + horizon_delta

        group = company_groups.get_group(company)
        followons = group[
            (group["funded_at"] > trigger_date) & (group["funded_at"] <= horizon_end)
        ]

        label = 1 if len(followons) > 0 else 0
        followon_date = followons.iloc[0]["funded_at"] if label == 1 else pd.NaT
        followon_type = followons.iloc[0]["funding_round_type"] if label == 1 else None

        results.append(
            {
                "company_permalink": company,
                "company_name": trigger["company_name"],
                "trigger_round_type": trigger["funding_round_type"],
                "trigger_round_code": trigger["funding_round_code"],
                "trigger_funded_at": trigger_date,
                "trigger_amount_usd": trigger.get("raised_amount_usd"),
                "label": label,
                "followon_funded_at": followon_date,
                "followon_round_type": followon_type,
            }
        )

    return pd.DataFrame(results)
