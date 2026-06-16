import pandas as pd
import numpy as np

TRAIN_END = "2012-01-01"
VAL_END = "2013-01-01"
TEST_END = "2014-01-01"


def assign_temporal_split(
    labeled_df: pd.DataFrame,
    train_end: str = TRAIN_END,
    val_end: str = VAL_END,
    test_end: str = TEST_END,
) -> pd.DataFrame:
    """
    Assign splits based on trigger_funded_at:
      train: trigger_funded_at < train_end
      val:   train_end <= trigger_funded_at < val_end
      test:  val_end <= trigger_funded_at < test_end
    """
    df = labeled_df.copy()
    df["trigger_funded_at"] = pd.to_datetime(df["trigger_funded_at"])

    conditions = [
        df["trigger_funded_at"] < pd.Timestamp(train_end),
        (df["trigger_funded_at"] >= pd.Timestamp(train_end))
        & (df["trigger_funded_at"] < pd.Timestamp(val_end)),
        (df["trigger_funded_at"] >= pd.Timestamp(val_end))
        & (df["trigger_funded_at"] < pd.Timestamp(test_end)),
    ]
    choices = ["train", "val", "test"]

    df["split"] = np.select(conditions, choices, default="excluded")
    df = df[df["split"] != "excluded"].reset_index(drop=True)

    return df


def validate_no_leakage(splits_df: pd.DataFrame) -> bool:
    """
    Verify temporal integrity of the split:
    1. No company appears in multiple splits
    2. All train dates < all val dates < all test dates
    3. Splits are non-empty
    """
    # Check 1: No company in multiple splits
    company_splits = splits_df.groupby("company_permalink")["split"].nunique()
    assert (
        company_splits.max() == 1
    ), f"Company appears in {company_splits.max()} splits!"

    # Check 2: Temporal ordering
    train = splits_df[splits_df["split"] == "train"]["trigger_funded_at"]
    val = splits_df[splits_df["split"] == "val"]["trigger_funded_at"]
    test = splits_df[splits_df["split"] == "test"]["trigger_funded_at"]

    assert train.max() < val.min(), "Train/val temporal overlap!"
    assert val.max() < test.min(), "Val/test temporal overlap!"

    # Check 3: Non-empty splits
    for split_name in ["train", "val", "test"]:
        count = (splits_df["split"] == split_name).sum()
        assert count > 0, f"Empty {split_name} split!"

    return True


def get_graph_cutoff_dates() -> dict[str, str]:
    """
    Return the graph edge cutoff date for each split.
    The graph for each split contains ONLY edges before this date.
    """
    return {
        "train": TRAIN_END,
        "val": VAL_END,
        "test": TEST_END,
    }


def print_split_summary(splits_df: pd.DataFrame) -> None:
    """Print summary statistics for each split."""
    for split_name in ["train", "val", "test"]:
        subset = splits_df[splits_df["split"] == split_name]
        if len(subset) == 0:
            print(f"  {split_name:>5}: 0 companies")
            continue
        pos_rate = subset["label"].mean()
        valid_dates = subset["trigger_funded_at"].dropna()
        date_min = valid_dates.min().strftime("%Y-%m-%d") if len(valid_dates) > 0 else "N/A"
        date_max = valid_dates.max().strftime("%Y-%m-%d") if len(valid_dates) > 0 else "N/A"
        print(
            f"  {split_name:>5}: {len(subset):>6,} companies | "
            f"positive rate: {100*pos_rate:.1f}% | "
            f"date range: {date_min} to {date_max}"
        )
