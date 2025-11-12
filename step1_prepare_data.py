#!/usr/bin/env python3
# step1_prepare_data.py

import argparse
import ast
import json
import os
import sys
import gzip
import io
from typing import Any, List, Optional

import pandas as pd


IRRELEVANT_COLS = [
    "Unnamed: 0",
    "tournamentTag",
    "time",            # keep if you need patch-time splits; otherwise drop
    "arena.name",      # example noisy cols—safe to drop if unused
    "arena.id",
    "type",            # e.g., classic/challenge/ladder—drop if not used
]


def smart_read(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Reads CSV (.csv or .csv.gz) or Parquet into a DataFrame.
    """
    lower = path.lower()
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return pd.read_parquet(path, engine="pyarrow")
    if lower.endswith(".csv") or lower.endswith(".txt"):
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    if lower.endswith(".csv.gz"):
        with gzip.open(path, "rb") as f:
            return pd.read_csv(io.BytesIO(f.read()), nrows=nrows, low_memory=False)
    raise ValueError(f"Unsupported file type: {path}")


def try_parse_list(x: Any) -> List[Any]:
    """
    Robustly parse a deck column which may be:
    - already a list
    - a Python-literal string like "['a','b']"
    - a JSON string like '["a","b"]'
    - malformed (returns [])
    """
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    # First try Python literal (fast for "['a','b']" style)
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return val
    except Exception:
        pass
    # Fallback: JSON
    try:
        import json as _json
        val = _json.loads(s)
        if isinstance(val, list):
            return val
    except Exception:
        pass
    return []


def clean_and_filter(
    df: pd.DataFrame,
    trophy_threshold: int,
    drop_cols: Optional[List[str]] = None,
    winner_col: str = "winner.cards.list",
    loser_col: str  = "loser.cards.list",
    trophies_col: str = "average.startingTrophies",
) -> pd.DataFrame:
    """
    Drop irrelevant columns, filter by trophies, and parse deck lists.
    """
    # Drop irrelevant columns (ignore missing)
    to_drop = (drop_cols or IRRELEVANT_COLS)
    df = df.drop(columns=[c for c in to_drop if c in df.columns], errors="ignore")

    # Ensure trophies column exists
    if trophies_col not in df.columns:
        raise KeyError(
            f"Expected trophies column '{trophies_col}' not found. "
            f"Available columns: {list(df.columns)[:20]}..."
        )

    # Filter high-level matches
    df = df[df[trophies_col] >= trophy_threshold].copy()

    # Parse decks
    for col in [winner_col, loser_col]:
        if col not in df.columns:
            raise KeyError(
                f"Expected deck column '{col}' not found. "
                f"Available columns: {list(df.columns)[:20]}..."
            )
        df[col] = df[col].apply(try_parse_list)

    # Basic sanity: keep only rows with 8-card decks on both sides (optional but recommended)
    df = df[df[winner_col].apply(len) == 8]
    df = df[df[loser_col].apply(len) == 8]

    # Reset index after filtering
    return df.reset_index(drop=True)


def summarize(df: pd.DataFrame,
              winner_col: str,
              loser_col: str,
              trophies_col: str) -> dict:
    # Count matches and basic stats
    n = len(df)
    avg_trophies = float(df[trophies_col].mean()) if trophies_col in df.columns and n else None

    # Unique cards seen (assuming IDs/names are hashable)
    def unique_cards(series):
        seen = set()
        for lst in series:
            seen.update(lst)
        return len(seen)

    uniq_w = unique_cards(df[winner_col]) if n else 0
    uniq_l = unique_cards(df[loser_col]) if n else 0

    # Example: distribution of deck sizes (should be 8 if cleaned)
    deck_sizes_w = df[winner_col].apply(len).value_counts().to_dict() if n else {}
    deck_sizes_l = df[loser_col].apply(len).value_counts().to_dict() if n else {}

    return {
        "matches_after_filter": n,
        "avg_starting_trophies": avg_trophies,
        "unique_cards_in_winner_decks": uniq_w,
        "unique_cards_in_loser_decks": uniq_l,
        "winner_deck_size_distribution": deck_sizes_w,
        "loser_deck_size_distribution": deck_sizes_l,
        "columns": list(df.columns),
    }


def main():
    p = argparse.ArgumentParser(description="Step 1: Clean & prepare Clash Royale matches for 4000+ trophies.")
    p.add_argument("--in", dest="in_path", required=True, help="Input file (CSV/CSV.GZ/Parquet)")
    p.add_argument("--out", dest="out_prefix", required=True, help="Output prefix (no extension)")
    p.add_argument("--trophy-threshold", type=int, default=4000, help="Minimum average.startingTrophies (default: 4000)")
    p.add_argument("--nrows", type=int, default=None, help="Read only first N rows (debugging)")
    p.add_argument("--sample", type=int, default=None, help="Optional: downsample cleaned matches to this many rows")
    p.add_argument("--winner-col", default="winner.cards.list", help="Column for winner deck list")
    p.add_argument("--loser-col", default="loser.cards.list", help="Column for loser deck list")
    p.add_argument("--trophies-col", default="average.startingTrophies", help="Column for average starting trophies")
    p.add_argument("--extra-drop-cols", nargs="*", default=None, help="Additional columns to drop (space-separated)")
    args = p.parse_args()

    # Read
    print(f"[Step1] Loading: {args.in_path}")
    df = smart_read(args.in_path, nrows=args.nrows)
    print(f"[Step1] Loaded shape: {df.shape}")

    # Clean + filter
    drop_cols = IRRELEVANT_COLS
    if args.extra_drop_cols:
        drop_cols = list(set(drop_cols + args.extra_drop_cols))

    df = clean_and_filter(
        df,
        trophy_threshold=args.trophy_threshold,
        drop_cols=drop_cols,
        winner_col=args.winner_col,
        loser_col=args.loser_col,
        trophies_col=args.trophies_col,
    )
    print(f"[Step1] After filter & parsing: {df.shape}")

    # Optional sample
    if args.sample is not None and args.sample < len(df):
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"[Step1] Downsampled to: {df.shape}")

    # Write outputs
    base = args.out_prefix
    csv_path = f"{base}.csv"
    parquet_path = f"{base}.parquet"
    meta_path = f"{base}_summary.json"

    print(f"[Step1] Writing CSV: {csv_path}")
    df.to_csv(csv_path, index=False)

    print(f"[Step1] Writing Parquet: {parquet_path}")
    try:
        df.to_parquet(parquet_path, engine="pyarrow", index=False)
    except Exception as e:
        print(f"[Step1] Parquet write skipped (install pyarrow to enable). Error: {e}")

    # Summary metadata
    summary = summarize(
        df,
        winner_col=args.winner_col,
        loser_col=args.loser_col,
        trophies_col=args.trophies_col,
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Step1] Wrote summary: {meta_path}")
    print("[Step1] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
