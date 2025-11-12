#!/usr/bin/env python3
# step1_prepare_data_chunked.py
# Stream the big CSV in chunks to avoid out-of-memory.

import argparse
import ast
import csv
import json
import os
import sys
from typing import Any, List, Optional, Dict
import pandas as pd

# Columns you probably don't need in Step 1 (safe to drop if present)
IRRELEVANT_COLS = [
    "Unnamed: 0", "tournamentTag", "time", "arena.name", "arena.id", "type"
]

def try_parse_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return val
    except Exception:
        pass
    try:
        import json as _json
        val = _json.loads(s)
        if isinstance(val, list):
            return val
    except Exception:
        pass
    return []

def process_chunk(df: pd.DataFrame,
                  trophy_threshold: int,
                  winner_col: str,
                  loser_col: str,
                  trophies_col: str) -> pd.DataFrame:
    # Drop irrelevant columns (ignore missing)
    df = df.drop(columns=[c for c in IRRELEVANT_COLS if c in df.columns], errors="ignore")

    # Filter by trophies
    if trophies_col not in df.columns:
        raise KeyError(f"Expected trophies column '{trophies_col}' not found.")
    df = df[df[trophies_col] >= trophy_threshold].copy()
    if df.empty:
        return df

    # Parse decks
    for col in (winner_col, loser_col):
        if col not in df.columns:
            raise KeyError(f"Expected deck column '{col}' not found.")
        df[col] = df[col].apply(try_parse_list)

    # Keep only rows with 8-card decks on both sides
    df = df[df[winner_col].apply(len) == 8]
    df = df[df[loser_col].apply(len) == 8]

    # Optional: ensure lists are stored as JSON strings so CSV/Parquet handle them cleanly
    df[winner_col] = df[winner_col].apply(json.dumps)
    df[loser_col]  = df[loser_col].apply(json.dumps)

    return df

def update_summary(summary: Dict, df: pd.DataFrame,
                   winner_col: str, loser_col: str, trophies_col: str):
    n = len(df)
    summary["matches_after_filter"] += n
    if n:
        summary["trophies_sum"] += float(df[trophies_col].sum())
        summary["trophies_count"] += n

        def add_uniques(series, target_set):
            for s in series:
                try:
                    lst = json.loads(s) if isinstance(s, str) else s
                except Exception:
                    lst = []
                target_set.update(lst)

        add_uniques(df[winner_col], summary["_uniq_w"])
        add_uniques(df[loser_col], summary["_uniq_l"])

        for size in df[winner_col].apply(lambda s: len(json.loads(s)) if isinstance(s, str) else len(s)).value_counts().to_dict().items():
            k, v = size
            summary["winner_deck_size_distribution"][str(k)] = summary["winner_deck_size_distribution"].get(str(k), 0) + v
        for size in df[loser_col].apply(lambda s: len(json.loads(s)) if isinstance(s, str) else len(s)).value_counts().to_dict().items():
            k, v = size
            summary["loser_deck_size_distribution"][str(k)] = summary["loser_deck_size_distribution"].get(str(k), 0) + v

def main():
    ap = argparse.ArgumentParser(description="Chunked Step 1: Clean & prepare matches (>=4000 trophies) without OOM.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input CSV (can be huge)")
    ap.add_argument("--out", dest="out_prefix", required=True, help="Output prefix (no extension)")
    ap.add_argument("--trophy-threshold", type=int, default=4000, help="Min average.startingTrophies (default 4000)")
    ap.add_argument("--chunksize", type=int, default=250000, help="Rows per chunk (tune to your RAM)")
    ap.add_argument("--winner-col", default="winner.cards.list", help="Winner deck column name")
    ap.add_argument("--loser-col", default="loser.cards.list", help="Loser deck column name")
    ap.add_argument("--trophies-col", default="average.startingTrophies", help="Trophies column name")
    ap.add_argument("--usecols", nargs="*", default=None,
                    help="Limit columns to reduce memory. If omitted, read all.")
    args = ap.parse_args()

    in_path = args.in_path
    out_csv = f"{args.out_prefix}.csv"
    out_parquet = f"{args.out_prefix}.parquet"
    out_summary = f"{args.out_prefix}_summary.json"

    # If user didn't specify usecols, pick a lean default set if present
    default_cols = [
        args.trophies_col, args.winner_col, args.loser_col,
        # add any known-needed cols here (e.g., elixir, rarity counts) to keep memory low:
        "winner.elixir.average", "loser.elixir.average",
        "winner.totalcard.level", "loser.totalcard.level",
        "winner.legendary.count", "loser.legendary.count"
    ]
    usecols = args.usecols or None

    # Initialize outputs
    wrote_header = False
    if os.path.exists(out_csv):
        os.remove(out_csv)

    # Summary accumulators
    summary = {
        "matches_after_filter": 0,
        "avg_starting_trophies": None,
        "unique_cards_in_winner_decks": 0,
        "unique_cards_in_loser_decks": 0,
        "winner_deck_size_distribution": {},
        "loser_deck_size_distribution": {},
        "columns": [],
        "trophies_sum": 0.0,
        "trophies_count": 0,
        "_uniq_w": set(),
        "_uniq_l": set(),
    }

    # Stream the CSV in chunks
    try:
        chunk_iter = pd.read_csv(
            in_path,
            usecols=usecols,
            chunksize=args.chunksize,
            low_memory=True,
            dtype=str,  # read as strings first; we only need to parse a few fields
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="skip",
            engine="c",
        )
    except TypeError:
        # pandas < 1.3 compatibility for on_bad_lines
        chunk_iter = pd.read_csv(
            in_path,
            usecols=usecols,
            chunksize=args.chunksize,
            low_memory=True,
            dtype=str,
            engine="c",
        )

    chunk_idx = 0
    for raw in chunk_iter:
        chunk_idx += 1
        # Convert trophies col to numeric (errors coerce to NaN → dropped)
        if args.trophies_col in raw.columns:
            raw[args.trophies_col] = pd.to_numeric(raw[args.trophies_col], errors="coerce")

        cleaned = process_chunk(
            raw,
            trophy_threshold=args.trophy_threshold,
            winner_col=args.winner_col,
            loser_col=args.loser_col,
            trophies_col=args.trophies_col,
        )
        if cleaned.empty:
            continue

        # Keep track of column order for summary
        if not summary["columns"]:
            summary["columns"] = list(cleaned.columns)

        # Append to CSV (streaming)
        cleaned.to_csv(out_csv, index=False, mode="a", header=not wrote_header)
        wrote_header = True

        # Update summary
        update_summary(summary, cleaned, args.winner_col, args.loser_col, args.trophies_col)

        print(f"[Step1-Chunked] Processed chunk {chunk_idx}, wrote {len(cleaned)} rows.")

    # Finalize summary
    if summary["trophies_count"] > 0:
        summary["avg_starting_trophies"] = summary["trophies_sum"] / summary["trophies_count"]
    summary["unique_cards_in_winner_decks"] = len(summary["_uniq_w"])
    summary["unique_cards_in_loser_decks"] = len(summary["_uniq_l"])
    # Drop internal fields
    del summary["_uniq_w"], summary["_uniq_l"], summary["trophies_sum"], summary["trophies_count"]

    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Step1-Chunked] Wrote CSV: {out_csv}")
    print(f"[Step1-Chunked] Wrote summary: {out_summary}")
    print("[Step1-Chunked] Done.")

    # Optional: try to convert CSV→Parquet (faster for next steps), streaming-friendly via read_csv again
    try:
        # Read the resulting smaller CSV in a single pass and write Parquet
        df_small = pd.read_csv(out_csv, low_memory=False)
        df_small.to_parquet(out_parquet, engine="pyarrow", index=False)
        print(f"[Step1-Chunked] Wrote Parquet: {out_parquet}")
    except Exception as e:
        print(f"[Step1-Chunked] Parquet write skipped (install pyarrow to enable). Error: {e}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
