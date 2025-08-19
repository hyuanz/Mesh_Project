from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def norm_date(date_str):
    if isinstance(date_str, pd.Series):
        return date_str.apply(lambda x: norm_date(x))
    try:
        return pd.to_datetime(date_str, format="%m/%d/%Y").strftime("%Y-%m-%d")
    except ValueError:
        try:
            return pd.to_datetime(date_str, format="%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            return pd.to_datetime(date_str).strftime("%Y-%m-%d")


def norm_amt(x, keep_sign: bool = True):
    v = pd.to_numeric(str(x).replace("$", "").replace(",", ""), errors="coerce")
    v = v.round(2)
    return v if keep_sign else v.abs()


def clean_desc(desc: str) -> str:
    if not isinstance(desc, str):
        return ""
    desc = desc.upper().strip()
    desc = re.sub(r"\bWWW\.[A-Z0-9\.-]+\.[A-Z]{2,}\b", "", desc)
    desc = re.sub(r"(#|NO\.?)?\s*\d{3,}", "", desc)
    desc = re.sub(r"[^\w\s]", " ", desc)
    desc = re.sub(r"\s+", " ", desc)
    return desc.strip()


def build_keys(df: pd.DataFrame, date_col: str, desc_col: str, amt_col: str) -> pd.DataFrame:
    df = df.copy()
    df.insert(0, "original_index", df.index)
    df["Date_norm"] = norm_date(df[date_col])
    df["Desc_norm"] = df[desc_col].apply(clean_desc)
    df["Amount_norm"] = df[amt_col].apply(norm_amt)
    df["base_key"] = df["Date_norm"] + "|" + df["Desc_norm"] + "|" + df["Amount_norm"].astype(str)
    df["occ"] = df.groupby("base_key").cumcount()
    df["match_key"] = df["base_key"] + "|occ=" + df["occ"].astype(str)
    return df


def duplicate_aware_reconcile(
    bank_df: pd.DataFrame,
    gl_df: pd.DataFrame,
    bank_date: str = "Date",
    bank_desc: str = "Description",
    bank_amt: str = "Amount",
    gl_date: str = "Transaction date",
    gl_desc: str = "Line description",
    gl_amt: str = "Category/Product/Service amount",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    b = build_keys(bank_df, bank_date, bank_desc, bank_amt)
    g = build_keys(gl_df, gl_date, gl_desc, gl_amt)

    matched_keys = (
        pd.merge(b[["match_key"]], g[["match_key"]], on="match_key", how="inner").drop_duplicates()
    )

    missing_in_gl = b.merge(matched_keys, on="match_key", how="left", indicator=True)
    missing_in_gl = (
        missing_in_gl[missing_in_gl["_merge"] == "left_only"].drop(columns="_merge").reset_index(drop=True)
    )
    missing_in_gl["new_index"] = range(len(missing_in_gl))

    missing_in_bank = g.merge(matched_keys, on="match_key", how="left", indicator=True)
    missing_in_bank = (
        missing_in_bank[missing_in_bank["_merge"] == "left_only"].drop(columns="_merge").reset_index(drop=True)
    )
    missing_in_bank["new_index"] = range(len(missing_in_bank))

    matched = pd.merge(
        b[["match_key", "original_index"]].rename(columns={"original_index": "original_index_bank"}),
        g[["match_key", "original_index"]].rename(columns={"original_index": "original_index_gl"}),
        on="match_key",
        how="inner",
    ).reset_index(drop=True)
    matched["new_index"] = range(len(matched))

    return matched, missing_in_gl, missing_in_bank


def run_reconciliation(
    bank_csv: Path | None = None,
    gl_csv: Path | None = None,
) -> dict[str, Any]:
    bank_path = bank_csv or (DATA_DIR / "bank_statement.csv")
    gl_path = gl_csv or (DATA_DIR / "general_ledger_report.csv")

    bank_df = pd.read_csv(bank_path)
    gl_df = pd.read_csv(gl_path)

    matched, missing_in_gl, missing_in_bank = duplicate_aware_reconcile(
        bank_df,
        gl_df,
        bank_date="Date",
        bank_desc="Description",
        bank_amt="Amount",
        gl_date="Transaction date",
        gl_desc="Line description",
        gl_amt="Category/Product/Service amount",
    )

    # Persist missing_in_gl for downstream categorization step
    out_missing_path = DATA_DIR / "missing_in_gl.csv"
    missing_in_gl.to_csv(out_missing_path, index=False)

    def to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
        return df.to_dict(orient="records")

    return {
        "matched": to_records(matched),
        "missing_in_gl": to_records(missing_in_gl),
        "missing_in_bank": to_records(missing_in_bank),
        "counts": {
            "matched": int(len(matched)),
            "missing_in_gl": int(len(missing_in_gl)),
            "missing_in_bank": int(len(missing_in_bank)),
        },
    }


