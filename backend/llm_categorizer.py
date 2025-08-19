from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
logger = logging.getLogger(__name__)


def load_csvs(
    coa_path: Path | None = None,
    missing_in_gl_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load COA and missing_in_gl CSVs from disk.

    Defaults to files under the repo-level data directory.
    """
    coa_csv = coa_path or (DATA_DIR / "sample_chart_of_accounts.csv")
    missing_csv = missing_in_gl_path or (DATA_DIR / "missing_in_gl.csv")

    coa_df = pd.read_csv(coa_csv)
    missing_in_gl = pd.read_csv(missing_csv, index_col="new_index").rename_axis(None)
    logger.info("Loaded CSVs: COA rows=%d, Missing-in-GL rows=%d", len(coa_df), len(missing_in_gl))
    return coa_df, missing_in_gl


def extract_unique_descs(desc_series: pd.Series, limit: int | None = 100) -> list[str]:
    """Return a list of unique, normalized Desc_norm strings (lowercased, trimmed)."""
    values = (
        desc_series.dropna().astype(str).str.strip().str.lower().replace("", pd.NA).dropna().unique().tolist()
    )
    values = values[: limit or len(values)]
    logger.info("Collected %d unique desc_norm strings for mapping", len(values))
    return values


# --- Simple rule-based mapping to enforce obvious service-type categories ---

AIRLINE_TOKENS = [
    "airlines",
    "airways",
    "airline",
    "southwest",
    "delta",
    "american",
    "united",
    "jetblue",
    "alaska",
    "spirit",
    "frontier",
]
HOTEL_TOKENS = [
    "hotel",
    "marriott",
    "hilton",
    "hyatt",
    "ihg",
    "holiday inn",
    "westin",
    "sheraton",
    "resort",
    "motel",
    "hampton",
    "doubletree",
    "embassy suites",
]
RIDESHARE_TOKENS = ["uber", "lyft", "taxi", "cab"]
MEAL_TOKENS = ["starbucks", "coffee", "cafe", "chipotle", "grill", "bar", "restaurant", "tapas", "blue bottle"]


def contains_any(text: str, tokens: list[str]) -> bool:
    return any(tok in text for tok in tokens)


def rule_based_mapping_for_desc(desc: str) -> dict[str, Any] | None:
    """Return a mapping dict for obvious cases; otherwise None.

    The account names here should exist in the provided COA.
    """
    t = desc.strip().lower()
    # Airlines
    if contains_any(t, AIRLINE_TOKENS):
        return {
            "desc_norm": desc,
            "account_name": "Travel:Airfare",
            "is_travel": True,
            "rationale": "Airline vendor implies airfare (travel).",
        }
    # Hotels and lodging
    if contains_any(t, HOTEL_TOKENS):
        return {
            "desc_norm": desc,
            "account_name": "Travel:Hotels",
            "is_travel": True,
            "rationale": "Hotel vendor implies lodging (travel).",
        }
    # Rideshare / taxi
    if contains_any(t, RIDESHARE_TOKENS):
        return {
            "desc_norm": desc,
            "account_name": "Travel:Taxis or shared rides",
            "is_travel": True,
            "rationale": "Ride-hailing/taxi implies ground transportation (travel).",
        }
    # Payroll
    if "gusto" in t or "payroll" in t:
        return {
            "desc_norm": desc,
            "account_name": "Payroll expenses",
            "is_travel": False,
            "rationale": "Payroll-related description maps to Payroll expenses.",
        }
    # Refunds
    if "refund" in t:
        return {
            "desc_norm": desc,
            "account_name": "Uncategorized Expense",
            "is_travel": False,
            "rationale": "Refund without context.",
        }
    # Meals & coffee shops
    if contains_any(t, MEAL_TOKENS):
        return {
            "desc_norm": desc,
            "account_name": "Meals:Travel meals",
            "is_travel": True,
            "rationale": "Restaurant/coffee likely during travel; treat as travel meals unless stated otherwise.",
        }
    return None


def apply_rule_based_mappings(desc_norms: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Apply simple rules. Return (mapped_items, remaining_descs)."""
    mapped: list[dict[str, Any]] = []
    remaining: list[str] = []
    for d in desc_norms:
        item = rule_based_mapping_for_desc(d)
        if item is not None:
            mapped.append(item)
        else:
            remaining.append(d)
    logging.info("Rule-based mapped %d/%d descriptions", len(mapped), len(desc_norms))
    return mapped, remaining


def build_mapping_prompt(desc_norms: list[str], coa_df: pd.DataFrame) -> str:
    # Provide the full COA catalog with the three relevant columns together
    cols = [c for c in ["Account Name", "Type", "Description"] if c in coa_df.columns]
    coa_records = (
        coa_df[cols]
        .fillna("")
        .astype(str)
        .to_dict(orient="records")
    )
    prompt = (
        "You are categorizing transactions.\n"
        "Goal: For each transaction description (desc_norm), pick the single best Account Name from the Chart of Accounts.\n"
        "- Evaluate COA entries using ALL THREE fields together: 'Account Name', 'Type', and 'Description' (treat them as one profile).\n"
        "- First, infer the likely store/vendor name from the description (e.g., 'APLPAY' -> 'Apple Pay', 'AMAZON COM' -> 'Amazon').\n"
        "- Guess the most appropriate category from the COA. If uncertain, still provide your best guess AND set not_sure=true.\n"
        "- Travel is only one signal; do NOT default decisions to travel. Use vendor/service type and context (including location hints) holistically.\n"
        "- The chosen account_name MUST be exactly one of the 'Account Name' values in the COA catalog provided below.\n\n"
        f"Descriptions (desc_norm): {desc_norms}\n\n"
        f"COA catalog (use these for matching; each item has Account Name, Type, Description):\n{coa_records}\n\n"
        "Return ONLY a JSON OBJECT with key 'items' containing objects with fields: "
        "desc_norm (str), account_name (str from COA), is_travel (bool), not_sure (bool), guessed_store (str), rationale (str)."
    )
    logger.info("Built mapping prompt for %d descriptions with full COA catalog (%d rows)", len(desc_norms), len(coa_records))
    return prompt


SYSTEM_PROMPT = (
    "You map transaction descriptions to accounting Account Names using the provided COA (Account Name + Type + Description). "
    "First infer vendor/store, then choose the best account. If uncertain, set not_sure=true and still provide your best guess. "
    "Return strict JSON OBJECT with an 'items' array: "
    "{\"items\": [{\"desc_norm\": str, \"account_name\": str, \"is_travel\": bool, \"not_sure\": bool, \"guessed_store\": str, \"rationale\": str}]}. "
    "account_name MUST be from the provided list."
)


def run_llm_mapping(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    """Call the preferred LLM provider. Tries `agents` first, then OpenAI if available.

    Returns the raw model output (expected to be JSON string).
    """
    # Try the agents API first
    try:
        from agents import Agent, Runner  # type: ignore
        from agents.lifecycle import RunHooksBase  # type: ignore

        agent = Agent(system_prompt=system_prompt)
        runner = Runner(agent, hooks=None)  # hooks optional
        logger.info("Calling LLM via agents backend")
        output = runner.run(prompt)
        logger.info("agents backend returned %d chars", len(str(output) or ""))
        return output
    except Exception:
        pass

    # Fallback to OpenAI (python SDK v1.x) if installed and key present
    try:
        from openai import OpenAI  # type: ignore

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")

        client = OpenAI(api_key=api_key)
        logger.info("Calling LLM via OpenAI backend")
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content or "[]"
        logger.info("OpenAI backend returned %d chars", len(content))
        return content
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "No LLM backend available. Install and configure `agents` or `openai`."
        ) from exc


def extract_json_from_text(text: str) -> str | None:
    """Extract a JSON array or object from text that might include prose or code fences."""
    # Try code fence first
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    # Try to find the first JSON-looking bracketed content
    bracket_match = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
    if bracket_match:
        return bracket_match.group(1)
    return None


def parse_llm_json(output: str) -> list[dict[str, Any]]:
    """Parse LLM JSON into a normalized list of mapping dicts."""
    try:
        candidate = output.strip() or ""
        try:
            data = json.loads(candidate)
        except Exception:
            extracted = extract_json_from_text(candidate) or ""
            data = json.loads(extracted)
        if isinstance(data, dict):
            # Preferred: items/data/mappings/result is a list
            for k in ["items", "data", "mappings", "result"]:
                if k in data and isinstance(data[k], list):
                    data = data[k]
                    break
            else:
                # If dict looks like a single mapping object, wrap it
                if {"desc_norm", "account_name"}.issubset(set(map(str, data.keys()))):
                    data = [data]
                else:
                    # If dict contains a list under some other key, try to find it
                    list_values = [v for v in data.values() if isinstance(v, list)]
                    if list_values:
                        data = list_values[0]
                    else:
                        # Maybe a dict-of-mappings: {"desc text": {..}, ...} or {"desc text": "Account Name"}
                        candidates = []
                        for k, v in data.items():
                            if isinstance(v, dict):
                                acc = v.get("account_name") or v.get("account") or ""
                                candidates.append({
                                    "desc_norm": str(k),
                                    "account_name": str(acc),
                                    "is_travel": bool(v.get("is_travel", False)),
                                    "rationale": str(v.get("rationale", "")),
                                })
                            elif isinstance(v, str):
                                candidates.append({
                                    "desc_norm": str(k),
                                    "account_name": v,
                                    "is_travel": False,
                                    "rationale": "",
                                })
                        data = candidates
        if not isinstance(data, list):
            raise ValueError("Top-level JSON is not a list or convertible list")
        cleaned: list[dict[str, Any]] = []
        for row in data:
            if not isinstance(row, dict):
                continue
            desc_norm = str(row.get("desc_norm", "")).strip()
            account_name = str(row.get("account_name", "")).strip()
            is_travel = bool(row.get("is_travel", False))
            not_sure = bool(row.get("not_sure", False))
            guessed_store = str(row.get("guessed_store", "")).strip()
            rationale = str(row.get("rationale", "")).strip()
            if desc_norm and account_name:
                cleaned.append(
                    {
                        "desc_norm": desc_norm,
                        "account_name": account_name,
                        "is_travel": is_travel,
                        "not_sure": not_sure,
                        "guessed_store": guessed_store,
                        "rationale": rationale,
                    }
                )
        logger.info("Parsed %d LLM mappings", len(cleaned))
        return cleaned
    except Exception:
        logger.exception("Failed to parse LLM JSON output")
        return []


def prepare_coa_vectorizer(coa_df: pd.DataFrame) -> tuple[TfidfVectorizer, Any]:
    """Fit a TF-IDF vectorizer on COA text fields and return (vectorizer, matrix)."""
    cols = [c for c in ["Account Name", "Type", "Description"] if c in coa_df.columns]
    text_series = coa_df[cols].fillna("").astype(str).agg(" ".join, axis=1)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(text_series)
    logger.info("Prepared TF-IDF vectorizer: rows=%d, vocab_size=%d", matrix.shape[0], matrix.shape[1])
    return vectorizer, matrix


def find_best_account_rows(
    vectorizer: TfidfVectorizer,
    matrix: Any,
    query_text: str,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Return [(row_index, score), ...] for the best COA rows matching the query text."""
    query_vec = vectorizer.transform([query_text])
    sims = cosine_similarity(query_vec, matrix)[0]
    best_idx = np.argsort(-sims)[:top_k]
    return [(int(i), float(sims[i])) for i in best_idx]


def build_mapping_matches_df(
    llm_mappings: Iterable[dict[str, Any]],
    coa_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix: Any,
    top_k: int = 3,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mapping in llm_mappings:
        # We now expect LLM to propose an account_name directly; still compute similarity for auditing
        proposed_account = mapping["account_name"]
        query_text = proposed_account
        matches = find_best_account_rows(vectorizer, matrix, query_text, top_k=top_k)
        for rank, (idx, score) in enumerate(matches, start=1):
            row = {
                "desc_norm": mapping["desc_norm"],
                "proposed_account_name": proposed_account,
                "is_travel": mapping["is_travel"],
                "not_sure": mapping.get("not_sure", False),
                "guessed_store": mapping.get("guessed_store", ""),
                "rationale": mapping["rationale"],
                "match_rank": rank,
                "match_score": score,
                "coa_index": idx,
                "coa_account_name": coa_df.iloc[idx]["Account Name"],
                "coa_type": coa_df.iloc[idx].get("Type", ""),
                "coa_description": coa_df.iloc[idx].get("Description", ""),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    logger.info("Built mapping_matches_df with %d rows", len(df))
    return df


def attach_best_account_to_transactions(
    missing_in_gl: pd.DataFrame,
    desc_norms: list[str],
    mapping_matches_df: pd.DataFrame,
    coa_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix: Any,
    use_fallback: bool = False,
    similarity_threshold: float = 0.0,
) -> pd.DataFrame:
    """Add `keyword` and `best_account_name` columns to the transactions DataFrame."""
    if mapping_matches_df.empty:
        logger.info("No mapping matches; returning original transactions")
        return missing_in_gl.copy()

    best_match = mapping_matches_df.sort_values(["desc_norm", "match_rank"]).drop_duplicates("desc_norm", keep="first")
    desc_to_account = dict(zip(best_match["desc_norm"], best_match["coa_account_name"]))
    desc_to_rationale = dict(zip(best_match["desc_norm"], best_match["rationale"]))
    desc_to_is_travel = dict(zip(best_match["desc_norm"], best_match["is_travel"]))
    desc_to_not_sure = dict(zip(best_match["desc_norm"], best_match.get("not_sure", pd.Series(dtype=bool))))
    desc_to_store = dict(zip(best_match["desc_norm"], best_match.get("guessed_store", pd.Series(dtype=str))))

    out = missing_in_gl.copy()
    norm = out["Desc_norm"].astype(str).str.strip().str.lower()
    out["desc_norm_normed"] = norm
    out["best_account_name"] = out["desc_norm_normed"].map(desc_to_account)
    out["is_travel"] = out["desc_norm_normed"].map(desc_to_is_travel)
    out["not_sure"] = out["desc_norm_normed"].map(desc_to_not_sure)
    out["guessed_store"] = out["desc_norm_normed"].map(desc_to_store)
    out["rationale"] = out["desc_norm_normed"].map(desc_to_rationale)
    attached = out["best_account_name"].notna().sum()
    logger.info("Attached best_account_name to %d/%d transactions via LLM mappings", attached, len(out))

    # Optional fallback: direct TF-IDF mapping for blanks
    if use_fallback:
        missing_mask = out["best_account_name"].isna()
        if missing_mask.any():
            to_fill = out.loc[missing_mask, "desc_norm_normed"].fillna("").astype(str).tolist()
            if to_fill:
                query_vec = vectorizer.transform(to_fill)
                sims = cosine_similarity(query_vec, matrix)
                best_indices = np.argmax(sims, axis=1)
                best_scores = sims[np.arange(sims.shape[0]), best_indices]
                filled = 0
                for i, (row_idx, best_i, score) in enumerate(zip(out.index[missing_mask], best_indices, best_scores)):
                    if score >= similarity_threshold:
                        out.at[row_idx, "best_account_name"] = coa_df.iloc[int(best_i)]["Account Name"]
                        filled += 1
                logger.info(
                    "Fallback TF-IDF attached %d additional accounts (threshold=%.3f)",
                    filled,
                    similarity_threshold,
                )
    return out


def end_to_end_categorization(limit_keywords: int | None = 100) -> dict[str, Any]:
    """Full pipeline to support the API: load, extract, LLM map, match, attach."""
    coa_df, missing_in_gl = load_csvs()
    desc_norms = extract_unique_descs(missing_in_gl["Desc_norm"], limit=limit_keywords)
    prompt = build_mapping_prompt(desc_norms, coa_df)
    raw = run_llm_mapping(prompt)
    mappings = parse_llm_json(raw)
    vectorizer, matrix = prepare_coa_vectorizer(coa_df)
    matches_df = build_mapping_matches_df(mappings, coa_df, vectorizer, matrix, top_k=3)
    enriched_missing = attach_best_account_to_transactions(
        missing_in_gl,
        desc_norms,
        matches_df,
        coa_df,
        vectorizer,
        matrix,
        use_fallback=False,
        similarity_threshold=0.0,
    )

    summary = {
        "descriptions": desc_norms,
        "llm_mappings": mappings,
        "mapping_matches": matches_df,
        "transactions_with_account": enriched_missing,
    }
    logger.info(
        "Pipeline summary: descriptions=%d, mappings=%d, matches=%d, txns=%d",
        len(summary["descriptions"]),
        len(summary["llm_mappings"]),
        len(summary["mapping_matches"]),
        len(summary["transactions_with_account"]),
    )
    return summary


