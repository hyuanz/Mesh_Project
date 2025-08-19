from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

import logging
from .llm_categorizer import end_to_end_categorization, load_csvs, extract_unique_descs
from .reconciliation import run_reconciliation


class CategorizeRequest(BaseModel):
    limit_keywords: int | None = 20


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Mesh Take Home - Categorization API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your FE origin(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reconcile")
def reconcile() -> dict[str, Any]:
    """Run Part 1 reconciliation and persist outputs.

    Returns counts and record samples for UI consumption.
    """
    result = run_reconciliation()
    return result


@app.post("/categorize")
def categorize(req: CategorizeRequest) -> dict[str, Any]:
    # Important: ensure the OpenAI key is not returned in any payload
    logger.info("/categorize called with limit_keywords=%s", req.limit_keywords)
    result = end_to_end_categorization(limit_keywords=req.limit_keywords)

    # Convert DataFrames to records for JSON response
    def df_to_records(df):
        return df.to_dict(orient="records")

    payload = {
        "descriptions": result["descriptions"],
        "llm_mappings": result["llm_mappings"],
        "mapping_matches": df_to_records(result["mapping_matches"]),
        "transactions_with_account": df_to_records(result["transactions_with_account"]),
    }
    logger.info(
        "Responding: descriptions=%d, mappings=%d, matches=%d, txns=%d",
        len(payload["descriptions"]),
        len(payload["llm_mappings"]),
        len(payload["mapping_matches"]),
        len(payload["transactions_with_account"]),
    )
    return payload


@app.get("/descriptions")
def get_descriptions(limit_keywords: int | None = 20) -> dict[str, Any]:
    """Return only the list of unique description strings (fast preparation phase)."""
    logger.info("/descriptions called with limit_keywords=%s", limit_keywords)
    coa_df, missing_in_gl = load_csvs()
    desc_norms = extract_unique_descs(missing_in_gl["Desc_norm"], limit=limit_keywords)
    return {"descriptions": desc_norms}


def create_app() -> FastAPI:
    return app


