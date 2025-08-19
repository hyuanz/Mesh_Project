## Mesh Take Home — Reconciliation & Categorization

### What the result looks like
See `Mesh Categorization.pdf` in the repository root. It illustrates the expected categorization output and how transactions map to Chart of Accounts (COA) entries. Use it as a visual reference for what the final result should resemble.

### Overview
- **Step 1 — Reconciliation (`backend/reconciliation.py`)**
  - Normalizes bank and general ledger data (date, description, amount) and builds a duplicate-aware `match_key` using an occurrence counter.
  - Finds matches between the bank statement and GL, and identifies two gaps:
    - `missing_in_gl`: transactions present in bank but not in GL (persisted to `data/missing_in_gl.csv`).
    - `missing_in_bank`: transactions present in GL but not in bank.
  - Returns counts and record samples for the UI/API.

- **Step 2 — Categorization (`backend/llm_categorizer.py`)**
  - Loads the COA (`data/sample_chart_of_accounts.csv`) and the `missing_in_gl.csv` produced in Step 1.
  - Extracts unique normalized descriptions and applies simple rule-based mappings for obvious cases (e.g., airlines, hotels, rideshare, meals).
  - Builds a prompt that includes the full COA profile fields and calls an LLM to propose `account_name` choices.
  - Parses the model’s JSON robustly and then computes TF‑IDF similarity over the COA to audit and rank the proposed accounts.
  - Attaches the best account back onto the transactions, producing an enriched table ready for display.


