from pathlib import Path
import pandas as pd
import re
import sys
from typing import List, Tuple, Optional

# --------------------------
# Configuration
# --------------------------
BASE_DIR = Path(__file__).resolve().parents[2]  # Adjust if running from different location
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------
# Utilities
# --------------------------
def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV safely with dtype=str to avoid mixed-type surprises."""
    path = Path(path)
    if not path.exists():
        print(f"[WARN] File not found: {path}. Returning empty DataFrame.")
        return pd.DataFrame()
    try:
        # Force str reading to avoid mixed-dtype inference issues
        df = pd.read_csv(path, dtype=str, low_memory=False, **kwargs)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        return pd.DataFrame()
    # Normalize + dedupe column names right after read
    df = normalize_and_dedupe_columns(df)
    return df


def normalize_and_dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names (strip, lowercase, replace spaces/dots) and ensure unique names.
    Returns a copy with updated columns.
    """
    if df.empty:
        return df

    # Normalize: strip, collapse whitespace, replace punctuation -> underscore, lowercase
    new_cols = []
    for c in df.columns:
        c_clean = str(c).strip()
        # collapse all whitespace to single space first, then replace with underscore
        c_clean = re.sub(r"\s+", " ", c_clean)
        c_clean = re.sub(r"[^\w\s]", "_", c_clean)  # replace punctuation with _
        c_clean = c_clean.replace(" ", "_")
        c_clean = c_clean.lower()
        new_cols.append(c_clean)

    # Deduplicate by appending suffix for repeats
    seen = {}
    out_cols = []
    for c in new_cols:
        if c not in seen:
            seen[c] = 0
            out_cols.append(c)
        else:
            seen[c] += 1
            out_cols.append(f"{c}_{seen[c]}")

    df = df.copy()
    df.columns = out_cols
    return df


def resolve_single_column(df: pd.DataFrame, logical_name: str) -> pd.Series:
    """
    Given a DataFrame and a logical_name (normalized form, e.g., 'cost_price'),
    find all matching columns (exact match or variants) and return a single Series:
    - If there is one match -> that Series
    - If multiple -> bfill across axis=1 to pick first non-null per row
    - If none -> returns a Series of NaNs
    """
    if df.empty:
        return pd.Series(dtype=float)

    # match columns by normalized equality or contains
    matches = [c for c in df.columns if c == logical_name or logical_name in c]
    if not matches:
        # try looser match: contains keywords
        matches = [c for c in df.columns if logical_name in c]
    if not matches:
        # fallback: return empty series same length as df
        return pd.Series([pd.NA] * len(df), index=df.index)

    if len(matches) == 1:
        return df[matches[0]]
    # multiple matches -> merge
    merged = df[matches].bfill(axis=1).iloc[:, 0]
    return merged


def series_drop_text_headers(s: pd.Series) -> pd.Series:
    """
    Remove rows that are likely to be header/summary rows embedded in file,
    by dropping any values that look alphabetic-only or contain common tokens.
    """
    if s.empty:
        return s

    # Convert to string and strip
    s_str = s.astype(str).str.strip()
    # Mark rows that look like header lines: contain letters and short words like 'stock', 'total', 'summary'
    mask_texty = s_str.str.contains(r"[A-Za-z]", na=False)
    # Also drop rows that are exactly 'nan', 'none', empty after stripping
    mask_empty = s_str.isin(["", "nan", "none", "na", "n/a"])
    # Build final mask of good rows (keep where NOT texty and not empty)
    keep_mask = ~(mask_texty | mask_empty)
    # For numeric-ish columns there may be strings like '411Rs 0.15 Per Day15.5' -> we'll let numeric cleaner handle it,
    # but pure alphabetic header rows should be removed.
    return s[keep_mask]


def clean_numeric_series(input_col) -> pd.Series:
    """
    Robustly convert an input Series (or DataFrame of duplicate columns) into numeric floats.
    Steps:
     - If DataFrame passed, merge duplicates using bfill
     - Strip currency symbols, commas, /-, and words like 'per day'
     - Remove pure alphabetic header rows
     - Coerce to numeric with pd.to_numeric(errors='coerce'), fillna(0)
    Returns float Series.
    """
    # Accept either Series or DataFrame (in case of duplicate columns)
    if isinstance(input_col, pd.DataFrame):
        # merge duplicates by taking first non-null
        s = input_col.bfill(axis=1).iloc[:, 0]
    else:
        s = input_col

    # If series is empty return numeric empty series
    if s is None or s.empty:
        return pd.Series(dtype=float)

    s = s.astype(str).str.strip()

    # Remove obvious header/text-only rows first (where they are pure words)
    s = s[~s.str.fullmatch(r"[A-Za-z\s\-/_.]*", na=False)].reindex(s.index)

    # Remove common tokens and currency symbols while keeping digits, dots, hyphen
    # Replace commas, currency symbols, sequences like 'rs' or 'per' or 'per day'
    s_clean = s.str.replace(r"(?i)rs|₹|\$|,|/|-{1,2}|per day|per_year|per_month|per|/day", "", regex=True)
    s_clean = s_clean.str.replace(r"[^\d.\-]", "", regex=True)  # keep digits, dot, minus
    s_clean = s_clean.str.replace(r"^\.+", "", regex=True)  # leading dots removal
    s_clean = s_clean.str.strip()
    # Coerce to numeric
    numeric = pd.to_numeric(s_clean, errors="coerce").fillna(0.0)
    return numeric


def safe_to_datetime(s: pd.Series) -> pd.Series:
    """Parse a date series with fallback; return datetime series (NaT if unparseable)."""
    if s is None or s.empty:
        return pd.Series(dtype="datetime64[ns]")
    # Try common ISO parse first, else fallback to pandas parser
    try:
        return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    except Exception:
        return pd.to_datetime(s, errors="coerce")


def save_df(df: pd.DataFrame, fname: str):
    out_path = OUT_DIR / fname
    try:
        df.to_csv(out_path, index=False)
        print(f"[SAVED] {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save {out_path}: {e}")


# --------------------------
# Source Processors
# --------------------------
def process_amazon(path: Path) -> pd.DataFrame:
    print("Processing Amazon Sales...")
    df = read_csv_safe(path)
    if df.empty:
        return pd.DataFrame()

    # Common column names to attempt to resolve
    # We expect: order id, date, status, sku, category, size, qty, amount, ship_city, ship_state
    # Use resolve_single_column to be tolerant to variants/duplicates
    col_map = {
        "order_id": ["order_id", "order", "orderid"],
        "date": ["date", "order_date"],
        "status": ["status"],
        "sku": ["sku", "sku_code", "sku_code_1"],
        "category": ["category"],
        "size": ["size"],
        "qty": ["qty", "quantity", "pcs", "pcs_1"],
        "amount": ["amount", "gross_amt", "gross_amount", "final_amt"],
        "ship_city": ["ship_city", "ship-city", "ship_city_1", "shipcity"],
        "ship_state": ["ship_state", "ship-state", "ship_state_1", "shipstate"],
    }

    out = pd.DataFrame(index=df.index)

    for out_col, candidates in col_map.items():
        # pick first match for candidates list
        found = None
        for cand in candidates:
            if cand in df.columns:
                found = cand
                break
        if found:
            out[out_col] = df[found]
        else:
            out[out_col] = pd.NA

    # Parse date, numeric cleaning
    out["date"] = safe_to_datetime(out["date"])
    out["amount"] = clean_numeric_series(out["amount"])
    # Quantity may include strings; coerce safely
    out["qty"] = pd.to_numeric(out["qty"].astype(str).str.replace(r"[^\d.-]", "", regex=True), errors="coerce").fillna(0).astype(int)

    out["channel"] = "amazon"

    # Reorder & rename to canonical casing
    out = out.rename(columns={
        "order_id": "Order ID", "date": "Date", "status": "Status", "sku": "SKU",
        "category": "Category", "size": "Size", "qty": "Qty", "amount": "Amount",
        "ship_city": "ship-city", "ship_state": "ship-state", "channel": "Channel"
    })[
        ["Order ID", "Date", "Status", "SKU", "Category", "Size", "Qty", "Amount", "ship-city", "ship-state", "Channel"]
    ]
    return out


def process_international(path: Path) -> pd.DataFrame:
    print("Processing International Sales...")
    df = read_csv_safe(path)
    if df.empty:
        return pd.DataFrame()

    # Try to find likely columns
    # Common raw header examples: date, customer, sku, pcs, gross amt
    possible_date = next((c for c in df.columns if c in ("date", "dt", "dated", "order_date")), None)
    possible_customer = next((c for c in df.columns if "customer" in c), None)
    possible_sku = next((c for c in df.columns if c in ("sku", "item_code", "sku_code")), None)
    possible_qty = next((c for c in df.columns if c in ("pcs", "qty", "quantity")), None)
    possible_amount = next((c for c in df.columns if "amount" in c or "gross" in c), None)

    # Build working frame
    working = pd.DataFrame(index=df.index)
    working["date"] = df[possible_date] if possible_date in df.columns else pd.NA
    working["customer_id"] = df[possible_customer] if possible_customer in df.columns else pd.NA
    working["sku"] = df[possible_sku] if possible_sku in df.columns else pd.NA
    working["qty"] = df[possible_qty] if possible_qty in df.columns else pd.NA
    working["amount_raw"] = df[possible_amount] if possible_amount in df.columns else pd.NA

    # Remove embedded header/summary rows where 'amount_raw' is pure alphabetic (like 'Stock' or 'Total')
    # Keep rows where amount_raw contains at least one digit
    if "amount_raw" in working.columns:
        amount_str = working["amount_raw"].astype(str)
        mask_has_digit = amount_str.str.contains(r"\d", na=False)
        working = working[mask_has_digit].copy()

    working["date"] = safe_to_datetime(working["date"])
    working["amount"] = clean_numeric_series(working["amount_raw"])
    working["qty"] = pd.to_numeric(working["qty"].astype(str).str.replace(r"[^\d.-]", "", regex=True), errors="coerce").fillna(0).astype(int)

    working["channel"] = "international"
    # Enforce consistent output columns
    out = working.rename(columns={"date": "Date", "customer_id": "Customer_ID", "sku": "SKU", "qty": "Qty", "amount": "Amount"})
    out = out[["Date", "Customer_ID", "SKU", "Qty", "Amount", "channel"]].rename(columns={"channel": "Channel"})
    return out


def process_inventory(path: Path) -> pd.DataFrame:
    print("Processing Inventory...")
    df = read_csv_safe(path)
    if df.empty:
        return pd.DataFrame()

    # find columns
    sku_col = next((c for c in df.columns if "sku" in c), None)
    stock_col = next((c for c in df.columns if "stock" in c or "quantity" in c), None)
    cat_col = next((c for c in df.columns if "category" in c), None)
    size_col = next((c for c in df.columns if "size" in c), None)
    color_col = next((c for c in df.columns if "color" in c), None)

    out = pd.DataFrame()
    out["SKU"] = df[sku_col] if sku_col in df.columns else pd.NA
    out["Stock_Quantity"] = clean_numeric_series(df[stock_col] if stock_col in df.columns else pd.Series(dtype=float))
    # Stock likely to be integer
    out["Stock_Quantity"] = out["Stock_Quantity"].astype(int)
    out["Category"] = df[cat_col] if cat_col in df.columns else pd.NA
    out["Size"] = df[size_col] if size_col in df.columns else pd.NA
    out["Color"] = df[color_col] if color_col in df.columns else pd.NA

    return out[["SKU", "Stock_Quantity", "Category", "Size", "Color"]]


def process_pricing(pricing_files: List[Tuple[Path, str]]) -> pd.DataFrame:
    print("Processing Pricing & Cost History...")
    frames = []

    for path, month_tag in pricing_files:
        df = read_csv_safe(path)
        if df.empty:
            continue

        # Heuristics to find relevant columns (SKU, Category, Cost/TP, Amazon MRP, Final MRP)
        sku_col = next((c for c in df.columns if c == "sku" or "sku" in c), None)
        cat_col = next((c for c in df.columns if "category" in c), None)
        # find cost-like column: tp, tp_1, cost, tp_1 etc.
        cost_col = next((c for c in df.columns if c.startswith("tp") or "cost" in c), None)
        amazon_mrp_col = next((c for c in df.columns if "amazon" in c and "mrp" in c), None)
        final_mrp_col = next((c for c in df.columns if "final" in c and "mrp" in c), None)

        # fallback if not found: try broad matches
        if sku_col is None:
            sku_col = next((c for c in df.columns if "sku" in c), None)
        if cost_col is None:
            cost_col = next((c for c in df.columns if "tp" in c or "cost" in c), None)

        # Safely extract / merge duplicates
        sku_s = resolve_single_column(df, "sku")
        cat_s = resolve_single_column(df, "category")
        cost_s = resolve_single_column(df, "cost_price") if "cost_price" in df.columns else (df[cost_col] if cost_col in df.columns else resolve_single_column(df, "tp"))
        amazon_mrp_s = resolve_single_column(df, "amazon_mrp") if "amazon_mrp" in df.columns else (df[amazon_mrp_col] if amazon_mrp_col in df.columns else pd.Series(dtype=float))
        final_mrp_s = resolve_single_column(df, "final_mrp") if "final_mrp" in df.columns else (df[final_mrp_col] if final_mrp_col in df.columns else pd.Series(dtype=float))

        out = pd.DataFrame({
            "SKU": sku_s,
            "Category": cat_s,
            "Cost_Price_raw": cost_s,
            "Amazon_MRP_raw": amazon_mrp_s,
            "Final_MRP_raw": final_mrp_s
        })

        out["Cost_Price"] = clean_numeric_series(out["Cost_Price_raw"])
        out["Amazon_MRP"] = clean_numeric_series(out["Amazon_MRP_raw"])
        out["Final_MRP"] = clean_numeric_series(out["Final_MRP_raw"])
        out["Report_Month"] = pd.to_datetime(month_tag, errors="coerce")

        out = out[["SKU", "Category", "Cost_Price", "Amazon_MRP", "Final_MRP", "Report_Month"]]
        # ensure unique colnames before append
        out = normalize_and_dedupe_columns(out)
        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["SKU", "Category", "Cost_Price", "Amazon_MRP", "Final_MRP", "Report_Month"])

    # Ensure all frames have unique columns, then concat
    frames = [normalize_and_dedupe_columns(f) for f in frames]
    # Reindex columns union to make concatenation predictable
    all_cols = []
    for f in frames:
        for c in f.columns:
            if c not in all_cols:
                all_cols.append(c)
    frames = [f.reindex(columns=all_cols, fill_value=pd.NA) for f in frames]
    fact_pricing = pd.concat(frames, ignore_index=True)
    # Convert numeric columns if present
    if "cost_price" in fact_pricing.columns:
        fact_pricing["cost_price"] = pd.to_numeric(fact_pricing["cost_price"], errors="coerce").fillna(0.0)
    if "amazon_mrp" in fact_pricing.columns:
        fact_pricing["amazon_mrp"] = pd.to_numeric(fact_pricing["amazon_mrp"], errors="coerce").fillna(0.0)
    if "final_mrp" in fact_pricing.columns:
        fact_pricing["final_mrp"] = pd.to_numeric(fact_pricing["final_mrp"], errors="coerce").fillna(0.0)

    # Rename back to canonical names
    rename_map = {}
    if "cost_price" in fact_pricing.columns:
        rename_map["cost_price"] = "Cost_Price"
    if "amazon_mrp" in fact_pricing.columns:
        rename_map["amazon_mrp"] = "Amazon_MRP"
    if "final_mrp" in fact_pricing.columns:
        rename_map["final_mrp"] = "Final_MRP"
    if "sku" in fact_pricing.columns:
        rename_map["sku"] = "SKU"
    if "category" in fact_pricing.columns:
        rename_map["category"] = "Category"
    if "report_month" in fact_pricing.columns:
        rename_map["report_month"] = "Report_Month"

    fact_pricing = fact_pricing.rename(columns=rename_map)

    # Ensure expected final order if those columns exist
    cols_order = [c for c in ["SKU", "Category", "Cost_Price", "Amazon_MRP", "Final_MRP", "Report_Month"] if c in fact_pricing.columns]
    fact_pricing = fact_pricing[cols_order]
    return fact_pricing


def process_financials(path: Path) -> pd.DataFrame:
    print("Processing Financials...")
    df = read_csv_safe(path)
    if df.empty:
        return pd.DataFrame()

    # Many files have side-by-side tables or header on second row; we attempt heuristics
    # Attempt to find 'particular' and 'amount' columns
    part_cols = [c for c in df.columns if "particular" in c]
    amt_cols = [c for c in df.columns if "amount" in c]

    frames = []
    for p, a in zip(part_cols, amt_cols):
        sub = df[[p, a]].dropna(how="all").copy()
        sub.columns = ["Particular", "Amount"]
        sub["Type"] = "Income" if "income" in p else "Expense"
        frames.append(sub)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        # fallback: attempt melting numeric columns
        numeric_cols = [c for c in df.columns if df[c].str.match(r"^\d+(\.\d+)?$", na=False).any()]
        if numeric_cols:
            melt = df.melt(value_vars=numeric_cols)
            combined = pd.DataFrame({"Particular": melt["variable"], "Amount": melt["value"], "Type": "Income"})
        else:
            # nothing to do
            return pd.DataFrame()

    # Clean amount
    combined["Amount"] = clean_numeric_series(combined["Amount"])
    return combined[["Particular", "Amount", "Type"]]


def process_warehouse(path: Path) -> pd.DataFrame:
    print("Processing Warehouse Comparison...")
    df = read_csv_safe(path)
    if df.empty:
        return pd.DataFrame()

    # Try header names first
    service_col = next((c for c in df.columns if "service" in c), None)
    shiprocket_col = next((c for c in df.columns if "shiprocket" in c), None)
    increff_col = next((c for c in df.columns if "increff" in c), None)

    if service_col and shiprocket_col and increff_col:
        sub = df[[service_col, shiprocket_col, increff_col]].copy()
        sub.columns = ["Service_Type", "Shiprocket_Price", "Increff_Price"]
    else:
        # fallback to positional extraction of first few rows/cols (like earlier approach)
        try:
            sub = df.iloc[1:6, 1:4].copy()
            sub.columns = ["Service_Type", "Shiprocket_Price", "Increff_Price"]
        except Exception:
            return pd.DataFrame()

    sub["Shiprocket_Price"] = clean_numeric_series(sub["Shiprocket_Price"])
    sub["Increff_Price"] = clean_numeric_series(sub["Increff_Price"])
    return sub[["Service_Type", "Shiprocket_Price", "Increff_Price"]]


# --------------------------
# Main runner
# --------------------------
def process_data():
    # File paths (relative to RAW_DIR)
    amazon_path = RAW_DIR / "Amazon Sale Report.csv"
    intl_path = RAW_DIR / "International sale Report.csv"
    inventory_path = RAW_DIR / "Sale Report.csv"
    expense_path = RAW_DIR / "Expense IIGF.csv"
    may22_path = RAW_DIR / "May-2022.csv"
    mar21_path = RAW_DIR / "P  L March 2021.csv"
    warehouse_path = RAW_DIR / "Cloud Warehouse Compersion Chart.csv"

    # Process
    amazon_df = process_amazon(amazon_path)
    intl_df = process_international(intl_path)
    inventory_df = process_inventory(inventory_path)
    financials_df = process_financials(expense_path)
    pricing_df = process_pricing([(may22_path, "2022-05-01"), (mar21_path, "2021-03-01")])
    warehouse_df = process_warehouse(warehouse_path)

    # Save outputs if not empty (and print warnings if empty)
    outputs = [
        (amazon_df, "fact_sales_amazon.csv"),
        (intl_df, "fact_sales_intl.csv"),
        (inventory_df, "dim_inventory.csv"),
        (financials_df, "fact_financials.csv"),
        (pricing_df, "fact_product_pricing.csv"),
        (warehouse_df, "dim_warehouse_pricing.csv"),
    ]

    for df, fname in outputs:
        if df is None or df.empty:
            print(f"[WARN] Output {fname} is empty — file will not be written.")
        else:
            save_df(df, fname)

    print("ETL run complete.")


if __name__ == "__main__":
    process_data()
