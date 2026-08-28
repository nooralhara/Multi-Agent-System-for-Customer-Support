import os
import re
import pandas as pd
from backend.config import DATA_DIR

_orders_df = None


def _load_orders() -> pd.DataFrame:
    global _orders_df
    if _orders_df is None:
        _orders_df = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"), dtype=str)
    return _orders_df


def extract_order_id(text: str) -> str | None:
    """Look for something that looks like an order id, e.g. O0001, #1234."""
    match = re.search(r"\b([A-Za-z]{1,6}\d{3,})\b", text)
    if match:
        return match.group(1)
    match = re.search(r"#(\d+)", text)
    if match:
        return match.group(1)
    return None


def get_order_by_id(order_id: str) -> dict | None:
    df = _load_orders()
    row = df[df["order_id"].str.lower() == order_id.lower()]
    if row.empty:
        return None
    return row.iloc[0].to_dict()