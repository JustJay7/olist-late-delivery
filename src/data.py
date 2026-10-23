"""Loading, target definition, population and train/test split.

These functions encode the problem definition (DECISIONS.md D1 to D7). Feature
engineering lives in src/features.py and must start from build_population().
"""
import pandas as pd

from src import config


def load_raw(name: str) -> pd.DataFrame:
    """Read one raw Olist table by its short name (see config.RAW_FILES)."""
    if name not in config.RAW_FILES:
        raise KeyError(f"Unknown table '{name}'. Options: {sorted(config.RAW_FILES)}")
    path = config.RAW_DIR / config.RAW_FILES[name]
    if not path.exists():
        path = path.with_name(path.name + ".gz")  # raw tables are committed gzipped (D12)
    if not path.exists():
        raise FileNotFoundError(f"Missing raw file: {path}")
    parse = config.ORDER_DATE_COLS if name == "orders" else None
    return pd.read_csv(path, parse_dates=parse)


def make_target(orders: pd.DataFrame) -> pd.Series:
    """1 if delivered on a later calendar day than promised, else 0.

    The promised date has no time of day (always 00:00), so comparing calendar
    days avoids counting an order delivered on the promised day as late.
    """
    delivered_day = orders["order_delivered_customer_date"].dt.normalize()
    if delivered_day.isna().any():
        raise ValueError("make_target needs a delivery date on every row; filter first.")
    return (delivered_day > orders["order_estimated_delivery_date"]).astype(int)


def build_population(orders: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the population rules and add the target.

    Returns the modelling population and a funnel table (rows kept at each step),
    which the README reports so every exclusion is visible.
    """
    funnel = [("Raw orders", len(orders))]

    df = orders[orders["order_status"] == "delivered"]
    funnel.append(("Status 'delivered'", len(df)))

    df = df.dropna(subset=["order_approved_at", "order_delivered_customer_date"])
    funnel.append(("Has approval and delivery dates", len(df)))

    impossible = df["order_delivered_customer_date"] < df["order_purchase_timestamp"]
    df = df[~impossible]
    funnel.append(("Delivery not before purchase", len(df)))

    df = df[df["order_purchase_timestamp"] >= config.MIN_PURCHASE_DATE]
    funnel.append((f"Purchased from {config.MIN_PURCHASE_DATE.date()}", len(df)))

    last_promise = config.EFFECTIVE_SNAPSHOT - pd.Timedelta(days=config.CENSOR_BUFFER_DAYS)
    df = df[df["order_estimated_delivery_date"] <= last_promise]
    funnel.append((f"Promised on or before {last_promise.date()} (censoring rule)", len(df)))

    df = df.copy()
    df[config.TARGET] = make_target(df)
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

    funnel_df = pd.DataFrame(funnel, columns=["step", "orders"])
    return df, funnel_df


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on orders purchased before TEST_START, test on the rest.

    The test set is touched exactly once, after the final model is chosen.
    """
    is_test = df["order_purchase_timestamp"] >= config.TEST_START
    train, test = df[~is_test].copy(), df[is_test].copy()
    if train.empty or test.empty:
        raise ValueError("Time split produced an empty side; check TEST_START.")
    return train, test
