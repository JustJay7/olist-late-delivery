"""Locked project decisions.

Every notebook and module imports from here instead of hard-coding values,
so a decision changed in DECISIONS.md is changed in exactly one place in code.
Do not edit without Jay's sign-off (see CONVENTIONS.md).
"""
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "reports" / "figures"

RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

ORDER_DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

# ---------------------------------------------------------------- reproducibility
RANDOM_STATE = 42

# ---------------------------------------------------------------- population (D2, D4)
# 2016 holds only a few hundred orders over a few months, too sparse to learn from.
MIN_PURCHASE_DATE = pd.Timestamp("2017-01-01")
# Last day with normal delivery volume; recorded deliveries collapse after 30 Aug 2018.
EFFECTIVE_SNAPSHOT = pd.Timestamp("2018-08-31")
# A late order is only visible if it was delivered before the snapshot. 95% of late
# orders arrive within 31 days of the promised date, so orders promised for the last
# 30 days before the snapshot are dropped to avoid under-counting late deliveries.
CENSOR_BUFFER_DAYS = 30

# ---------------------------------------------------------------- split (D6, D7)
# Train on the past, test on the future, as the model would be used in production.
TEST_START = pd.Timestamp("2018-04-01")
N_CV_SPLITS = 5  # TimeSeriesSplit folds inside the training period

# ---------------------------------------------------------------- evaluation (D8)
TARGET = "is_late"
FLAG_BUDGET = 0.10  # ops can review the riskiest 10% of orders each day
REPORT_BUDGETS = (0.05, 0.10, 0.20)  # recall also shown at these budgets (README A3)

# ---------------------------------------------------------------- leakage guard (D5)
# Known only after approval, so never usable as features at prediction time.
BANNED_FEATURE_COLUMNS = [
    "order_status",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "review_id",
    "review_score",
    "review_comment_title",
    "review_comment_message",
    "review_creation_date",
    "review_answer_timestamp",
]
