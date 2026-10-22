"""Join the independently built order-level feature table to the locked project population.

Run the merge script first (scripts/build_olist_model_table.py). The model
population and calendar-date label always come from src.data.build_population.
"""
from pathlib import Path
import pandas as pd
from src import config

MERGED_PATH = config.ROOT / 'data' / 'processed' / 'olist_merged_modeling.csv'

# These remain for audit/splitting in the population, not model features.
EXCLUDE_FROM_SOURCE = {
    'customer_id', 'order_purchase_timestamp', 'order_approved_at',
    'order_delivered_customer_date', 'order_estimated_delivery_date', 'is_late',
}

# Explicit whitelist: never silently use newly added, potentially leaky columns.
APPROVAL_FEATURES = [
    'promised_days_from_approval', 'approval_lag_hours', 'purchase_month',
    'purchase_weekday', 'purchase_hour', 'approval_month',
    'customer_zip_code_prefix', 'customer_state', 'item_count',
    'distinct_product_count', 'seller_count', 'total_price', 'total_freight',
    'mean_product_weight_g', 'max_product_weight_g', 'total_product_volume_cm3',
    'mean_distance_km', 'max_distance_km', 'same_state_item_share',
    'any_seller_state', 'primary_product_category', 'freight_to_price_ratio',
]


def integrate_features(population: pd.DataFrame, path: Path = MERGED_PATH) -> pd.DataFrame:
    """Preserve project population/labels/split and attach approval-time features."""
    if not population.order_id.is_unique:
        raise ValueError('Project population has duplicate order IDs')
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Run scripts/build_olist_model_table.py first.')
    merged = pd.read_csv(path, low_memory=False)
    if not merged.order_id.is_unique:
        raise ValueError('Merged source has duplicate order IDs')
    missing = set(APPROVAL_FEATURES) - set(merged.columns)
    if missing:
        raise ValueError(f'Missing approved features: {sorted(missing)}')
    if any(c in population.columns for c in APPROVAL_FEATURES):
        raise ValueError('Population already has merged features; refusing duplicate join')

    out = population.merge(
        merged[['order_id', *APPROVAL_FEATURES]],
        on='order_id', how='left', validate='one_to_one', indicator=True
    )
    unmatched = (out['_merge'] != 'both').sum()
    if unmatched:
        raise ValueError(f'{unmatched} population orders not found in merged source')
    out = out.drop(columns='_merge')
    assert len(out) == len(population)
    assert out[config.TARGET].equals(population[config.TARGET].reset_index(drop=True))
    return out


def model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Whitelisted X. Call *after* splitting; Mia fits preprocessing on train only."""
    missing = set(APPROVAL_FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing features: {sorted(missing)}')
    return frame[APPROVAL_FEATURES].copy()
