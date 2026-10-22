"""Guard against committed merged data disagreeing with the project labels."""
import pandas as pd
import pytest
from src import config
from src.data import load_raw, build_population


def test_merged_dataset_matches_project_labels():
    path = config.ROOT / 'data' / 'processed' / 'olist_merged_modeling.csv'
    if not path.exists():
        pytest.skip('Generate processed data first: python scripts/build_olist_model_table.py')
    merged = pd.read_csv(path, usecols=['order_id', 'is_late'])
    population, _ = build_population(load_raw('orders'))
    joined = population[['order_id', 'is_late']].merge(
        merged, on='order_id', how='left', validate='one_to_one',
        suffixes=('_expected', '_merged'), indicator=True,
    )
    assert joined['_merge'].eq('both').all(), 'Some filtered orders are missing from merged data'
    assert joined['is_late_expected'].eq(joined['is_late_merged']).all(), 'Merged target differs from project calendar-date definition'
    assert len(merged) == merged.order_id.nunique(), 'Merged data must contain one row per order'
