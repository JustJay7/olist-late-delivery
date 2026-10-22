import pandas as pd
import pytest
from src.integration import integrate_features, model_features, APPROVAL_FEATURES


def test_join_preserves_target_and_rows(tmp_path):
    pop = pd.DataFrame({'order_id': ['a', 'b'], 'is_late': [0, 1],
                        'order_delivered_customer_date': ['2020-01-01', '2020-01-03']})
    merged = pd.DataFrame({'order_id': ['b', 'a'], 'is_late': [0, 1],
                           **{col: [2, 1] for col in APPROVAL_FEATURES}})
    file = tmp_path / 'merged.csv'; merged.to_csv(file, index=False)
    out = integrate_features(pop, file)
    assert out['is_late'].tolist() == [0, 1]
    assert len(out) == 2 and out['item_count'].tolist() == [1, 2]
    assert 'is_late' not in model_features(out)
    assert 'order_delivered_customer_date' not in model_features(out)


def test_fails_on_missing_orders(tmp_path):
    pop = pd.DataFrame({'order_id': ['a', 'b'], 'is_late': [0, 1]})
    merged = pd.DataFrame({'order_id': ['a'], **{col: [1] for col in APPROVAL_FEATURES}})
    file = tmp_path / 'merged.csv'; merged.to_csv(file, index=False)
    with pytest.raises(ValueError, match='not found'):
        integrate_features(pop, file)
