"""Edge cases for the target and population rules (run: pytest -q)."""
import pandas as pd
import pytest

from src import config
from src.data import build_population, make_target, time_split


def _orders(rows):
    df = pd.DataFrame(rows)
    for c in config.ORDER_DATE_COLS:
        df[c] = pd.to_datetime(df[c], format="mixed")
    return df


BASE = dict(
    order_id="x", order_status="delivered",
    order_purchase_timestamp="2017-06-01 10:00", order_approved_at="2017-06-01 11:00",
    order_delivered_carrier_date="2017-06-03", order_estimated_delivery_date="2017-06-20",
)


def test_same_day_as_promise_is_on_time():
    df = _orders([{**BASE, "order_delivered_customer_date": "2017-06-20 23:59"}])
    assert make_target(df).tolist() == [0]


def test_next_day_is_late():
    df = _orders([{**BASE, "order_delivered_customer_date": "2017-06-21 00:01"}])
    assert make_target(df).tolist() == [1]


def test_missing_delivery_date_fails_loudly():
    df = _orders([{**BASE, "order_delivered_customer_date": None}])
    with pytest.raises(ValueError):
        make_target(df)


def test_population_rules():
    rows = [
        {**BASE, "order_id": "keep", "order_delivered_customer_date": "2017-06-10"},
        {**BASE, "order_id": "canceled", "order_status": "canceled", "order_delivered_customer_date": None},
        {**BASE, "order_id": "no_approval", "order_approved_at": None, "order_delivered_customer_date": "2017-06-10"},
        {**BASE, "order_id": "too_early", "order_purchase_timestamp": "2016-10-01",
         "order_delivered_customer_date": "2016-10-10"},
        {**BASE, "order_id": "censored", "order_estimated_delivery_date": "2018-08-20",
         "order_delivered_customer_date": "2018-08-15"},
    ]
    pop, funnel = build_population(_orders(rows))
    assert pop["order_id"].tolist() == ["keep"]
    assert funnel["orders"].iloc[0] == 5 and funnel["orders"].iloc[-1] == 1


def test_time_split_has_no_overlap():
    rows = [
        {**BASE, "order_id": "a", "order_delivered_customer_date": "2017-06-10"},
        {**BASE, "order_id": "b", "order_purchase_timestamp": "2018-05-01",
         "order_approved_at": "2018-05-01", "order_estimated_delivery_date": "2018-05-20",
         "order_delivered_customer_date": "2018-05-10"},
    ]
    pop, _ = build_population(_orders(rows))
    train, test = time_split(pop)
    assert train["order_purchase_timestamp"].max() < test["order_purchase_timestamp"].min()
