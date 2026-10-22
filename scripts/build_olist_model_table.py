"""Build one-row-per-order Olist dataset for late-delivery prediction at order approval.
Run: python build_olist_model_table.py --data-dir /path/to/nine/csvs --output merged_olist_orders.csv
Uses orders, customers, items, products, sellers, category translations, geolocation.
Reviews are excluded (post-delivery). Payments excluded by default because exact availability
at approval isn't documented; enable with --include-payments only if justified.
Post-approval fields are used ONLY to construct labels, never predictive features.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

NAMES = {
 'orders':'olist_orders_dataset.csv', 'customers':'olist_customers_dataset.csv',
 'items':'olist_order_items_dataset.csv','products':'olist_products_dataset.csv',
 'sellers':'olist_sellers_dataset.csv','translation':'product_category_name_translation.csv',
 'geo':'olist_geolocation_dataset.csv','payments':'olist_order_payments_dataset.csv',
}

def build(folder:Path, include_payments=False):
    def read(k): return pd.read_csv(folder/NAMES[k],low_memory=False)
    orders=read('orders'); customers=read('customers'); items=read('items')
    products=read('products'); sellers=read('sellers'); translation=read('translation')
    geo=read('geo')
    for c in ['order_purchase_timestamp','order_approved_at','order_delivered_customer_date','order_estimated_delivery_date']:
        orders[c]=pd.to_datetime(orders[c],errors='coerce')
    # Scope: prediction is made at approval. Labels can only be calculated on completed deliveries.
    orders=orders.loc[orders.order_status.eq('delivered') & orders.order_approved_at.notna() & orders.order_delivered_customer_date.notna() & orders.order_estimated_delivery_date.notna()].copy()
    orders['is_late']=(orders.order_delivered_customer_date>orders.order_estimated_delivery_date).astype('int8')
    orders['promised_days_from_approval']=(orders.order_estimated_delivery_date-orders.order_approved_at).dt.total_seconds()/86400
    orders['approval_lag_hours']=(orders.order_approved_at-orders.order_purchase_timestamp).dt.total_seconds()/3600
    orders['purchase_month']=orders.order_purchase_timestamp.dt.month
    orders['purchase_weekday']=orders.order_purchase_timestamp.dt.dayofweek
    orders['purchase_hour']=orders.order_purchase_timestamp.dt.hour
    orders['approval_month']=orders.order_approved_at.dt.month
    # Keep outcome dates in output for audits/temporal holdouts, never as model features.
    orders=orders.drop(columns=['order_delivered_carrier_date','order_status'])
    customers=customers[['customer_id','customer_zip_code_prefix','customer_state']].drop_duplicates('customer_id')
    out=orders.merge(customers,on='customer_id',how='left',validate='many_to_one')
    products=products.merge(translation.drop_duplicates('product_category_name'),on='product_category_name',how='left',validate='many_to_one')
    items=items.merge(products,on='product_id',how='left',validate='many_to_one')
    items=items.merge(sellers[['seller_id','seller_zip_code_prefix','seller_state']].drop_duplicates('seller_id'),on='seller_id',how='left',validate='many_to_one')
    items['product_volume_cm3']=items.product_length_cm*items.product_height_cm*items.product_width_cm
    items['same_state_seller_customer']=np.nan # calculated below using customer state
    items=items.drop(columns=['same_state_seller_customer'])
    # Geographic prefix can appear many times: calculate one centroid per ZIP prefix first.
    centers=geo.groupby('geolocation_zip_code_prefix',as_index=False).agg(lat=('geolocation_lat','median'),lon=('geolocation_lng','median'))
    seller_geo=centers.rename(columns={'geolocation_zip_code_prefix':'seller_zip_code_prefix','lat':'seller_lat','lon':'seller_lon'})
    customer_geo=centers.rename(columns={'geolocation_zip_code_prefix':'customer_zip_code_prefix','lat':'customer_lat','lon':'customer_lon'})
    items=items.merge(seller_geo,on='seller_zip_code_prefix',how='left',validate='many_to_one')
    # Link customer ZIP to each order item for distance aggregation, but never output item-level duplicates.
    loc=out[['order_id','customer_zip_code_prefix','customer_state']]
    items=items.merge(loc,on='order_id',how='inner',validate='many_to_one')
    items=items.merge(customer_geo,on='customer_zip_code_prefix',how='left',validate='many_to_one')
    a=np.radians(items['seller_lat']); b=np.radians(items['customer_lat'])
    da=a-b; dl=np.radians(items['seller_lon']-items['customer_lon'])
    h=np.sin(da/2)**2+np.cos(a)*np.cos(b)*np.sin(dl/2)**2
    items['distance_km']=6371*2*np.arcsin(np.sqrt(h.clip(0,1)))
    items['same_state']=np.where(items.seller_state.notna() & items.customer_state.notna(),(items.seller_state==items.customer_state).astype(float),np.nan)
    # Single-category flag: multiple product categories are summarized by primary (highest total price).
    cat=(items.assign(_price=items.price.fillna(0)).sort_values(['order_id','_price'],ascending=[True,False])
         .drop_duplicates('order_id')[['order_id','product_category_name_english']]
         .rename(columns={'product_category_name_english':'primary_product_category'}))
    by_order=items.groupby('order_id',as_index=False).agg(
        item_count=('order_item_id','size'),distinct_product_count=('product_id','nunique'),
        seller_count=('seller_id','nunique'),total_price=('price','sum'),total_freight=('freight_value','sum'),
        mean_product_weight_g=('product_weight_g','mean'),max_product_weight_g=('product_weight_g','max'),
        total_product_volume_cm3=('product_volume_cm3','sum'),mean_distance_km=('distance_km','mean'),
        max_distance_km=('distance_km','max'),same_state_item_share=('same_state','mean'),
        any_seller_state=('seller_state','first'))
    by_order=by_order.merge(cat,on='order_id',how='left',validate='one_to_one')
    out=out.merge(by_order,on='order_id',how='left',validate='one_to_one')
    out['freight_to_price_ratio']=out.total_freight/out.total_price.replace(0,np.nan)
    if include_payments:
        pay=read('payments')
        pay_summary=pay.groupby('order_id',as_index=False).agg(payment_count=('payment_sequential','size'),
            total_payment_value=('payment_value','sum'),max_installments=('payment_installments','max'))
        main_pay=pay.sort_values(['order_id','payment_value'],ascending=[True,False]).drop_duplicates('order_id')[['order_id','payment_type']].rename(columns={'payment_type':'primary_payment_type'})
        out=out.merge(pay_summary.merge(main_pay,on='order_id',validate='one_to_one'),on='order_id',how='left',validate='one_to_one')
    assert out.order_id.is_unique, 'Unexpected duplicate order IDs'
    print('Orders:',len(out),'| late:',int(out.is_late.sum()),'| late rate:',f'{out.is_late.mean():.2%}')
    print('Orders lacking item aggregates:',int(out.item_count.isna().sum()))
    print('Orders lacking customer state:',int(out.customer_state.isna().sum()))
    print('Orders lacking usable distance:',int(out.mean_distance_km.isna().sum()))
    print('Number of columns:',len(out.columns))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('.'))
    p.add_argument('--output',type=Path,default=Path('merged_olist_orders.csv'))
    p.add_argument('--include-payments',action='store_true',help='Only if payment data is demonstrably available at approval')
    a=p.parse_args()
    d=build(a.data_dir,a.include_payments)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    d.to_csv(a.output,index=False)
    print('Saved:',a.output)
    print('IMPORTANT: drop order_id, customer_id, purchase/approval raw timestamps, actual delivery date,')
    print('and estimated delivery date before model fitting (use engineered promise window instead).')
    print('For evaluation use a chronological split by order_approved_at, then never use it as a feature.')
