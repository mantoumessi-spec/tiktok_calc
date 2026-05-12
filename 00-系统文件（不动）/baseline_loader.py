"""
baseline_loader.py
功能：加载去年同期订单数据，计算去年同期订单层面指标
"""

import pandas as pd
import os
from typing import Dict, Optional


def load_last_year_orders(current_start, current_end, base_dir: str) -> Optional[pd.DataFrame]:
    """
    加载去年同期订单数据
    
    Args:
        current_start: 本期开始日期
        current_end: 本期结束日期
        base_dir: 项目根目录
    
    Returns:
        去年同期订单DataFrame，或None（如果文件不存在）
    """
    filepath = os.path.join(base_dir, '01-基础数据（年度更新）', '2025年全年订单底表.xlsx')
    if not os.path.exists(filepath):
        print(f"  去年同期订单底表不存在: {filepath}")
        return None
    
    df = pd.read_excel(filepath, sheet_name=0)
    
    # 去掉表头说明行
    if 'Order Status' in df.columns:
        df = df[df['Order Status'] != 'Current order status.'].copy()
    
    # 时间解析
    df['Created Time'] = pd.to_datetime(df['Created Time'], errors='coerce')
    
    # 计算去年同期起止
    last_year_start = pd.to_datetime(current_start) - pd.DateOffset(years=1)
    last_year_end = pd.to_datetime(current_end) - pd.DateOffset(years=1)
    
    mask = (df['Created Time'] >= last_year_start) & (df['Created Time'] <= last_year_end)
    df = df[mask].copy()
    
    if df.empty:
        print(f"  去年同期无数据 ({last_year_start.date()} ~ {last_year_end.date()})")
        return df
    
    print(f"  去年同期订单: {len(df)}行 ({last_year_start.date()} ~ {last_year_end.date()})")
    
    # 数据清洗（和本期保持一致）
    df['Seller SKU'] = df['Seller SKU'].astype(str).str.strip().str.upper()
    df['SKU ID'] = df['SKU ID'].astype(str).str.strip()
    df['营收'] = pd.to_numeric(df['营收'], errors='coerce').fillna(0)
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)
    
    # 退款标记
    df['Is_Refund'] = df['Order Status'].isin(['Cancelled', 'Canceled'])
    
    # 总营收（和本期逻辑一致：营收列已包含运费）
    df['总营收'] = df['营收']
    
    return df


def calculate_last_year_spu_orders(orders: pd.DataFrame, spu_map: pd.DataFrame) -> pd.DataFrame:
    """
    计算去年同期SPU级订单指标（仅订单层面，无成本/广告/联盟）
    
    Returns:
        DataFrame columns: [
            SPU, 退款前营收, 退款后营收, 退款金额, 退款数量, 退款前销量,
            销售收入, 销售数量, 净销售收入, 净销售数量, 样品数量
        ]
    """
    if orders.empty:
        return pd.DataFrame(columns=[
            'SPU', '退款前营收', '退款后营收', '退款金额', '退款数量', '退款前销量',
            '销售收入', '销售数量', '净销售收入', '净销售数量', '样品数量'
        ])
    
    # SPU映射
    spu_map = spu_map.copy()
    spu_map['SKU'] = spu_map['SKU'].astype(str).str.strip().str.upper()
    sku_to_spu = dict(zip(spu_map['SKU'], spu_map['SPU']))
    orders['SPU'] = orders['Seller SKU'].map(sku_to_spu)
    orders.loc[orders['SPU'].isna(), 'SPU'] = orders.loc[orders['SPU'].isna(), 'Seller SKU']
    
    # 退款前营收 = 所有订单总营收
    gross_revenue = orders.groupby('SPU')['总营收'].sum().reset_index()
    gross_revenue.columns = ['SPU', '退款前营收']
    
    # 正常订单
    normal = orders[(~orders['Is_Refund']) & (orders['总营收'] > 0)]
    normal_agg = normal.groupby('SPU').agg({'总营收': 'sum', 'Quantity': 'sum'}).reset_index()
    normal_agg.columns = ['SPU', '销售收入', '销售数量']
    
    # 退款订单
    refund = orders[orders['Is_Refund']]
    refund_agg = refund.groupby('SPU').agg({'总营收': 'sum', 'Quantity': 'sum'}).reset_index()
    refund_agg.columns = ['SPU', '退款金额', '退款数量']
    
    # 样品订单
    sample = orders[(orders['总营收'] == 0) & (~orders['Is_Refund'])]
    sample_agg = sample.groupby('SPU')['Quantity'].sum().reset_index()
    sample_agg.columns = ['SPU', '样品数量']
    
    # 合并
    result = normal_agg.merge(refund_agg, on='SPU', how='outer')
    result = result.merge(gross_revenue, on='SPU', how='left')
    result = result.merge(sample_agg, on='SPU', how='left')
    result = result.fillna(0)
    
    result['退款后营收'] = result['销售收入']
    result['净销售收入'] = result['销售收入']
    result['净销售数量'] = result['销售数量']
    result['退款前销量'] = result['销售数量'] + result['退款数量']
    
    # 排序
    result = result.sort_values('净销售收入', ascending=False)
    
    return result[[
        'SPU', '退款前营收', '退款后营收', '退款金额', '退款数量', '退款前销量',
        '销售收入', '销售数量', '净销售收入', '净销售数量', '样品数量'
    ]]
