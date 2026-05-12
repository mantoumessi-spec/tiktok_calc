"""
validation.py
功能：生成数据校验各子表
"""

import pandas as pd
from typing import Dict


def generate_validation_sheets(data: Dict[str, pd.DataFrame], result) -> Dict[str, pd.DataFrame]:
    """生成数据校验各子表，返回字典"""

    # ── 通用数据准备 ──────────────────────────────────────────────
    orders = data.get('orders', pd.DataFrame()).copy()
    if orders.empty:
        return {
            'sku_cost': pd.DataFrame(),
            'affiliate': pd.DataFrame(),
            'ads': pd.DataFrame(),
            'ads_unmapped': pd.DataFrame(),
            'other': pd.DataFrame(),
        }

    orders['Seller SKU'] = orders['Seller SKU'].astype(str).str.strip().str.upper()
    orders['营收'] = pd.to_numeric(orders['营收'], errors='coerce').fillna(0)
    orders['Quantity'] = pd.to_numeric(orders['Quantity'], errors='coerce').fillna(0)
    orders['Is_Refund'] = orders['Order Status'].isin(['Cancelled', 'Canceled'])

    # ── 1. SKU成本完整性 ─────────────────────────────────────────
    normal = orders[(~orders['Is_Refund']) & (orders['营收'] > 0)]
    sku_agg = normal.groupby('Seller SKU').agg({
        'Quantity': 'sum',
        '营收': 'sum'
    }).reset_index()
    sku_agg.columns = ['Seller SKU', '销售数量', '销售收入']

    purchase = data.get('purchase', pd.DataFrame())
    last_mile = data.get('last_mile', pd.DataFrame())
    tariff = data.get('tariff', pd.DataFrame())

    purchase_skus = set()
    if not purchase.empty and 'SKU' in purchase.columns:
        purchase_skus = set(purchase['SKU'].astype(str).str.strip().str.upper().dropna())

    last_mile_skus = set()
    if not last_mile.empty and 'SKU' in last_mile.columns:
        last_mile_skus = set(last_mile['SKU'].astype(str).str.strip().str.upper().dropna())

    tariff_skus = set()
    if not tariff.empty and 'SKU' in tariff.columns:
        tariff_skus = set(tariff['SKU'].astype(str).str.strip().str.upper().dropna())

    rows_sku = []
    for _, row in sku_agg.iterrows():
        sku = row['Seller SKU']
        has_p = '√' if sku in purchase_skus else ''
        has_f = '√' if sku in purchase_skus else ''
        has_l = '√' if sku in last_mile_skus else ''
        has_t = '√' if sku in tariff_skus else ''

        missing = []
        if sku not in purchase_skus:
            missing.append('采购')
        if sku not in purchase_skus:
            missing.append('头程')
        if sku not in last_mile_skus:
            missing.append('尾程')
        if sku not in tariff_skus:
            missing.append('关税')

        missing_str = '、'.join(missing) if missing else '无'
        status = '完整' if not missing else '缺失'

        rows_sku.append({
            'Seller SKU': sku,
            '销售数量': int(row['销售数量']),
            '销售收入': round(row['销售收入'], 2),
            '有采购成': has_p,
            '有头程成': has_f,
            '有尾程成': has_l,
            '有关税成': has_t,
            '成本缺失': missing_str,
            '状态': status,
        })

    df_sku_cost = pd.DataFrame(rows_sku)

    # ── 2. 联盟佣金校验 ─────────────────────────────────────────
    affiliate = data.get('affiliate', pd.DataFrame()).copy()
    if not affiliate.empty:
        affiliate['SKU ID'] = affiliate['SKU ID'].astype(str).str.strip()
        affiliate['Commission'] = (
            pd.to_numeric(affiliate.get('Est. standard commission payment', 0), errors='coerce').fillna(0) +
            pd.to_numeric(affiliate.get('Est. Shop Ads commission payment', 0), errors='coerce').fillna(0)
        )

        total_records = len(affiliate)
        unique_sku_ids = affiliate['SKU ID'].nunique()

        sku_ids_in_orders = set(orders['SKU ID'].astype(str).str.strip().dropna())
        affiliate_sku_ids = set(affiliate['SKU ID'].unique())
        matched = affiliate_sku_ids & sku_ids_in_orders
        unmatched = affiliate_sku_ids - sku_ids_in_orders

        matched_count = len(matched)
        unmatched_count = len(unmatched)
        match_rate = matched_count / unique_sku_ids * 100 if unique_sku_ids > 0 else 0

        raw_total = affiliate['Commission'].sum()
        profit_total = result.spu_profit['联盟佣金'].sum() if not result.spu_profit.empty else 0
        diff = raw_total - profit_total
        diff_rate = diff / raw_total * 100 if raw_total != 0 else 0

        df_affiliate = pd.DataFrame({
            '校验项': ['联盟订单记录数', '唯一SKU ID数', '匹配数', '未匹配数', '匹配率',
                     '原始佣金总额', '利润表佣金总额', '差异金额', '差异率'],
            '结果': [
                total_records,
                unique_sku_ids,
                matched_count,
                unmatched_count,
                f'{match_rate:.1f}%',
                round(raw_total, 2),
                round(profit_total, 2),
                round(diff, 2),
                f'{diff_rate:.2f}%'
            ]
        })
    else:
        df_affiliate = pd.DataFrame({'校验项': [], '结果': []})

    # ── 3. 广告费校验 ───────────────────────────────────────────
    ads = data.get('ads', pd.DataFrame()).copy()
    if not ads.empty and 'Product ID' in ads.columns:
        ads['PID_Clean'] = ads['Product ID'].astype(str).str.strip()
        ads['Cost'] = pd.to_numeric(ads['Cost'], errors='coerce').fillna(0)

        total_records = len(ads)
        unique_pids = ads['PID_Clean'].nunique()

        pid_map = data.get('pid_sku_map', pd.DataFrame())
        mapped_pids = set()
        if not pid_map.empty and 'product id' in pid_map.columns:
            mapped_pids = set(pid_map['product id'].astype(str).str.strip().unique())

        ads_pids = set(ads['PID_Clean'].unique())
        matched = ads_pids & mapped_pids
        unmatched = ads_pids - mapped_pids

        matched_count = len(matched)
        unmatched_count = len(unmatched)
        match_rate = matched_count / unique_pids * 100 if unique_pids > 0 else 0

        total_cost = ads['Cost'].sum()
        mapped_cost = ads[ads['PID_Clean'].isin(mapped_pids)]['Cost'].sum()
        unmapped_cost = ads[~ads['PID_Clean'].isin(mapped_pids)]['Cost'].sum()

        profit_ads = result.spu_profit['广告费'].sum() if not result.spu_profit.empty else 0
        diff = total_cost - profit_ads
        diff_rate = diff / total_cost * 100 if total_cost != 0 else 0

        df_ads = pd.DataFrame({
            '校验项': ['广告记录数', '唯一PID数', '匹配数', '未匹配数', '匹配率',
                     '广告总费用', '已映射费用', '未映射费用', '利润表广告费', '差异金额', '差异率'],
            '结果': [
                total_records,
                unique_pids,
                matched_count,
                unmatched_count,
                f'{match_rate:.1f}%',
                round(total_cost, 2),
                round(mapped_cost, 2),
                round(unmapped_cost, 2),
                round(profit_ads, 2),
                round(diff, 2),
                f'{diff_rate:.2f}%'
            ]
        })

        # 未匹配明细：直接从原始ads提取（使用正确列名）
        unmapped_df = ads[~ads['PID_Clean'].isin(mapped_pids)][['PID_Clean', 'Cost', 'Campaign name']].copy()
        unmapped_df.columns = ['Product ID', '花费', 'Campaign名称']
        # 按PID汇总花费
        unmapped_agg = unmapped_df.groupby(['Product ID', 'Campaign名称'])['花费'].sum().reset_index()
        df_ads_unmapped = unmapped_agg.sort_values('花费', ascending=False)
    else:
        df_ads = pd.DataFrame({'校验项': [], '结果': []})
        df_ads_unmapped = pd.DataFrame({'Product ID': [], '花费': [], 'Campaign名称': [], 'Ad组名称': []})

    # ── 4. 其他校验 ─────────────────────────────────────────────
    refund_orders = orders[orders['Is_Refund']]
    退款订单数 = len(refund_orders)
    退款金额原始 = refund_orders['营收'].sum()
    退款金额利润表 = result.spu_profit['退款金额'].sum() if not result.spu_profit.empty else 0
    退款差异 = 退款金额原始 - 退款金额利润表

    营收原始 = orders['营收'].sum()
    营收利润表 = result.spu_profit['退款前营收'].sum() if not result.spu_profit.empty else 0
    营收差异 = 营收原始 - 营收利润表

    sample_orders = orders[(orders['营收'] == 0) & (~orders['Is_Refund'])]
    样品订单数 = len(sample_orders)
    样品数量 = sample_orders['Quantity'].sum()
    利润表样品费 = result.spu_profit['样品费'].sum() if not result.spu_profit.empty else 0

    成本为0的SPU数 = len(result.spu_profit[result.spu_profit['成本'] == 0]) if not result.spu_profit.empty else 0

    # 排除表头行后的所有订单状态
    valid_status = orders['Order Status'].dropna()
    valid_status = valid_status[valid_status != 'Current order status.']
    unique_status = sorted(valid_status.unique())
    订单状态 = '、'.join(unique_status)

    df_other = pd.DataFrame({
        '校验项': [
            '退款订单数', '退款金额(原始)', '退款金额(利润表)', '退款差异',
            '退款前营收(原始)', '退款前营收(利润表)', '营收差异',
            '样品订单数', '样品数量', '利润表样品费',
            '成本为0的SPU数', '未知订单状态'
        ],
        '结果': [
            退款订单数,
            round(退款金额原始, 2),
            round(退款金额利润表, 2),
            round(退款差异, 2),
            round(营收原始, 2),
            round(营收利润表, 2),
            round(营收差异, 2),
            样品订单数,
            int(样品数量),
            round(利润表样品费, 2),
            成本为0的SPU数,
            订单状态
        ]
    })

    return {
        'sku_cost': df_sku_cost,
        'affiliate': df_affiliate,
        'ads': df_ads,
        'ads_unmapped': df_ads_unmapped,
        'other': df_other,
    }
