import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import altair as alt

# ================= 1. 页面基础配置 =================
st.set_page_config(
    page_title="华青TikTok 利润测算系统 (Web版)",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded" 
)

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    div.stButton > button:first-child {
        background-color: #ff0050; color: white; border-radius: 8px; 
        padding: 12px 24px; font-weight: 600; border: none; width: 100%; font-size: 18px;
    }
    div.stButton > button:first-child:hover {background-color: #d60043; color: white;}
    [data-testid="stMetricValue"] {font-size: 24px; font-weight: bold; color: #1e1e1e;}
    
    /* KPI 卡片样式 */
    .kpi-card {
        background-color: white; padding: 20px; border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px; border: 1px solid #e0e0e0;
    }
    .kpi-title {font-size: 16px; color: #666; margin-bottom: 5px;}
    
    /* 文本域样式 */
    .stTextArea textarea {font-family: 'Consolas', 'Courier New', monospace; font-size: 14px;}
    </style>
""", unsafe_allow_html=True)

# ================= 2. 全局配置 =================
EXCHANGE_RATE = 1 / 7.15 

# 输出列定义
TARGET_COLUMNS_SKU = [
    'SPU', 'SKU', '类别', '销量', 
    '退款前营收', '退款后营收', 
    '利润率', '利润额', 'ASP',
    '营业成本率', '运营成本率', '总营销费比',
    '单件采购成本', '单件头程', '单件关税', '单件尾程', 
    '退款单数', '退款营收', '退款率', 
    '总达人佣金', 
    '单件样品成本', '总样品费', 
    '总广告投放费', 
    '采购成本-占比', '头程-占比', '关税占比', '尾程-占比', 
    '仓租-占比', '其他物流成本-占比', '品牌费用-占比', '平台佣金-占比', 
    '其他和售后-占比', '达人佣金-占比', '样品费-占比', 
    '广告投放费-占比'
]

TARGET_COLUMNS_SPU = [col for col in TARGET_COLUMNS_SKU if col not in [
    'SKU', '单件采购成本', '单件头程', '单件关税', '单件尾程', '单件样品成本'
]]
TARGET_COLUMNS_SHOP = [col for col in TARGET_COLUMNS_SPU if col not in ['SPU', '类别']]
TARGET_COLUMNS_SHOP_FINAL = ['数据周期'] + TARGET_COLUMNS_SHOP

# ================= 3. 基础工具函数 =================

def normalize_headers(df):
    if df is None: return None
    df.columns = df.columns.astype(str).str.replace(r'[\u200b\ufeff]', '', regex=True).str.replace(r'\s+', ' ', regex=True).str.strip()
    return df

def clean_text(df, col_name):
    if col_name in df.columns:
        return df[col_name].astype(str).str.replace(r'[\u200b\ufeff]', '', regex=True).str.strip().str.upper()
    return df[col_name]

def convert_scientific_to_str(val):
    if pd.isna(val): return ""
    try:
        if isinstance(val, (int, float)): return str(int(val))
        s = str(val).strip()
        s = re.sub(r'[\u200b\ufeff]', '', s) 
        if 'E' in s.upper(): return str(int(float(s)))
        if s.endswith('.0'): return s[:-2]
        return s
    except: return str(val).strip()

def clean_money(val):
    if pd.isna(val): return 0
    s = str(val).strip()
    s = re.sub(r'[^\d\.\-]', '', s) 
    try: return float(s)
    except: return 0

def find_col_by_keyword(df, keywords):
    for col in df.columns:
        c_low = str(col).lower()
        if all(k.lower() in c_low for k in keywords):
            return col
    return None

def find_order_id_col(df):
    candidates = ['Order ID', 'Order Id', 'order id', 'order_id', '订单号', 'Main Order ID']
    for c in df.columns:
        if str(c).strip() in candidates: return c
    for c in df.columns:
        if 'order' in str(c).lower() and 'id' in str(c).lower(): return c
    return None

def find_affiliate_sku_col(df):
    for c in df.columns:
        if 'seller' in str(c).lower() and 'sku' in str(c).lower(): return c
    for c in df.columns:
        if 'sku' in str(c).lower() and 'product' not in str(c).lower(): return c
    return None

def get_cost_map(cost_df, keywords):
    if cost_df is None: return {}
    target_col = find_col_by_keyword(cost_df, keywords)
    if not target_col: return {}
    sku_col = find_col_by_keyword(cost_df, ['sku'])
    if not sku_col: return {}
    
    cost_df['SKU_Clean'] = clean_text(cost_df, sku_col)
    cost_df['Clean_Cost'] = cost_df[target_col].apply(clean_money)
    cost_df['USD'] = cost_df['Clean_Cost'] * EXCHANGE_RATE
    return dict(zip(cost_df['SKU_Clean'], cost_df['USD']))

def build_sku_to_spu_dict(df_spu_sku):
    if df_spu_sku is None: return {}
    mapping_dict = {}
    spu_col = find_col_by_keyword(df_spu_sku, ['spu'])
    if not spu_col: return {}
    candidate_cols = [c for c in df_spu_sku.columns if 'sku' in str(c).lower() and c != spu_col]
    for _, row in df_spu_sku.iterrows():
        target_spu = row[spu_col]
        if pd.isna(target_spu) or str(target_spu).strip() == '': continue
        target_spu = str(target_spu).strip()
        for col in candidate_cols:
            sku_val = row[col]
            if pd.notna(sku_val) and str(sku_val).strip() != '':
                mapping_dict[str(sku_val).strip().upper()] = target_spu
    return mapping_dict

def format_dataframe(df, target_columns):
    for col in target_columns:
        if col not in df.columns: df[col] = 0   
    df_out = df.reindex(columns=target_columns, fill_value=0)
    pct_columns = [
        '利润率', '退款率', '总营销费比', '营业成本率', '运营成本率',
        '采购成本-占比', '头程-占比', '关税占比', '尾程-占比', 
        '仓租-占比', '其他物流成本-占比', '品牌费用-占比', '平台佣金-占比', 
        '其他和售后-占比', '达人佣金-占比', '样品费-占比', '广告投放费-占比'
    ]
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    money_cols = [c for c in numeric_cols if c not in pct_columns]
    
    df_out[money_cols] = df_out[money_cols].fillna(0).round(2)
    
    for col in pct_columns:
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(0).apply(lambda x: f"{x:.2%}")
    return df_out

# ================= 4. 日期处理与分析引擎 =================

def ensure_date_column(df, label="未知"):
    if df is None or df.empty: return False
    if 'Date' in df.columns: return True
    
    col_date = 'Created Time'
    if col_date not in df.columns:
        col_date = find_col_by_keyword(df, ['created', 'time', 'date', '日期'])
    
    if col_date and col_date in df.columns:
        try:
            df['Date'] = pd.to_datetime(df[col_date], dayfirst=False, errors='coerce')
            if df['Date'].notna().sum() > 0: return True
            else:
                return False
        except: return False
    return False

def calculate_yoy_metrics(df_current, df_last):
    if not ensure_date_column(df_current, "今年订单"): return None
    
    def get_core_stats(df):
        df = df.copy()
        col_rev = '营收' if '营收' in df.columns else find_col_by_keyword(df, ['revenue', 'amount'])
        col_qty = 'Quantity' if 'Quantity' in df.columns else find_col_by_keyword(df, ['quantity'])
        col_status = 'Order Status' if 'Order Status' in df.columns else find_col_by_keyword(df, ['status'])
        
        if not col_rev: col_rev = find_col_by_keyword(df, ['amount'])
        if not col_rev or not col_qty: return 0, 0, 0, 0
        
        df[col_rev] = pd.to_numeric(df[col_rev], errors='coerce').fillna(0)
        df[col_qty] = pd.to_numeric(df[col_qty], errors='coerce').fillna(0)
        is_cancelled = df[col_status].astype(str).str.strip().isin(['Cancelled', 'Canceled'])
        
        gmv = df[col_rev].sum()
        net_rev = df.loc[~is_cancelled, col_rev].sum()
        sales_qty = df.loc[~is_cancelled, col_qty].sum()
        refund_rev = df.loc[is_cancelled, col_rev].sum()
        refund_rate = refund_rev / gmv if gmv > 0 else 0
        return gmv, net_rev, sales_qty, refund_rate

    curr_gmv, curr_net, curr_qty, curr_ref_rate = get_core_stats(df_current)
    return {'curr': (curr_gmv, curr_net, curr_qty, curr_ref_rate)}

def get_dual_trend_data(df_curr, df_last):
    if df_curr is None: return None, None
    ensure_date_column(df_curr)
    
    data_biweek = []
    data_monthly = []
    
    def process_df(df, year_label):
        if 'Date' not in df.columns: return
        
        col_rev = '营收' if '营收' in df.columns else find_col_by_keyword(df, ['revenue', 'amount'])
        col_status = 'Order Status' if 'Order Status' in df.columns else find_col_by_keyword(df, ['status'])
        
        if col_rev and col_status:
            df[col_rev] = pd.to_numeric(df[col_rev], errors='coerce').fillna(0)
            is_can = df[col_status].astype(str).str.strip().isin(['Cancelled', 'Canceled'])
            df_clean = df[~is_can].copy()
            
            # 月度
            df_clean['Month'] = df_clean['Date'].dt.strftime('%m')
            monthly_agg = df_clean.groupby('Month')[col_rev].sum().reset_index()
            for _, row in monthly_agg.iterrows():
                data_monthly.append({'X': row['Month'], 'Revenue': row[col_rev], 'Year': year_label})
                
            # 双周
            df_clean['DayOfYear'] = df_clean['Date'].dt.dayofyear
            df_clean['BiWeek'] = (df_clean['DayOfYear'] - 1) // 14 + 1
            biweek_agg = df_clean.groupby('BiWeek')[col_rev].sum().reset_index()
            for _, row in biweek_agg.iterrows():
                label = f"Bi-Week {int(row['BiWeek']):02d}"
                data_biweek.append({'X': label, 'Revenue': row[col_rev], 'Year': year_label})

    process_df(df_curr, '今年')
    if df_last is not None and not df_last.empty:
        if ensure_date_column(df_last, "2025趋势"):
            process_df(df_last, '去年')
            
    return pd.DataFrame(data_biweek), pd.DataFrame(data_monthly)

# ================= 5. 核心计算逻辑 =================

def calculate_metrics_final(df_base):
    df = df_base.copy()
    qty = df['销量'].replace(0, 1)
    rev_after = df['退款后营收']
    df['ASP'] = rev_after / qty
    
    if 'Refund_Orders' not in df.columns:
        df['Refund_Orders'] = df['退款单数'] if '退款单数' in df.columns else 0
        
    df['退款营收'] = df['Refund_Orders'] * df['ASP']
    df['退款前营收'] = rev_after + df['退款营收']
    
    rev_before_safe = df['退款前营收'].replace(0, 1)
    df['退款率'] = df['退款营收'] / rev_before_safe
    df['退款单数'] = df['Refund_Orders']

    for c in ['总达人佣金', '总样品费', '总广告投放费']:
        if c not in df.columns: df[c] = 0
        
    mkt_cost = df['总广告投放费'] + df['总达人佣金'] + df['总样品费']
    df['总营销费比'] = mkt_cost / rev_after.replace(0, 1)

    df['仓租'] = rev_after * 0.005
    df['其他物流成本'] = rev_after * 0.003
    df['品牌费用'] = rev_after * 0.003
    df['平台佣金'] = rev_after * 0.06
    df['其他和售后'] = rev_after * 0.003

    for c in ['采购成本', '头程', '尾程', '关税']:
        if c not in df.columns: df[c] = 0

    all_costs = sum(df[c] for c in [
        '采购成本', '头程', '尾程', '关税', 
        '仓租', '其他物流成本', '品牌费用', '平台佣金', '其他和售后', 
        '总达人佣金', '总样品费', '总广告投放费'
    ] if c in df.columns)
    
    df['利润额'] = rev_after - all_costs
    df['利润率'] = df['利润额'] / rev_after.replace(0, 1)

    rev_safe = rev_after.replace(0, 1)
    cogs = df['采购成本'] + df['头程'] + df['关税'] + df['尾程']
    df['营业成本率'] = cogs / rev_safe
    ops_cost = df['仓租'] + df['其他物流成本'] + df['品牌费用'] + df['平台佣金'] + df['其他和售后']
    df['运营成本率'] = ops_cost / rev_safe
    
    ratio_map = {
        '采购成本-占比': '采购成本', '头程-占比': '头程', '尾程-占比': '尾程', '关税占比': '关税',
        '仓租-占比': '仓租', '其他物流成本-占比': '其他物流成本',
        '品牌费用-占比': '品牌费用', '平台佣金-占比': '平台佣金', '其他和售后-占比': '其他和售后',
        '达人佣金-占比': '总达人佣金', '样品费-占比': '总样品费', '广告投放费-占比': '总广告投放费',
    }
    for r_col, val_col in ratio_map.items():
        if val_col in df.columns:
            df[r_col] = df[val_col] / rev_safe
        else:
            df[r_col] = 0
            
    return df

def run_calculation_logic(dfs):
    for key, df in dfs.items():
        if df is not None: dfs[key] = normalize_headers(df)
    
    df_orders = dfs['orders']
    if df_orders is None: return None, None, None, "无订单数据", None

    col_sku = 'Seller SKU' if 'Seller SKU' in df_orders.columns else 'SKU'
    df_orders[col_sku] = clean_text(df_orders, col_sku)
    df_orders['clean_order_id'] = df_orders['Order ID'].apply(convert_scientific_to_str)
    
    sku_to_spu_dict = build_sku_to_spu_dict(dfs['spu_sku'])
    if sku_to_spu_dict:
        df_orders['SPU'] = df_orders[col_sku].map(sku_to_spu_dict).fillna(df_orders[col_sku])
    else:
        if 'SPU' not in df_orders.columns: df_orders['SPU'] = df_orders[col_sku]

    time_str = "未知周期"
    max_date = None
    if ensure_date_column(df_orders, "今年订单"):
        try:
            dates = df_orders['Date'].dropna()
            if not dates.empty:
                time_str = f"{dates.min().strftime('%Y-%m-%d')} ~ {dates.max().strftime('%Y-%m-%d')}"
                max_date = dates.max()
        except: pass

    # === 佣金匹配 ===
    df_orders['Comm'] = 0
    sku_real_comm = None
    df_aff = dfs['affiliate']
    if df_aff is not None:
        aff_id = find_order_id_col(df_aff)
        aff_sku = find_affiliate_sku_col(df_aff)
        std_cols = [c for c in df_aff.columns if 'standard commission' in str(c).lower() and 'payment' in str(c).lower()]
        ads_cols = [c for c in df_aff.columns if 'shop ads commission' in str(c).lower() and 'payment' in str(c).lower()]

        if aff_id and aff_sku and (std_cols or ads_cols):
            df_aff['clean_order_id'] = df_aff[aff_id].apply(convert_scientific_to_str)
            df_aff['Mapped_SKU'] = clean_text(df_aff, aff_sku)
            df_aff['Total_Raw'] = 0
            for c in std_cols + ads_cols: df_aff['Total_Raw'] += df_aff[c].apply(clean_money)
            
            aff_grp = df_aff.groupby(['clean_order_id', 'Mapped_SKU'])['Total_Raw'].sum().reset_index()
            df_orders['sku_weight'] = df_orders.groupby(['clean_order_id', col_sku])['clean_order_id'].transform('count')
            merged = pd.merge(df_orders, aff_grp, left_on=['clean_order_id', col_sku], right_on=['clean_order_id', 'Mapped_SKU'], how='left')
            df_orders['Comm'] = merged['Total_Raw'].fillna(0) / df_orders['sku_weight'].fillna(1)

    sku_real_comm = df_orders.groupby(['SPU', col_sku])['Comm'].sum().reset_index().rename(columns={col_sku: 'SKU', 'Comm': '总达人佣金'})

    # === 订单分类 ===
    if 'Order ID' in df_orders.columns:
        df_orders = df_orders[~df_orders['Order ID'].astype(str).str.contains('Platform|Order ID', na=False)]
    
    col_rev = '营收'
    col_qty = 'Quantity'
    col_status = 'Order Status'
    df_orders[col_rev] = pd.to_numeric(df_orders[col_rev], errors='coerce').fillna(0)
    df_orders[col_qty] = pd.to_numeric(df_orders.get(col_qty, 1), errors='coerce').fillna(0)
    
    is_cancelled = df_orders[col_status].astype(str).str.strip().isin(['Cancelled', 'Canceled'])
    is_normal = (~is_cancelled) & (df_orders[col_rev] > 0)
    is_sample = (~is_cancelled) & (df_orders[col_rev] == 0)
    
    df_normal = df_orders[is_normal].copy()
    df_sample = df_orders[is_sample].copy()
    df_refund = df_orders[is_cancelled].copy()

    # === 成本准备 ===
    map_p = get_cost_map(dfs['purchase'], ['采购', 'CNY'])
    map_h = get_cost_map(dfs['head'], ['头程', 'CNY'])
    map_t = get_cost_map(dfs['tail'], ['尾程', 'CNY'])

    master_skus = df_orders[['SPU', col_sku]].drop_duplicates().rename(columns={col_sku: 'SKU'})

    # === SKU 统计 ===
    norm_stat = df_normal.groupby(['SPU', col_sku]).agg({col_rev: 'sum', col_qty: 'sum', 'Product Name': 'first'}).reset_index().rename(columns={col_sku: 'SKU', col_qty: '销量', col_rev: '退款后营收'})
    sku_stats = pd.merge(master_skus, norm_stat, on=['SPU', 'SKU'], how='left')
    sku_stats[['销量', '退款后营收']] = sku_stats[['销量', '退款后营收']].fillna(0)
    
    if 'Product Name' in df_orders.columns:
        pmap = df_orders.groupby(col_sku)['Product Name'].first().to_dict()
        sku_stats['Product Name'] = sku_stats['Product Name'].fillna(sku_stats['SKU'].map(pmap))

    if not df_refund.empty:
        ref_agg = df_refund.groupby(['SPU', col_sku])[col_qty].sum().reset_index().rename(columns={col_sku: 'SKU', col_qty: 'Refund_Orders'})
        sku_stats = pd.merge(sku_stats, ref_agg, on=['SPU', 'SKU'], how='left')
    else: sku_stats['Refund_Orders'] = 0
    sku_stats['Refund_Orders'] = sku_stats['Refund_Orders'].fillna(0)

    tmp_qty = sku_stats['销量'].replace(0, 1)
    sku_stats['ASP'] = sku_stats['退款后营收'] / tmp_qty
    sku_stats.loc[sku_stats['销量']==0, 'ASP'] = 0
    sku_stats['退款营收'] = sku_stats['Refund_Orders'] * sku_stats['ASP']
    sku_stats['退款前营收'] = sku_stats['退款后营收'] + sku_stats['退款营收']

    # === 样品与成本 ===
    sku_stats['总样品费'] = 0
    sku_stats['单件样品成本'] = 0
    if not df_sample.empty:
        df_sample['Unit_S'] = df_sample[col_sku].map(map_p).fillna(0) + df_sample[col_sku].map(map_h).fillna(0) + df_sample[col_sku].map(map_t).fillna(0)
        df_sample['Total_S'] = df_sample[col_qty] * df_sample['Unit_S']
        s_agg = df_sample.groupby(['SPU', col_sku])['Total_S'].sum().reset_index().rename(columns={col_sku: 'SKU', 'Total_S': '总样品费'})
        u_agg = df_sample.groupby(['SPU', col_sku])['Unit_S'].first().reset_index().rename(columns={col_sku: 'SKU', 'Unit_S': '单件样品成本'})
        sku_stats = pd.merge(sku_stats, s_agg, on=['SPU', 'SKU'], how='left')
        sku_stats = pd.merge(sku_stats, u_agg, on=['SPU', 'SKU'], how='left')
        if '总样品费_y' in sku_stats.columns: sku_stats['总样品费'] = sku_stats['总样品费_y'].fillna(0)
        else: sku_stats['总样品费'] = sku_stats.get('总样品费', 0).fillna(0)
        if '单件样品成本_y' in sku_stats.columns: sku_stats['单件样品成本'] = sku_stats['单件样品成本_y'].fillna(0)
        else: sku_stats['单件样品成本'] = sku_stats.get('单件样品成本', 0).fillna(0)

    def fill_sample_unit(row):
        if row['单件样品成本'] == 0: return map_p.get(row['SKU'], 0) + map_h.get(row['SKU'], 0) + map_t.get(row['SKU'], 0)
        return row['单件样品成本']
    sku_stats['单件样品成本'] = sku_stats.apply(fill_sample_unit, axis=1)

    sku_stats['单件采购成本'] = sku_stats['SKU'].map(map_p).fillna(0)
    sku_stats['单件头程'] = sku_stats['SKU'].map(map_h).fillna(0)
    sku_stats['单件尾程'] = sku_stats['SKU'].map(map_t).fillna(0)
    sku_stats['单件关税'] = 0
    sku_stats['采购成本'] = sku_stats['单件采购成本'] * sku_stats['销量']
    sku_stats['头程'] = sku_stats['单件头程'] * sku_stats['销量']
    sku_stats['尾程'] = sku_stats['单件尾程'] * sku_stats['销量']
    sku_stats['关税'] = 0

    sku_stats = pd.merge(sku_stats, sku_real_comm, on=['SPU', 'SKU'], how='left')
    sku_stats['总达人佣金'] = sku_stats['总达人佣金'].fillna(0)

    # === 广告分摊 ===
    sku_stats['总广告投放费'] = 0
    df_ads = dfs['ads']
    df_map = dfs['mapping']
    if df_ads is not None and df_map is not None:
        pid_c = find_col_by_keyword(df_map, ['product_id'])
        sku_mc = find_col_by_keyword(df_map, ['sku'])
        if pid_c and sku_mc:
            df_map[pid_c] = clean_text(df_map, pid_c)
            df_map[sku_mc] = clean_text(df_map, sku_mc)
            pid_grps = df_map.groupby(pid_c)[sku_mc].apply(list).reset_index()
            ad_pid = find_col_by_keyword(df_ads, ['product id'])
            if ad_pid:
                df_ads[ad_pid] = clean_text(df_ads, ad_pid)
                df_ads['Cost'] = pd.to_numeric(df_ads['Cost'], errors='coerce').fillna(0)
                rev_map = dict(zip(sku_stats['SKU'], sku_stats['退款前营收']))
                dist_list = []
                merged_ads = pd.merge(df_ads, pid_grps, left_on=ad_pid, right_on=pid_c, how='inner')
                for _, row in merged_ads.iterrows():
                    cost = row['Cost']
                    skus = row[sku_mc]
                    if not skus: continue
                    revs = {s: rev_map.get(s, 0) for s in skus}
                    tot = sum(revs.values())
                    for s in skus:
                        if tot > 0: share = cost * (revs[s] / tot)
                        else: share = cost / len(skus)
                        dist_list.append({'SKU': s, 'AdsCost': share})
                if dist_list:
                    ads_df = pd.DataFrame(dist_list)
                    ads_agg = ads_df.groupby('SKU')['AdsCost'].sum().reset_index().rename(columns={'AdsCost': '总广告投放费'})
                    sku_stats = pd.merge(sku_stats, ads_agg, on='SKU', how='left')
                    if '总广告投放费_y' in sku_stats.columns: sku_stats['总广告投放费'] = sku_stats['总广告投放费_y'].fillna(0)
                    else: sku_stats['总广告投放费'] = sku_stats.get('总广告投放费', 0).fillna(0)

    # === 汇总 ===
    df_sku_final = calculate_metrics_final(sku_stats)
    df_sku_out = format_dataframe(df_sku_final, TARGET_COLUMNS_SKU)

    sum_cols = ['销量', '退款后营收', '退款前营收', 'Refund_Orders', '退款营收', '采购成本', '头程', '尾程', '关税', '仓租', '其他物流成本', '品牌费用', '平台佣金', '其他和售后', '总达人佣金', '总样品费', '总广告投放费']
    spu_agg = sku_stats.groupby('SPU').agg({**{c: 'sum' for c in sum_cols if c in sku_stats.columns}, 'Product Name': 'first'}).reset_index().rename(columns={'Product Name': '类别'})
    df_spu_final = calculate_metrics_final(spu_agg).sort_values(by='退款后营收', ascending=False)
    df_spu_out = format_dataframe(df_spu_final, TARGET_COLUMNS_SPU)

    shop_agg = sku_stats.agg({c: 'sum' for c in sum_cols if c in sku_stats.columns}).to_frame().T
    df_shop_final = calculate_metrics_final(shop_agg)
    df_shop_final['数据周期'] = time_str
    df_shop_out = format_dataframe(df_shop_final, TARGET_COLUMNS_SHOP_FINAL)

    return df_shop_out, df_spu_out, df_sku_out, time_str, max_date

# ================= 7. 智能文件识别读取器 (修复版 V2.3) =================
def load_uploaded_files(uploaded_files):
    dfs = {
        'orders': None, 'orders_last_year': None, 'ads': None, 'affiliate': None,
        'spu_sku': None, 'mapping': None, 'purchase': None, 'head': None, 'tail': None
    }
    status_flags = {k: False for k in dfs.keys()}
    
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name.lower()
        
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(uploaded_file, dtype=str)
            else:
                df = pd.read_excel(uploaded_file, dtype=str)
        except Exception:
            continue
            
        # 修正逻辑：关键词匹配优先级
        if '2025' in filename or '去年' in filename:
            dfs['orders_last_year'] = df
            status_flags['orders_last_year'] = True
        elif '广告' in filename or 'ads' in filename:
            dfs['ads'] = df
            status_flags['ads'] = True
        elif '联盟' in filename or 'affiliate' in filename:
            dfs['affiliate'] = df
            status_flags['affiliate'] = True
        elif 'spu' in filename:
            dfs['spu_sku'] = df
            status_flags['spu_sku'] = True
        elif 'pid' in filename or 'mapping' in filename:
            dfs['mapping'] = df
            status_flags['mapping'] = True
        elif '采购' in filename or 'purchase' in filename:
            dfs['purchase'] = df
            status_flags['purchase'] = True
        elif '头程' in filename or 'head' in filename:
            dfs['head'] = df
            status_flags['head'] = True
        elif '尾程' in filename or 'tail' in filename:
            dfs['tail'] = df
            status_flags['tail'] = True
        elif '订单' in filename or 'order' in filename:
            # 只有当不是 2025 的订单时，才认为是今年的
            dfs['orders'] = df
            status_flags['orders'] = True
            
    return dfs, status_flags

# ================= 8. 主程序 =================
def main():
    st.title("🚀 华青ikTok 利润测算仪表盘 (Web协同版)")
    
    # --- 侧边栏：上传与设置 ---
    with st.sidebar:
        st.header("📂 1. 拖拽上传文件")
        st.info("💡 提示：一次性选中所有文件拖进来即可，系统会自动识别。")
        
        uploaded_files = st.file_uploader(
            "请上传业务数据表 (支持 xlsx/csv)", 
            accept_multiple_files=True,
            type=['xlsx', 'csv']
        )
        
        # 实时状态灯
        if uploaded_files:
            with st.spinner("⏳ 正在智能解析文件，请稍候..."):
                dfs, flags = load_uploaded_files(uploaded_files)
            
            st.markdown("### 📊 文件就位状态")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.write(f"{'✅' if flags['orders'] else '❌'} 今年订单")
                st.write(f"{'✅' if flags['ads'] else '❌'} 广告表")
                st.write(f"{'✅' if flags['purchase'] else '❌'} 采购成本")
                st.write(f"{'✅' if flags['spu_sku'] else '❌'} SPU映射")
            with col_s2:
                st.write(f"{'✅' if flags['orders_last_year'] else '⚠️'} 2025订单")
                st.write(f"{'✅' if flags['affiliate'] else '❌'} 联盟表")
                st.write(f"{'✅' if flags['head'] else '❌'} 头程成本")
                st.write(f"{'✅' if flags['tail'] else '❌'} 尾程成本")
        else:
            dfs = {}
            flags = {}

        st.divider()
        st.subheader("🎯 2. 目标设定")
        target_revenue = st.number_input("本月营收目标 ($)", value=0.0, step=1000.0)
        target_profit_rate = st.number_input("目标利润率 (%)", value=15.0, step=0.5) / 100.0

    # 主操作按钮
    if st.button("🚀 点击开始测算", type="primary", disabled=not flags.get('orders')):
        st.session_state['has_run'] = True
        
        with st.spinner("⏳ 正在进行：数据清洗、智能匹配、利润计算..."):
            try:
                df_shop, df_spu, df_sku, time_str, max_date = run_calculation_logic(dfs)
                if df_shop is None:
                    st.error("❌ 订单表数据为空或格式错误，请检查文件。")
                    st.stop()
                    
                st.session_state['data'] = {
                    'dfs': dfs, 
                    'df_shop': df_shop,
                    'df_spu': df_spu,
                    'df_sku': df_sku,
                    'time_str': time_str,
                    'max_date': max_date
                }
            except Exception as e:
                st.error(f"❌ 运行错误: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.session_state['has_run'] = False

    # 结果展示页面
    if st.session_state.get('has_run') and st.session_state.get('data'):
        data = st.session_state['data']
        df_shop = data['df_shop']
        df_spu = data['df_spu']
        df_sku = data['df_sku']
        time_str = data['time_str']
        max_date = data['max_date']
        dfs = data['dfs'] 

        # 计算 KPI
        shop_row = df_shop.iloc[0]
        curr_rev = shop_row['退款后营收']
        
        if pd.notna(max_date):
            days_in_month = pd.Period(max_date, freq='M').days_in_month
            time_progress = max_date.day / days_in_month
        else:
            time_progress = 0
            
        mtd_achieve = curr_rev / target_revenue if target_revenue > 0 else 0
        pace_status = "🔴 落后" if mtd_achieve < time_progress else "🟢 超前"
        
        yoy_data = calculate_yoy_metrics(dfs['orders'], dfs.get('orders_last_year'))
        trend_df_bw, trend_df_m = get_dual_trend_data(dfs['orders'], dfs.get('orders_last_year'))

        st.success(f"✅ 测算成功！数据周期: {time_str}")
        
        # === 1. 经营概览 ===
        st.markdown("### 📈 1. 经营概览 (KPI Dashboard)")
        kpi_col1, kpi_col2 = st.columns(2)
        
        with kpi_col1:
            st.markdown("""<div class="kpi-card"><div class="kpi-title">📊 KPI 考核与进度</div>""", unsafe_allow_html=True)
            st.write(f"⏳ 月度时间进度 ({time_progress:.1%})")
            st.progress(time_progress)
            c1, c2 = st.columns(2)
            c1.metric("🎯 目标营收", f"${target_revenue:,.0f}")
            c2.metric("💰 实际营收", f"${curr_rev:,.0f}", f"{mtd_achieve:.1%} (达成率)")
            st.write(f"**进度判定**: {pace_status} (MTD)")
            st.divider()
            c3, c4 = st.columns(2)
            c3.metric("🎯 目标利润率", f"{target_profit_rate:.1%}")
            c4.metric("💰 实际利润率", f"{shop_row['利润率']}", f"{(clean_money(shop_row['利润率'].strip('%'))/100 - target_profit_rate):.1%}")
            st.markdown("</div>", unsafe_allow_html=True)

        with kpi_col2:
            st.markdown("""<div class="kpi-card"><div class="kpi-title">🌍 大盘核心数据 (vs 去年同期)</div>""", unsafe_allow_html=True)
            curr = yoy_data['curr']
            m1, m2 = st.columns(2)
            m1.metric("GMV (退款前)", f"${shop_row['退款前营收']:,.0f}")
            m2.metric("净利润", f"${shop_row['利润额']:,.0f}")
            st.divider()
            m3, m4 = st.columns(2)
            m3.metric("综合退款率", shop_row['退款率'], "美区基准 10-20%", delta_color="inverse")
            m4.metric("营销费比", shop_row['总营销费比'], "含广告+达人+样品", delta_color="inverse")
            st.markdown("</div>", unsafe_allow_html=True)

        # === 2. 趋势图 ===
        st.markdown("### 📊 2. 营收趋势对比")
        t_tab1, t_tab2 = st.tabs(["📅 双周视图 (Bi-Weekly)", "🗓️ 月度视图 (Monthly)"])
        
        with t_tab1:
            if trend_df_bw is not None and not trend_df_bw.empty:
                chart = alt.Chart(trend_df_bw).mark_line(point=True).encode(
                    x=alt.X('X', title='双周周期'), y=alt.Y('Revenue', title='净营收 ($)'),
                    color=alt.Color('Year', title='年份', scale=alt.Scale(domain=['今年', '去年'], range=['#ff0050', '#c3cfe2'])),
                    tooltip=['Year', 'X', 'Revenue']
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            else: st.info("暂无数据")
            
        with t_tab2:
            if trend_df_m is not None and not trend_df_m.empty:
                chart = alt.Chart(trend_df_m).mark_line(point=True).encode(
                    x=alt.X('X', title='月份'), y=alt.Y('Revenue', title='净营收 ($)'),
                    color=alt.Color('Year', title='年份', scale=alt.Scale(domain=['今年', '去年'], range=['#ff0050', '#c3cfe2'])),
                    tooltip=['Year', 'X', 'Revenue']
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            else: st.info("暂无数据")

        st.markdown("---")
        
        # === 3. 爆品与亏损 ===
        st.markdown("### 🔥 3. 爆品与亏损分析")
        col_p1, col_p2 = st.columns([1.5, 1])
        with col_p1:
            st.markdown("#### 🏆 利润贡献 Top 10 SPU")
            if not df_spu.empty:
                df_spu_sort = df_spu.copy()
                df_spu_sort['利润额'] = pd.to_numeric(df_spu_sort['利润额'], errors='coerce')
                top_10 = df_spu_sort.sort_values(by='利润额', ascending=False).head(10)
                chart_top = alt.Chart(top_10).mark_bar().encode(
                    x=alt.X('利润额', title='净利润 ($)'), y=alt.Y('SPU', sort='-x'),
                    color=alt.value('#2ecc71'), tooltip=['SPU', '利润额', '销量']
                ).interactive()
                st.altair_chart(chart_top, use_container_width=True)
        
        with col_p2:
            st.markdown("#### 🚨 亏损警示榜 (负利润)")
            if not df_spu.empty:
                df_spu_loss = df_spu.copy()
                df_spu_loss['利润额'] = pd.to_numeric(df_spu_loss['利润额'], errors='coerce')
                loss_spus = df_spu_loss[df_spu_loss['利润额'] < 0].copy()
                if not loss_spus.empty:
                    loss_spus = loss_spus.sort_values(by='利润额', ascending=True)
                    cols_loss = ['SPU', '销量', '退款后营收', '利润额', '利润率']
                    st.dataframe(loss_spus[cols_loss], use_container_width=True, height=400)
                else: st.success("🎉 恭喜！本期没有亏损 SPU。")

        st.markdown("---")
        
        # === 4. AI 与 报表 ===
        tab1, tab2, tab3, tab4 = st.tabs(["🏠 店铺汇总", "📦 SPU 分析", "📄 SKU 明细", "🤖 AI 经营参谋"])
        with tab1: st.dataframe(df_shop, use_container_width=True)
        with tab2: st.dataframe(df_spu, use_container_width=True)
        with tab3: st.dataframe(df_sku, use_container_width=True)
        
        with tab4:
            st.markdown("#### 🧠 AI 智能经营分析")
            st.info("💡 点击下方按钮生成指令，发送给 ChatGPT/DeepSeek。")
            if st.button("✨ 生成 AI 分析提示词"):
                prompt = f"""
你是一名资深美国 TikTok Shop 电商操盘手。请根据以下业务数据，撰写一份专业的周报分析。

【1. KPI 考核与进度】
- 日期节点：{max_date.strftime('%Y-%m-%d') if pd.notna(max_date) else 'N/A'}
- 时间进度：{time_progress:.1%}
- 目标营收：${target_revenue:,.0f} | 实际营收：${curr_rev:,.0f} | 达成率：{mtd_achieve:.1%}
- 进度判定：{pace_status}
- 目标利润率：{target_profit_rate:.1%} | 实际利润率：{shop_row['利润率']}

【2. 大盘核心数据】
- GMV：${shop_row['退款前营收']:,.0f}
- 净利润：${shop_row['利润额']:,.0f}
- 综合退款率：{shop_row['退款率']} (美区基准通常在 10%-20%)
- 营销费比：{shop_row['总营销费比']} (含广告+达人+样品)

【3. 异常单品诊断】
[红榜 - 利润贡献前3]
{df_spu.head(3)[['SPU', '利润额', '利润率']].to_string(index=False)}

[黑榜 - 亏损严重前3]
{df_spu.tail(3)[['SPU', '利润额', '利润率', '退款率', '总营销费比']].to_string(index=False)}

【任务要求】
请输出一份结构清晰的分析报告，包含：
1. **经营摘要**：点评本周盈亏及 KPI 达成情况，解释为何{pace_status}。
2. **问题诊断**：分析亏损 SPU 原因（广告失控？退款过高？定价太低？）。
3. **行动计划**：给出 3 条具体的优化建议。
"""
                st.code(prompt, language='text')
            
            report_content = st.text_area("✏️ 在此粘贴 AI 生成的报告并修改...", height=400)
            if report_content:
                st.download_button("📥 导出报告 (.txt)", report_content, f"经营分析_{time_str}.txt", "text/plain")

        # 底部下载
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_sku.to_excel(writer, sheet_name='SKU明细', index=False)
            df_spu.to_excel(writer, sheet_name='SPU汇总', index=False)
            df_shop.to_excel(writer, sheet_name='店铺汇总', index=False)
        st.download_button("📥 下载完整 Excel 利润表", output.getvalue(), f"利润表_{time_str}.xlsx")

if __name__ == '__main__':
    main()