import streamlit as st
import pandas as pd
import numpy as np
import re
import altair as alt
import time
import os
from datetime import datetime

# ================= 1. 页面基础配置 =================
st.set_page_config(
    page_title="TikTok AI运营系统（利润&广告&达人）",
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

    .kpi-card {
        background-color: white; padding: 18px; border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 16px; border: 1px solid #e0e0e0;
    }
    .kpi-title {font-size: 16px; color: #666; margin-bottom: 6px;}
    .note {color:#666; font-size: 12px;}
    .warn {color:#b04a00; font-size: 12px;}
    .ok {color:#117a37; font-size: 12px;}

    .stProgress > div > div > div > div {background-color: #ff0050;}
    </style>
""", unsafe_allow_html=True)

# ================= 2. 全局配置 & 核心列名映射 =================
EXCHANGE_RATE = 1 / 7.15

# 可调参数（业务规则）
ROI_BEST = 2.0
ROI_LOSS = 1.0
DEFAULT_ROAS_FOR_CPA_LINE = 3.0
COST_OBSERVE = 50.0

CTR_FLOOR = 0.01
CVR_FLOOR = 0.01
CPM_FLOOR_HIGH = 25.0

RATE2S_FLOOR = 0.20
RATE6S_FLOOR = 0.10

COLUMN_CONFIG = {
    'orders': {
        'sku': 'Seller SKU',
        'order_id': 'Order ID',
        'revenue': '营收',
        'qty': 'Quantity',
        'status': 'Order Status',
        'time': 'Created Time',
        'product_name': 'Product Name'
    },
    'ads': {
        'pid': 'Product ID',
        'cost': 'Cost',
        'revenue': 'Gross revenue',
        'orders': 'SKU orders',
        'impressions': 'Product ad impressions',
        'clicks': 'Product ad clicks',
        'video_title': 'Video title',
        'video_id': 'Video ID',
        'ctr': 'Product ad click rate',
        'cvr': 'Ad conversion rate',
        'rate_2s': '2-second ad video view rate',
        'rate_6s': '6-second ad video view rate'
    },
    'affiliate': {
        'creator': 'Creator Username',
        'gmv': 'Payment Amount',
        'commission': 'Actual Commission Payment',
        'commission_est_std': 'Est. standard commission payment',
        'commission_est_ads': 'Est. Shop Ads commission payment',
        'content_type': 'Content Type',
        'order_id': 'Order ID',
        'sku': 'Seller Sku'
    },
    'transaction': {
        'pid': 'Product ID',
        'aff_gmv': 'Affiliate-attributed GMV',
        'videos': 'Videos',
        'lives': 'LIVE streams'
    }
}

TARGET_COLUMNS_SKU = [
    'SPU', 'SKU', '类别', '销量', '退款前营收', '退款后营收',
    '利润率', '利润额', 'ASP', '营业成本率', '运营成本率', '总营销费比',
    '单件采购成本', '单件头程', '单件关税', '单件尾程',
    '退款单数', '退款营收', '退款率', '总达人佣金',
    '单件样品成本', '总样品费', '总广告投放费',
    '采购成本-占比', '头程-占比', '关税占比', '尾程-占比',
    '仓租-占比', '其他物流成本-占比', '品牌费用-占比', '平台佣金-占比',
    '其他和售后-占比', '达人佣金-占比', '样品费-占比', '广告投放费-占比'
]
TARGET_COLUMNS_SPU = [col for col in TARGET_COLUMNS_SKU if col not in ['SKU', '单件采购成本', '单件头程', '单件关税', '单件尾程', '单件样品成本']]
TARGET_COLUMNS_SHOP = [col for col in TARGET_COLUMNS_SPU if col not in ['SPU', '类别']]
TARGET_COLUMNS_SHOP_FINAL = ['数据周期'] + TARGET_COLUMNS_SHOP

# ================= 3. 基础工具函数 =================
def normalize_headers(df):
    if df is None:
        return None
    df.columns = df.columns.astype(str).str.strip()
    return df

def clean_text(df, col_name):
    if col_name in df.columns:
        return df[col_name].astype(str).str.replace(r'[\u200b\ufeff]', '', regex=True).str.strip().str.upper()
    return df[col_name]

def convert_scientific_to_str(val):
    if pd.isna(val):
        return ""
    try:
        if isinstance(val, (int, float)):
            return str(int(val))
        s = str(val).strip()
        s = re.sub(r'[\u200b\ufeff]', '', s)
        if 'E' in s.upper():
            return str(int(float(s)))
        if s.endswith('.0'):
            return s[:-2]
        return s
    except:
        return str(val).strip()

def clean_money(val):
    if pd.isna(val):
        return 0.0
    s = str(val).strip()
    s = re.sub(r'[^\d\.\-]', '', s)
    try:
        return float(s)
    except:
        return 0.0

def clean_percent(val):
    if pd.isna(val):
        return 0.0
    s = str(val).strip().replace('%', '')
    try:
        return float(s) / 100.0
    except:
        return 0.0

def find_col_by_keyword_fuzzy(df, keywords):
    if df is None:
        return None
    for col in df.columns:
        c_low = str(col).lower()
        for k in keywords:
            if k in c_low:
                return col
    return None

def get_cost_map(cost_df, keywords):
    if cost_df is None:
        return {}
    target_col = find_col_by_keyword_fuzzy(cost_df, keywords)
    if not target_col:
        return {}
    sku_col = find_col_by_keyword_fuzzy(cost_df, ['sku'])
    if not sku_col:
        return {}
    cost_df = cost_df.copy()
    cost_df['SKU_Clean'] = cost_df[sku_col].astype(str).str.replace(r'[\u200b\ufeff]', '', regex=True).str.strip().str.upper()
    cost_df['Clean_Cost'] = cost_df[target_col].apply(clean_money)
    cost_df['USD'] = cost_df['Clean_Cost'] * EXCHANGE_RATE
    return dict(zip(cost_df['SKU_Clean'], cost_df['USD']))

def build_sku_to_spu_dict(df_spu_sku):
    if df_spu_sku is None:
        return {}
    mapping_dict = {}
    spu_col = find_col_by_keyword_fuzzy(df_spu_sku, ['spu'])
    if not spu_col:
        return {}
    candidate_cols = [c for c in df_spu_sku.columns if 'sku' in str(c).lower() and c != spu_col]
    for _, row in df_spu_sku.iterrows():
        target_spu = row[spu_col]
        if pd.isna(target_spu) or str(target_spu).strip() == '':
            continue
        target_spu = str(target_spu).strip()
        for col in candidate_cols:
            sku_val = row[col]
            if pd.notna(sku_val) and str(sku_val).strip() != '':
                mapping_dict[str(sku_val).strip().upper()] = target_spu
    return mapping_dict

def format_dataframe(df, target_columns):
    df_out = df.copy()
    for col in target_columns:
        if col not in df_out.columns:
            df_out[col] = 0
    df_out = df_out.reindex(columns=target_columns, fill_value=0)

    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    money_cols = [c for c in numeric_cols if '占比' not in c and '率' not in c and 'ASP' not in c]
    df_out[money_cols] = df_out[money_cols].fillna(0).round(2)

    for col in df_out.columns:
        if '占比' in col or '率' in col:
            df_out[col] = df_out[col].fillna(0).apply(lambda x: f"{x:.2%}")
    return df_out

def safe_div(a, b, default=0.0):
    try:
        return a / b if b not in [0, 0.0, None, np.nan] else default
    except:
        return default

# ================= 4. 日期处理 =================
def ensure_date_column(df):
    if df is None or df.empty:
        return False
    if 'Date' in df.columns:
        return True
    col_date = None
    for c in df.columns:
        if 'created time' in c.lower() or 'time posted' in c.lower():
            col_date = c
            break
    if col_date:
        try:
            df['Date'] = pd.to_datetime(df[col_date], dayfirst=False, errors='coerce')
            return df['Date'].notna().any()
        except:
            return False
    return False

def get_dual_trend_data(df_curr, df_last):
    if df_curr is None:
        return None, None
    ensure_date_column(df_curr)
    cols = ['X', 'Revenue', 'Year']
    data_biweek = []
    data_monthly = []

    rev_col_name = COLUMN_CONFIG['orders']['revenue']

    def process_df(df, year_label):
        if 'Date' not in df.columns:
            return
        col_rev = None
        for c in df.columns:
            if rev_col_name in c or 'revenue' in c.lower() or 'amount' in c.lower():
                col_rev = c
                break
        if not col_rev:
            return
        df = df.copy()
        df[col_rev] = pd.to_numeric(df[col_rev], errors='coerce').fillna(0)

        col_status = None
        for c in df.columns:
            if 'status' in c.lower():
                col_status = c
                break
        if col_status:
            is_can = df[col_status].astype(str).str.strip().isin(['Cancelled', 'Canceled'])
            df_clean = df[~is_can].copy()
        else:
            df_clean = df.copy()

        df_clean['Month'] = df_clean['Date'].dt.strftime('%m')
        monthly_agg = df_clean.groupby('Month')[col_rev].sum().reset_index()
        for _, row in monthly_agg.iterrows():
            data_monthly.append({'X': row['Month'], 'Revenue': row[col_rev], 'Year': year_label})

        df_clean['DayOfYear'] = df_clean['Date'].dt.dayofyear
        df_clean['BiWeek'] = (df_clean['DayOfYear'] - 1) // 14 + 1
        biweek_agg = df_clean.groupby('BiWeek')[col_rev].sum().reset_index()
        for _, row in biweek_agg.iterrows():
            label = f"Bi-Week {int(row['BiWeek']):02d}"
            data_biweek.append({'X': label, 'Revenue': row[col_rev], 'Year': year_label})

    process_df(df_curr, '今年')
    if df_last is not None and ensure_date_column(df_last):
        process_df(df_last, '去年')

    df_bw = pd.DataFrame(data_biweek, columns=cols) if data_biweek else pd.DataFrame(columns=cols)
    df_m = pd.DataFrame(data_monthly, columns=cols) if data_monthly else pd.DataFrame(columns=cols)
    return df_bw, df_m

# ================= 5. 核心利润计算 =================
def calculate_metrics_final(df_base):
    df = df_base.copy()
    qty = df['销量'].replace(0, 1)
    rev_after = df['退款后营收']
    df['ASP'] = rev_after / qty

    if 'Refund_Orders' not in df.columns:
        df['Refund_Orders'] = 0

    df['退款营收'] = df['Refund_Orders'] * df['ASP']
    df['退款前营收'] = rev_after + df['退款营收']

    rev_before_safe = df['退款前营收'].replace(0, 1)
    df['退款率'] = df['退款营收'] / rev_before_safe
    df['退款单数'] = df['Refund_Orders']

    for c in ['总达人佣金', '总样品费', '总广告投放费', '采购成本', '头程', '尾程', '关税']:
        if c not in df.columns:
            df[c] = 0
        else:
            df[c] = df[c].fillna(0)

    mkt_cost = df['总广告投放费'] + df['总达人佣金'] + df['总样品费']
    df['总营销费比'] = mkt_cost / rev_after.replace(0, 1)

    # 运营成本（按营收比例估算）
    df['仓租'] = rev_after * 0.005
    df['其他物流成本'] = rev_after * 0.003
    df['品牌费用'] = rev_after * 0.003
    df['平台佣金'] = rev_after * 0.06
    df['其他和售后'] = rev_after * 0.003

    all_costs = sum(df[c] for c in [
        '采购成本', '头程', '尾程', '关税',
        '仓租', '其他物流成本', '品牌费用', '平台佣金', '其他和售后',
        '总达人佣金', '总样品费', '总广告投放费'
    ])
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
        df[r_col] = df[val_col] / rev_safe if val_col in df.columns else 0

    return df

# ================= 6. 广告分析（V2 两层诊断） =================
def process_ads_data_v2(dfs, df_sku_final):
    df_ads = dfs.get('ads')
    df_mapping = dfs.get('mapping')
    df_spu_sku = dfs.get('spu_sku')

    if df_ads is None:
        return None, None, None, {}

    # Required columns
    col_pid = COLUMN_CONFIG['ads']['pid']
    col_cost = COLUMN_CONFIG['ads']['cost']
    col_rev = COLUMN_CONFIG['ads']['revenue']
    col_orders = COLUMN_CONFIG['ads']['orders']
    col_imp = COLUMN_CONFIG['ads']['impressions']
    col_clicks = COLUMN_CONFIG['ads']['clicks']
    col_video = COLUMN_CONFIG['ads']['video_title']

    required = [col_pid, col_cost, col_rev, col_orders, col_imp, col_clicks, col_video]
    if any(c not in df_ads.columns for c in required):
        return None, None, None, {"error": f"广告表缺少必要列：{[c for c in required if c not in df_ads.columns]}"}

    df_ads = df_ads.copy()
    df_ads['PID_Clean'] = clean_text(df_ads, col_pid)
    df_ads['Cost_Val'] = df_ads[col_cost].apply(clean_money)
    df_ads['Rev_Val'] = df_ads[col_rev].apply(clean_money)
    df_ads['Ord_Val'] = df_ads[col_orders].apply(clean_money)
    df_ads['Imp_Val'] = df_ads[col_imp].apply(clean_money)
    df_ads['Clk_Val'] = df_ads[col_clicks].apply(clean_money)
    df_ads['Vid_Title'] = clean_text(df_ads, col_video)

    # Rates: if exist use, else compute
    # CTR
    c_ctr = COLUMN_CONFIG['ads']['ctr']
    if c_ctr in df_ads.columns:
        df_ads['CTR'] = df_ads[c_ctr].apply(clean_percent)
    else:
        df_ads['CTR'] = df_ads.apply(lambda x: (x['Clk_Val'] / x['Imp_Val']) if x['Imp_Val'] > 0 else 0.0, axis=1)

    # CVR
    c_cvr = COLUMN_CONFIG['ads']['cvr']
    if c_cvr in df_ads.columns:
        df_ads['CVR'] = df_ads[c_cvr].apply(clean_percent)
    else:
        df_ads['CVR'] = df_ads.apply(lambda x: (x['Ord_Val'] / x['Clk_Val']) if x['Clk_Val'] > 0 else 0.0, axis=1)

    # 2s/6s rates
    c_2s = COLUMN_CONFIG['ads']['rate_2s']
    c_6s = COLUMN_CONFIG['ads']['rate_6s']
    df_ads['RATE_2S'] = df_ads[c_2s].apply(clean_percent) if c_2s in df_ads.columns else 0.0
    df_ads['RATE_6S'] = df_ads[c_6s].apply(clean_percent) if c_6s in df_ads.columns else 0.0

    # Build mapping: PID -> SKUs, SKU -> SPU
    pid_skus_map = {}
    if df_mapping is not None:
        df_mapping = df_mapping.copy()
        m_pid = find_col_by_keyword_fuzzy(df_mapping, ['product_id'])
        m_sku = find_col_by_keyword_fuzzy(df_mapping, ['sku'])
        if m_pid and m_sku:
            df_mapping['PID_Clean'] = clean_text(df_mapping, m_pid)
            df_mapping['SKU_Clean'] = clean_text(df_mapping, m_sku)
            pid_skus_map = df_mapping.groupby('PID_Clean')['SKU_Clean'].apply(lambda x: list(sorted(set(x.tolist())))).to_dict()

    sku_spu_map = build_sku_to_spu_dict(df_spu_sku) if df_spu_sku is not None else {}

    def pid_to_spu_str(pid):
        skus = pid_skus_map.get(pid, [])
        if not skus:
            return "未匹配"
        spus = sorted(list(set(sku_spu_map.get(s, s) for s in skus)))
        return ", ".join(spus) if spus else "未匹配"

    # ✅ 新增：广告明细行也带上 SPU（用于 video 映射）
    df_ads['SPU'] = df_ads['PID_Clean'].apply(pid_to_spu_str)

    # Prepare SKU margin map for CPA_Line (AUTO)
    sku_margin_map = {}
    sku_rev_weight_map = {}
    if df_sku_final is not None and not df_sku_final.empty:
        tmp = df_sku_final.copy()
        # Need: SKU, ASP, 运营成本率, 单件采购成本, 单件头程, 单件尾程, 单件关税(可选), 退款后营收(用作权重)
        for _, r in tmp.iterrows():
            sku = str(r.get('SKU', '')).strip().upper()
            if not sku:
                continue
            asp = float(r.get('ASP', 0) or 0)
            var_rate = float(r.get('运营成本率', 0) or 0)
            fixed = float(r.get('单件采购成本', 0) or 0) + float(r.get('单件头程', 0) or 0) + float(r.get('单件尾程', 0) or 0)
            fixed += float(r.get('单件关税', 0) or 0)
            margin = asp - (fixed + asp * var_rate)
            sku_margin_map[sku] = margin
            sku_rev_weight_map[sku] = float(r.get('退款后营收', 0) or 0)

    def compute_cpa_line(pid, aov):
        skus = pid_skus_map.get(pid, [])
        if skus and sku_margin_map:
            margins = []
            weights = []
            for s in skus:
                s2 = str(s).strip().upper()
                if s2 in sku_margin_map:
                    margins.append(sku_margin_map[s2])
                    weights.append(max(sku_rev_weight_map.get(s2, 0.0), 0.0))
            if margins:
                wsum = sum(weights)
                if wsum > 0:
                    return float(np.average(margins, weights=weights)), "AUTO"
                return float(np.mean(margins)), "AUTO"
        # DEFAULT
        return (aov / DEFAULT_ROAS_FOR_CPA_LINE) if aov > 0 else 0.0, "DEFAULT"

    # Aggregate to PID level
    df_prod = df_ads.groupby('PID_Clean').agg({
        'Cost_Val': 'sum',
        'Rev_Val': 'sum',
        'Ord_Val': 'sum',
        'Imp_Val': 'sum',
        'Clk_Val': 'sum'
    }).reset_index()

    df_prod.rename(columns={
        'PID_Clean': 'Product ID',
        'Cost_Val': 'Cost',
        'Rev_Val': 'Revenue',
        'Ord_Val': 'Orders'
    }, inplace=True)

    # Derived metrics
    df_prod['ROI'] = df_prod.apply(lambda x: (x['Revenue'] / x['Cost']) if x['Cost'] > 0 else 0.0, axis=1)
    df_prod['CPA'] = df_prod.apply(lambda x: (x['Cost'] / x['Orders']) if x['Orders'] > 0 else 0.0, axis=1)
    df_prod['CPM'] = df_prod.apply(lambda x: (x['Cost'] / x['Imp_Val'] * 1000) if x['Imp_Val'] > 0 else 0.0, axis=1)
    df_prod['CTR'] = df_prod.apply(lambda x: (x['Clk_Val'] / x['Imp_Val']) if x['Imp_Val'] > 0 else 0.0, axis=1)
    df_prod['CVR'] = df_prod.apply(lambda x: (x['Orders'] / x['Clk_Val']) if x['Clk_Val'] > 0 else 0.0, axis=1)
    df_prod['AOV'] = df_prod.apply(lambda x: (x['Revenue'] / x['Orders']) if x['Orders'] > 0 else 0.0, axis=1)

    df_prod['SPU'] = df_prod['Product ID'].apply(pid_to_spu_str)

    # Compute CPA_Line + source
    cpa_lines = []
    sources = []
    for _, r in df_prod.iterrows():
        line, src = compute_cpa_line(r['Product ID'], r['AOV'])
        cpa_lines.append(line)
        sources.append(src)
    df_prod['CPA_Line'] = cpa_lines
    df_prod['CPA_Line_Source'] = sources

    # Global thresholds (dynamic + floor)
    med_ctr = float(df_prod['CTR'].median()) if not df_prod.empty else CTR_FLOOR
    med_cvr = float(df_prod['CVR'].median()) if not df_prod.empty else CVR_FLOOR
    med_cpm = float(df_prod['CPM'].median()) if not df_prod.empty else CPM_FLOOR_HIGH

    thr_ctr_low = max(med_ctr, CTR_FLOOR)
    thr_cvr_low = max(med_cvr, CVR_FLOOR)
    thr_cpm_high = max(med_cpm, CPM_FLOOR_HIGH)

    def pid_status_and_diag(row):
        if row['Cost'] < COST_OBSERVE:
            return "⚪ 观察期", "花费太少，继续观察", "继续测/先别下结论"
        # 爆款
        if (row['ROI'] > ROI_BEST) and (row['CPA_Line'] > 0) and (row['CPA'] < row['CPA_Line']):
            return "🟢 爆款", "盈利且起量", "建议扩量（加预算/复制受众/复制素材）"
        # 亏损
        is_loss = (row['ROI'] < ROI_LOSS) or ((row['CPA_Line'] > 0) and (row['CPA'] > row['CPA_Line']))
        if is_loss:
            # 分叉诊断优先级：CVR低 -> CPM高 -> CTR低
            if row['CVR'] < thr_cvr_low:
                return "🔴 亏损", "CVR 低：流量来了接不住", "检查产品/价格/落地页/货不对板"
            if row['CPM'] > thr_cpm_high:
                return "🔴 亏损", "CPM 高：流量太贵", "调整受众/出新素材/避开高竞争时段"
            if row['CTR'] < thr_ctr_low:
                return "🔴 亏损", "CTR 低：没人点", "优化封面与开头3秒（钩子/对比/证据）"
            return "🔴 亏损", "综合偏低：ROI不达标", "优先改素材与商品页表达"
        # 灰区
        return "🟡 可优化", "接近盈亏线", "按 CTR/CVR/CPM 最弱项优化后继续测"

    df_prod[['Status', 'Diagnosis', 'Action']] = df_prod.apply(lambda x: pd.Series(pid_status_and_diag(x)), axis=1)

    # ✅ 需求2：产品盈亏诊断表格中 SPU 列放到 Product ID 左侧（只调顺序，不改数据）
    if 'SPU' in df_prod.columns and 'Product ID' in df_prod.columns:
        cols = df_prod.columns.tolist()
        new_cols = []
        for c in cols:
            if c not in ['SPU', 'Product ID']:
                new_cols.append(c)
        df_prod = df_prod[['SPU', 'Product ID'] + new_cols]

    # Video level aggregation
    df_video = df_ads.groupby('Vid_Title').agg({
        'Cost_Val': 'sum',
        'Rev_Val': 'sum',
        'Imp_Val': 'sum',
        'Clk_Val': 'sum',
        'Ord_Val': 'sum',
        'CTR': 'mean',
        'CVR': 'mean',
        'RATE_2S': 'mean',
        'RATE_6S': 'mean'
    }).reset_index()

    df_video.rename(columns={
        'Vid_Title': 'Video title',
        'Cost_Val': 'Cost',
        'Rev_Val': 'Revenue',
        'Ord_Val': 'Orders'
    }, inplace=True)

    df_video['ROI'] = df_video.apply(lambda x: (x['Revenue'] / x['Cost']) if x['Cost'] > 0 else 0.0, axis=1)

    # ✅ 需求3：video title 左侧增加 SPU + Product ID（从明细里取“最常出现的”映射）
    title_to_pid = df_ads.groupby('Vid_Title')['PID_Clean'].agg(lambda x: x.value_counts().idxmax() if len(x) else "").to_dict()
    title_to_spu = df_ads.groupby('Vid_Title')['SPU'].agg(lambda x: x.value_counts().idxmax() if len(x) else "未匹配").to_dict()
    df_video['Product ID'] = df_video['Video title'].map(lambda t: title_to_pid.get(str(t).strip().upper(), ""))
    df_video['SPU'] = df_video['Video title'].map(lambda t: title_to_spu.get(str(t).strip().upper(), "未匹配"))

    # Dynamic thresholds for video
    med_v_ctr = float(df_video['CTR'].median()) if not df_video.empty else CTR_FLOOR
    med_v_cvr = float(df_video['CVR'].median()) if not df_video.empty else CVR_FLOOR
    med_v_2s = float(df_video['RATE_2S'].median()) if not df_video.empty else RATE2S_FLOOR
    med_v_6s = float(df_video['RATE_6S'].median()) if not df_video.empty else RATE6S_FLOOR

    thr_v_ctr_high = max(med_v_ctr, CTR_FLOOR)
    thr_v_cvr_high = max(med_v_cvr, CVR_FLOOR)
    thr_v_2s_high = max(med_v_2s, RATE2S_FLOOR)
    # 6s 低：按“低于中位数且低于底线”更严格，这里取二者更小作为低阈值
    thr_v_6s_low = min(med_v_6s, RATE6S_FLOOR)
    thr_v_cvr_low = max(med_v_cvr, CVR_FLOOR)

    # ✅ 需求5：AI结论给出具体判断标准
    def classify_video(row):
        ctr = row.get('CTR', 0.0)
        cvr = row.get('CVR', 0.0)
        r2 = row.get('RATE_2S', 0.0)
        r6 = row.get('RATE_6S', 0.0)

        # 黄金：高CTR + 高2s + 高CVR
        if (ctr >= thr_v_ctr_high) and (r2 >= thr_v_2s_high) and (cvr >= thr_v_cvr_high):
            standard = f"标准：CTR≥{thr_v_ctr_high:.2%} 且 2s≥{thr_v_2s_high:.2%} 且 CVR≥{thr_v_cvr_high:.2%}"
            return "🥇 黄金素材", f"{standard}（开头吸睛+内容承接+转化精准）", "复制结构/开头套路，扩量投放"

        # 标题党：高CTR + 低6s
        if (ctr >= thr_v_ctr_high) and (r6 <= thr_v_6s_low):
            standard = f"标准：CTR≥{thr_v_ctr_high:.2%} 且 6s≤{thr_v_6s_low:.2%}"
            return "🎣 标题党", f"{standard}（骗点击，内容崩塌，用户6秒内大量流失）", "重剪前6秒承接，卖点前置"

        # 无效种草：高完播（2s或6s高） + 低CVR
        if ((r2 >= thr_v_2s_high) or (r6 >= max(med_v_6s, RATE6S_FLOOR))) and (cvr < thr_v_cvr_low):
            standard = f"标准：2s≥{thr_v_2s_high:.2%} 或 6s≥{max(med_v_6s, RATE6S_FLOOR):.2%} 且 CVR<{thr_v_cvr_low:.2%}"
            return "🌿 无效种草", f"{standard}（好看但不卖货/货不对板/缺证据镜头）", "强化购买理由/证据镜头/商品页一致性"

        standard = "标准：不满足黄金/标题党/无效种草（三类）或样本不足"
        return "🗑️ 其他", standard, "继续测试或归档"

    df_video[['Creative Type', 'AI_Conclusion', 'Next_Action']] = df_video.apply(lambda x: pd.Series(classify_video(x)), axis=1)

    # Hook vs Pitch matrix fields (for plotting)
    df_video['HookRate_2S'] = df_video['RATE_2S']

    # ✅ 需求3：把列顺序改成 SPU、Product ID、Video title 在最左侧
    if ('SPU' in df_video.columns) and ('Product ID' in df_video.columns) and ('Video title' in df_video.columns):
        vcols = df_video.columns.tolist()
        rest = [c for c in vcols if c not in ['SPU', 'Product ID', 'Video title']]
        df_video = df_video[['SPU', 'Product ID', 'Video title'] + rest]

    meta = {
        "thr_prod": {"CTR_low": thr_ctr_low, "CVR_low": thr_cvr_low, "CPM_high": thr_cpm_high},
        "thr_video": {
            "CTR_high": thr_v_ctr_high, "CVR_high": thr_v_cvr_high,
            "RATE2S_high": thr_v_2s_high, "RATE6S_low": thr_v_6s_low
        },
        "no_daily": True
    }

    # Return raw detail ads df also
    return df_prod, df_video, df_ads, meta

# ================= 7. 达人分析（V2） =================
def process_creator_data_v2(dfs, df_shop_raw, df_spu_raw):
    df_aff = dfs.get('affiliate')
    df_trans = dfs.get('transaction')
    res = {
        "overall": None,
        "leaderboard": None,
        "content_pie": None,
        "spu_perf": None,
        "commission_source_note": ""
    }

    # ---------- Overall & SPU performance from Transaction ----------
    if df_trans is not None:
        c_pid = COLUMN_CONFIG['transaction']['pid']
        c_aff_gmv = COLUMN_CONFIG['transaction']['aff_gmv']
        c_videos = COLUMN_CONFIG['transaction']['videos']
        c_lives = COLUMN_CONFIG['transaction']['lives']

        if c_pid in df_trans.columns and c_aff_gmv in df_trans.columns:
            df_trans = df_trans.copy()
            df_trans['PID_Clean'] = clean_text(df_trans, c_pid)
            df_trans['Affiliate_GMV'] = df_trans[c_aff_gmv].apply(clean_money)
            df_trans['Videos'] = pd.to_numeric(df_trans.get(c_videos, 0), errors='coerce').fillna(0)
            df_trans['Lives'] = pd.to_numeric(df_trans.get(c_lives, 0), errors='coerce').fillna(0)

            # Map PID -> SPU (first match)
            df_mapping = dfs.get('mapping')
            df_spu_sku = dfs.get('spu_sku')
            pid_skus_map = {}
            sku_spu_map = build_sku_to_spu_dict(df_spu_sku) if df_spu_sku is not None else {}

            if df_mapping is not None:
                df_mapping = df_mapping.copy()
                m_pid = find_col_by_keyword_fuzzy(df_mapping, ['product_id'])
                m_sku = find_col_by_keyword_fuzzy(df_mapping, ['sku'])
                if m_pid and m_sku:
                    df_mapping['PID_Clean'] = clean_text(df_mapping, m_pid)
                    df_mapping['SKU_Clean'] = clean_text(df_mapping, m_sku)
                    pid_skus_map = df_mapping.groupby('PID_Clean')['SKU_Clean'].apply(lambda x: list(sorted(set(x.tolist())))).to_dict()

            def get_spu(pid):
                skus = pid_skus_map.get(pid, [])
                spus = [sku_spu_map.get(s, s) for s in skus]
                return spus[0] if spus else "未匹配"

            df_trans['SPU'] = df_trans['PID_Clean'].apply(get_spu)

            spu_aff = df_trans.groupby('SPU').agg({
                'Affiliate_GMV': 'sum',
                'Videos': 'sum',
                'Lives': 'sum'
            }).reset_index()

            # Shop GMV before (refund-before) as denominator
            shop_gmv_before = 0.0
            if df_shop_raw is not None and not df_shop_raw.empty and '退款前营收' in df_shop_raw.columns:
                shop_gmv_before = float(df_shop_raw.iloc[0]['退款前营收'] or 0)

            # Overall achievement (shop level)
            aff_gmv_shop = float(df_trans['Affiliate_GMV'].sum())
            videos_shop = float(df_trans['Videos'].sum())
            res['overall'] = {
                "Affiliate_GMV": aff_gmv_shop,
                "Affiliate_Share": (aff_gmv_shop / shop_gmv_before) if shop_gmv_before > 0 else 0.0,
                "Videos": videos_shop,
                "Efficiency": (aff_gmv_shop / videos_shop) if videos_shop > 0 else 0.0
            }

            # SPU denominators from df_spu_raw (refund-before GMV)
            shop_gmv_map = {}
            if df_spu_raw is not None and not df_spu_raw.empty:
                if '退款前营收' in df_spu_raw.columns:
                    shop_gmv_map = dict(zip(df_spu_raw['SPU'], df_spu_raw['退款前营收']))
                else:
                    shop_gmv_map = dict(zip(df_spu_raw['SPU'], df_spu_raw.get('退款后营收', 0)))

            spu_aff['Shop_GMV_Before'] = spu_aff['SPU'].map(shop_gmv_map).fillna(0.0)
            spu_aff['Affiliate_Rate'] = spu_aff.apply(lambda x: (x['Affiliate_GMV'] / x['Shop_GMV_Before']) if x['Shop_GMV_Before'] > 0 else 0.0, axis=1)
            spu_aff['OutputPerVideo'] = spu_aff.apply(lambda x: (x['Affiliate_GMV'] / x['Videos']) if x['Videos'] > 0 else 0.0, axis=1)

            res['spu_perf'] = spu_aff.sort_values('Affiliate_GMV', ascending=False)

    # ---------- Leaderboard & content type pie from Affiliate ----------
    if df_aff is not None:
        c_name = COLUMN_CONFIG['affiliate']['creator']
        c_gmv = COLUMN_CONFIG['affiliate']['gmv']
        c_type = COLUMN_CONFIG['affiliate']['content_type']

        c_est_std = COLUMN_CONFIG['affiliate']['commission_est_std']
        c_est_ads = COLUMN_CONFIG['affiliate']['commission_est_ads']
        c_actual = COLUMN_CONFIG['affiliate']['commission']

        if c_name in df_aff.columns and c_gmv in df_aff.columns:
            df_aff = df_aff.copy()
            df_aff['GMV_Val'] = df_aff[c_gmv].apply(clean_money)

            # Commission logic V2: prefer est std + est ads; fallback to actual (with note)
            if (c_est_std in df_aff.columns) and (c_est_ads in df_aff.columns):
                df_aff['Comm_Val'] = df_aff[c_est_std].apply(clean_money) + df_aff[c_est_ads].apply(clean_money)
                res['commission_source_note'] = "佣金口径：Est.standard + Est.ShopAds（V2主口径）"
            elif c_actual in df_aff.columns:
                df_aff['Comm_Val'] = df_aff[c_actual].apply(clean_money)
                res['commission_source_note'] = "⚠️ 佣金口径退化：缺 Est.* 字段，使用 Actual Commission Payment"
            else:
                df_aff['Comm_Val'] = 0.0
                res['commission_source_note'] = "⚠️ 佣金字段缺失：Commission 将为 0"

            leaderboard = df_aff.groupby(c_name).agg({
                'GMV_Val': 'sum',
                'Comm_Val': 'sum',
                c_name: 'count'
            }).rename(columns={'GMV_Val': 'GMV', 'Comm_Val': 'Commission', c_name: 'Orders'}).reset_index()

            leaderboard['ROI'] = leaderboard.apply(lambda x: (x['GMV'] / x['Commission']) if x['Commission'] > 0 else 0.0, axis=1)
            res['leaderboard'] = leaderboard.sort_values('GMV', ascending=False)

            if c_type in df_aff.columns:
                pie_data = df_aff.groupby(c_type)['GMV_Val'].sum().reset_index()
                pie_data.columns = ['Type', 'GMV']
                res['content_pie'] = pie_data

    return res

# ================= 8. 主计算流程整合 =================
def run_calculation_logic_v2(dfs):
    # normalize headers
    for k, df in dfs.items():
        if df is not None:
            dfs[k] = normalize_headers(df)

    df_orders = dfs.get('orders')
    if df_orders is None:
        return None, {}

    # Required order columns
    col_sku = COLUMN_CONFIG['orders']['sku']
    col_rev = COLUMN_CONFIG['orders']['revenue']
    col_qty = COLUMN_CONFIG['orders']['qty']
    col_oid = COLUMN_CONFIG['orders']['order_id']
    missing = [c for c in [col_sku, col_rev, col_qty, col_oid] if c not in df_orders.columns]
    if missing:
        return None, {"error": f"订单表缺少核心列：{missing}"}

    df_orders = df_orders.copy()
    df_orders['SKU_Clean'] = clean_text(df_orders, col_sku)
    df_orders['OID_Clean'] = df_orders[col_oid].apply(convert_scientific_to_str)
    df_orders['Rev_Val'] = pd.to_numeric(df_orders[col_rev], errors='coerce').fillna(0.0)
    df_orders['Qty_Val'] = pd.to_numeric(df_orders[col_qty], errors='coerce').fillna(0.0)

    sku_to_spu_dict = build_sku_to_spu_dict(dfs.get('spu_sku'))
    df_orders['SPU'] = df_orders['SKU_Clean'].map(sku_to_spu_dict).fillna(df_orders['SKU_Clean'])

    time_str = "未知周期"
    max_date = None
    min_date = None
    if ensure_date_column(df_orders):
        dates = df_orders['Date'].dropna()
        if not dates.empty:
            min_date = dates.min()
            max_date = dates.max()
            time_str = f"{min_date.strftime('%Y-%m-%d')} ~ {max_date.strftime('%Y-%m-%d')}"

    # Cancelled / sample
    col_status = COLUMN_CONFIG['orders']['status']
    is_cancelled = df_orders[col_status].astype(str).str.strip().isin(['Cancelled', 'Canceled']) if col_status in df_orders.columns else False
    is_sample = (~is_cancelled) & (df_orders['Rev_Val'] == 0)

    # Cost maps
    map_p = get_cost_map(dfs.get('purchase'), ['采购', 'CNY'])
    map_h = get_cost_map(dfs.get('head'), ['头程', 'CNY'])
    map_t = get_cost_map(dfs.get('tail'), ['尾程', 'CNY'])

    # Sample cost
    df_sample = df_orders[is_sample].copy()
    sku_sample_cost = None
    if not df_sample.empty:
        df_sample['Unit_Cost'] = df_sample['SKU_Clean'].map(map_p).fillna(0) + df_sample['SKU_Clean'].map(map_h).fillna(0) + df_sample['SKU_Clean'].map(map_t).fillna(0)
        df_sample['Total_S'] = df_sample['Qty_Val'] * df_sample['Unit_Cost']
        sku_sample_cost = df_sample.groupby('SKU_Clean')['Total_S'].sum().reset_index().rename(columns={'SKU_Clean': 'SKU', 'Total_S': '总样品费'})

    # Affiliate commission (SKU-level) from affiliate orders (optional)
    df_aff = dfs.get('affiliate')
    sku_real_comm = None
    aff_note = ""
    if df_aff is not None:
        df_aff = df_aff.copy()
        c_oid = COLUMN_CONFIG['affiliate']['order_id']
        c_sku = COLUMN_CONFIG['affiliate']['sku']

        c_est_std = COLUMN_CONFIG['affiliate']['commission_est_std']
        c_est_ads = COLUMN_CONFIG['affiliate']['commission_est_ads']
        c_actual = COLUMN_CONFIG['affiliate']['commission']

        if c_oid in df_aff.columns and c_sku in df_aff.columns and ((c_est_std in df_aff.columns and c_est_ads in df_aff.columns) or (c_actual in df_aff.columns)):
            df_aff['OID_Clean'] = df_aff[c_oid].apply(convert_scientific_to_str)
            df_aff['SKU_Clean'] = clean_text(df_aff, c_sku)

            if (c_est_std in df_aff.columns) and (c_est_ads in df_aff.columns):
                df_aff['Comm_Val'] = df_aff[c_est_std].apply(clean_money) + df_aff[c_est_ads].apply(clean_money)
                aff_note = "佣金口径：Est.standard + Est.ShopAds（V2主口径）"
            else:
                df_aff['Comm_Val'] = df_aff[c_actual].apply(clean_money)
                aff_note = "⚠️ 佣金口径退化：缺 Est.*，使用 Actual Commission Payment"

            sku_real_comm = df_aff.groupby('SKU_Clean')['Comm_Val'].sum().reset_index().rename(columns={'SKU_Clean': 'SKU', 'Comm_Val': '总达人佣金'})

    # Ads cost allocation to SKU (optional)
    sku_ads_cost = None
    df_ads = dfs.get('ads')
    df_map = dfs.get('mapping')
    if df_ads is not None and df_map is not None:
        c_pid_ads = COLUMN_CONFIG['ads']['pid']
        c_cost_ads = COLUMN_CONFIG['ads']['cost']
        c_pid_map = find_col_by_keyword_fuzzy(df_map, ['product_id'])
        c_sku_map = find_col_by_keyword_fuzzy(df_map, ['sku'])
        if c_pid_ads in df_ads.columns and c_cost_ads in df_ads.columns and c_pid_map and c_sku_map:
            df_map = df_map.copy()
            df_map['PID_Clean'] = clean_text(df_map, c_pid_map)
            df_map['SKU_Clean'] = clean_text(df_map, c_sku_map)
            pid_grps = df_map.groupby('PID_Clean')['SKU_Clean'].apply(lambda x: list(sorted(set(x.tolist())))).reset_index()

            df_ads2 = df_ads.copy()
            df_ads2['PID_Clean'] = clean_text(df_ads2, c_pid_ads)
            df_ads2['Cost_Raw'] = df_ads2[c_cost_ads].apply(clean_money)

            # SKU revenue map from orders (exclude samples)
            sku_rev_map = df_orders[~is_sample & ~is_cancelled].groupby('SKU_Clean')['Rev_Val'].sum().to_dict()

            dist_list = []
            merged = pd.merge(df_ads2, pid_grps, on='PID_Clean', how='inner')
            for _, row in merged.iterrows():
                cost = float(row['Cost_Raw'] or 0)
                skus = row['SKU_Clean']
                if not skus:
                    continue
                revs = {s: float(sku_rev_map.get(s, 0.0)) for s in skus}
                tot = sum(revs.values())
                for s in skus:
                    share = cost * (revs[s] / tot) if tot > 0 else cost / max(len(skus), 1)
                    dist_list.append({'SKU': s, '总广告投放费': share})

            if dist_list:
                sku_ads_cost = pd.DataFrame(dist_list).groupby('SKU')['总广告投放费'].sum().reset_index()

    # Build SKU stats: refund-after revenue and qty from non-cancel and non-sample
    df_normal = df_orders[~is_cancelled].copy()
    df_refund = df_orders[is_cancelled].copy()

    sku_stats = df_normal.groupby(['SKU_Clean', 'SPU']).agg({'Rev_Val': 'sum', 'Qty_Val': 'sum'}).reset_index()
    sku_stats.rename(columns={'SKU_Clean': 'SKU', 'Rev_Val': '退款后营收', 'Qty_Val': '销量'}, inplace=True)

    # Merge add costs
    if sku_real_comm is not None:
        sku_stats = pd.merge(sku_stats, sku_real_comm, on='SKU', how='left')
    if sku_sample_cost is not None:
        sku_stats = pd.merge(sku_stats, sku_sample_cost, on='SKU', how='left')
    if sku_ads_cost is not None:
        sku_stats = pd.merge(sku_stats, sku_ads_cost, on='SKU', how='left')

    for c in ['总达人佣金', '总样品费', '总广告投放费']:
        if c not in sku_stats.columns:
            sku_stats[c] = 0.0
        else:
            sku_stats[c] = sku_stats[c].fillna(0.0)

    # Refund orders qty
    if not df_refund.empty:
        ref_agg = df_refund.groupby('SKU_Clean')['Qty_Val'].sum().reset_index().rename(columns={'SKU_Clean': 'SKU', 'Qty_Val': 'Refund_Orders'})
        sku_stats = pd.merge(sku_stats, ref_agg, on='SKU', how='left').fillna({'Refund_Orders': 0.0})
    else:
        sku_stats['Refund_Orders'] = 0.0

    # Unit costs
    sku_stats['单件采购成本'] = sku_stats['SKU'].map(map_p).fillna(0.0)
    sku_stats['单件头程'] = sku_stats['SKU'].map(map_h).fillna(0.0)
    sku_stats['单件尾程'] = sku_stats['SKU'].map(map_t).fillna(0.0)
    sku_stats['单件关税'] = 0.0

    sku_stats['采购成本'] = sku_stats['单件采购成本'] * sku_stats['销量']
    sku_stats['头程'] = sku_stats['单件头程'] * sku_stats['销量']
    sku_stats['尾程'] = sku_stats['单件尾程'] * sku_stats['销量']
    sku_stats['关税'] = 0.0

    # Profit metrics
    df_sku_raw = calculate_metrics_final(sku_stats)
    df_sku_out = format_dataframe(df_sku_raw, TARGET_COLUMNS_SKU)

    # Aggregate to SPU
    cols_to_sum = [
        '销量', '退款后营收', '退款前营收', 'Refund_Orders', '退款营收',
        '采购成本', '头程', '尾程', '关税',
        '仓租', '其他物流成本', '品牌费用', '平台佣金', '其他和售后',
        '总达人佣金', '总样品费', '总广告投放费'
    ]
    valid_cols = [c for c in cols_to_sum if c in df_sku_raw.columns]
    spu_agg = df_sku_raw.groupby('SPU')[valid_cols].sum().reset_index()
    df_spu_raw = calculate_metrics_final(spu_agg).sort_values(by='退款后营收', ascending=False)
    df_spu_out = format_dataframe(df_spu_raw, TARGET_COLUMNS_SPU)

    # Shop aggregate
    shop_agg = df_sku_raw[valid_cols].sum().to_frame().T
    df_shop_raw = calculate_metrics_final(shop_agg)
    df_shop_raw['数据周期'] = time_str
    df_shop_out = format_dataframe(df_shop_raw, TARGET_COLUMNS_SHOP_FINAL)

    # Ads diagnostics (V2)
    df_prod_ads, df_video_ads, df_ads_detail, ads_meta = process_ads_data_v2(dfs, df_sku_raw)

    # Creator data (V2)
    creator_data = process_creator_data_v2(dfs, df_shop_raw, df_spu_raw)
    if aff_note:
        creator_data['commission_source_note'] = (creator_data.get('commission_source_note', '') + " | " + aff_note).strip(" |")

    meta = {
        "time_str": time_str,
        "max_date": max_date,
        "min_date": min_date,
        "ads_meta": ads_meta
    }

    out = {
        "df_shop_out": df_shop_out, "df_spu_out": df_spu_out, "df_sku_out": df_sku_out,
        "df_shop_raw": df_shop_raw, "df_spu_raw": df_spu_raw, "df_sku_raw": df_sku_raw,
        "df_prod_ads": df_prod_ads, "df_video_ads": df_video_ads,
        "creator_data": creator_data,
        "dfs": dfs
    }
    return out, meta

# ================= 9. 智能文件识别读取器 =================
def load_uploaded_files(uploaded_files):
    dfs = {
        'orders': None, 'orders_last_year': None, 'ads': None, 'affiliate': None,
        'spu_sku': None, 'mapping': None, 'purchase': None, 'head': None, 'tail': None,
        'transaction': None
    }
    status_flags = {k: False for k in dfs.keys()}
    debug_logs = []

    file_list = uploaded_files if isinstance(uploaded_files, list) else []
    total = len(file_list)
    progress_bar = st.progress(0)
    status_text = st.empty()

    valid_exts = ['.csv', '.xlsx', '.xls']

    for i, file_obj in enumerate(file_list):
        is_local = isinstance(file_obj, str)
        fname = os.path.basename(file_obj) if is_local else file_obj.name
        fname_lower = fname.lower()

        if fname.startswith('.') or fname.startswith('~$'):
            continue
        if not any(fname_lower.endswith(ext) for ext in valid_exts):
            continue

        status_text.text(f"⏳ 正在解析: {fname}...")
        if total > 0:
            progress_bar.progress((i + 1) / total)

        try:
            if fname_lower.endswith('.csv'):
                df = pd.read_csv(file_obj, dtype=str, encoding='utf-8-sig')
            else:
                df = pd.read_excel(file_obj, dtype=str)

            df.columns = df.columns.astype(str).str.strip()
            cols = df.columns.tolist()
            log_info = f"📄 **{fname}**\n- 列前5: {cols[:5]}\n"
            match_type = "未匹配"

            # 1. Affiliate
            if COLUMN_CONFIG['affiliate']['creator'] in cols:
                dfs['affiliate'] = df; status_flags['affiliate'] = True
                match_type = "✅ 联盟订单表"

            # 2. Transaction
            elif COLUMN_CONFIG['transaction']['aff_gmv'] in cols:
                dfs['transaction'] = df; status_flags['transaction'] = True
                match_type = "✅ Transaction表"

            # 3. Ads
            elif 'Campaign name' in cols or 'ad group name' in cols:
                dfs['ads'] = df; status_flags['ads'] = True
                match_type = "✅ 广告表"

            # 4. last year orders
            elif '2025' in fname_lower:
                dfs['orders_last_year'] = df; status_flags['orders_last_year'] = True
                match_type = "✅ 去年订单表"

            # 5. main orders
            elif COLUMN_CONFIG['orders']['order_id'] in cols and COLUMN_CONFIG['orders']['sku'] in cols:
                dfs['orders'] = df; status_flags['orders'] = True
                match_type = "✅ 主订单表"

            # 6. Aux tables by filename
            elif 'spu' in fname_lower:
                dfs['spu_sku'] = df; status_flags['spu_sku'] = True; match_type = "SPU映射"
            elif 'pid' in fname_lower or 'mapping' in fname_lower:
                dfs['mapping'] = df; status_flags['mapping'] = True; match_type = "PID映射"
            elif '采购' in fname:
                dfs['purchase'] = df; status_flags['purchase'] = True; match_type = "采购表"
            elif '头程' in fname:
                dfs['head'] = df; status_flags['head'] = True; match_type = "头程表"
            elif '尾程' in fname:
                dfs['tail'] = df; status_flags['tail'] = True; match_type = "尾程表"

            log_info += f"- 判定结果: {match_type}"
            debug_logs.append(log_info)

        except Exception as e:
            st.error(f"❌ 读取文件 {fname} 失败: {str(e)}")
            debug_logs.append(f"❌ **{fname}** 读取失败: {str(e)}")

    time.sleep(0.2)
    status_text.text("✅ 解析完成！")
    progress_bar.empty()
    return dfs, status_flags, debug_logs

# ================= 10. 主程序 =================
def main():
    st.title("🚀 TikTok AI运营系统（利润 & 广告 & 达人）")

    with st.sidebar:
        st.header("📂 数据源设置")
        mode = st.radio("选择数据来源", ["⬆️ 手动上传文件", "📂 本地自动读取（调试用）"])

        uploaded_files = []
        if mode == "⬆️ 手动上传文件":
            st.info("💡 支持 xlsx/csv，自动忽略干扰文件。")
            uploaded_files = st.file_uploader("请上传业务数据表", accept_multiple_files=True, type=['xlsx', 'csv'])
        else:
            st.info("💡 正在扫描当前目录下的数据文件...")
            current_dir = os.getcwd()
            uploaded_files = [os.path.join(current_dir, f) for f in os.listdir(current_dir) if f.endswith(('.csv', '.xlsx', '.xls'))]
            st.write(f"找到 {len(uploaded_files)} 个文件")

        dfs, flags, logs = {}, {}, []
        if uploaded_files:
            dfs, flags, logs = load_uploaded_files(uploaded_files)

            st.markdown("### 📊 文件就位状态")
            with st.expander("财务核心数据", expanded=True):
                st.write(f"{'✅' if flags.get('orders') else '❌'} 订单表（必须）")
                st.write(f"{'✅' if flags.get('ads') else '❌'} 广告表")
                st.write(f"{'✅' if flags.get('purchase') else '❌'} 采购成本")
                st.write(f"{'✅' if flags.get('spu_sku') else '❌'} SPU映射")
                st.write(f"{'✅' if flags.get('mapping') else '❌'} PID映射（建议）")
            with st.expander("达人分析数据", expanded=True):
                st.write(f"{'✅' if flags.get('affiliate') else '❌'} 联盟订单表")
                st.write(f"{'✅' if flags.get('transaction') else '❌'} Transaction表")

            with st.expander("🕵️ 文件诊断详情（Debug）", expanded=False):
                for log in logs:
                    st.markdown(log)
                    st.divider()

        st.divider()
        st.subheader("🎯 目标设定（V2）")
        target_revenue = st.number_input("本月营收目标 ($)", value=0.0, step=1000.0)
        target_profit_rate = st.number_input("目标利润率（默认15%）", value=0.15, step=0.01, format="%.2f")

    if st.button("🚀 点击开始测算", type="primary", disabled=not flags.get('orders')):
        st.session_state['has_run'] = True
        with st.spinner("⏳ 正在进行：利润核算、广告诊断、达人分析..."):
            out, meta = run_calculation_logic_v2(dfs)
            if out is None:
                st.error(meta.get("error", "❌ 运行失败：未知错误"))
                st.session_state['has_run'] = False
            else:
                st.session_state['data'] = {"out": out, "meta": meta}
                st.session_state['targets'] = {"target_revenue": target_revenue, "target_profit_rate": target_profit_rate}

    if st.session_state.get('has_run') and st.session_state.get('data'):
        data = st.session_state['data']
        out = data['out']
        meta = data['meta']
        targets = st.session_state.get('targets', {"target_revenue": 0.0, "target_profit_rate": 0.15})

        df_shop_out = out['df_shop_out']
        df_spu_out = out['df_spu_out']
        df_sku_out = out['df_sku_out']

        df_shop_raw = out['df_shop_raw']
        df_spu_raw = out['df_spu_raw']
        df_prod_ads = out['df_prod_ads']
        df_video_ads = out['df_video_ads']
        creator_data = out['creator_data']
        dfs = out['dfs']

        time_str = meta.get("time_str", "未知周期")
        max_date = meta.get("max_date", None)
        min_date = meta.get("min_date", None)
        ads_meta = meta.get("ads_meta", {})

        # Shop core numbers from raw
        shop_row_raw = df_shop_raw.iloc[0]
        curr_rev_after = float(shop_row_raw.get('退款后营收', 0) or 0)
        curr_gmv_before = float(shop_row_raw.get('退款前营收', 0) or 0)
        curr_profit = float(shop_row_raw.get('利润额', 0) or 0)
        curr_profit_rate = float(shop_row_raw.get('利润率', 0) or 0)
        curr_refund_rate = float(shop_row_raw.get('退款率', 0) or 0)
        curr_mkt_rate = float(shop_row_raw.get('总营销费比', 0) or 0)

        target_revenue = float(targets.get("target_revenue", 0) or 0)
        target_profit_rate = float(targets.get("target_profit_rate", 0.15) or 0.15)

        mtd_achieve = (curr_rev_after / target_revenue) if target_revenue > 0 else 0.0

        # Time progress (best-effort)
        time_progress = 0.0
        if max_date is not None:
            try:
                d = pd.to_datetime(max_date)
                days_in_month = pd.Period(d.strftime("%Y-%m")).days_in_month
                time_progress = min(max(d.day / days_in_month, 0.0), 1.0)
            except:
                time_progress = 0.0

        # progress judgment
        def progress_label_revenue(ach, tp):
            if tp <= 0:
                return "—"
            if ach >= tp:
                return "🟢 进度健康"
            if ach >= tp - 0.05:
                return "🟡 轻微落后"
            return "🔴 明显落后"

        rev_judge = progress_label_revenue(mtd_achieve, time_progress) if target_revenue > 0 else "—"
        profit_judge = "🟢 达标" if curr_profit_rate >= target_profit_rate else "🔴 不达标"

        st.success(f"✅ 测算成功！数据周期: {time_str}")

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏠 经营总览", "📦 SPU 分析", "📄 SKU 明细", "📺 广告深度诊断", "🤝 达人合作分析", "🧠 AI 操盘手"])

        with tab1:
            st.markdown("### 📈 经营总览（V2）")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-title">📊 KPU 进度</div>
                  <b>营收目标</b>: ${target_revenue:,.0f} ｜ <b>实际</b>: ${curr_rev_after:,.0f}（达成 {mtd_achieve:.1%}）<br>
                  <b>月度时间进度</b>: {time_progress:.1%} ｜ <b>进度判定</b>: {rev_judge}<br><br>
                  <b>目标利润率</b>: {target_profit_rate:.1%} ｜ <b>实际利润率</b>: {curr_profit_rate:.1%} ｜ <b>判定</b>: {profit_judge}
                </div>
                """, unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-title">💰 大盘核心数据</div>
                  <b>退款前GMV</b>: ${curr_gmv_before:,.0f}<br>
                  <b>净利润</b>: ${curr_profit:,.0f}<br>
                  <b>退款率</b>: {curr_refund_rate:.1%} ｜ <b>营销费比</b>: {curr_mkt_rate:.1%}
                  <div class="note">退款前GMV = 退款后营收 +（退款单数×ASP估算退款额）</div>
                </div>
                """, unsafe_allow_html=True)

            # Trend charts
            trend_df_bw, trend_df_m = get_dual_trend_data(dfs.get('orders'), dfs.get('orders_last_year'))
            if trend_df_bw is not None and not trend_df_bw.empty:
                st.subheader("净营收趋势（双周/按月）")
                gran = st.radio("趋势粒度", ["双周（默认）", "按月"], horizontal=True)
                tdf = trend_df_bw if gran.startswith("双周") else trend_df_m
                chart = alt.Chart(tdf).mark_line(point=True).encode(
                    x=alt.X('X:N', title='周期', sort=None),
                    y=alt.Y('Revenue:Q', title='净营收 ($)'),
                    color=alt.Color('Year:N', title='年份'),
                    tooltip=[alt.Tooltip('Year:N'), alt.Tooltip('X:N'), alt.Tooltip('Revenue:Q', format=',.2f')]
                ).interactive()
                st.altair_chart(chart, use_container_width=True)

            st.subheader("店铺维度财务数据表")
            st.dataframe(df_shop_out, use_container_width=True)

        with tab2:
            # ✅ 需求1 + 需求6：SPU分析加入“盈亏分析”和“波士顿分析”
            st.markdown("## 📦 SPU 分析（V2）")

            st.markdown("### 🔥 1）盈亏分析（可视化）")
            left, right = st.columns([1.2, 1.0])

            # 利润贡献 Top10 SPU
            with left:
                st.markdown("#### 🏆 利润贡献 Top 10 SPU")
                if df_spu_raw is None or df_spu_raw.empty or ('SPU' not in df_spu_raw.columns) or ('利润额' not in df_spu_raw.columns):
                    st.warning("缺少 SPU 或 利润额 字段，无法生成利润贡献Top10。")
                else:
                    top_profit = df_spu_raw[['SPU', '利润额']].copy().sort_values('利润额', ascending=False).head(10)
                    bar = alt.Chart(top_profit).mark_bar().encode(
                        x=alt.X('利润额:Q', title='利润贡献值（净利润 $）'),
                        y=alt.Y('SPU:N', sort='-x', title='SPU'),
                        tooltip=[alt.Tooltip('SPU:N'), alt.Tooltip('利润额:Q', format=',.2f')]
                    ).properties(height=360)
                    st.altair_chart(bar, use_container_width=True)

            # 亏损警示榜
            with right:
                st.markdown("#### 🧨 亏损警示榜（负利润）")
                req_cols = ['SPU', '销量', '退款前营收', '退款单数', '退款后营收', '利润额', '利润率']
                miss = [c for c in req_cols if (df_spu_raw is None) or (c not in df_spu_raw.columns)]
                if miss:
                    st.warning(f"缺少字段：{miss}，无法生成亏损警示榜。")
                else:
                    loss_df = df_spu_raw[req_cols].copy()
                    loss_df = loss_df[loss_df['利润额'] < 0].sort_values('利润额', ascending=True).head(10)
                    loss_df = loss_df.rename(columns={'退款前营收': '退款前GMV', '退款单数': '退款数量'})
                    # 格式化展示
                    loss_df['利润率'] = loss_df['利润率'].apply(lambda x: f"{float(x):.2%}")
                    st.dataframe(loss_df, use_container_width=True, hide_index=True)

            st.divider()
            st.markdown("### 🧭 2）波士顿分析（SPU 矩阵）")

            # 波士顿矩阵：缺少“增长率/市场份额”的真实定义，这里用“营收占比（退款后营收占店铺）”作为 Share，
            # 用“利润率”作为 Quality/Margin 维度，便于运营快速分层。
            if df_spu_raw is None or df_spu_raw.empty:
                st.info("SPU raw 数据为空，无法做波士顿分析。")
            else:
                need = ['SPU', '退款后营收', '利润率']
                if any(c not in df_spu_raw.columns for c in need):
                    st.warning(f"缺少字段 {need}，无法做波士顿分析。")
                else:
                    bcg = df_spu_raw[['SPU', '退款后营收', '利润率', '销量']].copy()
                    total_rev = float(bcg['退款后营收'].sum() or 0)
                    bcg['营收占比'] = bcg['退款后营收'].apply(lambda x: (float(x) / total_rev) if total_rev > 0 else 0.0)

                    x_mid = float(bcg['营收占比'].median() if not bcg.empty else 0.0)
                    y_mid = float(bcg['利润率'].median() if not bcg.empty else 0.0)

                    def quad(row):
                        x = row['营收占比']; y = row['利润率']
                        if x >= x_mid and y >= y_mid:
                            return "⭐ Star（高占比&高利润）"
                        if x >= x_mid and y < y_mid:
                            return "🐄 Cash Cow（高占比&低利润）"
                        if x < x_mid and y >= y_mid:
                            return "❓ Question（低占比&高利润）"
                        return "🐶 Dog（低占比&低利润）"

                    bcg['象限'] = bcg.apply(quad, axis=1)

                    bubble = alt.Chart(bcg).mark_circle().encode(
                        x=alt.X('营收占比:Q', title='营收占比（退款后营收/店铺）'),
                        y=alt.Y('利润率:Q', title='利润率'),
                        size=alt.Size('退款后营收:Q', title='退款后营收($)'),
                        color=alt.Color('象限:N', title='波士顿象限'),
                        tooltip=[
                            alt.Tooltip('SPU:N'),
                            alt.Tooltip('象限:N'),
                            alt.Tooltip('退款后营收:Q', format=',.2f'),
                            alt.Tooltip('营收占比:Q', format='.2%'),
                            alt.Tooltip('利润率:Q', format='.2%'),
                            alt.Tooltip('销量:Q')
                        ]
                    ).properties(height=420).interactive()

                    vline = alt.Chart(pd.DataFrame({'x': [x_mid]})).mark_rule().encode(x='x:Q')
                    hline = alt.Chart(pd.DataFrame({'y': [y_mid]})).mark_rule().encode(y='y:Q')

                    st.altair_chart(bubble + vline + hline, use_container_width=True)

                    with st.expander("📌 波士顿象限口径说明（便于你对齐团队理解）", expanded=False):
                        st.markdown(f"""
- **横轴：营收占比** = SPU退款后营收 / 店铺退款后营收（用于代表“规模/份额”）
- **纵轴：利润率** = SPU利润额 / SPU退款后营收（用于代表“质量/可持续”）
- 分割线为**中位数**：营收占比≈{x_mid:.2%}，利润率≈{y_mid:.2%}
                        """)

            st.divider()
            st.markdown("### 📋 3）SPU 分析表（格式化展示）")
            st.dataframe(df_spu_out, use_container_width=True)

        with tab3:
            st.subheader("SKU 明细表（格式化展示）")
            st.dataframe(df_sku_out, use_container_width=True)

        with tab4:
            st.markdown("### 📺 广告深度诊断（V2：两层诊断）")

            if isinstance(ads_meta, dict) and ads_meta.get("error"):
                st.error(f"广告诊断不可用：{ads_meta.get('error')}")
            elif df_prod_ads is None or df_prod_ads.empty:
                st.info("💡 未上传广告表或广告表字段不足，暂无法进行广告诊断。")
            else:
                # Top KPIs
                total_cost = float(df_prod_ads['Cost'].sum() or 0)
                total_rev = float(df_prod_ads['Revenue'].sum() or 0)
                total_orders = float(df_prod_ads['Orders'].sum() or 0)
                total_imps = float(df_prod_ads['Imp_Val'].sum() or 0) if 'Imp_Val' in df_prod_ads.columns else 0.0

                roas = (total_rev / total_cost) if total_cost > 0 else 0.0
                cpa_all = (total_cost / total_orders) if total_orders > 0 else 0.0
                cpm_all = (total_cost / total_imps * 1000) if total_imps > 0 else 0.0

                ac1, ac2, ac3, ac4 = st.columns(4)
                ac1.metric("总广告费", f"${total_cost:,.0f}")
                ac2.metric("总 ROAS", f"{roas:.2f}")
                ac3.metric("整体 CPA", f"${cpa_all:,.2f}")
                ac4.metric("整体 CPM", f"${cpm_all:,.2f}")

                st.divider()
                st.subheader("层级一：产品（PID）盈亏诊断（这个品能不能打）")

                # Matrix chart
                c_chart = alt.Chart(df_prod_ads).mark_circle().encode(
                    x=alt.X('CPA:Q', title='CPA'),
                    y=alt.Y('ROI:Q', title='ROI (ROAS)'),
                    size=alt.Size('Cost:Q', title='Cost'),
                    color=alt.Color('Status:N', title='Status'),
                    tooltip=['Product ID', 'SPU', 'Status', 'Diagnosis', 'Cost', 'ROI', 'CPA', 'CPA_Line', 'CPA_Line_Source']
                ).interactive()
                st.altair_chart(c_chart, use_container_width=True)

                # Detail table
                df_show = df_prod_ads.copy()
                for c in ['ROI', 'CPA', 'CPM', 'CTR', 'CVR', 'CPA_Line']:
                    if c in df_show.columns:
                        df_show[c] = df_show[c].astype(float).round(2)

                # ✅ 需求2已经在 process_ads_data_v2 调顺序，这里只展示
                st.dataframe(df_show.sort_values('Cost', ascending=False), use_container_width=True)

                # Rule explanation
                thr = ads_meta.get("thr_prod", {})
                st.markdown(f"""
                <div class="kpi-card">
                  <div class="kpi-title">📌 判定规则说明（V2）</div>
                  <b>观察期</b>：Cost &lt; ${COST_OBSERVE:.0f}<br>
                  <b>🟢 爆款</b>：ROI &gt; {ROI_BEST:.1f} 且 CPA &lt; CPA毛利线（CPA_Line）<br>
                  <b>🔴 亏损</b>：ROI &lt; {ROI_LOSS:.1f} 或 CPA &gt; CPA_Line<br>
                  <b>🟡 可优化</b>：不属于爆款也不属于亏损<br><br>
                  <b>亏损分叉诊断优先级</b>：CVR低 → CPM高 → CTR低<br>
                  当前动态阈值（含底线）：CTR_low≈{thr.get('CTR_low', 0):.2%}，CVR_low≈{thr.get('CVR_low', 0):.2%}，CPM_high≈${thr.get('CPM_high', 0):.2f}<br>
                  <div class="warn">⚠️ 由于广告表无按天字段，本版本不提供“最近3天/近7天衰退预警（Fatigue Alert）”。</div>
                </div>
                """, unsafe_allow_html=True)

                st.divider()
                st.subheader("层级二：素材（Video）质量透视（这条视频好在哪）")

                if df_video_ads is None or df_video_ads.empty:
                    st.info("💡 未检测到视频标题字段或数据不足，无法进行素材分析。")
                else:
                    # Hook vs Pitch matrix (2s vs CTR)
                    df_v = df_video_ads.copy()
                    if 'HookRate_2S' in df_v.columns:
                        df_v['HookRate_2S'] = df_v['HookRate_2S'].astype(float)
                    if 'CTR' in df_v.columns:
                        df_v['CTR'] = df_v['CTR'].astype(float)

                    hook_chart = alt.Chart(df_v).mark_circle().encode(
                        x=alt.X('HookRate_2S:Q', title='2s 完播率（Hook）'),
                        y=alt.Y('CTR:Q', title='CTR（Pitch）'),
                        size=alt.Size('Cost:Q', title='Cost'),
                        color=alt.Color('Creative Type:N', title='类型'),
                        tooltip=['SPU', 'Product ID', 'Video title', 'Creative Type', 'ROI', 'CTR', 'CVR', 'RATE_2S', 'RATE_6S', 'AI_Conclusion']
                    ).interactive()
                    st.altair_chart(hook_chart, use_container_width=True)

                    # ✅ 需求4：表格格式化显示
                    show_cols = ['SPU', 'Product ID', 'Video title', 'Cost', 'Revenue', 'Orders', 'ROI',
                                 'CTR', 'CVR', 'RATE_2S', 'RATE_6S',
                                 'Creative Type', 'AI_Conclusion', 'Next_Action']
                    show_cols = [c for c in show_cols if c in df_video_ads.columns]
                    df_v_show = df_video_ads[show_cols].copy()

                    # ROI 2位；CTR/CVR/2s/6s 百分比2位
                    if 'ROI' in df_v_show.columns:
                        df_v_show['ROI'] = df_v_show['ROI'].astype(float).round(2)
                    for pc in ['CTR', 'CVR', 'RATE_2S', 'RATE_6S']:
                        if pc in df_v_show.columns:
                            df_v_show[pc] = df_v_show[pc].astype(float).apply(lambda x: f"{x:.2%}")

                    st.dataframe(df_v_show.sort_values('Cost', ascending=False), use_container_width=True)

                    # ✅ 补充：把“判断标准”也在页面说明里给到（便于团队对齐）
                    vthr = ads_meta.get("thr_video", {})
                    st.markdown(f"""
                    <div class="kpi-card">
                      <div class="kpi-title">🧪 素材分类标准（本周期动态阈值）</div>
                      <b>黄金素材</b>：CTR ≥ {vthr.get('CTR_high', 0):.2%} 且 2s ≥ {vthr.get('RATE2S_high', 0):.2%} 且 CVR ≥ {vthr.get('CVR_high', 0):.2%}<br>
                      <b>标题党</b>：CTR ≥ {vthr.get('CTR_high', 0):.2%} 且 6s ≤ {vthr.get('RATE6S_low', 0):.2%}<br>
                      <b>无效种草</b>： (2s ≥ {vthr.get('RATE2S_high', 0):.2%} 或 6s 高) 且 CVR 低于该周期 CVR 中位数/底线<br>
                      <div class="note">阈值由“中位数 + 底线”共同决定：既能自适应当期流量水位，也避免阈值过低失真。</div>
                    </div>
                    """, unsafe_allow_html=True)

        with tab5:
            st.markdown("### 🤝 达人合作分析（V2）")

            if creator_data.get('commission_source_note'):
                st.info(creator_data['commission_source_note'])

            # Overall achievement (from transaction)
            overall = creator_data.get('overall')
            if overall:
                oc1, oc2, oc3, oc4 = st.columns(4)
                oc1.metric("达人GMV（Transaction）", f"${overall['Affiliate_GMV']:,.0f}")
                oc2.metric("达人GMV占比（分母=退款前GMV）", f"{overall['Affiliate_Share']:.2%}")
                oc3.metric("上线视频数（Videos）", f"{overall['Videos']:,.0f}")
                oc4.metric("自建联视频效率（GMV/Video）", f"${overall['Efficiency']:,.2f}")
            else:
                st.warning("💡 未上传 Transaction 表：无法计算达人GMV占比、上线视频数、视频效率。")

            st.divider()

            # Leaderboard + pie
            if creator_data.get('leaderboard') is not None:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### 👑 达人贡献榜")
                    st.dataframe(creator_data['leaderboard'].head(15), use_container_width=True, hide_index=True)
                with c2:
                    st.markdown("#### 📊 场域分布（Content Type）")
                    if creator_data.get('content_pie') is not None:
                        pie = alt.Chart(creator_data['content_pie']).mark_arc(innerRadius=50).encode(
                            theta=alt.Theta('GMV:Q'), color='Type:N', tooltip=['Type', 'GMV']
                        )
                        st.altair_chart(pie, use_container_width=True)
            else:
                st.info("💡 未上传联盟订单表：无法生成达人贡献榜与场域分布。")

            st.divider()

            # SPU performance
            if creator_data.get('spu_perf') is not None:
                st.markdown("#### 📦 核心 SPU 带货表现（按 Affiliate GMV 降序）")
                df_sp = creator_data['spu_perf'].copy()
                if 'Affiliate_Rate' in df_sp.columns:
                    df_sp['Affiliate_Rate'] = df_sp['Affiliate_Rate'].astype(float).round(4)
                if 'OutputPerVideo' in df_sp.columns:
                    df_sp['OutputPerVideo'] = df_sp['OutputPerVideo'].astype(float).round(2)
                st.dataframe(df_sp, use_container_width=True, hide_index=True)
            else:
                st.info("💡 未上传 Transaction 或缺少 PID 映射：无法生成 SPU 级渗透与单视频产出。")

        with tab6:
            st.markdown("#### 🧠 AI 操盘手（占位）")
            if st.button("✨ 生成 Prompt"):
                st.code(f"请分析大盘数据：退款后营收 ${curr_rev_after:,.0f}，退款前GMV ${curr_gmv_before:,.0f}，利润率 {curr_profit_rate:.2%}...")

if __name__ == '__main__':
    main()


