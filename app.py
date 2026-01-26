import streamlit as st
import pandas as pd
import numpy as np
import io
import re
import altair as alt

# ================= 1. 页面基础配置 =================
st.set_page_config(
    page_title="华青TikTok 业务数据系统 (Pro版)",
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

def clean_percent(val):
    """清洗百分比字符串为小数"""
    if pd.isna(val): return 0.0
    s = str(val).strip().replace('%', '')
    try:
        return float(s) / 100.0
    except:
        return 0.0

def find_col_by_keyword(df, keywords):
    """
    注意：此函数逻辑是必须包含列表中所有关键词才算匹配。
    如果只需匹配其中一个，请调用时只传一个关键词，分多次调用。
    """
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

# ================= 5. 核心利润计算逻辑 (SKU级) =================

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

# ================= 6. 广告深度诊断引擎 (v3.4 - 修复视频列识别) =================

def process_ads_data(dfs, sku_stats_df):
    """
    广告数据处理、双重映射、智能诊断核心函数
    """
    df_ads = dfs['ads']
    df_mapping = dfs['mapping']
    df_spu_sku = dfs['spu_sku']
    
    if df_ads is None: return None, None, None, 0.0

    # 1. 基础清洗
    col_pid = find_col_by_keyword(df_ads, ['product id'])
    col_cost = find_col_by_keyword(df_ads, ['cost'])
    
    # --- 修复核心：尝试多种方式找视频列 ---
    col_video = find_col_by_keyword(df_ads, ['video title']) # 优先找 Title
    if not col_video:
        col_video = find_col_by_keyword(df_ads, ['video id']) # 其次找 ID
    
    col_rev = find_col_by_keyword(df_ads, ['gross revenue'])
    col_orders = find_col_by_keyword(df_ads, ['sku orders'])
    col_imp = find_col_by_keyword(df_ads, ['impressions'])
    col_clicks = find_col_by_keyword(df_ads, ['clicks'])
    col_ctr = find_col_by_keyword(df_ads, ['click rate'])
    col_cvr = find_col_by_keyword(df_ads, ['conversion rate'])
    
    col_2s = find_col_by_keyword(df_ads, ['2-second'])
    col_6s = find_col_by_keyword(df_ads, ['6-second'])

    if not (col_pid and col_cost): return None, None, None, 0.0

    df_ads[col_pid] = clean_text(df_ads, col_pid)
    df_ads['Cost'] = df_ads[col_cost].apply(clean_money)
    df_ads['Revenue'] = df_ads[col_rev].apply(clean_money) if col_rev else 0
    df_ads['Orders'] = df_ads[col_orders].apply(clean_money) if col_orders else 0
    df_ads['Impressions'] = df_ads[col_imp].apply(clean_money) if col_imp else 0
    df_ads['Clicks'] = df_ads[col_clicks].apply(clean_money) if col_clicks else 0
    
    # 百分比处理
    for c, target in [(col_ctr, 'CTR'), (col_cvr, 'CVR'), (col_2s, '2s_Rate'), (col_6s, '6s_Rate')]:
        if c: df_ads[target] = df_ads[c].apply(clean_percent)
        else: df_ads[target] = 0.0

    # 2. 计算 CPM 和 CPA (Row Level)
    df_ads['CPM'] = df_ads.apply(lambda x: (x['Cost'] / x['Impressions'] * 1000) if x['Impressions'] > 0 else 0, axis=1)

    # 3. 双重映射引擎 (PID -> SKU -> SPU)
    pid_skus_map = {}
    sku_spu_map = {}
    
    # 3.1 构建 PID -> SKU List
    if df_mapping is not None:
        m_pid = find_col_by_keyword(df_mapping, ['product_id'])
        m_sku = find_col_by_keyword(df_mapping, ['sku'])
        if m_pid and m_sku:
            df_mapping[m_pid] = clean_text(df_mapping, m_pid)
            df_mapping[m_sku] = clean_text(df_mapping, m_sku)
            # 去除空值
            valid_map = df_mapping.dropna(subset=[m_pid, m_sku])
            pid_skus_map = valid_map.groupby(m_pid)[m_sku].apply(list).to_dict()

    # 3.2 构建 SKU -> SPU
    if df_spu_sku is not None:
        sku_spu_map = build_sku_to_spu_dict(df_spu_sku)

    # 3.3 映射执行函数
    def get_spus_str(pid):
        skus = pid_skus_map.get(pid, [])
        if not skus: return "未匹配"
        spus = set()
        for s in skus:
            s_clean = str(s).strip().upper()
            if s_clean in sku_spu_map:
                spus.add(sku_spu_map[s_clean])
            else:
                spus.add(s_clean) # 如果找不到SPU，用SKU兜底
        return ", ".join(sorted(list(spus))) if spus else "未匹配"

    # 4. 保本 ROI 计算引擎
    # 准备 SKU 成本字典
    sku_cost_dict = {}
    if not sku_stats_df.empty:
        for _, row in sku_stats_df.iterrows():
            s = str(row['SKU']).strip().upper()
            asp = row['ASP']
            if asp <= 0: continue
            
            # 刚性绝对成本
            fixed_cost = row['单件采购成本'] + row['单件头程'] + row['单件尾程']
            # 变动比例成本 (平台佣金6% + 仓租0.5% + 物流0.3% + 品牌0.3% + 售后0.3% = 7.4%)
            var_rate = 0.06 + 0.005 + 0.003 + 0.003 + 0.003
            var_cost = asp * var_rate
            
            # 总非广告成本
            total_non_ads_cost = fixed_cost + var_cost
            
            # 单件毛利空间
            margin_val = asp - total_non_ads_cost
            
            # 保本 ROI = ASP / Margin
            if margin_val > 0.01: # 避免除以0
                breakeven_roi = asp / margin_val
            else:
                breakeven_roi = 999.0 # 成本倒挂，无法保本
            
            sku_cost_dict[s] = breakeven_roi

    def get_pid_breakeven(pid):
        skus = pid_skus_map.get(pid, [])
        if not skus: return 1.6 # 默认兜底
        rois = [sku_cost_dict.get(s, 1.6) for s in skus if s in sku_cost_dict]
        if not rois: return 1.6
        return sum(rois) / len(rois)

    # 5. 聚合：Product ID 维度 (用于红黑榜)
    df_prod = df_ads.groupby(col_pid).agg({
        'Cost': 'sum', 'Revenue': 'sum', 'Orders': 'sum',
        'Impressions': 'sum', 'Clicks': 'sum'
    }).reset_index()
    
    # 核心指标计算
    df_prod['ROI'] = df_prod.apply(lambda x: x['Revenue']/x['Cost'] if x['Cost']>0 else 0, axis=1)
    df_prod['CPA'] = df_prod.apply(lambda x: x['Cost']/x['Orders'] if x['Orders']>0 else 0, axis=1)
    df_prod['CPM'] = df_prod.apply(lambda x: x['Cost']/x['Impressions']*1000 if x['Impressions']>0 else 0, axis=1)
    
    # 计算衍生指标
    df_prod['CTR'] = df_prod.apply(lambda x: x['Clicks']/x['Impressions'] if x['Impressions']>0 else 0, axis=1)
    df_prod['CVR'] = df_prod.apply(lambda x: x['Orders']/x['Clicks'] if x['Clicks']>0 else 0, axis=1)
    
    # 注入 SPU 和 保本 ROI
    df_prod['SPU'] = df_prod[col_pid].apply(get_spus_str)
    df_prod['Product ID'] = df_prod[col_pid] # 保留 PID 列
    df_prod['Breakeven_ROI'] = df_prod[col_pid].apply(get_pid_breakeven)
    
    # 全局平均 CPM (用于判断流量贵不贵)
    global_avg_cpm = df_prod[df_prod['Impressions']>1000]['CPM'].median()
    if np.isnan(global_avg_cpm): global_avg_cpm = 20.0
    
    # === 5.1 智能分叉诊断树 (Logic Tree) ===
    def diagnose_row(row):
        # 0. 观察期过滤
        if row['Cost'] < 50: 
            return "观察期", "-"
            
        # 1. 严重亏损判定
        if row['Breakeven_ROI'] >= 999:
            return "🚨 严重亏损 (负毛利)", "成本结构崩坏"
            
        # 2. 红黑榜判定
        if row['ROI'] >= row['Breakeven_ROI']:
            # 盈利
            return "🌟 盈利爆款", "利润健康"
        else:
            # 亏损 - 开始分叉归因
            status = "📉 隐形亏损"
            reasons = []
            
            # 归因 A: 点击率低
            if row['CTR'] < 0.01: 
                reasons.append("素材太差(CTR<1%)")
            
            # 归因 B: 流量贵
            if row['CPM'] > 25 or row['CPM'] > global_avg_cpm * 1.3:
                reasons.append(f"流量太贵(CPM>${row['CPM']:.0f})")
                
            # 归因 C: 转化差
            if row['CVR'] < 0.01:
                reasons.append("内功不行(CVR<1%)")
                
            # 兜底
            if not reasons: reasons.append("ROI综合偏低")
            
            return status, " / ".join(reasons)
            
    # 应用诊断
    df_prod[['Status', 'Diagnosis']] = df_prod.apply(
        lambda x: pd.Series(diagnose_row(x)), axis=1
    )

    # 6. 聚合：视频素材维度
    df_video = None
    if col_video:
        df_ads[col_video] = clean_text(df_ads, col_video)
        df_video = df_ads.groupby(col_video).agg({
            'Cost': 'sum', 'Revenue': 'sum', 'Impressions': 'sum', 'Clicks': 'sum', 'Orders': 'sum'
        }).reset_index()
        
        # 加权计算率值
        wm = lambda x: np.average(x, weights=df_ads.loc[x.index, "Cost"]) if df_ads.loc[x.index, "Cost"].sum() > 0 else x.mean()
        rates = df_ads.groupby(col_video).agg({
            'CTR': wm, 'CVR': wm, '2s_Rate': wm, '6s_Rate': wm
        }).reset_index()
        
        df_video = pd.merge(df_video, rates, on=col_video)
        df_video['ROI'] = df_video.apply(lambda x: x['Revenue']/x['Cost'] if x['Cost']>0 else 0, axis=1)
        
        # 象限分类逻辑 - 严格执行您定义的 3 类
        avg_ctr_video = df_video['CTR'].median()
        avg_cvr_video = df_video['CVR'].median()
        avg_2s = df_video['2s_Rate'].median()
        avg_6s = df_video['6s_Rate'].median()
        
        def label_video(row):
            # 🥇 黄金素材：高 CTR + 高 2s 完播率 + 高 CVR
            if row['CTR'] > avg_ctr_video and row['2s_Rate'] > avg_2s and row['CVR'] > avg_cvr_video:
                return "🥇 黄金素材", "开头吸睛+内容种草+转化精准"
            
            # 🎣 标题党素材：高 CTR + 低 6s 完播率
            if row['CTR'] > avg_ctr_video and row['6s_Rate'] < avg_6s:
                return "🎣 标题党", "开头骗点击+内容崩塌"
            
            # 📉 无效种草素材：高 6s 完播率 (或 100%) + 低 CVR
            if row['6s_Rate'] > avg_6s and row['CVR'] < avg_cvr_video:
                return "📉 无效种草", "视频好看但无购买欲/货不对板"
            
            return "🗑️ 其他/待优化", "表现平庸"
            
        df_video[['Type', 'AI_Comment']] = df_video.apply(
            lambda x: pd.Series(label_video(x)), axis=1
        )

    return df_prod, df_video, df_ads, global_avg_cpm

# ================= 7. 主计算流程整合 =================

def run_calculation_logic(dfs):
    # 1. 基础处理
    for key, df in dfs.items():
        if df is not None: dfs[key] = normalize_headers(df)
    
    df_orders = dfs['orders']
    if df_orders is None: return None, None, None, None, None, "无订单数据", None

    # --- (复用之前的 orders 处理逻辑) ---
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

    # --- 成本 & 佣金计算 (复用逻辑) ---
    df_orders['Comm'] = 0
    sku_real_comm = None
    df_aff = dfs['affiliate']
    if df_aff is not None:
        # (保持原有佣金逻辑不变)
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

    # --- 订单清洗 ---
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

    # --- 成本映射 ---
    map_p = get_cost_map(dfs['purchase'], ['采购', 'CNY'])
    map_h = get_cost_map(dfs['head'], ['头程', 'CNY'])
    map_t = get_cost_map(dfs['tail'], ['尾程', 'CNY'])
    master_skus = df_orders[['SPU', col_sku]].drop_duplicates().rename(columns={col_sku: 'SKU'})

    # --- SKU Stats 计算 ---
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

    # --- 样品费 & 单件成本 ---
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
    
    sku_stats['单件采购成本'] = sku_stats['SKU'].map(map_p).fillna(0)
    sku_stats['单件头程'] = sku_stats['SKU'].map(map_h).fillna(0)
    sku_stats['单件尾程'] = sku_stats['SKU'].map(map_t).fillna(0)
    sku_stats['采购成本'] = sku_stats['单件采购成本'] * sku_stats['销量']
    sku_stats['头程'] = sku_stats['单件头程'] * sku_stats['销量']
    sku_stats['尾程'] = sku_stats['单件尾程'] * sku_stats['销量']
    sku_stats['关税'] = 0

    sku_stats = pd.merge(sku_stats, sku_real_comm, on=['SPU', 'SKU'], how='left')
    sku_stats['总达人佣金'] = sku_stats['总达人佣金'].fillna(0)

    # --- 广告分摊 ---
    # 这里依然保留 SKU 维度的分摊逻辑，用于经营报表
    sku_stats['总广告投放费'] = 0
    df_ads = dfs['ads']
    df_map = dfs['mapping']
    if df_ads is not None and df_map is not None:
        # (复用原有的分摊代码)
        pid_c = find_col_by_keyword(df_map, ['product_id'])
        sku_mc = find_col_by_keyword(df_map, ['sku'])
        if pid_c and sku_mc:
            df_map[pid_c] = clean_text(df_map, pid_c)
            df_map[sku_mc] = clean_text(df_map, sku_mc)
            pid_grps = df_map.groupby(pid_c)[sku_mc].apply(list).reset_index()
            ad_pid = find_col_by_keyword(df_ads, ['product id'])
            if ad_pid:
                df_ads[ad_pid] = clean_text(df_ads, ad_pid)
                # 重新读取 Cost (上面 process_ads_data 也会读，但这里是为了经营报表)
                ad_cost_col = find_col_by_keyword(df_ads, ['cost'])
                df_ads['Cost_Raw'] = df_ads[ad_cost_col].apply(clean_money)
                
                rev_map = dict(zip(sku_stats['SKU'], sku_stats['退款前营收']))
                dist_list = []
                merged_ads = pd.merge(df_ads, pid_grps, left_on=ad_pid, right_on=pid_c, how='inner')
                for _, row in merged_ads.iterrows():
                    cost = row['Cost_Raw']
                    skus = row[sku_mc]
                    if not skus: continue
                    revs = {s: rev_map.get(s, 0) for s in skus}
                    tot = sum(revs.values())
                    for s in skus:
                        if tot > 0: share = cost * (revs[s] / tot)
                        else: share = cost / len(skus)
                        dist_list.append({'SKU': s, 'AdsCost': share})
                if dist_list:
                    ads_df_dist = pd.DataFrame(dist_list)
                    ads_agg = ads_df_dist.groupby('SKU')['AdsCost'].sum().reset_index().rename(columns={'AdsCost': '总广告投放费'})
                    sku_stats = pd.merge(sku_stats, ads_agg, on='SKU', how='left')
                    if '总广告投放费_y' in sku_stats.columns: sku_stats['总广告投放费'] = sku_stats['总广告投放费_y'].fillna(0)
                    else: sku_stats['总广告投放费'] = sku_stats.get('总广告投放费', 0).fillna(0)

    # --- 经营报表汇总 ---
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

    # --- 调用广告深度分析 ---
    df_prod_ads, df_video_ads, _, avg_cpm = process_ads_data(dfs, sku_stats)

    return df_shop_out, df_spu_out, df_sku_out, df_prod_ads, df_video_ads, time_str, max_date, avg_cpm

# ================= 8. 智能文件识别读取器 =================
def load_uploaded_files(uploaded_files):
    dfs = {
        'orders': None, 'orders_last_year': None, 'ads': None, 'affiliate': None,
        'spu_sku': None, 'mapping': None, 'purchase': None, 'head': None, 'tail': None
    }
    status_flags = {k: False for k in dfs.keys()}
    
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name.lower()
        try:
            if filename.endswith('.csv'): df = pd.read_csv(uploaded_file, dtype=str)
            else: df = pd.read_excel(uploaded_file, dtype=str)
        except: continue
            
        if '2025' in filename or '去年' in filename:
            dfs['orders_last_year'] = df; status_flags['orders_last_year'] = True
        elif '广告' in filename or 'ads' in filename:
            dfs['ads'] = df; status_flags['ads'] = True
        elif '联盟' in filename or 'affiliate' in filename:
            dfs['affiliate'] = df; status_flags['affiliate'] = True
        elif 'spu' in filename:
            dfs['spu_sku'] = df; status_flags['spu_sku'] = True
        elif 'pid' in filename or 'mapping' in filename:
            dfs['mapping'] = df; status_flags['mapping'] = True
        elif '采购' in filename or 'purchase' in filename:
            dfs['purchase'] = df; status_flags['purchase'] = True
        elif '头程' in filename or 'head' in filename:
            dfs['head'] = df; status_flags['head'] = True
        elif '尾程' in filename or 'tail' in filename:
            dfs['tail'] = df; status_flags['tail'] = True
        elif '订单' in filename or 'order' in filename:
            dfs['orders'] = df; status_flags['orders'] = True
            
    return dfs, status_flags

# ================= 9. 主程序 =================
def main():
    st.title("🚀 华青TikTok 业务数据系统 (Pro版)")
    
    # --- 侧边栏 ---
    with st.sidebar:
        st.header("📂 1. 拖拽上传文件")
        st.info("💡 提示：一次性选中所有文件拖进来即可，系统会自动识别。")
        uploaded_files = st.file_uploader("请上传业务数据表 (支持 xlsx/csv)", accept_multiple_files=True, type=['xlsx', 'csv'])
        
        if uploaded_files:
            with st.spinner("⏳ 正在智能解析文件..."):
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
                st.write(f"{'✅' if flags['mapping'] else '⚠️'} PID映射")
                st.write(f"{'✅' if flags['head'] else '❌'} 头程成本")
                st.write(f"{'✅' if flags['tail'] else '❌'} 尾程成本")
        else:
            dfs = {}; flags = {}

        st.divider()
        st.subheader("🎯 2. 目标设定")
        target_revenue = st.number_input("本月营收目标 ($)", value=0.0, step=1000.0)
        target_profit_rate = st.number_input("目标利润率 (%)", value=15.0, step=0.5) / 100.0

    # 主操作按钮
    if st.button("🚀 点击开始测算", type="primary", disabled=not flags.get('orders')):
        st.session_state['has_run'] = True
        with st.spinner("⏳ 正在进行：全链路成本计算、广告归因、利润核算..."):
            try:
                # 运行核心逻辑
                res = run_calculation_logic(dfs)
                df_shop, df_spu, df_sku, df_prod_ads, df_video_ads, time_str, max_date, avg_cpm = res
                
                if df_shop is None: st.error("❌ 订单表为空或格式错误"); st.stop()
                    
                st.session_state['data'] = {
                    'dfs': dfs, 'df_shop': df_shop, 'df_spu': df_spu, 'df_sku': df_sku,
                    'df_prod_ads': df_prod_ads, 'df_video_ads': df_video_ads,
                    'time_str': time_str, 'max_date': max_date, 'avg_cpm': avg_cpm
                }
            except Exception as e:
                st.error(f"❌ 运行错误: {str(e)}")
                import traceback; st.code(traceback.format_exc()); st.session_state['has_run'] = False

    # 结果展示
    if st.session_state.get('has_run') and st.session_state.get('data'):
        data = st.session_state['data']
        df_shop = data['df_shop']; df_spu = data['df_spu']; df_sku = data['df_sku']
        df_prod_ads = data['df_prod_ads']; df_video_ads = data['df_video_ads']
        time_str = data['time_str']; max_date = data['max_date']; dfs = data['dfs']
        avg_cpm = data['avg_cpm']

        shop_row = df_shop.iloc[0]
        curr_rev = shop_row['退款后营收']
        
        if pd.notna(max_date):
            days_in_month = pd.Period(max_date, freq='M').days_in_month
            time_progress = max_date.day / days_in_month
        else: time_progress = 0
        mtd_achieve = curr_rev / target_revenue if target_revenue > 0 else 0
        pace_status = "🔴 落后" if mtd_achieve < time_progress else "🟢 超前"
        
        yoy_data = calculate_yoy_metrics(dfs['orders'], dfs.get('orders_last_year'))
        trend_df_bw, trend_df_m = get_dual_trend_data(dfs['orders'], dfs.get('orders_last_year'))

        st.success(f"✅ 测算成功！数据周期: {time_str}")
        
        # Tab 分页
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 经营总览", "📦 SPU 分析", "📄 SKU 明细", "📺 广告深度诊断", "🤖 AI 操盘手"])
        
        with tab1:
            st.markdown("### 📈 经营概览 (Dashboard)")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""<div class="kpi-card">
                <div class="kpi-title">📊 KPI 进度</div>
                <b>营收目标</b>: ${target_revenue:,.0f} | <b>实际</b>: ${curr_rev:,.0f} ({mtd_achieve:.1%})<br>
                <b>时间进度</b>: {time_progress:.1%} | <b>状态</b>: {pace_status}
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="kpi-card">
                <div class="kpi-title">💰 利润核心</div>
                <b>净利润</b>: ${shop_row['利润额']:,.0f} | <b>利润率</b>: {shop_row['利润率']}<br>
                <b>退款率</b>: {shop_row['退款率']} | <b>营销费比</b>: {shop_row['总营销费比']}
                </div>""", unsafe_allow_html=True)
            
            # --- 趋势维度切换控制 (v3.5 更新点) ---
            st.markdown("### 📊 趋势对比")
            trend_type = st.radio(
                "选择时间维度", 
                ["📅 按双周 (Bi-Week)", "🌙 按月度 (Monthly)"], 
                horizontal=True,
                label_visibility="collapsed"
            )

            # 根据选择加载不同数据
            data_to_plot = None
            x_title = ""
            
            if trend_type == "📅 按双周 (Bi-Week)":
                data_to_plot = trend_df_bw
                x_title = "双周周期"
            else:
                data_to_plot = trend_df_m
                x_title = "月份"

            if data_to_plot is not None and not data_to_plot.empty:
                chart = alt.Chart(data_to_plot).mark_line(point=True).encode(
                    x=alt.X('X', title=x_title, sort=None), # sort=None 保持原有排序
                    y=alt.Y('Revenue', title='净营收 ($)'),
                    color=alt.Color('Year', title='年份', scale=alt.Scale(domain=['今年', '去年'], range=['#ff0050', '#c3cfe2'])),
                    tooltip=['Year', 'X', 'Revenue']
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("⚠️ 暂无足够数据生成趋势图")

        with tab2: st.dataframe(df_spu, use_container_width=True)
        with tab3: st.dataframe(df_sku, use_container_width=True)

        with tab4:
            st.markdown("### 📺 广告深度诊断 (Ads Diagnosis)")
            if df_prod_ads is not None:
                # 1. 顶部：AI 全局综述 (新增)
                total_spu = len(df_prod_ads)
                profitable = len(df_prod_ads[df_prod_ads['Status'].str.contains('盈利')])
                loss_making = len(df_prod_ads[df_prod_ads['Status'].str.contains('亏损')])
                
                # 统计亏损主因
                loss_df = df_prod_ads[df_prod_ads['Status'].str.contains('亏损')]
                main_reason = "无明显聚集"
                if not loss_df.empty:
                    reason_counts = loss_df['Diagnosis'].value_counts()
                    main_reason = reason_counts.index[0] if not reason_counts.empty else "综合ROI低"

                st.info(f"""
                🤖 **AI 投放综述**：
                本次共分析 **{total_spu}** 个 SPU。其中 **{profitable}** 个盈利爆款，**{loss_making}** 个亏损预警。
                🚩 **最大短板**：亏损产品中，最常见的问题是 **“{main_reason}”**。
                🌊 **流量水位**：全局平均 CPM 为 **${avg_cpm:.2f}**。
                """)
                
                # 2. 核心指标卡
                ac1, ac2, ac3, ac4 = st.columns(4)
                ac1.metric("总广告费", f"${df_prod_ads['Cost'].sum():,.0f}")
                ac2.metric("总 ROAS", f"{df_prod_ads['Revenue'].sum()/df_prod_ads['Cost'].sum():.2f}")
                ac3.metric("平均 CPA", f"${df_prod_ads['Cost'].sum()/df_prod_ads['Orders'].sum():.2f}")
                ac4.metric("平均 CPM", f"${avg_cpm:.2f}")

                st.divider()
                
                # 3. 产品表现矩阵
                st.subheader("1. 产品表现矩阵 (Product Matrix)")
                c_chart = alt.Chart(df_prod_ads).mark_circle().encode(
                    x=alt.X('CPA', title='CPA (获客成本)'),
                    y=alt.Y('ROI', title='ROAS (投产比)'),
                    size='Cost',
                    color=alt.Color('Status', scale=alt.Scale(domain=['🌟 盈利爆款', '📉 隐形亏损', '🚨 严重亏损 (负毛利)', '观察期'], range=['#2ecc71', '#f1c40f', '#e74c3c', '#95a5a6'])),
                    tooltip=['SPU', 'Status', 'Cost', 'ROI', 'Breakeven_ROI', 'Diagnosis']
                ).interactive()
                st.altair_chart(c_chart, use_container_width=True)
                
                # 4. 详细诊断表 (修复排序问题)
                st.subheader("2. 详细诊断表 (SPU Level)")
                
                # 准备展示数据，保持数值类型以便排序
                df_show = df_prod_ads[['SPU', 'Product ID', 'Cost', 'ROI', 'Breakeven_ROI', 'CPA', 'CPM', 'CTR', 'CVR', 'Status', 'Diagnosis']].copy()
                
                # 默认按 Cost 降序排列
                df_show = df_show.sort_values(by='Cost', ascending=False)
                
                # 使用 column_config 进行格式化渲染，而不是转成字符串
                st.dataframe(
                    df_show,
                    use_container_width=True,
                    column_config={
                        "Cost": st.column_config.NumberColumn("Cost", format="$%d"),
                        "ROI": st.column_config.NumberColumn("ROI", format="%.2f"),
                        "Breakeven_ROI": st.column_config.NumberColumn("保本ROI", format="%.2f"),
                        "CPA": st.column_config.NumberColumn("CPA", format="$%.2f"),
                        "CPM": st.column_config.NumberColumn("CPM", format="$%.2f"),
                        "CTR": st.column_config.NumberColumn("CTR", format="%.2f%%"),
                        "CVR": st.column_config.NumberColumn("CVR", format="%.2f%%"),
                    },
                    hide_index=True
                )

                # 5. 素材维度深度分析 (新增)
                if df_video_ads is not None:
                    st.divider()
                    st.subheader("3. 素材内容诊所 (Creative Clinic)")
                    
                    # 统计素材分布
                    bad_creatives = df_video_ads[df_video_ads['Type'].isin(['🎣 标题党', '📉 无效种草'])]
                    wasted_budget = bad_creatives['Cost'].sum()
                    
                    st.warning(f"⚠️ **素材预警**：检测到 **{len(bad_creatives)}** 条问题素材（标题党/无效种草），共浪费预算 **${wasted_budget:,.0f}**，建议优先优化。")
                    
                    vc1, vc2 = st.columns(2)
                    
                    with vc1:
                        st.markdown("#### 🥇 黄金素材榜 (Top Winners)")
                        st.caption("标准：CTR高 + 2s完播高 + CVR高")
                        gold_df = df_video_ads[df_video_ads['Type'].str.contains('黄金')].sort_values('ROI', ascending=False).head(5)
                        st.dataframe(
                            gold_df[['Video title', 'ROI', 'AI_Comment']], 
                            use_container_width=True, hide_index=True,
                            column_config={"ROI": st.column_config.NumberColumn(format="%.2f")}
                        )

                    with vc2:
                        st.markdown("#### 🗑️ 问题素材榜 (Top Losers)")
                        st.caption("标准：标题党 (骗点击) 或 无效种草 (不转化)")
                        bad_df_show = bad_creatives.sort_values('Cost', ascending=False).head(5)
                        st.dataframe(
                            bad_df_show[['Video title', 'Cost', 'Type', 'AI_Comment']], 
                            use_container_width=True, hide_index=True,
                            column_config={"Cost": st.column_config.NumberColumn(format="$%d")}
                        )
                    
                    # 散点图
                    st.markdown("#### 素材分布图")
                    v_chart = alt.Chart(df_video_ads).mark_circle().encode(
                        x=alt.X('CTR', title='CTR (点击率)'),
                        y=alt.Y('CVR', title='CVR (转化率)'),
                        color=alt.Color('Type', legend=alt.Legend(title="素材类型")),
                        size='Cost',
                        tooltip=['Video title', 'Type', 'AI_Comment', 'CTR', 'CVR', '2s_Rate', '6s_Rate']
                    ).interactive()
                    st.altair_chart(v_chart, use_container_width=True)

        with tab5:
            st.markdown("#### 🧠 AI 操盘手")
            if st.button("✨ 生成全盘诊断 Prompt"):
                # 自动提取数据
                top_loss = df_prod_ads[df_prod_ads['Status'].str.contains('亏损')].sort_values('Cost', ascending=False).head(3)
                loss_txt = ""
                for _, r in top_loss.iterrows():
                    loss_txt += f"- SPU [{r['SPU']}]: 花费${r['Cost']:.0f}, 实际ROI {r['ROI']:.2f} (保本需 {r['Breakeven_ROI']:.2f}), 诊断: {r['Diagnosis']}\n"
                
                top_win = df_prod_ads[df_prod_ads['Status'].str.contains('爆款')].sort_values('Cost', ascending=False).head(3)
                win_txt = ""
                for _, r in top_win.iterrows():
                    win_txt += f"- SPU [{r['SPU']}]: 花费${r['Cost']:.0f}, 实际ROI {r['ROI']:.2f}\n"

                prompt = f"""
你是一名 TikTok Shop 资深操盘手。请根据以下系统自动归因的数据，撰写周报：

【1. 大盘数据】
- GMV: ${shop_row['退款前营收']:,.0f}, 净利: ${shop_row['利润额']:,.0f}
- 广告花费: ${df_prod_ads['Cost'].sum():,.0f}, 整体 ROAS: {df_prod_ads['Revenue'].sum()/df_prod_ads['Cost'].sum():.2f}
- 流量水温: 平均 CPM ${avg_cpm:.2f}

【2. 重点异常 (红黑榜)】
🚨 亏损严重 (Top 3 Losers):
{loss_txt}
🌟 盈利爆款 (Top 3 Winners):
{win_txt}

【3. 任务要求】
请输出结构化报告：
1. **止损行动**：针对上述亏损品，结合具体的诊断原因（如 CPM 贵、CTR 低），给出直接的操作建议（改素材？降出价？）。
2. **扩量机会**：针对盈利品，给出放量策略。
3. **素材风向**：基于 CPM 水位，判断当前大盘竞争态势。
"""
                st.code(prompt)

if __name__ == '__main__':
    main()