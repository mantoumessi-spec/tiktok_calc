import streamlit as st
import pandas as pd
import numpy as np
import re
import altair as alt
import time
import os
from datetime import datetime

from manual_content import (
    MANUAL_QUICKSTART_MD,
    MANUAL_HOME_MD,
    MANUAL_SPU_MD,
    MANUAL_ADS_MD,
    MANUAL_CREATOR_MD,
    MANUAL_AI_MD,
    MANUAL_RULES_MD,
)





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
    '采购成本率', '头程成本率', '关税成本率', '尾程费率',
    '仓租费率', '其他物流费用率', '品牌费率', '平台费率',
    '售后成本率', '达人佣金费率', '样品费率', '广告费率'
]
TARGET_COLUMNS_SPU = [col for col in TARGET_COLUMNS_SKU if col not in ['SKU', '单件采购成本', '单件头程', '单件关税', '单件尾程', '单件样品成本']]
TARGET_COLUMNS_SHOP = [col for col in TARGET_COLUMNS_SPU if col not in ['SPU', '类别']]
TARGET_COLUMNS_SHOP_FINAL = ['数据周期'] + TARGET_COLUMNS_SHOP + ['2025年同期退款后营收', '退款后营收_YOY%']

# ================= 加载2025年财务官报数据 =================
def load_2025_financial_report():
    """加载2025年财务官报，返回每月费率数据字典"""
    try:
        df_2025 = pd.read_excel('2025年TideWe财务管报.xlsx')
        # 第一列是指标名，其余是每月数据
        df_2025 = df_2025.set_index(df_2025.columns[0])
        # 列名是datetime，转为月份字符串
        month_cols = {}
        for col in df_2025.columns[1:]:  # 跳过全年累计
            if hasattr(col, 'strftime') and not isinstance(col, str):
                month = col.strftime('%Y-%m')
                month_cols[month] = df_2025[col].dropna().to_dict()  # 去掉nan
        return month_cols
    except Exception as e:
        print(f"无法加载2025年财务官报: {e}")
        return {}


def load_2025_orders_data(df_spu_sku=None, sku_to_spu_dict=None):
    """
    加载2025年订单数据，计算店铺/SPU/类目维度的月度营收
    返回：
    - shop_2025: {month: 退款后营收}
    - spu_2025: {(spu, month): 退款后营收}
    - category_2025: {(category, month): 退款后营收}
    """
    try:
        df_2025 = pd.read_excel('订单表_2025.xlsx')
        
        # 标准化列名
        df_2025.columns = df_2025.columns.astype(str).str.strip()
        
        # 找到关键列
        col_sku = None
        for c in df_2025.columns:
            if 'seller sku' in c.lower() or c.lower() == 'sku':
                col_sku = c
                break
        
        col_revenue = None
        for c in df_2025.columns:
            if '营收' in c or 'revenue' in c.lower():
                col_revenue = c
                break
        
        col_status = None
        for c in df_2025.columns:
            if 'status' in c.lower():
                col_status = c
                break
        
        col_time = None
        for c in df_2025.columns:
            if 'created time' in c.lower() or 'time' in c.lower():
                col_time = c
                break
        
        col_spu_in_order = None
        for c in df_2025.columns:
            if c.upper() == 'SPU':
                col_spu_in_order = c
                break
        
        if not col_sku or not col_revenue:
            print("2025年订单表缺少关键列")
            return {}, {}, {}
        
        # 数据清洗
        df = df_2025.copy()
        df['SKU_Clean'] = df[col_sku].astype(str).str.replace(r'[\u200b\ufeff]', '', regex=True).str.strip().str.upper()
        df['Rev_Val'] = pd.to_numeric(df[col_revenue], errors='coerce').fillna(0.0)
        
        # 退款状态处理
        if col_status:
            df['Is_Cancel'] = df[col_status].astype(str).str.strip().isin(['Cancelled', 'Canceled'])
        else:
            df['Is_Cancel'] = False
        
        # 时间处理
        if col_time:
            df['Date'] = pd.to_datetime(df[col_time], errors='coerce')
            df['Month'] = df['Date'].dt.strftime('%Y-%m')
        else:
            df['Month'] = '2025-01'  # 默认
        
        # 只保留非取消订单计算营收
        df_normal = df[~df['Is_Cancel']].copy()
        
        # 映射SPU
        if col_spu_in_order:
            df_normal['SPU'] = df_normal[col_spu_in_order]
        elif sku_to_spu_dict:
            df_normal['SPU'] = df_normal['SKU_Clean'].map(sku_to_spu_dict).fillna(df_normal['SKU_Clean'])
        else:
            df_normal['SPU'] = df_normal['SKU_Clean']
        
        # 店铺维度：按月汇总
        shop_2025 = df_normal.groupby('Month')['Rev_Val'].sum().to_dict()
        
        # SPU维度：按SPU+月汇总
        spu_2025 = df_normal.groupby(['SPU', 'Month'])['Rev_Val'].sum().to_dict()
        
        # 类目维度：需要映射三级类目
        category_2025 = {}
        # 类目数据需要从采购成本表获取映射，这里先返回空，后续处理
        
        return shop_2025, spu_2025, category_2025
        
    except Exception as e:
        print(f"无法加载2025年订单数据: {e}")
        return {}, {}, {}


def calculate_yoy_2025_by_category(purchase_df, sku_to_category_map):
    """
    基于2025年采购成本表中的三级类目，计算每个类目的月度营收
    （简化处理：用SKU在2025年的营收汇总）
    """
    # 这个函数需要在有2025年SKU映射到类目的基础上计算
    # 目前返回空，后续可以根据需要完善
    return {}

# ================= 3. 基础工具函数 =================

def _split_tokens(x):
    if pd.isna(x):
        return []
    s = str(x)
    parts = re.split(r"[,\s;/|]+", s)
    return [p.strip() for p in parts if p.strip()]

def build_spu_video_table(df_ads: pd.DataFrame, pid_to_spu: dict, observe_cost: float = 50.0) -> pd.DataFrame:
    """
    输入：广告明细（至少包含 Video title + PID + Cost/Revenue/Orders/Impressions/Clicks + 若有 2s/6s/CTR/CVR）
    输出：SPU-Video 粒度汇总表（含 ROI、Hook、CTR、CVR 等）
    """
    if df_ads is None or df_ads.empty:
        return pd.DataFrame()

    # 统一列名（按你项目常见写法做兼容）
    col_video = "Video title" if "Video title" in df_ads.columns else ("Video" if "Video" in df_ads.columns else None)
    col_pid = "PID" if "PID" in df_ads.columns else ("Product ID" if "Product ID" in df_ads.columns else None)
    if col_video is None or col_pid is None:
        return pd.DataFrame()

    df = df_ads.copy()

    # 尝试把核心数值列转成 float
    for c in ["Cost", "Revenue", "Orders", "Impressions", "Clicks"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # 统一 rate 字段为 0~1
    for c in ["CTR", "CVR", "2s", "6s", "2s_rate", "6s_rate", "Hook_2s", "Hold_6s"]:
        if c in df.columns:
            df[c] = normalize_rate_series(df[c])

    # 展开：一条广告行 -> 多个 SPU（等额分摊）
    alloc_cols = [c for c in ["Cost", "Revenue", "Orders", "Impressions", "Clicks"] if c in df.columns]
    rate_cols = [c for c in ["2s", "6s", "2s_rate", "6s_rate", "Hook_2s", "Hold_6s", "CTR", "CVR"] if c in df.columns]

    expanded = []
    for _, r in df.iterrows():
        pids = _split_tokens(r[col_pid])
        spus = set()
        for pid in pids:
            spu_val = pid_to_spu.get(str(pid), None)
            # 兼容 spu_val 可能是 "spu1,spu2" 或 list
            if spu_val is None:
                continue
            if isinstance(spu_val, (list, tuple, set)):
                spus |= set([str(x) for x in spu_val if str(x).strip()])
            else:
                spus |= set(_split_tokens(spu_val))

        if not spus:
            spus = {"(unmapped)"}

        w = 1.0 / len(spus)
        for spu in spus:
            rr = r.copy()
            rr["SPU"] = spu
            for c in alloc_cols:
                rr[c] = (rr[c] if pd.notna(rr[c]) else 0.0) * w

            # rate 字段用“Impressions 加权平均”最稳（如果没 Impressions 就用 Cost）
            if "Impressions" in df.columns and pd.notna(rr.get("Impressions", np.nan)) and rr["Impressions"] > 0:
                rr["_w_rate"] = rr["Impressions"]
            elif "Cost" in df.columns and pd.notna(rr.get("Cost", np.nan)) and rr["Cost"] > 0:
                rr["_w_rate"] = rr["Cost"]
            else:
                rr["_w_rate"] = 1.0

            expanded.append(rr)

    ex = pd.DataFrame(expanded)
    if ex.empty:
        return pd.DataFrame()

    # 聚合：SPU + Video
    group_cols = ["SPU", col_video]
    agg = {c: "sum" for c in alloc_cols}
    agg["_w_rate"] = "sum"
    for c in rate_cols:
        agg[c] = "sum"  # 先把 “rate * weight” 的合计做出来

    # 把 rate 变成加权平均：先乘权重
    for c in rate_cols:
        ex[c] = ex[c] * ex["_w_rate"]

    g = ex.groupby(group_cols, dropna=False).agg(agg).reset_index()

    # 恢复 rate
    for c in rate_cols:
        g[c] = g.apply(lambda x: safe_div(x[c], x["_w_rate"]), axis=1)

    # 衍生 CTR/CVR（如果有 clicks/impressions/orders）
    if "Clicks" in g.columns and "Impressions" in g.columns:
        g["CTR_calc"] = g.apply(lambda x: safe_div(x["Clicks"], x["Impressions"]), axis=1)
        if "CTR" not in g.columns or g["CTR"].isna().mean() > 0.5:
            g["CTR"] = g["CTR_calc"]

    if "Orders" in g.columns and "Clicks" in g.columns:
        g["CVR_calc"] = g.apply(lambda x: safe_div(x["Orders"], x["Clicks"]), axis=1)
        if "CVR" not in g.columns or g["CVR"].isna().mean() > 0.5:
            g["CVR"] = g["CVR_calc"]

    # 统一 Hook / Hold 字段名（页面使用这两个更直观）
    if "Hook_2s" not in g.columns:
        if "2s_rate" in g.columns: g["Hook_2s"] = g["2s_rate"]
        elif "2s" in g.columns: g["Hook_2s"] = g["2s"]
    if "Hold_6s" not in g.columns:
        if "6s_rate" in g.columns: g["Hold_6s"] = g["6s_rate"]
        elif "6s" in g.columns: g["Hold_6s"] = g["6s"]

    # ROI
    if "Revenue" in g.columns and "Cost" in g.columns:
        g["ROI"] = g.apply(lambda x: safe_div(x["Revenue"], x["Cost"]), axis=1)

    # 观察门槛
    g["Observe_Cost_Threshold"] = observe_cost
    return g


import re
import numpy as np
import pandas as pd

def _to_float_safe(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(",", "")
    if s == "" or s.lower() in {"nan", "none", "null", "-"}:
        return np.nan
    # 处理中文百分号
    s = s.replace("％", "%")
    # 去掉可能的多余字符
    s = re.sub(r"[^\d\.\-%]", "", s)
    if s in {"", "-", "%"}:
        return np.nan
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    return float(s)

def normalize_rate_series(series: pd.Series) -> pd.Series:
    """
    把各种输入统一成 0~1 的小数（Altair 可用 .0% 格式展示）。
    规则：
    - "2.3%" -> 0.023
    - 2.3 -> 如果列里存在 >1.5 的值，视作 0~100 的百分比 -> 0.023
    - 0.023 -> 保持不变
    """
    s = series.map(_to_float_safe)
    # 若最大值明显 > 1，基本可判定是 0~100 的“百分比数值”
    vmax = s.max(skipna=True)
    if pd.notna(vmax) and vmax > 1.5:
        s = s / 100.0
    return s

def safe_div(a, b):
    a = float(a) if pd.notna(a) else np.nan
    b = float(b) if pd.notna(b) else np.nan
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b

def normalize_headers(df):
    if df is None:
        return None
    df.columns = df.columns.astype(str).str.strip()
    return df

def clean_text(df, col_name):
    if col_name in df.columns:
        return df[col_name].astype(str).str.replace(r'[\u200b\ufeff]', '', regex=True).str.strip().str.upper()
    return pd.Series([""] * len(df), index=df.index)

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
    """
    基础成本映射（向后兼容）
    """
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


def get_cost_map_v2(cost_df, keywords, country_filter=None, platform_filter=None, platform_fallback=None):
    """
    增强版成本映射，支持国家和平台筛选
    
    参数:
    - cost_df: 成本表DataFrame
    - keywords: 成本列的关键词列表（如 ['采购', 'CNY']）
    - country_filter: 国家筛选值（如 '美国'），None表示不筛选
    - platform_filter: 平台筛选值（如 'TikTok'），None表示不筛选
    - platform_fallback: 备选平台值（如 'Shopify'），当platform_filter无匹配时使用
    
    返回:
    - cost_map: {SKU: USD成本} 字典
    - source_note: 成本来源备注（用于尾程成本追踪）
    """
    source_note = ""  # 成本来源备注
    
    if cost_df is None:
        return {}, source_note
    
    # 1. 找到成本列
    target_col = find_col_by_keyword_fuzzy(cost_df, keywords)
    if not target_col:
        return {}, source_note
    
    # 2. 找到SKU列
    sku_col = find_col_by_keyword_fuzzy(cost_df, ['sku'])
    if not sku_col:
        return {}, source_note
    
    # 3. 复制数据
    df = cost_df.copy()
    df['SKU_Clean'] = df[sku_col].astype(str).str.replace(r'[\u200b\ufeff]', '', regex=True).str.strip().str.upper()
    df['Clean_Cost'] = df[target_col].apply(clean_money)
    
    # 4. 国家筛选（如果指定）
    if country_filter:
        country_col = find_col_by_keyword_fuzzy(df, ['国家', 'country'])
        if country_col:
            # 标准化国家值进行比较
            df['country_norm'] = df[country_col].astype(str).str.strip().str.upper()
            country_norm_filter = str(country_filter).strip().upper()
            df = df[df['country_norm'] == country_norm_filter].copy()
            if df.empty:
                # 国家筛选后无数据，返回空（让调用方决定如何处理）
                return {}, source_note
    
    # 5. 平台筛选（如果指定）
    if platform_filter:
        platform_col = find_col_by_keyword_fuzzy(df, ['平台', 'platform', '销售渠道'])
        if platform_col:
            df['platform_norm'] = df[platform_col].astype(str).str.strip().str.upper()
            platform_norm_filter = str(platform_filter).strip().upper()
            
            # 先尝试主平台筛选
            df_primary = df[df['platform_norm'] == platform_norm_filter].copy()
            
            if not df_primary.empty:
                # 主平台有数据
                df = df_primary
                source_note = f"{platform_filter}"
            elif platform_fallback:
                # 主平台无数据，尝试备选平台
                platform_norm_fallback = str(platform_fallback).strip().upper()
                df_fallback = df[df['platform_norm'] == platform_norm_fallback].copy()
                
                if not df_fallback.empty:
                    df = df_fallback
                    source_note = f"⚠️ 使用{platform_fallback}数据（无{platform_filter}数据）"
                else:
                    # 备选平台也无数据，返回空
                    return {}, source_note
            else:
                # 无备选平台，返回空
                return {}, source_note
    
    # 6. 计算USD成本
    df['USD'] = df['Clean_Cost'] * EXCHANGE_RATE
    
    # 7. 提取三级类目映射（如果存在）
    category_map = {}
    category_col = find_col_by_keyword_fuzzy(df, ['三级类目', '类目', 'category'])
    if category_col:
        # 去重：同一个SKU可能有多行（不同国家/平台），取第一个非空的三级类目
        for _, row in df.iterrows():
            sku = row['SKU_Clean']
            cat = row.get(category_col)
            if pd.notna(cat) and str(cat).strip() and sku not in category_map:
                category_map[sku] = str(cat).strip()
    
    return dict(zip(df['SKU_Clean'], df['USD'])), source_note, category_map

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

def safe_div(a, b, default=np.nan):
    try:
        if pd.isna(a) or pd.isna(b) or float(b) == 0:
            return default
        return float(a) / float(b)
    except:
        return default

def load_traffic_funnel_data(df_traffic, current_month):
    """
    从流量和出单占比表提取多周期数据
    文件结构：
    - 第0行：指标名
    - 第1行：2026年1月
    - 第2行：2026年2月（上周期）
    - 第3行：2026年3月（本周期）
    """
    if df_traffic is None or df_traffic.empty:
        return None
    
    try:
        df = df_traffic.copy()
        
        # 渠道列定义 (起始列索引, 渠道名称)
        # 列结构：每8列一个渠道 (日期, Impression, Page views, CTR, CVR, 流量贡献占比, GMV占比, 空列)
        channel_defs = [
            (0, '全店'),
            (8, '短视频'),
            (16, '商品卡'),
            (24, '直播')
        ]
        
        result = {
            'current': {},      # 本周期 = 3月 = 第3行
            'last': {},         # 上周期 = 2月 = 第2行
            'ytd_pv': 0,        # 年度总计 = 1-3月累加
            'mom_change': {}
        }
        
        # 解析各渠道数据
        for start_col, channel_name in channel_defs:
            if start_col >= len(df.columns):
                continue
                
            # 获取指标列索引
            imp_col = df.columns[start_col + 1] if start_col + 1 < len(df.columns) else None
            pv_col = df.columns[start_col + 2] if start_col + 2 < len(df.columns) else None
            ctr_col = df.columns[start_col + 3] if start_col + 3 < len(df.columns) else None
            cvr_col = df.columns[start_col + 4] if start_col + 4 < len(df.columns) else None
            pv_ratio_col = df.columns[start_col + 5] if start_col + 5 < len(df.columns) else None
            gmv_ratio_col = df.columns[start_col + 6] if start_col + 6 < len(df.columns) else None
            
            # 本周期数据 = 第3行（2026年3月）
            if len(df) > 3:
                result['current'][channel_name] = {
                    'Product Impression': int(df.iloc[3].get(imp_col, 0)) if imp_col else 0,
                    'PV': int(df.iloc[3].get(pv_col, 0)) if pv_col else 0,
                    'CTR': float(df.iloc[3].get(ctr_col, 0)) if ctr_col else 0,
                    'CVR': float(df.iloc[3].get(cvr_col, 0)) if cvr_col else 0,
                    '流量贡献占比 (PV)': float(df.iloc[3].get(pv_ratio_col, 0)) if pv_ratio_col else 0,
                    'GMV占比': float(df.iloc[3].get(gmv_ratio_col, 0)) if gmv_ratio_col else 0,
                }
            
            # 上周期数据 = 第2行（2026年2月）按MTD比例换算（14/28）
            if len(df) > 2:
                # 2026年是平年，2月有28天，MTD比例 = 14/28 = 0.5
                mtd_ratio_feb = 14 / 28
                result['last'][channel_name] = {
                    'Product Impression': int(float(df.iloc[2].get(imp_col, 0)) * mtd_ratio_feb) if imp_col else 0,
                    'PV': int(float(df.iloc[2].get(pv_col, 0)) * mtd_ratio_feb) if pv_col else 0,
                    'CTR': float(df.iloc[2].get(ctr_col, 0)) if ctr_col else 0,  # 比率不换算
                    'CVR': float(df.iloc[2].get(cvr_col, 0)) if cvr_col else 0,  # 比率不换算
                    '流量贡献占比 (PV)': float(df.iloc[2].get(pv_ratio_col, 0)) if pv_ratio_col else 0,  # 占比不换算
                    'GMV占比': float(df.iloc[2].get(gmv_ratio_col, 0)) if gmv_ratio_col else 0,  # 占比不换算
                }
        
        # 计算环比变化（3月 vs 2月）
        for channel in ['全店', '短视频', '商品卡', '直播']:
            if channel in result['current'] and channel in result['last']:
                curr = result['current'][channel]
                last = result['last'][channel]
                
                result['mom_change'][channel] = {
                    'PV': (curr['PV'] - last['PV']) / last['PV'] if last['PV'] > 0 else 0,
                    'CTR': curr['CTR'] - last['CTR'],
                    'CVR': curr['CVR'] - last['CVR'],
                    'GMV占比': curr['GMV占比'] - last['GMV占比']
                }
        
        # 年度总计PV = 2026年1-3月累加（第1行 + 第2行 + 第3行）
        pv_col_idx = 2  # Page views列索引（全店）
        if len(df) > 3:
            # 有1月、2月、3月数据（第1行、第2行、第3行）
            jan_pv = int(df.iloc[1, pv_col_idx])  # 第1行 = 1月
            feb_pv = int(df.iloc[2, pv_col_idx])  # 第2行 = 2月
            mar_pv = int(df.iloc[3, pv_col_idx])  # 第3行 = 3月
            result['ytd_pv'] = jan_pv + feb_pv + mar_pv
        elif len(df) > 2:
            # 只有1月、2月数据
            result['ytd_pv'] = int(df.iloc[1:3][df.columns[pv_col_idx]].sum())
        elif len(df) > 1:
            # 只有1月数据
            result['ytd_pv'] = int(df.iloc[1][df.columns[pv_col_idx]])
        
        return result
        
    except Exception as e:
        print(f"加载流量漏斗数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None

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
        '采购成本率': '采购成本', '头程成本率': '头程', '尾程费率': '尾程', '关税成本率': '关税',
        '仓租费率': '仓租', '其他物流费用率': '其他物流成本',
        '品牌费率': '品牌费用', '平台费率': '平台佣金', '售后成本率': '其他和售后',
        '达人佣金费率': '总达人佣金', '样品费率': '总样品费', '广告费率': '总广告投放费',
    }
    for r_col, val_col in ratio_map.items():
        df[r_col] = df[val_col] / rev_safe if val_col in df.columns else 0

    return df


def build_sku_cost_mapping_audit(
    sku_stats: pd.DataFrame,
    sku_to_spu_dict: dict,
    map_p: dict,
    map_h: dict,
    map_t: dict,
    map_d: dict,
    sku_ads_cost: pd.DataFrame | None,
    sku_real_comm: pd.DataFrame | None,
    sku_sample_cost: pd.DataFrame | None,
    has_purchase: bool,
    has_head: bool,
    has_tail: bool,
    has_tariff: bool,
    has_ads: bool,
    has_affiliate: bool,
    has_samples: bool,
):
    """生成 SKU 成本映射的缺失检测表

    目标：校验订单表里每个 SKU 是否都能匹配到对应的成本/映射数据（采购/头程/尾程/关税/广告/佣金/样品）。
    """

    audit = sku_stats[['SKU', 'SPU']].drop_duplicates().copy()
    audit['SPU映射存在'] = audit['SKU'].isin(sku_to_spu_dict)

    # 映射存在性判断（只在对应文件存在时才检测）
    audit['采购成本缺失'] = has_purchase & (~audit['SKU'].isin(map_p.keys()))
    audit['头程成本缺失'] = has_head & (~audit['SKU'].isin(map_h.keys()))
    audit['尾程成本缺失'] = has_tail & (~audit['SKU'].isin(map_t.keys()))
    audit['关税成本缺失'] = has_tariff & (~audit['SKU'].isin(map_d.keys()))

    # 广告 / 佣金 / 样品：使用已计算的聚合表判断是否有对应 SKU 映射
    if has_ads and sku_ads_cost is not None:
        ads_skus = set(sku_ads_cost['SKU'].astype(str).str.strip().str.upper())
        audit['广告成本缺失'] = ~audit['SKU'].astype(str).str.strip().str.upper().isin(ads_skus)
    else:
        audit['广告成本缺失'] = False

    if has_affiliate and sku_real_comm is not None:
        comm_skus = set(sku_real_comm['SKU'].astype(str).str.strip().str.upper())
        audit['达人佣金缺失'] = ~audit['SKU'].astype(str).str.strip().str.upper().isin(comm_skus)
    else:
        audit['达人佣金缺失'] = False

    if has_samples and sku_sample_cost is not None:
        sample_skus = set(sku_sample_cost['SKU'].astype(str).str.strip().str.upper())
        audit['样品费缺失'] = ~audit['SKU'].astype(str).str.strip().str.upper().isin(sample_skus)
    else:
        audit['样品费缺失'] = False

    # 汇总缺失项
    def _merge_notes(row):
        notes = []
        if not row['SPU映射存在']:
            notes.append('SPU未匹配')
        for col, label in [
            ('采购成本缺失', '采购'),
            ('头程成本缺失', '头程'),
            ('尾程成本缺失', '尾程'),
            ('关税成本缺失', '关税'),
            ('广告成本缺失', '广告'),
            ('达人佣金缺失', '佣金'),
            ('样品费缺失', '样品费'),
        ]:
            if row.get(col):
                notes.append(label)
        return '、'.join(notes)

    audit['缺失项'] = audit.apply(_merge_notes, axis=1)

    # 只保留存在缺失的 SKU（方便展示）
    audit_missing = audit[audit['缺失项'] != ''].copy()
    if not audit_missing.empty:
        # 为方便阅读，按缺失项数量排序
        audit_missing['缺失项数量'] = audit_missing['缺失项'].str.count('、') + 1
        audit_missing = audit_missing.sort_values(['缺失项数量', 'SKU'], ascending=[False, True])
        audit_missing = audit_missing.drop(columns=['缺失项数量'])

    return audit_missing


def build_sku_calculation_detail(sku_stats, df_sku_raw):
    """
    生成 SKU 级别的详细计算明细表，用于手工计算对比
    
    输入：
    - sku_stats: 中间计算过程中的 SKU 统计数据（含单件成本、销量等）
    - df_sku_raw: calculate_metrics_final 后的 SKU 结果数据
    
    输出：
    - 包含详细计算过程的 DataFrame，方便导出 Excel 进行手工验证
    """
    if sku_stats is None or sku_stats.empty:
        return pd.DataFrame()
    
    # 从 sku_stats 获取基础数据
    detail = sku_stats.copy()
    
    # 确保关键列存在
    required_cols = ['SKU', 'SPU', '销量', '退款后营收', '单件采购成本', '单件头程', '单件尾程', '单件关税']
    for col in required_cols:
        if col not in detail.columns:
            detail[col] = 0.0
    
    # 计算各项成本明细
    detail['采购成本_计算'] = detail['单件采购成本'] * detail['销量']
    detail['头程_计算'] = detail['单件头程'] * detail['销量']
    detail['尾程_计算'] = detail['单件尾程'] * detail['销量']
    detail['关税_计算'] = detail['单件关税'] * detail['销量']
    
    # 运营成本计算（固定费率）
    detail['仓租_计算'] = detail['退款后营收'] * 0.005
    detail['其他物流成本_计算'] = detail['退款后营收'] * 0.003
    detail['品牌费用_计算'] = detail['退款后营收'] * 0.003
    detail['平台佣金_计算'] = detail['退款后营收'] * 0.06
    detail['其他和售后_计算'] = detail['退款后营收'] * 0.003
    
    # 营销成本
    detail['总广告投放费'] = detail.get('总广告投放费', 0.0)
    detail['总达人佣金'] = detail.get('总达人佣金', 0.0)
    detail['总样品费'] = detail.get('总样品费', 0.0)
    
    # 总成本计算
    detail['营业成本_计算'] = (
        detail['采购成本_计算'] + detail['头程_计算'] + 
        detail['尾程_计算'] + detail['关税_计算']
    )
    detail['运营成本_计算'] = (
        detail['仓租_计算'] + detail['其他物流成本_计算'] + 
        detail['品牌费用_计算'] + detail['平台佣金_计算'] + 
        detail['其他和售后_计算']
    )
    detail['营销成本_计算'] = (
        detail['总广告投放费'] + detail['总达人佣金'] + detail['总样品费']
    )
    detail['总成本_计算'] = (
        detail['营业成本_计算'] + detail['运营成本_计算'] + detail['营销成本_计算']
    )
    
    # 利润计算
    detail['利润额_计算'] = detail['退款后营收'] - detail['总成本_计算']
    detail['利润率_计算'] = detail.apply(
        lambda x: x['利润额_计算'] / x['退款后营收'] if x['退款后营收'] > 0 else 0.0, 
        axis=1
    )
    
    # 选择输出列（按计算逻辑排序）
    output_cols = [
        'SKU', 'SPU', '三级类目', '销量', '退款后营收',
        '单件采购成本', '采购成本_计算',
        '单件头程', '头程_计算',
        '单件尾程', '尾程_计算', '尾程成本备注',
        '单件关税', '关税_计算',
        '营业成本_计算',
        '仓租_计算', '其他物流成本_计算', '品牌费用_计算', 
        '平台佣金_计算', '其他和售后_计算',
        '运营成本_计算',
        '总广告投放费', '总达人佣金', '总样品费',
        '营销成本_计算',
        '总成本_计算',
        '利润额_计算', '利润率_计算'
    ]
    
    # 只保留存在的列
    existing_cols = [c for c in output_cols if c in detail.columns]
    return detail[existing_cols].sort_values(['SPU', 'SKU'])


def build_ads_cost_validation(df_ads, df_mapping, df_orders, sku_ads_cost):
    """
    广告成本分摊校验：对比广告表总花费 vs 程序分摊结果
    
    返回：
    - validation_summary: 汇总校验信息
    - unmapped_pids: 未被映射的 PID 列表
    - pid_detail: 每个 PID 的分摊详情
    """
    result = {
        'validation_summary': {},
        'unmapped_pids': pd.DataFrame(),
        'pid_detail': pd.DataFrame(),
        'errors': []
    }
    
    if df_ads is None:
        result['errors'].append('广告表未上传')
        return result
    
    col_pid = COLUMN_CONFIG['ads']['pid']
    col_cost = COLUMN_CONFIG['ads']['cost']
    
    if col_pid not in df_ads.columns:
        result['errors'].append(f'广告表缺少PID列: {col_pid}')
        return result
    if col_cost not in df_ads.columns:
        result['errors'].append(f'广告表缺少Cost列: {col_cost}')
        return result
    
    # 1. 计算广告表原始总花费
    df_ads_clean = df_ads.copy()
    df_ads_clean['PID_Clean'] = clean_text(df_ads_clean, col_pid)
    df_ads_clean['Cost_Clean'] = df_ads_clean[col_cost].apply(clean_money)
    
    # 按 PID 汇总原始广告花费
    ads_by_pid = df_ads_clean.groupby('PID_Clean').agg({
        'Cost_Clean': 'sum'
    }).reset_index()
    ads_by_pid.columns = ['PID', '原始广告花费']
    total_ads_raw = ads_by_pid['原始广告花费'].sum()
    
    # 2. 检查映射关系
    if df_mapping is None:
        result['errors'].append('PID映射表未上传，无法校验分摊')
        result['validation_summary'] = {
            '广告表总花费': total_ads_raw,
            '映射表状态': '缺失'
        }
        return result
    
    # 获取映射关系
    c_pid_map = find_col_by_keyword_fuzzy(df_mapping, ['product_id'])
    c_sku_map = find_col_by_keyword_fuzzy(df_mapping, ['sku'])
    
    if not c_pid_map or not c_sku_map:
        result['errors'].append('PID映射表缺少必要列')
        result['validation_summary'] = {
            '广告表总花费': total_ads_raw,
            '映射表状态': '列缺失'
        }
        return result
    
    df_map_clean = df_mapping.copy()
    df_map_clean['PID_Clean'] = clean_text(df_map_clean, c_pid_map)
    df_map_clean['SKU_Clean'] = clean_text(df_map_clean, c_sku_map)
    
    # 按 PID 统计映射的 SKU 数量
    pid_sku_count = df_map_clean.groupby('PID_Clean')['SKU_Clean'].nunique().reset_index()
    pid_sku_count.columns = ['PID', '映射SKU数']
    
    # 3. 找出未被映射的 PID
    mapped_pids = set(df_map_clean['PID_Clean'].unique())
    all_pids = set(ads_by_pid['PID'].unique())
    unmapped_pids = all_pids - mapped_pids
    
    unmapped_df = ads_by_pid[ads_by_pid['PID'].isin(unmapped_pids)].copy()
    unmapped_df = unmapped_df.sort_values('原始广告花费', ascending=False)
    unmapped_df['占比'] = unmapped_df['原始广告花费'] / total_ads_raw if total_ads_raw > 0 else 0
    
    # 4. 统计分摊结果
    if sku_ads_cost is not None and not sku_ads_cost.empty:
        total_allocated = sku_ads_cost['总广告投放费'].sum()
    else:
        total_allocated = 0
    
    # 5. 计算已映射 PID 的花费
    mapped_spent = ads_by_pid[ads_by_pid['PID'].isin(mapped_pids)]['原始广告花费'].sum()
    unmapped_spent = unmapped_df['原始广告花费'].sum() if not unmapped_df.empty else 0
    
    # 6. 汇总信息
    result['validation_summary'] = {
        '广告表总花费': total_ads_raw,
        '已映射PID花费': mapped_spent,
        '未映射PID花费': unmapped_spent,
        '未映射PID数量': len(unmapped_pids),
        '程序分摊总花费': total_allocated,
        '分摊差额': total_ads_raw - total_allocated,
        '映射覆盖率': mapped_spent / total_ads_raw if total_ads_raw > 0 else 0
    }
    
    result['unmapped_pids'] = unmapped_df
    
    # 7. 生成 PID 分摊详情
    pid_detail = ads_by_pid.merge(pid_sku_count, on='PID', how='left')
    pid_detail['映射SKU数'] = pid_detail['映射SKU数'].fillna(0).astype(int)
    pid_detail['是否映射'] = pid_detail['映射SKU数'] > 0
    
    # 估算分摊到各 SKU 的金额（简化版，实际分摊考虑了营收权重）
    # 这里主要是为了展示每个 PID 的理论分摊逻辑
    result['pid_detail'] = pid_detail.sort_values('原始广告花费', ascending=False)
    
    return result


# ================= 6. 广告分析（V2 两层诊断） =================
# （此部分保持你上一版逻辑不变，为节省篇幅，直接原样保留）
def process_ads_data_v2(dfs, df_sku_final):
    df_ads = dfs.get('ads')
    df_mapping = dfs.get('mapping')
    df_spu_sku = dfs.get('spu_sku')

    if df_ads is None:
        return None, None, None, {}

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

    c_ctr = COLUMN_CONFIG['ads']['ctr']
    if c_ctr in df_ads.columns:
        df_ads['CTR'] = df_ads[c_ctr].apply(clean_percent)
    else:
        df_ads['CTR'] = df_ads.apply(lambda x: (x['Clk_Val'] / x['Imp_Val']) if x['Imp_Val'] > 0 else 0.0, axis=1)

    c_cvr = COLUMN_CONFIG['ads']['cvr']
    if c_cvr in df_ads.columns:
        df_ads['CVR'] = df_ads[c_cvr].apply(clean_percent)
    else:
        df_ads['CVR'] = df_ads.apply(lambda x: (x['Ord_Val'] / x['Clk_Val']) if x['Clk_Val'] > 0 else 0.0, axis=1)

    c_2s = COLUMN_CONFIG['ads']['rate_2s']
    c_6s = COLUMN_CONFIG['ads']['rate_6s']
    df_ads['RATE_2S'] = df_ads[c_2s].apply(clean_percent) if c_2s in df_ads.columns else 0.0
    df_ads['RATE_6S'] = df_ads[c_6s].apply(clean_percent) if c_6s in df_ads.columns else 0.0

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

    sku_margin_map = {}
    sku_rev_weight_map = {}
    if df_sku_final is not None and not df_sku_final.empty:
        tmp = df_sku_final.copy()
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
        return (aov / DEFAULT_ROAS_FOR_CPA_LINE) if aov > 0 else 0.0, "DEFAULT"

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

    df_prod['ROI'] = df_prod.apply(lambda x: (x['Revenue'] / x['Cost']) if x['Cost'] > 0 else 0.0, axis=1)
    df_prod['CPA'] = df_prod.apply(lambda x: (x['Cost'] / x['Orders']) if x['Orders'] > 0 else 0.0, axis=1)
    df_prod['CPM'] = df_prod.apply(lambda x: (x['Cost'] / x['Imp_Val'] * 1000) if x['Imp_Val'] > 0 else 0.0, axis=1)
    df_prod['CTR'] = df_prod.apply(lambda x: (x['Clk_Val'] / x['Imp_Val']) if x['Imp_Val'] > 0 else 0.0, axis=1)
    df_prod['CVR'] = df_prod.apply(lambda x: (x['Orders'] / x['Clk_Val']) if x['Clk_Val'] > 0 else 0.0, axis=1)
    df_prod['AOV'] = df_prod.apply(lambda x: (x['Revenue'] / x['Orders']) if x['Orders'] > 0 else 0.0, axis=1)

    df_prod['SPU'] = df_prod['Product ID'].apply(pid_to_spu_str)

    cpa_lines = []
    sources = []
    for _, r in df_prod.iterrows():
        line, src = compute_cpa_line(r['Product ID'], r['AOV'])
        cpa_lines.append(line)
        sources.append(src)
    df_prod['CPA_Line'] = cpa_lines
    df_prod['CPA_Line_Source'] = sources

    med_ctr = float(df_prod['CTR'].median()) if not df_prod.empty else CTR_FLOOR
    med_cvr = float(df_prod['CVR'].median()) if not df_prod.empty else CVR_FLOOR
    med_cpm = float(df_prod['CPM'].median()) if not df_prod.empty else CPM_FLOOR_HIGH

    thr_ctr_low = max(med_ctr, CTR_FLOOR)
    thr_cvr_low = max(med_cvr, CVR_FLOOR)
    thr_cpm_high = max(med_cpm, CPM_FLOOR_HIGH)

    def pid_status_and_diag(row):
        if row['Cost'] < COST_OBSERVE:
            return "⚪ 观察期", "花费太少，继续观察", "继续测/先别下结论"
        if (row['ROI'] > ROI_BEST) and (row['CPA_Line'] > 0) and (row['CPA'] < row['CPA_Line']):
            return "🟢 爆款", "盈利且起量", "建议扩量（加预算/复制受众/复制素材）"
        is_loss = (row['ROI'] < ROI_LOSS) or ((row['CPA_Line'] > 0) and (row['CPA'] > row['CPA_Line']))
        if is_loss:
            if row['CVR'] < thr_cvr_low:
                return "🔴 亏损", "CVR 低：流量来了接不住", "检查产品/价格/落地页/货不对板"
            if row['CPM'] > thr_cpm_high:
                return "🔴 亏损", "CPM 高：流量太贵", "调整受众/出新素材/避开高竞争时段"
            if row['CTR'] < thr_ctr_low:
                return "🔴 亏损", "CTR 低：没人点", "优化封面与开头3秒（钩子/对比/证据）"
            return "🔴 亏损", "综合偏低：ROI不达标", "优先改素材与商品页表达"
        return "🟡 可优化", "接近盈亏线", "按 CTR/CVR/CPM 最弱项优化后继续测"

    df_prod[['Status', 'Diagnosis', 'Action']] = df_prod.apply(lambda x: pd.Series(pid_status_and_diag(x)), axis=1)

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

    med_v_ctr = float(df_video['CTR'].median()) if not df_video.empty else CTR_FLOOR
    med_v_cvr = float(df_video['CVR'].median()) if not df_video.empty else CVR_FLOOR
    med_v_2s = float(df_video['RATE_2S'].median()) if not df_video.empty else RATE2S_FLOOR
    med_v_6s = float(df_video['RATE_6S'].median()) if not df_video.empty else RATE6S_FLOOR

    thr_v_ctr_high = max(med_v_ctr, CTR_FLOOR)
    thr_v_cvr_high = max(med_v_cvr, CVR_FLOOR)
    thr_v_2s_high = max(med_v_2s, RATE2S_FLOOR)
    thr_v_6s_low = min(med_v_6s, RATE6S_FLOOR)
    thr_v_cvr_low = max(med_v_cvr, CVR_FLOOR)

    def classify_video(row):
        ctr = row.get('CTR', 0.0)
        cvr = row.get('CVR', 0.0)
        r2 = row.get('RATE_2S', 0.0)
        r6 = row.get('RATE_6S', 0.0)

        if (ctr >= thr_v_ctr_high) and (r2 >= thr_v_2s_high) and (cvr >= thr_v_cvr_high):
            return "🥇 黄金素材", "开头吸睛，内容种草，转化精准", "复制结构/开头套路，扩量投放"
        if (ctr >= thr_v_ctr_high) and (r6 <= thr_v_6s_low):
            return "🎣 标题党", "开头骗点击，但内容崩塌，用户流失", "重剪前6秒承接，卖点前置"
        if ((r2 >= thr_v_2s_high) or (r6 >= max(med_v_6s, RATE6S_FLOOR))) and (cvr < thr_v_cvr_low):
            return "🌿 无效种草", "视频好看但不转化，或货不对板", "强化购买理由/证据镜头/商品页一致性"
        return "🗑️ 其他", "表现平庸或样本不足", "继续测试或归档"

    df_video[['Creative Type', 'AI_Conclusion', 'Next_Action']] = df_video.apply(lambda x: pd.Series(classify_video(x)), axis=1)
    df_video['HookRate_2S'] = df_video['RATE_2S']

    meta = {
        "thr_prod": {"CTR_low": thr_ctr_low, "CVR_low": thr_cvr_low, "CPM_high": thr_cpm_high},
        "thr_video": {"CTR_high": thr_v_ctr_high, "CVR_high": thr_v_cvr_high, "RATE2S_high": thr_v_2s_high, "RATE6S_low": thr_v_6s_low},
        "no_daily": True
    }
    return df_prod, df_video, df_ads, meta

# ================= 7. 达人分析（V2） =================
# （保持不变）
def process_creator_data_v2(dfs, df_shop_raw, df_spu_raw):
    # 这里保持你原逻辑
    df_aff = dfs.get('affiliate')
    df_trans = dfs.get('transaction')
    res = {
        "overall": None,
        "leaderboard": None,
        "content_pie": None,
        "spu_perf": None,
        "commission_source_note": ""
    }

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

            shop_gmv_before = 0.0
            if df_shop_raw is not None and not df_shop_raw.empty and '退款前营收' in df_shop_raw.columns:
                shop_gmv_before = float(df_shop_raw.iloc[0]['退款前营收'] or 0)

            aff_gmv_shop = float(df_trans['Affiliate_GMV'].sum())
            videos_shop = float(df_trans['Videos'].sum())
            res['overall'] = {
                "Affiliate_GMV": aff_gmv_shop,
                "Affiliate_Share": (aff_gmv_shop / shop_gmv_before) if shop_gmv_before > 0 else 0.0,
                "Videos": videos_shop,
                "Efficiency": (aff_gmv_shop / videos_shop) if videos_shop > 0 else 0.0
            }

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

# ================= X. 通用增强函数（周对比 / SPU责任位 / Top10回流 / AI任务） =================

def compute_weekly_compare_orders(df_orders):
    """
    近7天 vs 上7天（稳定口径：订单表可算）
    返回：summary_df（行=指标，列=近7天/上7天/变化），以及note提示
    """
    if df_orders is None or df_orders.empty:
        return None, "未上传订单表或订单表为空"

    df = df_orders.copy()
    if not ensure_date_column(df):
        return None, "订单表缺少时间字段（Created Time），无法计算周对比"

    col_oid = COLUMN_CONFIG['orders']['order_id']
    col_rev = COLUMN_CONFIG['orders']['revenue']
    col_qty = COLUMN_CONFIG['orders']['qty']
    col_status = COLUMN_CONFIG['orders']['status']

    need = [col_oid, col_rev, col_qty]
    miss = [c for c in need if c not in df.columns]
    if miss:
        return None, f"订单表缺少必要列：{miss}"

    df['OID_Clean'] = df[col_oid].apply(convert_scientific_to_str)
    df['Rev_Val'] = pd.to_numeric(df[col_rev], errors='coerce').fillna(0.0)
    df['Qty_Val'] = pd.to_numeric(df[col_qty], errors='coerce').fillna(0.0)

    if col_status in df.columns:
        df['Is_Cancel'] = df[col_status].astype(str).str.strip().isin(['Cancelled', 'Canceled'])
    else:
        df['Is_Cancel'] = False

    # 以订单表内最大日期作为“本周结束日”（更贴近你实际数据更新节奏）
    end_date = pd.to_datetime(df['Date'].max()).normalize()
    curr_start = end_date - pd.Timedelta(days=6)
    prev_start = curr_start - pd.Timedelta(days=7)

    def agg_period(d0, d1):
        sub = df[(df['Date'] >= d0) & (df['Date'] <= d1)].copy()
        if sub.empty:
            return {
                "date_range": f"{d0.date()} ~ {d1.date()}",
                "net_rev": 0.0, "orders": 0, "cancel_orders": 0,
                "units": 0.0, "asp": 0.0, "cancel_rate": 0.0
            }

        total_orders = int(sub['OID_Clean'].nunique())
        cancel_orders = int(sub[sub['Is_Cancel']]['OID_Clean'].nunique())

        sub_ok = sub[~sub['Is_Cancel']].copy()
        net_rev = float(sub_ok['Rev_Val'].sum() or 0.0)
        units = float(sub_ok['Qty_Val'].sum() or 0.0)
        asp = (net_rev / units) if units > 0 else 0.0
        cancel_rate = (cancel_orders / total_orders) if total_orders > 0 else 0.0

        return {
            "date_range": f"{d0.date()} ~ {d1.date()}",
            "net_rev": net_rev,
            "orders": total_orders,
            "cancel_orders": cancel_orders,
            "units": units,
            "asp": asp,
            "cancel_rate": cancel_rate
        }

    curr = agg_period(curr_start, end_date)
    prev = agg_period(prev_start, curr_start - pd.Timedelta(days=1))

    def delta(a, b):
        # a=curr, b=prev
        return (a - b)

    def delta_pct(a, b):
        return (a / b - 1) if b not in [0, 0.0, None, np.nan] else np.nan

    rows = [
        ("日期范围", curr["date_range"], prev["date_range"], "—"),
        ("净营收($)", curr["net_rev"], prev["net_rev"], delta(curr["net_rev"], prev["net_rev"])),
        ("订单数", curr["orders"], prev["orders"], delta(curr["orders"], prev["orders"])),
        ("取消订单数", curr["cancel_orders"], prev["cancel_orders"], delta(curr["cancel_orders"], prev["cancel_orders"])),
        ("取消率", curr["cancel_rate"], prev["cancel_rate"], delta(curr["cancel_rate"], prev["cancel_rate"])),
        ("销量(件)", curr["units"], prev["units"], delta(curr["units"], prev["units"])),
        ("ASP($/件)", curr["asp"], prev["asp"], delta(curr["asp"], prev["asp"])),
        ("净营收变化%", np.nan, np.nan, delta_pct(curr["net_rev"], prev["net_rev"])),
    ]

    out = pd.DataFrame(rows, columns=["指标", "近7天", "上7天", "变化"])
    return out, ""


def infer_spu_owner(row):
    """
    第一责任位（单一归属，规则驱动）
    输出：投放 / 内容 / 达人 / 商品
    """
    cause = str(row.get('主因', '') or '')
    action = str(row.get('建议动作', '') or '')
    bucket = str(row.get('决策篮子', '') or '')

    # 1) 商品优先（退款）
    if "退款" in cause or "退款" in action:
        return "商品"

    # 2) 达人信号（集中度/矩阵/佣金）
    if ("达人" in cause) or ("佣金" in cause) or ("矩阵" in action) or ("达人" in action):
        return "达人"

    # 3) 营销偏高：看主因里最大项
    if "营销偏高" in cause:
        if "广告费" in cause:
            return "投放"
        if "样品费" in cause:
            return "达人"
        if "达人佣金" in cause:
            return "达人"
        return "投放"

    # 4) A 增投候选：默认投放（扩量动作通常由投放牵头）
    if str(bucket).startswith("A"):
        return "投放"

    # 5) 亏损兜底：先投放控损
    if "利润为负" in cause or "止损" in action:
        return "投放"

    # 6) 默认：内容（优化表达/证据镜头）
    return "内容"



def build_ads_spu_drag_table(df_prod_ads, topn=10):
    """
    广告拖累 SPU TopN（MVP：PID->SPU 逗号拆分，成本/营收等额分摊）
    """
    if df_prod_ads is None or df_prod_ads.empty:
        return None, "未检测到广告产品诊断数据"

    need_cols = ['SPU', 'Cost', 'Revenue', 'ROI', 'Status', 'Diagnosis', 'Action']
    miss = [c for c in need_cols if c not in df_prod_ads.columns]
    if miss:
        return None, f"广告产品表缺少字段：{miss}"

    df = df_prod_ads.copy()
    # 拆分 SPU（逗号分隔）
    df['SPU_List'] = df['SPU'].astype(str).apply(lambda x: [s.strip() for s in str(x).split(",") if s.strip() != ""] if str(x).strip() not in ["", "未匹配", "nan"] else ["未匹配"])
    df['SPU_Cnt'] = df['SPU_List'].apply(len).replace(0, 1)

    # 等额分摊（MVP）
    for c in ['Cost', 'Revenue']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        df[f'{c}_Alloc'] = df[c] / df['SPU_Cnt']

    # explode
    df_exp = df.explode('SPU_List').rename(columns={'SPU_List': 'SPU_Alloc'})
    df_exp['Is_Loss'] = df_exp['Status'].astype(str).str.contains("亏损")

    # 聚合
    agg = df_exp.groupby('SPU_Alloc').agg(
        Cost=('Cost_Alloc', 'sum'),
        Revenue=('Revenue_Alloc', 'sum'),
        LossPIDs=('Is_Loss', 'sum'),
        TotalPIDs=('Product ID', 'count') if 'Product ID' in df_exp.columns else ('Is_Loss', 'count')
    ).reset_index()

    agg['ROAS'] = agg.apply(lambda r: (r['Revenue'] / r['Cost']) if r['Cost'] > 0 else 0.0, axis=1)

    # 诊断与动作（取出现频率最高的标签）
    def top_mode(series):
        try:
            s = series.dropna().astype(str)
            s = s[s.str.strip() != ""]
            return s.value_counts().index[0] if not s.empty else "—"
        except:
            return "—"

    diag = df_exp.groupby('SPU_Alloc').agg(
        主诊断=('Diagnosis', top_mode),
        建议动作=('Action', top_mode),
    ).reset_index()

    out = agg.merge(diag, on='SPU_Alloc', how='left').rename(columns={'SPU_Alloc': 'SPU'})

    # 只看“拖累”更直观：优先按 LossPIDs，再按 Cost
    out = out.sort_values(['LossPIDs', 'Cost'], ascending=[False, False]).head(int(topn)).copy()

    return out, "口径说明：若 PID 对应多个 SPU，本榜单采用等额分摊 Cost/Revenue（MVP 口径），用于快速定位拖累面。"

def attach_spu_to_video_ads(df_video_ads, df_pid_spu_map=None):
    """
    给视频层级广告表 df_video_ads 补上 SPU 字段
    - 依赖 df_pid_spu_map: 至少包含 ['Product ID', 'SPU'] 两列
    - 若视频表里本来就有 SPU，则优先用原 SPU
    """
    if df_video_ads is None or df_video_ads.empty:
        return df_video_ads

    df = df_video_ads.copy()

    # 兼容列名
    pid_col_candidates = ['Product ID', 'ProductID', 'Product Id', 'product_id', 'pid']
    pid_col = next((c for c in pid_col_candidates if c in df.columns), None)

    # 如果视频表里已经有 SPU，就不强行覆盖
    if 'SPU' in df.columns:
        return df

    if pid_col is None:
        # 没 PID 也没 SPU，无法拆解
        df['SPU'] = '未匹配'
        return df

    # 处理映射表
    if df_pid_spu_map is None or df_pid_spu_map.empty:
        df['SPU'] = '未匹配'
        return df

    m = df_pid_spu_map.copy()
    if 'Product ID' not in m.columns or 'SPU' not in m.columns:
        df['SPU'] = '未匹配'
        return df

    m = m[['Product ID', 'SPU']].dropna()
    m['Product ID'] = m['Product ID'].astype(str).str.strip()
    m['SPU'] = m['SPU'].astype(str).str.strip()

    pid2spu = dict(zip(m['Product ID'], m['SPU']))

    df[pid_col] = df[pid_col].astype(str).str.strip()
    df['SPU'] = df[pid_col].map(pid2spu).fillna('未匹配')

    return df

def normalize_ads_for_spu_video(df: pd.DataFrame) -> pd.DataFrame:
    """
    把各种 TikTok 广告导出列名，统一成 build_spu_video_table 能识别的标准列名：
    Product ID / Video title / Cost / Revenue / Orders / Impressions / Clicks / CTR / CVR / 2s_rate / 6s_rate
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    # 1) 先按你系统 COLUMN_CONFIG['ads'] 做“强映射”
    cfg = COLUMN_CONFIG.get("ads", {})
    rename_map = {}

    def _rename_if_exists(src, dst):
        if src in df.columns and dst not in df.columns:
            rename_map[src] = dst

    _rename_if_exists(cfg.get("pid", ""), "Product ID")
    _rename_if_exists(cfg.get("video_title", ""), "Video title")
    _rename_if_exists(cfg.get("cost", ""), "Cost")
    _rename_if_exists(cfg.get("revenue", ""), "Revenue")
    _rename_if_exists(cfg.get("orders", ""), "Orders")
    _rename_if_exists(cfg.get("impressions", ""), "Impressions")
    _rename_if_exists(cfg.get("clicks", ""), "Clicks")
    _rename_if_exists(cfg.get("ctr", ""), "CTR")
    _rename_if_exists(cfg.get("cvr", ""), "CVR")
    _rename_if_exists(cfg.get("rate_2s", ""), "2s_rate")
    _rename_if_exists(cfg.get("rate_6s", ""), "6s_rate")

    if rename_map:
        df = df.rename(columns=rename_map)

    # 2) 再做一层“弱归一”（兼容大小写/其它导出叫法）
    df = _normalize_ads_detail_cols(df)

    return df


def build_pid_to_spu_map(df_mapping: pd.DataFrame, df_spu_sku: pd.DataFrame) -> dict:
    """
    用 mapping(PID->SKU) + spu_sku(SKU->SPU) 自动构建 pid_to_spu 字典
    返回：{ "12345": ["SPU_A", "SPU_B"], ... }
    """
    pid2spu = {}
    if df_mapping is None or df_spu_sku is None or df_mapping.empty or df_spu_sku.empty:
        return pid2spu

    m_pid = find_col_by_keyword_fuzzy(df_mapping, ['product_id', 'product id', 'pid'])
    m_sku = find_col_by_keyword_fuzzy(df_mapping, ['sku'])
    if not m_pid or not m_sku:
        return pid2spu

    mp = df_mapping.copy()
    mp['PID_Clean'] = clean_text(mp, m_pid)
    mp['SKU_Clean'] = clean_text(mp, m_sku)

    pid_skus = mp.groupby('PID_Clean')['SKU_Clean'].apply(
        lambda x: list(sorted(set([str(s).strip().upper() for s in x if str(s).strip() != ""])))
    ).to_dict()

    sku2spu = build_sku_to_spu_dict(df_spu_sku)

    for pid, skus in pid_skus.items():
        spus = sorted(set([sku2spu.get(s, s) for s in skus if str(s).strip() != ""]))
        if spus:
            pid2spu[str(pid).strip()] = spus

    return pid2spu



def build_spu_video_action_view_v3(df_spu_video: pd.DataFrame, observe_cost: float = 50.0) -> pd.DataFrame:
    if df_spu_video is None or df_spu_video.empty:
        return pd.DataFrame()

    df = df_spu_video.copy()

    # 必备列兜底
    for c in ["Cost", "Revenue", "Orders", "Impressions", "Clicks", "ROI", "CTR", "CVR", "Hook_2s", "Hold_6s"]:
        if c not in df.columns:
            df[c] = np.nan

    # 先标 Observe
    df["Video_Status"] = np.where(df["Cost"].fillna(0) < observe_cost, "Observe", "TBD")

    # 每个 SPU 内做分位数
    out = []
    for spu, g in df.groupby("SPU", dropna=False):
        gg = g.copy()
        # ROI 分位
        roi_series = gg.loc[gg["Video_Status"] != "Observe", "ROI"].dropna()
        if len(roi_series) >= 4:
            q25 = roi_series.quantile(0.25)
            q75 = roi_series.quantile(0.75)
        else:
            q25, q75 = 0.9, 1.2  # 样本不足时用保守默认

        # 中位数做短板判断
        med_hook = gg["Hook_2s"].median(skipna=True)
        med_ctr  = gg["CTR"].median(skipna=True)
        med_cvr  = gg["CVR"].median(skipna=True)

        def classify_row(r):
            if r["Video_Status"] == "Observe":
                return "Observe", "观察期", "先积累样本：让 Cost ≥ %.0f 再判定" % observe_cost

            roi = r["ROI"]
            if pd.isna(roi):
                return "Observe", "样本不足", "补齐数据后再判定"

            if roi >= q75 and roi >= 1.1:
                # Scale：同时给“复用建议”
                return "Scale", "ROI强", "加预算放量；复制同结构到同SPU/相似SPU"
            if roi <= q25 and roi < 0.9:
                return "Pause", "ROI弱", "暂停或大幅降预算；若要复活，先改 Hook/承接 再小额重测"

            # Optimize：找短板优先级
            reason = []
            if pd.notna(r["Hook_2s"]) and pd.notna(med_hook) and r["Hook_2s"] < med_hook:
                reason.append("Hook弱")
            if pd.notna(r["CTR"]) and pd.notna(med_ctr) and r["CTR"] < med_ctr:
                reason.append("CTR弱")
            if pd.notna(r["CVR"]) and pd.notna(med_cvr) and r["CVR"] < med_cvr:
                reason.append("CVR弱")

            if not reason:
                reason_txt = "可优化"
                action_txt = "小步优化：强化证据镜头/卖点节奏/CTA，继续观察 ROI 变化"
            else:
                reason_txt = "+".join(reason)
                if "Hook弱" in reason_txt:
                    action_txt = "优先改前3秒：痛点更直接+更强证据镜头；保持同卖点不变做AB"
                elif "CTR弱" in reason_txt:
                    action_txt = "改封面/首帧/主视觉：产品占比更大、利益点前置；同脚本重剪AB"
                else:
                    action_txt = "修承接：检查标题党/货不对板/落地页；增加“证明镜头”与清晰CTA"

            return "Optimize", reason_txt, action_txt

        status_reason_action = gg.apply(lambda r: classify_row(r), axis=1, result_type="expand")
        gg["Video_Status"] = status_reason_action[0]
        gg["Reason"] = status_reason_action[1]
        gg["Next_Action"] = status_reason_action[2]
        out.append(gg)

    res = pd.concat(out, ignore_index=True)
    # 更利于运营复制粘贴
    res["Evidence"] = (
        "Cost=" + res["Cost"].fillna(0).round(2).astype(str)
        + " ROI=" + res["ROI"].fillna(0).round(2).astype(str)
        + " CTR=" + (res["CTR"].fillna(0)*100).round(2).astype(str) + "%"
        + " CVR=" + (res["CVR"].fillna(0)*100).round(2).astype(str) + "%"
        + " Hook=" + (res["Hook_2s"].fillna(0)*100).round(2).astype(str) + "%"
    )
    return res




def build_ads_loss_pid_table(df_prod_ads, min_cost=0.0):
    """
    亏损 PID 全量清单（PID维度）
    - 亏损定义：Status 包含“亏损”
    - min_cost：过滤观察期噪音（默认建议用 COST_OBSERVE）
    """
    if df_prod_ads is None or df_prod_ads.empty:
        return None, "未检测到广告产品诊断数据"

    need_cols = ['Status', 'Cost', 'Revenue']
    miss = [c for c in need_cols if c not in df_prod_ads.columns]
    if miss:
        return None, f"广告产品表缺少字段：{miss}"

    df = df_prod_ads.copy()
    df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce').fillna(0.0)
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce').fillna(0.0)

    # 亏损判定
    is_loss = df['Status'].astype(str).str.contains("亏损", na=False)
    out = df[is_loss].copy()

    # 过滤观察期
    if float(min_cost or 0) > 0:
        out = out[out['Cost'] >= float(min_cost)].copy()

    # ROI/ROAS列兼容
    if 'ROI' not in out.columns:
        out['ROI'] = out.apply(lambda r: (r['Revenue'] / r['Cost']) if r['Cost'] > 0 else 0.0, axis=1)

    # 优先展示列（兼容你现有字段）
    prefer_cols = [
        'Product ID', 'SPU', 'Cost', 'Revenue', 'ROI',
        'Status', 'Diagnosis', 'Action',
        'CPA', 'CVR', 'CTR', 'CPM'
    ]
    show_cols = [c for c in prefer_cols if c in out.columns]
    out = out[show_cols].sort_values('Cost', ascending=False)

    note = "口径：亏损PID=Status包含“亏损”。建议优先处理 Cost 更高的亏损 PID。"
    return out.reset_index(drop=True), note


def build_creator_risk_table(spu_aff_perf, creator_stats, topn=10):
    """
    达人结构风险 SPU TopN（集中度/矩阵薄/效率低/达人渗透异常）
    """
    if (spu_aff_perf is None or spu_aff_perf.empty) and (creator_stats is None or creator_stats.empty):
        return None, "未检测到达人数据（缺 Transaction / affiliate 或字段不足）"

    base = None
    if spu_aff_perf is not None and not spu_aff_perf.empty:
        base = spu_aff_perf.copy()
    else:
        base = pd.DataFrame({'SPU': creator_stats['SPU'].astype(str).tolist()})

    if creator_stats is not None and not creator_stats.empty:
        base = base.merge(creator_stats, on='SPU', how='left')

    # 数值化兜底
    for c in ['Affiliate_Rate', 'Videos', 'OutputPerVideo', 'Affiliate_GMV', 'Creators', 'Top1Share', 'Top3Share']:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors='coerce').fillna(0.0)
        else:
            base[c] = 0.0

    # 风险评分
    med_eff = float(base['OutputPerVideo'].median() or 0.0)
    med_eff = med_eff if med_eff > 0 else 1.0

    def reason_and_score(r):
        score = 0
        reasons = []
        if r['Top1Share'] >= 0.70:
            score += 3; reasons.append("Top1≥70%（高度依赖）")
        elif r['Top1Share'] >= 0.45:
            score += 2; reasons.append("Top1≥45%（中度集中）")

        if 0 < r['Creators'] < 5:
            score += 2; reasons.append("出单达人数<5（矩阵薄）")

        if r['Videos'] >= 5 and r['OutputPerVideo'] < med_eff * 0.5:
            score += 1; reasons.append("单视频产出偏低")

        if r['Affiliate_Rate'] >= 0.50:
            score += 1; reasons.append("达人GMV占比偏高")

        if not reasons:
            reasons = ["结构正常/样本不足"]
        return score, "；".join(reasons)

    tmp = []
    for _, rr in base.iterrows():
        s, reason = reason_and_score(rr)
        tmp.append((s, reason))
    base['RiskScore'] = [x[0] for x in tmp]
    base['RiskReason'] = [x[1] for x in tmp]

    base = base.sort_values(['RiskScore', 'Affiliate_GMV'], ascending=[False, False]).head(int(topn)).copy()

    # 一句话建议
    def suggest(r):
        if "高度依赖" in r['RiskReason']:
            return "先做去依赖：扩达人池/多达人同结构测试，再扩量"
        if "矩阵薄" in r['RiskReason']:
            return "先补达人池：分层寄样+佣金策略，再谈放量"
        if "单视频产出偏低" in r['RiskReason']:
            return "优化选人&脚本：提升单视频产出后再加样品/加预算"
        if "达人GMV占比偏高" in r['RiskReason']:
            return "评估风控：降低单点依赖，补自然/广告承接"
        return "继续观察或补充数据"

    base['建议动作'] = base.apply(suggest, axis=1)
    return base, ""


def build_ai_tasks_from_spu(df_pl, target_profit_rate, topn=200):
    """
    从 SPU 决策表 df_pl 生成任务清单（待办+优先级+影响粗估）
    """
    if df_pl is None or df_pl.empty:
        return None

    df = df_pl.copy()

    # 数值化
    for c in ['退款后营收', '利润额', '利润率', '退款率', '总营销费比']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        else:
            df[c] = 0.0

    # 影响粗估（美元）
    def impact_usd(r):
        rev = float(r['退款后营收'] or 0)
        profit = float(r['利润额'] or 0)
        pr = float(r['利润率'] or 0)
        bucket = str(r.get('决策篮子', '') or '')

        if bucket.startswith("C"):
            return abs(profit) if profit < 0 else max(rev * 0.05, 0)  # 兜底
        if bucket.startswith("B"):
            # 以“拉到目标利润率”的潜在增益粗估
            gap = max(float(target_profit_rate) - pr, 0.0)
            return rev * gap
        if bucket.startswith("A"):
            # 增长任务：用当前利润额做“可扩张基准”
            return max(profit, 0.0)
        return 0.0

    df['Impact_USD'] = df.apply(impact_usd, axis=1)

    def impact_level(x):
        x = float(x or 0)
        if x >= 1000:
            return "高"
        if x >= 300:
            return "中"
        return "低"

    df['预估影响等级'] = df['Impact_USD'].apply(impact_level)
    df['预估影响($粗估)'] = df['Impact_USD'].apply(lambda x: f"${float(x or 0):,.0f}")

    # 优先级
    def priority(r):
        bucket = str(r.get('决策篮子', '') or '')
        if bucket.startswith("C"):
            return "P0"
        if bucket.startswith("B"):
            return "P1"
        if bucket.startswith("A"):
            return "P2"
        return "P2"

    df['优先级'] = df.apply(priority, axis=1)

    # 任务名（更像工单）
    def task_name(r):
        spu = str(r.get('SPU', '') or '')
        bucket = str(r.get('决策篮子', '') or '')
        cause = str(r.get('主因', '') or '')
        if bucket.startswith("C"):
            return f"【C止损】{spu} | {cause}"
        if bucket.startswith("B"):
            return f"【B修复】{spu} | {cause}"
        if bucket.startswith("A"):
            return f"【A增投】{spu} | 复制结构扩量"
        return f"【观察】{spu}"

    df['Task'] = df.apply(task_name, axis=1)

    # 证据字段
    def evidence(r):
        return f"营收${float(r['退款后营收'] or 0):,.0f}｜利润${float(r['利润额'] or 0):,.0f}｜利润率{float(r['利润率'] or 0):.2%}｜退款率{float(r['退款率'] or 0):.2%}｜营销费比{float(r['总营销费比'] or 0):.2%}"

    df['证据'] = df.apply(evidence, axis=1)

    keep = ['Task', '对象类型', '对象ID', '优先级', '第一责任位', '触发原因', '建议动作', '预估影响等级', '预估影响($粗估)', '证据']
    out = pd.DataFrame()
    out['Task'] = df['Task']
    out['对象类型'] = "SPU"
    out['对象ID'] = df['SPU'].astype(str)
    out['优先级'] = df['优先级']
    out['第一责任位'] = df.get('第一责任位', '—')
    out['触发原因'] = df.get('主因', '—')
    out['建议动作'] = df.get('建议动作', '—')
    out['预估影响等级'] = df['预估影响等级']
    out['预估影响($粗估)'] = df['预估影响($粗估)']
    out['证据'] = df['证据']

    # 只输出 A/B/C（减少噪音）
    out = out[out['Task'].str.contains("【A增投】|【B修复】|【C止损】", regex=True)].copy()
    # 按优先级 + 影响排序
    pr_order = {'P0': 0, 'P1': 1, 'P2': 2}
    out['__p'] = out['优先级'].map(pr_order).fillna(9)
    out['__impact'] = df.loc[out.index, 'Impact_USD'].values if len(out.index) <= len(df) else 0
    out = out.sort_values(['__p', '__impact'], ascending=[True, False]).drop(columns=['__p', '__impact']).head(int(topn))

    return out


# ================= 8. 主计算流程整合 =================
def run_calculation_logic_v2(dfs):
    for k, df in dfs.items():
        if df is not None:
            dfs[k] = normalize_headers(df)

    yoy_data = load_2025_financial_report()

    df_orders = dfs.get('orders')
    if df_orders is None:
        return None, {}

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
            # 添加月份列用于按月聚合
            df_orders['Month'] = df_orders['Date'].dt.strftime('%Y-%m')

    col_status = COLUMN_CONFIG['orders']['status']
    is_cancelled = df_orders[col_status].astype(str).str.strip().isin(['Cancelled', 'Canceled']) if col_status in df_orders.columns else False
    is_sample = (~is_cancelled) & (df_orders['Rev_Val'] == 0)

    # ✅ 更新：采购成本 - 筛选国家=美国，同时提取三级类目映射
    map_p, _, sku_category_map = get_cost_map_v2(
        dfs.get('purchase'), 
        ['采购', 'CNY'], 
        country_filter='美国'
    )
    
    # ✅ 更新：头程成本 - 筛选国家=美国
    map_h, _, _ = get_cost_map_v2(
        dfs.get('head'), 
        ['头程', 'CNY'], 
        country_filter='美国'
    )
    
    # ✅ 更新：尾程成本 - 先筛选国家=美国+平台=TikTok，如无则使用Shopify
    map_t, tail_source_note, _ = get_cost_map_v2(
        dfs.get('tail'), 
        ['尾程', 'CNY'], 
        country_filter='美国',
        platform_filter='TikTok',
        platform_fallback='Shopify'
    )

    # ✅ 新增：关税成本映射（可选）
    # 你的关税成本表：SKU、关税金额（列名包含"关税"即可）
    map_d = get_cost_map(dfs.get('tariff'), ['关税'])

    df_sample = df_orders[is_sample].copy()
    sku_sample_cost = None
    if not df_sample.empty:
        df_sample['Unit_Cost'] = (
            df_sample['SKU_Clean'].map(map_p).fillna(0) +
            df_sample['SKU_Clean'].map(map_h).fillna(0) +
            df_sample['SKU_Clean'].map(map_t).fillna(0) +
            df_sample['SKU_Clean'].map(map_d).fillna(0)
        )
        df_sample['Total_S'] = df_sample['Qty_Val'] * df_sample['Unit_Cost']
        sku_sample_cost = df_sample.groupby('SKU_Clean')['Total_S'].sum().reset_index().rename(columns={'SKU_Clean': 'SKU', 'Total_S': '总样品费'})

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

    df_normal = df_orders[~is_cancelled].copy()
    df_refund = df_orders[is_cancelled].copy()

    sku_stats = df_normal.groupby(['SKU_Clean', 'SPU', 'Month']).agg({'Rev_Val': 'sum', 'Qty_Val': 'sum'}).reset_index()
    sku_stats.rename(columns={'SKU_Clean': 'SKU', 'Rev_Val': '退款后营收', 'Qty_Val': '销量'}, inplace=True)

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

    if not df_refund.empty:
        ref_agg = df_refund.groupby(['SKU_Clean', 'Month'])['Qty_Val'].sum().reset_index().rename(columns={'SKU_Clean': 'SKU', 'Qty_Val': 'Refund_Orders'})
        sku_stats = pd.merge(sku_stats, ref_agg, on=['SKU', 'Month'], how='left').fillna({'Refund_Orders': 0.0})
    else:
        sku_stats['Refund_Orders'] = 0.0

    sku_stats['单件采购成本'] = sku_stats['SKU'].map(map_p).fillna(0.0)
    sku_stats['单件头程'] = sku_stats['SKU'].map(map_h).fillna(0.0)
    sku_stats['单件尾程'] = sku_stats['SKU'].map(map_t).fillna(0.0)
    
    # ✅ 新增：尾程成本来源备注
    if tail_source_note and '⚠️' in tail_source_note:
        sku_stats['尾程成本备注'] = tail_source_note
    else:
        sku_stats['尾程成本备注'] = ''
    
    # ✅ 新增：三级类目（从采购成本表映射）
    sku_stats['三级类目'] = sku_stats['SKU'].map(sku_category_map).fillna('未分类')

    # ✅ 新增：单件关税（从关税成本表映射）
    sku_stats['单件关税'] = sku_stats['SKU'].map(map_d).fillna(0.0)

    sku_stats['采购成本'] = sku_stats['单件采购成本'] * sku_stats['销量']
    sku_stats['头程'] = sku_stats['单件头程'] * sku_stats['销量']
    sku_stats['尾程'] = sku_stats['单件尾程'] * sku_stats['销量']
    sku_stats['关税'] = sku_stats['单件关税'] * sku_stats['销量']

    # ============== 新增：成本映射自检 ==============
    sku_audit_missing = build_sku_cost_mapping_audit(
        sku_stats=sku_stats,
        sku_to_spu_dict=sku_to_spu_dict,
        map_p=map_p,
        map_h=map_h,
        map_t=map_t,
        map_d=map_d,
        sku_ads_cost=sku_ads_cost,
        sku_real_comm=sku_real_comm,
        sku_sample_cost=sku_sample_cost,
        has_purchase=(dfs.get('purchase') is not None),
        has_head=(dfs.get('head') is not None),
        has_tail=(dfs.get('tail') is not None),
        has_tariff=(dfs.get('tariff') is not None),
        has_ads=(df_ads is not None and df_map is not None),
        has_affiliate=(df_aff is not None),
        has_samples=(not df_sample.empty)
    )

    df_sku_raw = calculate_metrics_final(sku_stats)
    df_sku_out = format_dataframe(df_sku_raw, TARGET_COLUMNS_SKU)

    cols_to_sum = [
        '销量', '退款后营收', '退款前营收', 'Refund_Orders', '退款营收',
        '采购成本', '头程', '尾程', '关税',
        '仓租', '其他物流成本', '品牌费用', '平台佣金', '其他和售后',
        '总达人佣金', '总样品费', '总广告投放费'
    ]
    valid_cols = [c for c in cols_to_sum if c in df_sku_raw.columns]
    
    # ============== SPU 维度汇总 ==============
    spu_agg = df_sku_raw.groupby(['SPU', 'Month'])[valid_cols].sum().reset_index()
    df_spu_raw = calculate_metrics_final(spu_agg).sort_values(by=['Month', '退款后营收'], ascending=[True, False])
    df_spu_out = format_dataframe(df_spu_raw, TARGET_COLUMNS_SPU)
    
    # ============== 三级类目维度汇总 ==============
    category_agg = df_sku_raw.groupby(['三级类目', 'Month'])[valid_cols].sum().reset_index()
    df_category_raw = calculate_metrics_final(category_agg).sort_values(by=['Month', '退款后营收'], ascending=[True, False])
    # 类目输出列（类似SPU，但用三级类目代替SPU）
    TARGET_COLUMNS_CATEGORY = [col for col in TARGET_COLUMNS_SPU if col != 'SPU'] + ['三级类目', '2025年同期退款后营收', '退款后营收_YOY%']
    df_category_out = format_dataframe(df_category_raw, TARGET_COLUMNS_CATEGORY)

    shop_agg = df_sku_raw.groupby('Month')[valid_cols].sum().reset_index()
    df_shop_raw = calculate_metrics_final(shop_agg).sort_values(by='Month')
    
    # 添加同比计算
    yoy_data = load_2025_financial_report()
    if yoy_data:
        # 2025年指标映射到2026年列名
        yoy_mapping = {
            '退款率': '退款率',
            '采购成本率': '营业成本率',  # 采购成本率对应营业成本率的一部分，但近似
            '头程成本率': '营业成本率',
            '关税成本率': '营业成本率',
            '尾程费率': '营业成本率',
            '广告费率': '总营销费比',
            '达人佣金费率': '总营销费比',
            '样品费率': '总营销费比',
            '品牌费率': '运营成本率'
        }
        for yoy_key, current_key in yoy_mapping.items():
            if yoy_key in yoy_data and current_key in df_shop_raw.columns:
                df_shop_raw[f'{current_key}_同比%'] = df_shop_raw.apply(
                    lambda row: (row[current_key] - yoy_data.get(row['Month'].replace('2026-', '2025-'), {}).get(yoy_key, 0)) / max(abs(yoy_data.get(row['Month'].replace('2026-', '2025-'), {}).get(yoy_key, 0)), 0.001) * 100 
                    if row['Month'].replace('2026-', '2025-') in yoy_data and yoy_key in yoy_data[row['Month'].replace('2026-', '2025-')] else None, axis=1
                )
    
    # ============== 新增：加载2025年订单数据，计算YOY营收对比 ==============
    shop_2025, spu_2025, category_2025 = load_2025_orders_data(dfs.get('spu_sku'), sku_to_spu_dict)
    
    # 店铺维度YOY营收对比
    if shop_2025:
        df_shop_raw['2025年同期退款后营收'] = df_shop_raw['Month'].apply(
            lambda m: shop_2025.get(m.replace('2026-', '2025-'), 0)
        )
        df_shop_raw['退款后营收_YOY%'] = df_shop_raw.apply(
            lambda row: (row['退款后营收'] - row['2025年同期退款后营收']) / max(abs(row['2025年同期退款后营收']), 0.001) * 100 
            if row['2025年同期退款后营收'] > 0 else None, axis=1
        )
    
    # SPU维度YOY营收对比
    if spu_2025:
        df_spu_raw['2025年同期退款后营收'] = df_spu_raw.apply(
            lambda row: spu_2025.get((row['SPU'], row['Month'].replace('2026-', '2025-')), 0), axis=1
        )
        df_spu_raw['退款后营收_YOY%'] = df_spu_raw.apply(
            lambda row: (row['退款后营收'] - row['2025年同期退款后营收']) / max(abs(row['2025年同期退款后营收']), 0.001) * 100 
            if row['2025年同期退款后营收'] > 0 else None, axis=1
        )
    
    # 类目维度YOY营收对比（简化：从采购成本表获取2025年类目营收映射）
    if sku_category_map and category_2025:
        # 如果已经有类目2025年数据，直接使用
        df_category_raw['2025年同期退款后营收'] = df_category_raw.apply(
            lambda row: category_2025.get((row['三级类目'], row['Month'].replace('2026-', '2025-')), 0), axis=1
        )
    elif sku_category_map:
        # 否则用2025年采购成本表中同一类目的SKU对应的营收估算
        # 这需要更复杂的处理，暂时先计算类目的2025年营收为0
        df_category_raw['2025年同期退款后营收'] = 0
    
    if '2025年同期退款后营收' in df_category_raw.columns:
        df_category_raw['退款后营收_YOY%'] = df_category_raw.apply(
            lambda row: (row['退款后营收'] - row['2025年同期退款后营收']) / max(abs(row['2025年同期退款后营收']), 0.001) * 100 
            if row['2025年同期退款后营收'] > 0 else None, axis=1
        )
    
    df_shop_raw['数据周期'] = df_shop_raw['Month']
    df_shop_out = format_dataframe(df_shop_raw, TARGET_COLUMNS_SHOP_FINAL)

    df_prod_ads, df_video_ads, df_ads_detail, ads_meta = process_ads_data_v2(dfs, df_sku_raw)
    creator_data = process_creator_data_v2(dfs, df_shop_raw, df_spu_raw)
    if aff_note:
        creator_data['commission_source_note'] = (creator_data.get('commission_source_note', '') + " | " + aff_note).strip(" |")

    # ============== 新增：生成 SKU 计算明细 ==============
    sku_calculation_detail = build_sku_calculation_detail(sku_stats, df_sku_raw)

    # ============== 新增：广告成本分摊校验 ==============
    ads_validation = build_ads_cost_validation(
        df_ads=dfs.get('ads'),
        df_mapping=dfs.get('mapping'),
        df_orders=dfs.get('orders'),
        sku_ads_cost=sku_ads_cost
    )

    meta = {
        "time_str": time_str,
        "max_date": max_date,
        "min_date": min_date,
        "ads_meta": ads_meta
    }

    out = {
        "df_shop_out": df_shop_out, "df_spu_out": df_spu_out, "df_sku_out": df_sku_out,
        "df_category_out": df_category_out, "df_category_raw": df_category_raw,  # 新增：三级类目维度
        "df_shop_raw": df_shop_raw, "df_spu_raw": df_spu_raw, "df_sku_raw": df_sku_raw,
        "df_prod_ads": df_prod_ads, "df_video_ads": df_video_ads,
        "creator_data": creator_data,
        "sku_audit_missing": sku_audit_missing,
        "dfs": dfs,
        # 新增：计算明细和广告校验
        "sku_calculation_detail": sku_calculation_detail,
        "ads_validation": ads_validation,
        # 新增：2025年财务数据用于YOY对比
        "yoy_data": yoy_data
    }
    
    # ============== 新增：加载流量漏斗数据 ==============
    traffic_data = None
    if dfs.get('traffic_funnel') is not None:
        # 获取当前月份（从订单数据推断）
        current_month = None
        if df_orders is not None and not df_orders.empty and 'Month' in df_orders.columns:
            # 确保 Month 列是字符串类型
            month_values = df_orders['Month'].dropna().astype(str)
            current_month = month_values.max() if len(month_values) > 0 else None
        traffic_data = load_traffic_funnel_data(dfs['traffic_funnel'], current_month)
    
    out['traffic_data'] = traffic_data
    
    return out, meta

# ================= 9. 智能文件识别读取器 =================

def _is_ads_detail_df(df: pd.DataFrame, filename: str = "") -> bool:
    cols = set([str(c).strip().lower() for c in df.columns])
    fn = (filename or "").lower()

    # 文件名强命中（你这个就叫“广告订单”）
    if "广告订单" in filename:
        return True

    # 结构命中：有 Video + Product + 花费（任意命中即可）
    has_video = ("video title" in cols) or ("video" in cols)
    has_pid   = ("product id" in cols) or ("pid" in cols)
    has_cost  = ("cost" in cols) or ("spend" in cols) or ("amount spent" in cols)

    # “广告明细”通常还会带 campaign / creative 这些字段，你这张图里就有
    looks_like_detail = has_video and has_pid and (has_cost or ("creative type" in cols) or ("campaign name" in cols))
    return looks_like_detail


def _normalize_ads_detail_cols(df: pd.DataFrame) -> pd.DataFrame:
    # 轻度列名归一，避免 TikTok 导出列名不一致
    rename = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ["amount spent", "spend", "cost($)", "cost (usd)"]:
            rename[c] = "Cost"
        elif cl in ["gross revenue", "total revenue", "gmv"]:
            rename[c] = "Revenue"
        elif cl in ["total orders", "order", "orders"]:
            rename[c] = "Orders"
        elif cl in ["video", "video name", "creative name"]:
            rename[c] = "Video title"
        elif cl in ["pid", "productid", "product id"]:
            rename[c] = "Product ID"
    return df.rename(columns=rename)



def load_uploaded_files(uploaded_files):
    dfs = {
        'orders': None, 'orders_last_year': None, 'ads': None, 'affiliate': None,
        'spu_sku': None, 'mapping': None, 'purchase': None, 'head': None, 'tail': None,
        'tariff': None,  # ✅ 新增：关税成本表
        'transaction': None
    }
    status_flags = {k: False for k in dfs.keys()}
    file_info = {k: {'name': '', 'rows': 0} for k in dfs.keys()}  # 记录文件名和行数
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

            # 规范列名：去除空白、零宽字符、换行，避免列名匹配失败
            df.columns = df.columns.astype(str).str.replace('\n', ' ').str.strip()
            cols = df.columns.tolist()
            log_info = f"📄 **{fname}**\n- 列前5: {cols[:5]}\n"
            match_type = "未匹配"

            if COLUMN_CONFIG['affiliate']['creator'] in cols:
                dfs['affiliate'] = df; status_flags['affiliate'] = True
                file_info['affiliate'] = {'name': fname, 'rows': len(df)}
                match_type = "✅ 联盟订单表"

            elif (COLUMN_CONFIG['transaction']['aff_gmv'] in cols and 
                  COLUMN_CONFIG['transaction']['pid'] in cols):
                dfs['transaction'] = df; status_flags['transaction'] = True
                file_info['transaction'] = {'name': fname, 'rows': len(df)}
                match_type = "✅ Transaction表"

            elif 'Campaign name' in cols or 'ad group name' in cols:
                dfs['ads'] = df; status_flags['ads'] = True
                file_info['ads'] = {'name': fname, 'rows': len(df)}
                match_type = "✅ 广告表"

            elif '2025' in fname_lower:
                dfs['orders_last_year'] = df; status_flags['orders_last_year'] = True
                file_info['orders_last_year'] = {'name': fname, 'rows': len(df)}
                match_type = "✅ 去年订单表"

            elif COLUMN_CONFIG['orders']['order_id'] in cols and COLUMN_CONFIG['orders']['sku'] in cols:
                dfs['orders'] = df; status_flags['orders'] = True
                file_info['orders'] = {'name': fname, 'rows': len(df)}
                match_type = "✅ 主订单表"

            elif 'spu' in fname_lower:
                dfs['spu_sku'] = df; status_flags['spu_sku'] = True
                file_info['spu_sku'] = {'name': fname, 'rows': len(df)}
                match_type = "SPU映射"

            elif 'pid' in fname_lower or 'mapping' in fname_lower:
                dfs['mapping'] = df; status_flags['mapping'] = True; match_type = "PID映射"
                file_info['mapping'] = {'name': fname, 'rows': len(df)}
                # 记录详细信息用于调试
                log_info += f"\n- 行数: {len(df)}"
                log_info += f"\n- 列名: {cols}"
                log_info += f"\n- 前3行预览:\n{df.head(3).to_string()}"

            elif '采购' in fname:
                dfs['purchase'] = df; status_flags['purchase'] = True
                file_info['purchase'] = {'name': fname, 'rows': len(df)}
                match_type = "采购表"

            elif '头程' in fname:
                dfs['head'] = df; status_flags['head'] = True
                file_info['head'] = {'name': fname, 'rows': len(df)}
                match_type = "头程表"

            elif '尾程' in fname:
                dfs['tail'] = df; status_flags['tail'] = True
                file_info['tail'] = {'name': fname, 'rows': len(df)}
                match_type = "尾程表"

            # ✅ 新增：关税成本（文件名含“关税”）
            elif '关税' in fname:
                dfs['tariff'] = df; status_flags['tariff'] = True; match_type = "关税成本表"

            # ✅ 新增：流量和出单占比数据（用于周会PV资产和流量漏斗）
            elif '流量' in fname and '占比' in fname:
                dfs['traffic_funnel'] = df; status_flags['traffic_funnel'] = True
                file_info['traffic_funnel'] = {'name': fname, 'rows': len(df)}
                match_type = "流量漏斗数据"

            log_info += f"- 判定结果: {match_type}"
            debug_logs.append(log_info)

        except Exception as e:
            st.error(f"❌ 读取文件 {fname} 失败: {str(e)}")
            debug_logs.append(f"❌ **{fname}** 读取失败: {str(e)}")

    time.sleep(0.2)
    status_text.text("✅ 解析完成！")
    progress_bar.empty()
    return dfs, status_flags, debug_logs, file_info

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
            st.markdown("---")
            st.subheader("📣 广告明细（层级二 Video 透视用）")

            ads_detail_file = st.file_uploader(
                "上传【广告明细】(文件名可叫：广告订单)",
                type=["xlsx", "xls", "csv"],
                key="uploader_ads_detail"
            )

            def _read_any(file):
                import pandas as pd
                name = (file.name or "").lower()
                if name.endswith(".csv"):
                    return pd.read_csv(file)

                # xlsx / xls：尽量读对“广告订单”sheet
                xls = pd.read_excel(file, sheet_name=None)
                if "广告订单" in xls:
                    return xls["广告订单"]
                return list(xls.values())[0]

            if ads_detail_file is not None:
                df_ads_detail = _read_any(ads_detail_file)

                # ✅ 核心：层级二优先从 session_state 拿
                st.session_state["df_ads_detail"] = df_ads_detail

                # ✅ 兼容老逻辑（若你某些函数还在 globals() 里找）
                globals()["df_ads_detail"] = df_ads_detail

                st.success(f"已加载广告明细：{ads_detail_file.name}（{len(df_ads_detail):,} 行）")

        else:
            st.info("💡 正在扫描当前目录下的数据文件...")
            current_dir = os.getcwd()
            uploaded_files = [os.path.join(current_dir, f) for f in os.listdir(current_dir) if f.endswith(('.csv', '.xlsx', '.xls'))]
            st.write(f"找到 {len(uploaded_files)} 个文件")

        dfs, flags, logs, file_info = {}, {}, [], {}
        if uploaded_files:
            dfs, flags, logs, file_info = load_uploaded_files(uploaded_files)

            st.markdown("### 📊 文件就位状态")
            with st.expander("财务核心数据", expanded=True):
                # 显示状态 + 文件名 + 行数
                def _fmt_file_status(key, label, optional=False):
                    has_file = flags.get(key)
                    info = file_info.get(key, {})
                    fname = info.get('name', '')
                    rows = info.get('rows', 0)
                    icon = '✅' if has_file else ('⚠️' if optional else '❌')
                    if has_file and fname:
                        return f"{icon} **{label}**  \n`{fname}` ({rows}行)"
                    elif has_file:
                        return f"{icon} **{label}**"
                    else:
                        return f"{icon} **{label}** (未找到)"
                
                st.markdown(_fmt_file_status('orders', '订单表（必须）'))
                st.markdown(_fmt_file_status('ads', '广告表'))
                st.markdown(_fmt_file_status('purchase', '采购成本'))
                st.markdown(_fmt_file_status('head', '头程成本'))
                st.markdown(_fmt_file_status('tail', '尾程成本'))
                st.markdown(_fmt_file_status('tariff', '关税成本', optional=True))
                st.markdown(_fmt_file_status('traffic_funnel', '流量漏斗数据', optional=True))
                st.markdown(_fmt_file_status('spu_sku', 'SPU映射'))
                st.markdown(_fmt_file_status('mapping', 'PID映射', optional=True))
                
            with st.expander("达人分析数据", expanded=True):
                st.markdown(_fmt_file_status('affiliate', '联盟订单表'))
                st.markdown(_fmt_file_status('transaction', 'Transaction表'))

            with st.expander("🕵️ 文件诊断详情（Debug）", expanded=False):
                for log in logs:
                    st.markdown(log)
                    st.divider()

        st.divider()
        st.subheader("🎯 目标设定（V2）")
        target_revenue = st.number_input("本月营收目标 ($)", value=0.0, step=1000.0)
        target_profit_rate = st.number_input("目标利润率（默认15%）", value=0.15, step=0.01, format="%.2f")

        st.divider()
        with st.expander("📘 系统使用手册（新同事必读）", expanded=False):

            with st.expander("🚀 快速上手（3分钟）", expanded=False):
                st.markdown(MANUAL_QUICKSTART_MD)

            with st.expander("🏠 经营总览", expanded=False):
                st.markdown(MANUAL_HOME_MD)

            with st.expander("📦 SPU 分析", expanded=False):
                st.markdown(MANUAL_SPU_MD)

            with st.expander("📺 广告深度诊断", expanded=False):
                st.markdown(MANUAL_ADS_MD)

            with st.expander("🤝 达人合作分析", expanded=False):
                st.markdown(MANUAL_CREATOR_MD)

            with st.expander("🧠 AI待办分析", expanded=False):
                st.markdown(MANUAL_AI_MD)

            with st.expander("📏 判定标准大全", expanded=False):
                st.markdown(MANUAL_RULES_MD)


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

        time_progress = 0.0
        if max_date is not None:
            try:
                d = pd.to_datetime(max_date)
                days_in_month = pd.Period(d.strftime("%Y-%m")).days_in_month
                time_progress = min(max(d.day / days_in_month, 0.0), 1.0)
            except:
                time_progress = 0.0

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

        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📋 周会概览",      # 新增，放在第一位
    "🏠 经营总览", 
    "📦 SPU 分析", 
    "📊 类目分析", 
    "📄 SKU 明细", 
    "📺 广告深度诊断", 
    "🤝 达人合作分析", 
    "🧠 AI 操盘手"
])

        # ===== ① 周会概览 =====
        with tab1:
            render_weekly_overview_page(out, meta, targets)

        with tab2:
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
            
            # ============== 新增：YOY同比分析 ==============
            with st.expander("📊 YOY 同比分析（vs 2025年同期）", expanded=True):
                if df_shop_raw is not None and not df_shop_raw.empty:
                    # 检查是否有YOY数据
                    has_yoy = '退款后营收_YOY%' in df_shop_raw.columns
                    
                    if has_yoy:
                        st.markdown("### 店铺维度同比")
                        
                        # 展示当前月与2025年同期对比
                        for _, row in df_shop_raw.iterrows():
                            month = row.get('Month', '未知')
                            rev_2026 = row.get('退款后营收', 0)
                            rev_2025 = row.get('2025年同期退款后营收', 0)
                            yoy_pct = row.get('退款后营收_YOY%', None)
                            
                            col_yoy1, col_yoy2, col_yoy3, col_yoy4 = st.columns(4)
                            col_yoy1.metric("月份", month)
                            col_yoy2.metric("2026年退款后营收", f"${rev_2026:,.0f}")
                            col_yoy3.metric("2025年同期", f"${rev_2025:,.0f}")
                            if yoy_pct is not None:
                                delta_color = "normal" if yoy_pct >= 0 else "inverse"
                                col_yoy4.metric("YOY增长率", f"{yoy_pct:+.1f}%", delta_color=delta_color)
                            else:
                                col_yoy4.metric("YOY增长率", "N/A")
                        
                        st.divider()
                        st.markdown("### 各项费率对比")
                        
                        # 费率对比表
                        rate_cols = ['退款率', '采购成本率', '头程成本率', '尾程费率', '关税成本率',
                                     '广告费率', '达人佣金费率', '样品费率', '品牌费率', '仓租费率',
                                     '其他物流费用率', '平台费率', '售后成本率']
                        
                        # 费率名称映射（从2026年列名到2025年财务管报名）
                        rate_name_mapping = {
                            '退款率': '退款率',
                            '采购成本率': '采购成本率',
                            '头程成本率': '头程成本率',
                            '尾程费率': '尾程费率',
                            '关税成本率': '关税成本率',
                            '广告费率': '广告费率',
                            '达人佣金费率': '达人佣金费率',
                            '样品费率': '样品费率',
                            '品牌费率': '品牌费率',
                            '仓租费率': '仓租费率',
                            '其他物流费用率': '其他物流费用',
                            '平台费率': '平台费率',
                            '售后成本率': '售后成本率'
                        }
                        
                        yoy_rate_data = []
                        yoy_data = out.get('yoy_data', {})
                        
                        for _, row in df_shop_raw.iterrows():
                            month = row.get('Month', '未知')
                            month_2025 = month.replace('2026-', '2025-') if isinstance(month, str) else None
                            
                            for rate_col in rate_cols:
                                if rate_col in row:
                                    yoy_col = f"{rate_col}_同比%"
                                    yoy_val = row.get(yoy_col, None)
                                    
                                    # 从yoy_data获取2025年费率
                                    rate_2025_name = rate_name_mapping.get(rate_col, rate_col)
                                    rate_2025_val = None
                                    if yoy_data and month_2025 in yoy_data and rate_2025_name in yoy_data[month_2025]:
                                        rate_2025_val = yoy_data[month_2025][rate_2025_name]
                                    
                                    # 计算YOY变化（百分点）
                                    rate_2026_val = row.get(rate_col, 0)
                                    if rate_2025_val is not None and rate_2026_val is not None:
                                        yoy_change_pp = (rate_2026_val - rate_2025_val) * 100  # 转换为百分点
                                        yoy_change_str = f"{yoy_change_pp:+.2f}"
                                    else:
                                        yoy_change_str = "-"
                                    
                                    yoy_rate_data.append({
                                        '月份': month,
                                        '费率项': rate_col,
                                        '2026年': f"{rate_2026_val:.2%}",
                                        '2025年': f"{rate_2025_val:.2%}" if rate_2025_val is not None else "-",
                                        'YOY变化(pp)': yoy_change_str
                                    })
                        
                        if yoy_rate_data:
                            df_yoy_rates = pd.DataFrame(yoy_rate_data)
                            st.dataframe(df_yoy_rates, use_container_width=True)
                    else:
                        st.info("💡 暂无YOY对比数据。请确保已上传2025年订单表和财务管报。")
                else:
                    st.info("💡 暂无数据")

            with st.expander("🧪 SKU 成本映射自检（缺失映射列表）", expanded=False):
                sku_audit_missing = out.get('sku_audit_missing')
                if sku_audit_missing is None:
                    st.info("⚠️ 未生成校验数据。请确认已上传订单表并执行测算。")
                elif sku_audit_missing.empty:
                    st.success("✅ 所有订单SKU的成本项（采购/头程/尾程/关税/广告/佣金/样品）与SPU映射均已匹配，无缺失。")
                else:
                    st.warning(f"⚠️ 发现 {len(sku_audit_missing)} 个 SKU 存在缺失映射（请确认对应成本/映射表是否完整）。")
                    st.dataframe(sku_audit_missing, use_container_width=True)


        with tab3:
            # =========================
            # SPU 决策面板（周会拍板版）
            # 顺序：盈亏分析（A/B/C清单）→ 诊断图谱（波士顿/退款影响/营销构成）→ SPU表
            # =========================

            # ================= 新增：计算明细与广告校验（用于手工对比调试） =================
            with st.expander("🔍 计算明细 & 广告校验（用于与手工计算对比）", expanded=True):
                col_dbg1, col_dbg2 = st.columns(2)
                
                with col_dbg1:
                    st.markdown("**📊 广告成本分摊校验**")
                    ads_val = out.get('ads_validation', {})
                    val_summary = ads_val.get('validation_summary', {})
                    
                    if val_summary:
                        st.write(f"广告表总花费: **${val_summary.get('广告表总花费', 0):,.2f}**")
                        st.write(f"程序分摊总花费: **${val_summary.get('程序分摊总花费', 0):,.2f}**")
                        st.write(f"分摊差额: **${val_summary.get('分摊差额', 0):,.2f}**")
                        st.write(f"未映射PID数量: **{val_summary.get('未映射PID数量', 0)}**")
                        st.write(f"未映射PID花费: **${val_summary.get('未映射PID花费', 0):,.2f}**")
                        st.write(f"映射覆盖率: **{val_summary.get('映射覆盖率', 0):.2%}**")
                        
                        if ads_val.get('errors'):
                            st.error("校验错误: " + "; ".join(ads_val['errors']))
                    else:
                        st.info("广告校验数据未生成")
                    
                    # 显示未映射的 PID
                    unmapped_df = ads_val.get('unmapped_pids')
                    if unmapped_df is not None and not unmapped_df.empty:
                        st.warning(f"⚠️ 发现 {len(unmapped_df)} 个未映射的 PID")
                        st.dataframe(unmapped_df.head(20), use_container_width=True, hide_index=True)
                        
                        # 提供下载
                        csv_unmapped = unmapped_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="⬇️ 下载未映射PID清单",
                            data=csv_unmapped,
                            file_name="unmapped_pids.csv",
                            mime="text/csv"
                        )
                
                with col_dbg2:
                    st.markdown("**📋 SKU计算明细导出**")
                    calc_detail = out.get('sku_calculation_detail')
                    if calc_detail is not None and not calc_detail.empty:
                        st.success(f"✅ 已生成 {len(calc_detail)} 个 SKU 的计算明细")
                        st.caption("此表包含每个SKU的成本计算过程，可与手工计算对比")
                        
                        # 提供下载
                        csv_detail = calc_detail.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="⬇️ 下载SKU计算明细CSV",
                            data=csv_detail,
                            file_name="sku_calculation_detail.csv",
                            mime="text/csv",
                            key="download_calc_detail"
                        )
                        
                        # 显示前几行预览
                        with st.expander("预览前5行", expanded=False):
                            st.dataframe(calc_detail.head(), use_container_width=True)
                    else:
                        st.info("计算明细未生成")
                
                # 显示 PID 分摊详情
                pid_detail = ads_val.get('pid_detail')
                if pid_detail is not None and not pid_detail.empty:
                    with st.expander("📑 PID 广告分摊详情", expanded=False):
                        st.write("此表展示每个PID的广告花费及映射情况：")
                        st.dataframe(pid_detail.head(50), use_container_width=True, hide_index=True)
                        
                        # 按是否映射分组统计
                        st.write("**映射状态统计：**")
                        map_stats = pid_detail.groupby('是否映射')['原始广告花费'].agg(['count', 'sum'])
                        st.dataframe(map_stats, use_container_width=True)

            # ---------- 侧边栏：阈值（可调 + 默认值） ----------
            with st.sidebar.expander("🧭 SPU 决策阈值（可调）", expanded=False):
                # 目标利润率：优先用你侧边栏“目标利润率”，这里给一个可覆盖的输入
                default_target_pr = float(targets.get("target_profit_rate", 0.15) or 0.15)
                spu_target_profit_rate = st.number_input("目标利润率", value=default_target_pr, step=0.01, format="%.2f")

                min_profit_abs = st.number_input("增投：最低利润额 ($)", value=300.0, step=50.0)
                min_rev_scale = st.number_input("增投：最低营收 ($)", value=1000.0, step=100.0)

                # ✅你要求的默认值
                refund_warn = st.number_input("退款率警戒线", value=0.05, step=0.01, format="%.2f")
                refund_high = st.number_input("退款率高危线", value=0.08, step=0.01, format="%.2f")

                mkt_cap = st.number_input("营销费比上限（增投候选需≤）", value=0.25, step=0.01, format="%.2f")
                mkt_high = st.number_input("营销费比高危线", value=0.30, step=0.01, format="%.2f")

                stoploss_rev = st.number_input("止损规模线（负利润且营收≥才上榜）", value=1000.0, step=100.0)

                show_labels = st.checkbox("图上显示 SPU 标签（仅Top N）", value=True)
                top_n = st.slider("图上显示标签 Top N（按退款后营收）", min_value=8, max_value=30, value=15, step=1)

            # ---------- 数据准备 ----------
            st.subheader("盈亏分析（SPU）")

            if df_spu_raw is None or df_spu_raw.empty:
                st.info("💡 SPU 原始数据为空，无法生成 SPU 决策面板。")
            else:
                df_pl = df_spu_raw.copy()

                # 必备字段兜底（避免缺列报错）
                must_cols = [
                    'SPU', '销量', '退款前营收', '退款后营收', '退款单数', '利润额', '利润率',
                    '总广告投放费', '总达人佣金', '总样品费', '总营销费比', '退款率'
                ]
                for c in must_cols:
                    if c not in df_pl.columns:
                        df_pl[c] = 0

                # 数值化
                num_cols = [
                    '销量', '退款前营收', '退款后营收', '退款单数', '利润额', '利润率',
                    '总广告投放费', '总达人佣金', '总样品费', '总营销费比', '退款率'
                ]
                for c in num_cols:
                    df_pl[c] = pd.to_numeric(df_pl[c], errors='coerce').fillna(0.0)

                # 派生指标（营销拆分）
                rev = df_pl['退款后营收'].replace(0, np.nan)
                df_pl['广告费比'] = (df_pl['总广告投放费'] / rev).fillna(0.0)
                df_pl['佣金比'] = (df_pl['总达人佣金'] / rev).fillna(0.0)
                df_pl['样品费比'] = (df_pl['总样品费'] / rev).fillna(0.0)

                # 盈亏标签
                df_pl['盈亏'] = df_pl['利润率'].apply(lambda x: "✅ 盈利" if x >= 0 else "❌ 亏损")

                # 决策篮子：A增投 / B可修复 / C止损 / D其他
                def decide_bucket(r):
                    revenue = float(r.get('退款后营收', 0) or 0)
                    profit = float(r.get('利润额', 0) or 0)
                    pr = float(r.get('利润率', 0) or 0)
                    rr = float(r.get('退款率', 0) or 0)
                    mr = float(r.get('总营销费比', 0) or 0)

                    # C：止损预警（拖累型）
                    if (profit < 0 and revenue >= stoploss_rev) or (rr >= refund_high) or (mr >= mkt_high):
                        return "C 止损预警"

                    # A：增投候选（稳 + 能放量）
                    if (pr >= spu_target_profit_rate) and (profit >= min_profit_abs) and (revenue >= min_rev_scale) and (rr <= refund_warn) and (mr <= mkt_cap):
                        return "A 增投候选"

                    # B：可修复（不一定亏，但明显短板）
                    # 1) 有规模，利润率未达标但接近目标
                    if (revenue >= min_rev_scale) and (pr < spu_target_profit_rate) and (pr >= -0.05):
                        return "B 可修复"
                    # 2) 利润率达标但退款率/营销费比超线
                    if (pr >= spu_target_profit_rate) and ((rr > refund_warn) or (mr > mkt_cap)):
                        return "B 可修复"
                    # 3) 小规模亏损：先修，不一定要立刻止损
                    if (profit < 0) and (revenue < stoploss_rev):
                        return "B 可修复"

                    return "D 其他观察"

                df_pl['决策篮子'] = df_pl.apply(decide_bucket, axis=1)

                # 主因判定（运营向：只给一个最该先动的原因）
                def root_cause(r):
                    rr = float(r.get('退款率', 0) or 0)
                    mr = float(r.get('总营销费比', 0) or 0)
                    ads = float(r.get('广告费比', 0) or 0)
                    aff = float(r.get('佣金比', 0) or 0)
                    samp = float(r.get('样品费比', 0) or 0)

                    if rr >= refund_warn:
                        return "退款偏高"
                    if mr >= mkt_cap:
                        biggest = max([(ads, "广告费"), (aff, "达人佣金"), (samp, "样品费")], key=lambda x: x[0])[1]
                        return f"营销偏高（{biggest}为主）"
                    if float(r.get('利润率', 0) or 0) < 0:
                        return "利润为负（先控投放&控寄样）"
                    return "结构正常"

                def action_suggest(r):
                    bucket = r.get('决策篮子', '')
                    cause = r.get('主因', '')
                    if bucket.startswith("A"):
                        return "增投：复制高转化素材结构 + 扩人群/扩达人，优先稳 ROAS"
                    if bucket.startswith("C"):
                        if "退款" in cause:
                            return "止损：先停投放/停寄样，补证据镜头+使用边界/尺码说明，降退款再回测"
                        if "广告" in cause:
                            return "止损：先降预算/切人群，换新Hook+证据镜头，再决定是否回拉"
                        if "达人" in cause:
                            return "止损：暂停低效寄样，改KOL分层+佣金策略，先保正毛利"
                        return "止损：暂停扩量，优先修商品页表达+素材证据后再回测"
                    if bucket.startswith("B"):
                        if "退款" in cause:
                            return "修复：补强证据镜头/尺码说明/使用限制，先把退款压到警戒线以下"
                        if "营销" in cause:
                            return "修复：控投放节奏+优化承接，寄样按ROI分层，降低营销费比"
                        return "修复：优化投放结构+卖点前置，达标后再进A"
                    return "观察：数据量偏小或结构未触发阈值，继续测"

                df_pl['主因'] = df_pl.apply(root_cause, axis=1)
                df_pl['建议动作'] = df_pl.apply(action_suggest, axis=1)
                                # ================= 新增：第一责任位（不影响A/B/C，仅用于分工） =================
                df_pl['第一责任位'] = df_pl.apply(infer_spu_owner, axis=1)


                # =========================
                # ✅ 轻联动：达人结构数据准备（不影响A/B/C）
                # 来源优先级：
                # - 订单级“联盟订单表 affiliate”：可得 Creator → 用于集中度/达人数量
                # - SPU级“Transaction表”：可得 Affiliate_Rate / Videos / OutputPerVideo（已有 creator_data['spu_perf']）
                # =========================

                creator_stats = None         # SPU维度：Creators / Top1Share / Top3Share / HealthTag
                creator_long_for_viz = None  # 用于可视化：每个SPU的Top达人占比
                spu_aff_perf = None          # Transaction汇总：Affiliate_Rate / Videos / OutputPerVideo（若有）

                # 1) Transaction 表的 SPU 汇总（你 Tab5 已有逻辑产出 creator_data['spu_perf']）
                try:
                    if creator_data and creator_data.get('spu_perf') is not None:
                        spu_aff_perf = creator_data['spu_perf'].copy()
                        # 规范字段名，后面表格联动更稳
                        # spu_aff_perf: ['SPU','Affiliate_GMV','Videos','Lives','Shop_GMV_Before','Affiliate_Rate','OutputPerVideo']
                except:
                    spu_aff_perf = None

                # 2) affiliate 表：计算“出单达人集中度”（Creators / Top1 / Top3）
                try:
                    df_aff = dfs.get('affiliate')
                    df_spu_sku = dfs.get('spu_sku')
                    if df_aff is not None and df_spu_sku is not None and not df_aff.empty and not df_spu_sku.empty:
                        c_creator = COLUMN_CONFIG['affiliate']['creator']
                        c_gmv = COLUMN_CONFIG['affiliate']['gmv']
                        c_sku = COLUMN_CONFIG['affiliate']['sku']

                        if c_creator in df_aff.columns and c_gmv in df_aff.columns and c_sku in df_aff.columns:
                            tmp = df_aff.copy()
                            tmp['Creator'] = tmp[c_creator].astype(str).str.strip()
                            tmp['SKU_Clean'] = clean_text(tmp, c_sku)
                            tmp['GMV_Val'] = tmp[c_gmv].apply(clean_money)

                            # SKU -> SPU 映射
                            sku_to_spu_dict = build_sku_to_spu_dict(df_spu_sku)
                            tmp['SPU'] = tmp['SKU_Clean'].map(sku_to_spu_dict).fillna(tmp['SKU_Clean'])

                            # 过滤无效
                            tmp = tmp[(tmp['SPU'].astype(str).str.strip() != "") & (tmp['Creator'].astype(str).str.strip() != "")].copy()

                            if not tmp.empty:
                                # SPU-达人 GMV 聚合
                                g = tmp.groupby(['SPU', 'Creator'], as_index=False)['GMV_Val'].sum()
                                g = g[g['GMV_Val'] > 0].copy()

                                # 计算集中度
                                stats_rows = []
                                viz_rows = []

                                for spu, sub in g.groupby('SPU'):
                                    sub = sub.sort_values('GMV_Val', ascending=False).copy()
                                    total = float(sub['GMV_Val'].sum() or 0)
                                    creators = int(sub['Creator'].nunique())

                                    top1 = float(sub.iloc[0]['GMV_Val']) if len(sub) >= 1 else 0.0
                                    top3 = float(sub.head(3)['GMV_Val'].sum()) if len(sub) >= 3 else float(sub['GMV_Val'].sum())

                                    top1_share = (top1 / total) if total > 0 else 0.0
                                    top3_share = (top3 / total) if total > 0 else 0.0

                                    # 健康度标签（可按你习惯微调阈值）
                                    if top1_share >= 0.70:
                                        tag = "🔴 高度依赖"
                                    elif top1_share >= 0.45:
                                        tag = "🟠 中度集中"
                                    else:
                                        tag = "🟢 矩阵健康"

                                    stats_rows.append({
                                        'SPU': spu,
                                        'Creators': creators,
                                        'Top1Share': top1_share,
                                        'Top3Share': top3_share,
                                        'CreatorHealth': tag
                                    })

                                    # 可视化数据：取Top3 + Others（更清爽）
                                    topN = sub.head(3).copy()
                                    for i, r in topN.iterrows():
                                        viz_rows.append({
                                            'SPU': spu,
                                            'Group': f"Top{(topN.index.get_loc(i)+1)}",
                                            'Share': (float(r['GMV_Val']) / total) if total > 0 else 0.0
                                        })
                                    others_share = 1.0 - sum([x['Share'] for x in viz_rows if x['SPU'] == spu])
                                    others_share = max(others_share, 0.0)
                                    viz_rows.append({'SPU': spu, 'Group': 'Others', 'Share': others_share})

                                creator_stats = pd.DataFrame(stats_rows) if stats_rows else None
                                creator_long_for_viz = pd.DataFrame(viz_rows) if viz_rows else None
                except:
                    creator_stats = None
                    creator_long_for_viz = None

                # =========================
                # ✅ 轻联动：在“建议动作”末尾追加风险提示（不影响A/B/C）
                # =========================
                if creator_stats is not None and not creator_stats.empty:
                    cs_map = creator_stats.set_index('SPU').to_dict(orient='index')

                    def append_creator_risk(row):
                        spu = str(row.get('SPU', ''))
                        base = str(row.get('建议动作', ''))
                        info = cs_map.get(spu)
                        if not info:
                            return base

                        top1 = float(info.get('Top1Share', 0) or 0)
                        creators = int(info.get('Creators', 0) or 0)

                        # 仅追加一条最关键提示，避免噪音
                        tip = ""
                        if top1 >= 0.70:
                            tip = "｜⚠️ 达人过度集中：Top1≥70%，先做去依赖测试（扩达人矩阵）"
                        elif creators > 0 and creators < 5:
                            tip = "｜⚠️ 达人矩阵偏薄：出单达人<5，先补达人池再扩量"

                        return base + tip if tip else base

                    df_pl['建议动作'] = df_pl.apply(append_creator_risk, axis=1)

                                

                # ---------- 0) 周会总览 KPI ----------
                st.markdown("### 🧾 周会总览")
                profit_cnt = int((df_pl['利润额'] >= 0).sum())
                loss_cnt = int((df_pl['利润额'] < 0).sum())
                profit_sum = float(df_pl.loc[df_pl['利润额'] >= 0, '利润额'].sum())
                loss_sum = float(df_pl.loc[df_pl['利润额'] < 0, '利润额'].sum())  # 负数
                net_sum = float(df_pl['利润额'].sum())

                a_cnt = int((df_pl['决策篮子'] == "A 增投候选").sum())
                b_cnt = int((df_pl['决策篮子'] == "B 可修复").sum())
                c_cnt = int((df_pl['决策篮子'] == "C 止损预警").sum())

                k1, k2, k3, k4, k5, k6 = st.columns(6)
                k1.metric("盈利 SPU 数", f"{profit_cnt}")
                k2.metric("亏损 SPU 数", f"{loss_cnt}")
                k3.metric("盈利总额", f"${profit_sum:,.0f}")
                k4.metric("亏损总额", f"${loss_sum:,.0f}")
                k5.metric("净利润合计", f"${net_sum:,.0f}")
                k6.metric("A/B/C 数量", f"{a_cnt}/{b_cnt}/{c_cnt}")
                
                # ================= 新增：SPU维度YOY同比Top10 =================
                if df_spu_raw is not None and '退款后营收_YOY%' in df_spu_raw.columns:
                    st.divider()
                    with st.expander("📊 SPU 维度 YOY 同比（vs 2025年同期）", expanded=False):
                        # YOY增长最快Top10
                        st.markdown("#### 🚀 YOY 增长最快 Top 10 SPU")
                        df_yoy_growth = df_spu_raw[df_spu_raw['2025年同期退款后营收'] > 0].copy()
                        df_yoy_growth = df_yoy_growth.sort_values('退款后营收_YOY%', ascending=False).head(10)
                        
                        if not df_yoy_growth.empty:
                            show_yoy = df_yoy_growth[['SPU', '退款后营收', '2025年同期退款后营收', '退款后营收_YOY%']].copy()
                            show_yoy['退款后营收'] = show_yoy['退款后营收'].apply(lambda x: f"${float(x or 0):,.0f}")
                            show_yoy['2025年同期退款后营收'] = show_yoy['2025年同期退款后营收'].apply(lambda x: f"${float(x or 0):,.0f}")
                            show_yoy['退款后营收_YOY%'] = show_yoy['退款后营收_YOY%'].apply(lambda x: f"{float(x or 0):+.1f}%")
                            st.dataframe(show_yoy, use_container_width=True, hide_index=True)
                        else:
                            st.info("暂无YOY增长数据")
                        
                        # YOY下降最多Top10
                        st.markdown("#### 📉 YOY 下降最多 Top 10 SPU")
                        df_yoy_decline = df_spu_raw[df_spu_raw['2025年同期退款后营收'] > 0].copy()
                        df_yoy_decline = df_yoy_decline.sort_values('退款后营收_YOY%', ascending=True).head(10)
                        
                        if not df_yoy_decline.empty:
                            show_decline = df_yoy_decline[['SPU', '退款后营收', '2025年同期退款后营收', '退款后营收_YOY%']].copy()
                            show_decline['退款后营收'] = show_decline['退款后营收'].apply(lambda x: f"${float(x or 0):,.0f}")
                            show_decline['2025年同期退款后营收'] = show_decline['2025年同期退款后营收'].apply(lambda x: f"${float(x or 0):,.0f}")
                            show_decline['退款后营收_YOY%'] = show_decline['退款后营收_YOY%'].apply(lambda x: f"{float(x or 0):+.1f}%")
                            st.dataframe(show_decline, use_container_width=True, hide_index=True)
                        else:
                            st.info("暂无YOY下降数据")

                # ================= 0.8) Top10 利润贡献 & Top10 亏损（新增，放在周会总览下面） =================
                st.divider()
                st.markdown("### 🏆 Top 10 利润贡献 & 🔻 Top 10 亏损（SPU）")

                def _fmt_money(x): 
                    return f"${float(x or 0):,.0f}"
                def _fmt_pct(x): 
                    return f"{float(x or 0):.2%}"

                top_show_cols = ['SPU', '第一责任位', '退款后营收', '利润额', '利润率', '退款率', '总营销费比', '主因', '建议动作']

                for c in top_show_cols:
                    if c not in df_pl.columns:
                        df_pl[c] = 0

                left_top, right_top = st.columns(2)

                with left_top:
                    st.markdown("#### 🟢 Top 10 利润贡献（按利润额）")
                    df_top_profit = df_pl.sort_values('利润额', ascending=False).head(10).copy()
                    if df_top_profit.empty:
                        st.info("暂无数据。")
                    else:
                        df_tp = df_top_profit[top_show_cols].copy()
                        df_tp['退款后营收'] = df_tp['退款后营收'].apply(_fmt_money)
                        df_tp['利润额'] = df_tp['利润额'].apply(_fmt_money)
                        for c in ['利润率', '退款率', '总营销费比']:
                            df_tp[c] = df_tp[c].apply(_fmt_pct)
                        st.dataframe(df_tp, use_container_width=True, hide_index=True)

                with right_top:
                    st.markdown("#### 🔴 Top 10 亏损（按利润额）")
                    df_top_loss = df_pl.sort_values('利润额', ascending=True).head(10).copy()
                    if df_top_loss.empty:
                        st.info("暂无数据。")
                    else:
                        df_tl = df_top_loss[top_show_cols].copy()
                        df_tl['退款后营收'] = df_tl['退款后营收'].apply(_fmt_money)
                        df_tl['利润额'] = df_tl['利润额'].apply(_fmt_money)
                        for c in ['利润率', '退款率', '总营销费比']:
                            df_tl[c] = df_tl[c].apply(_fmt_pct)
                        st.dataframe(df_tl, use_container_width=True, hide_index=True)





                # ---------- 0.5) 利润率区间结构（营收占比 & 贡献利润率） ----------
                st.divider()
                st.markdown("### 🧩 利润率区间结构（营收占比 & 贡献利润率）")

                # 可调：低利润上限（默认 10%），中利润上限=目标利润率（你侧边栏已有默认）
                # 这里做成轻量可调，不影响原有阈值体系
                low_profit_upper = st.number_input("低利润区间上限（默认 10%）", value=0.10, step=0.01, format="%.2f")

                # 防止用户把阈值设反
                low_profit_upper = float(max(low_profit_upper, 0.0))
                mid_profit_upper = float(max(spu_target_profit_rate, low_profit_upper))

                # 统一口径：利润率 = 利润额 / 退款后营收（你已确认）
                # 区间贡献利润率：区间利润额 / 区间退款后营收（加权口径）
                df_seg = df_pl[['SPU', '退款后营收', '利润额', '利润率']].copy()
                df_seg['退款后营收'] = pd.to_numeric(df_seg['退款后营收'], errors='coerce').fillna(0.0)
                df_seg['利润额'] = pd.to_numeric(df_seg['利润额'], errors='coerce').fillna(0.0)
                df_seg['利润率'] = pd.to_numeric(df_seg['利润率'], errors='coerce').fillna(0.0)

                # 仅统计有营收的SPU（避免 0/0 的噪声）
                df_seg = df_seg[df_seg['退款后营收'] > 0].copy()

                if df_seg.empty:
                    st.info("💡 本周期没有可用于区间结构分析的SPU（退款后营收为0）。")
                else:
                    # 区间定义：亏损 / 低利润 / 中利润 / 健康
                    # 亏损：<0；低利润：0~low；中利润：low~target；健康：>=target
                    def profit_band(pr):
                        pr = float(pr or 0)
                        if pr < 0:
                            return "🔴 亏损（<0）"
                        if pr < low_profit_upper:
                            return f"🟠 低利润（0–{low_profit_upper:.0%}）"
                        if pr < mid_profit_upper:
                            return f"🟡 中利润（{low_profit_upper:.0%}–{mid_profit_upper:.0%}）"
                        return f"🟢 健康（≥{mid_profit_upper:.0%}）"

                    df_seg['利润率区间'] = df_seg['利润率'].apply(profit_band)

                    # 汇总
                    seg = df_seg.groupby('利润率区间').agg(
                        SPU数=('SPU', 'nunique'),
                        区间营收=('退款后营收', 'sum'),
                        区间利润=('利润额', 'sum')
                    ).reset_index()

                    # 排序：按“亏损→低→中→健康”
                    order_labels = [
                        "🔴 亏损（<0）",
                        f"🟠 低利润（0–{low_profit_upper:.0%}）",
                        f"🟡 中利润（{low_profit_upper:.0%}–{mid_profit_upper:.0%}）",
                        f"🟢 健康（≥{mid_profit_upper:.0%}）"
                    ]
                    seg['__order__'] = seg['利润率区间'].apply(lambda x: order_labels.index(x) if x in order_labels else 999)
                    seg = seg.sort_values('__order__').drop(columns='__order__')

                    total_rev = float(seg['区间营收'].sum() or 0)
                    total_profit = float(seg['区间利润'].sum() or 0)

                    seg['营收占比'] = seg['区间营收'].apply(lambda x: (x / total_rev) if total_rev > 0 else 0.0)
                    # 利润占比：当 total_profit 为负或为0时，占比会出现“符号反直觉”，这里仍给出，并在页面提示
                    seg['利润占比'] = seg['区间利润'].apply(lambda x: (x / total_profit) if total_profit != 0 else 0.0)
                    seg['区间贡献利润率'] = seg.apply(lambda r: (r['区间利润'] / r['区间营收']) if r['区间营收'] > 0 else 0.0, axis=1)

                    # ---- 可视化：营收占比 100% 堆叠条（最直观）----
                    seg_plot = seg[['利润率区间', '营收占比']].copy()
                    seg_plot.rename(columns={'营收占比': 'Share'}, inplace=True)

                    bar = alt.Chart(seg_plot).mark_bar().encode(
                        x=alt.X('Share:Q', stack='normalize', title='营收占比', axis=alt.Axis(format='%')),
                        y=alt.Y('营收占比:Q', title=None, axis=None),  # 只用一根横条即可
                        color=alt.Color('利润率区间:N', title='利润率区间'),
                        tooltip=[
                            alt.Tooltip('利润率区间:N', title='区间'),
                            alt.Tooltip('Share:Q', title='营收占比', format='.2%')
                        ]
                    ).properties(height=60)

                    st.altair_chart(bar, use_container_width=True)

                    # ---- 数据表：可复制/可筛选 ----
                    seg_show = seg.copy()
                    seg_show['区间营收'] = seg_show['区间营收'].apply(lambda x: f"${float(x or 0):,.0f}")
                    seg_show['区间利润'] = seg_show['区间利润'].apply(lambda x: f"${float(x or 0):,.0f}")
                    seg_show['营收占比'] = seg_show['营收占比'].apply(lambda x: f"{float(x or 0):.2%}")
                    seg_show['利润占比'] = seg_show['利润占比'].apply(lambda x: f"{float(x or 0):.2%}")
                    seg_show['区间贡献利润率'] = seg_show['区间贡献利润率'].apply(lambda x: f"{float(x or 0):.2%}")

                    st.dataframe(seg_show, use_container_width=True, hide_index=True)

                    st.caption("说明：利润率口径 = 利润额 / 退款后营收；区间贡献利润率 = 区间利润额 / 区间退款后营收。若全店利润为负，'利润占比' 会出现符号反直觉（数学上仍成立），此时更建议看“营收占比 + 区间贡献利润率”。")





                # ---------- 1) 三张行动清单 A/B/C ----------
                st.divider()
                st.markdown("### ✅ 周会拍板：A 增投候选 / B 可修复 / C 止损预警")

               
                    # ✅ A：规则说明区块（默认折叠，点开才展开）
                with st.expander("📌 查看 A / B / C 判定标准（点击展开）", expanded=False):
                    st.info(
                        f"""
                **判定口径（当期 SPU）**：以「退款后营收 / 利润额 / 利润率 / 退款率 / 总营销费比」做周会决策。

                **A 增投候选（稳 + 可放量）同时满足：**
                - 利润率 ≥ {spu_target_profit_rate:.0%}
                - 利润额 ≥ ${min_profit_abs:,.0f}
                - 退款后营收 ≥ ${min_rev_scale:,.0f}
                - 退款率 ≤ {refund_warn:.0%}
                - 总营销费比 ≤ {mkt_cap:.0%}

                **C 止损预警（任一条触发即可进 C）：**
                - 负利润 且 退款后营收 ≥ ${stoploss_rev:,.0f}
                - 退款率 ≥ {refund_high:.0%}
                - 总营销费比 ≥ {mkt_high:.0%}

                **B 可修复（不进 A、不进 C，但存在明显短板）：**
                - 利润率未达标但接近目标且有规模；或
                - 利润率达标但退款率/营销费比超线；或
                - 小规模亏损（优先修复而非立刻止损）
                """
                    )


                colA, colB, colC = st.columns(3)

                def fmt_money(x): return f"${float(x or 0):,.0f}"
                def fmt_pct(x): return f"{float(x or 0):.2%}"

                show_cols_action = ['SPU', '第一责任位', '退款后营收', '利润额', '利润率', '退款率', '总营销费比',
                    '广告费比', '佣金比', '样品费比', '主因', '建议动作']


                with colA:
                    st.markdown("#### A ✅ 增投候选")
                    dfA = df_pl[df_pl['决策篮子'] == "A 增投候选"].copy().sort_values(['利润额', '退款后营收'], ascending=False).head(20)
                    if dfA.empty:
                        st.info("暂无 A：可以适当放宽阈值或等待数据累积。")
                    else:
                        dfA_show = dfA[show_cols_action].copy()
                        dfA_show['退款后营收'] = dfA_show['退款后营收'].apply(fmt_money)
                        dfA_show['利润额'] = dfA_show['利润额'].apply(fmt_money)
                        for c in ['利润率', '退款率', '总营销费比', '广告费比', '佣金比', '样品费比']:
                            dfA_show[c] = dfA_show[c].apply(fmt_pct)
                        st.dataframe(dfA_show, use_container_width=True, hide_index=True)

                with colB:
                    st.markdown("#### B 🛠 可修复")
                    dfB = df_pl[df_pl['决策篮子'] == "B 可修复"].copy().sort_values(['退款后营收'], ascending=False).head(20)
                    if dfB.empty:
                        st.info("暂无 B：当前SPU要么很健康，要么已经触发止损。")
                    else:
                        dfB_show = dfB[show_cols_action].copy()
                        dfB_show['退款后营收'] = dfB_show['退款后营收'].apply(fmt_money)
                        dfB_show['利润额'] = dfB_show['利润额'].apply(fmt_money)
                        for c in ['利润率', '退款率', '总营销费比', '广告费比', '佣金比', '样品费比']:
                            dfB_show[c] = dfB_show[c].apply(fmt_pct)
                        st.dataframe(dfB_show, use_container_width=True, hide_index=True)

                with colC:
                    st.markdown("#### C 🧨 止损预警")
                    dfC = df_pl[df_pl['决策篮子'] == "C 止损预警"].copy().sort_values(['利润额'], ascending=True).head(20)
                    if dfC.empty:
                        st.success("✅ 暂无 C：本周期没有明显拖累项。")
                    else:
                        dfC_show = dfC[show_cols_action].copy()
                        dfC_show['退款后营收'] = dfC_show['退款后营收'].apply(fmt_money)
                        dfC_show['利润额'] = dfC_show['利润额'].apply(fmt_money)
                        for c in ['利润率', '退款率', '总营销费比', '广告费比', '佣金比', '样品费比']:
                            dfC_show[c] = dfC_show[c].apply(fmt_pct)
                        st.dataframe(dfC_show, use_container_width=True, hide_index=True)

                # ---------- 2) 诊断图谱：波士顿 + 退款影响 + 营销构成 ----------
                st.divider()
                st.markdown("### 📍 诊断图谱（看清“为什么”）")

                # ✅ 颜色：利润率正=绿，负=红
                profit_color_scale = alt.Scale(
                    domain=["🟢 利润率为正", "🔴 利润率为负"],
                    range=["#2ecc71", "#e74c3c"]
                )

        
                 # 2.1 波士顿分析：营收 x 利润率（y固定 -100%~100%，0线；颜色=决策篮子A/B/C/D；仅TopN且仅A/C打标签；保留全量灰点）
                st.markdown("#### 波士顿分析（SPU）")

                bdf = df_pl[['SPU', '退款后营收', '利润率', '利润额', '退款率', '总营销费比', '决策篮子']].copy()
                bdf.rename(columns={'退款后营收': 'Revenue', '利润率': 'ProfitRate'}, inplace=True)

                # 显示用 clip（避免极端值把图拉爆），tooltip 用真实值
                bdf['ProfitRate_Display'] = bdf['ProfitRate'].clip(-1.0, 1.0)

                # 决策篮子压缩成更直观的显示（A/B/C/D）
                def _bucket_short(x: str) -> str:
                    x = str(x or "")
                    if x.startswith("A"):
                        return "A 增投候选"
                    if x.startswith("B"):
                        return "B 可修复"
                    if x.startswith("C"):
                        return "C 止损预警"
                    return "D 其他观察"

                bdf['Bucket'] = bdf['决策篮子'].apply(_bucket_short)

                # 颜色映射（A绿 / B橙 / C红 / D灰）
                bucket_domain = ["A 增投候选", "B 可修复", "C 止损预警", "D 其他观察"]
                bucket_range = ["#22c55e", "#f59e0b", "#ef4444", "#9ca3af"]  # green/orange/red/gray

                # Top N：默认按退款后营收排序（你已确认）
                top_n_int = int(top_n) if 'top_n' in locals() else 15
                top_df = bdf.sort_values('Revenue', ascending=False).head(top_n_int).copy()

                # 仅给 TopN 且仅 A/C 打标签，防止糊图
                label_df = top_df[top_df['Bucket'].isin(["A 增投候选", "C 止损预警"])].copy()

                # ---------- 底层：全量SPU灰点（保留结构） ----------
                base_all = alt.Chart(bdf).encode(
                    x=alt.X('Revenue:Q', title='退款后营收 ($)', scale=alt.Scale(zero=True)),
                    y=alt.Y('ProfitRate_Display:Q', title='利润率', scale=alt.Scale(domain=[-1, 1]), axis=alt.Axis(format='%')),
                    size=alt.Size('Revenue:Q', legend=None),
                    tooltip=[
                        alt.Tooltip('SPU:N'),
                        alt.Tooltip('Revenue:Q', title='退款后营收', format=',.0f'),
                        alt.Tooltip('ProfitRate:Q', title='利润率', format='.2%'),
                        alt.Tooltip('利润额:Q', title='利润额', format=',.0f'),
                        alt.Tooltip('退款率:Q', title='退款率', format='.2%'),
                        alt.Tooltip('总营销费比:Q', title='营销费比', format='.2%'),
                        alt.Tooltip('Bucket:N', title='决策篮子')
                    ]
                )

                all_points = base_all.mark_circle(opacity=0.15, color="#9ca3af")

                # ---------- 上层：A/B/C/D 决策色 ----------
                highlight_points = alt.Chart(bdf).encode(
                    x=alt.X('Revenue:Q', title='退款后营收 ($)', scale=alt.Scale(zero=True)),
                    y=alt.Y('ProfitRate_Display:Q', title='利润率', scale=alt.Scale(domain=[-1, 1]), axis=alt.Axis(format='%')),
                    size=alt.Size('Revenue:Q', legend=None),
                    color=alt.Color('Bucket:N', title='决策篮子',
                                    scale=alt.Scale(domain=bucket_domain, range=bucket_range)),
                    tooltip=[
                        alt.Tooltip('SPU:N'),
                        alt.Tooltip('Revenue:Q', title='退款后营收', format=',.0f'),
                        alt.Tooltip('ProfitRate:Q', title='利润率', format='.2%'),
                        alt.Tooltip('利润额:Q', title='利润额', format=',.0f'),
                        alt.Tooltip('退款率:Q', title='退款率', format='.2%'),
                        alt.Tooltip('总营销费比:Q', title='营销费比', format='.2%'),
                        alt.Tooltip('Bucket:N', title='决策篮子')
                    ]
                ).mark_circle(opacity=0.90)

                # 0线
                zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color="#111827", opacity=0.8).encode(y='y:Q')

                # 标签层：仅 TopN 且仅 A/C
                labels = alt.Chart(label_df).encode(
                    x=alt.X('Revenue:Q'),
                    y=alt.Y('ProfitRate_Display:Q'),
                    text=alt.Text('SPU:N')
                ).mark_text(align='left', dx=7, dy=-7, fontSize=11, color="#111827")

                # 是否显示标签（沿用你侧边栏开关 show_labels）
                if 'show_labels' in locals() and show_labels:
                    chart = (all_points + highlight_points + labels + zero_line).interactive()
                else:
                    chart = (all_points + highlight_points + zero_line).interactive()

                st.altair_chart(chart, use_container_width=True)

                # 2.2 退款影响图：退款率 x 利润率（红/绿）
                st.markdown("#### 退款影响图（Refund Impact Map）")

                rdf = df_pl[['SPU', '退款后营收', '利润率', '退款率', '决策篮子']].copy()
                rdf.rename(columns={'退款后营收': 'Revenue', '利润率': 'ProfitRate', '退款率': 'RefundRate'}, inplace=True)

                rdf['ProfitRate_Display'] = rdf['ProfitRate'].clip(-1.0, 1.0)
                rdf['ProfitSign'] = rdf['ProfitRate'].apply(lambda x: "🟢 利润率为正" if x >= 0 else "🔴 利润率为负")

                rbase = alt.Chart(rdf).encode(
                    x=alt.X('RefundRate:Q', title='退款率', scale=alt.Scale(zero=True), axis=alt.Axis(format='%')),
                    y=alt.Y('ProfitRate_Display:Q', title='利润率', scale=alt.Scale(domain=[-1, 1]), axis=alt.Axis(format='%')),
                    size=alt.Size('Revenue:Q', legend=None),
                    color=alt.Color('ProfitSign:N', title='利润状态', scale=profit_color_scale),
                    tooltip=[
                        alt.Tooltip('SPU:N'),
                        alt.Tooltip('RefundRate:Q', title='退款率', format='.2%'),
                        alt.Tooltip('ProfitRate:Q', title='利润率(真实)', format='.2%'),
                        alt.Tooltip('Revenue:Q', title='退款后营收', format=',.0f'),
                        alt.Tooltip('决策篮子:N', title='篮子')
                    ]
                )

                rdots = rbase.mark_circle(opacity=0.90)
                rzero = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color="#333").encode(y='y:Q')
                st.altair_chart((rdots + rzero).interactive(), use_container_width=True)

                
                
                
                # 2.3 营销构成（Top N SPU 堆叠占比条形图）——默认折叠 + 支持指定 SPU
                with st.expander("#### 营销构成（Top SPU）", expanded=False):
                    view_mode = st.radio("视图", ["Top N（默认）", "指定SPU"], horizontal=True)

                    # 预备数据
                    sdf_all = df_pl[['SPU', '退款后营收', '总广告投放费', '总达人佣金', '总样品费']].copy()
                    sdf_all['退款后营收'] = pd.to_numeric(sdf_all['退款后营收'], errors='coerce').fillna(0.0)
                    sdf_all['总广告投放费'] = pd.to_numeric(sdf_all['总广告投放费'], errors='coerce').fillna(0.0)
                    sdf_all['总达人佣金'] = pd.to_numeric(sdf_all['总达人佣金'], errors='coerce').fillna(0.0)
                    sdf_all['总样品费'] = pd.to_numeric(sdf_all['总样品费'], errors='coerce').fillna(0.0)

                    if view_mode == "Top N（默认）":
                        sdf = sdf_all.sort_values('退款后营收', ascending=False).head(int(top_n)).copy()
                    else:
                        spu_options = sorted([x for x in sdf_all['SPU'].dropna().astype(str).unique().tolist() if x.strip() != ""])
                        picked_spu = st.multiselect("选择要看的 SPU（可多选）", options=spu_options, default=[])
                        if not picked_spu:
                            st.info("请选择至少 1 个 SPU 才会展示营销构成。")
                            sdf = pd.DataFrame(columns=sdf_all.columns)
                        else:
                            sdf = sdf_all[sdf_all['SPU'].astype(str).isin([str(x) for x in picked_spu])].copy()
                            # 指定SPU视图下，按营收排序更好读
                            sdf = sdf.sort_values('退款后营收', ascending=False)

                    if sdf.empty:
                        pass
                    else:
                        denom = sdf['退款后营收'].replace(0, np.nan)
                        sdf['广告费比'] = (sdf['总广告投放费'] / denom).fillna(0.0)
                        sdf['佣金比'] = (sdf['总达人佣金'] / denom).fillna(0.0)
                        sdf['样品费比'] = (sdf['总样品费'] / denom).fillna(0.0)

                        long_rows = []
                        for _, r in sdf.iterrows():
                            spu = str(r['SPU'])
                            long_rows.append({'SPU': spu, 'Type': '广告费比', 'Rate': float(r['广告费比'] or 0)})
                            long_rows.append({'SPU': spu, 'Type': '佣金比', 'Rate': float(r['佣金比'] or 0)})
                            long_rows.append({'SPU': spu, 'Type': '样品费比', 'Rate': float(r['样品费比'] or 0)})

                        mkt_long = pd.DataFrame(long_rows)

                        mkt_chart = alt.Chart(mkt_long).mark_bar().encode(
                            y=alt.Y('SPU:N', sort='-x', title='SPU'),
                            x=alt.X('Rate:Q', title='占退款后营收比例', axis=alt.Axis(format='%')),
                            color=alt.Color('Type:N', title='营销构成'),
                            tooltip=[alt.Tooltip('SPU:N'), alt.Tooltip('Type:N'), alt.Tooltip('Rate:Q', format='.2%')]
                        )
                        st.altair_chart(mkt_chart, use_container_width=True)

                    # =========================
                    # ✅ 新增：达人结构（轻联动，不改A/B/C）
                    # 1) 默认折叠
                    # 2) 默认展示：Top10营收SPU（按退款后营收）
                    # 3) 支持指定SPU（默认不选）
                    # =========================
                    with st.expander("🤝 达人结构（轻联动：集中度 & 矩阵健康）", expanded=False):

                        if creator_stats is None and spu_aff_perf is None:
                            st.info("💡 未检测到达人数据（缺 affiliate 或 transaction 表 / 或字段不足），无法展示达人结构。")
                        else:
                            view_mode2 = st.radio("视图", ["Top10 营收SPU（默认）", "指定SPU"], horizontal=True)

                            # Top10：按退款后营收（你的系统核心口径）
                            top10_spu_list = df_pl.sort_values('退款后营收', ascending=False).head(10)['SPU'].astype(str).tolist()

                            if view_mode2 == "Top10 营收SPU（默认）":
                                chosen_spus = top10_spu_list
                            else:
                                spu_options_all = sorted([x for x in df_pl['SPU'].dropna().astype(str).unique().tolist() if x.strip() != ""])
                                chosen_spus = st.multiselect("选择要看的 SPU（可多选）", options=spu_options_all, default=[])  # ✅ 默认不选

                            if not chosen_spus:
                                st.info("请选择至少 1 个 SPU 才会展示达人结构。")
                            else:
                                # —— 1) 汇总表：SPU 的达人结构核心指标 ——
                                show = pd.DataFrame({'SPU': chosen_spus})

                                # 拼接 Transaction 汇总（Affiliate_Rate / Videos / OutputPerVideo）
                                if spu_aff_perf is not None and not spu_aff_perf.empty:
                                    keep_cols = ['SPU', 'Affiliate_Rate', 'Videos', 'OutputPerVideo', 'Affiliate_GMV']
                                    keep_cols = [c for c in keep_cols if c in spu_aff_perf.columns]
                                    show = show.merge(spu_aff_perf[keep_cols], on='SPU', how='left')

                                # 拼接 affiliate 计算的集中度（Creators / Top1Share / Top3Share / HealthTag）
                                if creator_stats is not None and not creator_stats.empty:
                                    show = show.merge(creator_stats, on='SPU', how='left')

                                # 再拼接 SPU 财务关键字段（更利于周会理解）
                                fin_cols = ['SPU', '退款后营收', '利润额', '利润率', '决策篮子']
                                fin_tmp = df_pl[fin_cols].copy()
                                show = show.merge(fin_tmp, on='SPU', how='left')

                                # 格式化
                                def _m(x): return f"${float(x or 0):,.0f}"
                                def _p(x): return f"{float(x or 0):.2%}"

                                if '退款后营收' in show.columns: show['退款后营收'] = show['退款后营收'].apply(_m)
                                if '利润额' in show.columns: show['利润额'] = show['利润额'].apply(_m)
                                if '利润率' in show.columns: show['利润率'] = show['利润率'].apply(_p)

                                if 'Affiliate_Rate' in show.columns: show['Affiliate_Rate'] = show['Affiliate_Rate'].apply(_p)
                                if 'OutputPerVideo' in show.columns:
                                    show['OutputPerVideo'] = show['OutputPerVideo'].apply(lambda x: f"${float(x or 0):,.0f}")
                                if 'Top1Share' in show.columns: show['Top1Share'] = show['Top1Share'].apply(_p)
                                if 'Top3Share' in show.columns: show['Top3Share'] = show['Top3Share'].apply(_p)

                                # 字段顺序（缺则自动跳过）
                                col_order = [
                                    'SPU', '决策篮子',
                                    '退款后营收', '利润额', '利润率',
                                    'Affiliate_Rate', 'Videos', 'OutputPerVideo',
                                    'Creators', 'Top1Share', 'Top3Share', 'CreatorHealth'
                                ]
                                col_order = [c for c in col_order if c in show.columns]

                                st.dataframe(show[col_order], use_container_width=True, hide_index=True)

                                # —— 2) 可视化：Top3 + Others 的堆叠占比（基于 affiliate GMV）——
                                if creator_long_for_viz is not None and not creator_long_for_viz.empty:
                                    viz = creator_long_for_viz[creator_long_for_viz['SPU'].astype(str).isin([str(x) for x in chosen_spus])].copy()
                                    if not viz.empty:
                                        chart = alt.Chart(viz).mark_bar().encode(
                                            y=alt.Y('SPU:N', sort='-x', title='SPU'),
                                            x=alt.X('Share:Q', title='达人GMV结构占比（Top3 + Others）', axis=alt.Axis(format='%')),
                                            color=alt.Color('Group:N', title='结构'),
                                            tooltip=[alt.Tooltip('SPU:N'), alt.Tooltip('Group:N'), alt.Tooltip('Share:Q', format='.2%')]
                                        )
                                        st.altair_chart(chart, use_container_width=True)
                                    else:
                                        st.caption("说明：当前选择的 SPU 在 affiliate 表中无可用达人GMV记录，因此无法绘制结构图。")

                                st.caption("说明：本模块为轻联动，仅提供内容侧风险提示，不改变 A/B/C 规则。")

                
                
                # ---------- 3) SPU 分析表（格式化展示）放到最后 ----------
                st.divider()
                st.subheader("SPU 分析表（格式化展示）")

                # 过滤器：按篮子 + 搜索SPU
                basket_options = ["全部", "A 增投候选", "B 可修复", "C 止损预警", "D 其他观察"]
                flt1, flt2 = st.columns([0.35, 0.65])
                with flt1:
                    chosen_bucket = st.selectbox("按决策篮子筛选", basket_options, index=0)
                with flt2:
                    keyword = st.text_input("搜索 SPU（支持模糊）", value="")

                # 把篮子列合并到格式化表中（不改原 df_spu_out 的结构，做一个展示副本）
                df_spu_show = df_spu_out.copy()
                if 'SPU' in df_spu_show.columns:
                    bucket_map = dict(zip(df_pl['SPU'], df_pl['决策篮子']))
                    cause_map = dict(zip(df_pl['SPU'], df_pl['主因']))
                    action_map = dict(zip(df_pl['SPU'], df_pl['建议动作']))
                    df_spu_show.insert(0, "决策篮子", df_spu_show['SPU'].map(bucket_map).fillna("—"))
                    df_spu_show.insert(1, "主因", df_spu_show['SPU'].map(cause_map).fillna("—"))
                    df_spu_show.insert(2, "建议动作", df_spu_show['SPU'].map(action_map).fillna("—"))
                # =========================
                # ✅ SPU 分析表：新增达人联动列（轻联动）
                # =========================
                if 'SPU' in df_spu_show.columns:
                    # 1) Transaction 汇总列
                    if spu_aff_perf is not None and not spu_aff_perf.empty:
                        perf_map = spu_aff_perf.set_index('SPU').to_dict(orient='index')
                        df_spu_show['达人GMV占比'] = df_spu_show['SPU'].map(lambda x: perf_map.get(x, {}).get('Affiliate_Rate', np.nan))
                        df_spu_show['达人视频数'] = df_spu_show['SPU'].map(lambda x: perf_map.get(x, {}).get('Videos', np.nan))
                        df_spu_show['单视频产出'] = df_spu_show['SPU'].map(lambda x: perf_map.get(x, {}).get('OutputPerVideo', np.nan))
                    else:
                        df_spu_show['达人GMV占比'] = np.nan
                        df_spu_show['达人视频数'] = np.nan
                        df_spu_show['单视频产出'] = np.nan

                    # 2) affiliate 集中度列
                    if creator_stats is not None and not creator_stats.empty:
                        cs_map = creator_stats.set_index('SPU').to_dict(orient='index')
                        df_spu_show['出单达人数'] = df_spu_show['SPU'].map(lambda x: cs_map.get(x, {}).get('Creators', np.nan))
                        df_spu_show['Top1达人占比'] = df_spu_show['SPU'].map(lambda x: cs_map.get(x, {}).get('Top1Share', np.nan))
                        df_spu_show['Top3达人占比'] = df_spu_show['SPU'].map(lambda x: cs_map.get(x, {}).get('Top3Share', np.nan))
                        df_spu_show['内容健康度'] = df_spu_show['SPU'].map(lambda x: cs_map.get(x, {}).get('CreatorHealth', "—"))
                    else:
                        df_spu_show['出单达人数'] = np.nan
                        df_spu_show['Top1达人占比'] = np.nan
                        df_spu_show['Top3达人占比'] = np.nan
                        df_spu_show['内容健康度'] = "—"

                    # 3) 格式化显示（避免全是小数）
                    for c in ['达人GMV占比', 'Top1达人占比', 'Top3达人占比']:
                        if c in df_spu_show.columns:
                            df_spu_show[c] = pd.to_numeric(df_spu_show[c], errors='coerce').fillna(np.nan).apply(
                                lambda x: f"{x:.2%}" if pd.notna(x) else "—"
                            )
                    if '单视频产出' in df_spu_show.columns:
                        df_spu_show['单视频产出'] = pd.to_numeric(df_spu_show['单视频产出'], errors='coerce').fillna(np.nan).apply(
                            lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
                        )
                    if '达人视频数' in df_spu_show.columns:
                        df_spu_show['达人视频数'] = pd.to_numeric(df_spu_show['达人视频数'], errors='coerce').fillna(np.nan).apply(
                            lambda x: f"{int(x)}" if pd.notna(x) else "—"
                        )
                    if '出单达人数' in df_spu_show.columns:
                        df_spu_show['出单达人数'] = pd.to_numeric(df_spu_show['出单达人数'], errors='coerce').fillna(np.nan).apply(
                            lambda x: f"{int(x)}" if pd.notna(x) else "—"
                        )

                # 应用筛选
                if chosen_bucket != "全部":
                    df_spu_show = df_spu_show[df_spu_show.get("决策篮子", "") == chosen_bucket].copy()

                if keyword.strip():
                    kw = keyword.strip().upper()
                    if 'SPU' in df_spu_show.columns:
                        df_spu_show = df_spu_show[df_spu_show['SPU'].astype(str).str.upper().str.contains(kw, na=False)].copy()

                st.dataframe(df_spu_show, use_container_width=True)


        with tab4:
            # ================= 三级类目分析 =================
            st.subheader("📊 三级类目经营分析")
            
            df_category_out_local = out.get('df_category_out')
            df_category_raw_local = out.get('df_category_raw')
            
            if df_category_out_local is None or df_category_out_local.empty:
                st.info("💡 暂无类目分析数据。请确保采购成本表中包含'三级类目'列。")
            else:
                # 显示类目汇总KPI
                total_categories = df_category_raw_local['三级类目'].nunique() if '三级类目' in df_category_raw_local.columns else 0
                total_revenue = df_category_raw_local['退款后营收'].sum() if '退款后营收' in df_category_raw_local.columns else 0
                total_profit = df_category_raw_local['利润额'].sum() if '利润额' in df_category_raw_local.columns else 0
                
                k1, k2, k3 = st.columns(3)
                k1.metric("类目数量", f"{total_categories}")
                k2.metric("类目总营收", f"${total_revenue:,.0f}")
                k3.metric("类目总利润", f"${total_profit:,.0f}")
                
                st.divider()
                
                # ============== 新增：类目维度YOY同比 ==============
                if '退款后营收_YOY%' in df_category_raw_local.columns:
                    with st.expander("📊 类目维度 YOY 同比（vs 2025年同期）", expanded=True):
                        # YOY增长最快Top10类目
                        st.markdown("#### 🚀 YOY 增长最快 Top 10 类目")
                        df_cat_yoy = df_category_raw_local[df_category_raw_local['2025年同期退款后营收'] > 0].copy() if '2025年同期退款后营收' in df_category_raw_local.columns else df_category_raw_local.copy()
                        
                        if not df_cat_yoy.empty and '退款后营收_YOY%' in df_cat_yoy.columns:
                            df_cat_yoy_growth = df_cat_yoy.sort_values('退款后营收_YOY%', ascending=False).head(10)
                            show_cat_yoy = df_cat_yoy_growth[['三级类目', '退款后营收', '2025年同期退款后营收', '退款后营收_YOY%']].copy()
                            show_cat_yoy['退款后营收'] = show_cat_yoy['退款后营收'].apply(lambda x: f"${float(x or 0):,.0f}")
                            show_cat_yoy['2025年同期退款后营收'] = show_cat_yoy['2025年同期退款后营收'].apply(lambda x: f"${float(x or 0):,.0f}")
                            show_cat_yoy['退款后营收_YOY%'] = show_cat_yoy['退款后营收_YOY%'].apply(lambda x: f"{float(x or 0):+.1f}%" if pd.notna(x) else "N/A")
                            st.dataframe(show_cat_yoy, use_container_width=True, hide_index=True)
                        else:
                            st.info("暂无类目YOY对比数据（需要2025年采购成本表中的类目数据）")
                
                st.divider()
                
                # 显示类目排名
                st.markdown("### 🏆 类目营收排名（Top 20）")
                if '三级类目' in df_category_raw_local.columns and '退款后营收' in df_category_raw_local.columns:
                    top_cats = df_category_raw_local.nlargest(20, '退款后营收')[['三级类目', '销量', '退款后营收', '利润额', '利润率', '总营销费比']].copy()
                    
                    # 格式化显示
                    for col in ['退款后营收', '利润额']:
                        if col in top_cats.columns:
                            top_cats[col] = top_cats[col].apply(lambda x: f"${float(x or 0):,.0f}")
                    for col in ['利润率', '总营销费比']:
                        if col in top_cats.columns:
                            top_cats[col] = top_cats[col].apply(lambda x: f"{float(x or 0):.2%}")
                    
                    st.dataframe(top_cats, use_container_width=True, hide_index=True)
                
                st.divider()
                
                # 显示完整类目数据表
                st.markdown("### 📋 类目详细数据表")
                st.dataframe(df_category_out_local, use_container_width=True)
                
                # 提供下载
                csv_category = df_category_out_local.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="⬇️ 下载类目分析数据 CSV",
                    data=csv_category,
                    file_name="category_analysis.csv",
                    mime="text/csv"
                )

        with tab5:
            st.subheader("📄 SKU 明细表（格式化展示）")
            st.dataframe(df_sku_out, use_container_width=True)

        with tab6:
                        # ================= 新增：广告拖累 SPU Top10（回流SPU视角） =================

            def _scatter_bubble(df, x, y, size="Cost", color="Video_Status", tooltip_cols=None, title=""):
                if df is None or df.empty or x not in df.columns or y not in df.columns:
                    st.info("暂无足够数据生成图表")
                    return

                tooltip_cols = tooltip_cols or ["SPU", "Video title", "Cost", "Revenue", "Orders", "ROI", "Hook_2s", "CTR", "CVR", "Video_Status"]
                tooltip_cols = [c for c in tooltip_cols if c in df.columns]

                chart = (
                    alt.Chart(df)
                    .mark_circle(opacity=0.75)
                    .encode(
                        x=alt.X(x, axis=alt.Axis(format=".0%") if "Hook" in x or x in {"CTR","CVR","Hook_2s","Hold_6s"} else alt.Axis()),
                        y=alt.Y(y, axis=alt.Axis(format=".0%") if "Hook" in y or y in {"CTR","CVR","Hook_2s","Hold_6s"} else alt.Axis()),
                        size=alt.Size(size, legend=None),
                        color=alt.Color(color, legend=alt.Legend(title="Status")),
                        tooltip=tooltip_cols
                    )
                    .properties(height=340, title=title)
                )
                st.altair_chart(chart, use_container_width=True)

            def render_video_quality_v3(df_spu_video_action: pd.DataFrame, observe_cost: float = 50.0):
                st.subheader("层级二｜素材（Video）质量透视 V3")

                if df_spu_video_action is None or df_spu_video_action.empty:
                    st.warning("暂无 SPU-Video 数据（请检查广告明细是否包含 Video title + PID + Cost 等字段）")
                    return

                # 统一视频列名（兼容）
                video_col = "Video title" if "Video title" in df_spu_video_action.columns else "Video"

                t1, t2, t3, t4 = st.tabs(["① 素材总览", "② SPU 素材拆解", "③ 跨SPU资产库", "④ 一键导出工单"])

                # ① 总览
                with t1:
                    df = df_spu_video_action.copy()
                    total_videos = df[video_col].nunique()
                    total_cost = df["Cost"].sum()
                    total_rev = df["Revenue"].sum() if "Revenue" in df.columns else np.nan
                    total_roi = safe_div(total_rev, total_cost) if pd.notna(total_rev) else np.nan

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("视频数", f"{total_videos}")
                    c2.metric("总花费", f"{total_cost:,.2f}")
                    c3.metric("总营收", f"{(total_rev if pd.notna(total_rev) else 0):,.2f}")
                    c4.metric("整体 ROI", f"{(total_roi if pd.notna(total_roi) else 0):.2f}")

                    st.write("Status 分布（数量）")
                    st.dataframe(df["Video_Status"].value_counts(dropna=False).rename("count").reset_index().rename(columns={"index":"Video_Status"}), use_container_width=True)

                # ② SPU 素材拆解
                with t2:
                    spus = sorted(df_spu_video_action["SPU"].dropna().unique().tolist())
                    default_spu = spus[0] if spus else "(unmapped)"
                    spu_sel = st.selectbox("选择 SPU", spus, index=0 if default_spu in spus else 0)

                    colA, colB, colC = st.columns([1,1,1])
                    with colA:
                        min_cost = st.number_input("最低 Cost（过滤观察期）", min_value=0.0, value=float(observe_cost), step=10.0)
                    with colB:
                        status_filter = st.multiselect("筛选 Status", options=sorted(df_spu_video_action["Video_Status"].unique().tolist()),
                                                    default=sorted(df_spu_video_action["Video_Status"].unique().tolist()))
                    with colC:
                        st.caption("图表使用 % 展示（内部为 0~1）")

                    d = df_spu_video_action[(df_spu_video_action["SPU"] == spu_sel) & (df_spu_video_action["Cost"].fillna(0) >= min_cost)]
                    if status_filter:
                        d = d[d["Video_Status"].isin(status_filter)]

                    st.markdown("**图A：Hook × CTR（找前3秒 vs 点进来）**")
                    _scatter_bubble(d, "Hook_2s", "CTR", title="Hook(2s) vs CTR")

                    st.markdown("**图B：CTR × CVR（找标题党/承接问题）**")
                    _scatter_bubble(d, "CTR", "CVR", title="CTR vs CVR")

                    st.markdown("**动作清单（可直接派活）**")
                    show_cols = [c for c in ["SPU", video_col, "Cost", "Revenue", "Orders", "ROI", "Hook_2s", "Hold_6s", "CTR", "CVR", "Video_Status", "Reason", "Next_Action", "Evidence"] if c in d.columns]
                    st.dataframe(d[show_cols].sort_values(["Video_Status","Cost"], ascending=[True, False]), use_container_width=True, height=520)

                    csv = d[show_cols].to_csv(index=False).encode("utf-8-sig")
                    st.download_button("下载该 SPU 的动作清单 CSV", data=csv, file_name=f"spu_video_action_{spu_sel}.csv", mime="text/csv")

                # ③ 资产库
                with t3:
                    df = df_spu_video_action.copy()
                    df_big = df[df["Cost"].fillna(0) >= observe_cost]

                    st.markdown("**可复用素材 Top（优先 Hook+CTR+ROI）**")
                    top = df_big.copy()
                    top["Score"] = top["Hook_2s"].fillna(0)*0.4 + top["CTR"].fillna(0)*0.4 + (top["ROI"].fillna(0)/3.0)*0.2
                    top = top.sort_values("Score", ascending=False).head(30)
                    cols = [c for c in ["SPU", video_col, "Cost", "ROI", "Hook_2s", "CTR", "CVR", "Video_Status", "Next_Action"] if c in top.columns]
                    st.dataframe(top[cols], use_container_width=True, height=420)

                    st.markdown("**高风险素材（高CTR低CVR）**")
                    risk = df_big[(df_big["CTR"].fillna(0) >= df_big["CTR"].quantile(0.7)) & (df_big["CVR"].fillna(0) <= df_big["CVR"].quantile(0.3))]
                    risk = risk.sort_values("Cost", ascending=False).head(30)
                    cols2 = [c for c in ["SPU", video_col, "Cost", "ROI", "CTR", "CVR", "Reason", "Next_Action"] if c in risk.columns]
                    st.dataframe(risk[cols2], use_container_width=True, height=420)

                # ④ 工单导出
                with t4:
                    df = df_spu_video_action.copy()
                    video_col = "Video title" if "Video title" in df.columns else "Video"

                    ticket_cols = [c for c in ["SPU", video_col, "Video_Status", "Reason", "Next_Action", "Evidence", "Cost", "Revenue", "Orders", "ROI", "Hook_2s", "Hold_6s", "CTR", "CVR"] if c in df.columns]
                    tickets = df[ticket_cols].copy()
                    tickets["Generated_At"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

                    st.dataframe(tickets.sort_values(["Video_Status","Cost"], ascending=[True, False]), use_container_width=True, height=520)
                    csv = tickets.to_csv(index=False).encode("utf-8-sig")
                    st.download_button("下载全店素材工单 CSV", data=csv, file_name="video_tickets_all.csv", mime="text/csv")
       

                        
            st.subheader("🧲 广告拖累诊断（回流SPU视角）")

            # 观察期过滤阈值：默认 COST_OBSERVE（若你没定义该常量，则默认 0）
            default_min_cost = float(COST_OBSERVE) if 'COST_OBSERVE' in globals() else 0.0
            min_cost_filter = st.number_input(
                "最低花费门槛（过滤观察期噪音，填 0 表示不过滤）",
                value=float(default_min_cost),
                step=10.0,
                format="%.0f"
            )
            # ====== [新增] SPU -> 利润率 映射（用于广告页联动展示）======
            spu_pr_map = {}

            # 尝试从全局变量里找“SPU经营表”（避免你变量名不叫 df_spu_raw 导致报错）
            _spu_df = (
                globals().get("df_spu_raw")
                or globals().get("df_spu")
                or globals().get("df_spu_kpi")
                or globals().get("df_spu_fin")
            )

            def _pick_col(df, candidates):
                for c in candidates:
                    if c in df.columns:
                        return c
                return None

            def _fmt_pct_safe(x):
                try:
                    return f"{float(x):.2%}" if pd.notna(x) else "—"
                except:
                    return "—"

            if _spu_df is not None and isinstance(_spu_df, pd.DataFrame) and (not _spu_df.empty):
                spu_col = _pick_col(_spu_df, ["SPU", "spu", "Spu"])
                pr_col  = _pick_col(_spu_df, ["利润率", "profit_rate", "Profit_Rate", "ProfitRate", "profitRate"])

                if (spu_col is not None) and (pr_col is not None):
                    _tmp = _spu_df[[spu_col, pr_col]].copy()
                    _tmp[spu_col] = _tmp[spu_col].astype(str).str.strip()
                    _tmp[pr_col] = pd.to_numeric(_tmp[pr_col], errors="coerce")
                    spu_pr_map = dict(zip(_tmp[spu_col], _tmp[pr_col]))

            # PID行里 SPU 可能是 "A, B, C"，做成 "A:xx% | B:yy%" 的展示
            def _spu_profit_rate_text(spu_str):
                spus = [s.strip() for s in str(spu_str).split(",") if s.strip()]
                if not spus:
                    return "—"
                parts = []
                for s in spus:
                    pr = spu_pr_map.get(s, np.nan)
                    parts.append(f"{s}:{_fmt_pct_safe(pr)}")
                return " | ".join(parts)

            # 单个 SPU 的利润率（拖累SPU榜单每行一个SPU时用）
            def _spu_profit_rate_single(spu):
                pr = spu_pr_map.get(str(spu).strip(), np.nan)
                return _fmt_pct_safe(pr)

            tab1, tab2 = st.tabs(["亏损PID清单（全量）", "广告拖累SPU Top10（原口径）"])

            with tab2:
                loss_pid_df, loss_note = build_ads_loss_pid_table(df_prod_ads, min_cost=min_cost_filter)
                if loss_pid_df is None or loss_pid_df.empty:
                    st.info(f"💡 暂无亏损PID或数据不足：{loss_note}")
                else:
                    show = loss_pid_df.copy()
                    if 'Cost' in show.columns:
                        show['Cost'] = show['Cost'].apply(lambda x: f"${float(x or 0):,.0f}")
                    if 'Revenue' in show.columns:
                        show['Revenue'] = show['Revenue'].apply(lambda x: f"${float(x or 0):,.0f}")
                    if 'ROI' in show.columns:
                        show['ROI'] = show['ROI'].apply(lambda x: f"{float(x or 0):.2f}")
                    st.dataframe(show, use_container_width=True, hide_index=True)
                    st.caption(loss_note)

            with tab3:
                drag_df, drag_note = build_ads_spu_drag_table(df_prod_ads, topn=10)
                if drag_df is None or drag_df.empty:
                    st.info(f"💡 暂无法生成：{drag_note}")
                else:
                    show = drag_df.copy()
                                        # [新增] 序号列
                    show.insert(0, "序号", range(1, len(show) + 1))
                    # [新增] SPU 利润率（这里每行一个SPU，直接映射）
                    if 'SPU' in show.columns:
                        show.insert(2, "利润率", show['SPU'].map(lambda x: spu_pr_map.get(str(x).strip(), np.nan)).apply(_fmt_pct_safe))
                    else:
                        show.insert(2, "利润率", "—")
                    show['Cost'] = show['Cost'].apply(lambda x: f"${float(x or 0):,.0f}")
                    show['Revenue'] = show['Revenue'].apply(lambda x: f"${float(x or 0):,.0f}")
                    show['ROAS'] = show['ROAS'].apply(lambda x: f"{float(x or 0):.2f}")
                    st.dataframe(show, use_container_width=True, hide_index=True)
                    st.caption(drag_note)

                

            st.markdown("### 📺 广告深度诊断（V2：两层诊断）")
            if isinstance(ads_meta, dict) and ads_meta.get("error"):
                st.error(f"广告诊断不可用：{ads_meta.get('error')}")
            elif df_prod_ads is None or df_prod_ads.empty:
                st.info("💡 未上传广告表或广告表字段不足，暂无法进行广告诊断。")
            else:
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

                with st.expander("📈 展开查看：ROI × CPA 气泡图（进阶）", expanded=False):
                    c_chart = alt.Chart(df_prod_ads).mark_circle().encode(
                        x=alt.X('CPA:Q', title='CPA'),
                        y=alt.Y('ROI:Q', title='ROI (ROAS)'),
                        size=alt.Size('Cost:Q', title='Cost'),
                        color=alt.Color('Status:N', title='Status'),
                        tooltip=['Product ID', 'SPU', 'Status', 'Diagnosis', 'Cost', 'ROI', 'CPA', 'CPA_Line', 'CPA_Line_Source']
                    ).interactive()
                    st.altair_chart(c_chart, use_container_width=True)

                df_show = df_prod_ads.copy()
                for c in ['ROI', 'CPA', 'CPM', 'CTR', 'CVR', 'CPA_Line']:
                    if c in df_show.columns:
                        df_show[c] = df_show[c].astype(float).round(2)
                                            # [新增] 调整列顺序：SPU 放到 Product ID 左侧
                    cols = list(df_show.columns)
                    if ("SPU" in cols) and ("Product ID" in cols):
                        new_order = ["SPU", "Product ID"] + [c for c in cols if c not in ["SPU", "Product ID"]]
                        df_show = df_show[new_order]

                st.dataframe(df_show.sort_values('Cost', ascending=False), use_container_width=True)

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
                # ================= 层级二：素材（Video）质量透视（安全兜底版） =================

                OBSERVE_COST = 50.0  # 或换成 COST_OBSERVE

                def _get_global_df(*names):
                    for n in names:
                        v = globals().get(n)
                        if isinstance(v, pd.DataFrame):
                            return v
                    return None

                # 1) 优先：session_state 里的广告明细
                df_ads_detail = st.session_state.get("df_ads_detail")

                # 2) 如果没有广告明细：直接用你已经识别到的广告表 dfs['ads']
                if df_ads_detail is None or (hasattr(df_ads_detail, "empty") and df_ads_detail.empty):
                    df_ads_detail = dfs.get("ads")  # ✅ 关键：fallback

                # 3) 统一列名（确保 build_spu_video_table 识别 Cost/Revenue/...）
                df_ads_detail = normalize_ads_for_spu_video(df_ads_detail)

                # 4) 自动构建 PID -> SPU 映射（不再依赖 globals 碰运气）
                pid_to_spu = build_pid_to_spu_map(dfs.get("mapping"), dfs.get("spu_sku"))

                # Debug（建议先开着）
                with st.expander("🔍 Debug：层级二取数检查", expanded=False):
                    st.write("df_ads_detail 是否为空：", df_ads_detail is None or (hasattr(df_ads_detail, "empty") and df_ads_detail.empty))
                    if isinstance(df_ads_detail, pd.DataFrame):
                        st.write("shape:", df_ads_detail.shape)
                        st.write("columns:", list(df_ads_detail.columns)[:50])
                        st.write("pid_to_spu 样本(前5):", list(pid_to_spu.items())[:5])
                        st.dataframe(df_ads_detail.head(5), use_container_width=True)

                if df_ads_detail is None or df_ads_detail.empty:
                    st.info("💡 未检测到广告数据：请检查是否已上传广告表（ads）或广告明细（ads_detail）。")
                else:
                    df_spu_video = build_spu_video_table(df_ads_detail, pid_to_spu, observe_cost=OBSERVE_COST)
                    df_spu_video_action = build_spu_video_action_view_v3(df_spu_video, observe_cost=OBSERVE_COST)
                    render_video_quality_v3(df_spu_video_action, observe_cost=OBSERVE_COST)

            
        with tab7:
            st.markdown("### 🤝 达人合作分析（V2）")
            if creator_data.get('commission_source_note'):
                st.info(creator_data['commission_source_note'])

            overall = creator_data.get('overall')
            if overall:
                oc1, oc2, oc3, oc4 = st.columns(4)
                oc1.metric("达人GMV（Transaction）", f"${overall['Affiliate_GMV']:,.0f}")
                oc2.metric("达人GMV占比（分母=退款前GMV）", f"{overall['Affiliate_Share']:.2%}")
                oc3.metric("上线视频数（Videos）", f"{overall['Videos']:,.0f}")
                oc4.metric("自建联视频效率（GMV/Video）", f"${overall['Efficiency']:,.2f}")
            else:
                st.warning("💡 未上传 Transaction 表：无法计算达人GMV占比、上线视频数、视频效率。")
            

            
            # ================= 新增：达人结构风险 SPU Top10（回流SPU视角） =================
            st.subheader("⚠️ 达人结构风险 SPU Top10（集中度/矩阵薄/效率）")

            # 复用：从 affiliate 表算集中度（与你 Tab2 口径一致）
            creator_stats_local = None
            try:
                df_aff_local = dfs.get('affiliate')
                df_spu_sku_local = dfs.get('spu_sku')
                if df_aff_local is not None and df_spu_sku_local is not None and (not df_aff_local.empty) and (not df_spu_sku_local.empty):
                    c_creator = COLUMN_CONFIG['affiliate']['creator']
                    c_gmv = COLUMN_CONFIG['affiliate']['gmv']
                    c_sku = COLUMN_CONFIG['affiliate']['sku']
                    if c_creator in df_aff_local.columns and c_gmv in df_aff_local.columns and c_sku in df_aff_local.columns:
                        tmp = df_aff_local.copy()
                        tmp['Creator'] = tmp[c_creator].astype(str).str.strip()
                        tmp['SKU_Clean'] = clean_text(tmp, c_sku)
                        tmp['GMV_Val'] = tmp[c_gmv].apply(clean_money)
                        sku_to_spu_dict = build_sku_to_spu_dict(df_spu_sku_local)
                        tmp['SPU'] = tmp['SKU_Clean'].map(sku_to_spu_dict).fillna(tmp['SKU_Clean'])
                        tmp = tmp[(tmp['SPU'].astype(str).str.strip() != "") & (tmp['Creator'].astype(str).str.strip() != "")].copy()

                        g = tmp.groupby(['SPU', 'Creator'], as_index=False)['GMV_Val'].sum()
                        g = g[g['GMV_Val'] > 0].copy()

                        rows = []
                        for spu, sub in g.groupby('SPU'):
                            sub = sub.sort_values('GMV_Val', ascending=False).copy()
                            total = float(sub['GMV_Val'].sum() or 0)
                            creators = int(sub['Creator'].nunique())
                            top1 = float(sub.iloc[0]['GMV_Val']) if len(sub) >= 1 else 0.0
                            top3 = float(sub.head(3)['GMV_Val'].sum()) if len(sub) >= 3 else float(sub['GMV_Val'].sum())
                            top1_share = (top1 / total) if total > 0 else 0.0
                            top3_share = (top3 / total) if total > 0 else 0.0
                            if top1_share >= 0.70:
                                tag = "🔴 高度依赖"
                            elif top1_share >= 0.45:
                                tag = "🟠 中度集中"
                            else:
                                tag = "🟢 矩阵健康"
                            rows.append({'SPU': spu, 'Creators': creators, 'Top1Share': top1_share, 'Top3Share': top3_share, 'CreatorHealth': tag})

                        creator_stats_local = pd.DataFrame(rows) if rows else None
            except:
                creator_stats_local = None

            risk_df, risk_note = build_creator_risk_table(creator_data.get('spu_perf'), creator_stats_local, topn=10)
            if risk_df is None:
                st.info(f"💡 暂无法生成：{risk_note}")
            else:
                show = risk_df.copy()
                # 格式化
                show['Affiliate_Rate'] = show['Affiliate_Rate'].apply(lambda x: f"{float(x or 0):.2%}")
                show['OutputPerVideo'] = show['OutputPerVideo'].apply(lambda x: f"${float(x or 0):,.0f}")
                show['Top1Share'] = show['Top1Share'].apply(lambda x: f"{float(x or 0):.2%}")
                show['Top3Share'] = show['Top3Share'].apply(lambda x: f"{float(x or 0):.2%}")
                # 展示列
                cols = ['SPU', 'RiskScore', 'RiskReason', '建议动作', 'Affiliate_Rate', 'Videos', 'OutputPerVideo',
                        'Creators', 'Top1Share', 'Top3Share', 'CreatorHealth']
                cols = [c for c in cols if c in show.columns]
                st.dataframe(show[cols], use_container_width=True, hide_index=True)
                if risk_note:
                    st.caption(risk_note)

            st.divider()
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

        with tab8:
            st.markdown("### 🧠 AI 操盘手（任务清单｜人执行）")

            # 任务清单需要 df_pl（SPU决策表）
            # df_pl 在 tab2 内生成，但 with 不形成作用域，仍可在 tab6 使用；若为空则提示
            try:
                df_pl_for_task = df_pl  # 来自 tab2
            except:
                df_pl_for_task = None

            if df_pl_for_task is None or df_pl_for_task.empty:
                st.info("💡 暂无任务清单：请先确保 SPU 分析页已生成 A/B/C 决策结果（或本周期无SPU数据）。")
            else:
                tasks = build_ai_tasks_from_spu(df_pl_for_task, target_profit_rate=target_profit_rate, topn=200)
                if tasks is None or tasks.empty:
                    st.info("💡 当前无 A/B/C 任务（可能全部在观察或数据不足）。")
                else:
                    # 筛选器
                    c1, c2, c3 = st.columns([0.33, 0.33, 0.34])
                    with c1:
                        p_sel = st.multiselect("按优先级筛选", options=["P0", "P1", "P2"], default=["P0", "P1", "P2"])
                    with c2:
                        o_sel = st.multiselect("按责任位筛选", options=["投放", "内容", "达人", "商品"], default=["投放", "内容", "达人", "商品"])
                    with c3:
                        imp_sel = st.multiselect("按影响等级筛选", options=["高", "中", "低"], default=["高", "中", "低"])

                    view = tasks.copy()
                    view = view[view['优先级'].isin(p_sel)]
                    view = view[view['第一责任位'].isin(o_sel)]
                    view = view[view['预估影响等级'].isin(imp_sel)]

                    st.dataframe(view, use_container_width=True, hide_index=True)

                    # 下载
                    csv = view.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="⬇️ 下载任务清单 CSV",
                        data=csv,
                        file_name="AI_tasks.csv",
                        mime="text/csv"
                    )

                    st.caption("说明：任务清单为规则驱动的‘工单化输出’，AI不执行；预估影响为粗估，用于排优先级。")



# ================= 周会概览页面渲染函数 =================

def render_weekly_summary_card(time_str, shop_metrics, spu_metrics, traffic_data, targets, df_shop_raw=None):
    """本周经营结论卡片"""
    st.markdown("### ① 本周经营结论")
    
    curr_rev = shop_metrics.get('退款后营收', 0)
    target_rev = targets.get('target_revenue', 0)
    curr_profit_rate = shop_metrics.get('利润率', 0)
    target_profit_rate = targets.get('target_profit_rate', 0.15)
    
    rev_achieve = (curr_rev / target_rev) if target_rev > 0 else 0
    
    # 计算时间进度（假设3月1-14日，时间进度约45%）
    time_progress = 45  # 3月1-14日约45%的月度进度
    gap_pp = time_progress - rev_achieve * 100
    
    # 计算火鸡包营收占比
    fire_chicken_ratio = 0
    try:
        df_orders = pd.read_excel('店铺订单.xlsx')
        df_orders = df_orders[df_orders['Order Status'] != 'Canceled'].copy()
        df_orders['营收'] = pd.to_numeric(df_orders['营收'], errors='coerce').fillna(0)
        
        # 读取采购成本表获取SKU->类目映射
        df_purchase = pd.read_excel('2602采购成本-含关税.xlsx')
        sku_category_map = {}
        if 'SKU' in df_purchase.columns and '三级类目' in df_purchase.columns:
            for _, row in df_purchase.iterrows():
                sku = str(row.get('SKU', '')).strip()
                category = str(row.get('三级类目', '')).strip()
                if sku and category and category != 'nan':
                    sku_category_map[sku] = category
        
        # 计算火鸡包营收
        fire_chicken_rev = 0
        total_rev = df_orders['营收'].sum()
        for _, row in df_orders.iterrows():
            seller_sku = str(row.get('Seller SKU', '')).strip()
            revenue = row['营收']
            category = sku_category_map.get(seller_sku, '')
            if '火鸡包' in category:
                fire_chicken_rev += revenue
        
        if total_rev > 0:
            fire_chicken_ratio = fire_chicken_rev / total_rev * 100
    except:
        pass
    
    # 构建固定格式结论
    conclusion_text = f"本周期营收${curr_rev:,.0f}，MTD达成率{rev_achieve*100:.0f}%，落后时间进度{gap_pp:.0f}pp，利润率{curr_profit_rate:.2%}，核心原因为低客单导致采购占比上涨+关税，火鸡包营收占比{fire_chicken_ratio:.1f}%，但为亏损，需尽快拉升。"
    
    st.markdown(f'''
    <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px;">
        <div style="font-size: 18px; font-weight: 600; margin-bottom: 10px;">💡 核心结论</div>
        <div style="font-size: 16px; line-height: 1.6;">{conclusion_text}</div>
    </div>
    ''', unsafe_allow_html=True)


def render_weekly_kpi_section(df_shop_raw, df_spu_raw, traffic_data, targets):
    """核心指标总览 - 分为销售类和品牌类"""
    st.markdown("### ② 核心指标总览")
    
    # 基础数据获取（2026年）
    curr_rev = float(df_shop_raw.iloc[0].get('退款后营收', 0)) if not df_shop_raw.empty else 0
    curr_profit_rate = float(df_shop_raw.iloc[0].get('利润率', 0)) if not df_shop_raw.empty else 0
    curr_mkt_rate = float(df_shop_raw.iloc[0].get('总营销费比', 0)) if not df_shop_raw.empty else 0
    
    curr_pv = traffic_data.get('current', {}).get('全店', {}).get('PV', 0) if traffic_data else 0
    last_pv = traffic_data.get('last', {}).get('全店', {}).get('PV', 0) if traffic_data and traffic_data.get('last') else 0
    
    # 计算当前周期视频播放量（达人视频+官号）
    curr_playback = calc_video_playback()
    
    # 读取2025年同期数据进行同比计算（MTD：3月1-14日）
    data_2025 = load_2025_financial_data(target_month=3, mtd_days=14)
    
    # 从订单表_2025计算MTD营收
    rev_2025_mtd = calc_2025_mtd_revenue(target_month=3, mtd_days=14)
    profit_rate_2025 = data_2025.get('毛利率', 0)
    mkt_rate_2025 = data_2025.get('广告费率', 0) + data_2025.get('达人佣金费率', 0)  # 营销费近似
    
    rev_yoy = ((curr_rev - rev_2025_mtd) / rev_2025_mtd * 100) if rev_2025_mtd > 0 else 0
    profit_rate_yoy = (curr_profit_rate - profit_rate_2025) * 100  # 百分点变化
    mkt_rate_yoy = (curr_mkt_rate - mkt_rate_2025) * 100
    
    # PV环比
    pv_mom = ((curr_pv - last_pv) / last_pv) if last_pv > 0 else 0
    
    # 第一行：销售类指标（带同比变化）
    st.markdown("**📊 销售类指标**")
    c1, c2, c3 = st.columns(3)
    
    # 退款后营收（带同比）
    rev_color = "#16a34a" if rev_yoy >= 0 else "#dc2626"
    rev_icon = "↑" if rev_yoy >= 0 else "↓"
    c1.markdown(f"""
    <div style="text-align: center;">
        <div style="font-size: 12px; color: #64748b; margin-bottom: 4px;">退款后营收</div>
        <div style="font-size: 24px; font-weight: bold; color: #0f172a;">${curr_rev:,.0f}</div>
        <div style="font-size: 11px; color: {rev_color};">同比: {rev_icon} {abs(rev_yoy):.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    profit_color = "#16a34a" if profit_rate_yoy >= 0 else "#dc2626"
    profit_icon = "↑" if profit_rate_yoy >= 0 else "↓"
    c2.markdown(f"""
    <div style="text-align: center;">
        <div style="font-size: 12px; color: #64748b; margin-bottom: 4px;">利润率</div>
        <div style="font-size: 24px; font-weight: bold; color: #0f172a;">{curr_profit_rate:.2%}</div>
        <div style="font-size: 11px; color: {profit_color};">同比: {profit_icon} {abs(profit_rate_yoy):.1f}pp</div>
    </div>
    """, unsafe_allow_html=True)
    
    mkt_color = "#16a34a" if mkt_rate_yoy <= 0 else "#dc2626"  # 营销费下降是好事
    mkt_icon = "↓" if mkt_rate_yoy <= 0 else "↑"
    c3.markdown(f"""
    <div style="text-align: center;">
        <div style="font-size: 12px; color: #64748b; margin-bottom: 4px;">营销费比</div>
        <div style="font-size: 24px; font-weight: bold; color: #0f172a;">{curr_mkt_rate:.2%}</div>
        <div style="font-size: 11px; color: {mkt_color};">同比: {mkt_icon} {abs(mkt_rate_yoy):.1f}pp</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 第二行：品牌类指标
    st.markdown("**📈 品牌类指标**")
    c4, c5 = st.columns(2)
    
    # 播放量（计算值）
    playback_display = f"{curr_playback:,.0f}" if curr_playback > 0 else "暂无数据"
    c4.markdown(f"""
    <div style="text-align: center;">
        <div style="font-size: 12px; color: #64748b; margin-bottom: 4px;">播放量</div>
        <div style="font-size: 24px; font-weight: bold; color: {'#0f172a' if curr_playback > 0 else '#94a3b8'};">{playback_display}</div>
        <div style="font-size: 11px; color: #94a3b8;">同比: --</div>
    </div>
    """, unsafe_allow_html=True)
    
    pv_color = "#16a34a" if pv_mom >= 0 else "#dc2626"
    pv_icon = "↑" if pv_mom >= 0 else "↓"
    c5.markdown(f"""
    <div style="text-align: center;">
        <div style="font-size: 12px; color: #64748b; margin-bottom: 4px;">Page View</div>
        <div style="font-size: 24px; font-weight: bold; color: #0f172a;">{curr_pv:,.0f}</div>
        <div style="font-size: 11px; color: {pv_color};">环比: {pv_icon} {abs(pv_mom):.1%}</div>
    </div>
    """, unsafe_allow_html=True)


def load_2025_financial_data(target_month=3, mtd_days=14):
    """从2025年财务管报读取指定月份的费率数据（营收从订单表计算）
    
    参数:
    - target_month: 目标月份
    - mtd_days: MTD天数（默认14天）
    """
    try:
        file_path = "2025年TideWe财务管报.xlsx"
        df = pd.read_excel(file_path, sheet_name='Sheet1')
        
        # 列名是datetime对象，需要找到对应的列
        target_col = None
        for col in df.columns:
            if hasattr(col, 'month') and col.month == target_month and col.year == 2025:
                target_col = col
                break
        
        # 如果没找到，尝试字符串匹配
        if target_col is None:
            target_col_str = f"2025-{target_month:02d}-01 00:00:00"
            for col in df.columns:
                if str(col) == target_col_str:
                    target_col = col
                    break
        
        result = {
            '营收': 0,  # 将从订单表_2025计算
            '退款率': 0,
            '毛利率': 0,
            '采购成本率': 0,
            '头程成本率': 0,
            '关税成本率': 0,
            '尾程费率': 0,
            '广告费率': 0,
            '达人佣金费率': 0,
            '样品费率': 0,
            '品牌费率': 0,
            '仓租费率': 0,
            '其他物流费用率': 0,
            '售后成本率': 0
        }
        
        if target_col is None:
            print(f"未找到2025年{target_month}月的数据列")
            return result
        
        # 遍历每一行，根据第一列的指标名提取数据（仅费率，营收从订单表计算）
        for idx, row in df.iterrows():
            indicator = str(row.get('Unnamed: 0', '')).strip() if pd.notna(row.get('Unnamed: 0')) else ''
            value = row.get(target_col, 0)
            
            if indicator == '退款率':
                result['退款率'] = float(value) if pd.notna(value) else 0
            elif indicator == '毛利率':
                result['毛利率'] = float(value) if pd.notna(value) else 0
            elif indicator == '采购成本率':
                result['采购成本率'] = float(value) if pd.notna(value) else 0
            elif indicator == '头程成本率':
                result['头程成本率'] = float(value) if pd.notna(value) else 0
            elif indicator == '关税成本率':
                result['关税成本率'] = float(value) if pd.notna(value) else 0
            elif indicator == '尾程费率':
                result['尾程费率'] = float(value) if pd.notna(value) else 0
            elif indicator == '广告费率':
                result['广告费率'] = float(value) if pd.notna(value) else 0
            elif indicator == '达人佣金费率':
                result['达人佣金费率'] = float(value) if pd.notna(value) else 0
            elif indicator == '样品费率':
                result['样品费率'] = float(value) if pd.notna(value) else 0
            elif indicator == '品牌费率':
                result['品牌费率'] = float(value) if pd.notna(value) else 0
            elif indicator == '仓租费率':
                result['仓租费率'] = float(value) if pd.notna(value) else 0
            elif indicator == '其他物流费用':
                result['其他物流费用率'] = float(value) if pd.notna(value) else 0
            elif indicator == '售后成本率':
                result['售后成本率'] = float(value) if pd.notna(value) else 0
        
        return result
    except Exception as e:
        print(f"读取2025年财务数据失败: {e}")
        import traceback
        traceback.print_exc()
        return {}


def calc_video_playback():
    """计算当前统计周期内的视频播放量
    
    计算逻辑：
    1. 从Video_List文件计算：(GMV / Shoppable video GPM) × 1000
    2. 从官号播放量文件读取官方账号播放量
    3. 两者相加得到总播放量
    
    返回: 总播放量数值
    """
    total_playback = 0
    
    try:
        # 1. 计算达人视频播放量
        video_file = "Video_List_20260301-20260314_20260317022914.xlsx"
        df_video = pd.read_excel(video_file)
        
        # 转换数值列
        df_video['GMV'] = pd.to_numeric(df_video['GMV'], errors='coerce').fillna(0)
        df_video['Shoppable video GPM'] = pd.to_numeric(df_video['Shoppable video GPM'], errors='coerce').fillna(0)
        
        # 计算每个视频的播放量并求和
        df_video['播放量'] = (df_video['GMV'] / df_video['Shoppable video GPM'] * 1000).replace([float('inf'), -float('inf')], 0).fillna(0)
        creator_playback = int(df_video['播放量'].sum())
        total_playback += creator_playback
        
    except Exception as e:
        print(f"计算达人视频播放量失败: {e}")
    
    try:
        # 2. 读取官号播放量
        official_file = "官号播放量.xlsx"
        df_official = pd.read_excel(official_file)
        
        # 查找TideWe官方账号的播放量
        if '播放量' in df_official.columns:
            official_playback = pd.to_numeric(df_official['播放量'], errors='coerce').fillna(0).sum()
            total_playback += int(official_playback)
        
    except Exception as e:
        print(f"读取官号播放量失败: {e}")
    
    return total_playback


def calc_2025_mtd_revenue(target_month=3, mtd_days=14):
    """从订单表_2025计算指定月份的MTD营收（排除Canceled订单）
    
    参数:
    - target_month: 目标月份
    - mtd_days: MTD天数（默认14天）
    
    返回: MTD营收金额
    """
    try:
        file_path = "订单表_2025.xlsx"
        df = pd.read_excel(file_path)
        
        # 转换Created Time为日期
        df['Created Time'] = pd.to_datetime(df['Created Time'], errors='coerce')
        
        # 筛选指定日期范围的订单
        year = 2025
        start_date = pd.to_datetime(f'{year}-{target_month:02d}-01')
        end_date = pd.to_datetime(f'{year}-{target_month:02d}-{mtd_days + 1}')  # 15号0点之前
        
        mask = (df['Created Time'] >= start_date) & (df['Created Time'] < end_date)
        mtd_orders = df[mask]
        
        # 排除Canceled状态的订单（退款订单）
        if 'Order Status' in mtd_orders.columns:
            # 排除Canceled订单，只保留非Canceled状态的订单
            mtd_orders = mtd_orders[mtd_orders['Order Status'] != 'Canceled']
        
        # 计算营收总和
        revenue_col = '营收'
        if revenue_col not in mtd_orders.columns:
            print(f"未找到营收列")
            return 0
        
        # 处理营收数据（已经是数值格式）
        mtd_revenue = pd.to_numeric(mtd_orders[revenue_col], errors='coerce').fillna(0).sum()
        
        return float(mtd_revenue)
    except Exception as e:
        print(f"计算2025年MTD营收失败: {e}")
        import traceback
        traceback.print_exc()
        return 0


def render_weekly_deviation_analysis(df_shop_raw, traffic_data):
    """经营偏差分析 - 费率同比对比+结论"""
    st.markdown("### ③ 经营偏差分析")
    
    # 第一部分：各项费率同比对比表
    st.markdown("**【各项费率同期】**")
    
    # 当前数据（2026年）- 从df_shop_raw获取
    curr_rev = float(df_shop_raw.iloc[0].get('退款后营收', 0)) if not df_shop_raw.empty else 0
    curr_profit_rate = float(df_shop_raw.iloc[0].get('利润率', 0)) if not df_shop_raw.empty else 0
    curr_refund_rate = float(df_shop_raw.iloc[0].get('退款率', 0)) if not df_shop_raw.empty else 0
    
    # 读取2025年同期数据（MTD：3月1-14日）
    data_2025 = load_2025_financial_data(target_month=3, mtd_days=14)
    # 从订单表_2025计算MTD营收
    rev_2025 = calc_2025_mtd_revenue(target_month=3, mtd_days=14)
    # 将营收添加到data_2025字典中
    data_2025['营收'] = rev_2025
    
    # 计算2026年ASP（从店铺订单.xlsx读取销量）
    asp_2026 = 0
    try:
        df_orders_2026 = pd.read_excel('店铺订单.xlsx')
        df_orders_2026_clean = df_orders_2026[df_orders_2026['Order Status'] != 'Canceled'].copy()
        qty_2026 = pd.to_numeric(df_orders_2026_clean['Quantity'], errors='coerce').fillna(0).sum()
        asp_2026 = curr_rev / qty_2026 if qty_2026 > 0 else 0
    except:
        pass
    
    # 计算2025年同期ASP（从订单表_2025读取销量）
    asp_2025 = 0
    try:
        df_orders_2025 = pd.read_excel('订单表_2025.xlsx')
        df_orders_2025['Created Time'] = pd.to_datetime(df_orders_2025['Created Time'], errors='coerce')
        df_orders_2025_clean = df_orders_2025[
            (df_orders_2025['Created Time'] >= '2025-03-01') & 
            (df_orders_2025['Created Time'] < '2025-03-15') &
            (df_orders_2025['Order Status'] != 'Canceled')
        ].copy()
        qty_2025 = pd.to_numeric(df_orders_2025_clean['Quantity'], errors='coerce').fillna(0).sum()
        asp_2025 = rev_2025 / qty_2025 if qty_2025 > 0 else 0
    except:
        pass
    
    # 将ASP添加到data_2025字典中
    data_2025['ASP'] = asp_2025
    
    # 构建转置表格数据（行=时间，列=指标）
    # 定义要展示的指标（简化的列名）
    indicators = [
        ('营收', '营收($)', lambda x: f"${x:,.0f}" if x != 0 else "-", lambda c, p: f"{'↑' if c > p else '↓'}${abs(c-p):,.0f}" if p != 0 else "-"),
        ('ASP', 'ASP($)', lambda x: f"${x:.2f}" if x != 0 else "-", lambda c, p: f"{'↑' if c > p else '↓'}${abs(c-p):.2f}" if p != 0 else "-"),
        ('毛利率', '利润率', lambda x: f"{x:.2%}", lambda c, p: f"{'↑' if c > p else '↓'}{abs(c-p)*100:.2f}pp" if p != 0 else "-"),
        ('退款率', '退款率', lambda x: f"{x:.2%}", lambda c, p: f"{'↑' if c > p else '↓'}{abs(c-p)*100:.2f}pp" if p != 0 else "-"),
        ('采购成本率', '采购成本', lambda x: f"{x:.2%}", lambda c, p: f"{'↑' if c > p else '↓'}{abs(c-p)*100:.2f}pp" if p != 0 else "-"),
        ('头程成本率', '头程成本', lambda x: f"{x:.2%}", lambda c, p: f"{'↑' if c > p else '↓'}{abs(c-p)*100:.2f}pp" if p != 0 else "-"),
        ('尾程费率', '尾程费率', lambda x: f"{x:.2%}", lambda c, p: f"{'↑' if c > p else '↓'}{abs(c-p)*100:.2f}pp" if p != 0 else "-"),
        ('关税成本率', '关税成本', lambda x: f"{x:.2%}", lambda c, p: f"{'↑' if c > p else '↓'}{abs(c-p)*100:.2f}pp" if p != 0 else "-"),
        ('广告费率', '广告费率', lambda x: f"{x:.2%}", lambda c, p: f"{'↑' if c > p else '↓'}{abs(c-p)*100:.2f}pp" if p != 0 else "-"),
        ('达人佣金费率', '达人佣金', lambda x: f"{x:.2%}", lambda c, p: f"{'↑' if c > p else '↓'}{abs(c-p)*100:.2f}pp" if p != 0 else "-"),
        ('样品费率', '样品费率', lambda x: f"{x:.2%}", lambda c, p: f"{'↑' if c > p else '↓'}{abs(c-p)*100:.2f}pp" if p != 0 else "-"),
        ('品牌费率', '品牌费率', lambda x: f"{x:.2%}", lambda c, p: f"{'↑' if c > p else '↓'}{abs(c-p)*100:.2f}pp" if p != 0 else "-"),
        ('仓租费率', '仓租费率', lambda x: f"{x:.2%}", lambda c, p: f"{'↑' if c > p else '↓'}{abs(c-p)*100:.2f}pp" if p != 0 else "-"),
        ('售后成本率', '售后成本', lambda x: f"{x:.2%}", lambda c, p: f"{'↑' if c > p else '↓'}{abs(c-p)*100:.2f}pp" if p != 0 else "-"),
    ]
    
    # 2026年数据字典
    curr_data = {
        '营收': curr_rev,
        'ASP': asp_2026,
        '毛利率': curr_profit_rate,
        '退款率': curr_refund_rate,
        '采购成本率': float(df_shop_raw.iloc[0].get('采购成本率', 0)) if not df_shop_raw.empty else 0,
        '头程成本率': float(df_shop_raw.iloc[0].get('头程成本率', 0)) if not df_shop_raw.empty else 0,
        '尾程费率': float(df_shop_raw.iloc[0].get('尾程费率', 0)) if not df_shop_raw.empty else 0,
        '关税成本率': float(df_shop_raw.iloc[0].get('关税成本率', 0)) if not df_shop_raw.empty else 0,
        '广告费率': float(df_shop_raw.iloc[0].get('广告费率', 0)) if not df_shop_raw.empty else 0,
        '达人佣金费率': float(df_shop_raw.iloc[0].get('达人佣金费率', 0)) if not df_shop_raw.empty else 0,
        '样品费率': float(df_shop_raw.iloc[0].get('样品费率', 0)) if not df_shop_raw.empty else 0,
        '品牌费率': float(df_shop_raw.iloc[0].get('品牌费率', 0)) if not df_shop_raw.empty else 0,
        '仓租费率': float(df_shop_raw.iloc[0].get('仓租费率', 0)) if not df_shop_raw.empty else 0,
        '售后成本率': float(df_shop_raw.iloc[0].get('售后成本率', 0)) if not df_shop_raw.empty else 0,
    }
    
    # 准备三行数据（显示具体的数据周期）
    row_2026 = {'时间': '2026年3月1-14日'}
    row_2025 = {'时间': '2025年3月1-14日'}
    row_change = {'时间': 'YOY变化'}
    
    major_changes = []
    
    for key, short_name, fmt_func, change_func in indicators:
        curr_val = curr_data.get(key, 0)
        prev_val = data_2025.get(key, 0)
        
        row_2026[short_name] = fmt_func(curr_val)
        row_2025[short_name] = fmt_func(prev_val) if prev_val != 0 else "-"
        
        # 计算变化
        if key == '营收':
            change_str = f"{'↑' if curr_val > prev_val else '↓'}${abs(curr_val-prev_val):,.0f}" if prev_val != 0 else "-"
            if prev_val != 0 and abs(curr_val - prev_val) > 10:
                major_changes.append(f"营收{'上升' if curr_val > prev_val else '下降'}${abs(curr_val-prev_val):,.0f}")
        elif key == 'ASP':
            # ASP用绝对值差异（$），颜色标注：>$5绿色，<$-5红色
            change_val = curr_val - prev_val
            if prev_val == 0:
                change_str = "-"
            elif change_val > 5:
                change_str = f"<span style='color: #16a34a;'>↑${change_val:.2f}</span>"
            elif change_val < -5:
                change_str = f"<span style='color: #dc2626;'>↓${abs(change_val):.2f}</span>"
            elif change_val > 0:
                change_str = f"↑${change_val:.2f}"
            else:
                change_str = f"↓${abs(change_val):.2f}"
            
            if prev_val != 0 and abs(change_val) > 3:
                direction = "上升" if change_val > 0 else "下降"
                major_changes.append(f"{short_name}{direction}${abs(change_val):.2f}")
        else:
            change_pp = (curr_val - prev_val) * 100
            # YOY变化颜色标注：>5pp绿色，<-5pp红色
            if prev_val == 0:
                change_str = "-"
            elif change_pp > 5:
                change_str = f"<span style='color: #16a34a;'>↑{change_pp:.2f}pp</span>"
            elif change_pp < -5:
                change_str = f"<span style='color: #dc2626;'>↓{abs(change_pp):.2f}pp</span>"
            elif change_pp > 0:
                change_str = f"↑{change_pp:.2f}pp"
            else:
                change_str = f"↓{abs(change_pp):.2f}pp"
            
            if prev_val != 0 and abs(change_pp) > 3:
                direction = "上升" if change_pp > 0 else "下降"
                major_changes.append(f"{short_name}{direction}{abs(change_pp):.1f}pp")
        
        row_change[short_name] = change_str
    
    # 构建转置后的DataFrame（用于markdown表格显示）
    df_transposed = pd.DataFrame([row_2026, row_2025, row_change])
    
    # 使用HTML表格显示以支持颜色渲染
    html_table = "<table style='width:100%; border-collapse: collapse;'>"
    
    # 表头
    html_table += "<tr>"
    for col in df_transposed.columns:
        html_table += f"<th style='border:1px solid #ddd; padding:8px; text-align:left; background-color:#f2f2f2;'>{col}</th>"
    html_table += "</tr>"
    
    # 数据行
    for _, row in df_transposed.iterrows():
        html_table += "<tr>"
        for col in df_transposed.columns:
            val = row[col]
            html_table += f"<td style='border:1px solid #ddd; padding:8px; text-align:left;'>{val}</td>"
        html_table += "</tr>"
    
    html_table += "</table>"
    
    st.markdown(html_table, unsafe_allow_html=True)
    
    # 第二部分：结论性总结
    st.markdown("**【偏差结论】**")
    
    conclusions = []
    
    if major_changes:
        conclusions.append(f"主要指标变动：{'; '.join(major_changes[:3])}")
    
    # 营收分析
    if rev_2025 > 0:
        rev_change_pct = (curr_rev - rev_2025) / rev_2025 * 100
        if rev_change_pct > 20:
            conclusions.append(f"营收同比增长{rev_change_pct:.1f}%，业务扩张良好")
        elif rev_change_pct < -20:
            conclusions.append(f"营收同比下降{abs(rev_change_pct):.1f}%，需关注销售目标达成")
    
    # 利润率分析
    profit_change_pp = (curr_profit_rate - data_2025.get('毛利率', 0)) * 100
    if abs(profit_change_pp) > 5:
        direction = "提升" if profit_change_pp > 0 else "下滑"
        conclusions.append(f"利润率同比{direction}{abs(profit_change_pp):.1f}pp")
    
    # 费率分析
    if data_2025.get('关税成本率', 0) > 0.05:
        conclusions.append("关税成本占比较高，建议优化供应链布局")
    
    ad_change = (curr_data.get('广告费率', 0) - data_2025.get('广告费率', 0)) * 100
    if ad_change < -3:
        conclusions.append(f"广告费率同比下降{abs(ad_change):.1f}pp，投放效率提升")
    elif ad_change > 3:
        conclusions.append(f"广告费率同比上升{ad_change:.1f}pp，需关注投放ROI")
    
    if not conclusions:
        conclusions.append("各项指标同比变动在合理区间内，经营结构相对稳定")
    
    st.markdown(f"""
    <div style="padding: 12px; background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                border-left: 4px solid #0284c7; border-radius: 6px; margin-top: 10px;">
        <div style="font-weight: 600; color: #0369a1; margin-bottom: 6px;">💡 关键洞察</div>
        <div style="color: #334155; line-height: 1.6;">{'; '.join(conclusions)}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 第二部分：三级类目和SPU对比分析
    st.markdown("---")
    render_category_comparison(df_shop_raw, {})
    st.markdown("---")
    render_spu_comparison(df_shop_raw, {})


def get_category_mapping(sku_category_map, spu_category_map, sku_or_spu, spu_short=None):
    """获取三级类目映射
    
    参数:
    - sku_category_map: SKU->类目映射字典
    - spu_category_map: SPU->类目映射字典（短格式SPU）
    - sku_or_spu: 要查找的SKU或SPU
    - spu_short: 短格式SPU（用于前缀匹配）
    
    返回: (类目名称, 映射方式)
    """
    # 1. 优先SKU映射
    if sku_or_spu in sku_category_map:
        return sku_category_map[sku_or_spu], "SKU映射"
    
    # 2. 其次SPU映射（前缀匹配）
    if spu_short and spu_short in spu_category_map:
        return spu_category_map[spu_short], "SPU映射"
    
    # 3. 尝试前缀匹配映射表中的SPU
    if spu_short:
        for map_spu, category in spu_category_map.items():
            if str(map_spu).startswith(str(spu_short)) or str(spu_short).startswith(str(map_spu)):
                return category, "SPU前缀映射"
    
    return "未分类", "未映射"


def render_category_comparison(df_shop_raw, dfs=None):
    """【By类目】Top 10 三级类目营收对比"""
    st.markdown("**【By类目】Top 10 三级类目营收对比**")
    
    try:
        # 读取2026年订单数据（当前周期）
        df_2026 = None
        if dfs and 'orders' in dfs:
            df_2026 = dfs.get('orders')
        
        # 如果dfs中没有，直接读取本地文件
        if df_2026 is None or df_2026.empty:
            try:
                df_2026 = pd.read_excel('店铺订单.xlsx')
            except:
                st.info("💡 未找到2026年订单数据（店铺订单.xlsx）")
                return
        
        # 读取2025年订单数据
        df_2025 = pd.read_excel('订单表_2025.xlsx')
        
        # 读取SPU-SKU映射表
        df_spu_sku = pd.read_excel('spu_sku映射表.xlsx')
        
        # 读取采购成本表获取三级类目
        df_purchase = pd.read_excel('2602采购成本-含关税.xlsx')
        
        # 构建SKU->SPU映射（短格式）
        sku_to_spu_short = {}
        spu_to_category = {}
        
        # 从采购成本表提取SKU->类目映射
        sku_category_map = {}
        if 'SKU' in df_purchase.columns and '三级类目' in df_purchase.columns:
            for _, row in df_purchase.iterrows():
                sku = str(row.get('SKU', '')).strip()
                category = str(row.get('三级类目', '')).strip()
                if sku and category and category != 'nan':
                    sku_category_map[sku] = category
        
        # 从spu_sku映射表提取信息
        for _, row in df_spu_sku.iterrows():
            spu_full = str(row.get('SPU', '')).strip()
            sku = str(row.get('在售SKU', '')).strip()
            
            # 提取短格式SPU（如BL001-TE -> BL001）
            spu_short = spu_full.split('-')[0] if '-' in spu_full else spu_full
            
            if sku:
                sku_to_spu_short[sku] = spu_short
        
        # 处理2026年数据 - 分别统计非Canceled和Canceled订单
        df_2026['营收'] = pd.to_numeric(df_2026['营收'], errors='coerce').fillna(0)
        df_2026['Quantity'] = pd.to_numeric(df_2026['Quantity'], errors='coerce').fillna(0)
        
        # 非Canceled订单（正常订单）
        df_2026_clean = df_2026[df_2026['Order Status'] != 'Canceled'].copy()
        # Canceled订单（退款订单）
        df_2026_cancelled = df_2026[df_2026['Order Status'] == 'Canceled'].copy()
        
        # 通过SKU关联SPU和类目 - 统计所有订单（包含Canceled）
        category_data_2026 = {}  # 总销量（包含Canceled）和营收（非Canceled）
        category_cancelled_qty_2026 = {}  # 记录各类目的退款订单销量
        sku_mapped = 0
        sku_unmapped = 0
        
        # 先统计非Canceled订单（营收+销量）
        for _, row in df_2026_clean.iterrows():
            seller_sku = str(row.get('Seller SKU', '')).strip()
            revenue = row['营收']
            quantity = row['Quantity']
            
            # 获取短格式SPU
            spu_short = sku_to_spu_short.get(seller_sku)
            
            # 获取类目
            category, map_method = get_category_mapping(
                sku_category_map, {}, seller_sku, spu_short
            )
            
            if category != "未分类":
                sku_mapped += 1
            else:
                sku_unmapped += 1
            
            if category not in category_data_2026:
                category_data_2026[category] = {'营收': 0, '销量': 0}
            category_data_2026[category]['营收'] += revenue
            category_data_2026[category]['销量'] += quantity
        
        # 再统计Canceled订单的销量（加到总销量中，同时记录退款销量）
        for _, row in df_2026_cancelled.iterrows():
            seller_sku = str(row.get('Seller SKU', '')).strip()
            quantity = row['Quantity']
            
            spu_short = sku_to_spu_short.get(seller_sku)
            category, map_method = get_category_mapping(
                sku_category_map, {}, seller_sku, spu_short
            )
            
            # 将Canceled订单销量加到总销量
            if category not in category_data_2026:
                category_data_2026[category] = {'营收': 0, '销量': 0}
            category_data_2026[category]['销量'] += quantity
            
            # 单独记录退款销量
            if category not in category_cancelled_qty_2026:
                category_cancelled_qty_2026[category] = 0
            category_cancelled_qty_2026[category] += quantity
        
        # 处理2025年数据（3月1-14日）
        df_2025['Created Time'] = pd.to_datetime(df_2025['Created Time'], errors='coerce')
        df_2025_period = df_2025[
            (df_2025['Created Time'] >= '2025-03-01') & 
            (df_2025['Created Time'] < '2025-03-15')
        ].copy()
        df_2025_period['营收'] = pd.to_numeric(df_2025_period['营收'], errors='coerce').fillna(0)
        df_2025_period['Quantity'] = pd.to_numeric(df_2025_period['Quantity'], errors='coerce').fillna(0)
        
        # 非Canceled订单
        df_2025_clean = df_2025_period[df_2025_period['Order Status'] != 'Canceled'].copy()
        # Canceled订单
        df_2025_cancelled = df_2025_period[df_2025_period['Order Status'] == 'Canceled'].copy()
        
        # 2025年已有SPU字段，需要映射到短格式
        category_data_2025 = {}  # 总销量（包含Canceled）和营收（非Canceled）
        category_cancelled_qty_2025 = {}  # 记录各类目的退款订单销量
        
        # 先统计非Canceled订单
        for _, row in df_2025_clean.iterrows():
            spu_full = str(row.get('SPU', '')).strip()
            revenue = row['营收']
            quantity = row['Quantity']
            
            # 获取短格式SPU
            spu_short = spu_full.split('-')[0] if '-' in spu_full else spu_full
            
            # 获取类目（通过SKU映射或SPU前缀匹配）
            category = "未分类"
            for sku, s_short in sku_to_spu_short.items():
                if s_short == spu_short:
                    if sku in sku_category_map:
                        category = sku_category_map[sku]
                        break
            
            if category not in category_data_2025:
                category_data_2025[category] = {'营收': 0, '销量': 0}
            category_data_2025[category]['营收'] += revenue
            category_data_2025[category]['销量'] += quantity
        
        # 再统计Canceled订单的销量（加到总销量中，同时记录退款销量）
        for _, row in df_2025_cancelled.iterrows():
            spu_full = str(row.get('SPU', '')).strip()
            quantity = row['Quantity']
            
            spu_short = spu_full.split('-')[0] if '-' in spu_full else spu_full
            category = "未分类"
            for sku, s_short in sku_to_spu_short.items():
                if s_short == spu_short:
                    if sku in sku_category_map:
                        category = sku_category_map[sku]
                        break
            
            # 将Canceled订单销量加到总销量
            if category not in category_data_2025:
                category_data_2025[category] = {'营收': 0, '销量': 0}
            category_data_2025[category]['销量'] += quantity
            
            # 单独记录退款销量
            if category not in category_cancelled_qty_2025:
                category_cancelled_qty_2025[category] = 0
            category_cancelled_qty_2025[category] += quantity
        
        # 合并数据并计算Top 10
        all_categories = set(category_data_2026.keys()) | set(category_data_2025.keys())
        
        merged_data = []
        for category in all_categories:
            rev_2026 = category_data_2026.get(category, {'营收': 0, '销量': 0})['营收']
            qty_2026_total = category_data_2026.get(category, {'营收': 0, '销量': 0})['销量']
            rev_2025 = category_data_2025.get(category, {'营收': 0, '销量': 0})['营收']
            qty_2025_total = category_data_2025.get(category, {'营收': 0, '销量': 0})['销量']
            
            # 获取Canceled订单销量
            qty_2026_cancelled = category_cancelled_qty_2026.get(category, 0)
            qty_2025_cancelled = category_cancelled_qty_2025.get(category, 0)
            
            # 有效销量 = 总销量（非Canceled + Canceled）- Canceled订单销量 = 非Canceled订单销量
            qty_2026 = qty_2026_total - qty_2026_cancelled
            qty_2025 = qty_2025_total - qty_2025_cancelled
            
            # 计算ASP（退款后营收 / 有效销量）
            asp_2026 = rev_2026 / qty_2026 if qty_2026 > 0 else 0
            asp_2025 = rev_2025 / qty_2025 if qty_2025 > 0 else 0
            
            # 计算YOY
            rev_yoy = ((rev_2026 - rev_2025) / rev_2025 * 100) if rev_2025 > 0 else (float('inf') if rev_2026 > 0 else 0)
            qty_yoy = ((qty_2026 - qty_2025) / qty_2025 * 100) if qty_2025 > 0 else (float('inf') if qty_2026 > 0 else 0)
            asp_yoy = ((asp_2026 - asp_2025) / asp_2025 * 100) if asp_2025 > 0 else (float('inf') if asp_2026 > 0 else 0)
            
            merged_data.append({
                '三级类目': category,
                '2026营收($)': rev_2026,
                '2025营收($)': rev_2025,
                '营收YOY': rev_yoy,
                '2026销量': qty_2026,
                '2025销量': qty_2025,
                '销量YOY': qty_yoy,
                '2026 ASP': asp_2026,
                '2025 ASP': asp_2025,
                'ASP YOY': asp_yoy
            })
        
        # 按2026营收排序取Top 10
        merged_data.sort(key=lambda x: x['2026营收($)'], reverse=True)
        top10 = merged_data[:10]
        
        # 计算总结数据
        if top10:
            total_rev = sum(d['2026营收($)'] for d in top10)
            top3_rev = sum(d['2026营收($)'] for d in top10[:3])
            top3_ratio = top3_rev / total_rev * 100 if total_rev > 0 else 0
            
            # 找火鸡包数据
            fire_chicken_yoy = 0
            for d in top10:
                if '火鸡包' in d['三级类目']:
                    fire_chicken_yoy = d['营收YOY']
                    break
            
            yoy_str = f"YOY+{fire_chicken_yoy:.0f}%" if fire_chicken_yoy > 0 else f"YOY{fire_chicken_yoy:.0f}%"
            st.caption(f"💡 Top 3类目贡献{top3_ratio:.0f}%营收，火鸡包{yoy_str}领跑但ASP承压")
        
        # 显示表格
        if top10:
            df_display = pd.DataFrame(top10)
            
            # 格式化显示函数（>20%绿色、<-20%红色，其他黑色；"新增"改为"/"）
            def format_yoy(x):
                if x == float('inf'):
                    return "/"
                elif x > 20:
                    return f"<span style='color: #16a34a;'>↑{x:.1f}%</span>"
                elif x < -20:
                    return f"<span style='color: #dc2626;'>↓{abs(x):.1f}%</span>"
                elif x > 0:
                    return f"↑{x:.1f}%"
                elif x < 0:
                    return f"↓{abs(x):.1f}%"
                else:
                    return "-"
            
            # 格式化数值列
            df_display['2026营收($)'] = df_display['2026营收($)'].apply(lambda x: f"${x:,.0f}")
            df_display['2025营收($)'] = df_display['2025营收($)'].apply(lambda x: f"${x:,.0f}" if x > 0 else "-")
            df_display['营收YOY'] = df_display['营收YOY'].apply(format_yoy)
            df_display['2026销量'] = df_display['2026销量'].apply(lambda x: f"{x:,.0f}")
            df_display['2025销量'] = df_display['2025销量'].apply(lambda x: f"{x:,.0f}" if x > 0 else "-")
            df_display['销量YOY'] = df_display['销量YOY'].apply(format_yoy)
            df_display['2026 ASP'] = df_display['2026 ASP'].apply(lambda x: f"${x:.2f}")
            df_display['2025 ASP'] = df_display['2025 ASP'].apply(lambda x: f"${x:.2f}" if x > 0 else "-")
            df_display['ASP YOY'] = df_display['ASP YOY'].apply(format_yoy)
            
            # 使用HTML显示带颜色的表格，营收列不加粗
            html_table = df_display.to_html(escape=False, index=False)
            # 为营收列添加正常字重样式
            html_table = html_table.replace('<th>2026营收($)</th>', '<th style="font-weight: normal;">2026营收($)</th>')
            html_table = html_table.replace('<th>2025营收($)</th>', '<th style="font-weight: normal;">2025营收($)</th>')
            st.write(html_table, unsafe_allow_html=True)
            
            # 显示映射统计
            total_skus = sku_mapped + sku_unmapped
            st.caption(f"📋 映射统计：共{total_skus}个SKU，成功映射{sku_mapped}个({sku_mapped/total_skus*100:.1f}%)，未分类{sku_unmapped}个")
            
            # 显示未分类详情（如果存在）
            if "未分类" in [d['三级类目'] for d in merged_data]:
                unclassified_revenue = category_data_2026.get('未分类', {}).get('营收', 0)
                if unclassified_revenue > 0:
                    st.caption(f"⚠️ 未分类类目共{unclassified_revenue:,.0f}营收，建议检查SPU-SKU映射表或采购成本表中的三级类目配置")
        else:
            st.info("暂无类目数据")
            
    except Exception as e:
        st.error(f"类目对比分析失败: {e}")
        import traceback
        traceback.print_exc()


def render_spu_comparison(df_shop_raw, dfs=None):
    """【By SPU】Top 10 SPU营收对比"""
    st.markdown("**【By SPU】Top 10 SPU营收对比**")
    
    try:
        # 读取2026年订单数据（当前周期）
        df_2026 = None
        if dfs and 'orders' in dfs:
            df_2026 = dfs.get('orders')
        
        # 如果dfs中没有，直接读取本地文件
        if df_2026 is None or df_2026.empty:
            try:
                df_2026 = pd.read_excel('店铺订单.xlsx')
            except:
                st.info("💡 未找到2026年订单数据（店铺订单.xlsx）")
                return
        
        # 读取2025年订单数据
        df_2025 = pd.read_excel('订单表_2025.xlsx')
        
        # 读取SPU-SKU映射表
        df_spu_sku = pd.read_excel('spu_sku映射表.xlsx')
        
        # 构建SKU->SPU映射
        sku_to_spu_short = {}
        for _, row in df_spu_sku.iterrows():
            spu_full = str(row.get('SPU', '')).strip()
            sku = str(row.get('在售SKU', '')).strip()
            
            # 提取短格式SPU
            spu_short = spu_full.split('-')[0] if '-' in spu_full else spu_full
            
            if sku:
                sku_to_spu_short[sku] = spu_short
        
        # 处理2026年数据 - 分别统计非Canceled和Canceled订单
        df_2026['营收'] = pd.to_numeric(df_2026['营收'], errors='coerce').fillna(0)
        df_2026['Quantity'] = pd.to_numeric(df_2026['Quantity'], errors='coerce').fillna(0)
        
        # 非Canceled订单
        df_2026_clean = df_2026[df_2026['Order Status'] != 'Canceled'].copy()
        # Canceled订单
        df_2026_cancelled = df_2026[df_2026['Order Status'] == 'Canceled'].copy()
        
        # 通过SKU关联SPU - 统计所有订单（包含Canceled）
        spu_data_2026 = {}  # 总销量（包含Canceled）和营收（非Canceled）
        spu_cancelled_qty_2026 = {}  # 记录各SPU的退款订单销量
        sku_mapped = 0
        sku_unmapped = 0
        
        # 先统计非Canceled订单
        for _, row in df_2026_clean.iterrows():
            seller_sku = str(row.get('Seller SKU', '')).strip()
            revenue = row['营收']
            quantity = row['Quantity']
            
            # 获取SPU
            spu = sku_to_spu_short.get(seller_sku)
            
            if spu:
                sku_mapped += 1
            else:
                sku_unmapped += 1
                spu = "未关联"
            
            if spu not in spu_data_2026:
                spu_data_2026[spu] = {'营收': 0, '销量': 0}
            spu_data_2026[spu]['营收'] += revenue
            spu_data_2026[spu]['销量'] += quantity
        
        # 再统计Canceled订单的销量（加到总销量中，同时记录退款销量）
        for _, row in df_2026_cancelled.iterrows():
            seller_sku = str(row.get('Seller SKU', '')).strip()
            quantity = row['Quantity']
            
            spu = sku_to_spu_short.get(seller_sku)
            if not spu:
                spu = "未关联"
            
            # 将Canceled订单销量加到总销量
            if spu not in spu_data_2026:
                spu_data_2026[spu] = {'营收': 0, '销量': 0}
            spu_data_2026[spu]['销量'] += quantity
            
            # 单独记录退款销量
            if spu not in spu_cancelled_qty_2026:
                spu_cancelled_qty_2026[spu] = 0
            spu_cancelled_qty_2026[spu] += quantity
        
        # 处理2025年数据（3月1-14日）
        df_2025['Created Time'] = pd.to_datetime(df_2025['Created Time'], errors='coerce')
        df_2025_period = df_2025[
            (df_2025['Created Time'] >= '2025-03-01') & 
            (df_2025['Created Time'] < '2025-03-15')
        ].copy()
        df_2025_period['营收'] = pd.to_numeric(df_2025_period['营收'], errors='coerce').fillna(0)
        df_2025_period['Quantity'] = pd.to_numeric(df_2025_period['Quantity'], errors='coerce').fillna(0)
        
        # 非Canceled订单
        df_2025_clean = df_2025_period[df_2025_period['Order Status'] != 'Canceled'].copy()
        # Canceled订单
        df_2025_cancelled = df_2025_period[df_2025_period['Order Status'] == 'Canceled'].copy()
        
        # 2025年已有SPU字段，需要标准化为短格式
        spu_data_2025 = {}  # 总销量（包含Canceled）和营收（非Canceled）
        spu_cancelled_qty_2025 = {}  # 记录各SPU的退款订单销量
        
        # 先统计非Canceled订单
        for _, row in df_2025_clean.iterrows():
            spu_full = str(row.get('SPU', '')).strip()
            revenue = row['营收']
            quantity = row['Quantity']
            
            # 转换为短格式
            spu_short = spu_full.split('-')[0] if '-' in spu_full else spu_full
            
            if spu_short not in spu_data_2025:
                spu_data_2025[spu_short] = {'营收': 0, '销量': 0}
            spu_data_2025[spu_short]['营收'] += revenue
            spu_data_2025[spu_short]['销量'] += quantity
        
        # 再统计Canceled订单的销量（加到总销量中，同时记录退款销量）
        for _, row in df_2025_cancelled.iterrows():
            spu_full = str(row.get('SPU', '')).strip()
            quantity = row['Quantity']
            
            spu_short = spu_full.split('-')[0] if '-' in spu_full else spu_full
            
            # 将Canceled订单销量加到总销量
            if spu_short not in spu_data_2025:
                spu_data_2025[spu_short] = {'营收': 0, '销量': 0}
            spu_data_2025[spu_short]['销量'] += quantity
            
            # 单独记录退款销量
            if spu_short not in spu_cancelled_qty_2025:
                spu_cancelled_qty_2025[spu_short] = 0
            spu_cancelled_qty_2025[spu_short] += quantity
        
        # 合并数据并计算Top 10
        all_spus = set(spu_data_2026.keys()) | set(spu_data_2025.keys())
        
        merged_data = []
        for spu in all_spus:
            rev_2026 = spu_data_2026.get(spu, {'营收': 0, '销量': 0})['营收']
            qty_2026_total = spu_data_2026.get(spu, {'营收': 0, '销量': 0})['销量']
            rev_2025 = spu_data_2025.get(spu, {'营收': 0, '销量': 0})['营收']
            qty_2025_total = spu_data_2025.get(spu, {'营收': 0, '销量': 0})['销量']
            
            # 获取Canceled订单销量
            qty_2026_cancelled = spu_cancelled_qty_2026.get(spu, 0)
            qty_2025_cancelled = spu_cancelled_qty_2025.get(spu, 0)
            
            # 有效销量 = 非Canceled订单销量 - Canceled订单销量（方案B）
            qty_2026 = qty_2026_total - qty_2026_cancelled
            qty_2025 = qty_2025_total - qty_2025_cancelled
            
            # 计算ASP（退款后营收 / 有效销量）
            asp_2026 = rev_2026 / qty_2026 if qty_2026 > 0 else 0
            asp_2025 = rev_2025 / qty_2025 if qty_2025 > 0 else 0
            
            # 计算YOY
            rev_yoy = ((rev_2026 - rev_2025) / rev_2025 * 100) if rev_2025 > 0 else (float('inf') if rev_2026 > 0 else 0)
            qty_yoy = ((qty_2026 - qty_2025) / qty_2025 * 100) if qty_2025 > 0 else (float('inf') if qty_2026 > 0 else 0)
            asp_yoy = ((asp_2026 - asp_2025) / asp_2025 * 100) if asp_2025 > 0 else (float('inf') if asp_2026 > 0 else 0)
            
            merged_data.append({
                'SPU': spu,
                '2026营收($)': rev_2026,
                '2025营收($)': rev_2025,
                '营收YOY': rev_yoy,
                '2026销量': qty_2026,
                '2025销量': qty_2025,
                '销量YOY': qty_yoy,
                '2026 ASP': asp_2026,
                '2025 ASP': asp_2025,
                'ASP YOY': asp_yoy
            })
        
        # 按2026营收排序取Top 10
        merged_data.sort(key=lambda x: x['2026营收($)'], reverse=True)
        top10 = merged_data[:10]
        
        # 计算总结数据
        if top10:
            top3_spus = [d['SPU'] for d in top10[:3]]
            top3_str = "/".join(top3_spus)
            
            # 检查是否有新品（2025营收为0的）
            new_products = [d['SPU'] for d in top10 if d['2025营收($)'] == 0]
            new_product_str = ""
            if new_products:
                new_product_str = f"，新品{new_products[0]}快速起量"
            
            st.caption(f"💡 {top3_str}为TOP3{new_product_str}")
        
        # 显示表格
        if top10:
            df_display = pd.DataFrame(top10)
            
            # 格式化显示函数（>20%绿色、<-20%红色，其他黑色；"新增"改为"/"）
            def format_yoy(x):
                if x == float('inf'):
                    return "/"
                elif x > 20:
                    return f"<span style='color: #16a34a;'>↑{x:.1f}%</span>"
                elif x < -20:
                    return f"<span style='color: #dc2626;'>↓{abs(x):.1f}%</span>"
                elif x > 0:
                    return f"↑{x:.1f}%"
                elif x < 0:
                    return f"↓{abs(x):.1f}%"
                else:
                    return "-"
            
            # 格式化数值列
            df_display['2026营收($)'] = df_display['2026营收($)'].apply(lambda x: f"${x:,.0f}")
            df_display['2025营收($)'] = df_display['2025营收($)'].apply(lambda x: f"${x:,.0f}" if x > 0 else "-")
            df_display['营收YOY'] = df_display['营收YOY'].apply(format_yoy)
            df_display['2026销量'] = df_display['2026销量'].apply(lambda x: f"{x:,.0f}")
            df_display['2025销量'] = df_display['2025销量'].apply(lambda x: f"{x:,.0f}" if x > 0 else "-")
            df_display['销量YOY'] = df_display['销量YOY'].apply(format_yoy)
            df_display['2026 ASP'] = df_display['2026 ASP'].apply(lambda x: f"${x:.2f}")
            df_display['2025 ASP'] = df_display['2025 ASP'].apply(lambda x: f"${x:.2f}" if x > 0 else "-")
            df_display['ASP YOY'] = df_display['ASP YOY'].apply(format_yoy)
            
            # 使用HTML显示带颜色的表格，营收列不加粗
            html_table = df_display.to_html(escape=False, index=False)
            # 为营收列添加正常字重样式
            html_table = html_table.replace('<th>2026营收($)</th>', '<th style="font-weight: normal;">2026营收($)</th>')
            html_table = html_table.replace('<th>2025营收($)</th>', '<th style="font-weight: normal;">2025营收($)</th>')
            st.write(html_table, unsafe_allow_html=True)
            
            # 显示映射统计
            total_skus = sku_mapped + sku_unmapped
            st.caption(f"📋 映射统计：共{total_skus}个SKU，成功关联SPU{sku_mapped}个({sku_mapped/total_skus*100:.1f}%)，未关联{sku_unmapped}个")
            
            # 显示未关联详情（如果存在）
            if "未关联" in [d['SPU'] for d in merged_data]:
                unclassified_revenue = spu_data_2026.get('未关联', {}).get('营收', 0)
                if unclassified_revenue > 0:
                    st.caption(f"⚠️ 未关联SPU的SKU共{unclassified_revenue:,.0f}营收，建议检查SPU-SKU映射表配置")
        else:
            st.info("暂无SPU数据")
            
    except Exception as e:
        st.error(f"SPU对比分析失败: {e}")
        import traceback
        traceback.print_exc()


def render_traffic_funnel_summary(traffic_data):
    """流量漏斗与转化效率 - 表格形式展示"""
    st.markdown("### ④ 流量漏斗与转化效率")
    
    if traffic_data is None or 'current' not in traffic_data:
        st.info("💡 未上传流量数据，请在侧边栏上传「流量和出单占比」文件")
        return
    
    current = traffic_data.get('current', {}).get('全店', {})
    last = traffic_data.get('last', {}).get('全店', {}) if traffic_data.get('last') else {}
    
    # 获取指标数据
    curr_impression = current.get('Product Impression', 0)
    curr_pv = current.get('PV', 0)
    curr_ctr = current.get('CTR', 0)
    curr_cvr = current.get('CVR', 0)
    
    last_impression = last.get('Product Impression', 0)
    last_pv = last.get('PV', 0)
    last_cvr = last.get('CVR', 0)
    
    # 计算环比变化
    imp_change = (curr_impression - last_impression) / last_impression if last_impression > 0 else 0
    pv_change = (curr_pv - last_pv) / last_pv if last_pv > 0 else 0
    cvr_change_pp = (curr_cvr - last_cvr) * 100
    
    # 【流量获取漏斗】表格形式
    st.markdown("**【流量获取漏斗】**")
    
    # 添加总结
    imp_change_str = f"环比+{imp_change:.0%}" if imp_change > 0 else f"环比{imp_change:.0%}"
    st.caption(f"💡 Impression{imp_change_str}，但CTR {curr_ctr:.2%}仍有优化空间")
    
    funnel_data = [
        {
            '层级': 'Product Impressions\n（产品展现量）',
            '数值': f"{curr_impression:,.0f}",
            '环比': f"{'↑' if imp_change >= 0 else '↓'} {abs(imp_change):.1%}"
        },
        {
            '层级': 'CTR（点击率）',
            '数值': f"{curr_ctr:.2%}",
            '环比': '-'
        },
        {
            '层级': 'Page Views\n（页面浏览量）',
            '数值': f"{curr_pv:,.0f}",
            '环比': f"{'↑' if pv_change >= 0 else '↓'} {abs(pv_change):.1%}"
        },
        {
            '层级': 'CVR（访客→购买转化率）',
            '数值': f"{curr_cvr:.2%}",
            '环比': f"{'↑' if cvr_change_pp >= 0 else '↓'} {abs(cvr_change_pp):.2f}pp"
        }
    ]
    
    df_funnel = pd.DataFrame(funnel_data)
    st.dataframe(df_funnel, use_container_width=True, hide_index=True)


def render_channel_structure(traffic_data):
    """出单渠道结构 - 饼图展示"""
    st.markdown("### ⑤ 出单渠道结构")
    
    if traffic_data is None or 'current' not in traffic_data:
        st.info("💡 未上传流量数据")
        return
    
    current = traffic_data.get('current', {})
    last = traffic_data.get('last', {}) if traffic_data.get('last') else {}
    
    # 获取各渠道GMV占比数据
    channels = ['短视频', '商品卡', '直播']
    channel_data = []
    
    for ch in channels:
        if ch in current:
            curr_gmv_ratio = current[ch].get('GMV占比', 0)
            last_gmv_ratio = last.get(ch, {}).get('GMV占比', 0) if last else 0
            change_pp = (curr_gmv_ratio - last_gmv_ratio) * 100
            channel_data.append({
                '渠道': ch,
                'GMV占比': curr_gmv_ratio,
                '环比变化pp': change_pp
            })
    
    if not channel_data:
        st.info("暂无渠道结构数据")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 【GMV占比】饼图
        st.markdown("**【GMV占比】本周期**")
        
        # 计算总结
        short_video_ratio = 0
        short_video_change = 0
        for d in channel_data:
            if d['渠道'] == '短视频':
                short_video_ratio = d['GMV占比'] * 100
                short_video_change = d['环比变化pp']
                break
        
        trend_str = "微有下滑" if short_video_change < 0 else "稳步提升"
        st.caption(f"💡 短视频占比{short_video_ratio:.0f}%为主力，{trend_str}")
        
        try:
            import plotly.express as px
            df_pie = pd.DataFrame(channel_data)
            colors = ['#0ea5e9', '#22c55e', '#f59e0b']  # 蓝、绿、橙
            fig = px.pie(df_pie, values='GMV占比', names='渠道', 
                        color_discrete_sequence=colors,
                        hole=0.4)  # 环形图
            fig.update_traces(textinfo='label+percent', textfont_size=12)
            fig.update_layout(height=300, showlegend=True, 
                            legend=dict(orientation='h', yanchor='bottom', y=-0.1))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            # 如果plotly失败，显示文字版
            for d in channel_data:
                st.markdown(f"**{d['渠道']}**: {d['GMV占比']:.1%}")
    
    with col2:
        # 【环比变化】
        st.markdown("**【环比变化】**")
        
        for d in channel_data:
            ch = d['渠道']
            change = d['环比变化pp']
            change_str = f"{'↑' if change >= 0 else '↓'} {abs(change):.1f}pp"
            change_color = "#16a34a" if change >= 0 else "#dc2626"
            
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; 
                        padding: 10px; margin: 6px 0; background: #f8fafc; border-radius: 6px;">
                <span style="font-weight: 500; color: #475569;">{ch}</span>
                <span style="color: {change_color}; font-weight: 600;">{change_str}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # 【流量→GMV效率对比】- 确保显示三个渠道
    st.markdown("**【流量→GMV效率对比】**")
    
    # 添加总结
    st.caption(f"💡 商品卡转化效率较高，短视频转化效率待加强")
    
    efficiency_insights = []
    for ch in channels:
        if ch in current:
            pv_ratio = current[ch].get('流量贡献占比 (PV)', 0)
            gmv_ratio = current[ch].get('GMV占比', 0)
            
            if pv_ratio > 0:
                efficiency = gmv_ratio / pv_ratio if pv_ratio > 0 else 0
                
                if efficiency > 1.5:
                    insight = f"<b>{ch}</b>：PV占{pv_ratio:.0%}→GMV占{gmv_ratio:.0%}，转化效率<b style='color:#16a34a'>高</b>（{efficiency:.1f}x）"
                elif efficiency < 0.7:
                    insight = f"<b>{ch}</b>：PV占{pv_ratio:.0%}→GMV占{gmv_ratio:.0%}，流量多但转化待提升（{efficiency:.1f}x）"
                else:
                    insight = f"<b>{ch}</b>：PV占{pv_ratio:.0%}→GMV占{gmv_ratio:.0%}，转化效率正常（{efficiency:.1f}x）"
                
                efficiency_insights.append(insight)
    
    if efficiency_insights:
        st.markdown(f"""
        <div style="padding: 12px; background: #f8fafc; border-radius: 6px; font-size: 13px; color: #475569; line-height: 2;">
            {'<br>'.join(efficiency_insights)}
        </div>
        """, unsafe_allow_html=True)


def render_pv_asset_tracking(traffic_data):
    """PV人群资产"""
    st.markdown("### ⑥ PV人群资产")
    
    if traffic_data is None or 'current' not in traffic_data:
        st.info("💡 未上传流量数据")
        return
    
    current = traffic_data.get('current', {}).get('全店', {})
    last = traffic_data.get('last', {}).get('全店', {}) if traffic_data.get('last') else {}
    
    # 本周期PV
    curr_pv = current.get('PV', 0)
    last_pv = last.get('PV', 0) if last else 0
    pv_change = (curr_pv - last_pv) / last_pv if last_pv > 0 else 0
    
    # 年度总计PV = 2026年1-3月累加
    # 数据行：第1行=1月，第2行=2月，第3行=3月
    yearly_total = curr_pv  # 默认至少是本周期
    try:
        import pandas as pd
        df_traffic = pd.read_excel('流量和出单占比.xlsx')
        pv_col = df_traffic.columns[2]  # 全店Page views列
        
        # 累加1-3月：第1行(1月) + 第2行(2月) + 第3行(3月)
        if len(df_traffic) > 3:
            # 有1月、2月、3月数据
            jan_pv = int(df_traffic.iloc[1][pv_col])  # 第1行 = 1月
            feb_pv = int(df_traffic.iloc[2][pv_col])  # 第2行 = 2月
            mar_pv = int(df_traffic.iloc[3][pv_col])  # 第3行 = 3月
            yearly_total = jan_pv + feb_pv + mar_pv
        elif len(df_traffic) > 2:
            # 只有1月、2月数据
            jan_pv = int(df_traffic.iloc[1][pv_col])  # 第1行 = 1月
            feb_pv = int(df_traffic.iloc[2][pv_col])  # 第2行 = 2月
            yearly_total = jan_pv + feb_pv
        else:
            yearly_total = curr_pv
    except Exception as e:
        yearly_total = curr_pv
    
    # 三个指标卡片
    st.markdown("""
    <style>
    .pv-card {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        border: 1px solid #bae6fd;
    }
    .pv-card .label {
        font-size: 12px;
        color: #0369a1;
        margin-bottom: 8px;
        font-weight: 500;
    }
    .pv-card .value {
        font-size: 24px;
        font-weight: bold;
        color: #0c4a6e;
        margin-bottom: 4px;
    }
    .pv-card .change {
        font-size: 12px;
        color: #64748b;
    }
    .pv-card .change.up {
        color: #16a34a;
    }
    .pv-card .change.down {
        color: #dc2626;
    }
    </style>
    """, unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    
    with c1:
        change_class = "up" if pv_change >= 0 else "down"
        change_icon = "↑" if pv_change >= 0 else "↓"
        st.markdown(f"""
        <div class="pv-card">
            <div class="label">本周期PV</div>
            <div class="value">{curr_pv:,.0f}</div>
            <div class="change {change_class}">{change_icon} {abs(pv_change):.1%} 环比</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"""
        <div class="pv-card">
            <div class="label">年度总计</div>
            <div class="value">{yearly_total:,.0f}</div>
            <div class="change">2026年累计</div>
        </div>
        """, unsafe_allow_html=True)
    
    with c3:
        st.markdown(f"""
        <div class="pv-card">
            <div class="label">月度目标达成</div>
            <div class="value">待配置</div>
            <div class="change">（后续提供目标）</div>
        </div>
        """, unsafe_allow_html=True)


def render_weekly_top3_issues(df_pl, df_prod_ads, creator_stats):
    """重点问题 Top 3"""
    st.markdown("### ⑥ 重点问题 Top 3")
    
    problems = []
    
    if df_pl is not None and not df_pl.empty:
        worst_profit = df_pl.nsmallest(1, '利润额')
        if not worst_profit.empty:
            row = worst_profit.iloc[0]
            problems.append({
                'rank': 1,
                'type': '🔴 利润亏损',
                'spu': row['SPU'],
                'value': f"${row['利润额']:,.0f}",
                'cause': row.get('主因', '利润为负'),
                'action': row.get('建议动作', '止损处理')
            })
    
    if df_prod_ads is not None and not df_prod_ads.empty and 'Status' in df_prod_ads.columns:
        loss_ads = df_prod_ads[df_prod_ads['Status'].astype(str).str.contains("亏损")]
        if not loss_ads.empty:
            worst_ads = loss_ads.nlargest(1, 'Cost')
            if not worst_ads.empty:
                row = worst_ads.iloc[0]
                problems.append({
                    'rank': 2,
                    'type': '🔴 广告拖累',
                    'spu': row.get('SPU', '未知'),
                    'value': f"${row['Cost']:,.0f} 花费",
                    'cause': row.get('Diagnosis', '广告亏损'),
                    'action': row.get('Action', '优化素材或止损')
                })
    
    if creator_stats is not None and not creator_stats.empty:
        if 'RiskScore' in creator_stats.columns:
            worst_creator = creator_stats.nlargest(1, 'RiskScore')
            if not worst_creator.empty:
                row = worst_creator.iloc[0]
                problems.append({
                    'rank': 3,
                    'type': '🟠 达人风险',
                    'spu': row['SPU'],
                    'value': row.get('RiskReason', '达人集中'),
                    'cause': row.get('RiskReason', 'Top1占比过高'),
                    'action': row.get('建议动作', '扩达人池')
                })
    
    for p in problems:
        bg_color = '#fef2f2' if '🔴' in p['type'] else '#fff7ed'
        border_color = '#dc2626' if '🔴' in p['type'] else '#ea580c'
        st.markdown(f'''
        <div style="padding: 12px; margin: 8px 0; background: {bg_color}; 
                    border-left: 4px solid {border_color}; border-radius: 4px;">
            <div style="font-weight: 600; font-size: 16px;">#{p['rank']} {p['type']} | {p['spu']}</div>
            <div style="margin-top: 4px; color: #666;">影响：{p['value']}</div>
            <div style="margin-top: 4px;">主因：{p['cause']}</div>
            <div style="margin-top: 4px; color: #0284c7;">动作：{p['action']}</div>
        </div>
        ''', unsafe_allow_html=True)


def render_weekly_resource_request(df_pl, df_prod_ads, targets):
    """资源需求（占位版）"""
    st.markdown("### ⑦ 资源需求")
    
    reqs = []
    
    if df_pl is not None and not df_pl.empty and '决策篮子' in df_pl.columns:
        a_cnt = int((df_pl['决策篮子'].astype(str).str.startswith("A")).sum())
        if a_cnt > 0:
            reqs.append(f"💰 **预算需求**：{a_cnt} 个A清单SPU建议追加预算扩量")
        
        c_cnt = int((df_pl['决策篮子'].astype(str).str.startswith("C")).sum())
        if c_cnt > 0:
            reqs.append(f"👥 **人力需求**：{c_cnt} 个C清单SPU需协同止损处理")
    
    if reqs:
        for r in reqs:
            st.markdown(f"- {r}")
    else:
        st.info("本周期暂无特殊资源需求")


def render_weekly_next_actions(df_pl, top_n=5):
    """下周关键动作"""
    st.markdown("### ⑧ 下周关键动作")
    
    if df_pl is None or df_pl.empty:
        st.info("暂无动作数据")
        return
    
    if '决策篮子' not in df_pl.columns:
        st.info("暂无决策篮子数据，请先访问 SPU 分析页面生成数据")
        return
    
    actions = []
    
    c_list = df_pl[df_pl['决策篮子'].astype(str).str.startswith("C")].sort_values('利润额').head(2)
    for _, row in c_list.iterrows():
        actions.append({
            'priority': 'P0',
            'color': '#dc2626',
            'task': f"【C止损】{row['SPU']}",
            'action': str(row.get('建议动作', '止损处理'))[:30] + "..."
        })
    
    b_list = df_pl[df_pl['决策篮子'].astype(str).str.startswith("B")].sort_values('退款后营收', ascending=False).head(2)
    for _, row in b_list.iterrows():
        actions.append({
            'priority': 'P1',
            'color': '#ea580c',
            'task': f"【B修复】{row['SPU']}",
            'action': str(row.get('建议动作', '优化修复'))[:30] + "..."
        })
    
    a_list = df_pl[df_pl['决策篮子'].astype(str).str.startswith("A")].sort_values('利润额', ascending=False).head(1)
    for _, row in a_list.iterrows():
        actions.append({
            'priority': 'P2',
            'color': '#16a34a',
            'task': f"【A增投】{row['SPU']}",
            'action': str(row.get('建议动作', '扩量投放'))[:30] + "..."
        })
    
    for a in actions[:top_n]:
        st.markdown(f'''
        <div style="padding: 10px; margin: 6px 0; background: #f8fafc; border-radius: 4px; 
                    border-left: 4px solid {a['color']}; display: flex; align-items: center;">
            <span style="background: {a['color']}; color: white; padding: 2px 8px; border-radius: 4px; 
                        font-size: 12px; font-weight: 600; margin-right: 10px;">{a['priority']}</span>
            <span style="font-weight: 500;">{a['task']}</span>
            <span style="margin-left: 10px; color: #666; font-size: 14px;">{a['action']}</span>
        </div>
        ''', unsafe_allow_html=True)


def render_weekly_overview_page(out, meta, targets):
    """周会概览页面主入口"""
    st.markdown("## 🏢 周会概览")
    
    time_str = meta.get("time_str", "未知周期")
    df_shop_raw = out.get('df_shop_raw')
    df_spu_raw = out.get('df_spu_raw')
    df_sku_raw = out.get('df_sku_raw')
    df_prod_ads = out.get('df_prod_ads')
    creator_data = out.get('creator_data')
    traffic_data = out.get('traffic_data')
    
    # SPU决策表df_pl
    if 'df_pl' in st.session_state:
        df_pl = st.session_state['df_pl']
    else:
        df_pl = df_spu_raw.copy() if df_spu_raw is not None else None
    
    st.caption(f"📅 数据周期：{time_str}")
    
    # ① 经营结论
    shop_metrics = {
        '退款后营收': df_shop_raw.iloc[0].get('退款后营收', 0) if not df_shop_raw.empty else 0,
        '利润率': df_shop_raw.iloc[0].get('利润率', 0) if not df_shop_raw.empty else 0
    }
    spu_metrics = {
        'profit_cnt': int((df_spu_raw['利润额'] >= 0).sum()) if df_spu_raw is not None else 0,
        'loss_cnt': int((df_spu_raw['利润额'] < 0).sum()) if df_spu_raw is not None else 0
    }
    render_weekly_summary_card(time_str, shop_metrics, spu_metrics, traffic_data, targets, df_shop_raw)
    
    # ② 核心指标
    render_weekly_kpi_section(df_shop_raw, df_spu_raw, traffic_data, targets)
    
    # ③ 经营偏差分析
    render_weekly_deviation_analysis(df_shop_raw, traffic_data)
    
    # ④ 流量漏斗
    render_traffic_funnel_summary(traffic_data)
    
    # ⑤ 渠道结构
    render_channel_structure(traffic_data)
    
    # ⑥ PV人群资产追踪
    render_pv_asset_tracking(traffic_data)
    
    # ⑦ 重点问题Top3
    creator_stats = creator_data.get('spu_perf') if creator_data else None
    render_weekly_top3_issues(df_pl, df_prod_ads, creator_stats)
    
    # ⑧ 资源需求
    render_weekly_resource_request(df_pl, df_prod_ads, targets)
    
    # ⑨ 下周动作
    render_weekly_next_actions(df_pl, top_n=5)
    
    # ⑩ 重点项目（占位）
    st.markdown("### ⑩ 重点项目摘要")
    st.info("📝 预留区域：可放置大促准备、新品上线、重点项目进度等信息")



if __name__ == '__main__':
    main()
