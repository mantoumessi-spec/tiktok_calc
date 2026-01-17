import streamlit as st
import pandas as pd
import numpy as np
import io

# ================= 🎨 界面美化配置 =================
def set_style():
    # 1. 设置页面基础信息
    st.set_page_config(
        page_title="华青-TikTok 利润测算系统",
        page_icon="💰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # 2. 注入自定义 CSS 样式
    st.markdown("""
        <style>
        /* 全局字体优化 */
        html, body, [class*="css"] {
            font-family: 'PingFang SC', 'Microsoft YaHei', 'Helvetica Neue', sans-serif;
        }

        /* === 主背景设置 (柔和的渐变灰蓝 - 商务风) === */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }

        /* === 侧边栏样式优化 === */
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e0e0e0;
            box-shadow: 2px 0 5px rgba(0,0,0,0.05);
        }
        
        /* === 侧边栏标题 === */
        section[data-testid="stSidebar"] h2 {
            color: #2c3e50;
            font-weight: 600;
        }

        /* === 主标题 (H1) 样式 === */
        h1 {
            color: #2c3e50;
            text-align: center;
            font-weight: 700;
            padding-bottom: 15px;
            border-bottom: 3px solid #3498db;
            margin-bottom: 25px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }

        /* === 按钮样式 (TikTok 品牌红) === */
        div.stButton > button:first-child {
            background-color: #ff0050; 
            color: white;
            border-radius: 6px;
            border: none;
            padding: 12px 24px;
            font-size: 16px;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            width: 100%;
        }
        div.stButton > button:first-child:hover {
            background-color: #d60043;
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }

        /* === 上传组件样式 === */
        [data-testid="stFileUploader"] {
            background-color: #fcfcfc;
            padding: 10px;
            border-radius: 8px;
            border: 1px dashed #b0b0b0;
        }

        /* === 数据表格卡片化 === */
        .stDataFrame {
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            margin-bottom: 20px; /* 增加表格间距 */
        }
        
        /* === 成功/错误提示框美化 === */
        .stAlert {
            border-radius: 8px;
            border: 1px solid rgba(0,0,0,0.05);
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* === 下载按钮特别样式 === */
        div.stDownloadButton > button {
            background-color: #27ae60 !important;
            color: white !important;
            border-radius: 6px;
            font-weight: bold;
        }

        /* === 分割线样式 === */
        hr {
            margin-top: 30px;
            margin-bottom: 30px;
            border: 0;
            border-top: 1px solid #dcdcdc;
        }
        </style>
    """, unsafe_allow_html=True)

# ================= 配置区域 =================
EXCHANGE_RATE = 1 / 7.15 

# 目标表头配置
TARGET_COLUMNS_SKU = [
    'SPU', 'SKU', '类别', '销量', '营收', '利润率', '利润额', 'ASP',
    '单件采购成本', '单件头程', '单件关税', '单件尾程', '单件仓租', 
    '单件其他物流成本', '单件品牌费用', '单件平台佣金', 
    '退款单数', '退款率', 
    '单件其他和售后', '单件达人佣金', '单件样品费', '单件广告投放费', 
    '总广告投放费', 
    '采购成本-占比', '头程-占比', '关税占比', '尾程-占比', 
    '仓租-占比', '其他物流成本-占比', '品牌费用-占比', '平台佣金-占比', 
    '退款-占比', '其他和售后-占比', '达人佣金-占比', '样品费-占比', 
    '广告投放费-占比', '总营销费比'
]
TARGET_COLUMNS_SPU = [col for col in TARGET_COLUMNS_SKU if col != 'SKU']
TARGET_COLUMNS_SHOP = [col for col in TARGET_COLUMNS_SPU if col not in ['SPU', '类别']]

# ================= 辅助函数 =================
def clean_text(df, col_name):
    if col_name in df.columns:
        return df[col_name].astype(str).str.strip()
    return df[col_name]

def find_col_by_keyword(df, keywords):
    for col in df.columns:
        if all(k in col for k in keywords):
            return col
    return None

def get_cost_map(cost_df, keywords):
    if cost_df is None: return {}
    target_col = find_col_by_keyword(cost_df, keywords)
    if not target_col: return {}
    cost_df['SKU'] = clean_text(cost_df, 'SKU')
    cost_df['USD'] = pd.to_numeric(cost_df[target_col], errors='coerce').fillna(0) * EXCHANGE_RATE
    return dict(zip(cost_df['SKU'], cost_df['USD']))

def format_dataframe(df, target_columns):
    df_out = df.reindex(columns=target_columns, fill_value=0)
    pct_columns = [
        '利润率', '退款率', '总营销费比', '采购成本-占比', '头程-占比', '关税占比', '尾程-占比', 
        '仓租-占比', '其他物流成本-占比', '品牌费用-占比', '平台佣金-占比', '退款-占比', 
        '其他和售后-占比', '达人佣金-占比', '样品费-占比', '广告投放费-占比'
    ]
    numeric_cols = df_out.select_dtypes(include=[np.number]).columns
    money_cols = [c for c in numeric_cols if c not in pct_columns]
    df_out[money_cols] = df_out[money_cols].fillna(0).round(2)
    for col in pct_columns:
        if col in df_out.columns:
            df_out[col] = df_out[col].fillna(0).apply(lambda x: f"{x:.2%}")
    return df_out

def calculate_metrics(df_base):
    df = df_base.copy()
    qty = df['销量'].replace(0, 1)
    rev = df['营收'].replace(0, 1)
    
    # 单件计算
    cost_cols = {
        '单件采购成本': '采购成本', '单件头程': '头程', '单件尾程': '尾程',
        '单件关税': '关税', '单件仓租': '仓租', '单件其他物流成本': '其他物流成本',
        '单件品牌费用': '品牌费用', '单件平台佣金': '平台佣金', 
        '单件其他和售后': '其他和售后', '单件达人佣金': '达人佣金', 
        '单件样品费': '样品费', '单件广告投放费': '总广告投放费'
    }
    for unit_col, total_col in cost_cols.items():
        if total_col in df.columns: df[unit_col] = df[total_col] / qty
        else: df[unit_col] = 0

    df['ASP'] = df['营收'] / qty
    total_orders = (df['Valid_Orders'] + df['Refund_Orders']).replace(0, 1)
    df['退款率'] = df['Refund_Orders'] / total_orders
    df['退款单数'] = df['Refund_Orders']

    ratio_cols = {
        '采购成本-占比': '采购成本', '头程-占比': '头程', '尾程-占比': '尾程',
        '关税占比': '关税', '仓租-占比': '仓租', '其他物流成本-占比': '其他物流成本',
        '品牌费用-占比': '品牌费用', '平台佣金-占比': '平台佣金', 
        '其他和售后-占比': '其他和售后', '达人佣金-占比': '达人佣金', 
        '样品费-占比': '样品费', '广告投放费-占比': '总广告投放费',
        '退款-占比': '退款'
    }
    for r_col, t_col in ratio_cols.items():
        if t_col in df.columns: df[r_col] = df[t_col] / rev
        else: df[r_col] = 0

    mkt_cost = df['总广告投放费'] + df['达人佣金'] + df['样品费']
    df['总营销费比'] = mkt_cost / rev

    all_costs = sum(df[c] for c in [
        '采购成本', '头程', '尾程', '关税', '仓租', '其他物流成本', 
        '品牌费用', '平台佣金', '其他和售后', '达人佣金', '样品费', '总广告投放费'
    ] if c in df.columns)
    df['利润额'] = df['营收'] - all_costs
    df['利润率'] = df['利润额'] / rev
    return df

# ================= 主程序 =================
def main():
    set_style() # 应用美化

    st.title("📊 TikTok 利润测算系统")
    st.markdown("""
    <div style='text-align: center; color: #555; margin-bottom: 25px; font-size: 14px;'>
        🚀 专为团队打造的财务分析神器 | 支持 <b>SKU / SPU / 店铺</b> 全维度透视 | 自动清洗脏数据
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### 📂 第一步：上传原始数据")
        st.info("请确保文件名包含关键字（如'订单'、'广告'等）")
        f_orders = st.file_uploader("1. 订单表 (OrderSKUList)", type=['xlsx', 'csv'])
        f_ads = st.file_uploader("2. 广告表", type=['xlsx', 'csv'])
        f_mapping = st.file_uploader("3. 映射表 (ID关联)", type=['xlsx', 'csv'])
        f_affiliate = st.file_uploader("4. 联盟订单", type=['xlsx', 'csv'])
        
        st.markdown("---")
        st.markdown("### 📂 第二步：上传成本表")
        f_purchase = st.file_uploader("5. 采购成本", type=['xlsx', 'csv'])
        f_head = st.file_uploader("6. 头程成本", type=['xlsx', 'csv'])
        f_tail = st.file_uploader("7. 尾程成本", type=['xlsx', 'csv'])

    if st.button("🚀 开始全自动测算", type="primary"):
        if not f_orders:
            st.error("❌ 无法开始：请至少上传【订单表】！")
        else:
            with st.spinner("⏳ 正在清洗数据、匹配广告费、分摊成本... 请稍候"):
                try:
                    # 读取与清洗
                    df_orders = pd.read_excel(f_orders) if f_orders.name.endswith('xlsx') else pd.read_csv(f_orders)
                    df_ads = pd.read_excel(f_ads) if f_ads and f_ads.name.endswith('xlsx') else pd.read_csv(f_ads) if f_ads else None
                    df_mapping = pd.read_excel(f_mapping) if f_mapping and f_mapping.name.endswith('xlsx') else pd.read_csv(f_mapping) if f_mapping else None
                    df_affiliate = pd.read_excel(f_affiliate) if f_affiliate and f_affiliate.name.endswith('xlsx') else pd.read_csv(f_affiliate) if f_affiliate else None
                    df_purchase = pd.read_excel(f_purchase) if f_purchase and f_purchase.name.endswith('xlsx') else pd.read_csv(f_purchase) if f_purchase else None
                    df_head = pd.read_excel(f_head) if f_head and f_head.name.endswith('xlsx') else pd.read_csv(f_head) if f_head else None
                    df_tail = pd.read_excel(f_tail) if f_tail and f_tail.name.endswith('xlsx') else pd.read_csv(f_tail) if f_tail else None

                    for df in [df_orders, df_ads, df_mapping, df_affiliate, df_purchase, df_head, df_tail]:
                        if df is not None: df.columns = df.columns.str.strip()

                    # === 逻辑处理 ===
                    if 'Order ID' in df_orders.columns:
                        df_orders = df_orders[~df_orders['Order ID'].astype(str).str.contains('Platform|Order ID', na=False)]

                    col_sku = 'Seller SKU' if 'Seller SKU' in df_orders.columns else 'SKU'
                    col_rev = '营收'
                    col_qty = 'Quantity'
                    col_status = 'Order Status'
                    
                    df_orders[col_rev] = pd.to_numeric(df_orders[col_rev], errors='coerce').fillna(0)
                    df_orders[col_qty] = pd.to_numeric(df_orders.get(col_qty, 1), errors='coerce').fillna(0)
                    df_orders[col_sku] = clean_text(df_orders, col_sku) 
                    df_orders[col_status] = df_orders[col_status].astype(str).str.strip()

                    is_cancelled = df_orders[col_status].isin(['Cancelled', 'Canceled'])
                    is_sample = (df_orders[col_rev] == 0) & (~is_cancelled)
                    is_normal = (df_orders[col_rev] > 0) & (~is_cancelled)

                    df_normal = df_orders[is_normal].copy()
                    df_sample = df_orders[is_sample].copy()
                    df_refund = df_orders[is_cancelled].copy()

                    map_purchase = get_cost_map(df_purchase, ['采购', 'CNY'])
                    map_head = get_cost_map(df_head, ['头程', 'CNY'])
                    map_tail = get_cost_map(df_tail, ['尾程', 'CNY'])

                    sku_stats = df_normal.groupby(['SPU', col_sku]).agg({
                        col_rev: 'sum', col_qty: 'sum', 'Product Name': 'first', 'Order ID': 'nunique'
                    }).reset_index().rename(columns={col_sku: 'SKU', col_qty: '销量', col_rev: '营收', 'Order ID': 'Valid_Orders'})

                    # 样品费
                    if not df_sample.empty:
                        df_sample['Unit_Purchase'] = df_sample[col_sku].map(map_purchase).fillna(0)
                        df_sample['Unit_Head'] = df_sample[col_sku].map(map_head).fillna(0)
                        df_sample['Sample_Cost'] = df_sample[col_qty] * (df_sample['Unit_Purchase'] + df_sample['Unit_Head'])
                        sample_agg = df_sample.groupby(['SPU', col_sku])['Sample_Cost'].sum().reset_index().rename(columns={col_sku: 'SKU', 'Sample_Cost': '样品费'})
                        sku_stats = pd.merge(sku_stats, sample_agg, on=['SPU', 'SKU'], how='left')
                    else: sku_stats['样品费'] = 0
                    sku_stats['样品费'] = sku_stats['样品费'].fillna(0)

                    # 退款
                    if not df_refund.empty:
                        refund_stats = df_refund.groupby(['SPU', col_sku]).agg({col_rev: 'sum', 'Order ID': 'nunique'}).reset_index().rename(columns={col_sku: 'SKU', col_rev: '退款', 'Order ID': 'Refund_Orders'})
                        sku_stats = pd.merge(sku_stats, refund_stats, on=['SPU', 'SKU'], how='left')
                    else:
                        sku_stats['退款'] = 0; sku_stats['Refund_Orders'] = 0
                    sku_stats[['退款', 'Refund_Orders']] = sku_stats[['退款', 'Refund_Orders']].fillna(0)

                    # 成本
                    sku_stats['采购成本'] = sku_stats.apply(lambda x: map_purchase.get(x['SKU'], 0) * x['销量'], axis=1)
                    sku_stats['头程'] = sku_stats.apply(lambda x: map_head.get(x['SKU'], 0) * x['销量'], axis=1)
                    sku_stats['尾程'] = sku_stats.apply(lambda x: map_tail.get(x['SKU'], 0) * x['销量'], axis=1)
                    sku_stats['平台佣金'] = sku_stats['营收'] * 0.06

                    # 达人佣金
                    if df_affiliate is not None:
                        df_affiliate['Order ID'] = clean_text(df_affiliate, 'Order ID')
                        df_normal['Order ID'] = clean_text(df_normal, 'Order ID')
                        c1, c2 = 'Est. standard commission payment', 'Est. Shop Ads commission payment'
                        for c in [c1, c2]: df_affiliate[c] = pd.to_numeric(df_affiliate.get(c, 0), errors='coerce').fillna(0)
                        comm_map = df_affiliate.groupby('Order ID')[[c1, c2]].sum().sum(axis=1)
                        df_normal['Comm'] = df_normal['Order ID'].map(comm_map).fillna(0)
                        aff_sum = df_normal.groupby(['SPU', col_sku])['Comm'].sum().reset_index().rename(columns={col_sku: 'SKU', 'Comm': '达人佣金'})
                        sku_stats = pd.merge(sku_stats, aff_sum, on=['SPU', 'SKU'], how='left')
                    else: sku_stats['达人佣金'] = 0
                    sku_stats['达人佣金'] = sku_stats['达人佣金'].fillna(0)

                    # 广告费
                    if df_ads is not None:
                        p_map = dict(zip(clean_text(df_mapping, 'Product ID'), clean_text(df_mapping, 'Product Name')))
                        spu_map = dict(zip(clean_text(df_orders, 'Product Name'), clean_text(df_orders, 'SPU')))
                        df_ads['SPU'] = clean_text(df_ads, 'Product ID').map(p_map).map(spu_map).fillna('Unknown')
                        df_ads['Cost'] = pd.to_numeric(df_ads['Cost'], errors='coerce').fillna(0)
                        spu_ads = df_ads.groupby('SPU')['Cost'].sum()
                        spu_rev = sku_stats.groupby('SPU')['营收'].sum()
                        sku_stats = pd.merge(sku_stats, spu_ads.rename('SPU_Ads'), on='SPU', how='left')
                        sku_stats = pd.merge(sku_stats, spu_rev.rename('SPU_Rev'), on='SPU', how='left')
                        sku_stats['总广告投放费'] = sku_stats.apply(lambda x: x['SPU_Ads'] * (x['营收']/x['SPU_Rev']) if x['SPU_Rev'] > 0 else 0, axis=1).fillna(0)
                    else: sku_stats['总广告投放费'] = 0

                    sku_stats['仓租'] = sku_stats['营收'] * 0.005
                    sku_stats['其他物流成本'] = sku_stats['营收'] * 0.003
                    sku_stats['品牌费用'] = sku_stats['营收'] * 0.003
                    sku_stats['其他和售后'] = sku_stats['营收'] * 0.003
                    sku_stats['关税'] = 0; sku_stats['类别'] = ''

                    # 生成三大报表
                    df_sku_final = calculate_metrics(sku_stats)
                    df_sku_out = format_dataframe(df_sku_final, TARGET_COLUMNS_SKU)

                    sum_cols = ['销量', '营收', 'Valid_Orders', 'Refund_Orders', '退款', '采购成本', '头程', '尾程', '关税', '仓租', '其他物流成本', '品牌费用', '平台佣金', '其他和售后', '达人佣金', '样品费', '总广告投放费']
                    spu_agg = sku_stats.groupby('SPU').agg({**{c: 'sum' for c in sum_cols}, '类别': 'first'}).reset_index()
                    df_spu_final = calculate_metrics(spu_agg).sort_values(by='营收', ascending=False)
                    df_spu_out = format_dataframe(df_spu_final, TARGET_COLUMNS_SPU)

                    shop_agg = sku_stats.agg({c: 'sum' for c in sum_cols}).to_frame().T
                    df_shop_final = calculate_metrics(shop_agg)
                    df_shop_out = format_dataframe(df_shop_final, TARGET_COLUMNS_SHOP)

                    # === 结果展示区 (垂直全览模式) ===
                    st.success("✅ 测算成功！数据如下：")
                    
                    # 1. 下载按钮
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df_sku_out.to_excel(writer, sheet_name='SKU明细', index=False)
                        df_spu_out.to_excel(writer, sheet_name='SPU汇总', index=False)
                        df_shop_out.to_excel(writer, sheet_name='店铺汇总', index=False)
                    
                    st.download_button(
                        label="📥 点击下载最终利润表 (Excel)",
                        data=output.getvalue(),
                        file_name="利润表_最终计算结果.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                    
                    st.markdown("---")

                    # 2. 垂直展示三大表格
                    st.markdown("### 🏠 1. 店铺总览 (全店)")
                    st.dataframe(df_shop_out, use_container_width=True)

                    st.markdown("### 📦 2. SPU 汇总 (Top 10)")
                    st.dataframe(df_spu_out.head(10), use_container_width=True)

                    st.markdown("### 📄 3. SKU 明细 (Top 10)")
                    st.dataframe(df_sku_out.head(10), use_container_width=True)

                except Exception as e:
                    st.error(f"❌ 运行出错: {e}")

if __name__ == '__main__':
    main()