"""
calculator_v3.py
功能：SPU级利润计算器 - 严格根据计算公式详细说明.xlsx执行
修改记录：新增成本拆分输出（采购/头程/尾程/关税各自独立列）
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from typing import Dict, Tuple, List
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from formula_config import get_formula_config


@dataclass
class CalculationResult:
    """计算结果容器"""
    store_matrix: pd.DataFrame         # 店铺维度3行矩阵（本期值/同比/占比）
    spu_matrix: pd.DataFrame           # SPU维度3行矩阵
    category_matrix: pd.DataFrame      # 三级类目维度3行矩阵
    store_profit: pd.DataFrame         # 店铺原始数值
    spu_profit: pd.DataFrame           # SPU原始数值
    cost_allocation: pd.DataFrame
    order_detail: pd.DataFrame
    unmapped_ads: List[Dict]
    summary_stats: Dict
    output_path: str


class ProfitCalculator:
    """
    利润计算器 - 所有公式从计算公式详细说明.xlsx读取
    """

    def __init__(self, data: Dict[str, pd.DataFrame], last_year_orders: pd.DataFrame = None, formula_file: str = None):
        self.data = data
        self.last_year_orders = last_year_orders

        if formula_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            possible_paths = [
                os.path.join(current_dir, '计算公式详细说明.xlsx'),
                os.path.join(os.path.dirname(current_dir), '00-系统文件（不动）', '计算公式详细说明.xlsx'),
                '00-系统文件（不动）/计算公式详细说明.xlsx',
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    formula_file = path
                    break

        self.formula_config = get_formula_config(formula_file)

        self.EXCHANGE_RATE = self.formula_config.get_param('exchange_rate')
        self.PLATFORM_COMMISSION_RATE = self.formula_config.get_param('platform_commission_rate')
        self.OTHER_FEE_RATE = self.formula_config.get_param('other_fee_rate')

        # 加载需排除的订单编号（不计算样品费）
        excluded_df = self.data.get('excluded_orders', pd.DataFrame())
        if not excluded_df.empty and 'Order ID' in excluded_df.columns:
            self.excluded_order_ids = set(excluded_df['Order ID'].astype(str).str.strip())
        else:
            self.excluded_order_ids = set()
        if self.excluded_order_ids:
            print(f"  需排除样品费的订单: {len(self.excluded_order_ids)} 个")

        # 加载 SKU-三级类目映射
        cat_df = self.data.get('sku_category_map', pd.DataFrame())
        if not cat_df.empty and 'SKU' in cat_df.columns and '三级类目' in cat_df.columns:
            self.sku_to_category = dict(zip(
                cat_df['SKU'].astype(str).str.strip().str.upper(),
                cat_df['三级类目'].astype(str).str.strip()
            ))
        else:
            self.sku_to_category = {}
        if self.sku_to_category:
            print(f"  SKU-三级类目映射: {len(self.sku_to_category)} 条")

        print(f"\n【配置参数】汇率={self.EXCHANGE_RATE}, 平台佣金率={self.PLATFORM_COMMISSION_RATE}, 其他费用率={self.OTHER_FEE_RATE}")

    def calculate(self) -> CalculationResult:
        print("\n" + "=" * 80)
        print("利润计算 - 严格按公式说明文档执行")
        print("=" * 80)

        print("\n【Step 1/5】数据预处理...")
        orders = self._prepare_orders()
        affiliate = self._prepare_affiliate()
        ads, unmapped_ads = self._prepare_ads()
        costs = self._prepare_costs()

        print("【Step 2/5】SPU级订单聚合...")
        spu_orders = self._aggregate_spu_orders(orders)

        print("【Step 3/5】费用映射与分摊...")
        spu_affiliate = self._map_affiliate_to_spu(affiliate, orders)
        spu_ads = self._allocate_ads_to_spu(ads, spu_orders)

        print("【Step 4/5】SPU成本计算（含拆分）...")
        spu_costs = self._calculate_spu_costs(orders, costs)

        print("【Step 4.5/5】样品费计算...")
        sample_fees = self._calculate_sample_fee(orders, costs)

        print("【Step 5/5】利润汇总计算...")
        profit_table = self._merge_profit_data(spu_orders, spu_costs, spu_affiliate, spu_ads, sample_fees)

        print("【Step 5.5/5】店铺维度汇总...")
        store_profit = self._calculate_store_profit(profit_table)

        print("【Step 6/5】去年同期订单指标计算...")
        ly_spu_orders = None
        ly_store = None
        if self.last_year_orders is not None and not self.last_year_orders.empty:
            ly_spu_orders = self._calculate_last_year_spu_orders(self.last_year_orders)
            ly_store = self._calculate_last_year_store(ly_spu_orders)
            print(f"  去年同期SPU数: {len(ly_spu_orders)}")

        print("【Step 7/5】构建3行矩阵...")
        store_matrix = self._build_store_matrix(store_profit, ly_store)
        spu_matrix = self._build_spu_matrix(profit_table, ly_spu_orders)
        category_matrix = self._build_category_matrix(profit_table, ly_spu_orders)

        output_path = self._generate_output(profit_table, spu_orders, spu_affiliate, spu_ads)

        stats = self._generate_summary(profit_table)
        self._print_summary(stats)

        return CalculationResult(
            store_matrix=store_matrix,
            spu_matrix=spu_matrix,
            category_matrix=category_matrix,
            store_profit=store_profit,
            spu_profit=profit_table,
            cost_allocation=spu_ads,
            order_detail=orders,
            unmapped_ads=unmapped_ads,
            summary_stats=stats,
            output_path=output_path
        )

    def _prepare_orders(self) -> pd.DataFrame:
        df = self.data['orders'].copy()

        df['Seller SKU'] = df['Seller SKU'].astype(str).str.strip().str.upper()
        df['SKU ID'] = df['SKU ID'].astype(str).str.strip()
        df['营收'] = pd.to_numeric(df['营收'], errors='coerce').fillna(0)
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce').fillna(0)

        if 'Shipping Fee After Discount' in df.columns:
            df['运费'] = pd.to_numeric(df['Shipping Fee After Discount'], errors='coerce').fillna(0)
        else:
            df['运费'] = 0

        # 营收列已包含 Shipping Fee After Discount，无需再加运费
        # 营收 = SKU Subtotal After Discount + Platform Discount + Shipping Fee After Discount
        df['总营收'] = df['营收']
        df['Is_Refund'] = df['Order Status'].isin(['Cancelled', 'Canceled'])

        # 获取NAN SKU回退映射（从配置读取，支持数据更换后灵活调整）
        nan_fallback = self.formula_config.get_param('nan_sku_fallback') or 'ERL001-BK-R-PN'

        spu_map = self.data['spu_map'].copy()
        spu_map['SKU'] = spu_map['SKU'].astype(str).str.strip().str.upper()
        sku_to_spu = dict(zip(spu_map['SKU'], spu_map['SPU']))

        # 先修正 NAN 的 Seller SKU，让后续 SPU 映射和成本查找都正确
        df.loc[df['Seller SKU'] == 'NAN', 'Seller SKU'] = nan_fallback

        # SPU 映射（NAN已修正，会自动通过 spu_map 正确映射）
        df['SPU'] = df['Seller SKU'].map(sku_to_spu)
        # 未映射的SKU使用自身名称作为SPU
        df.loc[df['SPU'].isna(), 'SPU'] = df.loc[df['SPU'].isna(), 'Seller SKU']

        return df

    def _prepare_affiliate(self) -> pd.DataFrame:
        df = self.data['affiliate'].copy()
        if df.empty:
            return df

        df['SKU ID'] = df['SKU ID'].astype(str).str.strip()
        df['Est_Standard'] = pd.to_numeric(df['Est. standard commission payment'], errors='coerce').fillna(0)
        df['Est_ShopAds'] = pd.to_numeric(df['Est. Shop Ads commission payment'], errors='coerce').fillna(0)
        df['Commission'] = df['Est_Standard'] + df['Est_ShopAds']

        orders = self._prepare_orders()
        sku_id_to_spu = dict(zip(orders['SKU ID'].astype(str).str.strip(), orders['SPU']))
        df['SPU'] = df['SKU ID'].map(sku_id_to_spu)
        df.loc[df['SPU'].isna(), 'SPU'] = 'SKU_' + df.loc[df['SPU'].isna(), 'SKU ID'].str[-6:]

        return df

    def _prepare_ads(self) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        准备广告数据
        返回: (广告DataFrame, 未匹配PID列表)
        """
        df = self.data['ads'].copy()
        unmapped_ads = []
        
        if df.empty:
            return df, unmapped_ads

        df['PID_Clean'] = df['Product ID'].astype(str).str.strip()
        df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce').fillna(0)

        pid_map = self.data['pid_sku_map'].copy()
        pid_map['PID_Clean'] = pid_map['product id'].astype(str).str.strip()
        # 一个PID可能映射多个SKU，保留第一个（这些SKU通常属于同一SPU）
        pid_to_sku = pid_map.groupby('PID_Clean')['seller_sku'].first().to_dict()
        df['Seller SKU'] = df['PID_Clean'].map(pid_to_sku).str.upper()

        spu_map = self.data['spu_map'].copy()
        spu_map['SKU'] = spu_map['SKU'].astype(str).str.strip().str.upper()
        sku_to_spu = dict(zip(spu_map['SKU'], spu_map['SPU']))
        df['SPU'] = df['Seller SKU'].str.upper().map(sku_to_spu)
        df.loc[df['SPU'].isna() & df['Seller SKU'].notna(), 'SPU'] = \
            df.loc[df['SPU'].isna() & df['Seller SKU'].notna(), 'Seller SKU']

        # 记录未匹配的PID
        mapped_spus = set(spu_map['SPU'].unique())
        unmapped_df = df[~df['SPU'].isin(mapped_spus)].copy()
        
        for _, row in unmapped_df.iterrows():
            unmapped_ads.append({
                'PID': row['PID_Clean'],
                '花费': float(row['Cost']),
                'Campaign名称': str(row.get('Campaign Name', ''))[:50],
                'Ad组名称': str(row.get('Ad Group Name', ''))[:50]
            })

        return df, unmapped_ads

    def _prepare_costs(self) -> Dict[str, pd.DataFrame]:
        costs = {}
        for name, table_name in [
            ('purchase', 'purchase'),
            ('first_mile', 'first_mile'),
            ('last_mile', 'last_mile'),
            ('tariff', 'tariff')
        ]:
            df = self.data[table_name].copy()
            if not df.empty:
                df['SKU'] = df['SKU'].astype(str).str.strip().str.upper()
            costs[name] = df
        return costs

    def _aggregate_spu_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        # 退款前营收 = 所有订单的总营收（包括退款订单在取消前的金额）
        gross_revenue = orders.groupby('SPU')['总营收'].sum().reset_index()
        gross_revenue.columns = ['SPU', '退款前营收']

        # 正常订单（非退款且营收>0）
        normal_orders = orders[(~orders['Is_Refund']) & (orders['总营收'] > 0)]
        normal_agg = normal_orders.groupby('SPU').agg({
            '总营收': 'sum',
            'Quantity': 'sum'
        }).reset_index()
        normal_agg.columns = ['SPU', '销售收入', '销售数量']

        # 退款订单
        refund_orders = orders[orders['Is_Refund']]
        refund_agg = refund_orders.groupby('SPU').agg({
            '总营收': 'sum',
            'Quantity': 'sum'
        }).reset_index()
        refund_agg.columns = ['SPU', '退款金额', '退款数量']

        # 样品订单（营收为0且非退款）
        sample_orders = orders[(orders['总营收'] == 0) & (~orders['Is_Refund'])]
        sample_agg = sample_orders.groupby('SPU')['Quantity'].sum().reset_index()
        sample_agg.columns = ['SPU', '样品数量']

        # 合并数据
        spu_orders = normal_agg.merge(refund_agg, on='SPU', how='outer')
        spu_orders = spu_orders.merge(gross_revenue, on='SPU', how='left')
        spu_orders = spu_orders.merge(sample_agg, on='SPU', how='left')
        spu_orders = spu_orders.fillna(0)

        # 净销售收入 = 退款前营收 - 退款金额 = 销售收入
        spu_orders['净销售收入'] = spu_orders['销售收入']
        spu_orders['净销售数量'] = spu_orders['销售数量']

        # 退款前销量 = 正常订单销量 + 退款订单销量
        spu_orders['退款前销量'] = spu_orders['销售数量'] + spu_orders['退款数量']

        return spu_orders

    def _map_affiliate_to_spu(self, affiliate: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
        if affiliate.empty:
            return pd.DataFrame(columns=['SPU', '联盟佣金'])
        spu_affiliate = affiliate.groupby('SPU')['Commission'].sum().reset_index()
        spu_affiliate.columns = ['SPU', '联盟佣金']
        return spu_affiliate

    def _allocate_ads_to_spu(self, ads: pd.DataFrame, spu_orders: pd.DataFrame) -> pd.DataFrame:
        if ads.empty:
            return pd.DataFrame(columns=['SPU', '广告费'])

        spu_map = self.data['spu_map'].copy()
        mapped_spus = set(spu_map['SPU'].unique())

        mapped_ads = ads[ads['SPU'].isin(mapped_spus)]
        unmapped_ads = ads[~ads['SPU'].isin(mapped_spus)]

        mapped_spend = mapped_ads.groupby('SPU')['Cost'].sum().reset_index()
        mapped_spend.columns = ['SPU', '已映射广告费']

        total_unmapped = unmapped_ads['Cost'].sum()
        if total_unmapped > 0:
            total_revenue = spu_orders['净销售收入'].sum()
            spu_orders['分摊比例'] = spu_orders['净销售收入'] / total_revenue if total_revenue > 0 else 0
            spu_orders['分摊广告费'] = spu_orders['分摊比例'] * total_unmapped
        else:
            spu_orders['分摊广告费'] = 0

        ads_result = spu_orders[['SPU', '分摊广告费']].merge(mapped_spend, on='SPU', how='left').fillna(0)
        ads_result['广告费'] = ads_result['分摊广告费'] + ads_result['已映射广告费']

        return ads_result[['SPU', '广告费', '已映射广告费', '分摊广告费']]

    def _map_spu_to_category(self, spu_map: pd.DataFrame) -> pd.DataFrame:
        """
        基于 spu_map 反向映射，为每个 SPU 确定三级类目
        规则：取该 SPU 下第一个在 sku_to_category 中有映射的 SKU 的三级类目
        """
        if not self.sku_to_category or spu_map.empty:
            return pd.DataFrame(columns=['SPU', '三级类目'])

        spu_map = spu_map.copy()
        spu_map['SKU'] = spu_map['SKU'].astype(str).str.strip().str.upper()

        result = []
        for spu, group in spu_map.groupby('SPU'):
            category = ''
            for sku in group['SKU']:
                cat = self.sku_to_category.get(sku)
                if cat:
                    category = cat
                    break
            result.append({'SPU': spu, '三级类目': category})

        return pd.DataFrame(result)

    def _get_sku_cost_breakdown(self, sku: str, costs: Dict) -> Dict[str, float]:
        """
        获取SKU单位成本明细（CNY）
        返回: {'采购': x, '头程': x, '尾程': x, '关税': x}
        """
        result = {'采购': 0.0, '头程': 0.0, '尾程': 0.0, '关税': 0.0}

        # 采购成本（CNY）- 列名可能是'采购成本'或'采购成本（CNY）'
        df_purchase = costs['purchase']
        if not df_purchase.empty:
            row = df_purchase[df_purchase['SKU'] == sku]
            if not row.empty:
                # 尝试多种可能的列名
                for col in row.columns:
                    if col.strip() in ['采购成本', '采购成本（CNY）', '采购成本(CNY)']:
                        val = row[col].iloc[0]
                        if pd.notna(val):
                            result['采购'] = float(val)
                            break

        # 头程成本（CNY）- 与采购在同一文件
        if not costs['purchase'].empty:
            row = costs['purchase'][costs['purchase']['SKU'] == sku]
            if not row.empty:
                for col in row.columns:
                    if col.strip() in ['头程成本', '头程成本（CNY）', '头程成本(CNY)']:
                        val = row[col].iloc[0]
                        if pd.notna(val):
                            result['头程'] = float(val)
                            break

        # 尾程成本（CNY）- 列名可能是'尾程单价'或其他包含'尾程'的列
        df_last = costs['last_mile']
        if not df_last.empty:
            row = df_last[df_last['SKU'] == sku]
            if not row.empty:
                for col in row.columns:
                    col_clean = col.strip()
                    # 匹配尾程相关列名
                    if '尾程' in col_clean or 'last mile' in col_clean.lower():
                        val = row[col].iloc[0]
                        if pd.notna(val):
                            result['尾程'] = float(val)
                            break

        # 关税成本（CNY）- 列名可能是'关税\nRMB'或'标准关税金额\nRMB'
        df_tariff = costs['tariff']
        if not df_tariff.empty:
            row = df_tariff[df_tariff['SKU'] == sku]
            if not row.empty:
                for col in row.columns:
                    col_clean = col.strip()
                    if col_clean in ['关税\nRMB', '标准关税金额\nRMB', '关税RMB', '关税']:
                        val = row[col].iloc[0]
                        if pd.notna(val):
                            result['关税'] = float(val)
                            break

        return result

    def _calculate_spu_costs(self, orders: pd.DataFrame, costs: Dict) -> pd.DataFrame:
        """
        计算SPU成本（含采购/头程/尾程/关税拆分）
        公式: SUM(各成本项 × 销售数量) / 汇率
        """
        # 各SKU销售数量（仅正常销售订单：非退款且营收>0）
        sku_outbound = {}
        for _, row in orders[(~orders['Is_Refund']) & (orders['总营收'] > 0)].iterrows():
            sku = row['Seller SKU']
            qty = row['Quantity']
            sku_outbound[sku] = sku_outbound.get(sku, 0) + qty

        # 获取SPU映射
        spu_map = self.data['spu_map'].copy()
        spu_map['SKU'] = spu_map['SKU'].astype(str).str.strip().str.upper()
        sku_to_spu = dict(zip(spu_map['SKU'], spu_map['SPU']))

        # 按SPU汇总各类成本（CNY）
        spu_costs = {}
        for sku, qty in sku_outbound.items():
            spu = sku_to_spu.get(sku, sku)
            if spu not in spu_costs:
                spu_costs[spu] = {'采购': 0.0, '头程': 0.0, '尾程': 0.0, '关税': 0.0, '出库数量': 0}

            breakdown = self._get_sku_cost_breakdown(sku, costs)
            spu_costs[spu]['采购'] += breakdown['采购'] * qty
            spu_costs[spu]['头程'] += breakdown['头程'] * qty
            spu_costs[spu]['尾程'] += breakdown['尾程'] * qty
            spu_costs[spu]['关税'] += breakdown['关税'] * qty
            spu_costs[spu]['出库数量'] += qty

        result = []
        for spu, data in spu_costs.items():
            result.append({
                'SPU': spu,
                '采购成本_CNY': data['采购'],
                '头程成本_CNY': data['头程'],
                '尾程成本_CNY': data['尾程'],
                '关税成本_CNY': data['关税'],
                '成本_CNY': data['采购'] + data['头程'] + data['尾程'] + data['关税'],
                '出库数量': data['出库数量'],
            })

        return pd.DataFrame(result) if result else pd.DataFrame(
            columns=['SPU', '采购成本_CNY', '头程成本_CNY', '尾程成本_CNY', '关税成本_CNY', '成本_CNY', '出库数量']
        )

    def _calculate_sample_fee(self, orders: pd.DataFrame, costs: Dict) -> pd.DataFrame:
        sample_orders = orders[(orders['营收'] == 0) & (~orders['Is_Refund'])].copy()
        if self.excluded_order_ids:
            sample_orders = sample_orders[~sample_orders['Order ID'].astype(str).str.strip().isin(self.excluded_order_ids)]
        if sample_orders.empty:
            return pd.DataFrame(columns=['SPU', '样品费'])

        sample_skus = sample_orders['Seller SKU'].unique()
        sku_unit_cost = {}
        for sku in sample_skus:
            bd = self._get_sku_cost_breakdown(sku, costs)
            sku_unit_cost[sku] = sum(bd.values())

        sample_orders['样品成本_CNY'] = sample_orders['Seller SKU'].map(sku_unit_cost)
        sample_orders['样品费_CNY'] = sample_orders['Quantity'] * sample_orders['样品成本_CNY']

        spu_sample_fee = sample_orders.groupby('SPU')['样品费_CNY'].sum().reset_index()
        spu_sample_fee['样品费'] = spu_sample_fee['样品费_CNY'] / self.EXCHANGE_RATE

        return spu_sample_fee[['SPU', '样品费']]

    def _calculate_store_profit(self, spu_profit: pd.DataFrame) -> pd.DataFrame:
        """
        计算店铺维度汇总（所有SPU加总）
        """
        if spu_profit.empty:
            return pd.DataFrame(columns=[
                '净销售收入', '销售数量', '采购成本', '头程成本', '尾程成本', '关税成本',
                '成本', '联盟佣金', '广告费', '平台佣金', '其他费用', '样品费',
                '毛利润', '毛利率'
            ])

        # 数值列求和
        numeric_cols = [
            '退款前营收', '退款后营收', '退款金额', '退款数量', '退款前销量',
            '销售收入', '销售数量', '净销售收入', '净销售数量',
            '采购成本', '头程成本', '尾程成本', '关税成本', '成本',
            '联盟佣金', '广告费', '平台佣金', '其他费用', '样品费', '毛利润'
        ]

        store = {}
        for col in numeric_cols:
            store[col] = spu_profit[col].sum() if col in spu_profit.columns else 0

        # 整体毛利率 = 总毛利润 / 总净销售收入
        net_revenue = store.get('净销售收入', 0)
        store['毛利率'] = store['毛利润'] / net_revenue if net_revenue != 0 else 0

        # 保留核心输出列
        output_cols = [
            '退款前营收', '退款后营收', '退款金额', '退款数量', '退款前销量',
            '销售收入', '销售数量', '净销售收入', '净销售数量',
            '采购成本', '头程成本', '尾程成本', '关税成本', '成本',
            '联盟佣金', '广告费', '平台佣金', '其他费用', '样品费',
            '毛利润', '毛利率'
        ]

        return pd.DataFrame([{k: store.get(k, 0) for k in output_cols}])

    def _merge_profit_data(self, spu_orders: pd.DataFrame, spu_costs: pd.DataFrame,
                           spu_affiliate: pd.DataFrame, spu_ads: pd.DataFrame,
                           sample_fees: pd.DataFrame) -> pd.DataFrame:
        # 合并三级类目
        spu_category = self._map_spu_to_category(self.data.get('spu_map', pd.DataFrame()))
        spu_orders = spu_orders.merge(spu_category, on='SPU', how='left')
        spu_orders['三级类目'] = spu_orders['三级类目'].fillna('')

        profit = spu_orders.merge(spu_costs, on='SPU', how='outer')
        profit = profit.merge(spu_affiliate, on='SPU', how='left')
        profit = profit.merge(spu_ads[['SPU', '广告费']], on='SPU', how='left')
        profit = profit.merge(sample_fees, on='SPU', how='left')
        profit = profit.fillna(0)

        # 成本转换为USD（各项拆分）
        profit['采购成本'] = profit['采购成本_CNY'] / self.EXCHANGE_RATE
        profit['头程成本'] = profit['头程成本_CNY'] / self.EXCHANGE_RATE
        profit['尾程成本'] = profit['尾程成本_CNY'] / self.EXCHANGE_RATE
        profit['关税成本'] = profit['关税成本_CNY'] / self.EXCHANGE_RATE
        profit['成本'] = profit['成本_CNY'] / self.EXCHANGE_RATE

        # 平台佣金 = 退款后营收 × 平台佣金率
        profit['平台佣金'] = profit['净销售收入'] * self.PLATFORM_COMMISSION_RATE

        # 其他费用 = 退款后营收 × 其他费用率
        profit['其他费用'] = profit['净销售收入'] * self.OTHER_FEE_RATE

        if '样品费' not in profit.columns:
            profit['样品费'] = 0

        # 添加退款后营收列（等于净销售收入，用于直观对比）
        profit['退款后营收'] = profit['净销售收入']

        # 毛利润
        profit['毛利润'] = (profit['净销售收入'] - profit['成本'] - profit['联盟佣金'] -
                          profit['广告费'] - profit['平台佣金'] - profit['其他费用'] -
                          profit['样品费'])

        profit['毛利率'] = profit.apply(
            lambda x: x['毛利润'] / x['净销售收入'] if x['净销售收入'] != 0 else 0, axis=1
        )

        # 排序：未分类末尾，其余按净销售收入降序
        profit['排序'] = profit.apply(
            lambda x: (1 if x['SPU'] == '未分类' else 0, -x['净销售收入']), axis=1
        )
        profit = profit.sort_values('排序').drop('排序', axis=1)

        output_cols = [
            'SPU', '三级类目', '退款前营收', '退款后营收', '退款金额', '退款数量',
            '退款前销量', '销售收入', '销售数量', '净销售收入', '净销售数量',
            '采购成本', '头程成本', '尾程成本', '关税成本', '成本',
            '联盟佣金', '广告费', '平台佣金', '其他费用', '样品费',
            '毛利润', '毛利率'
        ]

        return profit[output_cols]

    def _generate_output(self, profit_table: pd.DataFrame, spu_orders: pd.DataFrame,
                         spu_affiliate: pd.DataFrame, spu_ads: pd.DataFrame) -> str:
        # Excel 输出由启动测算.py 统一处理，此处返回空字符串
        return ''

    def _generate_summary(self, profit_table: pd.DataFrame) -> Dict:
        return {
            '总净销售收入': profit_table['净销售收入'].sum(),
            '总采购成本': profit_table['采购成本'].sum(),
            '总头程成本': profit_table['头程成本'].sum(),
            '总尾程成本': profit_table['尾程成本'].sum(),
            '总关税成本': profit_table['关税成本'].sum(),
            '总成本': profit_table['成本'].sum(),
            '总联盟佣金': profit_table['联盟佣金'].sum(),
            '总广告费': profit_table['广告费'].sum(),
            '总平台佣金': profit_table['平台佣金'].sum(),
            '总其他费用': profit_table['其他费用'].sum(),
            '总样品费': profit_table['样品费'].sum(),
            '总毛利润': profit_table['毛利润'].sum(),
            '整体毛利率': (profit_table['毛利润'].sum() / profit_table['净销售收入'].sum()
                        if profit_table['净销售收入'].sum() > 0 else 0),
            '总退款金额': profit_table['退款金额'].sum(),
            '总销售数量': profit_table['销售数量'].sum(),
        }

    def _print_summary(self, stats: Dict):
        print("\n" + "=" * 80)
        print("【计算结果摘要】")
        print("=" * 80)
        print(f"  净销售收入:      ${stats['总净销售收入']:>12,.2f}")
        print(f"  采购成本:        ${stats['总采购成本']:>12,.2f}")
        print(f"  头程成本:        ${stats['总头程成本']:>12,.2f}")
        print(f"  尾程成本:        ${stats['总尾程成本']:>12,.2f}")
        print(f"  关税成本:        ${stats['总关税成本']:>12,.2f}")
        print(f"  总联盟佣金:      ${stats['总联盟佣金']:>12,.2f}")
        print(f"  总广告费:        ${stats['总广告费']:>12,.2f}")
        print(f"  总平台佣金:      ${stats['总平台佣金']:>12,.2f}")
        print(f"  总其他费用:      ${stats['总其他费用']:>12,.2f}")
        print(f"  总样品费:        ${stats['总样品费']:>12,.2f}")
        print(f"  ─────────────────────────────────────")
        print(f"  总毛利润:        ${stats['总毛利润']:>12,.2f}")
        print(f"  整体毛利率:      {stats['整体毛利率']*100:>12.2f}%")
        print("=" * 80)


    # ── 去年同期与矩阵构建方法 ─────────────────────────────────────

    def _calculate_last_year_spu_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        """计算去年同期SPU级订单指标（仅订单层面，无成本/广告/联盟）"""
        if orders.empty:
            return pd.DataFrame(columns=[
                'SPU', '三级类目', '退款前营收', '退款后营收', '退款金额', '退款数量',
                '退款前销量', '销售收入', '销售数量', '净销售收入', '净销售数量', '样品数量'
            ])

        spu_map = self.data['spu_map'].copy()
        spu_map['SKU'] = spu_map['SKU'].astype(str).str.strip().str.upper()
        sku_to_spu = dict(zip(spu_map['SKU'], spu_map['SPU']))
        orders['SPU'] = orders['Seller SKU'].map(sku_to_spu)
        orders.loc[orders['SPU'].isna(), 'SPU'] = orders.loc[orders['SPU'].isna(), 'Seller SKU']

        gross_revenue = orders.groupby('SPU')['总营收'].sum().reset_index()
        gross_revenue.columns = ['SPU', '退款前营收']

        normal = orders[(~orders['Is_Refund']) & (orders['总营收'] > 0)]
        normal_agg = normal.groupby('SPU').agg({'总营收': 'sum', 'Quantity': 'sum'}).reset_index()
        normal_agg.columns = ['SPU', '销售收入', '销售数量']

        refund = orders[orders['Is_Refund']]
        refund_agg = refund.groupby('SPU').agg({'总营收': 'sum', 'Quantity': 'sum'}).reset_index()
        refund_agg.columns = ['SPU', '退款金额', '退款数量']

        sample = orders[(orders['总营收'] == 0) & (~orders['Is_Refund'])]
        sample_agg = sample.groupby('SPU')['Quantity'].sum().reset_index()
        sample_agg.columns = ['SPU', '样品数量']

        result = normal_agg.merge(refund_agg, on='SPU', how='outer')
        result = result.merge(gross_revenue, on='SPU', how='left')
        result = result.merge(sample_agg, on='SPU', how='left')
        result = result.fillna(0)

        result['退款后营收'] = result['销售收入']
        result['净销售收入'] = result['销售收入']
        result['净销售数量'] = result['销售数量']
        result['退款前销量'] = result['销售数量'] + result['退款数量']

        # 增加三级类目
        spu_category = self._map_spu_to_category(self.data.get('spu_map', pd.DataFrame()))
        result = result.merge(spu_category, on='SPU', how='left')
        result['三级类目'] = result['三级类目'].fillna('')

        result = result.sort_values('净销售收入', ascending=False)
        return result[[
            'SPU', '三级类目', '退款前营收', '退款后营收', '退款金额', '退款数量',
            '退款前销量', '销售收入', '销售数量', '净销售收入', '净销售数量', '样品数量'
        ]]

    def _calculate_last_year_store(self, ly_spu: pd.DataFrame) -> pd.DataFrame:
        """去年同期店铺汇总"""
        if ly_spu.empty:
            return pd.DataFrame(columns=[
                '退款前营收', '退款后营收', '退款金额', '退款数量', '退款前销量',
                '销售收入', '销售数量', '净销售收入', '净销售数量', '样品数量'
            ])
        numeric_cols = [
            '退款前营收', '退款后营收', '退款金额', '退款数量', '退款前销量',
            '销售收入', '销售数量', '净销售收入', '净销售数量', '样品数量'
        ]
        store = {col: ly_spu[col].sum() for col in numeric_cols}
        return pd.DataFrame([store])

    @staticmethod
    def _yoy(cur_val, ly_val):
        """同比变化格式化"""
        if ly_val is None or ly_val == 0:
            return '—'
        change = (cur_val - ly_val) / ly_val * 100
        arrow = '↑' if change >= 0 else '↓'
        return f'同{change:+.1f}% {arrow}'

    @staticmethod
    def _ratio(val, denominator):
        """占比格式化"""
        if denominator == 0:
            return '—'
        return f'{val / denominator * 100:.1f}%'

    def _build_store_matrix(self, current: pd.DataFrame, last_year: pd.DataFrame = None) -> pd.DataFrame:
        """构建店铺维度3行矩阵"""
        if current.empty:
            return pd.DataFrame()

        cur = current.iloc[0]
        ly = {}
        if last_year is not None and not last_year.empty:
            ly_row = last_year.iloc[0]
            for col in last_year.columns:
                ly[col] = ly_row[col]

        net_revenue = cur.get('净销售收入', 0)
        cur_asp = cur.get('退款前营收', 0) / cur.get('退款前销量', 1) if cur.get('退款前销量', 0) > 0 else 0
        ly_asp = ly.get('退款前营收', 0) / ly.get('退款前销量', 1) if ly.get('退款前销量', 0) > 0 else 0

        cols = [
            '退款前营收', '退款后营收', '退款后销量', 'ASP', '采购成本', '头程成本',
            '尾程成本', '关税成本', '联盟佣金', '广告费', '平台佣金', '其他费用',
            '样品费', '退款金额', '毛利润', '毛利率', '备注'
        ]

        row1, row2, row3 = {}, {}, {}
        for col in cols:
            if col == 'ASP':
                row1[col] = round(cur_asp, 2)
                row2[col] = self._yoy(cur_asp, ly_asp)
                row3[col] = '—'
            elif col == '退款后销量':
                row1[col] = int(cur.get('净销售数量', 0))
                row2[col] = self._yoy(cur.get('净销售数量', 0), ly.get('净销售数量', None))
                row3[col] = '—'
            elif col == '毛利率':
                row1[col] = f"{cur.get('毛利率', 0) * 100:.2f}%"
                row2[col] = '—'
                row3[col] = f"{cur.get('毛利率', 0) * 100:.1f}%"
            elif col == '备注':
                row1[col] = row2[col] = row3[col] = ''
            elif col in current.columns:
                cur_val = cur.get(col, 0)
                ly_val = ly.get(col, None)
                row1[col] = round(cur_val, 2)
                if col in ['退款前营收', '退款后营收', '退款金额', '销售收入', '销售数量', '净销售收入', '净销售数量', '样品数量']:
                    row2[col] = self._yoy(cur_val, ly_val)
                else:
                    row2[col] = '—'
                if col in ['退款后销量', 'ASP']:
                    row3[col] = '—'
                else:
                    row3[col] = self._ratio(cur_val, net_revenue)
            else:
                row1[col] = 0
                row2[col] = '—'
                row3[col] = '—'

        return pd.DataFrame([row1, row2, row3])

    def _build_spu_matrix(self, current: pd.DataFrame, last_year: pd.DataFrame = None) -> pd.DataFrame:
        """构建SPU维度3行矩阵（数值行 + 同比行 + 占比行）"""
        if current.empty:
            return pd.DataFrame()

        total_net_revenue = current['净销售收入'].sum()
        ly_dict = {}
        if last_year is not None and not last_year.empty:
            for _, row in last_year.iterrows():
                ly_dict[row['SPU']] = row

        rows = []
        for _, cur in current.iterrows():
            spu = cur['SPU']
            ly = ly_dict.get(spu, {})
            net_revenue = cur.get('净销售收入', 0)
            cur_asp = cur.get('退款前营收', 0) / cur.get('退款前销量', 1) if cur.get('退款前销量', 0) > 0 else 0
            ly_asp = ly.get('退款前营收', 0) / ly.get('退款前销量', 1) if ly.get('退款前销量', 0) > 0 else 0

            row1 = {
                'SPU': spu, '三级类目': cur.get('三级类目', ''),
                '营收占比': f"{net_revenue / total_net_revenue * 100:.1f}%" if total_net_revenue > 0 else '—',
                '退款前营收': round(cur.get('退款前营收', 0), 2),
                '退款后营收': round(cur.get('退款后营收', 0), 2),
                'ASP': round(cur_asp, 2),
                '退款前销量': int(cur.get('退款前销量', 0)),
                '退款后销量': int(cur.get('净销售数量', 0)),
                '采购成本': round(cur.get('采购成本', 0), 2),
                '头程成本': round(cur.get('头程成本', 0), 2),
                '尾程成本': round(cur.get('尾程成本', 0), 2),
                '关税成本': round(cur.get('关税成本', 0), 2),
                '联盟佣金': round(cur.get('联盟佣金', 0), 2),
                '广告费': round(cur.get('广告费', 0), 2),
                '平台佣金': round(cur.get('平台佣金', 0), 2),
                '其他费用': round(cur.get('其他费用', 0), 2),
                '样品费': round(cur.get('样品费', 0), 2),
                '退款金额': round(cur.get('退款金额', 0), 2),
                '毛利润': round(cur.get('毛利润', 0), 2),
                '毛利率': f"{cur.get('毛利率', 0) * 100:.2f}%",
            }
            row2 = {
                'SPU': '', '三级类目': '', '营收占比': '',
                '退款前营收': self._yoy(cur.get('退款前营收', 0), ly.get('退款前营收', None)),
                '退款后营收': self._yoy(cur.get('退款后营收', 0), ly.get('退款后营收', None)),
                'ASP': self._yoy(cur_asp, ly_asp),
                '退款前销量': self._yoy(cur.get('退款前销量', 0), ly.get('退款前销量', None)),
                '退款后销量': self._yoy(cur.get('净销售数量', 0), ly.get('净销售数量', None)),
                '采购成本': '—', '头程成本': '—', '尾程成本': '—', '关税成本': '—',
                '联盟佣金': '—', '广告费': '—', '平台佣金': '—', '其他费用': '—',
                '样品费': '—',
                '退款金额': self._yoy(cur.get('退款金额', 0), ly.get('退款金额', None)),
                '毛利润': '—', '毛利率': '—',
            }
            row3 = {
                'SPU': '', '三级类目': '', '营收占比': '',
                '退款前营收': self._ratio(cur.get('退款前营收', 0), net_revenue),
                '退款后营收': '100.0%', 'ASP': '—',
                '退款前销量': '—', '退款后销量': '—',
                '采购成本': self._ratio(cur.get('采购成本', 0), net_revenue),
                '头程成本': self._ratio(cur.get('头程成本', 0), net_revenue),
                '尾程成本': self._ratio(cur.get('尾程成本', 0), net_revenue),
                '关税成本': self._ratio(cur.get('关税成本', 0), net_revenue),
                '联盟佣金': self._ratio(cur.get('联盟佣金', 0), net_revenue),
                '广告费': self._ratio(cur.get('广告费', 0), net_revenue),
                '平台佣金': self._ratio(cur.get('平台佣金', 0), net_revenue),
                '其他费用': self._ratio(cur.get('其他费用', 0), net_revenue),
                '样品费': self._ratio(cur.get('样品费', 0), net_revenue),
                '退款金额': self._ratio(cur.get('退款金额', 0), net_revenue),
                '毛利润': f"{cur.get('毛利率', 0) * 100:.1f}%",
                '毛利率': '—',
            }
            rows.extend([row1, row2, row3])

        return pd.DataFrame(rows)

    def _build_category_matrix(self, current: pd.DataFrame, last_year: pd.DataFrame = None) -> pd.DataFrame:
        """构建三级类目维度3行矩阵（数值行 + 同比行 + 占比行）"""
        if current.empty:
            return pd.DataFrame()

        # 本期按三级类目汇总
        cat_cur = current.groupby('三级类目').agg({
            '退款前营收': 'sum', '退款后营收': 'sum', '退款金额': 'sum', '退款数量': 'sum',
            '退款前销量': 'sum', '销售收入': 'sum', '销售数量': 'sum',
            '净销售收入': 'sum', '净销售数量': 'sum',
            '采购成本': 'sum', '头程成本': 'sum', '尾程成本': 'sum', '关税成本': 'sum',
            '成本': 'sum', '联盟佣金': 'sum', '广告费': 'sum', '平台佣金': 'sum',
            '其他费用': 'sum', '样品费': 'sum', '毛利润': 'sum'
        }).reset_index()
        cat_cur['毛利率'] = cat_cur.apply(lambda x: x['毛利润'] / x['净销售收入'] if x['净销售收入'] != 0 else 0, axis=1)

        # 去年同期按三级类目汇总
        ly_dict = {}
        if last_year is not None and not last_year.empty:
            cat_ly = last_year.groupby('三级类目').agg({
                '退款前营收': 'sum', '退款后营收': 'sum', '退款金额': 'sum', '退款数量': 'sum',
                '退款前销量': 'sum', '销售收入': 'sum', '销售数量': 'sum',
                '净销售收入': 'sum', '净销售数量': 'sum', '样品数量': 'sum'
            }).reset_index()
            for _, row in cat_ly.iterrows():
                ly_dict[row['三级类目']] = row

        total_net_revenue = cat_cur['净销售收入'].sum()
        rows = []
        for _, cur in cat_cur.iterrows():
            raw_cat = cur['三级类目']
            if pd.isna(raw_cat) or str(raw_cat).strip() == '':
                cat = '未分类'
            else:
                cat = raw_cat
            ly = ly_dict.get(raw_cat, {})
            net_revenue = cur.get('净销售收入', 0)
            cur_asp = cur.get('退款前营收', 0) / cur.get('退款前销量', 1) if cur.get('退款前销量', 0) > 0 else 0
            ly_asp = ly.get('退款前营收', 0) / ly.get('退款前销量', 1) if ly.get('退款前销量', 0) > 0 else 0

            row1 = {
                '三级类目': cat,
                '营收占比': f"{net_revenue / total_net_revenue * 100:.1f}%" if total_net_revenue > 0 else '—',
                '退款前营收': round(cur.get('退款前营收', 0), 2),
                '退款后营收': round(cur.get('退款后营收', 0), 2),
                'ASP': round(cur_asp, 2),
                '退款前销量': int(cur.get('退款前销量', 0)),
                '退款后销量': int(cur.get('净销售数量', 0)),
                '采购成本': round(cur.get('采购成本', 0), 2),
                '头程成本': round(cur.get('头程成本', 0), 2),
                '尾程成本': round(cur.get('尾程成本', 0), 2),
                '关税成本': round(cur.get('关税成本', 0), 2),
                '联盟佣金': round(cur.get('联盟佣金', 0), 2),
                '广告费': round(cur.get('广告费', 0), 2),
                '平台佣金': round(cur.get('平台佣金', 0), 2),
                '其他费用': round(cur.get('其他费用', 0), 2),
                '样品费': round(cur.get('样品费', 0), 2),
                '退款金额': round(cur.get('退款金额', 0), 2),
                '毛利润': round(cur.get('毛利润', 0), 2),
                '毛利率': f"{cur.get('毛利率', 0) * 100:.2f}%",
            }
            row2 = {
                '三级类目': '', '营收占比': '',
                '退款前营收': self._yoy(cur.get('退款前营收', 0), ly.get('退款前营收', None)),
                '退款后营收': self._yoy(cur.get('退款后营收', 0), ly.get('退款后营收', None)),
                'ASP': self._yoy(cur_asp, ly_asp),
                '退款前销量': self._yoy(cur.get('退款前销量', 0), ly.get('退款前销量', None)),
                '退款后销量': self._yoy(cur.get('净销售数量', 0), ly.get('净销售数量', None)),
                '采购成本': '—', '头程成本': '—', '尾程成本': '—', '关税成本': '—',
                '联盟佣金': '—', '广告费': '—', '平台佣金': '—', '其他费用': '—',
                '样品费': '—',
                '退款金额': self._yoy(cur.get('退款金额', 0), ly.get('退款金额', None)),
                '毛利润': '—', '毛利率': '—',
            }
            row3 = {
                '三级类目': '', '营收占比': '',
                '退款前营收': self._ratio(cur.get('退款前营收', 0), net_revenue),
                '退款后营收': '100.0%', 'ASP': '—',
                '退款前销量': '—', '退款后销量': '—',
                '采购成本': self._ratio(cur.get('采购成本', 0), net_revenue),
                '头程成本': self._ratio(cur.get('头程成本', 0), net_revenue),
                '尾程成本': self._ratio(cur.get('尾程成本', 0), net_revenue),
                '关税成本': self._ratio(cur.get('关税成本', 0), net_revenue),
                '联盟佣金': self._ratio(cur.get('联盟佣金', 0), net_revenue),
                '广告费': self._ratio(cur.get('广告费', 0), net_revenue),
                '平台佣金': self._ratio(cur.get('平台佣金', 0), net_revenue),
                '其他费用': self._ratio(cur.get('其他费用', 0), net_revenue),
                '样品费': self._ratio(cur.get('样品费', 0), net_revenue),
                '退款金额': self._ratio(cur.get('退款金额', 0), net_revenue),
                '毛利润': f"{cur.get('毛利率', 0) * 100:.1f}%",
                '毛利率': '—',
            }
            rows.extend([row1, row2, row3])

        return pd.DataFrame(rows)


if __name__ == '__main__':
    from data_loader_v3 import DataLoader

    print("=" * 80)
    print("利润计算器V4测试 - 含成本拆分输出")
    print("=" * 80)

    loader = DataLoader()
    data = loader.load_all()

    calc = ProfitCalculator(data)
    result = calc.calculate()

    print(f"\n输出文件: {result.output_path}")
