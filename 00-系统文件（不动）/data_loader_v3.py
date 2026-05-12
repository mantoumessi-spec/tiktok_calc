"""
data_loader_v3.py
功能：数据加载器，读取Excel并验证列名
"""

import pandas as pd
import os
from typing import Dict, Optional
from config_v3 import get_config, CURRENT_DATA_DIR, BASE_DATA_DIR
from matcher_v3 import validate_columns, MatchError


class DataLoadError(Exception):
    """数据加载错误"""
    pass


class DataLoader:
    """数据加载器"""
    
    def __init__(self):
        self.config = get_config()
        self.data: Dict[str, pd.DataFrame] = {}
    
    def load_excel(self, filepath: str, table_name: str = '') -> pd.DataFrame:
        """
        读取Excel文件
        
        Args:
            filepath: 文件路径
            table_name: 数据表名称（用于特殊处理）
        
        Returns:
            DataFrame
        """
        if not os.path.exists(filepath):
            raise DataLoadError(f"文件不存在: {filepath}")
        
        try:
            # 特殊处理：广告订单和PID映射表，保持Product ID为字符串
            if '广告' in filepath or 'pid' in filepath.lower():
                df = pd.read_excel(
                    filepath, 
                    sheet_name=0,
                    converters={'Product ID': str, 'product id': str, 'product_id': str}
                )
            else:
                df = pd.read_excel(filepath, sheet_name=0)
            return df
        except Exception as e:
            # 尝试作为CSV读取（有些文件扩展名是.xlsx但实际是CSV）
            try:
                df = pd.read_csv(filepath)
                return df
            except Exception as csv_e:
                raise DataLoadError(f"读取文件失败 {filepath}: Excel: {str(e)} | CSV: {str(csv_e)}")
    
    def clean_data(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """
        清洗数据
        
        Args:
            df: 原始DataFrame
            table_name: 表名
        
        Returns:
            清洗后的DataFrame
        """
        df = df.copy()
        
        # 获取该表的列配置
        all_columns = self.config.get_all_columns(table_name)
        
        # 标准化列名（去空格）
        df.columns = df.columns.astype(str).str.strip()
        
        # 特殊处理：SPU映射表（多列SKU展开）
        if table_name == 'spu_map':
            df = self._expand_spu_map(df)
            # 展开后列名为 SKU 和 SPU，无需再验证原始列名
            return df
        
        # 特殊处理：SKU-三级类目映射表
        if table_name == 'sku_category_map':
            if 'SKU' in df.columns:
                df['SKU'] = df['SKU'].astype(str).str.strip().str.upper()
            if '三级类目' in df.columns:
                df['三级类目'] = df['三级类目'].astype(str).str.strip()
            return df
        
        # 货币字段清洗（移除$、,等符号）
        money_columns = ['营收', '采购金额', '头程金额', '尾程金额', '关税金额', 
                        '佣金金额', '样品金额', 'Cost', '花费']
        
        for col in df.columns:
            if any(m in col for m in money_columns):
                df[col] = self._clean_money(df[col])
        
        # 数值字段清洗
        qty_columns = ['Quantity', '数量', '销量']
        for col in df.columns:
            if any(q in col for q in qty_columns):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # 字符串字段清洗（去空格、转大写用于匹配）
        sku_columns = ['SKU ID', 'Seller SKU', 'SKU', 'product_id', 'Product ID']
        for col in df.columns:
            if col in sku_columns:
                df[col] = df[col].astype(str).str.strip()
                # 过滤无效值（但保留可能有SKU ID的nan行）
                if col == 'Seller SKU':
                    # 只过滤明确的无效值，保留nan（可能是NaN转换来的）
                    df = df[~df[col].str.contains('seller sku input|platform', case=False, na=False)]
                else:
                    df = df[~df[col].str.contains('seller sku input|nan|platform', case=False, na=False)]
        
        # PID字段已在读取时处理（converters保持字符串），这里只清洗空格
        
        return df
    
    def _clean_pid(self, val) -> str:
        """清洗PID字段，处理科学计数法（使用Decimal避免精度丢失）"""
        from decimal import Decimal, InvalidOperation
        
        if pd.isna(val):
            return ''
        s = str(val).strip()
        # 如果是科学计数法（包含e+），使用Decimal避免精度丢失
        if 'e+' in s.lower() or 'e-' in s.lower():
            try:
                # 使用Decimal保持精度
                d = Decimal(s)
                return str(int(d))
            except (InvalidOperation, ValueError):
                return s
        return s
    
    def _expand_spu_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        展开SPU映射表（多列SKU转多行）
        
        原始格式: SPU | 在售SKU | 关联SKU1 | 关联SKU2
        展开格式: SKU | SPU
        """
        if '在售SKU' not in df.columns or 'SPU' not in df.columns:
            return df
        
        # 收集所有SKU列
        sku_columns = ['在售SKU']
        if '关联SKU1' in df.columns:
            sku_columns.append('关联SKU1')
        if '关联SKU2' in df.columns:
            sku_columns.append('关联SKU2')
        
        # 展开为多行
        rows = []
        for _, row in df.iterrows():
            spu = row['SPU']
            for col in sku_columns:
                if pd.notna(row[col]) and str(row[col]).strip():
                    rows.append({'SKU': str(row[col]).strip(), 'SPU': spu})
        
        result = pd.DataFrame(rows)
        print(f"  SPU映射表展开: {len(df)}行 -> {len(result)}行")
        return result
    
    def _clean_money(self, series: pd.Series) -> pd.Series:
        """清洗货币字段"""
        return series.astype(str).str.replace(r'[$,]', '', regex=True) \
                          .str.replace(r'[()]', '-', regex=True) \
                          .apply(lambda x: pd.to_numeric(x, errors='coerce')) \
                          .fillna(0)
    
    def load_table(self, table_id: str, custom_path: Optional[str] = None) -> pd.DataFrame:
        """
        加载单个数据表
        
        Args:
            table_id: 数据表ID
            custom_path: 自定义文件路径（可选）
        
        Returns:
            DataFrame
        """
        # 获取数据源配置
        source_config = self.config.get_data_source(table_id)
        if not source_config:
            raise DataLoadError(f"未知数据表: {table_id}")
        
        # 确定文件路径
        if custom_path:
            filepath = custom_path
        else:
            # 根据filepath字段解析
            filepath = source_config['filepath']
            filepath = os.path.join(os.path.dirname(CURRENT_DATA_DIR), filepath)
        
        # 检查文件是否存在
        if not os.path.exists(filepath):
            if source_config['required']:
                raise DataLoadError(f"必须文件不存在: {filepath}\n{source_config['desc']}")
            else:
                print(f"警告: 可选文件不存在，将使用空数据: {filepath}")
                return pd.DataFrame()
        
        # 读取数据
        print(f"正在加载: {table_id} -> {os.path.basename(filepath)}")
        df = self.load_excel(filepath)
        
        # 数据清洗
        df = self.clean_data(df, table_id)
        
        # 验证必须列（SPU映射表特殊处理，已在clean_data中展开）
        if source_config['required'] and table_id != 'spu_map':
            validate_columns(df, table_id)
        
        print(f"  成功加载: {len(df)}行 x {len(df.columns)}列")
        
        return df
    
    def load_all(self, custom_paths: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
        """
        加载所有数据表
        
        Args:
            custom_paths: 自定义文件路径 {table_id: filepath}
        
        Returns:
            {表名: DataFrame}
        """
        custom_paths = custom_paths or {}
        
        # 获取所有数据表配置
        all_tables = ['orders', 'purchase', 'first_mile', 'last_mile', 
                     'tariff', 'affiliate', 'ads', 'spu_map', 'pid_sku_map', 
                     'sku_category_map', 'excluded_orders']
        
        for table_id in all_tables:
            try:
                self.data[table_id] = self.load_table(table_id, custom_paths.get(table_id))
            except DataLoadError as e:
                # 检查是否是必须表
                source_config = self.config.get_data_source(table_id)
                if source_config and source_config['required']:
                    raise
                else:
                    print(f"  跳过: {table_id}（{e}）")
                    self.data[table_id] = pd.DataFrame()
        
        return self.data
    
    def get_data(self) -> Dict[str, pd.DataFrame]:
        """获取已加载的数据"""
        return self.data
    
    def print_summary(self):
        """打印数据摘要"""
        print("\n" + "=" * 60)
        print("数据加载摘要")
        print("=" * 60)
        
        for table_id, df in self.data.items():
            if df.empty:
                status = "空"
            else:
                status = f"{len(df)}行 x {len(df.columns)}列"
            print(f"  {table_id:15s}: {status}")


if __name__ == '__main__':
    # 测试数据加载
    print("数据加载器测试")
    print("=" * 60)
    
    loader = DataLoader()
    
    try:
        data = loader.load_all()
        loader.print_summary()
        
        # 打印orders表的前几行
        if 'orders' in data and not data['orders'].empty:
            print("\n【orders表样例】")
            print(data['orders'].head())
            
    except Exception as e:
        print(f"\n加载失败: {e}")
