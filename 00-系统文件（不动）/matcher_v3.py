"""
matcher_v3.py
功能：数据匹配和验证
"""

import pandas as pd
from typing import List

class MatchError(Exception):
    """匹配错误"""
    pass

def validate_columns(df: pd.DataFrame, required_columns, table_name: str = '') -> bool:
    """
    验证DataFrame是否包含必需的列
    
    Args:
        df: DataFrame
        required_columns: 必需的列名列表或字符串
        table_name: 表名（用于错误信息）
    
    Returns:
        bool: 验证是否通过
    """
    # 如果不是列表，尝试转换或跳过验证
    if not isinstance(required_columns, list):
        return True
    
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise MatchError(f"表 {table_name} 缺少必需的列: {missing}")
    return True
