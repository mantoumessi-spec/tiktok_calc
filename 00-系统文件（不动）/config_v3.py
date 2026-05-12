"""
config_v3.py
功能：配置管理
"""

import pandas as pd
import os
from typing import Dict, List

# 路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DATA_DIR = os.path.join(BASE_DIR, '01-基础数据（年度更新）')
CURRENT_DATA_DIR = os.path.join(BASE_DIR, '02-本期数据（每次测算更新）')

# 文件映射
FILE_MAPPING = {
    'orders': ('店铺订单.xlsx', '02-本期数据（每次测算更新）'),
    'purchase': ('采购成本.xlsx', '02-本期数据（每次测算更新）'),
    'first_mile': ('采购成本.xlsx', '02-本期数据（每次测算更新）'),
    'last_mile': ('尾程成本.xlsx', '02-本期数据（每次测算更新）'),
    'tariff': ('关税成本.xlsx', '02-本期数据（每次测算更新）'),
    'affiliate': ('联盟订单.xlsx', '02-本期数据（每次测算更新）'),
    'ads': ('广告订单.xlsx', '02-本期数据（每次测算更新）'),
    'spu_map': ('spu_sku映射表.xlsx', '01-基础数据（年度更新）'),
    'pid_sku_map': ('pid_sku映射表.xlsx', '01-基础数据（年度更新）'),
    'sku_category_map': ('SKU-三级类目映射.xlsx', '01-基础数据（年度更新）'),
    'excluded_orders': ('需排除的订单编号.xlsx', '02-本期数据（每次测算更新）'),
}


class FormulaConfig:
    """配置管理"""
    
    def get_columns(self, table_name: str) -> List[str]:
        """获取列配置（已简化，返回空列表由代码自动适配）"""
        return []
    
    def get_all_columns(self, table_name: str) -> List[str]:
        """获取所有列"""
        return self.get_columns(table_name)
    
    def get_data_source(self, table_id: str) -> dict:
        """获取数据源配置"""
        desc_map = {
            'orders': '店铺订单数据',
            'purchase': '采购成本数据',
            'first_mile': '头程物流成本',
            'last_mile': '尾程物流成本',
            'tariff': '关税成本数据',
            'affiliate': '联盟佣金数据',
            'ads': '广告花费数据',
            'spu_map': 'SPU-SKU映射表',
            'pid_sku_map': 'PID-SKU映射表',
            'sku_category_map': 'SKU-三级类目映射表',
        }
        if table_id in FILE_MAPPING:
            filename, folder = FILE_MAPPING[table_id]
            return {
                'file': filename,
                'folder': folder,
                'filepath': os.path.join(BASE_DIR, folder, filename),
                'required': True,
                'desc': desc_map.get(table_id, table_id)
            }
        return {'file': '', 'folder': '', 'filepath': '', 'required': False, 'desc': ''}


# 单例
_config_instance = None

def get_config():
    """获取配置实例"""
    global _config_instance
    if _config_instance is None:
        _config_instance = FormulaConfig()
    return _config_instance
