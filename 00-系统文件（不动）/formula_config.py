"""
formula_config.py
功能：从计算公式详细说明.xlsx读取公式配置
"""

import pandas as pd
import re
from typing import Dict, Any


class FormulaConfig:
    """公式配置管理器 - 从Excel文件读取所有计算规则"""
    
    def __init__(self, formula_file: str = "00-系统文件（不动）/计算公式详细说明.xlsx"):
        self.formula_file = formula_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """从Excel文件加载所有配置"""
        import os
        
        config = {
            'formulas': {},      # 费用项 -> 公式详情
            'sample_steps': [],  # 样品费计算步骤
            'params': {          # 关键参数
                'exchange_rate': 7.2,  # 默认汇率
                'platform_commission_rate': 0.09,  # 平台佣金率
                'other_fee_rate': 0.015,  # 其他费用率
                'nan_sku_fallback': 'ERL001-BK-R-PN',  # NAN SKU默认映射
                # 广告问题归因阈值
                'diagnosis_ctr_low': 0.015,      # CTR低于此值判定为"CTR偏低"
                'diagnosis_cvr_low': 0.02,       # CVR低于此值判定为"CVR偏低"
                'diagnosis_cpm_high': 10.0,      # CPM高于此值判定为"CPM偏高"
                'diagnosis_roas_breakeven': 2.5, # ROAS低于此值触发归因诊断
                # 备注占位文字
                'note_placeholder': '点击添加备注…',  # 表格备注单元格占位提示
            }
        }
        
        # ── 优先从「基础参数表.xlsx」读取参数（用户可编辑）──
        base_param_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         '01-基础数据（年度更新）', '基础参数表.xlsx'),
            '01-基础数据（年度更新）/基础参数表.xlsx',
        ]
        param_file = None
        for path in base_param_paths:
            if os.path.exists(path):
                param_file = path
                break
        
        if param_file:
            try:
                df_params = pd.read_excel(param_file, sheet_name='基础参数')
                for _, row in df_params.iterrows():
                    name = str(row['参数名称']).strip() if pd.notna(row['参数名称']) else ''
                    val = row['参数值']
                    if name == '汇率' and pd.notna(val):
                        config['params']['exchange_rate'] = float(val)
                    elif name == '平台佣金率' and pd.notna(val):
                        config['params']['platform_commission_rate'] = float(val)
                    elif name == '其他费用率' and pd.notna(val):
                        config['params']['other_fee_rate'] = float(val)
                print(f"  ✅ 已从基础参数表加载参数: {param_file}")
            except Exception as e:
                print(f"  ⚠️ 基础参数表读取失败: {e}，使用默认参数")
        else:
            print("  ⚠️ 未找到基础参数表，使用默认参数")
        
        try:
            # 读取费用计算公式汇总
            df_formulas = pd.read_excel(self.formula_file, sheet_name='费用计算公式汇总')
            
            for _, row in df_formulas.iterrows():
                item_name = str(row['费用项']).strip() if pd.notna(row['费用项']) else ''
                if item_name:
                    config['formulas'][item_name] = {
                        'category': str(row['类别']).strip() if pd.notna(row['类别']) else '',
                        'formula': str(row['公式']).strip() if pd.notna(row['公式']) else '',
                        'description': str(row['详细说明']).strip() if pd.notna(row['详细说明']) else '',
                        'data_file': str(row['数据文件']).strip() if pd.notna(row['数据文件']) else '',
                        'column': str(row['列名']).strip() if pd.notna(row['列名']) else '',
                        'match_rule': str(row['匹配规则']).strip() if pd.notna(row['匹配规则']) else '',
                        'unit': str(row['单位']).strip() if pd.notna(row['单位']) else ''
                    }
                    
                    # 从公式中提取关键参数
                    self._extract_params(config['params'], row['公式'])
            
            # 读取样品费计算步骤
            df_steps = pd.read_excel(self.formula_file, sheet_name='样品费计算步骤')
            for _, row in df_steps.iterrows():
                config['sample_steps'].append({
                    'step': row['步骤'],
                    'operation': str(row['操作']).strip() if pd.notna(row['操作']) else '',
                    'description': str(row['说明']).strip() if pd.notna(row['说明']) else '',
                    'data_source': str(row['数据来源']).strip() if pd.notna(row['数据来源']) else '',
                    'condition': str(row['列/条件']).strip() if pd.notna(row['列/条件']) else ''
                })
                # 从列/条件中提取汇率（如"除以汇率7.1"）
                if pd.notna(row['列/条件']):
                    self._extract_params(config['params'], row['列/条件'])
                
        except Exception as e:
            print(f"警告: 无法读取公式配置文件: {e}")
            print("使用默认配置")
        
        return config
    
    def _extract_params(self, params: Dict, formula: str):
        """从公式中提取关键参数"""
        if pd.isna(formula):
            return
        
        formula_str = str(formula).lower()
        
        # 提取汇率 - 支持 "除以汇率7.1" 或 "汇率 7.1" 格式
        if '汇率' in formula_str:
            # 匹配 "除以汇率7.1" 或 "汇率 7.1" 或 "汇率: 7.1" 等格式
            match = re.search(r'汇率\s*:?\s*(\d+\.?\d*)', formula_str)
            if match:
                params['exchange_rate'] = float(match.group(1))
        
        # 提取平台佣金率
        if '平台佣金' in formula_str or '9%' in formula_str:
            match = re.search(r'(\d+\.?\d*)\s*%', formula_str)
            if match:
                params['platform_commission_rate'] = float(match.group(1)) / 100
        
        # 提取其他费用率
        if '其他费用' in formula_str or '1.5%' in formula_str:
            match = re.search(r'(\d+\.?\d*)\s*%', formula_str)
            if match:
                params['other_fee_rate'] = float(match.group(1)) / 100
        
        # 提取NAN SKU映射配置
        if 'nan_sku_fallback' in formula_str or 'NAN' in formula_str.upper():
            # 匹配格式: ERL001-BK-R-PN 或类似SKU格式
            match = re.search(r'([A-Z]{2,}\d{3,}[A-Z0-9-]+)', str(formula_str).upper())
            if match:
                params['nan_sku_fallback'] = match.group(1)
    
    def get_formula(self, item_name: str) -> Dict[str, str]:
        """获取指定费用项的公式配置"""
        return self.config['formulas'].get(item_name, {})
    
    def get_param(self, param_name: str) -> Any:
        """获取配置参数"""
        return self.config['params'].get(param_name)
    
    def get_all_formulas(self) -> Dict[str, Dict[str, str]]:
        """获取所有公式配置"""
        return self.config['formulas']
    
    def get_sample_steps(self) -> list:
        """获取样品费计算步骤"""
        return self.config['sample_steps']
    
    def print_config(self):
        """打印当前配置"""
        print("=" * 80)
        print("【公式配置】")
        print("=" * 80)
        print(f"\n关键参数:")
        for k, v in self.config['params'].items():
            print(f"  {k}: {v}")
        
        print(f"\n已加载公式项 ({len(self.config['formulas'])}个):")
        for name, cfg in self.config['formulas'].items():
            print(f"  - {name}: {cfg.get('formula', 'N/A')[:50]}...")


# 单例模式
_formula_config = None

def get_formula_config(formula_file: str = None) -> FormulaConfig:
    """获取公式配置实例（单例）"""
    global _formula_config
    if _formula_config is None:
        _formula_config = FormulaConfig(formula_file)
    return _formula_config


if __name__ == '__main__':
    # 测试配置加载
    config = get_formula_config()
    config.print_config()
