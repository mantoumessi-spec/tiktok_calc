"""
html_report.py
功能：生成HTML利润报告
排版参考：profit_report_20260507_093253.html
"""

import pandas as pd
from typing import Dict


def _css() -> str:
    return """
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC','Microsoft YaHei',sans-serif;background:#f0f2f5;color:#2c3e50;line-height:1.6;font-size:14px}

/* 页头 */
.page-header{background:linear-gradient(135deg,#1a252f 0%,#2c3e50 50%,#1a5276 100%);color:#fff;padding:24px 32px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px}
.header-title{font-size:24px;font-weight:700;letter-spacing:1px}
.header-meta{font-size:13px;opacity:.8;margin-top:4px}
.header-kpis{display:flex;gap:20px;flex-wrap:wrap;margin-top:12px}
.kpi-card{background:rgba(255,255,255,.12);border-radius:8px;padding:12px 20px;text-align:center;min-width:120px}
.kpi-label{font-size:12px;opacity:.8;margin-bottom:4px}
.kpi-value{font-size:20px;font-weight:700}

/* Tab导航 */
.main-nav{background:#fff;padding:0 24px;border-bottom:1px solid #e5e7eb;position:sticky;top:0;z-index:100}
.main-nav-inner{display:flex;gap:4px;max-width:1400px;margin:0 auto}
.main-tab-btn{padding:16px 24px;border:none;background:transparent;color:#64748b;font-size:14px;font-weight:600;cursor:pointer;transition:all .2s;border-bottom:3px solid transparent;margin-bottom:-1px}
.main-tab-btn:hover{color:#334155}
.main-tab-btn.active{color:#2563eb;border-bottom-color:#2563eb}
.main-tab-content{display:none;padding-bottom:24px}
.main-tab-content.active{display:block}

/* 区块 */
.section{background:#fff;border-radius:10px;padding:24px;margin:16px 24px;box-shadow:0 2px 8px rgba(0,0,0,.06)}
.section-title{font-size:16px;font-weight:700;color:#1a252f;margin-bottom:16px;display:flex;align-items:center;gap:10px}

/* 表格 */
.table-wrap{overflow-x:auto}
.data-table{width:100%;border-collapse:collapse;font-size:13px;white-space:nowrap}
.data-table th{background:#2c3e50;color:#fff;padding:8px 12px;text-align:center;font-weight:600}
.data-table td{padding:7px 12px;text-align:center;border-bottom:1px solid #ecf0f1}
.val-row td{font-weight:600;color:#2c3e50}
.pct-row td{color:#7f8c8d;font-size:12px}
.profit-pos{color:#27ae60!important;font-weight:700!important}
.profit-neg{color:#e74c3c!important;font-weight:700!important}
.yoy-up{color:#27ae60;font-size:11px;margin-top:2px;font-weight:600}
.yoy-down{color:#e74c3c;font-size:11px;margin-top:2px;font-weight:600}
.yoy-flat{color:#999;font-size:11px;margin-top:2px}

/* SPU卡片 */
.spu-controls{display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap}
.ctrl-btn{padding:6px 16px;border-radius:6px;border:1px solid #bdc3c7;background:#fff;cursor:pointer;font-size:13px;color:#2c3e50;transition:all .2s}
.ctrl-btn:hover{background:#2c3e50;color:#fff;border-color:#2c3e50}
.ctrl-btn.active{background:#2c3e50;color:#fff;border-color:#2c3e50}
.spu-count{font-size:13px;font-weight:400;color:#888;margin-left:8px}
.spu-card{border-radius:8px;margin-bottom:8px;border:1px solid #e8e8e8;overflow:hidden;transition:box-shadow .2s}
.spu-card:hover{box-shadow:0 4px 12px rgba(0,0,0,.1)}
.health-green{border-left:5px solid #27ae60}
.health-yellow{border-left:5px solid #f39c12}
.health-red{border-left:5px solid #e74c3c}
.spu-header{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;cursor:pointer;background:#fafafa;flex-wrap:wrap;gap:8px}
.spu-header:hover{background:#f0f2f5}
.spu-header-left{display:flex;align-items:center;gap:10px}
.spu-toggle{font-size:11px;color:#999;transition:transform .2s;display:inline-block}
.spu-name{font-size:14px;font-weight:700;color:#1a252f}
.spu-category{font-size:12px;color:#666;background:#e8e8e8;padding:2px 8px;border-radius:4px}
.spu-header-right{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
.spu-rev-ratio{font-size:12px;color:#2563eb;font-weight:600;background:#dbeafe;padding:2px 8px;border-radius:4px}
.spu-profit-rate{font-size:14px;font-weight:700}
.spu-profit-val{font-size:15px;font-weight:700}
.spu-body{padding:16px;background:#fff;border-top:1px solid #ecf0f1}

/* 校验 */
.val-section{margin:16px 24px;background:#fff;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.06);overflow:hidden}
.val-header{padding:14px 20px;background:#f8f9fa;border-bottom:1px solid #e8e8e8}
.val-title{font-size:15px;font-weight:700;color:#2c3e50}
.val-body{padding:16px 20px}
.val-table{width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap}
.val-table th{background:#34495e;color:#fff;padding:6px 10px;text-align:center;font-weight:600;font-size:11px}
.val-table td{padding:5px 10px;text-align:center;border-bottom:1px solid #ecf0f1;color:#2c3e50}
.val-table tbody tr:hover{background:#f0f2f5}
.status-ok{color:#27ae60;font-weight:600}
.status-miss{color:#e74c3c;font-weight:600}
.yellow-cell{background:#fff3cd!important}

/* 响应式 */
@media(max-width:768px){
  .section{margin:12px}
  .spu-header{flex-direction:column;align-items:flex-start}
}
"""


def _format_money(val):
    if pd.isna(val) or val == '':
        return ''
    try:
        return f"${float(val):,.0f}"
    except:
        return str(val)


def _format_yoy(yoy_str: str) -> str:
    if yoy_str == '—' or pd.isna(yoy_str) or str(yoy_str).strip() == '':
        return '<div class="yoy-flat">—</div>'
    s = str(yoy_str)
    if '↑' in s:
        return f'<div class="yoy-up">{s}</div>'
    elif '↓' in s:
        return f'<div class="yoy-down">{s}</div>'
    return f'<div class="yoy-flat">{s}</div>'


def _store_table(store_matrix: pd.DataFrame) -> str:
    """店铺汇总表格"""
    if store_matrix.empty or len(store_matrix) < 3:
        return '<p>无数据</p>'

    row_val = store_matrix.iloc[0]
    row_yoy = store_matrix.iloc[1]
    row_pct = store_matrix.iloc[2]

    cols = list(store_matrix.columns)
    # 移除备注列从表头（备注列单独处理）
    display_cols = [c for c in cols if c != '备注']

    thead = '<tr>' + ''.join(f'<th>{c}</th>' for c in display_cols) + '<th>备注</th></tr>'

    # val-row
    val_cells = []
    for col in display_cols:
        v = row_val.get(col, '')
        yoy = row_yoy.get(col, '—')
        if col in ['退款后销量', '退款前销量']:
            cell = f'<div>{int(v) if v else 0} 件</div>{_format_yoy(yoy)}'
        elif col == 'ASP':
            cell = f'<div>${v}</div>{_format_yoy(yoy)}'
        elif col == '毛利率':
            cell = f'{v}'
        else:
            cell = f'{_format_money(v)}{_format_yoy(yoy)}'
        val_cells.append(f'<td>{cell}</td>')
    val_cells.append('<td></td>')
    val_row = '<tr class="val-row">' + ''.join(val_cells) + '</tr>'

    # pct-row
    pct_cells = []
    for col in display_cols:
        p = row_pct.get(col, '—')
        cls = ''
        if col == '毛利润':
            try:
                profit_pct = float(str(p).replace('%', ''))
                cls = 'profit-pos' if profit_pct >= 0 else 'profit-neg'
            except:
                pass
        pct_cells.append(f'<td class="{cls}">{p}</td>')
    pct_cells.append('<td></td>')
    pct_row = '<tr class="pct-row">' + ''.join(pct_cells) + '</tr>'

    return f'<table class="data-table"><thead>{thead}</thead><tbody>{val_row}{pct_row}</tbody></table>'


def _spu_cards(spu_matrix: pd.DataFrame) -> str:
    """SPU卡片列表"""
    if spu_matrix.empty:
        return '<p>无数据</p>'

    cards = []
    # 每3行为一个SPU
    for i in range(0, len(spu_matrix), 3):
        if i + 2 >= len(spu_matrix):
            break
        row_val = spu_matrix.iloc[i]
        row_yoy = spu_matrix.iloc[i + 1]
        row_pct = spu_matrix.iloc[i + 2]

        spu = row_val.get('SPU', '')
        cat = row_val.get('三级类目', '')
        rev_ratio = row_val.get('营收占比', '')
        profit = row_val.get('毛利润', 0)
        margin = row_val.get('毛利率', '0%')
        net_rev = row_val.get('退款后营收', 0)
        ad_cost = row_val.get('广告费', 0)
        refund_amt = row_val.get('退款金额', 0)
        gross_rev = row_val.get('退款前营收', 0)

        # 计算广告占比和退款率
        ad_ratio = f"{ad_cost / net_rev * 100:.1f}%" if net_rev and net_rev > 0 else "0.0%"
        refund_rate = f"{refund_amt / gross_rev * 100:.1f}%" if gross_rev and gross_rev > 0 else "0.0%"

        # 健康状态
        try:
            margin_val = float(str(margin).replace('%', ''))
            health = 'health-green' if margin_val >= 0 else 'health-red'
        except:
            health = 'health-green'

        # 卡片内表格
        display_cols = ['三级类目', '退款前营收', '退款后营收', 'ASP', '退款前销量', '退款后销量',
                        '采购成本', '头程成本', '尾程成本', '关税成本', '联盟佣金', '广告费',
                        '平台佣金', '其他费用', '样品费', '退款金额', '毛利润', '备注']

        thead = '<tr>' + ''.join(f'<th>{c}</th>' for c in display_cols) + '</tr>'

        val_cells = []
        for col in display_cols:
            v = row_val.get(col, '')
            yoy = row_yoy.get(col, '—')
            if col in ['退款后销量', '退款前销量']:
                cell = f'<div>{int(v) if v else 0}</div>{_format_yoy(yoy)}'
            elif col == 'ASP':
                cell = f'<div>${v}</div>{_format_yoy(yoy)}'
            elif col == '三级类目':
                cell = f'{v}'
            elif col == '备注':
                cell = ''
            else:
                cell = f'{_format_money(v)}{_format_yoy(yoy)}'
            val_cells.append(f'<td>{cell}</td>')
        val_row = '<tr class="val-row">' + ''.join(val_cells) + '</tr>'

        pct_cells = []
        for col in display_cols:
            p = row_pct.get(col, '—')
            cls = ''
            if col == '毛利润':
                try:
                    profit_pct = float(str(p).replace('%', ''))
                    cls = 'profit-pos' if profit_pct >= 0 else 'profit-neg'
                except:
                    pass
            pct_cells.append(f'<td class="{cls}">{p}</td>')
        pct_row = '<tr class="pct-row">' + ''.join(pct_cells) + '</tr>'

        inner_table = f'<table class="data-table"><thead>{thead}</thead><tbody>{val_row}{pct_row}</tbody></table>'

        profit_color = 'profit-pos' if profit >= 0 else 'profit-neg'
        profit_str = f"${_format_money(profit).replace('$','')}" if profit >= 0 else f"-${abs(profit):,.0f}"

        card = f'''
<div class="spu-card {health}" id="spu-{spu}" data-revenue="{net_rev}">
  <div class="spu-header" onclick="toggleSPU('spu-{spu}')">
    <div class="spu-header-left">
      <span class="spu-toggle" id="toggle-spu-{spu}">▶</span>
      <span class="spu-name">{spu}</span>
      <span class="spu-category">{cat}</span>
    </div>
    <div class="spu-header-right">
      <span class="spu-rev-ratio">营收占比 {rev_ratio}</span>
      <span style="font-size:12px;color:#8e44ad">广告占比 {ad_ratio}</span>
      <span style="font-size:12px;color:#e67e22">退款率 {refund_rate}</span>
      <span class="spu-profit-rate {profit_color}">{margin}</span>
      <span class="spu-profit-val {profit_color}">{profit_str}</span>
    </div>
  </div>
  <div class="spu-body" id="body-spu-{spu}" style="display:none">
    <div class="table-wrap">{inner_table}</div>
  </div>
</div>
'''
        cards.append(card)

    # 控制按钮
    controls = f'''
<div class="spu-controls" style="flex-wrap:wrap;gap:8px;">
  <button class="ctrl-btn active" onclick="expandAll()">展开全部</button>
  <button class="ctrl-btn" onclick="collapseAll()">收起全部</button>
</div>
'''
    return controls + '\n'.join(cards)


def _category_table(category_matrix: pd.DataFrame) -> str:
    """三级类目汇总表格"""
    if category_matrix.empty:
        return '<p>无数据</p>'

    display_cols = ['三级类目', '营收占比', '退款前营收', '退款后营收', 'ASP', '退款前销量', '退款后销量',
                    '采购成本', '头程成本', '尾程成本', '关税成本', '联盟佣金', '广告费',
                    '平台佣金', '其他费用', '样品费', '退款金额', '毛利润', '毛利率', '备注']

    thead = '<tr>' + ''.join(f'<th>{c}</th>' for c in display_cols) + '</tr>'
    tbody = []

    for i in range(0, len(category_matrix), 3):
        if i + 2 >= len(category_matrix):
            break
        row_val = category_matrix.iloc[i]
        row_yoy = category_matrix.iloc[i + 1]
        row_pct = category_matrix.iloc[i + 2]

        val_cells = []
        for col in display_cols:
            v = row_val.get(col, '')
            yoy = row_yoy.get(col, '—')
            if col in ['退款后销量', '退款前销量']:
                cell = f'<div>{int(v) if v else 0}</div>{_format_yoy(yoy)}'
            elif col == 'ASP':
                cell = f'<div>${v}</div>{_format_yoy(yoy)}'
            elif col in ['三级类目', '营收占比', '备注']:
                cell = f'{v}'
            else:
                cell = f'{_format_money(v)}{_format_yoy(yoy)}'
            val_cells.append(f'<td>{cell}</td>')
        tbody.append('<tr class="val-row">' + ''.join(val_cells) + '</tr>')

        pct_cells = []
        for col in display_cols:
            p = row_pct.get(col, '—')
            cls = ''
            if col == '毛利润':
                try:
                    profit_pct = float(str(p).replace('%', ''))
                    cls = 'profit-pos' if profit_pct >= 0 else 'profit-neg'
                except:
                    pass
            pct_cells.append(f'<td class="{cls}">{p}</td>')
        tbody.append('<tr class="pct-row">' + ''.join(pct_cells) + '</tr>')

    return f'<table class="data-table"><thead>{thead}</thead><tbody>{"".join(tbody)}</tbody></table>'


def _validation_section(validation: Dict[str, pd.DataFrame]) -> str:
    """数据校验区块"""
    sections = []

    # 1. SKU成本完整性
    df = validation.get('sku_cost', pd.DataFrame())
    if not df.empty:
        rows = []
        for _, r in df.iterrows():
            cls = 'status-ok' if r.get('状态') == '完整' else 'status-miss'
            cells = ''.join(f'<td>{r.get(c, "")}</td>' for c in df.columns)
            # 标黄缺失项
            yellow_style = ' class="yellow-cell"' if r.get('状态') == '缺失' else ''
            rows.append(f'<tr{yellow_style}>{cells}</tr>')
        table = f'<table class="val-table"><thead><tr>{"".join(f"<th>{c}</th>" for c in df.columns)}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
        sections.append(f'<div class="val-section"><div class="val-header"><div class="val-title">📦 SKU成本完整性校验</div></div><div class="val-body">{table}</div></div>')

    # 2. 联盟佣金校验
    df = validation.get('affiliate', pd.DataFrame())
    if not df.empty:
        rows = ''.join(f'<tr><td style="text-align:left;font-weight:600">{r["校验项"]}</td><td style="font-weight:700">{r["结果"]}</td></tr>' for _, r in df.iterrows())
        table = f'<table class="val-table" style="max-width:400px"><tbody>{rows}</tbody></table>'
        sections.append(f'<div class="val-section"><div class="val-header"><div class="val-title">🤝 联盟佣金校验</div></div><div class="val-body">{table}</div></div>')

    # 3. 广告费校验
    df = validation.get('ads', pd.DataFrame())
    if not df.empty:
        rows = ''.join(f'<tr><td style="text-align:left;font-weight:600">{r["校验项"]}</td><td style="font-weight:700">{r["结果"]}</td></tr>' for _, r in df.iterrows())
        table = f'<table class="val-table" style="max-width:400px"><tbody>{rows}</tbody></table>'
        sections.append(f'<div class="val-section"><div class="val-header"><div class="val-title">📢 广告费校验</div></div><div class="val-body">{table}</div></div>')

    # 4. 广告未匹配明细
    df = validation.get('ads_unmapped', pd.DataFrame())
    if not df.empty:
        thead = '<tr>' + ''.join(f'<th>{c}</th>' for c in df.columns) + '</tr>'
        tbody = ''.join(f'<tr>{"".join(f"<td>{r[c]}</td>" for c in df.columns)}</tr>' for _, r in df.iterrows())
        table = f'<table class="val-table"><thead>{thead}</thead><tbody>{tbody}</tbody></table>'
        sections.append(f'<div class="val-section"><div class="val-header"><div class="val-title">📢 广告费未匹配明细</div></div><div class="val-body">{table}</div></div>')

    # 5. 其他校验
    df = validation.get('other', pd.DataFrame())
    if not df.empty:
        rows = ''.join(f'<tr><td style="text-align:left;font-weight:600">{r["校验项"]}</td><td style="font-weight:700">{r["结果"]}</td></tr>' for _, r in df.iterrows())
        table = f'<table class="val-table" style="max-width:500px"><tbody>{rows}</tbody></table>'
        sections.append(f'<div class="val-section"><div class="val-header"><div class="val-title">📋 其他校验</div></div><div class="val-body">{table}</div></div>')

    return '\n'.join(sections)


def _js() -> str:
    return """
function switchMainTab(tabId) {
    document.querySelectorAll('.main-tab-content').forEach(function(el) { el.classList.remove('active'); });
    document.querySelectorAll('.main-tab-btn').forEach(function(el) { el.classList.remove('active'); });
    document.getElementById('main-tab-' + tabId).classList.add('active');
    event.target.classList.add('active');
}
function toggleSPU(spuId) {
    var body = document.getElementById('body-' + spuId);
    var toggle = document.getElementById('toggle-' + spuId);
    if (body.style.display === 'none') {
        body.style.display = 'block';
        toggle.textContent = '▼';
    } else {
        body.style.display = 'none';
        toggle.textContent = '▶';
    }
}
function expandAll() {
    document.querySelectorAll('.spu-body').forEach(function(el) { el.style.display = 'block'; });
    document.querySelectorAll('.spu-toggle').forEach(function(el) { el.textContent = '▼'; });
}
function collapseAll() {
    document.querySelectorAll('.spu-body').forEach(function(el) { el.style.display = 'none'; });
    document.querySelectorAll('.spu-toggle').forEach(function(el) { el.textContent = '▶'; });
}
"""


def generate_html(result, validation, output_path: str, report_period: str, timestamp: str,
                  summary_stats: dict) -> str:
    """生成HTML报告"""

    # KPI数据
    net_revenue = summary_stats.get('总净销售收入', 0)
    gross_profit = summary_stats.get('总毛利润', 0)
    margin = summary_stats.get('整体毛利率', 0) * 100
    total_qty = summary_stats.get('总销售数量', 0)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>爆单猫-TikTok利润测算报告</title>
<style>
{_css()}
</style>
</head>
<body>

<div class="page-header">
  <div class="header-left">
    <div class="header-title">爆单猫-TikTok利润测算报告</div>
    <div class="header-meta">生成时间：{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]} &nbsp;|&nbsp; 数据周期：{report_period or '-'}</div>
    <div class="header-kpis">
      <div class="kpi-card">
        <div class="kpi-label">退款后营收</div>
        <div class="kpi-value">${net_revenue:,.0f}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">毛利润</div>
        <div class="kpi-value" style="color:{'#27ae60' if gross_profit >= 0 else '#e74c3c'}">${gross_profit:,.0f}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">毛利率</div>
        <div class="kpi-value" style="color:{'#27ae60' if margin >= 0 else '#e74c3c'}">{margin:.1f}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">销售数量</div>
        <div class="kpi-value">{int(total_qty)} 件</div>
      </div>
    </div>
  </div>
</div>

<nav class="main-nav">
    <div class="main-nav-inner">
        <button class="main-tab-btn active" onclick="switchMainTab('profit')">📊 利润概览</button>
        <button class="main-tab-btn" onclick="switchMainTab('validation')">📋 数据校验</button>
    </div>
</nav>

<div id="main-tab-profit" class="main-tab-content active">

<div class="section">
  <div class="section-title">📋 店铺利润汇总</div>
  <div class="table-wrap">
    {_store_table(result.store_matrix)}
  </div>
</div>

<div class="section">
  <div class="section-title">📦 SPU 利润明细 <span class="spu-count">共 {len(result.spu_matrix) // 3} 个 SPU</span></div>
  {_spu_cards(result.spu_matrix)}
</div>

<div class="section">
  <div class="section-title">📂 三级类目利润汇总</div>
  <div class="table-wrap">
    {_category_table(result.category_matrix)}
  </div>
</div>

</div>

<div id="main-tab-validation" class="main-tab-content">
{_validation_section(validation)}
</div>

<script>
{_js()}
</script>

</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  HTML报告: {output_path}")
    return output_path
