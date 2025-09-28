import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dateutil.relativedelta import relativedelta

DATA_DIR = "data"
PROJECT_CSV = os.path.join(DATA_DIR, "projects.csv")
MASTERS_JSON = os.path.join(DATA_DIR, "masters.json")
FISCAL_START_MONTH = 7
DEFAULT_FISCAL_YEAR = 2025
FISCAL_YEAR_OPTIONS = list(range(2024, 2029))
DEFAULT_BAR_COLOR = "#E67E22"

PROJECT_NUMERIC_COLUMNS = [
    "受注予定額",
    "受注金額",
    "予算原価",
    "予定原価",
    "実績原価",
    "粗利率",
    "進捗率",
    "月平均必要人数",
]

PROJECT_DATE_COLUMNS = [
    "着工日",
    "竣工日",
    "実際着工日",
    "実際竣工日",
    "回収開始日",
    "回収終了日",
    "支払開始日",
    "支払終了日",
]

PROJECT_BASE_COLUMNS = [
    "id",
    "案件名",
    "得意先",
    "元請区分",
    "工種",
    "ステータス",
    *PROJECT_DATE_COLUMNS,
    *PROJECT_NUMERIC_COLUMNS,
    "現場所在地",
    "担当者",
    "協力会社",
    "備考",
    "リスクメモ",
]


@dataclass
class FilterState:
    fiscal_year: int
    period_from: Optional[date]
    period_to: Optional[date]
    status: List[str]
    category: List[str]
    contractor_level: List[str]
    client: List[str]
    manager: List[str]
    prefecture: List[str]
    margin_range: Tuple[float, float]
    filter_mode: str
    search_text: str
    search_targets: List[str]
    color_key: str
    show_grid: bool
    label_density: str
    bar_color: str


def ensure_data_files() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(PROJECT_CSV):
        sample = pd.DataFrame(
            [
                {
                    "id": "P001",
                    "案件名": "高田小学校 体育館 新築 型枠工事",
                    "得意先": "金子技建",
                    "元請区分": "二次",
                    "工種": "型枠",
                    "ステータス": "施工中",
                    "着工日": "2025-07-10",
                    "竣工日": "2025-10-30",
                    "実際着工日": "2025-07-12",
                    "実際竣工日": "",
                    "受注予定額": 24000000,
                    "受注金額": 25000000,
                    "予算原価": 18000000,
                    "予定原価": 19000000,
                    "実績原価": 13500000,
                    "粗利率": 24,
                    "進捗率": 55,
                    "月平均必要人数": 6,
                    "回収開始日": "2025-08-15",
                    "回収終了日": "2025-11-30",
                    "支払開始日": "2025-07-31",
                    "支払終了日": "2025-12-15",
                    "現場所在地": "福岡",
                    "担当者": "山中",
                    "協力会社": "九州型枠工業",
                    "備考": "体育館の基礎および型枠一式",
                    "リスクメモ": "鉄筋納期に注意",
                },
                {
                    "id": "P002",
                    "案件名": "熊本・橋脚下部工（P3・フーチング）",
                    "得意先": "佐藤組",
                    "元請区分": "一次",
                    "工種": "土木",
                    "ステータス": "施工中",
                    "着工日": "2025-08-01",
                    "竣工日": "2025-12-20",
                    "実際着工日": "2025-08-05",
                    "実際竣工日": "",
                    "受注予定額": 33000000,
                    "受注金額": 32000000,
                    "予算原価": 25000000,
                    "予定原価": 24500000,
                    "実績原価": 16200000,
                    "粗利率": 23,
                    "進捗率": 48,
                    "月平均必要人数": 7,
                    "回収開始日": "2025-09-01",
                    "回収終了日": "2026-01-31",
                    "支払開始日": "2025-08-31",
                    "支払終了日": "2026-02-28",
                    "現場所在地": "熊本",
                    "担当者": "近藤",
                    "協力会社": "熊本土木サービス",
                    "備考": "河川敷工事の夜間作業あり",
                    "リスクメモ": "増水時は待機",
                },
                {
                    "id": "P003",
                    "案件名": "下大利 5階建（商住複合）",
                    "得意先": "新宮開発",
                    "元請区分": "二次",
                    "工種": "建築",
                    "ステータス": "受注",
                    "着工日": "2025-09-15",
                    "竣工日": "2026-02-28",
                    "実際着工日": "",
                    "実際竣工日": "",
                    "受注予定額": 57000000,
                    "受注金額": 58000000,
                    "予算原価": 43000000,
                    "予定原価": 44000000,
                    "実績原価": 0,
                    "粗利率": 24,
                    "進捗率": 10,
                    "月平均必要人数": 8,
                    "回収開始日": "2025-10-01",
                    "回収終了日": "2026-04-30",
                    "支払開始日": "2025-09-30",
                    "支払終了日": "2026-05-31",
                    "現場所在地": "福岡",
                    "担当者": "山中",
                    "協力会社": "九州建設パートナーズ",
                    "備考": "地下躯体に注意",
                    "リスクメモ": "地中障害物調査待ち",
                },
                {
                    "id": "P004",
                    "案件名": "みやま市 動物愛護施設（JV）",
                    "得意先": "金子技建",
                    "元請区分": "一次",
                    "工種": "建築",
                    "ステータス": "見積",
                    "着工日": "2025-11-15",
                    "竣工日": "2026-05-31",
                    "実際着工日": "",
                    "実際竣工日": "",
                    "受注予定額": 58000000,
                    "受注金額": 60000000,
                    "予算原価": 45000000,
                    "予定原価": 46000000,
                    "実績原価": 0,
                    "粗利率": 23,
                    "進捗率": 5,
                    "月平均必要人数": 9,
                    "回収開始日": "2026-01-15",
                    "回収終了日": "2026-06-30",
                    "支払開始日": "2025-11-30",
                    "支払終了日": "2026-07-15",
                    "現場所在地": "福岡",
                    "担当者": "山中",
                    "協力会社": "九州建設パートナーズ",
                    "備考": "JV案件",
                    "リスクメモ": "JV調整会議が必要",
                },
                {
                    "id": "P005",
                    "案件名": "朝倉市 私立病院 新設",
                    "得意先": "高野組",
                    "元請区分": "二次",
                    "工種": "建築",
                    "ステータス": "見積",
                    "着工日": "2025-12-01",
                    "竣工日": "2026-06-15",
                    "実際着工日": "",
                    "実際竣工日": "",
                    "受注予定額": 47000000,
                    "受注金額": 45000000,
                    "予算原価": 34000000,
                    "予定原価": 35000000,
                    "実績原価": 0,
                    "粗利率": 22,
                    "進捗率": 0,
                    "月平均必要人数": 7,
                    "回収開始日": "2026-02-01",
                    "回収終了日": "2026-07-31",
                    "支払開始日": "2025-12-31",
                    "支払終了日": "2026-08-31",
                    "現場所在地": "福岡",
                    "担当者": "近藤",
                    "協力会社": "九州医療建設",
                    "備考": "未定要素あり",
                    "リスクメモ": "医療機器仕様待ち",
                },
            ]
        )

    if not os.path.exists(MASTERS_JSON):
        masters = {
            "clients": [{"name": name, "active": True} for name in ["金子技建", "佐藤組", "新宮開発", "高野組"]],
            "categories": [{"name": name, "active": True} for name in ["建築", "土木", "型枠", "その他"]],
            "managers": [{"name": name, "active": True} for name in ["山中", "近藤", "田中"]],
            "holidays": [],
            "currency_format": "#,###",
            "decimal_places": 0,
            "history": [],
        }
        with open(MASTERS_JSON, "w", encoding="utf-8") as f:
            json.dump(masters, f, ensure_ascii=False, indent=2)


def normalize_master_entries(entries: List) -> List[Dict[str, object]]:
    normalized: List[Dict[str, object]] = []
    seen = set()
    for entry in entries or []:
        if isinstance(entry, dict):
            name = str(entry.get("name", "")).strip()
            active = bool(entry.get("active", True))
        else:
            name = str(entry).strip()
            active = True
        if not name or name in seen:
            continue
        normalized.append({"name": name, "active": active})
        seen.add(name)
    return normalized


def ensure_master_structure(masters: Dict[str, List]) -> Dict[str, List]:
    masters = masters or {}
    for key in ["clients", "categories", "managers"]:
        masters[key] = normalize_master_entries(masters.get(key, []))
    masters.setdefault("holidays", [])
    masters.setdefault("currency_format", "#,###")
    masters.setdefault("decimal_places", 0)
    masters.setdefault("history", [])
    return masters


def get_active_master_values(masters: Dict[str, List], key: str) -> List[str]:
    return [entry["name"] for entry in masters.get(key, []) if entry.get("active", True)]


def load_masters() -> Dict[str, List]:
    with open(MASTERS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ensure_master_structure(data)


def save_masters(masters: Dict[str, List]) -> None:
    with open(MASTERS_JSON, "w", encoding="utf-8") as f:
        json.dump(ensure_master_structure(masters), f, ensure_ascii=False, indent=2)


def load_projects() -> pd.DataFrame:
    df = pd.read_csv(PROJECT_CSV)
    for col in PROJECT_BASE_COLUMNS:
        if col not in df.columns:
            if col in PROJECT_DATE_COLUMNS:
                df[col] = pd.NaT
            elif col in PROJECT_NUMERIC_COLUMNS:
                df[col] = 0.0
            else:
                df[col] = ""
    for col in PROJECT_DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    for col in PROJECT_NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    missing_cols = [c for c in PROJECT_BASE_COLUMNS if c not in df.columns]
    if missing_cols:
        df = df.reindex(columns=list(df.columns) + missing_cols)
    return df[PROJECT_BASE_COLUMNS]


def save_projects(df: pd.DataFrame) -> None:
    out_df = df.copy()
    out_df = out_df.reindex(columns=PROJECT_BASE_COLUMNS)
    out_df.sort_values(by="着工日", inplace=True, ignore_index=True)
    out_df.to_csv(PROJECT_CSV, index=False)


def get_fiscal_year_range(year: int) -> Tuple[date, date]:
    start = date(year, FISCAL_START_MONTH, 1)
    end = start + relativedelta(years=1) - relativedelta(days=1)
    return start, end


def apply_filters(df: pd.DataFrame, filters: FilterState) -> pd.DataFrame:
    result = df.copy()
    if filters.period_from:
        result = result[result["竣工日"].fillna(date.min) >= filters.period_from]
    if filters.period_to:
        result = result[result["着工日"].fillna(date.max) <= filters.period_to]
    if filters.margin_range:
        low, high = filters.margin_range
        result = result[(result["粗利率"] >= low) & (result["粗利率"] <= high)]

    def build_search_condition(dataframe: pd.DataFrame) -> pd.Series:
        if not filters.search_text.strip():
            return pd.Series(True, index=dataframe.index)
        search_text = filters.search_text.strip().lower()
        columns = filters.search_targets or ["案件名", "得意先"]
        mask = pd.Series(False, index=dataframe.index)
        for col in columns:
            if col in dataframe.columns:
                mask = mask | dataframe[col].fillna("").astype(str).str.lower().str.contains(search_text)
        return mask

    if filters.filter_mode == "AND":
        if filters.status:
            result = result[result["ステータス"].isin(filters.status)]
        if filters.category:
            result = result[result["工種"].isin(filters.category)]
        if filters.contractor_level:
            result = result[result["元請区分"].isin(filters.contractor_level)]
        if filters.client:
            result = result[result["得意先"].isin(filters.client)]
        if filters.manager:
            result = result[result["担当者"].isin(filters.manager)]
        if filters.prefecture:
            result = result[result["現場所在地"].isin(filters.prefecture)]
        search_condition = build_search_condition(result)
        result = result[search_condition]
    else:
        masks: List[pd.Series] = []
        if filters.status:
            masks.append(result["ステータス"].isin(filters.status))
        if filters.category:
            masks.append(result["工種"].isin(filters.category))
        if filters.contractor_level:
            masks.append(result["元請区分"].isin(filters.contractor_level))
        if filters.client:
            masks.append(result["得意先"].isin(filters.client))
        if filters.manager:
            masks.append(result["担当者"].isin(filters.manager))
        if filters.prefecture:
            masks.append(result["現場所在地"].isin(filters.prefecture))
        search_condition = build_search_condition(result)
        if not search_condition.all():
            masks.append(search_condition)
        if masks:
            combined = masks[0]
            for m in masks[1:]:
                combined = combined | m
            result = result[combined]
    return result


def generate_color_map(values: pd.Series, key: str, default_color: str) -> Dict[str, str]:
    palettes = {
        "ステータス": ["#E67E22", "#3498DB", "#27AE60", "#9B59B6", "#F39C12", "#95A5A6"],
        "工種": ["#D35400", "#1ABC9C", "#8E44AD", "#2ECC71", "#F1C40F"],
        "元請区分": ["#E67E22", "#F39C12", "#E74C3C", "#9B59B6"],
    }
    palette = palettes.get(key, [default_color])
    unique_vals = [v for v in values.dropna().unique().tolist() if v != ""]
    color_map = {val: palette[i % len(palette)] for i, val in enumerate(unique_vals)}
    color_map["未設定"] = default_color
    return color_map


def coerce_date(value) -> Optional[date]:
    if value in (None, "", pd.NaT):
        return None
    if isinstance(value, date):
        return value
    try:
        return pd.to_datetime(value, errors="coerce").date()
    except (TypeError, ValueError):
        return None


def format_date(value) -> str:
    coerced = coerce_date(value)
    return coerced.strftime("%Y-%m-%d") if coerced else "-"


def calculate_expected_progress(row: pd.Series, today: date) -> float:
    start = coerce_date(row.get("着工日")) or coerce_date(row.get("実際着工日"))
    end = coerce_date(row.get("竣工日"))
    if not start or not end or start >= end:
        return 0.0
    if today <= start:
        return 0.0
    if today >= end:
        return 100.0
    total_days = (end - start).days
    elapsed_days = (today - start).days
    return max(0.0, min(100.0, elapsed_days / total_days * 100))


def determine_risk_level(row: pd.Series) -> Tuple[str, str]:
    risk_order = {"低": 0, "中": 1, "高": 2}
    level = "低"
    reasons: List[str] = []
    if row.get("予算超過", False):
        level = "高"
        reasons.append("予算超過")
    progress_gap = row.get("進捗差異", 0)
    if progress_gap < -30:
        level = "高"
        reasons.append("進捗大幅遅れ")
    elif progress_gap < -10 and risk_order[level] < risk_order["中"]:
        level = "中"
        reasons.append("進捗遅れ")
    delay_days = row.get("遅延日数", 0)
    if delay_days > 0:
        level = "高"
        reasons.append(f"遅延{int(delay_days)}日")
    if not reasons and row.get("リスクメモ"):
        level = "中"
        reasons.append(str(row.get("リスクメモ")))
    comment = "、".join(dict.fromkeys([r for r in reasons if r])) or "安定"
    return level, comment


def enrich_projects(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["粗利額"] = enriched["受注金額"] - enriched["予定原価"]
    with np.errstate(divide="ignore", invalid="ignore"):
        enriched["原価率"] = np.where(
            enriched["受注金額"] != 0,
            (enriched["予定原価"] / enriched["受注金額"]) * 100,
            0,
        )
    enriched["受注差異"] = enriched["受注金額"] - enriched["受注予定額"]
    enriched["予算乖離額"] = enriched["予定原価"] - enriched["予算原価"]
    enriched["予算超過"] = enriched["予算乖離額"] > 0
    enriched["完成工事高"] = enriched["受注金額"] * (enriched["進捗率"] / 100)
    enriched["実行粗利"] = enriched["受注金額"] - enriched["実績原価"]
    today = date.today()
    enriched["想定進捗率"] = enriched.apply(lambda row: calculate_expected_progress(row, today), axis=1)
    enriched["進捗差異"] = enriched["進捗率"] - enriched["想定進捗率"]
    actual_end = pd.to_datetime(enriched["実際竣工日"], errors="coerce")
    planned_end = pd.to_datetime(enriched["竣工日"], errors="coerce")
    delay = (actual_end - planned_end).dt.days
    enriched["遅延日数"] = delay.where(delay > 0, 0).fillna(0)
    levels_comments = enriched.apply(determine_risk_level, axis=1)
    enriched["リスクレベル"] = [lc[0] for lc in levels_comments]
    enriched["リスクコメント"] = [lc[1] for lc in levels_comments]
    return enriched


def allocate_value(value: float, start, end, month_start: pd.Timestamp, month_end: pd.Timestamp) -> float:
    start_date = coerce_date(start)
    end_date = coerce_date(end)
    if start_date is None or end_date is None:
        return 0.0
    total_start = pd.to_datetime(start_date)
    total_end = pd.to_datetime(end_date)
    if total_start > month_end or total_end < month_start:
        return 0.0
    overlap_start = max(total_start, month_start)
    overlap_end = min(total_end, month_end)
    if overlap_start > overlap_end:
        return 0.0
    total_days = (total_end - total_start).days + 1
    if total_days <= 0:
        return 0.0
    overlap_days = (overlap_end - overlap_start).days + 1
    return float(value) * (overlap_days / total_days)


def summarize_resources(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        empty = pd.DataFrame(columns=["担当", "必要人数"])
        return empty, empty
    manager = (
        df.groupby("担当者")["月平均必要人数"]
        .sum()
        .reset_index()
        .rename(columns={"担当者": "担当", "月平均必要人数": "必要人数"})
        .sort_values("必要人数", ascending=False)
    )
    partner = (
        df.groupby("協力会社")["月平均必要人数"]
        .sum()
        .reset_index()
        .rename(columns={"協力会社": "協力会社", "月平均必要人数": "必要人数"})
        .sort_values("必要人数", ascending=False)
    )
    return manager, partner




def create_timeline(df: pd.DataFrame, filters: FilterState, fiscal_range: Tuple[date, date]) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            xaxis_title="期間",
            yaxis_title="案件名",
            template="plotly_white",
            height=500,
        )
        return fig

    color_key = filters.color_key
    color_map = generate_color_map(df[color_key], color_key, filters.bar_color)
    legend_tracker: Dict[str, bool] = {}

    fig = go.Figure()
    for _, row in df.iterrows():
        planned_start = format_date(row.get("着工日"))
        planned_end = format_date(row.get("竣工日"))
        actual_end = format_date(row.get("実際竣工日"))
        hover_text = (
            f"案件名: {row['案件名']}<br>期間: {planned_start}〜{planned_end}<br>"
            f"得意先: {row['得意先']}<br>工種: {row['工種']}<br>ステータス: {row['ステータス']}<br>"
            f"進捗率: {row['進捗率']:.1f}% (想定 {row['想定進捗率']:.1f}%)<br>"
            f"遅延日数: {int(row['遅延日数'])}日 / 実竣工: {actual_end}<br>"
            f"粗利額: {row['粗利額']:,}円 / 原価率: {row['原価率']:.1f}%<br>"
            f"受注差異: {row['受注差異']:,}円 / 予算乖離: {row['予算乖離額']:,}円<br>"
            f"完成工事高: {row['完成工事高']:,}円 / 実行粗利: {row['実行粗利']:,}円<br>"
            f"担当者: {row['担当者']} / 協力会社: {row['協力会社']}<br>"
            f"月平均必要人数: {row['月平均必要人数']:.1f}人<br>"
            f"回収: {format_date(row.get('回収開始日'))}〜{format_date(row.get('回収終了日'))}<br>"
            f"支払: {format_date(row.get('支払開始日'))}〜{format_date(row.get('支払終了日'))}<br>"
            f"リスク: {row['リスクレベル']} ({row['リスクコメント']})<br>備考: {row['備考']}"
        )
        raw_value = row[color_key]
        legend_value = str(raw_value) if pd.notna(raw_value) and raw_value != "" else "未設定"
        showlegend = False
        if legend_value not in legend_tracker:
            legend_tracker[legend_value] = True
            showlegend = True
        bar_color = color_map.get(raw_value, filters.bar_color)
        risk_level = row.get("リスクレベル", "低")
        border_color = {"高": "#E74C3C", "中": "#F39C12"}.get(risk_level)
        fig.add_trace(
            go.Bar(
                x=[(pd.to_datetime(row["竣工日"]) - pd.to_datetime(row["着工日"]) + pd.Timedelta(days=1)).days],
                y=[row["案件名"]],
                base=pd.to_datetime(row["着工日"]),
                orientation="h",
                marker=dict(
                    color=bar_color,
                    line=dict(color=border_color or bar_color, width=3 if border_color else 1),
                ),
                hovertemplate=hover_text,
                name=legend_value,
                legendgroup=legend_value,
                showlegend=showlegend,
                text=[f"{row['進捗率']:.0f}%"],
                texttemplate="%{text}",
                textposition="inside",
            )
        )
        annotation_symbol = {"高": "⚠️", "中": "△"}.get(risk_level)
        if annotation_symbol:
            fig.add_annotation(
                x=pd.to_datetime(row["竣工日"]) + pd.Timedelta(days=1),
                y=row["案件名"],
                text=annotation_symbol,
                showarrow=False,
                font=dict(size=16, color=border_color or "#2C3E50"),
            )

    start, end = fiscal_range
    month_starts = pd.date_range(start, end, freq="MS")
    label_font = {"高": 14, "中": 12, "低": 10}
    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        height=max(400, 40 * len(df) + 200),
        title=f"{filters.fiscal_year}年度 タイムライン",
        xaxis=dict(
            range=[pd.to_datetime(start), pd.to_datetime(end)],
            tickformat="%m/%d",
            showgrid=filters.show_grid,
            tickmode="array",
            tickvals=month_starts,
            ticktext=[d.strftime("%m") for d in month_starts],
        ),
        yaxis=dict(autorange="reversed", tickfont=dict(size=label_font.get(filters.label_density, 12))),
    )

    for month in range(12):
        line_date = pd.to_datetime(start) + relativedelta(months=month)
        fig.add_vline(
            x=line_date,
            line_width=1,
            line_dash="dash",
            line_color="#BDC3C7",
            opacity=0.6,
        )
    today = pd.Timestamp(date.today())
    if start <= today.date() <= end:
        fig.add_vline(
            x=today,
            line_width=2,
            line_color="#E74C3C",
        )
        fig.add_annotation(
            x=today,
            xref="x",
            y=1,
            yref="paper",
            text="今日",
            showarrow=False,
            xanchor="left",
            yanchor="bottom",
            font=dict(color="#E74C3C"),
            bgcolor="rgba(255, 255, 255, 0.8)",
            borderpad=4,
        )
    fig.update_yaxes(tickmode="linear")
    return fig


def validate_projects(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if df["id"].isna().any() or (df["id"].astype(str).str.strip() == "").any():
        errors.append("id は必須です。")
    if df["id"].duplicated().any():
        errors.append("id が重複しています。重複しないようにしてください。")
    for col in ["受注予定額", "受注金額", "予算原価", "予定原価", "実績原価"]:
        if col in df.columns and (df[col] < 0).any():
            errors.append(f"{col} は 0 以上にしてください。")
    if "進捗率" in df.columns and (~df["進捗率"].between(0, 100, inclusive="both")).any():
        errors.append("進捗率は 0〜100 の範囲にしてください。")
    for idx, row in df.iterrows():
        if pd.isna(row["着工日"]) or pd.isna(row["竣工日"]):
            errors.append(f"行 {idx + 1}: 着工日・竣工日は必須です。")
            continue
        if row["竣工日"] < row["着工日"]:
            errors.append(f"行 {idx + 1}: 竣工日は着工日以降にしてください。")
        actual_start = coerce_date(row.get("実際着工日"))
        actual_end = coerce_date(row.get("実際竣工日"))
        if actual_start and actual_end and actual_end < actual_start:
            errors.append(f"行 {idx + 1}: 実際竣工日は実際着工日以降にしてください。")
        cash_start = coerce_date(row.get("回収開始日"))
        cash_end = coerce_date(row.get("回収終了日"))
        if cash_start and cash_end and cash_end < cash_start:
            errors.append(f"行 {idx + 1}: 回収終了日は回収開始日以降にしてください。")
        pay_start = coerce_date(row.get("支払開始日"))
        pay_end = coerce_date(row.get("支払終了日"))
        if pay_start and pay_end and pay_end < pay_start:
            errors.append(f"行 {idx + 1}: 支払終了日は支払開始日以降にしてください。")
        if not (-100 <= row["粗利率"] <= 100):
            errors.append(f"行 {idx + 1}: 粗利率は -100〜100 の範囲にしてください。")
    return len(errors) == 0, errors


def compute_monthly_aggregation(df: pd.DataFrame, fiscal_range: Tuple[date, date]) -> pd.DataFrame:
    if df.empty:
        start, end = fiscal_range
        months = pd.date_range(start, end, freq="MS")
        return pd.DataFrame(
            {
                "年月": months,
                "受注金額": 0,
                "予定原価": 0,
                "粗利": 0,
                "粗利率": 0,
                "延べ人数": 0,
                "キャッシュイン": 0,
                "キャッシュアウト": 0,
                "キャッシュフロー": 0,
                "累計キャッシュフロー": 0,
            }
        )

    start, end = fiscal_range
    months = pd.date_range(start, end, freq="MS")
    records: List[Dict[str, float]] = []
    for month_start in months:
        month_end = month_start + relativedelta(months=1) - relativedelta(days=1)
        month_revenue = 0.0
        month_cost = 0.0
        month_manpower = 0.0
        month_cash_in = 0.0
        month_cash_out = 0.0
        for _, row in df.iterrows():
            month_revenue += allocate_value(
                row["受注金額"], row.get("着工日"), row.get("竣工日"), month_start, month_end
            )
            month_cost += allocate_value(
                row["予定原価"], row.get("着工日"), row.get("竣工日"), month_start, month_end
            )
            month_manpower += allocate_value(
                row["月平均必要人数"], row.get("着工日"), row.get("竣工日"), month_start, month_end
            )
            month_cash_in += allocate_value(
                row["受注金額"],
                row.get("回収開始日") or row.get("着工日"),
                row.get("回収終了日") or row.get("竣工日"),
                month_start,
                month_end,
            )
            month_cash_out += allocate_value(
                row["予定原価"],
                row.get("支払開始日") or row.get("着工日"),
                row.get("支払終了日") or row.get("竣工日"),
                month_start,
                month_end,
            )
        gross = month_revenue - month_cost
        gross_margin = gross / month_revenue * 100 if month_revenue else 0
        records.append(
            {
                "年月": month_start,
                "受注金額": month_revenue,
                "予定原価": month_cost,
                "粗利": gross,
                "粗利率": gross_margin,
                "延べ人数": month_manpower,
                "キャッシュイン": month_cash_in,
                "キャッシュアウト": month_cash_out,
            }
        )
    monthly_df = pd.DataFrame(records)
    monthly_df["キャッシュフロー"] = monthly_df["キャッシュイン"] - monthly_df["キャッシュアウト"]
    monthly_df["累計キャッシュフロー"] = monthly_df["キャッシュフロー"].cumsum()
    return monthly_df


def render_sidebar(df: pd.DataFrame, masters: Dict[str, List[str]]) -> FilterState:
    st.sidebar.header("条件設定")
    fiscal_year = st.sidebar.selectbox("事業年度", FISCAL_YEAR_OPTIONS, index=FISCAL_YEAR_OPTIONS.index(DEFAULT_FISCAL_YEAR))
    start, end = get_fiscal_year_range(fiscal_year)

    st.sidebar.subheader("フィルタ")
    period_from = st.sidebar.date_input("期間 From", value=start)
    period_to = st.sidebar.date_input("期間 To", value=end)
    if period_from > period_to:
        st.sidebar.warning("期間 From が To より後になっています。値を入れ替えました。")
        period_from, period_to = period_to, period_from

    status_options = sorted(df["ステータス"].dropna().unique())
    category_options = get_active_master_values(masters, "categories")
    contractor_options = sorted(df["元請区分"].dropna().unique())
    client_options = get_active_master_values(masters, "clients")
    manager_options = get_active_master_values(masters, "managers")
    prefecture_options = sorted(df["現場所在地"].dropna().unique())

    status = st.sidebar.multiselect("案件ステータス", status_options)
    category = st.sidebar.multiselect("工種", category_options)
    contractor = st.sidebar.multiselect("元請区分", contractor_options)
    clients = st.sidebar.multiselect("主要取引先", client_options)
    managers = st.sidebar.multiselect("担当者", manager_options)
    prefectures = st.sidebar.multiselect("現場所在地 (都道府県)", prefecture_options)
    margin_min, margin_max = st.sidebar.slider("粗利率レンジ (%)", -100, 100, (-100, 100))
    filter_mode = st.sidebar.radio("条件の組み合わせ", ["AND", "OR"], index=0)
    search_text = st.sidebar.text_input("フリーワード検索", placeholder="案件名・得意先など")
    search_targets = st.sidebar.multiselect(
        "検索対象",
        ["案件名", "得意先", "担当者", "協力会社", "工種"],
        default=["案件名", "得意先"],
    )

    st.sidebar.subheader("表示設定")
    color_key = st.sidebar.selectbox("色分けキー", ["ステータス", "工種", "元請区分"])
    bar_color = st.sidebar.color_picker("バー基調色", DEFAULT_BAR_COLOR)
    show_grid = st.sidebar.checkbox("月グリッド線を表示", True)
    label_density = st.sidebar.selectbox("ラベル密度", ["高", "中", "低"], index=1)

    st.sidebar.subheader("CSV 入出力")
    export_target = st.sidebar.radio("エクスポート対象", ["案件データ", "月次集計"], index=0, key="export_target_radio")
    export_format = st.sidebar.selectbox("出力形式", ["CSV", "Excel"], index=0)
    st.session_state["export_target"] = export_target
    st.session_state["export_format"] = export_format
    st.session_state["export_placeholder"] = st.sidebar.empty()

    uploaded = st.sidebar.file_uploader("データインポート", type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        mode = st.sidebar.radio("取り込み方法", ["マージ", "置換"], index=0, key="import_mode")
        if st.sidebar.button("インポート実行"):
            import_projects(uploaded, mode)
            st.sidebar.success("インポートが完了しました。ページを再読み込みしてください。")

    template_df = pd.DataFrame(columns=PROJECT_BASE_COLUMNS)
    st.sidebar.download_button(
        "テンプレートダウンロード",
        data=prepare_export(template_df, "CSV"),
        file_name="projects_template.csv",
        mime="text/csv",
    )

    return FilterState(
        fiscal_year=fiscal_year,
        period_from=period_from,
        period_to=period_to,
        status=status,
        category=category,
        contractor_level=contractor,
        client=clients,
        manager=managers,
        prefecture=prefectures,
        margin_range=(margin_min, margin_max),
        filter_mode=filter_mode,
        search_text=search_text,
        search_targets=search_targets,
        color_key=color_key,
        show_grid=show_grid,
        label_density=label_density,
        bar_color=bar_color,
    )


def prepare_export(df: Optional[pd.DataFrame], file_format: str = "CSV"):
    if df is None:
        return b"" if file_format == "Excel" else ""
    if file_format == "Excel":
        buffer = BytesIO()
        df.to_excel(buffer, index=False)
        buffer.seek(0)
        return buffer.getvalue()
    return df.to_csv(index=False)


def load_uploaded_dataframe(uploaded) -> pd.DataFrame:
    name = getattr(uploaded, "name", "").lower()
    try:
        uploaded.seek(0)
    except Exception:
        pass
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    return pd.read_csv(uploaded)


def import_projects(uploaded, mode: str) -> None:
    try:
        new_df = load_uploaded_dataframe(uploaded)
        for col in PROJECT_DATE_COLUMNS:
            if col in new_df.columns:
                new_df[col] = pd.to_datetime(new_df[col], errors="coerce").dt.date
        for col in PROJECT_NUMERIC_COLUMNS:
            if col in new_df.columns:
                new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
        current_df = load_projects()
        new_df = new_df.reindex(columns=current_df.columns, fill_value=None)
        if mode == "置換":
            save_projects(new_df)
        else:
            merged = current_df.set_index("id")
            new_df = new_df.set_index("id")
            merged.update(new_df)
            missing = new_df.loc[~new_df.index.isin(merged.index)]
            merged = pd.concat([merged, missing])
            merged.reset_index(inplace=True)
            save_projects(merged)
    except Exception as exc:
        st.sidebar.error(f"インポート中にエラーが発生しました: {exc}")


def render_projects_tab(full_df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    st.subheader("案件一覧")
    display_df = enrich_projects(filtered_df) if not filtered_df.empty else filtered_df.copy()
    if display_df.empty:
        st.info("条件に合致する案件がありません。フィルタを変更するか、新規行を追加してください。")
    display_df.reset_index(drop=True, inplace=True)

    alert_df = display_df[display_df.get("予算超過", False) == True]
    if not alert_df.empty:
        st.warning("予算超過となっている案件があります。詳細を確認してください。")
        st.dataframe(
            alert_df[["案件名", "予算乖離額", "担当者", "リスクコメント"]],
            use_container_width=True,
        )

    column_order = [
        "id",
        "案件名",
        "得意先",
        "元請区分",
        "工種",
        "ステータス",
        "着工日",
        "竣工日",
        "実際着工日",
        "実際竣工日",
        "受注予定額",
        "受注金額",
        "予算原価",
        "予定原価",
        "実績原価",
        "粗利率",
        "進捗率",
        "月平均必要人数",
        "回収開始日",
        "回収終了日",
        "支払開始日",
        "支払終了日",
        "現場所在地",
        "担当者",
        "協力会社",
        "備考",
        "リスクメモ",
        "粗利額",
        "原価率",
        "受注差異",
        "予算乖離額",
        "予算超過",
        "完成工事高",
        "実行粗利",
        "想定進捗率",
        "進捗差異",
        "遅延日数",
        "リスクレベル",
        "リスクコメント",
    ]
    column_order = [col for col in column_order if col in display_df.columns]

    column_config = {
        "着工日": st.column_config.DateColumn("着工日"),
        "竣工日": st.column_config.DateColumn("竣工日"),
        "実際着工日": st.column_config.DateColumn("実際着工日"),
        "実際竣工日": st.column_config.DateColumn("実際竣工日"),
        "回収開始日": st.column_config.DateColumn("回収開始日"),
        "回収終了日": st.column_config.DateColumn("回収終了日"),
        "支払開始日": st.column_config.DateColumn("支払開始日"),
        "支払終了日": st.column_config.DateColumn("支払終了日"),
        "受注予定額": st.column_config.NumberColumn("受注予定額", format="%d"),
        "受注金額": st.column_config.NumberColumn("受注金額", format="%d"),
        "予算原価": st.column_config.NumberColumn("予算原価", format="%d"),
        "予定原価": st.column_config.NumberColumn("予定原価", format="%d"),
        "実績原価": st.column_config.NumberColumn("実績原価", format="%d"),
        "粗利率": st.column_config.NumberColumn("粗利率", format="%0.1f"),
        "進捗率": st.column_config.NumberColumn("進捗率", format="%0.1f"),
        "月平均必要人数": st.column_config.NumberColumn("月平均必要人数", format="%0.1f"),
        "粗利額": st.column_config.NumberColumn("粗利額", format="%d", disabled=True),
        "原価率": st.column_config.NumberColumn("原価率", format="%0.1f", disabled=True),
        "受注差異": st.column_config.NumberColumn("受注差異", format="%d", disabled=True),
        "予算乖離額": st.column_config.NumberColumn("予算乖離額", format="%d", disabled=True),
        "完成工事高": st.column_config.NumberColumn("完成工事高", format="%d", disabled=True),
        "実行粗利": st.column_config.NumberColumn("実行粗利", format="%d", disabled=True),
        "想定進捗率": st.column_config.NumberColumn("想定進捗率", format="%0.1f", disabled=True),
        "進捗差異": st.column_config.NumberColumn("進捗差異", format="%0.1f", disabled=True),
        "遅延日数": st.column_config.NumberColumn("遅延日数", format="%d", disabled=True),
        "予算超過": st.column_config.CheckboxColumn("予算超過", disabled=True),
        "リスクレベル": st.column_config.TextColumn("リスクレベル", disabled=True),
        "リスクコメント": st.column_config.TextColumn("リスクコメント", disabled=True),
    }

    edited = st.data_editor(
        display_df,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_order=column_order,
        column_config=column_config,
        key="project_editor",
    )
    if st.button("保存", type="primary"):
        try:
            for col in PROJECT_DATE_COLUMNS:
                if col in edited.columns:
                    edited[col] = pd.to_datetime(edited[col], errors="coerce").dt.date
            for col in PROJECT_NUMERIC_COLUMNS:
                if col in edited.columns:
                    edited[col] = pd.to_numeric(edited[col], errors="coerce").fillna(0)
            persist_columns = [col for col in full_df.columns if col in edited.columns]
            persist_df = edited[persist_columns].copy()
            valid, errors = validate_projects(persist_df)
            if not valid:
                for msg in errors:
                    st.error(msg)
                return
            remaining = full_df[~full_df["id"].isin(persist_df["id"])]
            combined = pd.concat([persist_df, remaining], ignore_index=True)
            save_projects(combined)
            st.success("保存しました。ページを再読み込みしてください。")
        except Exception as exc:
            st.error(f"保存中にエラーが発生しました: {exc}")


def render_summary_tab(df: pd.DataFrame, monthly: pd.DataFrame) -> None:
    st.subheader("集計 / 分析")
    enriched = enrich_projects(df) if not df.empty else df

    total_revenue = enriched["受注金額"].sum()
    gross_profit = enriched["粗利額"].sum()
    gross_margin = gross_profit / total_revenue * 100 if total_revenue else 0
    order_diff = enriched["受注差異"].sum()
    completion_value = enriched["完成工事高"].sum()
    budget_over_count = int(enriched.get("予算超過", pd.Series(dtype=bool)).sum()) if not enriched.empty else 0
    cumulative_cash = monthly["累計キャッシュフロー"].iloc[-1] if not monthly.empty else 0

    st.markdown("### KPIサマリー")
    kpi_cols = st.columns(4)
    kpi_cols[0].metric("総粗利額", f"{gross_profit:,.0f} 円", f"粗利率 {gross_margin:,.1f}%")
    kpi_cols[1].metric("受注差異", f"{order_diff:,.0f} 円")
    kpi_cols[2].metric("完成工事高", f"{completion_value:,.0f} 円")
    kpi_cols[3].metric("予算超過案件", f"{budget_over_count} 件")

    st.markdown("### 月次推移")
    trend_fig = go.Figure()
    trend_fig.add_bar(x=monthly["年月"], y=monthly["受注金額"], name="受注金額")
    trend_fig.add_bar(x=monthly["年月"], y=monthly["予定原価"], name="予定原価")
    trend_fig.add_trace(go.Scatter(x=monthly["年月"], y=monthly["粗利"], mode="lines+markers", name="粗利"))
    trend_fig.add_trace(
        go.Scatter(
            x=monthly["年月"],
            y=monthly["粗利率"],
            mode="lines",
            name="粗利率",
            yaxis="y2",
        )
    )
    trend_fig.update_layout(
        template="plotly_white",
        barmode="group",
        yaxis=dict(title="金額"),
        yaxis2=dict(title="粗利率 (%)", overlaying="y", side="right"),
        height=480,
    )
    st.plotly_chart(trend_fig, use_container_width=True)

    st.markdown("### キャッシュフロー見通し")
    cash_fig = go.Figure()
    cash_fig.add_bar(x=monthly["年月"], y=monthly["キャッシュイン"], name="キャッシュイン")
    cash_fig.add_bar(x=monthly["年月"], y=-monthly["キャッシュアウト"], name="キャッシュアウト")
    cash_fig.add_trace(
        go.Scatter(
            x=monthly["年月"],
            y=monthly["累計キャッシュフロー"],
            mode="lines+markers",
            name="累計キャッシュフロー",
            yaxis="y2",
        )
    )
    cash_fig.update_layout(
        template="plotly_white",
        barmode="relative",
        yaxis=dict(title="キャッシュフロー"),
        yaxis2=dict(title="累計 (円)", overlaying="y", side="right"),
        height=420,
    )
    st.plotly_chart(cash_fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if enriched.empty:
            st.info("対象データがありません。")
        else:
            pie1 = go.Figure(data=[go.Pie(labels=enriched["工種"], values=enriched["受注金額"], hole=0.3)])
            pie1.update_layout(title="工種別構成比")
            st.plotly_chart(pie1, use_container_width=True)
    with col2:
        if enriched.empty:
            st.info("対象データがありません。")
        else:
            pie2 = go.Figure(data=[go.Pie(labels=enriched["得意先"], values=enriched["受注金額"], hole=0.3)])
            pie2.update_layout(title="得意先別構成比")
            st.plotly_chart(pie2, use_container_width=True)

    if enriched.empty:
        st.info("粗利率の分布を表示するデータがありません。")
    else:
        hist = go.Figure(data=[go.Histogram(x=enriched["粗利率"], nbinsx=10)])
        hist.update_layout(title="粗利率ヒストグラム", template="plotly_white")
        st.plotly_chart(hist, use_container_width=True)

    if not enriched.empty:
        st.markdown("### 原価率分析")
        project_ratio = enriched[["案件名", "受注金額", "予定原価", "原価率", "リスクレベル"]]
        st.dataframe(project_ratio.sort_values("原価率", ascending=False), use_container_width=True)

        category_summary = (
            enriched.groupby("工種")[["受注金額", "予定原価", "粗利額"]]
            .sum()
            .assign(原価率=lambda x: np.where(x["受注金額"] != 0, x["予定原価"] / x["受注金額"] * 100, 0))
            .reset_index()
        )
        st.dataframe(category_summary, use_container_width=True)

    st.markdown("### 月次サマリー")
    st.dataframe(
        monthly.assign(年月=monthly["年月"].dt.strftime("%Y-%m")),
        use_container_width=True,
    )


def render_settings_tab(masters: Dict[str, List[str]]) -> None:
    st.subheader("設定")
    st.markdown("### マスタ管理")

    def render_master_editor(label: str, key: str) -> pd.DataFrame:
        st.markdown(f"#### {label}")
        upload = st.file_uploader(f"{label}一括取込 (CSV/Excel)", type=["csv", "xlsx", "xls"], key=f"{key}_upload")
        if upload is not None and st.button(f"{label}を取り込む", key=f"{key}_import"):
            try:
                imported = load_uploaded_dataframe(upload)
                first_column = imported.columns[0]
                values = imported[first_column].dropna().astype(str).tolist()
                masters[key] = normalize_master_entries(values)
                st.success(f"{label}を読み込みました。保存ボタンで確定してください。")
            except Exception as exc:
                st.error(f"{label}の取り込みに失敗しました: {exc}")
        entries = normalize_master_entries(masters.get(key, []))
        masters[key] = entries
        base_df = pd.DataFrame(entries)
        if base_df.empty:
            base_df = pd.DataFrame({"name": [], "active": []})
        base_df["active"] = base_df.get("active", True)
        editor = st.data_editor(
            base_df,
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "name": st.column_config.TextColumn("名称"),
                "active": st.column_config.CheckboxColumn("有効")
            },
            use_container_width=True,
            key=f"{key}_editor",
        )
        return editor

    clients_df = render_master_editor("得意先", "clients")
    categories_df = render_master_editor("工種", "categories")
    managers_df = render_master_editor("担当者", "managers")

    st.markdown("### 休日カレンダー")
    holidays_df = pd.DataFrame({"休日": masters.get("holidays", [])})
    holidays_edit = st.data_editor(holidays_df, num_rows="dynamic", hide_index=True)

    st.markdown("### 表示設定")
    currency_format = st.text_input("通貨フォーマット", masters.get("currency_format", "#,###"))
    decimal_places = st.number_input("小数点以下桁数", min_value=0, max_value=4, value=int(masters.get("decimal_places", 0)))

    if st.button("設定を保存", type="primary"):
        masters["clients"] = normalize_master_entries(clients_df.to_dict("records"))
        masters["categories"] = normalize_master_entries(categories_df.to_dict("records"))
        masters["managers"] = normalize_master_entries(managers_df.to_dict("records"))
        masters["holidays"] = [
            d.strftime("%Y-%m-%d") if isinstance(d, (datetime, pd.Timestamp)) else str(d)
            for d in holidays_edit["休日"].dropna().tolist()
        ]
        masters["currency_format"] = currency_format or "#,###"
        masters["decimal_places"] = decimal_places
        history = masters.get("history", [])
        history_entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "clients": len(masters["clients"]),
            "categories": len(masters["categories"]),
            "managers": len(masters["managers"]),
        }
        history.append(history_entry)
        masters["history"] = history[-50:]
        save_masters(masters)
        st.success("設定を保存しました。")

    if masters.get("history"):
        with st.expander("更新履歴"):
            history_df = pd.DataFrame(masters["history"])
            st.dataframe(history_df.sort_values("timestamp", ascending=False), use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="工事受注案件 予定表", layout="wide")
    ensure_data_files()
    masters = load_masters()

    try:
        projects_df = load_projects()
    except Exception as exc:
        st.error(f"データの読み込みに失敗しました: {exc}")
        return

    filters = render_sidebar(projects_df, masters)
    fiscal_range = get_fiscal_year_range(filters.fiscal_year)
    filtered_df = apply_filters(projects_df, filters)
    enriched_filtered_df = enrich_projects(filtered_df) if not filtered_df.empty else filtered_df
    monthly_df = compute_monthly_aggregation(filtered_df, fiscal_range)
    st.session_state["monthly"] = monthly_df

    export_placeholder = st.session_state.get("export_placeholder")
    export_target = st.session_state.get("export_target", "案件データ")
    export_format = st.session_state.get("export_format", "CSV")
    export_source = enriched_filtered_df if export_target == "案件データ" else monthly_df
    if export_placeholder is not None:
        mime = (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if export_format == "Excel"
            else "text/csv"
        )
        file_name = (
            "projects_export.xlsx" if export_target == "案件データ" and export_format == "Excel" else
            "projects_export.csv" if export_target == "案件データ" else
            "monthly_summary.xlsx" if export_format == "Excel" else
            "monthly_summary.csv"
        )
        export_placeholder.download_button(
            "エクスポート",
            data=prepare_export(export_source, export_format),
            file_name=file_name,
            mime=mime,
        )

    tabs = st.tabs(["タイムライン", "案件一覧", "集計/分析", "設定"])

    with tabs[0]:
        st.subheader("タイムライン")
        timeline_fig = create_timeline(enriched_filtered_df, filters, fiscal_range)
        st.plotly_chart(timeline_fig, use_container_width=True)
        if not enriched_filtered_df.empty:
            st.markdown("### リスクサマリー")
            risk_table = enriched_filtered_df[[
                "案件名",
                "リスクレベル",
                "リスクコメント",
                "予算乖離額",
                "進捗差異",
                "遅延日数",
            ]]
            risk_order = {"高": 3, "中": 2, "低": 1}
            risk_table = risk_table.assign(優先度=risk_table["リスクレベル"].map(risk_order).fillna(0))
            st.dataframe(
                risk_table.sort_values(["優先度", "予算乖離額"], ascending=[False, False]).drop(columns="優先度"),
                use_container_width=True,
            )

            st.markdown("### リソース稼働状況")
            manager_summary, partner_summary = summarize_resources(enriched_filtered_df)
            res_col1, res_col2 = st.columns(2)
            res_col1.dataframe(manager_summary, use_container_width=True)
            res_col2.dataframe(partner_summary, use_container_width=True)

    with tabs[1]:
        render_projects_tab(projects_df, filtered_df)

    with tabs[2]:
        render_summary_tab(enriched_filtered_df, monthly_df)

    with tabs[3]:
        render_settings_tab(masters)


if __name__ == "__main__":
    main()
