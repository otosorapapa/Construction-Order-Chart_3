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

BRAND_COLORS = {
    "navy": "#0B1F3A",
    "slate": "#2F3C48",
    "mist": "#F4F6FA",
    "cloud": "#E8ECF3",
    "gold": "#C9A227",
    "sky": "#4D7EA8",
    "teal": "#6AA5A9",
    "crimson": "#B03038",
}

BRAND_COLORWAY = [
    BRAND_COLORS["navy"],
    BRAND_COLORS["sky"],
    "#8FAACF",
    BRAND_COLORS["teal"],
    BRAND_COLORS["gold"],
    "#7B8C9E",
]

BRAND_TEMPLATE = go.layout.Template(
    layout=dict(
        font=dict(
            family="'Noto Sans JP', 'Hiragino Sans', 'Segoe UI', sans-serif",
            color=BRAND_COLORS["slate"],
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hoverlabel=dict(font=dict(family="'Noto Sans JP', 'Hiragino Sans', 'Segoe UI', sans-serif")),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=BRAND_COLORS["cloud"],
            borderwidth=1,
        ),
        colorway=BRAND_COLORWAY,
    )
)

DEFAULT_BAR_COLOR = BRAND_COLORS["navy"]

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
        "ステータス": [
            BRAND_COLORS["navy"],
            BRAND_COLORS["sky"],
            "#8FAACF",
            BRAND_COLORS["teal"],
            BRAND_COLORS["gold"],
            "#7B8C9E",
        ],
        "工種": [
            BRAND_COLORS["navy"],
            BRAND_COLORS["gold"],
            BRAND_COLORS["sky"],
            BRAND_COLORS["teal"],
            "#9AA8BC",
        ],
        "元請区分": [
            BRAND_COLORS["navy"],
            BRAND_COLORS["sky"],
            BRAND_COLORS["gold"],
            BRAND_COLORS["teal"],
        ],
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


def style_risk_table(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    if df.empty:
        return df.style

    def highlight(row: pd.Series) -> List[str]:
        styles: List[str] = []
        for col in row.index:
            style = ""
            if col == "リスクレベル":
                if row[col] == "高":
                    style = "color: #B03038; font-weight: 600;"
                elif row[col] == "中":
                    style = "color: #C9A227; font-weight: 600;"
            if col == "遅延日数" and row[col] > 0:
                style = "color: #B03038; font-weight: 600;"
            if col == "進捗差異" and row[col] < -10:
                style = "color: #B03038; font-weight: 600;"
            if col == "予算乖離額" and row[col] > 0:
                style = "color: #B03038; font-weight: 600;"
            styles.append(style)
        return styles

    return (
        df.style.format(
            {
                "予算乖離額": "{:+,.0f} 円",
                "進捗差異": "{:+.1f} %",
                "遅延日数": lambda v: f"{int(v)}日",
            }
        )
        .apply(highlight, axis=1)
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#F4F6FA"),
                        ("color", "#2F3C48"),
                        ("font-weight", "600"),
                        ("border-bottom", "1px solid #E0E6F0"),
                    ],
                }
            ]
        )
    )




def create_timeline(df: pd.DataFrame, filters: FilterState, fiscal_range: Tuple[date, date]) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            xaxis_title="期間",
            yaxis_title="案件名",
            template=BRAND_TEMPLATE,
            height=500,
            plot_bgcolor="white",
            paper_bgcolor="white",
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
        border_color = {"高": BRAND_COLORS["crimson"], "中": BRAND_COLORS["gold"]}.get(risk_level)
        fig.add_trace(
            go.Bar(
                x=[(pd.to_datetime(row["竣工日"]) - pd.to_datetime(row["着工日"]) + pd.Timedelta(days=1)).days],
                y=[row["案件名"]],
                base=pd.to_datetime(row["着工日"]),
                orientation="h",
                marker=dict(
                    color=bar_color,
                    line=dict(color=border_color or "rgba(12,31,58,0.3)", width=3 if border_color else 1),
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
                font=dict(size=16, color=border_color or BRAND_COLORS["slate"]),
            )

    start, end = fiscal_range
    month_starts = pd.date_range(start, end, freq="MS")
    label_font = {"高": 14, "中": 12, "低": 10}
    fig.update_layout(
        barmode="stack",
        template=BRAND_TEMPLATE,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(400, 40 * len(df) + 200),
        title=f"{filters.fiscal_year}年度 タイムライン",
        xaxis=dict(
            range=[pd.to_datetime(start), pd.to_datetime(end)],
            tickformat="%m/%d",
            showgrid=filters.show_grid,
            tickmode="array",
            tickvals=month_starts,
            ticktext=[d.strftime("%m") for d in month_starts],
            gridcolor=BRAND_COLORS["cloud"],
            linecolor=BRAND_COLORS["cloud"],
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=label_font.get(filters.label_density, 12)),
            gridcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=80, b=40, l=10, r=10, pad=10),
    )

    for month in range(12):
        line_date = pd.to_datetime(start) + relativedelta(months=month)
        fig.add_vline(
            x=line_date,
            line_width=1,
            line_dash="dash",
            line_color=BRAND_COLORS["cloud"],
            opacity=0.6,
        )
    today = pd.Timestamp(date.today())
    if start <= today.date() <= end:
        fig.add_vline(
            x=today,
            line_width=2,
            line_color=BRAND_COLORS["crimson"],
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
            font=dict(color=BRAND_COLORS["crimson"]),
            bgcolor="rgba(255, 255, 255, 0.85)",
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
    fiscal_year = st.sidebar.selectbox(
        "事業年度",
        FISCAL_YEAR_OPTIONS,
        index=FISCAL_YEAR_OPTIONS.index(DEFAULT_FISCAL_YEAR),
        help="対象とする事業年度を選択すると全体の集計期間が更新されます。",
    )
    start, end = get_fiscal_year_range(fiscal_year)

    with st.sidebar.expander("フィルタ", expanded=False):
        st.caption("フィルターを変更するとグラフや表が即座に更新されます。")
        period_from = st.date_input("期間 From", value=start, help="着工・竣工日の範囲で絞り込みます。")
        period_to = st.date_input("期間 To", value=end, help="着工・竣工日の範囲で絞り込みます。")
        if period_from > period_to:
            st.warning("期間 From が To より後になっています。値を入れ替えました。")
            period_from, period_to = period_to, period_from

        status_options = sorted(df["ステータス"].dropna().unique())
        category_options = get_active_master_values(masters, "categories")
        contractor_options = sorted(df["元請区分"].dropna().unique())
        client_options = get_active_master_values(masters, "clients")
        manager_options = get_active_master_values(masters, "managers")
        prefecture_options = sorted(df["現場所在地"].dropna().unique())

        status = st.multiselect(
            "案件ステータス",
            status_options,
            placeholder="ステータス名を検索…",
            help="キーワード検索や Enter キーで素早く選択できます。",
        )
        category = st.multiselect(
            "工種",
            category_options,
            placeholder="工種名を検索…",
            help="複数選択や削除はタップ/クリックで直感的に操作できます。",
        )
        contractor = st.multiselect(
            "元請区分",
            contractor_options,
            placeholder="元請区分を検索…",
        )
        clients = st.multiselect(
            "主要取引先",
            client_options,
            placeholder="取引先を検索…",
        )
        managers = st.multiselect(
            "担当者",
            manager_options,
            placeholder="担当者を検索…",
        )
        prefectures = st.multiselect(
            "現場所在地 (都道府県)",
            prefecture_options,
            placeholder="所在地を検索…",
        )
        margin_min, margin_max = st.slider("粗利率レンジ (%)", -100, 100, (-100, 100))
        filter_mode = st.radio("条件の組み合わせ", ["AND", "OR"], index=0)
        search_text = st.text_input("フリーワード検索", placeholder="案件名・得意先など")
        search_targets = st.multiselect(
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


def toggle_sidebar_visibility() -> None:
    st.session_state["sidebar_visible"] = not st.session_state.get("sidebar_visible", True)


def apply_brand_theme() -> None:
    sidebar_visible = st.session_state.get("sidebar_visible", True)
    sidebar_transform = "translateX(0)" if sidebar_visible else "translateX(-108%)"
    sidebar_shadow = "0 24px 48px rgba(11, 31, 58, 0.15)" if sidebar_visible else "none"
    st.markdown(
        f"""
        <style>
        :root {{
            --brand-navy: {BRAND_COLORS['navy']};
            --brand-slate: {BRAND_COLORS['slate']};
            --brand-mist: {BRAND_COLORS['mist']};
            --brand-cloud: {BRAND_COLORS['cloud']};
            --brand-gold: {BRAND_COLORS['gold']};
            --brand-sky: {BRAND_COLORS['sky']};
            --brand-crimson: {BRAND_COLORS['crimson']};
            --sidebar-transform: {sidebar_transform};
            --sidebar-shadow: {sidebar_shadow};
        }}

        html, body, [data-testid="stAppViewContainer"], [data-testid="block-container"] {{
            background-color: var(--brand-mist) !important;
            color: var(--brand-slate);
            font-family: 'Noto Sans JP', 'Hiragino Sans', 'Segoe UI', sans-serif;
        }}

        [data-testid="block-container"] {{
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }}

        h1, h2, h3, h4 {{
            font-family: 'Noto Sans JP', 'Hiragino Sans', 'Segoe UI', sans-serif;
            color: var(--brand-navy);
            letter-spacing: 0.01em;
        }}

        .page-title {{
            font-size: 2.1rem;
            font-weight: 600;
            margin-bottom: 0.1rem;
        }}

        .page-subtitle {{
            font-size: 0.95rem;
            color: #60738a;
            margin-bottom: 1.5rem;
        }}

        .kpi-card {{
            background: white;
            border-radius: 18px;
            padding: 1.2rem 1.4rem;
            box-shadow: 0 12px 32px rgba(11, 31, 58, 0.08);
            border: 1px solid rgba(12, 31, 58, 0.06);
            display: flex;
            gap: 1rem;
            align-items: center;
            height: 100%;
        }}

        .kpi-card.alert {{
            border-color: rgba(176, 48, 56, 0.3);
            box-shadow: 0 16px 40px rgba(176, 48, 56, 0.12);
        }}

        .kpi-icon {{
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: grid;
            place-items: center;
            font-size: 1.6rem;
            background: rgba(77, 126, 168, 0.1);
            color: var(--brand-sky);
        }}

        .kpi-title {{
            font-size: 0.9rem;
            color: #5b6c82;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .kpi-value {{
            font-size: 1.6rem;
            font-weight: 600;
            color: var(--brand-navy);
            margin: 0.2rem 0;
        }}

        .kpi-subtitle {{
            font-size: 0.85rem;
            color: #7a889d;
        }}

        .fiscal-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            background: white;
            border-radius: 999px;
            padding: 0.35rem 0.9rem;
            font-size: 0.85rem;
            color: var(--brand-slate);
            box-shadow: inset 0 0 0 1px rgba(11, 31, 58, 0.08);
        }}

        .sidebar-toggle {{
            display: none;
            margin-bottom: 0.5rem;
        }}

        [data-testid="stSidebar"] {{
            background: white;
            border-right: 1px solid var(--brand-cloud);
            padding: 1.5rem 1.4rem 3rem;
            width: 320px;
            transition: transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out;
            box-shadow: var(--sidebar-shadow);
        }}

        [data-testid="stSidebar"] .stRadio > label {{
            font-weight: 500;
            color: var(--brand-slate);
        }}

        [data-testid="stSidebar"] .stMultiSelect, [data-testid="stSidebar"] .stSelectbox, [data-testid="stSidebar"] .stSlider {{
            margin-bottom: 1rem;
        }}

        [data-testid="stSidebar"]::-webkit-scrollbar {{ width: 6px; }}
        [data-testid="stSidebar"]::-webkit-scrollbar-thumb {{
            background: rgba(91, 108, 130, 0.25);
            border-radius: 3px;
        }}

        @media (max-width: 1200px) {{
            [data-testid="stSidebar"] {{
                position: fixed;
                inset: 0 auto 0 0;
                z-index: 1000;
                transform: var(--sidebar-transform);
                width: min(88vw, 320px);
            }}

            .sidebar-toggle {{
                display: inline-flex !important;
            }}
        }}

        .sidebar-toggle button {{
            width: 100%;
            border-radius: 999px !important;
            background: var(--brand-navy) !important;
            color: white !important;
            border: none;
            box-shadow: 0 10px 24px rgba(11, 31, 58, 0.18);
        }}

        .sidebar-toggle button:hover {{
            background: #10284f !important;
        }}

        div[data-testid="stMarkdownContainer"] .risk-high {{
            color: var(--brand-crimson);
            font-weight: 600;
        }}

        div[data-testid="stMarkdownContainer"] .risk-medium {{
            color: var(--brand-gold);
            font-weight: 600;
        }}

        .element-container:has(.stDataFrame) {{
            border-radius: 18px;
            background: white;
            padding: 0.6rem 0.6rem 0.2rem;
            box-shadow: 0 8px 24px rgba(11, 31, 58, 0.05);
            margin-bottom: 1.2rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(fiscal_year: int, fiscal_range: Tuple[date, date]) -> None:
    header_cols = st.columns([1.1, 4, 2])
    with header_cols[0]:
        st.markdown('<div class="sidebar-toggle">', unsafe_allow_html=True)
        st.button("☰ フィルタ", key="sidebar_toggle_button", on_click=toggle_sidebar_visibility)
        st.markdown('</div>', unsafe_allow_html=True)
    with header_cols[1]:
        st.markdown('<div class="page-title">工事受注ダッシュボード</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="page-subtitle">案件状況・粗利・キャッシュを俯瞰し迅速な意思決定を支援します</div>',
            unsafe_allow_html=True,
        )
    with header_cols[2]:
        fiscal_from, fiscal_to = fiscal_range
        st.markdown(
            f"<div style='display:flex;justify-content:flex-end'><span class='fiscal-pill'>FY {fiscal_year} : {fiscal_from:%Y.%m} - {fiscal_to:%Y.%m}</span></div>",
            unsafe_allow_html=True,
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
        "受注予定額": st.column_config.NumberColumn("受注予定額", format="%d", min_value=0),
        "受注金額": st.column_config.NumberColumn("受注金額", format="%d", min_value=0),
        "予算原価": st.column_config.NumberColumn("予算原価", format="%d", min_value=0),
        "予定原価": st.column_config.NumberColumn("予定原価", format="%d", min_value=0),
        "実績原価": st.column_config.NumberColumn("実績原価", format="%d", min_value=0),
        "粗利率": st.column_config.NumberColumn("粗利率", format="%0.1f", min_value=-100, max_value=100),
        "進捗率": st.column_config.NumberColumn("進捗率", format="%0.1f", min_value=0, max_value=100),
        "月平均必要人数": st.column_config.NumberColumn("月平均必要人数", format="%0.1f", min_value=0),
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

    column_config.update(
        {
            "id": st.column_config.TextColumn("案件ID", required=True, pinned="left"),
            "案件名": st.column_config.TextColumn("案件名", required=True, pinned="left", width="large"),
            "ステータス": st.column_config.TextColumn("ステータス", pinned="left"),
            "竣工日": st.column_config.DateColumn("竣工日", pinned="left"),
        }
    )

    edited = st.data_editor(
        display_df,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_order=column_order,
        column_config=column_config,
        key="project_editor",
    )

    # 入力値の即時バリデーション
    preview_df = edited.copy()
    try:
        for col in PROJECT_DATE_COLUMNS:
            if col in preview_df.columns:
                preview_df[col] = pd.to_datetime(preview_df[col], errors="coerce").dt.date
        for col in PROJECT_NUMERIC_COLUMNS:
            if col in preview_df.columns:
                preview_df[col] = pd.to_numeric(preview_df[col], errors="coerce")
        preview_valid, preview_errors = validate_projects(preview_df)
    except Exception as exc:
        preview_valid = False
        preview_errors = [f"入力値の検証中にエラーが発生しました: {exc}"]

    if not preview_valid and preview_errors:
        st.warning("入力内容に修正が必要です。保存前にエラーを解消してください。")
        for msg in preview_errors:
            st.error(msg)

    action_cols = st.columns([1, 1, 4])
    save_clicked = action_cols[0].button("変更を保存", type="primary")
    cancel_clicked = action_cols[1].button("キャンセル", help="最後に保存した状態に戻します。")
    if cancel_clicked:
        st.experimental_rerun()

    if save_clicked:
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
            st.success("保存が完了しました。必要に応じてページを再読み込みしてください。")
            st.toast("案件データを保存しました。", icon="✅")
        except Exception as exc:
            st.error(f"保存中にエラーが発生しました: {exc}")

    st.markdown("#### 案件詳細プレビュー")
    st.caption("一覧の行をクリックすると詳細が表示されます。")
    summary_view = display_df[[col for col in ["id", "案件名", "ステータス", "竣工日", "得意先"] if col in display_df.columns]]
    st.dataframe(
        summary_view,
        hide_index=True,
        use_container_width=True,
        column_config={
            "id": st.column_config.TextColumn("案件ID", pinned="left"),
            "案件名": st.column_config.TextColumn("案件名", width="large", pinned="left"),
            "竣工日": st.column_config.DateColumn("竣工日"),
        },
        key="project_selector",
    )

    selection_state = st.session_state.get("project_selector")
    selected_indices: List[int] = []
    if isinstance(selection_state, dict):
        selected_indices = selection_state.get("selection", {}).get("rows", [])  # type: ignore[arg-type]

    if selected_indices:
        selected_row = display_df.iloc[selected_indices[0]]
        with st.expander(f"{selected_row['案件名']} の詳細", expanded=True):
            detail_cols = st.columns(2)
            detail_cols[0].markdown(f"**案件ID**: {selected_row['id']}")
            detail_cols[0].markdown(f"**ステータス**: {selected_row['ステータス']}")
            detail_cols[0].markdown(f"**工種**: {selected_row['工種']}")
            detail_cols[0].markdown(f"**元請区分**: {selected_row['元請区分']}")
            detail_cols[1].markdown(f"**担当者**: {selected_row['担当者']}")
            detail_cols[1].markdown(f"**得意先**: {selected_row['得意先']}")
            detail_cols[1].markdown(f"**現場所在地**: {selected_row['現場所在地']}")
            st.markdown("**期間**")
            st.markdown(
                f"着工日: {format_date(selected_row['着工日'])} / 竣工日: {format_date(selected_row['竣工日'])}"
            )
            st.markdown("**備考**")
            st.write(selected_row.get("備考", "-"))
            st.markdown("**リスクメモ**")
            st.write(selected_row.get("リスクメモ", "-"))
            st.caption("添付ファイルは案件詳細ページから確認・追加できます。")
    else:
        st.info("詳細を確認したい案件を一覧から選択してください。")


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
    kpi_data = [
        {
            "icon": "💰",
            "title": "Gross Profit",
            "value": f"{gross_profit:,.0f} 円",
            "subtitle": f"粗利率 {gross_margin:,.1f}%",
        },
        {
            "icon": "📦",
            "title": "Order Delta",
            "value": f"{order_diff:,.0f} 円",
            "subtitle": "受注金額 - 受注予定額",
        },
        {
            "icon": "🏗️",
            "title": "Completion Value",
            "value": f"{completion_value:,.0f} 円",
            "subtitle": f"完成工事高 / 累計CF {cumulative_cash:,.0f} 円",
        },
        {
            "icon": "⚠️" if budget_over_count else "✅",
            "title": "Budget Alerts",
            "value": f"{budget_over_count} 件",
            "subtitle": "予算超過案件数",
            "alert": budget_over_count > 0,
        },
    ]
    kpi_cols = st.columns(len(kpi_data))
    for col, card in zip(kpi_cols, kpi_data):
        alert_class = " alert" if card.get("alert") else ""
        col.markdown(
            f"""
            <div class="kpi-card{alert_class}">
                <div class="kpi-icon">{card['icon']}</div>
                <div>
                    <div class="kpi-title">{card['title']}</div>
                    <div class="kpi-value">{card['value']}</div>
                    <div class="kpi-subtitle">{card['subtitle']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### 月次推移")
    trend_fig = go.Figure()
    trend_fig.add_bar(
        x=monthly["年月"],
        y=monthly["受注金額"],
        name="受注金額",
        marker=dict(color=BRAND_COLORS["navy"], line=dict(width=0)),
    )
    trend_fig.add_bar(
        x=monthly["年月"],
        y=monthly["予定原価"],
        name="予定原価",
        marker=dict(color=BRAND_COLORS["sky"], line=dict(width=0)),
    )
    trend_fig.add_trace(
        go.Scatter(
            x=monthly["年月"],
            y=monthly["粗利"],
            mode="lines+markers",
            name="粗利",
            marker=dict(color=BRAND_COLORS["gold"], size=8),
            line=dict(color=BRAND_COLORS["gold"], width=3),
        )
    )
    trend_fig.add_trace(
        go.Scatter(
            x=monthly["年月"],
            y=monthly["粗利率"],
            mode="lines",
            name="粗利率",
            yaxis="y2",
            line=dict(color=BRAND_COLORS["teal"], width=2, dash="dot"),
        )
    )
    trend_fig.update_layout(
        template=BRAND_TEMPLATE,
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(title="金額", gridcolor=BRAND_COLORS["cloud"], zerolinecolor=BRAND_COLORS["cloud"]),
        yaxis2=dict(title="粗利率 (%)", overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"),
        height=480,
        margin=dict(t=60, b=40, l=10, r=10, pad=10),
    )
    st.plotly_chart(trend_fig, use_container_width=True)

    st.markdown("### キャッシュフロー見通し")
    cash_fig = go.Figure()
    cash_fig.add_bar(
        x=monthly["年月"],
        y=monthly["キャッシュイン"],
        name="キャッシュイン",
        marker=dict(color=BRAND_COLORS["teal"], line=dict(width=0)),
    )
    cash_fig.add_bar(
        x=monthly["年月"],
        y=-monthly["キャッシュアウト"],
        name="キャッシュアウト",
        marker=dict(color="#8FAACF", line=dict(width=0)),
    )
    cash_fig.add_trace(
        go.Scatter(
            x=monthly["年月"],
            y=monthly["累計キャッシュフロー"],
            mode="lines+markers",
            name="累計キャッシュフロー",
            yaxis="y2",
            marker=dict(color=BRAND_COLORS["navy"], size=7),
            line=dict(color=BRAND_COLORS["navy"], width=3),
        )
    )
    cash_fig.update_layout(
        template=BRAND_TEMPLATE,
        barmode="relative",
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(title="キャッシュフロー", gridcolor=BRAND_COLORS["cloud"], zerolinecolor=BRAND_COLORS["cloud"]),
        yaxis2=dict(title="累計 (円)", overlaying="y", side="right", gridcolor="rgba(0,0,0,0)"),
        height=420,
        margin=dict(t=60, b=40, l=10, r=10, pad=10),
    )
    st.plotly_chart(cash_fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if enriched.empty:
            st.info("対象データがありません。")
        else:
            pie1 = go.Figure(
                data=[
                    go.Pie(
                        labels=enriched["工種"],
                        values=enriched["受注金額"],
                        hole=0.45,
                        marker=dict(colors=BRAND_COLORWAY, line=dict(color="white", width=2)),
                        textinfo="label+percent",
                    )
                ]
            )
            pie1.update_layout(
                title="工種別構成比",
                template=BRAND_TEMPLATE,
                showlegend=False,
            )
            st.plotly_chart(pie1, use_container_width=True)
    with col2:
        if enriched.empty:
            st.info("対象データがありません。")
        else:
            pie2 = go.Figure(
                data=[
                    go.Pie(
                        labels=enriched["得意先"],
                        values=enriched["受注金額"],
                        hole=0.45,
                        marker=dict(colors=BRAND_COLORWAY, line=dict(color="white", width=2)),
                        textinfo="label+percent",
                    )
                ]
            )
            pie2.update_layout(
                title="得意先別構成比",
                template=BRAND_TEMPLATE,
                showlegend=False,
            )
            st.plotly_chart(pie2, use_container_width=True)

    if enriched.empty:
        st.info("粗利率の分布を表示するデータがありません。")
    else:
        hist = go.Figure(
            data=[
                go.Histogram(
                    x=enriched["粗利率"],
                    nbinsx=10,
                    marker=dict(color=BRAND_COLORS["navy"], opacity=0.75),
                )
            ]
        )
        hist.update_layout(
            title="粗利率ヒストグラム",
            template=BRAND_TEMPLATE,
            bargap=0.1,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(title="粗利率", gridcolor=BRAND_COLORS["cloud"]),
            yaxis=dict(title="件数", gridcolor=BRAND_COLORS["cloud"]),
        )
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
    monthly_view = monthly.assign(年月=monthly["年月"].dt.strftime("%Y-%m")).style.format(
        {
            "受注金額": "{:,.0f}",
            "予定原価": "{:,.0f}",
            "粗利": "{:,.0f}",
            "粗利率": "{:.1f}",
            "延べ人数": "{:.1f}",
            "キャッシュイン": "{:,.0f}",
            "キャッシュアウト": "{:,.0f}",
            "キャッシュフロー": "{:,.0f}",
            "累計キャッシュフロー": "{:,.0f}",
        }
    )
    st.dataframe(monthly_view, use_container_width=True)


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
    if "sidebar_visible" not in st.session_state:
        st.session_state["sidebar_visible"] = True
    apply_brand_theme()
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

    render_page_header(filters.fiscal_year, fiscal_range)

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
            sorted_risk = risk_table.sort_values(["優先度", "予算乖離額"], ascending=[False, False]).drop(columns="優先度")
            st.dataframe(style_risk_table(sorted_risk), use_container_width=True, height=360)

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
