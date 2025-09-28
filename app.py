import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from io import BytesIO
from typing import Dict, List, Optional, Set, Tuple

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
            font=dict(color=BRAND_COLORS["slate"], size=12),
        ),
        colorway=BRAND_COLORWAY,
    )
)

DEFAULT_BAR_COLOR = BRAND_COLORS["navy"]


MODAL_SUPPORTED = hasattr(st, "modal")


if MODAL_SUPPORTED:

    def modal_container(title: str, key: Optional[str] = None):
        """Return the native Streamlit modal context manager when available."""

        return st.modal(title, key=key)


else:

    @contextmanager
    def modal_container(title: str, key: Optional[str] = None):
        """Fallback context manager that emulates a modal in older Streamlit versions."""

        with st.container():
            st.markdown(f"### {title}")
            st.caption("この環境ではモーダル表示に対応していないため、フォームをページ内に表示しています。")
            yield

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


STATUS_BADGE_MAP = {
    "施工中": ("🔧", "info"),
    "受注": ("✅", "success"),
    "見積": ("📝", "info"),
    "完了": ("🏁", "success"),
    "失注": ("⚠️", "alert"),
}

RISK_BADGE_MAP = {
    "高": ("⚠️", "alert"),
    "中": ("⚡", "warn"),
    "低": ("🛡️", "success"),
}


def build_badge(label: str, icon: str, tone: str) -> str:
    return f"<span class='status-badge {tone}'>{icon} {label}</span>"


def format_status_badge(status: str) -> str:
    if not status:
        return "-"
    icon, tone = STATUS_BADGE_MAP.get(status, ("📁", "info"))
    return build_badge(status, icon, tone)


def format_risk_badge(level: str) -> str:
    if not level:
        return "-"
    icon, tone = RISK_BADGE_MAP.get(level, ("ℹ️", "info"))
    return build_badge(level, icon, tone)


def switch_main_tab(tab_label: str) -> None:
    """Programmatically switch the main content tab."""
    st.session_state["main_tabs"] = tab_label
    st.experimental_rerun()


def trigger_new_project_modal() -> None:
    """Open the project creation modal and jump to the project list tab."""
    st.session_state["show_project_modal"] = True
    st.session_state["main_tabs"] = "案件一覧"
    st.experimental_rerun()


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


def hex_to_rgb(color: str) -> Optional[Tuple[int, int, int]]:
    """Convert a hex color string (e.g. #0B1F3A) to an RGB tuple."""

    if not isinstance(color, str):
        return None
    cleaned = color.strip().lstrip("#")
    if len(cleaned) == 3:
        cleaned = "".join(ch * 2 for ch in cleaned)
    if len(cleaned) != 6:
        return None
    try:
        return tuple(int(cleaned[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return None


def get_contrasting_text_color(color: str) -> str:
    """Return a text color (white or navy) that contrasts with the given fill color."""

    rgb = hex_to_rgb(color)
    if rgb is None:
        return BRAND_COLORS["navy"]

    def to_linear(channel: float) -> float:
        channel = channel / 255
        return channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4

    r, g, b = (to_linear(c) for c in rgb)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#FFFFFF" if luminance < 0.55 else BRAND_COLORS["navy"]


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
    numeric_defaults = {
        "受注金額": 0.0,
        "予定原価": 0.0,
        "受注予定額": 0.0,
        "予算原価": 0.0,
        "進捗率": 0.0,
        "実績原価": 0.0,
    }
    for column, default in numeric_defaults.items():
        if column not in enriched.columns:
            enriched[column] = default
    if "実際竣工日" not in enriched.columns:
        enriched["実際竣工日"] = pd.NaT
    if "竣工日" not in enriched.columns:
        enriched["竣工日"] = pd.NaT
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
                        ("background-color", "#1e3a6f"),
                        ("color", "#ffffff"),
                        ("font-weight", "600"),
                        ("border-bottom", "1px solid #d5deeb"),
                    ],
                },
                {
                    "selector": "tbody tr:nth-child(odd)",
                    "props": [
                        ("background-color", "#eef3fb"),
                    ],
                },
                {
                    "selector": "tbody tr:nth-child(even)",
                    "props": [
                        ("background-color", "#ffffff"),
                    ],
                }
            ]
        )
    )




def create_timeline(df: pd.DataFrame, filters: FilterState, fiscal_range: Tuple[date, date]) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(
                title="期間",
                tickfont=dict(color=BRAND_COLORS["slate"]),
                titlefont=dict(color=BRAND_COLORS["slate"]),
                gridcolor=BRAND_COLORS["cloud"],
                linecolor=BRAND_COLORS["cloud"],
            ),
            yaxis=dict(
                title="案件名",
                tickfont=dict(color=BRAND_COLORS["slate"]),
                titlefont=dict(color=BRAND_COLORS["slate"]),
                gridcolor="rgba(0,0,0,0)",
            ),
            template=BRAND_TEMPLATE,
            height=500,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig

    color_key = filters.color_key
    if color_key in df.columns:
        color_source = df[color_key]
    else:
        color_source = pd.Series(["未設定"] * len(df), index=df.index)
    color_map = generate_color_map(color_source, color_key, filters.bar_color)
    legend_tracker: Dict[str, bool] = {}

    def safe_float(value, default: float = 0.0) -> float:
        if pd.isna(value):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def safe_int(value, default: int = 0) -> int:
        if pd.isna(value):
            return default
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default

    def safe_str(value, default: str = "-") -> str:
        if value is None or pd.isna(value):
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return str(value)

    fig = go.Figure()
    for _, row in df.iterrows():
        planned_start_dt = pd.to_datetime(row.get("着工日"), errors="coerce")
        planned_end_dt = pd.to_datetime(row.get("竣工日"), errors="coerce")

        if pd.isna(planned_start_dt) or pd.isna(planned_end_dt):
            continue

        duration_days = (planned_end_dt - planned_start_dt).days + 1
        if duration_days <= 0:
            continue

        planned_start = format_date(planned_start_dt)
        planned_end = format_date(planned_end_dt)
        actual_end = format_date(row.get("実際竣工日"))

        progress = safe_float(row.get("進捗率"))
        expected_progress = safe_float(row.get("想定進捗率"))
        delay_days = safe_int(row.get("遅延日数"))
        gross_profit = safe_int(row.get("粗利額"))
        cost_ratio = safe_float(row.get("原価率"))
        order_diff = safe_int(row.get("受注差異"))
        budget_diff = safe_int(row.get("予算乖離額"))
        completion_value = safe_int(row.get("完成工事高"))
        actual_profit = safe_int(row.get("実行粗利"))
        avg_people = safe_float(row.get("月平均必要人数"))

        client = safe_str(row.get("得意先"))
        category = safe_str(row.get("工種"))
        status = safe_str(row.get("ステータス"))
        manager = safe_str(row.get("担当者"))
        partner = safe_str(row.get("協力会社"))
        risk_level = safe_str(row.get("リスクレベル"), "低")
        risk_comment = safe_str(row.get("リスクコメント"))
        notes = safe_str(row.get("備考"))
        hover_text = (
            f"案件名: {row['案件名']}<br>期間: {planned_start}〜{planned_end}<br>"
            f"得意先: {client}<br>工種: {category}<br>ステータス: {status}<br>"
            f"進捗率: {progress:.1f}% (想定 {expected_progress:.1f}%)<br>"
            f"遅延日数: {delay_days}日 / 実竣工: {actual_end}<br>"
            f"粗利額: {gross_profit:,}円 / 原価率: {cost_ratio:.1f}%<br>"
            f"受注差異: {order_diff:,}円 / 予算乖離: {budget_diff:,}円<br>"
            f"完成工事高: {completion_value:,}円 / 実行粗利: {actual_profit:,}円<br>"
            f"担当者: {manager} / 協力会社: {partner}<br>"
            f"月平均必要人数: {avg_people:.1f}人<br>"
            f"回収: {format_date(row.get('回収開始日'))}〜{format_date(row.get('回収終了日'))}<br>"
            f"支払: {format_date(row.get('支払開始日'))}〜{format_date(row.get('支払終了日'))}<br>"
            f"リスク: {risk_level} ({risk_comment})<br>備考: {notes}"
        )
        raw_value = row.get(color_key, None)
        has_raw_value = pd.notna(raw_value) and str(raw_value).strip() != ""
        legend_value = str(raw_value) if has_raw_value else "未設定"
        showlegend = False
        if legend_value not in legend_tracker:
            legend_tracker[legend_value] = True
            showlegend = True
        color_lookup_key = raw_value if has_raw_value else "未設定"
        bar_color = color_map.get(color_lookup_key, filters.bar_color)
        border_color = {"高": BRAND_COLORS["crimson"], "中": BRAND_COLORS["gold"]}.get(risk_level)
        fig.add_trace(
            go.Bar(
                x=[duration_days],
                y=[row["案件名"]],
                base=planned_start_dt,
                orientation="h",
                marker=dict(
                    color=bar_color,
                    line=dict(color=border_color or "rgba(12,31,58,0.3)", width=3 if border_color else 1),
                ),
                hovertemplate=hover_text,
                name=legend_value,
                legendgroup=legend_value,
                showlegend=showlegend,
                text=[f"{progress:.0f}%"],
                texttemplate="%{text}",
                textposition="inside",
                textfont=dict(
                    color=[get_contrasting_text_color(bar_color)],
                    family="'Noto Sans JP', 'Hiragino Sans', 'Segoe UI', sans-serif",
                    size=12,
                ),
            )
        )
        annotation_symbol = {"高": "⚠️", "中": "△"}.get(risk_level)
        if annotation_symbol:
            fig.add_annotation(
                x=planned_end_dt + pd.Timedelta(days=1),
                y=row["案件名"],
                text=annotation_symbol,
                showarrow=False,
                font=dict(size=16, color=border_color or BRAND_COLORS["slate"]),
            )

    start, end = fiscal_range
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    month_starts = pd.date_range(start_ts, end_ts, freq="MS")

    tick_candidates: List[pd.Timestamp] = []
    for month_start in month_starts:
        month_end = min(
            end_ts,
            month_start + relativedelta(months=1) - pd.Timedelta(days=1),
        )

        for offset in [0, 6, 12, 18, 24]:
            candidate = month_start + pd.Timedelta(days=offset)
            if candidate < start_ts or candidate > month_end or candidate > end_ts:
                continue
            tick_candidates.append(candidate)

        if month_end >= start_ts:
            tick_candidates.append(month_end)

    tick_candidates.sort()
    tick_values: List[pd.Timestamp] = []
    seen: Set[pd.Timestamp] = set()
    for candidate in tick_candidates:
        if candidate not in seen:
            seen.add(candidate)
            tick_values.append(candidate)
    label_font = {"高": 14, "中": 12, "低": 10}
    project_count = max(1, len(fig.data))
    fig.update_layout(
        barmode="stack",
        template=BRAND_TEMPLATE,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=max(400, 40 * project_count + 200),
        title=dict(
            text=f"{filters.fiscal_year}年度 タイムライン",
            font=dict(color=BRAND_COLORS["slate"]),
        ),
        xaxis=dict(
            range=[start_ts, end_ts],
            tickformat="%m/%d",
            showgrid=filters.show_grid,
            tickmode="array",
            tickvals=tick_values,
            ticktext=[f"{val.month}/{val.day}" for val in tick_values],
            gridcolor=BRAND_COLORS["cloud"],
            linecolor=BRAND_COLORS["cloud"],
            tickfont=dict(color=BRAND_COLORS["slate"]),
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(
                size=label_font.get(filters.label_density, 12),
                color=BRAND_COLORS["slate"],
            ),
            gridcolor="rgba(0,0,0,0)",
        ),
        margin=dict(t=80, b=40, l=10, r=10, pad=10),
    )

    if filters.show_grid:
        for month_start in month_starts:
            month_end = min(
                end_ts,
                month_start + relativedelta(months=1) - pd.Timedelta(days=1),
            )

            fig.add_vline(
                x=month_start,
                line_width=1,
                line_dash="dash",
                line_color=BRAND_COLORS["cloud"],
                opacity=0.6,
            )

            for offset in [6, 12, 18, 24]:
                guide_date = month_start + pd.Timedelta(days=offset)
                if guide_date < start_ts or guide_date > month_end or guide_date > end_ts:
                    continue
                fig.add_vline(
                    x=guide_date,
                    line_width=0.6,
                    line_dash="dot",
                    line_color=BRAND_COLORS["cloud"],
                    opacity=0.35,
                )

            if month_end >= start_ts:
                fig.add_vline(
                    x=month_end,
                    line_width=1,
                    line_dash="dash",
                    line_color=BRAND_COLORS["cloud"],
                    opacity=0.5,
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
    fig.update_yaxes(tickmode="linear", tickfont=dict(color=BRAND_COLORS["slate"]))
    fig.update_xaxes(tickfont=dict(color=BRAND_COLORS["slate"]))
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


def render_control_panel(df: pd.DataFrame, masters: Dict[str, List[str]]) -> FilterState:
    st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
    with st.container():
        col_period, col_display, col_export = st.columns([1.35, 1.5, 1.15])

        with col_period:
            st.markdown("#### 集計期間")
            fiscal_year = st.selectbox(
                "事業年度",
                FISCAL_YEAR_OPTIONS,
                index=FISCAL_YEAR_OPTIONS.index(DEFAULT_FISCAL_YEAR),
                help="対象年度を変更すると、各種グラフ・表の期間が自動調整されます。",
                key="fiscal_year_select",
            )
            start, end = get_fiscal_year_range(fiscal_year)

        with col_display:
            st.markdown("#### 表示設定")
            color_key = st.selectbox(
                "色分けキー",
                ["ステータス", "工種", "元請区分"],
                help="ガントチャートや円グラフの色分け基準を切り替えます。",
                key="color_key_select",
            )
            bar_color = st.color_picker(
                "バー基調色",
                DEFAULT_BAR_COLOR,
                help="タイムラインのバー色をチームカラーに合わせて変更できます。",
                key="bar_color_picker",
            )
            show_grid = st.checkbox(
                "月グリッド線を表示",
                True,
                help="タイムラインに月単位のガイドを表示します。",
                key="show_grid_checkbox",
            )
            label_density = st.selectbox(
                "ラベル密度",
                ["高", "中", "低"],
                index=1,
                help="チャート上のラベル表示量を調整します。",
                key="label_density_select",
            )

        with col_export:
            st.markdown("#### データ入出力")
            export_target = st.radio(
                "エクスポート対象",
                ["案件データ", "月次集計"],
                index=0,
                horizontal=True,
                key="export_target_radio",
                help="ダウンロードするデータセットを選択します。",
            )
            export_format = st.selectbox(
                "出力形式",
                ["CSV", "Excel"],
                index=0,
                key="export_format_select",
                help="必要な形式でファイルを出力します。",
            )
            st.session_state["export_target"] = export_target
            st.session_state["export_format"] = export_format
            st.session_state["export_placeholder"] = st.empty()

        st.markdown("#### クイックアクション")
        render_quick_actions()

        with st.expander("詳細フィルタを表示", expanded=False):
            st.caption("条件を絞り込むと一覧・グラフが即座に更新されます。")

            period_state_key = "period_range_state"
            if period_state_key not in st.session_state:
                st.session_state[period_state_key] = (start, end)
                st.session_state["period_range_year"] = fiscal_year
            elif st.session_state.get("period_range_year") != fiscal_year:
                st.session_state[period_state_key] = (start, end)
                st.session_state["period_range_year"] = fiscal_year

            current_range = st.date_input(
                "対象期間",
                value=st.session_state.get(period_state_key, (start, end)),
                min_value=start - relativedelta(years=1),
                max_value=end + relativedelta(years=1),
                format="YYYY-MM-DD",
                help="期間をドラッグして選択できます。開始日と終了日は自動的に並び替えられます。",
                key="period_range_picker",
            )

            if isinstance(current_range, tuple):
                period_from, period_to = current_range
            else:
                period_from = current_range
                period_to = current_range

            if period_from and period_to and period_from > period_to:
                period_from, period_to = period_to, period_from

            st.session_state[period_state_key] = (period_from, period_to)

            status_options = sorted(df["ステータス"].dropna().unique())
            category_options = get_active_master_values(masters, "categories")
            contractor_options = sorted(df["元請区分"].dropna().unique())
            client_options = get_active_master_values(masters, "clients")
            manager_options = get_active_master_values(masters, "managers")
            prefecture_options = sorted(df["現場所在地"].dropna().unique())

            filter_cols = st.columns(3)
            with filter_cols[0]:
                status = st.multiselect(
                    "案件ステータス",
                    status_options,
                    placeholder="ステータス名を検索…",
                    help="進捗に応じた案件のみ抽出します。",
                )
                contractor = st.multiselect(
                    "元請区分",
                    contractor_options,
                    placeholder="元請区分を検索…",
                    help="自社/一次/二次などの区分を指定します。",
                )
                margin_min, margin_max = st.slider(
                    "粗利率レンジ (%)",
                    -100,
                    100,
                    (-100, 100),
                    help="粗利率の下限・上限を同時に指定できます。",
                )

            with filter_cols[1]:
                category = st.multiselect(
                    "工種",
                    category_options,
                    placeholder="工種名を検索…",
                    help="複数選択や削除はタップ/クリックで直感的に操作できます。",
                )
                clients = st.multiselect(
                    "主要取引先",
                    client_options,
                    placeholder="取引先を検索…",
                    help="取引先名を入力すると候補が絞り込まれます。",
                )
                filter_mode = st.radio(
                    "条件の組み合わせ",
                    ["AND", "OR"],
                    index=0,
                    horizontal=True,
                    help="AND: 全条件を満たす案件 / OR: いずれかの条件に合致する案件を表示します。",
                )

            with filter_cols[2]:
                managers = st.multiselect(
                    "担当者",
                    manager_options,
                    placeholder="担当者を検索…",
                    help="担当者名で案件を絞り込めます。",
                )
                prefectures = st.multiselect(
                    "現場所在地 (都道府県)",
                    prefecture_options,
                    placeholder="所在地を検索…",
                    help="地域別の案件を確認するときに活用できます。",
                )
                search_text = st.text_input(
                    "フリーワード検索",
                    placeholder="案件名・得意先・協力会社など",
                    help="部分一致で検索します。スペース区切りで複数キーワードも可能です。",
                )
                search_targets = st.multiselect(
                    "検索対象",
                    ["案件名", "得意先", "担当者", "協力会社", "工種"],
                    default=["案件名", "得意先"],
                    help="フリーワード検索の対象カラムを指定します。",
                )

            st.markdown("##### データ取り込み")
            upload_cols = st.columns([2, 1])
            with upload_cols[0]:
                uploaded = st.file_uploader(
                    "案件データを取り込む",
                    type=["csv", "xlsx", "xls"],
                    help="CSV/Excel 形式で案件一覧を一括更新できます。",
                )
            with upload_cols[1]:
                mode = st.radio(
                    "取り込み方法",
                    ["マージ", "置換"],
                    index=0,
                    help="マージ: 既存案件を維持し差分を追加 / 置換: ファイル内容で上書き",
                )
                if uploaded is not None and st.button(
                    "インポート実行",
                    use_container_width=True,
                    help="読み込んだデータを適用します。",
                ):
                    import_projects(uploaded, mode)
                    st.success("インポートが完了しました。ページを再読み込みしてください。")

            template_df = pd.DataFrame(columns=PROJECT_BASE_COLUMNS)
            st.download_button(
                "テンプレートダウンロード",
                data=prepare_export(template_df, "CSV"),
                file_name="projects_template.csv",
                mime="text/csv",
                help="案件登録用のフォーマットを取得します。",
            )

    st.markdown("</div>", unsafe_allow_html=True)

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


def apply_brand_theme() -> None:
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
            --accent-green: #2F9E5B;
            --surface-bg: #f7f9fc;
            --surface-panel: #e7eef8;
            --surface-card: #ffffff;
            --surface-outline: #d5deeb;
            --text-strong: #1c2734;
            --text-muted: #5b6c82;
            --text-invert: #ffffff;
        }}

        html, body, [data-testid="stAppViewContainer"], [data-testid="block-container"] {{
            background-color: var(--surface-bg) !important;
            color: var(--brand-slate);
            font-family: 'Noto Sans JP', 'Hiragino Sans', 'Segoe UI', sans-serif;
        }}

        [data-testid="block-container"] {{
            padding-top: 1.2rem;
            padding-bottom: 3rem;
            max-width: 1240px;
        }}

        h1, h2, h3, h4 {{
            font-family: 'Noto Sans JP', 'Hiragino Sans', 'Segoe UI', sans-serif;
            color: var(--brand-navy);
            letter-spacing: 0.01em;
        }}

        label, .stMarkdown p {{
            color: var(--text-strong);
        }}

        [data-testid="stTextInput"] label,
        [data-testid="stNumberInput"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stMultiselect"] label,
        [data-testid="stDateInput"] label,
        [data-testid="stRadio"] label,
        [data-testid="stSlider"] label {{
            font-weight: 600;
            color: var(--text-strong);
        }}

        [data-testid="stRadio"] div[role="radiogroup"] label p,
        [data-testid="stSelectbox"] label p,
        [data-testid="stMultiselect"] label p,
        [data-testid="stTextInput"] label p {{
            color: inherit !important;
        }}

        .page-title {{
            font-size: 2.25rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }}

        .page-subtitle {{
            font-size: 1rem;
            color: var(--text-muted);
            margin-bottom: 1.1rem;
        }}

        .kpi-card {{
            background: linear-gradient(145deg, rgba(30, 76, 156, 0.95), rgba(11, 31, 58, 0.95));
            border-radius: 18px;
            padding: 1.3rem 1.5rem;
            box-shadow: 0 18px 36px rgba(11, 31, 58, 0.16);
            border: 1px solid rgba(12, 31, 58, 0.18);
            display: flex;
            gap: 1rem;
            align-items: center;
            height: 100%;
            color: var(--text-invert);
        }}

        .kpi-card.alert {{
            border-color: rgba(176, 48, 56, 0.45);
            box-shadow: 0 18px 44px rgba(176, 48, 56, 0.25);
        }}

        .kpi-icon {{
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: grid;
            place-items: center;
            font-size: 1.6rem;
            background: rgba(255, 255, 255, 0.18);
            color: #dce7f8;
        }}

        .kpi-title {{
            font-size: 0.9rem;
            color: rgba(255, 255, 255, 0.75);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .kpi-value {{
            font-size: 1.6rem;
            font-weight: 600;
            color: var(--text-invert);
            margin: 0.2rem 0;
        }}

        .kpi-subtitle {{
            font-size: 0.85rem;
            color: rgba(255, 255, 255, 0.7);
        }}

        .fiscal-pill {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            background: rgba(30, 76, 156, 0.12);
            border-radius: 999px;
            padding: 0.35rem 0.9rem;
            font-size: 0.85rem;
            color: var(--brand-navy);
            box-shadow: inset 0 0 0 1px rgba(30, 76, 156, 0.25);
        }}

        .control-panel {{
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(231, 238, 248, 0.88));
            border-radius: 22px;
            padding: 1.1rem 1.4rem 1.25rem;
            border: 1px solid var(--surface-outline);
            box-shadow: 0 22px 44px rgba(11, 31, 58, 0.14);
            margin-bottom: 1.2rem;
        }}

        .control-panel h4,
        .control-panel h5,
        .control-panel label {{
            color: var(--brand-slate);
        }}

        .control-panel .stButton > button {{
            border-radius: 14px;
            background: linear-gradient(145deg, var(--brand-sky), var(--brand-navy));
            color: white;
            border: none;
            font-weight: 600;
            box-shadow: 0 14px 24px rgba(11, 31, 58, 0.18);
        }}

        .control-panel .stButton > button:hover {{
            background: linear-gradient(145deg, #4d86c0, #10274d);
        }}

        .control-panel div[data-baseweb="select"],
        .control-panel div[data-baseweb="input"],
        .control-panel div[data-baseweb="textarea"],
        .control-panel [data-testid="stDateInput"] div[data-baseweb="input"],
        .control-panel [data-testid="stColorPicker"] div[data-testid="stColorPickerValue"] {{
            background: rgba(255, 255, 255, 0.96);
            border-radius: 12px;
            border: 1px solid rgba(47, 60, 72, 0.18);
            color: var(--text-strong);
            box-shadow: inset 0 1px 2px rgba(11, 31, 58, 0.08);
        }}

        .control-panel div[data-baseweb="select"] span,
        .control-panel div[data-baseweb="select"] input,
        .control-panel div[data-baseweb="input"] input,
        .control-panel div[data-baseweb="textarea"] textarea,
        .control-panel [data-testid="stDateInput"] input,
        .control-panel [data-testid="stColorPicker"] input {{
            color: var(--text-strong) !important;
        }}

        [data-testid="stRadio"] div[role="radiogroup"] {{
            gap: 0.45rem;
            flex-wrap: wrap;
        }}

        [data-testid="stRadio"] div[role="radiogroup"] > label {{
            border-radius: 999px;
            padding: 0.35rem 0.85rem;
            border: 1px solid rgba(30, 76, 156, 0.22);
            background: rgba(255, 255, 255, 0.92);
            color: var(--text-strong) !important;
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            transition: background 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease, color 0.2s ease;
            font-weight: 600;
        }}

        [data-testid="stRadio"] div[role="radiogroup"] > label:hover {{
            background: rgba(77, 126, 168, 0.16);
            border-color: rgba(77, 126, 168, 0.38);
            color: var(--brand-navy) !important;
            box-shadow: 0 8px 18px rgba(11, 31, 58, 0.12);
        }}

        [data-testid="stRadio"] div[role="radiogroup"] > label:has(div[aria-checked="true"]) {{
            background: linear-gradient(135deg, rgba(30, 76, 156, 0.88), rgba(11, 31, 58, 0.92));
            border-color: rgba(11, 31, 58, 0.65);
            color: var(--text-invert) !important;
            box-shadow: 0 10px 22px rgba(11, 31, 58, 0.22);
        }}

        [data-testid="stRadio"] div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] p,
        [data-testid="stRadio"] div[role="radiogroup"] > label div[data-testid="stMarkdownContainer"] span {{
            color: inherit !important;
            margin: 0;
        }}

        [data-testid="stRadio"] div[role="radiogroup"] > label:not(:has(div[aria-checked="true"])) > div:nth-of-type(2),
        [data-testid="stRadio"] div[role="radiogroup"] > label:not(:has(div[aria-checked="true"])) > div:nth-of-type(2) *,
        [data-testid="stRadio"] div[role="radiogroup"] > label:not(:has(div[aria-checked="true"])) div[data-testid="stMarkdownContainer"] {{
            color: #3b4350 !important;
        }}

        [data-testid="stRadio"] div[role="radiogroup"] > label:has(div[aria-checked="true"]) > div:nth-of-type(2),
        [data-testid="stRadio"] div[role="radiogroup"] > label:has(div[aria-checked="true"]) > div:nth-of-type(2) *,
        [data-testid="stRadio"] div[role="radiogroup"] > label:has(div[aria-checked="true"]) div[data-testid="stMarkdownContainer"] p,
        [data-testid="stRadio"] div[role="radiogroup"] > label:has(div[aria-checked="true"]) div[data-testid="stMarkdownContainer"] span {{
            color: var(--text-invert) !important;
        }}

        .quick-actions {{
            margin-top: 0.4rem;
        }}

        .quick-actions .stButton > button {{
            background: rgba(30, 76, 156, 0.12) !important;
            color: #3a4553 !important;
            border-radius: 12px !important;
            border: 1px solid rgba(30, 76, 156, 0.3) !important;
            font-weight: 600 !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .quick-actions .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 14px 28px rgba(30, 76, 156, 0.18) !important;
        }}

        .quick-hint {{
            font-size: 0.8rem;
            color: var(--text-muted);
            padding-top: 0.35rem;
        }}

        .status-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.25rem;
            font-size: 0.85rem;
            font-weight: 600;
            border-radius: 999px;
            padding: 0.25rem 0.65rem;
            background: rgba(30, 76, 156, 0.12);
            color: var(--brand-navy);
        }}

        .status-badge.info {{
            background: rgba(77, 126, 168, 0.18);
            color: var(--brand-sky);
        }}

        .status-badge.success {{
            background: rgba(47, 158, 91, 0.15);
            color: #2F9E5B;
        }}

        .status-badge.warn {{
            background: rgba(201, 162, 39, 0.18);
            color: var(--brand-gold);
        }}

        .status-badge.alert {{
            background: rgba(176, 48, 56, 0.15);
            color: var(--brand-crimson);
        }}

        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] p {{
            color: var(--text-strong) !important;
        }}

        [data-testid="stFileUploader"] section {{
            background: rgba(255, 255, 255, 0.9);
            border: 1px dashed rgba(77, 126, 168, 0.55);
            border-radius: 14px;
            color: var(--text-strong);
        }}

        [data-testid="stFileUploader"] section:hover {{
            border-color: var(--brand-sky);
            background: rgba(233, 240, 252, 0.9);
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
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(231, 238, 248, 0.92));
            padding: 0.6rem 0.6rem 0.2rem;
            box-shadow: 0 10px 26px rgba(11, 31, 58, 0.1);
            margin-bottom: 1.2rem;
            border: 1px solid rgba(30, 76, 156, 0.15);
        }}

        [data-testid="stDataFrame"] table thead tr th {{
            background: linear-gradient(135deg, rgba(30, 76, 156, 0.95), rgba(11, 31, 58, 0.95));
            color: var(--text-invert) !important;
            font-weight: 600 !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.25) !important;
        }}

        [data-testid="stDataFrame"] table tbody tr:nth-child(odd) {{
            background-color: rgba(239, 244, 252, 0.85);
        }}

        [data-testid="stDataFrame"] table tbody tr:nth-child(even) {{
            background-color: rgba(255, 255, 255, 0.95);
        }}

        [data-testid="stDataFrame"] table tbody tr:hover {{
            background-color: rgba(77, 126, 168, 0.16) !important;
        }}

        .help-fab {{
            position: fixed;
            bottom: 26px;
            right: 32px;
            background: var(--brand-navy);
            color: white !important;
            padding: 0.75rem 1.1rem;
            border-radius: 999px;
            font-weight: 600;
            text-decoration: none;
            box-shadow: 0 20px 36px rgba(11, 31, 58, 0.22);
            z-index: 1200;
        }}

        .help-fab:hover {{
            background: #10284f;
        }}

        button[data-testid="baseButton-secondary"],
        div[data-testid="stFormSubmitButton"] button:not([data-testid="baseButton-primary"]) {{
            background: var(--surface-card) !important;
            color: var(--text-strong) !important;
            border: 1px solid rgba(47, 60, 72, 0.25) !important;
            box-shadow: none !important;
        }}

        button[data-testid="baseButton-secondary"]:hover,
        div[data-testid="stFormSubmitButton"] button:not([data-testid="baseButton-primary"]):hover {{
            border-color: var(--brand-sky) !important;
            color: var(--brand-navy) !important;
        }}

        div[data-testid="stAlert"] {{
            border-radius: 14px;
            border: 1px solid rgba(47, 60, 72, 0.18);
        }}

        div[data-testid="stAlert"] p {{
            color: var(--text-strong) !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(fiscal_year: int, fiscal_range: Tuple[date, date]) -> None:
    col_title, col_meta = st.columns([3.5, 2])
    with col_title:
        st.markdown('<div class="page-title">工事受注ダッシュボード</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="page-subtitle">主要指標とリスクをワンクリックで把握し、現場の次のアクションにつなげます。</div>',
            unsafe_allow_html=True,
        )
    with col_meta:
        fiscal_from, fiscal_to = fiscal_range
        st.markdown(
            f"<div style='display:flex;justify-content:flex-end;gap:0.4rem;align-items:center;'>"
            f"<span class='fiscal-pill'>FY {fiscal_year} : {fiscal_from:%Y.%m} - {fiscal_to:%Y.%m}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_quick_actions() -> None:
    actions = [
        {
            "label": "＋ 新規案件を登録",
            "description": "案件一覧タブを開き、登録フォームを立ち上げます。",
            "callback": trigger_new_project_modal,
        },
        {
            "label": "📊 最新の受注状況を見る",
            "description": "タイムラインで最新の受注状況を確認します。",
            "callback": lambda: switch_main_tab("タイムライン"),
        },
        {
            "label": "💹 工事別の粗利を確認",
            "description": "集計/分析タブの粗利指標へ移動します。",
            "callback": lambda: switch_main_tab("集計/分析"),
        },
        {
            "label": "⚙️ マスタ設定を開く",
            "description": "各種マスタや休日設定を編集します。",
            "callback": lambda: switch_main_tab("設定"),
        },
    ]
    st.markdown("<div class='quick-actions'>", unsafe_allow_html=True)
    cols = st.columns(len(actions))
    for idx, (col, action) in enumerate(zip(cols, actions)):
        with col:
            if st.button(
                action["label"],
                use_container_width=True,
                key=f"qa_{idx}",
                help=action["description"],
            ):
                action["callback"]()
            st.caption(action["description"])
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        "<a class='help-fab' href='#onboarding-guide' title='初めての方はこちらから操作手順を確認できます'>❓ チュートリアル</a>",
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
        st.error(f"インポート中にエラーが発生しました: {exc}")


def render_projects_tab(full_df: pd.DataFrame, filtered_df: pd.DataFrame, masters: Dict[str, List[str]]) -> None:
    st.subheader("案件一覧")
    col_add, col_draft, col_hint1, col_hint2 = st.columns([1.2, 1, 2.2, 2.2])
    if col_add.button(
        "＋ 新規案件を追加",
        type="primary",
        use_container_width=True,
        help="案件登録フォームを開きます。",
    ):
        st.session_state["show_project_modal"] = True

    draft_exists = bool(st.session_state.get("project_form_draft"))
    if col_draft.button(
        "下書きを開く",
        use_container_width=True,
        disabled=not draft_exists,
        help="保存済みの下書きがある場合に再開できます。",
    ) and draft_exists:
        st.session_state["show_project_modal"] = True

    col_hint1.markdown("<div class='quick-hint'>案件登録は専用フォームから行えます。</div>", unsafe_allow_html=True)
    col_hint2.markdown(
        "<div class='quick-hint'>編集後は下の保存ボタンで確定してください。</div>",
        unsafe_allow_html=True,
    )

    if st.session_state.get("show_project_modal"):
        status_options = sorted([s for s in full_df["ステータス"].dropna().unique() if s])
        contractor_options = sorted([c for c in full_df["元請区分"].dropna().unique() if c])
        clients = get_active_master_values(masters, "clients")
        categories = get_active_master_values(masters, "categories")
        managers = get_active_master_values(masters, "managers")
        today = date.today()

        def find_index(options_list: List[str], value: str) -> int:
            if not options_list:
                return 0
            try:
                return options_list.index(value)
            except ValueError:
                return 0

        default_draft = {
            "id": "",
            "案件名": "",
            "得意先": clients[0] if clients else "",
            "工種": categories[0] if categories else "",
            "元請区分": contractor_options[0] if contractor_options else "",
            "ステータス": status_options[0] if status_options else "",
            "着工日": today,
            "竣工日": today + relativedelta(months=3),
            "受注金額": 0,
            "予定原価": 0,
            "粗利率": 0,
            "担当者": managers[0] if managers else "",
            "月平均必要人数": 0.0,
            "備考": "",
        }
        draft = {**default_draft, **st.session_state.get("project_form_draft", {})}

        with modal_container("新規案件を登録", key="project_modal"):
            st.markdown("案件の基本情報を入力してください。必須項目は * で示しています。")
            with st.form("project_create_form"):
                id_value = st.text_input("* 案件ID", value=draft.get("id", ""))
                name_value = st.text_input("* 案件名", value=draft.get("案件名", ""))
                col_master = st.columns(2)
                client_value = col_master[0].selectbox(
                    "得意先",
                    clients or [""],
                    index=find_index(clients, draft.get("得意先", clients[0] if clients else "")),
                )
                category_value = col_master[1].selectbox(
                    "工種",
                    categories or [""],
                    index=find_index(categories, draft.get("工種", categories[0] if categories else "")),
                )
                col_secondary = st.columns(2)
                contractor_value = col_secondary[0].selectbox(
                    "元請区分",
                    contractor_options or [""],
                    index=find_index(contractor_options, draft.get("元請区分", contractor_options[0] if contractor_options else "")),
                )
                status_value = col_secondary[1].selectbox(
                    "ステータス",
                    status_options or [""],
                    index=find_index(status_options, draft.get("ステータス", status_options[0] if status_options else "")),
                )
                date_cols = st.columns(2)
                start_value = date_cols[0].date_input("* 着工日", value=draft.get("着工日", today))
                end_value = date_cols[1].date_input("* 竣工日", value=draft.get("竣工日", today + relativedelta(months=3)))
                finance_cols = st.columns(2)
                order_value = finance_cols[0].number_input(
                    "受注金額", min_value=0, value=int(draft.get("受注金額", 0))
                )
                cost_value = finance_cols[1].number_input(
                    "予定原価", min_value=0, value=int(draft.get("予定原価", 0))
                )
                extra_cols = st.columns(2)
                margin_value = extra_cols[0].number_input(
                    "粗利率(%)", min_value=-100, max_value=100, value=int(draft.get("粗利率", 0))
                )
                manager_value = extra_cols[1].selectbox(
                    "担当者",
                    managers or [""],
                    index=find_index(managers, draft.get("担当者", managers[0] if managers else "")),
                )
                manpower_value = st.number_input(
                    "月平均必要人数", min_value=0.0, value=float(draft.get("月平均必要人数", 0.0)), step=0.5
                )
                note_value = st.text_area("備考", value=draft.get("備考", ""))

                submit_col1, submit_col2, submit_col3 = st.columns([1, 1, 2])
                save_new = submit_col1.form_submit_button("登録して保存", type="primary")
                save_draft = submit_col2.form_submit_button("下書きを保存")
                cancel_modal = submit_col3.form_submit_button("閉じる")

                new_record = {
                    "id": id_value.strip(),
                    "案件名": name_value.strip(),
                    "得意先": client_value,
                    "工種": category_value,
                    "元請区分": contractor_value,
                    "ステータス": status_value,
                    "着工日": start_value,
                    "竣工日": end_value,
                    "受注金額": order_value,
                    "予定原価": cost_value,
                    "粗利率": margin_value,
                    "担当者": manager_value,
                    "月平均必要人数": manpower_value,
                    "備考": note_value,
                }

                if save_draft:
                    st.session_state["project_form_draft"] = new_record
                    st.toast("下書きを保存しました。", icon="📝")

                if cancel_modal:
                    st.session_state["show_project_modal"] = False
                    st.experimental_rerun()

                if save_new:
                    errors: List[str] = []
                    if not new_record["id"]:
                        errors.append("案件IDは必須です。")
                    if not new_record["案件名"]:
                        errors.append("案件名は必須です。")
                    if new_record["竣工日"] < new_record["着工日"]:
                        errors.append("竣工日は着工日以降に設定してください。")
                    existing_ids = set(full_df["id"].astype(str).str.strip())
                    if new_record["id"] in existing_ids:
                        errors.append("同じ案件IDが既に存在します。")

                    if errors:
                        for msg in errors:
                            st.error(msg)
                    else:
                        st.session_state.pop("project_form_draft", None)
                        st.session_state["show_project_modal"] = False
                        persist_record = {col: new_record.get(col, "") for col in PROJECT_BASE_COLUMNS}
                        persist_record["受注予定額"] = persist_record.get("受注予定額") or order_value
                        for numeric_col in PROJECT_NUMERIC_COLUMNS:
                            persist_record.setdefault(numeric_col, 0)
                        for date_col in PROJECT_DATE_COLUMNS:
                            persist_record.setdefault(date_col, None)
                        persist_record["受注予定額"] = persist_record.get("受注予定額", 0)
                        persist_df = pd.concat([full_df, pd.DataFrame([persist_record])], ignore_index=True)
                        save_projects(persist_df)
                        st.success("新規案件を保存しました。案件一覧を更新します。")
                        st.experimental_rerun()

    display_df = enrich_projects(filtered_df) if not filtered_df.empty else filtered_df.copy()
    if display_df.empty:
        st.info("条件に合致する案件がありません。フィルタを変更するか、新規行を追加してください。")
    display_df.reset_index(drop=True, inplace=True)

    alert_series = (
        display_df["予算超過"]
        if "予算超過" in display_df.columns
        else pd.Series(False, index=display_df.index)
    )
    alert_df = display_df[alert_series == True]
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
        "受注予定額": st.column_config.NumberColumn("受注予定額", format="%,d 円", min_value=0),
        "受注金額": st.column_config.NumberColumn("受注金額", format="%,d 円", min_value=0),
        "予算原価": st.column_config.NumberColumn("予算原価", format="%,d 円", min_value=0),
        "予定原価": st.column_config.NumberColumn("予定原価", format="%,d 円", min_value=0),
        "実績原価": st.column_config.NumberColumn("実績原価", format="%,d 円", min_value=0),
        "粗利率": st.column_config.NumberColumn("粗利率", format="%.1f %%", min_value=-100, max_value=100),
        "進捗率": st.column_config.NumberColumn("進捗率", format="%.1f %%", min_value=0, max_value=100),
        "月平均必要人数": st.column_config.NumberColumn("月平均必要人数", format="%.1f 人", min_value=0),
        "粗利額": st.column_config.NumberColumn("粗利額", format="%,d 円", disabled=True),
        "原価率": st.column_config.NumberColumn("原価率", format="%.1f %%", disabled=True),
        "受注差異": st.column_config.NumberColumn("受注差異", format="%,d 円", disabled=True),
        "予算乖離額": st.column_config.NumberColumn("予算乖離額", format="%,d 円", disabled=True),
        "完成工事高": st.column_config.NumberColumn("完成工事高", format="%,d 円", disabled=True),
        "実行粗利": st.column_config.NumberColumn("実行粗利", format="%,d 円", disabled=True),
        "想定進捗率": st.column_config.NumberColumn("想定進捗率", format="%.1f %%", disabled=True),
        "進捗差異": st.column_config.NumberColumn("進捗差異", format="%.1f %%", disabled=True),
        "遅延日数": st.column_config.NumberColumn("遅延日数", format="%d 日", disabled=True),
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
        num_rows="fixed",
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
            status_badge = format_status_badge(selected_row.get("ステータス", ""))
            risk_badge = format_risk_badge(selected_row.get("リスクレベル", ""))
            detail_cols = st.columns(2)
            detail_cols[0].markdown(f"**案件ID**: {selected_row['id']}")
            detail_cols[0].markdown(
                f"**ステータス**: {status_badge}",
                unsafe_allow_html=True,
            )
            detail_cols[0].markdown(f"**工種**: {selected_row['工種']}")
            detail_cols[0].markdown(f"**元請区分**: {selected_row['元請区分']}")
            detail_cols[1].markdown(f"**担当者**: {selected_row['担当者']}")
            detail_cols[1].markdown(f"**得意先**: {selected_row['得意先']}")
            detail_cols[1].markdown(f"**現場所在地**: {selected_row['現場所在地']}")
            st.markdown("**期間**")
            st.markdown(
                f"着工日: {format_date(selected_row['着工日'])} / 竣工日: {format_date(selected_row['竣工日'])}"
            )
            progress_value = float(selected_row.get("進捗率", 0) or 0)
            st.markdown(f"**進捗率**: {progress_value:.1f}%")
            st.progress(min(max(progress_value / 100, 0.0), 1.0))
            st.markdown("**リスク指標**", unsafe_allow_html=True)
            st.markdown(risk_badge, unsafe_allow_html=True)
            st.markdown("**リスクメモ**")
            st.write(selected_row.get("リスクメモ", "-"))
            st.markdown("**備考**")
            st.write(selected_row.get("備考", "-"))
            st.caption("添付ファイルは案件詳細ページから確認・追加できます。")
    else:
        st.info("詳細を確認したい案件を一覧から選択してください。")


def render_summary_tab(df: pd.DataFrame, monthly: pd.DataFrame) -> None:
    st.subheader("集計 / 分析")
    enriched = enrich_projects(df)

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
        xaxis=dict(
            title="年月",
            tickfont=dict(color=BRAND_COLORS["slate"]),
            titlefont=dict(color=BRAND_COLORS["slate"]),
            gridcolor=BRAND_COLORS["cloud"],
            linecolor=BRAND_COLORS["cloud"],
        ),
        yaxis=dict(
            title="金額",
            gridcolor=BRAND_COLORS["cloud"],
            zerolinecolor=BRAND_COLORS["cloud"],
            tickfont=dict(color=BRAND_COLORS["slate"]),
            titlefont=dict(color=BRAND_COLORS["slate"]),
        ),
        yaxis2=dict(
            title="粗利率 (%)",
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color=BRAND_COLORS["slate"]),
            titlefont=dict(color=BRAND_COLORS["slate"]),
        ),
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
        xaxis=dict(
            title="年月",
            tickfont=dict(color=BRAND_COLORS["slate"]),
            titlefont=dict(color=BRAND_COLORS["slate"]),
            gridcolor=BRAND_COLORS["cloud"],
            linecolor=BRAND_COLORS["cloud"],
        ),
        yaxis=dict(
            title="キャッシュフロー",
            gridcolor=BRAND_COLORS["cloud"],
            zerolinecolor=BRAND_COLORS["cloud"],
            tickfont=dict(color=BRAND_COLORS["slate"]),
            titlefont=dict(color=BRAND_COLORS["slate"]),
        ),
        yaxis2=dict(
            title="累計 (円)",
            overlaying="y",
            side="right",
            gridcolor="rgba(0,0,0,0)",
            tickfont=dict(color=BRAND_COLORS["slate"]),
            titlefont=dict(color=BRAND_COLORS["slate"]),
        ),
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
            xaxis=dict(
                title="粗利率",
                gridcolor=BRAND_COLORS["cloud"],
                tickfont=dict(color=BRAND_COLORS["slate"]),
                titlefont=dict(color=BRAND_COLORS["slate"]),
            ),
            yaxis=dict(
                title="件数",
                gridcolor=BRAND_COLORS["cloud"],
                tickfont=dict(color=BRAND_COLORS["slate"]),
                titlefont=dict(color=BRAND_COLORS["slate"]),
            ),
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
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {display: none !important;}
        [data-testid="stSidebarNavSeparator"] {display: none !important;}
        .settings-nav {display:flex; gap:0.75rem; flex-wrap:wrap; margin:0.5rem 0 1.2rem;}
        .settings-nav a {
            background: var(--brand-navy);
            color: #fff;
            padding: 0.45rem 1rem;
            border-radius: 999px;
            font-size: 0.85rem;
            text-decoration: none;
            box-shadow: 0 8px 18px rgba(11,31,58,0.18);
        }
        .settings-nav a:hover {background: #10284f;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="settings-nav">
            <a href="#settings-masters">マスタ管理</a>
            <a href="#settings-holidays">休日カレンダー</a>
            <a href="#settings-display">表示設定</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div id='settings-masters'></div>", unsafe_allow_html=True)
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
        controls = st.columns([1.2, 1, 3])
        modal_flag = f"{key}_show_modal"
        draft_key = f"{key}_draft"
        if controls[0].button(f"＋ {label}を追加", key=f"{key}_add"):
            st.session_state[modal_flag] = True
        draft_exists = bool(st.session_state.get(draft_key))
        if controls[1].button("下書きを開く", key=f"{key}_open_draft", disabled=not draft_exists) and draft_exists:
            st.session_state[modal_flag] = True
        controls[2].markdown(
            "<div class='quick-hint'>新規追加はフォームから行い、最後に設定保存ボタンで確定します。</div>",
            unsafe_allow_html=True,
        )

        entries = normalize_master_entries(masters.get(key, []))
        masters[key] = entries
        base_df = pd.DataFrame(entries)
        if base_df.empty:
            base_df = pd.DataFrame({"name": [], "active": []})
        base_df["active"] = base_df.get("active", True)
        if st.session_state.get(modal_flag):
            draft = st.session_state.get(draft_key, {"name": "", "active": True})
            with modal_container(f"{label}を新規追加", key=f"{key}_modal"):
                with st.form(f"{key}_form"):
                    name_value = st.text_input("* 名称", value=draft.get("name", ""))
                    active_value = st.checkbox("有効", value=bool(draft.get("active", True)))
                    modal_cols = st.columns([1, 1, 2])
                    submit_new = modal_cols[0].form_submit_button("登録", type="primary")
                    submit_draft = modal_cols[1].form_submit_button("下書きを保存")
                    cancel_modal = modal_cols[2].form_submit_button("閉じる")

                    if submit_draft:
                        st.session_state[draft_key] = {"name": name_value, "active": active_value}
                        st.toast("下書きを保存しました。", icon="📝")

                    if cancel_modal:
                        st.session_state[modal_flag] = False
                        st.experimental_rerun()

                    if submit_new:
                        errors: List[str] = []
                        cleaned = name_value.strip()
                        if not cleaned:
                            errors.append("名称は必須です。")
                        elif cleaned in [entry["name"] for entry in entries]:
                            errors.append("同じ名称が既に登録されています。")
                        if errors:
                            for msg in errors:
                                st.error(msg)
                        else:
                            entries.append({"name": cleaned, "active": active_value})
                            masters[key] = entries
                            st.session_state.pop(draft_key, None)
                            st.session_state[modal_flag] = False
                            st.success(f"{label}を追加しました。設定保存で反映されます。")
                            st.experimental_rerun()

        editor = st.data_editor(
            base_df,
            num_rows="fixed",
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

    st.markdown("<div id='settings-holidays'></div>", unsafe_allow_html=True)
    st.markdown("### 休日カレンダー")
    holidays_df = pd.DataFrame({"休日": masters.get("holidays", [])})
    holidays_edit = st.data_editor(holidays_df, num_rows="dynamic", hide_index=True)

    st.markdown("<div id='settings-display'></div>", unsafe_allow_html=True)
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
    apply_brand_theme()
    ensure_data_files()
    masters = load_masters()

    try:
        projects_df = load_projects()
    except Exception as exc:
        st.error(f"データの読み込みに失敗しました: {exc}")
        return

    header_year = st.session_state.get("fiscal_year_select", DEFAULT_FISCAL_YEAR)
    stored_range = st.session_state.get("period_range_state")
    if (
        isinstance(stored_range, tuple)
        and len(stored_range) == 2
        and all(isinstance(v, date) for v in stored_range)
    ):
        header_range = stored_range  # type: ignore[assignment]
    else:
        header_range = get_fiscal_year_range(header_year)

    render_page_header(header_year, header_range)

    filters = render_control_panel(projects_df, masters)
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

    tab_labels = ["タイムライン", "案件一覧", "集計/分析", "設定"]
    if "main_tabs" not in st.session_state:
        st.session_state["main_tabs"] = tab_labels[0]
    selected_tab = st.radio(
        "表示タブ",
        tab_labels,
        horizontal=True,
        key="main_tabs",
        label_visibility="collapsed",
    )

    st.divider()

    if selected_tab == "タイムライン":
        st.markdown("<div id='timeline-section'></div>", unsafe_allow_html=True)
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

    elif selected_tab == "案件一覧":
        st.markdown("<div id='project-section'></div>", unsafe_allow_html=True)
        render_projects_tab(projects_df, filtered_df, masters)

    elif selected_tab == "集計/分析":
        st.markdown("<div id='analysis-section'></div>", unsafe_allow_html=True)
        render_summary_tab(enriched_filtered_df, monthly_df)

    else:
        render_settings_tab(masters)

    st.markdown("<div id='onboarding-guide'></div>", unsafe_allow_html=True)
    with st.expander("クイックチュートリアル / オンボーディング", expanded=False):
        st.markdown(
            """
            1. 左上の「☰ フィルタ」で事業年度や期間を切り替えます。
            2. 「＋ 新規案件を登録」から案件フォームを開き、必須項目を入力して保存します。
            3. タイムラインで進捗とリスクを把握し、集計タブで粗利やキャッシュを確認してください。
            4. 設定タブから得意先・工種などのマスタや休日を整備できます。
            """
        )


if __name__ == "__main__":
    main()
