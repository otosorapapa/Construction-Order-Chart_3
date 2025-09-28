import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

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
                    "受注金額": 25_000_000,
                    "予定原価": 19_000_000,
                    "粗利率": 24,
                    "月平均必要人数": 6,
                    "現場所在地": "福岡",
                    "担当者": "山中",
                    "備考": "体育館の基礎および型枠一式",
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
                    "受注金額": 32_000_000,
                    "予定原価": 24_500_000,
                    "粗利率": 23,
                    "月平均必要人数": 7,
                    "現場所在地": "熊本",
                    "担当者": "近藤",
                    "備考": "河川敷工事の夜間作業あり",
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
                    "受注金額": 58_000_000,
                    "予定原価": 44_000_000,
                    "粗利率": 24,
                    "月平均必要人数": 8,
                    "現場所在地": "福岡",
                    "担当者": "山中",
                    "備考": "地下躯体に注意",
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
                    "受注金額": 60_000_000,
                    "予定原価": 46_000_000,
                    "粗利率": 23,
                    "月平均必要人数": 9,
                    "現場所在地": "福岡",
                    "担当者": "山中",
                    "備考": "JV案件",
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
                    "受注金額": 45_000_000,
                    "予定原価": 35_000_000,
                    "粗利率": 22,
                    "月平均必要人数": 7,
                    "現場所在地": "福岡",
                    "担当者": "近藤",
                    "備考": "未定要素あり",
                },
            ]
        )
        sample.to_csv(PROJECT_CSV, index=False)
    if not os.path.exists(MASTERS_JSON):
        masters = {
            "clients": ["金子技建", "佐藤組", "新宮開発", "高野組"],
            "categories": ["建築", "土木", "型枠", "その他"],
            "managers": ["山中", "近藤", "田中"],
            "holidays": [],
            "currency_format": "#,###",
            "decimal_places": 0,
        }
        with open(MASTERS_JSON, "w", encoding="utf-8") as f:
            json.dump(masters, f, ensure_ascii=False, indent=2)


def load_masters() -> Dict[str, List[str]]:
    with open(MASTERS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_masters(masters: Dict[str, List[str]]) -> None:
    with open(MASTERS_JSON, "w", encoding="utf-8") as f:
        json.dump(masters, f, ensure_ascii=False, indent=2)


def load_projects() -> pd.DataFrame:
    df = pd.read_csv(PROJECT_CSV)
    date_cols = ["着工日", "竣工日"]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    numeric_cols = ["受注金額", "予定原価", "粗利率", "月平均必要人数"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def save_projects(df: pd.DataFrame) -> None:
    out_df = df.copy()
    out_df.sort_values(by="着工日", inplace=True, ignore_index=True)
    out_df.to_csv(PROJECT_CSV, index=False)


def get_fiscal_year_range(year: int) -> Tuple[date, date]:
    start = date(year, FISCAL_START_MONTH, 1)
    end = start + relativedelta(years=1) - relativedelta(days=1)
    return start, end


def apply_filters(df: pd.DataFrame, filters: FilterState) -> pd.DataFrame:
    result = df.copy()
    if filters.period_from:
        result = result[result["竣工日"] >= filters.period_from]
    if filters.period_to:
        result = result[result["着工日"] <= filters.period_to]
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
    if filters.margin_range:
        low, high = filters.margin_range
        result = result[(result["粗利率"] >= low) & (result["粗利率"] <= high)]
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
        hover_text = (
            f"案件名: {row['案件名']}<br>期間: {row['着工日']}〜{row['竣工日']}<br>"
            f"得意先: {row['得意先']}<br>工種: {row['工種']}<br>ステータス: {row['ステータス']}<br>"
            f"粗利率: {row['粗利率']}%<br>備考: {row['備考']}"
        )
        raw_value = row[color_key]
        legend_value = str(raw_value) if pd.notna(raw_value) and raw_value != "" else "未設定"
        showlegend = False
        if legend_value not in legend_tracker:
            legend_tracker[legend_value] = True
            showlegend = True
        bar_color = color_map.get(raw_value, filters.bar_color)
        fig.add_trace(
            go.Bar(
                x=[(pd.to_datetime(row["竣工日"]) - pd.to_datetime(row["着工日"]) + pd.Timedelta(days=1)).days],
                y=[row["案件名"]],
                base=pd.to_datetime(row["着工日"]),
                orientation="h",
                marker=dict(color=bar_color),
                hovertemplate=hover_text,
                name=legend_value,
                legendgroup=legend_value,
                showlegend=showlegend,
            )
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
    for idx, row in df.iterrows():
        if pd.isna(row["着工日"]) or pd.isna(row["竣工日"]):
            errors.append(f"行 {idx + 1}: 着工日・竣工日は必須です。")
            continue
        if row["竣工日"] < row["着工日"]:
            errors.append(f"行 {idx + 1}: 竣工日は着工日以降にしてください。")
        if row["受注金額"] < 0:
            errors.append(f"行 {idx + 1}: 受注金額は 0 以上にしてください。")
        if row["予定原価"] < 0:
            errors.append(f"行 {idx + 1}: 予定原価は 0 以上にしてください。")
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
        for _, row in df.iterrows():
            proj_start = pd.to_datetime(row["着工日"])
            proj_end = pd.to_datetime(row["竣工日"])
            overlap_start = max(proj_start, month_start)
            overlap_end = min(proj_end, month_end)
            if overlap_start > overlap_end:
                continue
            total_days = (proj_end - proj_start).days + 1
            overlap_days = (overlap_end - overlap_start).days + 1
            ratio = overlap_days / total_days if total_days > 0 else 0
            month_revenue += row["受注金額"] * ratio
            month_cost += row["予定原価"] * ratio
            month_manpower += row["月平均必要人数"] * ratio
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
            }
        )
    return pd.DataFrame(records)


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
    category_options = masters.get("categories", [])
    contractor_options = sorted(df["元請区分"].dropna().unique())
    client_options = masters.get("clients", [])
    manager_options = masters.get("managers", [])
    prefecture_options = sorted(df["現場所在地"].dropna().unique())

    status = st.sidebar.multiselect("案件ステータス", status_options)
    category = st.sidebar.multiselect("工種", category_options)
    contractor = st.sidebar.multiselect("元請区分", contractor_options)
    clients = st.sidebar.multiselect("主要取引先", client_options)
    managers = st.sidebar.multiselect("担当者", manager_options)
    prefectures = st.sidebar.multiselect("現場所在地 (都道府県)", prefecture_options)
    margin_min, margin_max = st.sidebar.slider("粗利率レンジ (%)", -100, 100, (-100, 100))

    st.sidebar.subheader("表示設定")
    color_key = st.sidebar.selectbox("色分けキー", ["ステータス", "工種", "元請区分"])
    bar_color = st.sidebar.color_picker("バー基調色", DEFAULT_BAR_COLOR)
    show_grid = st.sidebar.checkbox("月グリッド線を表示", True)
    label_density = st.sidebar.selectbox("ラベル密度", ["高", "中", "低"], index=1)

    st.sidebar.subheader("CSV 入出力")
    export_target = st.sidebar.radio("エクスポート対象", ["案件データ", "月次集計"], index=0, key="export_target_radio")
    st.session_state["export_target"] = export_target
    st.session_state["export_placeholder"] = st.sidebar.empty()

    uploaded = st.sidebar.file_uploader("CSV インポート", type="csv")
    if uploaded is not None:
        mode = st.sidebar.radio("取り込み方法", ["マージ", "置換"], index=0, key="import_mode")
        if st.sidebar.button("インポート実行"):
            import_projects(uploaded, mode)
            st.sidebar.success("インポートが完了しました。ページを再読み込みしてください。")

    st.sidebar.download_button(
        "テンプレートダウンロード",
        data=prepare_export(pd.DataFrame(columns=df.columns)),
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
        color_key=color_key,
        show_grid=show_grid,
        label_density=label_density,
        bar_color=bar_color,
    )


def prepare_export(df: pd.DataFrame) -> str:
    if df is None:
        return ""
    return df.to_csv(index=False)


def import_projects(uploaded, mode: str) -> None:
    try:
        new_df = pd.read_csv(uploaded)
        for col in ["着工日", "竣工日"]:
            new_df[col] = pd.to_datetime(new_df[col], errors="coerce").dt.date
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
    display_df = filtered_df.copy()
    if display_df.empty:
        st.info("条件に合致する案件がありません。フィルタを変更するか、新規行を追加してください。")
    display_df.reset_index(drop=True, inplace=True)
    edited = st.data_editor(
        display_df,
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            "着工日": st.column_config.DateColumn("着工日"),
            "竣工日": st.column_config.DateColumn("竣工日"),
            "受注金額": st.column_config.NumberColumn("受注金額", format="%d"),
            "予定原価": st.column_config.NumberColumn("予定原価", format="%d"),
            "粗利率": st.column_config.NumberColumn("粗利率", format="%d"),
            "月平均必要人数": st.column_config.NumberColumn("月平均必要人数", format="%0.1f"),
        },
        key="project_editor",
    )
    if st.button("保存", type="primary"):
        try:
            edited["着工日"] = pd.to_datetime(edited["着工日"], errors="coerce").dt.date
            edited["竣工日"] = pd.to_datetime(edited["竣工日"], errors="coerce").dt.date
            edited["受注金額"] = pd.to_numeric(edited["受注金額"], errors="coerce").fillna(0)
            edited["予定原価"] = pd.to_numeric(edited["予定原価"], errors="coerce").fillna(0)
            edited["粗利率"] = pd.to_numeric(edited["粗利率"], errors="coerce").fillna(0)
            edited["月平均必要人数"] = pd.to_numeric(edited["月平均必要人数"], errors="coerce").fillna(0)
            valid, errors = validate_projects(edited)
            if not valid:
                for msg in errors:
                    st.error(msg)
                return
            remaining = full_df[~full_df["id"].isin(edited["id"])]
            combined = pd.concat([edited, remaining], ignore_index=True)
            save_projects(combined)
            st.success("保存しました。ページを再読み込みしてください。")
        except Exception as exc:
            st.error(f"保存中にエラーが発生しました: {exc}")


def render_summary_tab(df: pd.DataFrame, monthly: pd.DataFrame) -> None:
    st.subheader("集計 / 分析")

    fig = go.Figure()
    fig.add_bar(x=monthly["年月"], y=monthly["受注金額"], name="受注金額")
    fig.add_bar(x=monthly["年月"], y=monthly["予定原価"], name="予定原価")
    fig.add_trace(go.Scatter(x=monthly["年月"], y=monthly["粗利"], mode="lines+markers", name="粗利"))
    fig.add_trace(
        go.Scatter(
            x=monthly["年月"],
            y=monthly["粗利率"],
            mode="lines",
            name="粗利率",
            yaxis="y2",
        )
    )
    fig.update_layout(
        template="plotly_white",
        barmode="group",
        yaxis=dict(title="金額"),
        yaxis2=dict(title="粗利率 (%)", overlaying="y", side="right"),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if df.empty:
            st.info("対象データがありません。")
        else:
            pie1 = go.Figure(data=[go.Pie(labels=df["工種"], values=df["受注金額"], hole=0.3)])
            pie1.update_layout(title="工種別構成比")
            st.plotly_chart(pie1, use_container_width=True)
    with col2:
        if df.empty:
            st.info("対象データがありません。")
        else:
            pie2 = go.Figure(data=[go.Pie(labels=df["得意先"], values=df["受注金額"], hole=0.3)])
            pie2.update_layout(title="得意先別構成比")
            st.plotly_chart(pie2, use_container_width=True)

    if df.empty:
        st.info("粗利率の分布を表示するデータがありません。")
    else:
        hist = go.Figure(data=[go.Histogram(x=df["粗利率"], nbinsx=10)])
        hist.update_layout(title="粗利率ヒストグラム", template="plotly_white")
        st.plotly_chart(hist, use_container_width=True)

    st.dataframe(
        monthly.assign(年月=monthly["年月"].dt.strftime("%Y-%m")),
        use_container_width=True,
    )


def render_settings_tab(masters: Dict[str, List[str]]) -> None:
    st.subheader("設定")
    st.markdown("### マスタ管理")
    clients_text = st.text_area("得意先一覧 (改行区切り)", "\n".join(masters.get("clients", [])))
    categories_text = st.text_area("工種一覧 (改行区切り)", "\n".join(masters.get("categories", [])))
    managers_text = st.text_area("担当者一覧 (改行区切り)", "\n".join(masters.get("managers", [])))

    st.markdown("### 休日カレンダー")
    holidays_df = pd.DataFrame({"休日": masters.get("holidays", [])})
    holidays_edit = st.data_editor(holidays_df, num_rows="dynamic", hide_index=True)

    st.markdown("### 表示設定")
    currency_format = st.text_input("通貨フォーマット", masters.get("currency_format", "#,###"))
    decimal_places = st.number_input("小数点以下桁数", min_value=0, max_value=4, value=int(masters.get("decimal_places", 0)))

    if st.button("設定を保存", type="primary"):
        masters["clients"] = [c.strip() for c in clients_text.splitlines() if c.strip()]
        masters["categories"] = [c.strip() for c in categories_text.splitlines() if c.strip()]
        masters["managers"] = [c.strip() for c in managers_text.splitlines() if c.strip()]
        masters["holidays"] = [
            d.strftime("%Y-%m-%d") if isinstance(d, (datetime, pd.Timestamp)) else str(d)
            for d in holidays_edit["休日"].dropna().tolist()
        ]
        masters["currency_format"] = currency_format or "#,###"
        masters["decimal_places"] = decimal_places
        save_masters(masters)
        st.success("設定を保存しました。")


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
    monthly_df = compute_monthly_aggregation(filtered_df, fiscal_range)
    st.session_state["monthly"] = monthly_df

    export_placeholder = st.session_state.get("export_placeholder")
    export_target = st.session_state.get("export_target", "案件データ")
    export_source = filtered_df if export_target == "案件データ" else monthly_df
    if export_placeholder is not None:
        export_placeholder.download_button(
            "エクスポート",
            data=prepare_export(export_source),
            file_name="projects_export.csv" if export_target == "案件データ" else "monthly_summary.csv",
            mime="text/csv",
        )

    tabs = st.tabs(["タイムライン", "案件一覧", "集計/分析", "設定"])

    with tabs[0]:
        st.subheader("タイムライン")
        timeline_fig = create_timeline(filtered_df, filters, fiscal_range)
        st.plotly_chart(timeline_fig, use_container_width=True)

    with tabs[1]:
        render_projects_tab(projects_df, filtered_df)

    with tabs[2]:
        render_summary_tab(filtered_df, monthly_df)

    with tabs[3]:
        render_settings_tab(masters)


if __name__ == "__main__":
    main()
