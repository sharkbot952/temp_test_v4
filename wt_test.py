import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import hashlib

st.set_page_config(page_title="試験版",layout="wide")

# =====================================================
# 固定設定
# =====================================================

# アプリ直下の data フォルダを基準にする（Streamlit Cloud 対応）
APP_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = APP_DIR / 'data'
CSV_PATH = DEFAULT_BASE_DIR / 'Taiki_temp.csv'
ENCODING = "utf-8-sig"
DATE_COL = "DATE"
METRIC = "depth_avg"

HOT_RED = "#d32f2f"  # 強めの赤

# =====================================================
# 共通ユーティリティ
# =====================================================
def safe_row_mean(df, cols):
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].mean(axis=1, skipna=True)

def year_color_map(years):
    palette = px.colors.qualitative.Set2
    return {y: palette[i % len(palette)] for i, y in enumerate(sorted(years))}

def rgba_from_color(color, alpha):
    color = str(color).strip()
    if color.startswith("rgb"):
        r, g, b = [int(x) for x in color[4:-1].split(",")]
        return f"rgba({r},{g},{b},{alpha})"
    if color.startswith("#"):
        r, g, b = px.colors.hex_to_rgb(color)
        return f"rgba({r},{g},{b},{alpha})"
    return f"rgba(0,0,0,{alpha})"

def rolling_ma(series: pd.Series, x: pd.Series, mode: str):
    if mode == "datetime":
        s = series.copy()
        s.index = pd.to_datetime(x)
        return s.sort_index().rolling("7D", min_periods=1).mean().reindex(s.index).values
    else:
        return series.rolling(window=7, min_periods=1).mean().values

def add_band(fig, x, ymin, ymax, color, alpha=0.30):
    fig.add_trace(go.Scatter(
        x=x, y=ymin, mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=x, y=ymax, mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor=rgba_from_color(color, alpha),
        showlegend=False,
        hoverinfo="skip"
    ))

def add_lines(fig, x, y_mean, y_ma, color, name, show_ma: bool):
    if show_ma:
        fig.add_trace(go.Scatter(
            x=x, y=y_mean, mode="lines",
            line=dict(color=rgba_from_color(color, 0.35), width=1.2),
            name=name,
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_ma, mode="lines",
            line=dict(color=color, width=2.6, dash="dot"),
            showlegend=False,
            hoverinfo="skip"
        ))
    else:
        fig.add_trace(go.Scatter(
            x=x, y=y_mean, mode="lines",
            line=dict(color=color, width=2.0),
            name=name,
            showlegend=True
        ))

# =====================================================
# 要約モード用ユーティリティ
# =====================================================
def dekad(day: int):
    if day <= 10:
        return "上旬"
    elif day <= 20:
        return "中旬"
    else:
        return "下旬"

def build_month_dekad_by_year(df, month, years):
    d = df[[DATE_COL, "Year", "Month", "Day", METRIC]].dropna().copy()
    d = d[d["Month"] == month]
    d["Dekad"] = d["Day"].apply(dekad)

    out = {}
    for dk in ["上旬", "中旬", "下旬"]:
        out[dk] = {}
        for y in sorted(years, reverse=True):
            g = d[(d["Year"] == y) & (d["Dekad"] == dk)]
            if g.empty:
                out[dk][y] = None
            else:
                out[dk][y] = {
                    "mean": g[METRIC].mean(),
                    "median": g[METRIC].median(),
                    "min": g[METRIC].min(),
                    "max": g[METRIC].max(),
                }
    return out

# =====================================================
# データ読み込み（ttl設定で一定時間ごとに強制再読込）
# =====================================================
@st.cache_data(show_spinner="データ読み込み中...", ttl=600)
def load_raw(csv_path: Path, _hash_val: str):
    df = pd.read_csv(str(csv_path), encoding=ENCODING)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).copy()

    df["1m_avg"] = safe_row_mean(df, ["1m(UML)", "1m(Tele)"])
    df["2m_avg"] = safe_row_mean(df, ["2m(UML)", "2m(Tele)"])
    df["3m_avg"] = safe_row_mean(df, ["3m(UML)", "3m(Tele)"])
    df[METRIC] = df[["1m_avg", "2m_avg", "3m_avg"]].mean(axis=1, skipna=True)

    df["Year"] = df[DATE_COL].dt.year
    df["Month"] = df[DATE_COL].dt.month
    df["Day"] = df[DATE_COL].dt.day
    return df

# --- 物理ファイルのチェック ---
p = Path(CSV_PATH)
if not p.exists():
    st.error(f"CSV が見つかりません: {CSV_PATH}（{p.resolve()}）")
    st.stop()

# ハッシュ計算（ファイルサイズも考慮してキャッシュ破りを確実にする）
csv_bytes = p.read_bytes()
file_hash = f"{hashlib.sha1(csv_bytes).hexdigest()}_{len(csv_bytes)}"

df_raw = load_raw(p, file_hash)

# デバッグ表示（サイドバー）
with st.sidebar:
    st.divider()
    st.subheader("📡 Data Sync Status")
 st.write(f'**App dir:** `{APP_DIR}`')
 st.write(f'**CSV path:** `{p.resolve()}`')
    last_date = df_raw[DATE_COL].max()
    st.write(f"**最新データの日時:**")
    st.code(last_date.strftime('%Y-%m-%d %H:%M'))
    st.write(f"**Hash:** `{file_hash[:12]}`")
    if st.button("強制キャッシュクリア"):
        st.cache_data.clear()
        st.rerun()

years = sorted(df_raw["Year"].dropna().unique().tolist())
CURRENT_YEAR = max(years)

# =====================================================
# UI：表示モード
# =====================================================
mode = st.radio("", ["要約", "グラフ"], horizontal=True, index=0)

# =====================================================
# 要約表示
# =====================================================
if mode == "要約":
    selected_month = st.selectbox(
        "月",
        options=list(range(1, 13)),
        index=pd.Timestamp.today().month - 1
    )

    summary = build_month_dekad_by_year(df_raw, selected_month, years)

    for dk in ["上旬", "中旬", "下旬"]:
        st.markdown(
            f"<div style='font-weight:600; margin-top:12px; border-bottom:1px solid #eee;'>{dk}</div>",
            unsafe_allow_html=True
        )

        for y, info in summary[dk].items():
            if info is None:
                st.markdown(
                    f"<div style='margin-left:1em; color:#999;'>{y}年：データなし</div>",
                    unsafe_allow_html=True
                )
            else:
                is_hot = (info["mean"] >= 20.0) or (info["max"] >= 20.0)
                color_main = HOT_RED if is_hot else "#000000"
                color_range = HOT_RED if is_hot else "#666666"
                style = "font-weight:bold; background-color:#f0f8ff; padding:2px 4px; border-radius:4px;" if y == CURRENT_YEAR else ""

                st.markdown(
                    f"""
                    <div style="margin-left:1em; margin-top:4px; {style}">
                        {y}年：
                        <span style="font-size:1.1em; color:{color_main};">
                            {info['mean']:.1f}℃
                        </span>
                        <span style="color:{color_range}; font-size:0.9em;">
                            （{info['min']:.1f}–{info['max']:.1f}）
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.markdown(
        """
        <div style="border-left:3px solid #ccc; margin-top:20px; padding-left:8px; color:#666; font-size:0.85em;">
        ※ 天候・時化・魚の状態メモ（将来追加）
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================
# グラフ表示
# =====================================================
else:
    c1, c2, c3 = st.columns([1.1, 1.1, 3.0])

    with c1:
        agg_label = st.radio("集計", ["日時", "日平均"], horizontal=True)
    with c2:
        smooth_label = st.radio("平滑化", ["なし", "移動平均(7日)"], horizontal=True)
    with c3:
        selected_years = st.multiselect(
            "年",
            years,
            default=years[-2:] if len(years) >= 2 else years
        )

    if not selected_years:
        st.stop()

    agg_mode = "datetime" if agg_label == "日時" else "daily"
    show_ma = (smooth_label != "なし")
    colors = year_color_map(selected_years)

    def build_timeseries_stats(df):
        d = df[["Year", DATE_COL, METRIC]].dropna().copy()
        d["X"] = d[DATE_COL].dt.floor("D") if agg_mode == "daily" else d[DATE_COL]
        return d.groupby(["Year", "X"])[METRIC].agg(["mean", "min", "max"]).reset_index()

    def build_same_monthday_stats(df):
        d = df[["Year", DATE_COL, "Month", "Day", METRIC]].dropna().copy()
        d = d[~((d["Month"] == 2) & (d["Day"] == 29))]
        if agg_mode == "daily":
            d["AlignX"] = pd.to_datetime(dict(year=2001, month=d["Month"], day=d["Day"]))
        else:
            t = d[DATE_COL].dt
            d["AlignX"] = pd.to_datetime(dict(
                year=2001, month=d["Month"], day=d["Day"],
                hour=t.hour, minute=t.minute, second=t.second
            ))
        return d.groupby(["Year", "AlignX"])[METRIC].agg(["mean", "min", "max"]).reset_index()

    ts_stats = build_timeseries_stats(df_raw)
    md_stats = build_same_monthday_stats(df_raw)

    tab_ts, tab_md = st.tabs(["時系列", "同月日比較"])

    with tab_ts:
        fig = go.Figure()
        for y in selected_years:
            d = ts_stats[ts_stats["Year"] == y]
            if d.empty: continue
            add_band(fig, d["X"], d["min"], d["max"], colors[y])
            ma = rolling_ma(d["mean"], d["X"], agg_mode) if show_ma else None
            add_lines(fig, d["X"], d["mean"], ma, colors[y], str(y), show_ma)
        fig.update_layout(template="plotly_white", height=520, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

    with tab_md:
        fig = go.Figure()
        for y in selected_years:
            d = md_stats[md_stats["Year"] == y]
            if d.empty: continue
            add_band(fig, d["AlignX"], d["min"], d["max"], colors[y])
            ma = rolling_ma(d["mean"], d["AlignX"], agg_mode) if show_ma else None
            add_lines(fig, d["AlignX"], d["mean"], ma, colors[y], str(y), show_ma)
        fig.update_layout(template="plotly_white", height=520, hovermode="x unified")
        fig.update_xaxes(tickformat="%m/%d")
        st.plotly_chart(fig, use_container_width=True)