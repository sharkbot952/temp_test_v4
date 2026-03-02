
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import hashlib

st.set_page_config(page_title="試験版", layout="wide")

# =====================================================
# 固定設定
# =====================================================
APP_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = APP_DIR / 'data'
CSV_PATH = DEFAULT_BASE_DIR / 'Taiki_temp.csv'

ENCODING = "utf-8-sig"
DATE_COL = "DATE"
METRIC = "depth_avg"  # 水温（1–3mの平均）
HOT_RED = "#d32f2f"   # 強めの赤

# =====================================================
# UI（ピル型ボタン）ユーティリティ
# =====================================================

def pill_toggle(options, default, key, label=""):
    """segmented_control（ピル）優先。無ければ radio(horizontal) にフォールバック。"""
    try:
        return st.segmented_control(
            label,
            options=options,
            default=default,
            key=key,
            label_visibility="collapsed",
        )
    except Exception:
        idx = options.index(default) if default in options else 0
        return st.radio(
            label,
            options,
            index=idx,
            horizontal=True,
            key=key,
            label_visibility="collapsed",
        )

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
        return (
            s.sort_index()
            .rolling("7D", min_periods=1)
            .mean()
            .reindex(s.index)
            .values
        )
    else:
        return series.rolling(window=7, min_periods=1).mean().values


# --- 拡張：yaxis指定（既存呼び出しはそのまま）---

def add_band(fig, x, ymin, ymax, color, alpha=0.30, yaxis="y"):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=ymin,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
            yaxis=yaxis,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=ymax,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=rgba_from_color(color, alpha),
            showlegend=False,
            hoverinfo="skip",
            yaxis=yaxis,
        )
    )


def add_lines(
    fig,
    x,
    y_mean,
    y_ma,
    color,
    name,
    show_ma: bool,
    yaxis="y",
    base_width=2.0,
    base_dash=None,
    ma_width=2.6,
    ma_dash="dot",
    mean_alpha=0.35,
):
    """平均線＋（任意で）移動平均線。見分け用に線種/太さを指定可能。"""
    if show_ma:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_mean,
                mode="lines",
                line=dict(color=rgba_from_color(color, mean_alpha), width=1.2, dash=base_dash),
                name=name,
                showlegend=True,
                yaxis=yaxis,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_ma,
                mode="lines",
                line=dict(color=color, width=ma_width, dash=ma_dash),
                showlegend=False,
                hoverinfo="skip",
                yaxis=yaxis,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_mean,
                mode="lines",
                line=dict(color=color, width=base_width, dash=base_dash),
                name=name,
                showlegend=True,
                yaxis=yaxis,
            )
        )


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
# データ読み込み
# =====================================================

@st.cache_data(show_spinner="データ読み込み中...", ttl=600)
def load_raw(csv_path: Path, _hash_val: str):
    df = pd.read_csv(str(csv_path), encoding=ENCODING)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).copy()

    # 水温（既存）
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

csv_bytes = p.read_bytes()
file_hash = f"{hashlib.sha1(csv_bytes).hexdigest()}_{len(csv_bytes)}"
df_raw = load_raw(p, file_hash)

years = sorted(df_raw["Year"].dropna().unique().tolist())
CURRENT_YEAR = max(years)

# =====================================================
# UI：表示モード
# =====================================================
mode = pill_toggle(["要約", "グラフ"], default="要約", key="mode")

# =====================================================
# 要約表示（既存のまま：水温のみ）
# =====================================================
if mode == "要約":
    selected_month = st.selectbox(
        "月",
        options=list(range(1, 13)),
        index=pd.Timestamp.today().month - 1,
    )

    summary = build_month_dekad_by_year(df_raw, selected_month, years)

    for dk in ["上旬", "中旬", "下旬"]:
        st.markdown(
            f"<div style='font-weight:600; margin-top:12px; border-bottom:1px solid #eee;'>{dk}</div>",
            unsafe_allow_html=True,
        )
        for y, info in summary[dk].items():
            if info is None:
                st.markdown(
                    f"<div style='margin-left:1em; color:#999;'>{y}年：データなし</div>",
                    unsafe_allow_html=True,
                )
            else:
                is_hot = (info["mean"] >= 20.0) or (info["max"] >= 20.0)
                color_main = HOT_RED if is_hot else "#000000"
                color_range = HOT_RED if is_hot else "#666666"
                style = (
                    "font-weight:bold; background-color:#f0f8ff; padding:2px 4px; border-radius:4px;"
                    if y == CURRENT_YEAR
                    else ""
                )
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
                    unsafe_allow_html=True,
                )

    st.markdown(
        """
        <div style="border-left:3px solid #ccc; margin-top:20px; padding-left:8px; color:#666; font-size:0.85em;">
          ※ 天候・時化・魚の状態メモ（将来追加）
        </div>
        """,
        unsafe_allow_html=True,
    )

# =====================================================
# グラフ表示（副軸：Sal/DOを追加。副軸は生データ＝点、MA＝破線）
# =====================================================
else:
    # 既存の3列UIを壊さず、副軸だけ1列追加
    c0, c1, c2, c3 = st.columns([1.0, 1.1, 1.1, 3.0])

    with c0:
        sec_label = pill_toggle(["なし", "Sal", "DO"], default="なし", key="sec_label", label="副軸")

    with c1:
        agg_label = pill_toggle(["日時", "日平均"], default="日時", key="agg_label", label="集計")

    with c2:
        smooth_label = pill_toggle(["なし", "移動平均(7日)"], default="なし", key="smooth_label", label="平滑化")

    with c3:
        selected_years = st.multiselect(
            "年",
            years,
            default=years[-2:] if len(years) >= 2 else years,
        )

    if not selected_years:
        st.stop()

    agg_mode = "datetime" if agg_label == "日時" else "daily"
    show_ma = (smooth_label != "なし")

    # 水温用カラー（既存）
    colors = year_color_map(selected_years)
    # 副軸用カラー（別パレットで混同を回避）
    sec_palette = px.colors.qualitative.Dark24
    colors_sec = {y: sec_palette[i % len(sec_palette)] for i, y in enumerate(sorted(selected_years))}

    # 副軸メトリクス（CSV列名）
    sec_metric = {"Sal": "Sal(1m)", "DO": "DO(3m)"}.get(sec_label)
    if sec_metric and sec_metric not in df_raw.columns:
        st.warning(f"副軸の列が見つかりません: {sec_metric}（副軸はOFFにしました）")
        sec_metric = None
        sec_label = "なし"

    # ---- 水温（既存ロジック） ----
    def build_timeseries_stats(df, metric=METRIC):
        d = df[["Year", DATE_COL, metric]].dropna().copy()
        d["X"] = d[DATE_COL].dt.floor("D") if agg_mode == "daily" else d[DATE_COL]
        return d.groupby(["Year", "X"])[metric].agg(["mean", "min", "max"]).reset_index()

    def build_same_monthday_stats(df, metric=METRIC):
        d = df[["Year", DATE_COL, "Month", "Day", metric]].dropna().copy()
        d = d[~((d["Month"] == 2) & (d["Day"] == 29))]

        if agg_mode == "daily":
            d["AlignX"] = pd.to_datetime(dict(year=2001, month=d["Month"], day=d["Day"]))
        else:
            t = d[DATE_COL].dt
            d["AlignX"] = pd.to_datetime(
                dict(
                    year=2001,
                    month=d["Month"],
                    day=d["Day"],
                    hour=t.hour,
                    minute=t.minute,
                    second=t.second,
                )
            )

        return d.groupby(["Year", "AlignX"])[metric].agg(["mean", "min", "max"]).reset_index()

    # ---- 副軸：生データ（点用） ----
    def build_raw_timeseries(df, metric):
        d = df[["Year", DATE_COL, metric]].dropna().copy()
        d["X"] = d[DATE_COL].dt.floor("D") if agg_mode == "daily" else d[DATE_COL]
        return d[["Year", "X", metric]]

    def build_raw_same_monthday(df, metric):
        d = df[["Year", DATE_COL, "Month", "Day", metric]].dropna().copy()
        d = d[~((d["Month"] == 2) & (d["Day"] == 29))]

        if agg_mode == "daily":
            d["AlignX"] = pd.to_datetime(dict(year=2001, month=d["Month"], day=d["Day"]))
        else:
            t = d[DATE_COL].dt
            d["AlignX"] = pd.to_datetime(
                dict(
                    year=2001,
                    month=d["Month"],
                    day=d["Day"],
                    hour=t.hour,
                    minute=t.minute,
                    second=t.second,
                )
            )

        return d[["Year", "AlignX", metric]]

    ts_stats = build_timeseries_stats(df_raw, METRIC)
    md_stats = build_same_monthday_stats(df_raw, METRIC)

    if sec_metric:
        ts_sec = build_timeseries_stats(df_raw, sec_metric)      # min/max + mean
        md_sec = build_same_monthday_stats(df_raw, sec_metric)
        ts_sec_raw = build_raw_timeseries(df_raw, sec_metric)    # 生データ
        md_sec_raw = build_raw_same_monthday(df_raw, sec_metric)
    else:
        ts_sec = md_sec = ts_sec_raw = md_sec_raw = None

    tab_ts, tab_md = st.tabs(["時系列", "同月日比較"])

    # =====================
    # 時系列
    # =====================
    with tab_ts:
        fig = go.Figure()

        # --- 水温（第一Y軸）：既存と同じ ---
        for y in selected_years:
            d = ts_stats[ts_stats["Year"] == y]
            if d.empty:
                continue
            add_band(fig, d["X"], d["min"], d["max"], colors[y], alpha=0.25, yaxis="y")
            ma = rolling_ma(d["mean"], d["X"], agg_mode) if show_ma else None
            add_lines(
                fig,
                d["X"],
                d["mean"],
                ma,
                colors[y],
                f"{y} 水温",
                show_ma,
                yaxis="y",
                base_width=2.0,
                base_dash=None,
                ma_width=2.6,
                ma_dash="dot",  # 水温MAはdot固定
            )

        # --- 副軸（第二Y軸）：生データ=点、移動平均=破線（dash）。dotは使わない ---
        if ts_sec is not None:
            for y in selected_years:
                d2 = ts_sec[ts_sec["Year"] == y]
                if d2.empty:
                    continue

                # バンドは薄く（任意）
                add_band(fig, d2["X"], d2["min"], d2["max"], colors_sec[y], alpha=0.08, yaxis="y2")

                # 生データ点
                r2 = ts_sec_raw[ts_sec_raw["Year"] == y]
                if not r2.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=r2["X"],
                            y=r2[sec_metric],
                            mode="markers",
                            marker=dict(size=5, color=colors_sec[y], opacity=0.35),
                            name=f"{y} {sec_label}（生）",
                            yaxis="y2",
                        )
                    )

                # 移動平均（破線）
                if show_ma:
                    ma2 = rolling_ma(d2["mean"], d2["X"], agg_mode)
                    fig.add_trace(
                        go.Scatter(
                            x=d2["X"],
                            y=ma2,
                            mode="lines",
                            line=dict(color=colors_sec[y], width=2.2, dash="dash"),
                            name=f"{y} {sec_label}（移動平均）",
                            yaxis="y2",
                        )
                    )

        layout = dict(template="plotly_white", height=520, hovermode="x unified")
        if ts_sec is not None:
            layout.update(
                dict(
                    yaxis=dict(title="水温 (℃)"),
                    yaxis2=dict(
                        title="Sal" if sec_label == "Sal" else "DO",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                    ),
                )
            )
        fig.update_layout(**layout)

        st.plotly_chart(fig, use_container_width=True)

    # =====================
    # 同月日比較
    # =====================
    with tab_md:
        fig = go.Figure()

        # --- 水温（第一Y軸）：既存と同じ ---
        for y in selected_years:
            d = md_stats[md_stats["Year"] == y]
            if d.empty:
                continue
            add_band(fig, d["AlignX"], d["min"], d["max"], colors[y], alpha=0.25, yaxis="y")
            ma = rolling_ma(d["mean"], d["AlignX"], agg_mode) if show_ma else None
            add_lines(
                fig,
                d["AlignX"],
                d["mean"],
                ma,
                colors[y],
                f"{y} 水温",
                show_ma,
                yaxis="y",
                base_width=2.0,
                base_dash=None,
                ma_width=2.6,
                ma_dash="dot",
            )

        # --- 副軸（第二Y軸）：生データ=点、移動平均=破線 ---
        if md_sec is not None:
            for y in selected_years:
                d2 = md_sec[md_sec["Year"] == y]
                if d2.empty:
                    continue

                add_band(fig, d2["AlignX"], d2["min"], d2["max"], colors_sec[y], alpha=0.08, yaxis="y2")

                r2 = md_sec_raw[md_sec_raw["Year"] == y]
                if not r2.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=r2["AlignX"],
                            y=r2[sec_metric],
                            mode="markers",
                            marker=dict(size=2, color=colors_sec[y], opacity=0.35),
                            name=f"{y} {sec_label}（生）",
                            yaxis="y2",
                        )
                    )

                if show_ma:
                    ma2 = rolling_ma(d2["mean"], d2["AlignX"], agg_mode)
                    fig.add_trace(
                        go.Scatter(
                            x=d2["AlignX"],
                            y=ma2,
                            mode="lines",
                            line=dict(color=colors_sec[y], width=2.2, dash="dash"),
                            name=f"{y} {sec_label}（移動平均）",
                            yaxis="y2",
                        )
                    )

        layout = dict(template="plotly_white", height=520, hovermode="x unified")
        if md_sec is not None:
            layout.update(
                dict(
                    yaxis=dict(title="水温 (℃)"),
                    yaxis2=dict(
                        title="Sal" if sec_label == "Sal" else "DO",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                    ),
                )
            )
        fig.update_layout(**layout)
        fig.update_xaxes(tickformat="%m/%d")

        st.plotly_chart(fig, use_container_width=True)
