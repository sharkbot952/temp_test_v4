import os
import hashlib
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import xarray as xr

import plotly.graph_objects as go
import plotly.express as px


# =====================================================
# ページ設定
# =====================================================
st.set_page_config(page_title="試験版", layout="wide")

# =====================================================
# 固定設定（パス）
# =====================================================
APP_DIR = Path(__file__).resolve().parent

# 通常は repo直下の ./data を使う想定
DEFAULT_BASE_DIR = APP_DIR / "data"

# ユーザー指定のクローンフォルダ（フォールバック）
FALLBACK_BASE_DIR = Path(r"C:\Users\dgr5505\Documents\GitHub\temp_test_v4\data")

BASE_DIR = DEFAULT_BASE_DIR if DEFAULT_BASE_DIR.exists() else FALLBACK_BASE_DIR

# 水温CSV
CSV_PATH = BASE_DIR / "Taiki_temp.csv"
ENCODING = "utf-8-sig"
DATE_COL = "DATE"
METRIC = "depth_avg"  # 水温（1–3mの平均）
HOT_RED = "#d32f2f"

# 波浪NetCDF（同じ data に置いている前提）
WAV_MY_PATH = BASE_DIR / "wav_20200101_20221031_asahihama_BOXWIDE.nc"
WAV_ANFC_PATH = BASE_DIR / "wav_202211_20260321_asahihama_BOXWIDE.nc"

# 既定の抽出ポイント（決め打ち）
POINT_MY_DEFAULT = (42.0, 143.1999969482422)         # (lat, lon)
POINT_ANFC_DEFAULT = (42.16666666666666, 143.41666666666663)

# 既定の期間
RANGE_MY_DEFAULT = ("2020-01-01", "2022-10-31")
RANGE_ANFC_DEFAULT = ("2022-11-01", "2026-03-21")

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
# 共通ユーティリティ（水温側：既存）
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
# 要約モード用（水温側：既存）
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
# データ読み込み（水温）
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

# 物理ファイルチェック（水温CSV）
p = Path(CSV_PATH)
if not p.exists():
    st.error(f"CSV が見つかりません: {CSV_PATH}（{p.resolve()}）")
    st.stop()

csv_bytes = p.read_bytes()
file_hash = f"{hashlib.sha1(csv_bytes).hexdigest()}_{len(csv_bytes)}"
df_raw = load_raw(p, file_hash)

years = sorted(df_raw["Year"].dropna().unique().tolist())
CURRENT_YEAR = max(years) if years else None


# =====================================================
# 波浪：計算ユーティリティ
# =====================================================
def pick_coord(ds, candidates):
    for c in candidates:
        if c in ds.coords or c in ds.dims:
            return c
    raise KeyError(f"Coordinate not found: {candidates} / coords={list(ds.coords)} dims={list(ds.dims)}")

def pick_var(dsobj, base_names):
    # exact
    for b in base_names:
        if b in dsobj.data_vars:
            return b
    # prefix（例：VHM0_SW1）
    for v in dsobj.data_vars:
        if any(v.startswith(b) for b in base_names):
            return v
    raise KeyError(f"Variable not found: {base_names} / available={list(dsobj.data_vars)}")

def qscale01(s, qlow=0.05, qhigh=0.95):
    s = s.astype(float)
    lo = np.nanquantile(s, qlow)
    hi = np.nanquantile(s, qhigh)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        mn, mx = np.nanmin(s), np.nanmax(s)
        return (s - mn) / (mx - mn + 1e-12)
    x = (s - lo) / (hi - lo)
    return np.clip(x, 0, 1)

def ang_diff(a, b):
    return (a - b + 180) % 360 - 180

def choose_normal(axis_deg, offshore_bearing_deg):
    n1 = (axis_deg + 90) % 360
    n2 = (axis_deg - 90) % 360
    d1 = abs(ang_diff(n1, offshore_bearing_deg))
    d2 = abs(ang_diff(n2, offshore_bearing_deg))
    return n1 if d1 <= d2 else n2

def dir_weight_gauss(dir_deg, target_deg, sigma_deg):
    delta = ang_diff(dir_deg, target_deg)
    return np.exp(-0.5 * (delta / sigma_deg) ** 2)

def dir_weight_multi(dir_deg_array, targets, mode="max"):
    comps = []
    for t, s, w in targets:
        comps.append(w * dir_weight_gauss(dir_deg_array, t, s))
    M = np.vstack(comps)
    if mode == "max":
        return np.max(M, axis=0)
    elif mode == "sum":
        return np.clip(np.sum(M, axis=0), 0, 1)
    else:
        raise ValueError("DIR_MODE must be 'max' or 'sum'")

def build_dir_targets(params):
    entrance_target = choose_normal(params["ENTRANCE_AXIS_DEG"], params["OFFSHORE_BEARING_DEG"])
    outer_target    = choose_normal(params["OUTER_BW_AXIS_DEG"], params["OFFSHORE_BEARING_DEG"])
    inner_target    = choose_normal(params["INNER_BW_AXIS_DEG"], params["OFFSHORE_BEARING_DEG"])
    targets = [
        (entrance_target, params["SIG_ENTRANCE"], params["W_ENTRANCE"]),
        (outer_target,    params["SIG_OUTER"],    params["W_OUTER"]),
        (inner_target,    params["SIG_INNER"],    params["W_INNER"]),
    ]
    return targets

@st.cache_data(show_spinner=False, ttl=600)
def load_wave_daily(fn: str, lat0: float, lon0: float, start_date: str, end_date: str,
                    use_jst: bool, drop_leapday: bool,
                    q_low: float, q_high: float, smooth_days: int,
                    allow_180_flip: bool, dir_mode: str, params: dict):
    """
    NetCDFから daily DataFrame を作る（列：Hmax, Tp_mean, Dir_mean, score, score_map, H_idx, T_idx, D_idx）
    """
    ds = xr.open_dataset(fn)

    lat_name = pick_coord(ds, ["lat", "latitude", "LATITUDE", "nav_lat"])
    lon_name = pick_coord(ds, ["lon", "longitude", "LONGITUDE", "nav_lon"])

    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    if use_jst:
        start_utc = start - pd.Timedelta(hours=9)
        end_utc   = end   - pd.Timedelta(hours=9)
    else:
        start_utc, end_utc = start, end

    ds = ds.sel(time=slice(start_utc, end_utc))
    pt = ds.sel({lat_name: lat0, lon_name: lon0}, method="nearest")

    v_h = pick_var(pt, ["VHM0"])
    v_t = pick_var(pt, ["VTPK"])
    v_d = pick_var(pt, ["VMDR"])

    df = pt[[v_h, v_t, v_d]].to_dataframe().reset_index()
    tcol = "time" if "time" in df.columns else [c for c in df.columns if "time" in c.lower()][0]
    df[tcol] = pd.to_datetime(df[tcol])

    if use_jst:
        df["time_jst"] = df[tcol] + pd.Timedelta(hours=9)
        df = df.set_index("time_jst")
    else:
        df = df.set_index(tcol)

    df = df.replace([np.inf, -np.inf], np.nan)

    # daily化（波高=日最大、周期/方向=日平均）
    daily = pd.DataFrame({
        "Hmax": df[v_h].resample("D").max(),
        "Tp_mean": df[v_t].resample("D").mean(),
        "Dir_mean": df[v_d].resample("D").mean(),
    }).dropna()

    daily["md"] = daily.index.strftime("%m-%d")
    if drop_leapday:
        daily = daily[daily["md"] != "02-29"]
    daily["year"] = daily.index.year

    # スコア化
    daily["H_idx"] = qscale01(daily["Hmax"], q_low, q_high)
    daily["T_idx"] = qscale01(daily["Tp_mean"], q_low, q_high)

    targets = build_dir_targets(params)
    dir0 = daily["Dir_mean"].values
    D1 = dir_weight_multi(dir0, targets, mode=dir_mode)

    # VMDR 180反転も保険で評価
    if allow_180_flip:
        D2 = dir_weight_multi((dir0 + 180) % 360, targets, mode=dir_mode)
        daily["D_idx"] = np.maximum(D1, D2)
    else:
        daily["D_idx"] = D1

    daily["score"] = daily["H_idx"] * daily["T_idx"] * daily["D_idx"]

    if smooth_days and smooth_days > 1:
        daily["score_map"] = daily["score"].rolling(smooth_days, center=True, min_periods=1).mean()
    else:
        daily["score_map"] = daily["score"]

    lat_sel = float(pt[lat_name].values)
    lon_sel = float(pt[lon_name].values)

    meta = {
        "selected_point": (lat_sel, lon_sel),
        "vars": (v_h, v_t, v_d),
    }
    return daily, meta

def classify_alerts(daily: pd.DataFrame,
                    thr_score=0.5,
                    warn_days=4,
                    watch_days=2,
                    lookback_days=60):
    """
    daily（index=日付）から警報コメントを返す
    """
    if daily is None or daily.empty or "score" not in daily.columns:
        return "NO_DATA", "データがありません。", {}

    d = daily.sort_index().copy()
    d = d.iloc[-lookback_days:] if len(d) > lookback_days else d

    flag = (d["score"] >= thr_score).astype(int)
    grp = (flag.diff() != 0).cumsum()

    # 直近側の「score>=thr」の連続区間を取得
    active = None
    for _, g in d[flag == 1].groupby(grp[flag == 1]):
        active = g

    if active is None:
        mx = float(d["score"].max())
        dt = d["score"].idxmax()
        return "OK", f"現時点で注意喚起レベル（score≥{thr_score:.2f}）はなし（直近最大 score={mx:.2f} / {dt.date()}）", {"max_score": mx}

    start = active.index.min()
    end   = active.index.max()
    dur = (end - start).days + 1

    peak_idx = active["score"].idxmax()
    peak_score = float(active.loc[peak_idx, "score"])
    hmax = float(active["Hmax"].max()) if "Hmax" in active.columns else np.nan
    tpmax = float(active["Tp_mean"].max()) if "Tp_mean" in active.columns else np.nan

    if dur >= warn_days:
        status = "ALERT"
        head = "警報"
    elif dur >= watch_days:
        status = "WATCH"
        head = "注意報"
    else:
        status = "WATCH"
        head = "注意報（単発）"

    remain = warn_days - dur
    tail = ""
    if status == "WATCH" and remain > 0:
        tail = f"（あと{remain}日継続で警報）"

    msg = (f"{head}：score≥{thr_score:.2f} が {dur}日連続 "
           f"({start.date()}–{end.date()})  "
           f"最大score={peak_score:.2f}, Hmax={hmax:.1f}m, Tp={tpmax:.1f}s {tail}")

    detail = {
        "status": status,
        "start": start.date(),
        "end": end.date(),
        "duration_days": int(dur),
        "peak_date": peak_idx.date(),
        "peak_score": peak_score,
        "Hmax_max": hmax,
        "Tp_max": tpmax,
        "thr_score": thr_score,
        "warn_days": warn_days,
        "watch_days": watch_days,
    }
    return status, msg, detail

def plot_wave_recent(daily: pd.DataFrame, n=45):
    if daily is None or daily.empty:
        return None
    d = daily.sort_index().tail(n).copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.index, y=d["score"],
        mode="lines+markers",
        name="score",
        line=dict(color="#1f77b4", width=2),
    ))
    if "Hmax" in d.columns:
        fig.add_trace(go.Scatter(
            x=d.index, y=d["Hmax"],
            mode="lines",
            name="Hmax (m)",
            yaxis="y2",
            line=dict(color="#ff7f0e", width=2, dash="dot"),
        ))
    fig.update_layout(
        template="plotly_white",
        height=320,
        hovermode="x unified",
        yaxis=dict(title="score (0-1)", range=[0, 1]),
        yaxis2=dict(title="Hmax (m)", overlaying="y", side="right", showgrid=False),
        margin=dict(l=40, r=40, t=30, b=30),
    )
    return fig


# =====================================================
# UI：表示モード
# =====================================================
mode = pill_toggle(["要約", "グラフ", "波浪警報"], default="要約", key="mode")


# =====================================================
# 要約表示（水温のみ：既存）
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
                    if (CURRENT_YEAR is not None and y == CURRENT_YEAR)
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
# 波浪警報（コメント中心）
# =====================================================
elif mode == "波浪警報":
    st.subheader("旭浜：うねり影響の注意喚起（コメント中心）")

    # ファイルの存在チェック（事前）
    if not WAV_MY_PATH.exists():
        st.warning(f"MY NetCDF が見つかりません: {WAV_MY_PATH}")
    if not WAV_ANFC_PATH.exists():
        st.warning(f"ANFC NetCDF が見つかりません: {WAV_ANFC_PATH}")

    with st.sidebar:
        st.markdown("### 波浪データ")
        source = pill_toggle(["ANFC（解析・予測）", "MY（再解析）"], default="ANFC（解析・予測）", key="wav_source")

        fn_default = str(WAV_ANFC_PATH) if source.startswith("ANFC") else str(WAV_MY_PATH)
        fn = st.text_input("NetCDFパス", value=fn_default)

        st.markdown("### 抽出ポイント（nearest）")
        if source.startswith("ANFC"):
            lat0 = st.number_input("lat", value=float(POINT_ANFC_DEFAULT[0]), format="%.6f")
            lon0 = st.number_input("lon", value=float(POINT_ANFC_DEFAULT[1]), format="%.6f")
            d0, d1 = RANGE_ANFC_DEFAULT
        else:
            lat0 = st.number_input("lat", value=float(POINT_MY_DEFAULT[0]), format="%.6f")
            lon0 = st.number_input("lon", value=float(POINT_MY_DEFAULT[1]), format="%.6f")
            d0, d1 = RANGE_MY_DEFAULT

        st.markdown("### 期間（JST日付）")
        use_jst = st.checkbox("JSTとして扱う（UTC+9）", value=True)
        drop_leap = st.checkbox("2/29を除外", value=True)

        cA, cB = st.columns(2)
        with cA:
            start_date = st.date_input("開始日", value=pd.to_datetime(d0))
        with cB:
            end_date = st.date_input("終了日", value=pd.to_datetime(d1))

        st.markdown("### 警報ロジック（コメント）")
        thr_score = st.slider("score閾値", 0.0, 1.0, 0.50, 0.05)
        warn_days = st.number_input("警報：連続日数", min_value=2, max_value=14, value=4, step=1)
        watch_days = st.number_input("注意報：連続日数", min_value=1, max_value=13, value=2, step=1)
        lookback_days = st.number_input("判定に使う直近日数", min_value=14, max_value=365, value=60, step=1)

        st.markdown("### スコア計算（合成）")
        q_low = st.slider("分位下限（正規化）", 0.00, 0.20, 0.05, 0.01)
        q_high = st.slider("分位上限（正規化）", 0.80, 1.00, 0.95, 0.01)
        smooth_days = st.slider("平滑（日）", 1, 7, 3)
        allow_flip = st.checkbox("方向180°反転も評価（保険）", value=True)
        dir_mode = st.selectbox("方向合成", ["max", "sum"], index=0)

        st.markdown("### 旭浜の方向帯（あなたの整理）")
        ENTRANCE_AXIS_DEG = st.number_input("港口軸（deg）", value=20.0, format="%.1f")
        OUTER_BW_AXIS_DEG = st.number_input("沖側防波堤軸（deg）", value=45.0, format="%.1f")
        INNER_BW_AXIS_DEG = st.number_input("沿岸側防波堤軸（deg）", value=135.0, format="%.1f")
        OFFSHORE_BEARING_DEG = st.number_input("外洋側方位（deg）", value=90.0, format="%.1f")

        SIG_ENTRANCE = st.number_input("SIG 港口", value=22.0, format="%.1f")
        W_ENTRANCE = st.number_input("W 港口", value=1.00, format="%.2f")
        SIG_OUTER = st.number_input("SIG 沖防", value=30.0, format="%.1f")
        W_OUTER = st.number_input("W 沖防", value=0.75, format="%.2f")
        SIG_INNER = st.number_input("SIG 沿防", value=35.0, format="%.1f")
        W_INNER = st.number_input("W 沿防", value=0.55, format="%.2f")

        run = st.button("更新（計算）", type="primary")

    if run:
        if not fn or not Path(fn).exists():
            st.error(f"NetCDFが見つかりません: {fn}")
            st.stop()

        params = dict(
            ENTRANCE_AXIS_DEG=float(ENTRANCE_AXIS_DEG),
            OUTER_BW_AXIS_DEG=float(OUTER_BW_AXIS_DEG),
            INNER_BW_AXIS_DEG=float(INNER_BW_AXIS_DEG),
            OFFSHORE_BEARING_DEG=float(OFFSHORE_BEARING_DEG),
            SIG_ENTRANCE=float(SIG_ENTRANCE),
            W_ENTRANCE=float(W_ENTRANCE),
            SIG_OUTER=float(SIG_OUTER),
            W_OUTER=float(W_OUTER),
            SIG_INNER=float(SIG_INNER),
            W_INNER=float(W_INNER),
        )

        with st.spinner("NetCDF読み込み→日別→スコア化中..."):
            daily, meta = load_wave_daily(
                fn=str(fn),
                lat0=float(lat0),
                lon0=float(lon0),
                start_date=str(start_date),
                end_date=str(end_date),
                use_jst=bool(use_jst),
                drop_leapday=bool(drop_leap),
                q_low=float(q_low),
                q_high=float(q_high),
                smooth_days=int(smooth_days),
                allow_180_flip=bool(allow_flip),
                dir_mode=str(dir_mode),
                params=params,
            )

        pt = meta["selected_point"]
        st.caption(f"selected point=({pt[0]:.4f},{pt[1]:.4f})  vars={meta['vars']}")

        status, msg, detail = classify_alerts(
            daily,
            thr_score=float(thr_score),
            warn_days=int(warn_days),
            watch_days=int(watch_days),
            lookback_days=int(lookback_days),
        )

        # コメント表示（運用向け）
        if status == "ALERT":
            st.error(msg)
        elif status == "WATCH":
            st.warning(msg)
        elif status == "OK":
            st.success(msg)
        else:
            st.info(msg)

        # 直近の推移を軽く可視化（ヒートマップより運用向け）
        fig = plot_wave_recent(daily, n=45)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("直近データ（参考）")
        show_cols = [c for c in ["score", "Hmax", "Tp_mean", "Dir_mean", "H_idx", "T_idx", "D_idx"] if c in daily.columns]
        st.dataframe(daily.sort_index().tail(14)[show_cols], use_container_width=True)

        st.download_button(
            "daily（CSV）をダウンロード",
            data=daily.to_csv(index=True).encode("utf-8"),
            file_name="wave_daily.csv",
            mime="text/csv",
        )


# =====================================================
# グラフ表示（水温：既存）
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

    # ---- 副軸（Sal/DO）用：近い時間をまとめて帯（min–max）を作る ----
    SEC_BIN_HOURS = 3

    def build_timeseries_stats_sec(df, metric, bin_hours=SEC_BIN_HOURS):
        d = df[["Year", DATE_COL, metric]].dropna().copy()
        if agg_mode == "daily":
            d["X"] = d[DATE_COL].dt.floor("D")
        else:
            d["X"] = d[DATE_COL].dt.floor(f"{bin_hours}H")
        return d.groupby(["Year", "X"])[metric].agg(["mean", "min", "max"]).reset_index()

    def build_same_monthday_stats_sec(df, metric, bin_hours=SEC_BIN_HOURS):
        d = df[["Year", DATE_COL, "Month", "Day", metric]].dropna().copy()
        d = d[~((d["Month"] == 2) & (d["Day"] == 29))]
        if agg_mode == "daily":
            d["AlignX"] = pd.to_datetime(dict(year=2001, month=d["Month"], day=d["Day"]))
        else:
            t = d[DATE_COL].dt
            h = (t.hour // bin_hours) * bin_hours
            d["AlignX"] = pd.to_datetime(dict(year=2001, month=d["Month"], day=d["Day"], hour=h, minute=0, second=0))
        return d.groupby(["Year", "AlignX"])[metric].agg(["mean", "min", "max"]).reset_index()

    ts_stats = build_timeseries_stats(df_raw, METRIC)
    md_stats = build_same_monthday_stats(df_raw, METRIC)

    if sec_metric:
        ts_sec = build_timeseries_stats_sec(df_raw, sec_metric)
        md_sec = build_same_monthday_stats_sec(df_raw, sec_metric)
    else:
        ts_sec = md_sec = None

    tab_ts, tab_md = st.tabs(["時系列", "同月日比較"])

    # =====================
    # 時系列
    # =====================
    with tab_ts:
        fig = go.Figure()

        # 水温（第一Y軸）
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
                ma_dash="dot",
            )

        # 副軸（第二Y軸）
        if ts_sec is not None:
            for y in selected_years:
                d2 = ts_sec[ts_sec["Year"] == y]
                if d2.empty:
                    continue
                if show_ma:
                    ma2 = rolling_ma(d2["mean"], d2["X"], agg_mode)
                    fig.add_trace(
                        go.Scatter(
                            x=d2["X"],
                            y=ma2,
                            mode="lines",
                            line=dict(color=colors_sec[y], width=2.4, dash="dash"),
                            name=f"{y} {sec_label}（移動平均）",
                            yaxis="y2",
                        )
                    )
                else:
                    add_band(fig, d2["X"], d2["min"], d2["max"], colors_sec[y], alpha=0.14, yaxis="y2")
                    add_lines(
                        fig,
                        d2["X"],
                        d2["mean"],
                        None,
                        colors_sec[y],
                        f"{y} {sec_label}",
                        False,
                        yaxis="y2",
                        base_width=1.8,
                        base_dash="dash",
                        mean_alpha=0.65,
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

        # 水温（第一Y軸）
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

        # 副軸（第二Y軸）
        if md_sec is not None:
            for y in selected_years:
                d2 = md_sec[md_sec["Year"] == y]
                if d2.empty:
                    continue
                if show_ma:
                    ma2 = rolling_ma(d2["mean"], d2["AlignX"], agg_mode)
                    fig.add_trace(
                        go.Scatter(
                            x=d2["AlignX"],
                            y=ma2,
                            mode="lines",
                            line=dict(color=colors_sec[y], width=2.4, dash="dash"),
                            name=f"{y} {sec_label}（移動平均）",
                            yaxis="y2",
                        )
                    )
                else:
                    add_band(fig, d2["AlignX"], d2["min"], d2["max"], colors_sec[y], alpha=0.14, yaxis="y2")
                    add_lines(
                        fig,
                        d2["AlignX"],
                        d2["mean"],
                        None,
                        colors_sec[y],
                        f"{y} {sec_label}",
                        False,
                        yaxis="y2",
                        base_width=1.8,
                        base_dash="dash",
                        mean_alpha=0.65,
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