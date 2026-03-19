import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# =====================================================
# ページ設定
# =====================================================
st.set_page_config(page_title="", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CSV_PATH = DATA_DIR / "Taiki_temp.csv"

# ★点切り抜き済みWAV（位置は確定済みなのでPOINTは不要）
FN_MY   = DATA_DIR / "cmem_wav_MY.nc"
FN_ANFC = DATA_DIR / "cmem_wav_ANFC.nc"

# =====================================================
# 水温側：固定設定
# =====================================================
ENCODING = "utf-8-sig"
DATE_COL = "DATE"
METRIC = "depth_avg"  # 水温（1–3m平均）
HOT_RED = "#d32f2f"

# =====================================================
# 波浪側：固定設定（スコア等はUIで変更させない）
# =====================================================
# ★点ncなので位置指定は不要（旧POINT_*は廃止）
POINT_MY = None
POINT_ANFC = None

USE_JST = True
DROP_LEAPDAY = True

# 旭浜：方向帯（固定）
ENTRANCE_AXIS_DEG = 20.0
OUTER_BW_AXIS_DEG = 45.0
INNER_BW_AXIS_DEG = 135.0
OFFSHORE_BEARING_DEG = 90.0
SIG_ENTRANCE, W_ENTRANCE = 22.0, 1.00
SIG_OUTER, W_OUTER = 30.0, 0.75
SIG_INNER, W_INNER = 35.0, 0.55
DIR_MODE = "max"
ALLOW_180_FLIP = True

# 合成スコア（固定）
Q_LOW, Q_HIGH = 0.05, 0.95
SMOOTH_DAYS = 3

# --- 追加（A+B）: スコア強調パラメータ ---
DIR_EXP_Q = 0.60        # A: 方向の指数（<1 で頭打ち緩和、0.5〜0.8推奨）
EXCESS_AH = 0.80        # B: 波高の超過分ブースト係数
EXCESS_AT = 0.50        # B: 周期の超過分ブースト係数
EXCESS_P  = 1.20        # B: 超過分の非線形（1.0〜1.5推奨）


# コメント警報（固定）
ALERT_SCORE = 0.4
ALERT_DAYS = 4
WATCH_DAYS = 2
LOOKBACK_DAYS = 60

# 表示（固定）
RECENT_DAYS_TABLE = 10  # 表は最新10日

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
# 水温：共通ユーティリティ
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

def add_line(fig, x, y, color, name, yaxis="y", width=2.2, dash=None, alpha=1.0):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(color=rgba_from_color(color, alpha) if alpha < 1.0 else color, width=width, dash=dash),
            name=name,
            showlegend=True,
            yaxis=yaxis,
        )
    )

# =====================================================
# 水温：要約ユーティリティ
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
# 水温：データ読み込み
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

# =====================================================
# 波浪：共通関数
# =====================================================
def pick_coord(ds, candidates):
    for c in candidates:
        if c in ds.coords or c in ds.dims:
            return c
    raise KeyError(f"Coordinate not found: {candidates} / coords={list(ds.coords)} dims={list(ds.dims)}")

def pick_var(dsobj, base_names):
    for b in base_names:
        if b in dsobj.data_vars:
            return b
    for v in dsobj.data_vars:
        if any(v.startswith(b) for b in base_names):
            return v
    raise KeyError(f"Variable not found: {base_names} / available={list(dsobj.data_vars)}")

def scale01_lohi(s, lo, hi):
    s = s.astype(float)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        mn, mx = np.nanmin(s), np.nanmax(s)
        return (s - mn) / (mx - mn + 1e-12)
    x = (s - lo) / (hi - lo)
    return np.clip(x, 0, 1)

def calc_lohi(s, qlow=0.05, qhigh=0.95):
    s = s.astype(float)
    lo = np.nanquantile(s, qlow)
    hi = np.nanquantile(s, qhigh)
    return float(lo), float(hi)

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
    if mode == "sum":
        return np.clip(np.sum(M, axis=0), 0, 1)
    raise ValueError("DIR_MODE must be 'max' or 'sum'")

def build_dir_targets():
    entrance_target = choose_normal(ENTRANCE_AXIS_DEG, OFFSHORE_BEARING_DEG)
    outer_target = choose_normal(OUTER_BW_AXIS_DEG, OFFSHORE_BEARING_DEG)
    inner_target = choose_normal(INNER_BW_AXIS_DEG, OFFSHORE_BEARING_DEG)
    return [
        (entrance_target, SIG_ENTRANCE, W_ENTRANCE),
        (outer_target, SIG_OUTER, W_OUTER),
        (inner_target, SIG_INNER, W_INNER),
    ]

@st.cache_data(show_spinner=False, ttl=600)
def load_wave_daily(fn: str, point_latlon, date_range, ref_quantiles=None):
    """NetCDF -> 日別（Hmax, Tp_mean, Dir_mean, score）
       ※点切り抜き済みncなら point_latlon=None でOK（位置指定を不要化）
    """
    start_date, end_date = date_range

    ds = xr.open_dataset(fn)

    time_name = pick_coord(ds, ["time", "time_counter", "TIME", "valid_time"])
    lat_name  = pick_coord(ds, ["lat", "latitude", "LATITUDE", "nav_lat"])
    lon_name  = pick_coord(ds, ["lon", "longitude", "LONGITUDE", "nav_lon"])

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    if USE_JST:
        start_utc = start - pd.Timedelta(hours=9)
        end_utc = end - pd.Timedelta(hours=9)
    else:
        start_utc, end_utc = start, end

    ds = ds.sel({time_name: slice(start_utc, end_utc)})

    # ---- ここが修正点：点切り抜き済みncなら位置指定を不要化 ----
    if point_latlon is None:
        # lat/lon が単一点ならそのまま
        if (lat_name in ds.coords and ds[lat_name].size == 1) and (lon_name in ds.coords and ds[lon_name].size == 1):
            pt = ds
        else:
            ds.close()
            raise ValueError("point_latlon is None but dataset has multiple lat/lon cells. Provide point_latlon.")
    else:
        lat0, lon0 = point_latlon
        pt = ds.sel({lat_name: lat0, lon_name: lon0}, method="nearest")

    v_h = pick_var(pt, ["VHM0"])
    v_t = pick_var(pt, ["VTPK"])
    v_d = pick_var(pt, ["VMDR"])

    df = pt[[v_h, v_t, v_d]].to_dataframe().reset_index()

    tcol = time_name if time_name in df.columns else (
        "time" if "time" in df.columns else [c for c in df.columns if "time" in c.lower()][0]
    )
    df[tcol] = pd.to_datetime(df[tcol])

    if USE_JST:
        df["time_jst"] = df[tcol] + pd.Timedelta(hours=9)
        df = df.set_index("time_jst")
    else:
        df = df.set_index(tcol)

    df = df.replace([np.inf, -np.inf], np.nan)

    daily = pd.DataFrame(
        {
            "Hmax": df[v_h].resample("D").max(),
            "Tp_mean": df[v_t].resample("D").mean(),
            "Dir_mean": df[v_d].resample("D").mean(),
        }
    ).dropna()

    ds.close()

    if daily.empty:
        return daily, {}

    if DROP_LEAPDAY:
        md = daily.index.strftime("%m-%d")
        daily = daily[md != "02-29"].copy()

    # 正規化基準
    if ref_quantiles is None:
        H_lo, H_hi = calc_lohi(daily["Hmax"], Q_LOW, Q_HIGH)
        T_lo, T_hi = calc_lohi(daily["Tp_mean"], Q_LOW, Q_HIGH)
    else:
        H_lo, H_hi, T_lo, T_hi = ref_quantiles

    daily["H_idx"] = scale01_lohi(daily["Hmax"], H_lo, H_hi)
    daily["T_idx"] = scale01_lohi(daily["Tp_mean"], T_lo, T_hi)

    targets = build_dir_targets()
    dir0 = daily["Dir_mean"].values
    D1 = dir_weight_multi(dir0, targets, mode=DIR_MODE)
    if ALLOW_180_FLIP:
        D2 = dir_weight_multi((dir0 + 180) % 360, targets, mode=DIR_MODE)
        daily["D_idx"] = np.maximum(D1, D2)
    else:
        daily["D_idx"] = D1


    # --- A) 方向は活かすが、頭打ちを緩める（単調性は維持） ---
    d_eff = np.clip(daily["D_idx"].astype(float), 0, 1) ** DIR_EXP_Q
    score_base = daily["H_idx"].astype(float) * daily["T_idx"].astype(float) * d_eff

    # --- B) p95超の"超過分"を危険度へ戻す（越波級を強調） ---
    denH = max(H_hi - H_lo, 1e-12)
    denT = max(T_hi - T_lo, 1e-12)
    H_excess = np.clip((daily["Hmax"].astype(float) - H_hi) / denH, 0, None)
    T_excess = np.clip((daily["Tp_mean"].astype(float) - T_hi) / denT, 0, None)
    boost = 1.0 + EXCESS_AH * (H_excess ** EXCESS_P) + EXCESS_AT * (T_excess ** EXCESS_P)

    daily["score"] = np.clip(score_base * boost, 0, 1)

    if SMOOTH_DAYS and SMOOTH_DAYS > 1:
        daily["score_map"] = daily["score"].rolling(SMOOTH_DAYS, center=True, min_periods=1).mean()
    else:
        daily["score_map"] = daily["score"]

    return daily, {}

def classify_alerts(daily: pd.DataFrame):
    """score>=ALERT_SCORE 連続ALERT_DAYS以上=警報、WATCH_DAYS以上=注意報"""
    if daily is None or daily.empty or "score" not in daily.columns:
        return "NO_DATA", "データがありません（期間/格子点/欠損の可能性）。", {}

    d = daily.sort_index().copy()
    d = d.iloc[-LOOKBACK_DAYS:] if len(d) > LOOKBACK_DAYS else d

    flag = (d["score"] >= ALERT_SCORE).astype(int)
    grp = (flag.diff() != 0).cumsum()

    active = None
    for _, g in d[flag == 1].groupby(grp[flag == 1]):
        active = g

    if active is None:
        mx = float(d["score"].max())
        dt = d["score"].idxmax()
        return "OK", f"現時点で注意喚起なし（直近最大 score={mx:.2f} / {dt.date()}）", {"max_score": mx}

    start = active.index.min()
    end = active.index.max()
    dur = (end - start).days + 1
    peak_idx = active["score"].idxmax()
    peak_score = float(active.loc[peak_idx, "score"])
    hmax = float(active["Hmax"].max())
    tpmax = float(active["Tp_mean"].max())

    if dur >= ALERT_DAYS:
        status = "ALERT"
        head = "警報"
    elif dur >= WATCH_DAYS:
        status = "WATCH"
        head = "注意報"
    else:
        status = "WATCH"
        head = "注意報（単発）"

    remain = ALERT_DAYS - dur
    tail = f"（あと{remain}日継続で警報）" if status == "WATCH" and remain > 0 else ""

    msg = (
        f"{head}：score≥{ALERT_SCORE:.2f} が {dur}日連続 "
        f"({start.date()}–{end.date()}) "
        f"最大score={peak_score:.2f}, 波高={hmax:.1f}m, 周期={tpmax:.1f}s {tail}"
    )
    return status, msg, {"duration_days": int(dur), "peak_score": peak_score}


def plot_wave(daily: pd.DataFrame, title: str):
    d = daily.sort_index().copy()
    fig = go.Figure()
    if "kind" in d.columns:
        for k, g in d.groupby("kind"):
            fig.add_trace(go.Scatter(x=g.index, y=g["score"], mode="lines+markers",
                                     name=f"スコア({k})", line=dict(width=2)))
            fig.add_trace(go.Scatter(x=g.index, y=g["Hmax"], mode="lines",
                                     name=f"波高(m)({k})", yaxis="y2",
                                     line=dict(width=2, dash="dot")))
    else:
        fig.add_trace(go.Scatter(x=d.index, y=d["score"], mode="lines+markers",
                                 name="スコア", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=d.index, y=d["Hmax"], mode="lines",
                                 name="波高(m)", yaxis="y2",
                                 line=dict(width=2, dash="dot")))

    fig.update_layout(
        template="plotly_white",
        height=320,
        hovermode="x unified",
        title=title,
        yaxis=dict(title="スコア (0–1)", range=[0, 1]),
        yaxis2=dict(title="波高(m)", overlaying="y", side="right", showgrid=False),
        margin=dict(l=40, r=40, t=40, b=30),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
    )
    return fig
# =====================================================
# 起動：水温CSVの読み込み
# =====================================================
df_raw = None
years = []
CURRENT_YEAR = None

if CSV_PATH.exists():
    csv_bytes = CSV_PATH.read_bytes()
    file_hash = f"{hashlib.sha1(csv_bytes).hexdigest()}_{len(csv_bytes)}"
    df_raw = load_raw(CSV_PATH, file_hash)
    years = sorted(df_raw["Year"].dropna().unique().tolist())
    CURRENT_YEAR = max(years) if years else None

# =====================================================
# UI：モード
# =====================================================
mode = pill_toggle(["水温", "うねり", "グラフ"], default="水温", key="mode")

# =====================================================
# モード：水温要約
# =====================================================
if mode == "水温":
    if df_raw is None:
        st.error(f"CSV が見つかりません: {CSV_PATH}（repoの data/ に置いてください）")
        st.stop()

    selected_month = st.selectbox("月", options=list(range(1, 13)), index=pd.Timestamp.today().month - 1)
    summary = build_month_dekad_by_year(df_raw, selected_month, years)

    for dk in ["上旬", "中旬", "下旬"]:
        st.markdown(
            f"<div style='font-weight:600; margin-top:12px; border-bottom:1px solid #eee;'>{dk}</div>",
            unsafe_allow_html=True,
        )

        for y, info in summary[dk].items():
            if info is None:
                continue

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
                  <span style="font-size:1.1em; color:{color_main};">{info['mean']:.1f}℃</span>
                  <span style="color:{color_range}; font-size:0.9em;">（{info['min']:.1f}–{info['max']:.1f}）</span>
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
# モード：うねり
# =====================================================
elif mode == "うねり":
    @st.cache_data(show_spinner=False, ttl=600)
    def get_time_range(fn: str):
        ds = xr.open_dataset(fn)
        time_name = pick_coord(ds, ["time", "time_counter", "TIME", "valid_time"])
        t = pd.to_datetime(ds[time_name].values)
        tmin = pd.Timestamp(t.min())
        tmax = pd.Timestamp(t.max())
        ds.close()
        if USE_JST:
            tmin = tmin + pd.Timedelta(hours=9)
            tmax = tmax + pd.Timedelta(hours=9)
        return tmin, tmax

    anfc_range = get_time_range(str(FN_ANFC)) if FN_ANFC.exists() else None
    my_range   = get_time_range(str(FN_MY)) if FN_MY.exists() else None

    if anfc_range is not None:
        end_def = anfc_range[1].normalize()
        start_def = (end_def - pd.Timedelta(days=21)).normalize()
        if start_def < anfc_range[0].normalize():
            start_def = anfc_range[0].normalize()
    elif my_range is not None:
        end_def = my_range[1].normalize()
        start_def = (end_def - pd.Timedelta(days=21)).normalize()
        if start_def < my_range[0].normalize():
            start_def = my_range[0].normalize()
    else:
        st.error("波浪ファイルが見つかりません（data/ に .nc を置いてください）")
        st.stop()

    cst, ced = st.columns([1, 1])
    with cst:
        start_date = st.date_input("開始日", value=start_def.date(), key="wav_start")
    with ced:
        end_date = st.date_input("終了日", value=end_def.date(), key="wav_end")

    if start_date > end_date:
        st.error("開始日が終了日より後になっています。")
        st.stop()

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # ★固定のRANGE_* は使わず、ファイルの実データ範囲でMY/ANFCを判定する
    if (my_range is not None) and (start_ts <= my_range[1].normalize()) and (end_ts >= my_range[0].normalize()):
        hit_my = True
    else:
        hit_my = False

    if (anfc_range is not None) and (start_ts <= anfc_range[1].normalize()) and (end_ts >= anfc_range[0].normalize()):
        hit_anfc = True
    else:
        hit_anfc = False


    if hit_my and hit_anfc:
        use_kind = "BOTH"
        # MY と ANFC をまたぐ場合は後段で両方読み込んで結合する
        fn = None
        point = None
        avail = None
    elif hit_anfc:
        use_kind = "ANFC"
        fn = str(FN_ANFC)
        point = POINT_ANFC  # None
        avail = anfc_range
    elif hit_my:
        use_kind = "MY"
        fn = str(FN_MY)
        point = POINT_MY  # None
        avail = my_range
    else:
        st.error("選択した期間がデータ範囲外です。開始日・終了日を見直してください。")
        st.stop()

    if (fn is not None) and (not Path(fn).exists()):
        st.error("ファイルがありません（data/ に .nc を置いてください）")
        st.stop()


    # データ範囲に合わせてクランプ（BOTH の場合は各系列で個別にクランプ）
    if (use_kind != "BOTH") and (avail is not None):
        s_clamp = max(start_ts, avail[0].normalize())
        e_clamp = min(end_ts, avail[1].normalize())
        if (s_clamp != start_ts) or (e_clamp != end_ts):
            st.info(f"期間をデータ範囲に合わせて補正しました：{s_clamp.date()} – {e_clamp.date()}")
            start_ts, end_ts = s_clamp, e_clamp

    with st.spinner("波浪を計算中..."):

        # --- 参照（正規化）基準：MY+ANFC が揃う場合は全期間で共通化（比較のため） ---
        daily_ref = None
        if (my_range is not None) and FN_MY.exists() and (anfc_range is not None) and FN_ANFC.exists():
            my_s = my_range[0].normalize()
            my_e = my_range[1].normalize()
            an_s = anfc_range[0].normalize()
            an_e = anfc_range[1].normalize()
            daily_my_ref, _ = load_wave_daily(str(FN_MY), POINT_MY, (my_s.strftime("%Y-%m-%d"), my_e.strftime("%Y-%m-%d")))
            daily_an_ref, _ = load_wave_daily(str(FN_ANFC), POINT_ANFC, (an_s.strftime("%Y-%m-%d"), an_e.strftime("%Y-%m-%d")))
            daily_ref = pd.concat([daily_my_ref, daily_an_ref], axis=0).sort_index()
        else:
            # フォールバック：選択した系列の全期間を参照にする
            if use_kind == "ANFC":
                ref_s = anfc_range[0].normalize()
                ref_e = anfc_range[1].normalize()
                daily_ref, _ = load_wave_daily(str(FN_ANFC), POINT_ANFC, (ref_s.strftime("%Y-%m-%d"), ref_e.strftime("%Y-%m-%d")))
            else:
                ref_s = my_range[0].normalize()
                ref_e = my_range[1].normalize()
                daily_ref, _ = load_wave_daily(str(FN_MY), POINT_MY, (ref_s.strftime("%Y-%m-%d"), ref_e.strftime("%Y-%m-%d")))

        if daily_ref is None or daily_ref.empty:
            daily = daily_ref
        else:
            H_lo, H_hi = calc_lohi(daily_ref["Hmax"], Q_LOW, Q_HIGH)
            T_lo, T_hi = calc_lohi(daily_ref["Tp_mean"], Q_LOW, Q_HIGH)
            ref_q = (H_lo, H_hi, T_lo, T_hi)

            if use_kind == "MY":
                daily, _ = load_wave_daily(str(FN_MY), POINT_MY,
                                           (start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")),
                                           ref_quantiles=ref_q)
                if daily is not None and not daily.empty:
                    daily["kind"] = "MY"

            elif use_kind == "ANFC":
                daily, _ = load_wave_daily(str(FN_ANFC), POINT_ANFC,
                                           (start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d")),
                                           ref_quantiles=ref_q)
                if daily is not None and not daily.empty:
                    daily["kind"] = "ANFC"

            else:  # BOTH
                # MY 区間
                s_my = start_ts
                e_my = min(end_ts, my_range[1].normalize())
                daily_my, _ = load_wave_daily(str(FN_MY), POINT_MY,
                                              (s_my.strftime("%Y-%m-%d"), e_my.strftime("%Y-%m-%d")),
                                              ref_quantiles=ref_q)
                if daily_my is not None and not daily_my.empty:
                    daily_my["kind"] = "MY"

                # ANFC 区間
                s_an = max(start_ts, anfc_range[0].normalize())
                e_an = end_ts
                daily_an, _ = load_wave_daily(str(FN_ANFC), POINT_ANFC,
                                              (s_an.strftime("%Y-%m-%d"), e_an.strftime("%Y-%m-%d")),
                                              ref_quantiles=ref_q)
                if daily_an is not None and not daily_an.empty:
                    daily_an["kind"] = "ANFC"

                daily = pd.concat([daily_my, daily_an], axis=0).sort_index()
                if daily is not None and not daily.empty:
                    daily = daily[~daily.index.duplicated(keep="last")]
    status, msg, _ = classify_alerts(daily)
    if status == "ALERT":
        st.error(msg)
    elif status == "WATCH":
        st.warning(msg)
    elif status == "OK":
        st.success(msg)
    else:
        st.info(msg)

    if daily is not None and not daily.empty:
        st.plotly_chart(plot_wave(daily, f"期間推移（{use_kind}）" if use_kind != "BOTH" else "期間推移（MY+ANFC）"), use_container_width=True)

        cols = [c for c in ["score", "Hmax", "Tp_mean", "Dir_mean", "H_idx", "T_idx", "D_idx"] if c in daily.columns]
        dshow = daily.sort_index().tail(RECENT_DAYS_TABLE)[cols].copy()
        rename = {
            "score": "スコア",
            "Hmax": "波高(m)",
            "Tp_mean": "周期(s)",
            "Dir_mean": "波向(°)",
            "H_idx": "波高idx",
            "T_idx": "周期idx",
            "D_idx": "方向idx",
        }
        st.dataframe(dshow.rename(columns=rename), use_container_width=True)

        st.download_button(
            f"{use_kind} daily CSV",
            daily.to_csv(index=True).encode("utf-8"),
            (("my_anfc_daily.csv") if use_kind=="BOTH" else f"{use_kind.lower()}_daily.csv"),
            "text/csv",
        )

# =====================================================
# モード：グラフ（水温）
# =====================================================
else:
    if df_raw is None:
        st.error(f"CSV が見つかりません: {CSV_PATH}（repoの data/ に置いてください）")
        st.stop()

    c0, c1, c3 = st.columns([1.0, 1.1, 3.0])

    with c0:
        sec_label = pill_toggle(["なし", "Sal", "DO"], default="なし", key="sec_label", label="副軸")
    with c1:
        agg_label = pill_toggle(["日時", "日平均"], default="日時", key="agg_label", label="集計")
    with c3:
        selected_years = st.multiselect("年", years, default=years[-2:] if len(years) >= 2 else years)

    if not selected_years:
        st.stop()

    agg_mode = "datetime" if agg_label == "日時" else "daily"

    colors = year_color_map(selected_years)
    colors_sec = {y: colors[y] for y in selected_years}

    sec_metric = {"Sal": "Sal(1m)", "DO": "DO(3m)"}.get(sec_label)
    if sec_metric and sec_metric not in df_raw.columns:
        st.warning(f"副軸の列が見つかりません: {sec_metric}（副軸はOFFにしました）")
        sec_metric = None
        sec_label = "なし"

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
            d["AlignX"] = pd.to_datetime(dict(year=2001, month=d["Month"], day=d["Day"], hour=t.hour, minute=t.minute, second=t.second))
        return d.groupby(["Year", "AlignX"])[metric].agg(["mean", "min", "max"]).reset_index()

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

    legend_cfg = dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0)

    with tab_ts:
        fig = go.Figure()
        for y in selected_years:
            d = ts_stats[ts_stats["Year"] == y]
            if d.empty:
                continue
            add_band(fig, d["X"], d["min"], d["max"], colors[y], alpha=0.25, yaxis="y")
            add_line(fig, d["X"], d["mean"], colors[y], f"{y} 水温", yaxis="y", width=2.4)

        if ts_sec is not None:
            for y in selected_years:
                d2 = ts_sec[ts_sec["Year"] == y]
                if d2.empty:
                    continue
                add_band(fig, d2["X"], d2["min"], d2["max"], colors_sec[y], alpha=0.12, yaxis="y2")
                add_line(fig, d2["X"], d2["mean"], colors_sec[y], f"{y} {sec_label}",
                         yaxis="y2", width=1.5, dash="dash", alpha=0.55)

        layout = dict(
            template="plotly_white",
            height=520,
            hovermode="x unified",
            legend=legend_cfg,
            margin=dict(l=40, r=30, t=40, b=90),
        )
        if ts_sec is not None:
            layout.update(dict(
                yaxis=dict(title="水温 (℃)"),
                yaxis2=dict(title="Sal" if sec_label == "Sal" else "DO", overlaying="y", side="right", showgrid=False),
            ))
        fig.update_layout(**layout)
        st.plotly_chart(fig, use_container_width=True)

    with tab_md:
        fig = go.Figure()
        for y in selected_years:
            d = md_stats[md_stats["Year"] == y]
            if d.empty:
                continue
            add_band(fig, d["AlignX"], d["min"], d["max"], colors[y], alpha=0.25, yaxis="y")
            add_line(fig, d["AlignX"], d["mean"], colors[y], f"{y} 水温", yaxis="y", width=2.4)

        if md_sec is not None:
            for y in selected_years:
                d2 = md_sec[md_sec["Year"] == y]
                if d2.empty:
                    continue
                add_band(fig, d2["AlignX"], d2["min"], d2["max"], colors_sec[y], alpha=0.12, yaxis="y2")
                add_line(fig, d2["AlignX"], d2["mean"], colors_sec[y], f"{y} {sec_label}",
                         yaxis="y2", width=1.5, dash="dash", alpha=0.55)

        layout = dict(
            template="plotly_white",
            height=520,
            hovermode="x unified",
            legend=legend_cfg,
            margin=dict(l=40, r=30, t=40, b=90),
        )
        if md_sec is not None:
            layout.update(dict(
                yaxis=dict(title="水温 (℃)"),
                yaxis2=dict(title="Sal" if sec_label == "Sal" else "DO", overlaying="y", side="right", showgrid=False),
            ))
        fig.update_layout(**layout)
        fig.update_xaxes(tickformat="%m/%d")
        st.plotly_chart(fig, use_container_width=True)