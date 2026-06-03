import hashlib
import re
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="", layout="wide")

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
CSV_PATH = DATA_DIR / "Taiki_temp.csv"
CMEM_THETAO_CSV_PATH = DATA_DIR / "cmem_thetao.csv"

TEST_DATA_PATH = DATA_DIR / "test_data.csv"
GROWTH_DATA_PATH = DATA_DIR / "Total_growth_data.csv"
ENV_CSV_PATH = DATA_DIR / "asahihama_env2.csv"
ENV_ENCODING = "utf-8-sig"
FN_MY   = DATA_DIR / "cmem_wav_MY.nc"
FN_ANFC = DATA_DIR / "cmem_wav_ANFC.nc"

ENCODING = "utf-8-sig"
DATE_COL = "DATE"
METRIC = "depth_avg"
HOT_RED = "#d32f2f"

POINT_MY = None
POINT_ANFC = None

USE_JST = True
JST_TZ = "Asia/Tokyo"
DROP_LEAPDAY = True

ENTRANCE_AXIS_DEG = 20.0
OUTER_BW_AXIS_DEG = 45.0
INNER_BW_AXIS_DEG = 135.0
OFFSHORE_BEARING_DEG = 90.0
SIG_ENTRANCE, W_ENTRANCE = 22.0, 1.00
SIG_OUTER, W_OUTER = 30.0, 0.75
SIG_INNER, W_INNER = 35.0, 0.55
DIR_MODE = "max"
ALLOW_180_FLIP = True

Q_LOW, Q_HIGH = 0.05, 0.95
SMOOTH_DAYS = 3

DIR_EXP_Q = 0.60
EXCESS_AH = 0.80
EXCESS_AT = 0.50
EXCESS_P  = 1.20

ALERT_SCORE = 0.4
ALERT_DAYS = 4
WATCH_DAYS = 2
LOOKBACK_DAYS = 60

RECENT_DAYS_TABLE = 10


SEC_BIN_HOURS = 3
FILL_MISSING_TIME = True
FILL_FREQ = "30min"

GAP_BREAK = pd.Timedelta("3D")

def pill_toggle(options, default, key, label=""):
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

def insert_gaps_by_time(x, y, gap: pd.Timedelta):
    if x is None or y is None:
        return x, y

    xi = pd.to_datetime(pd.Series(x), errors="coerce")
    yi = pd.Series(y)

    order = np.argsort(xi.values)
    xi = xi.iloc[order].reset_index(drop=True)
    yi = yi.iloc[order].reset_index(drop=True)

    dt = xi.diff()

    out_x, out_y = [], []
    for i in range(len(xi)):
        if i > 0 and pd.notna(dt.iloc[i]) and dt.iloc[i] >= gap:
            out_x.append(xi.iloc[i])
            out_y.append(None)
        out_x.append(xi.iloc[i])
        out_y.append(yi.iloc[i])

    return out_x, out_y


def segment_by_gap(x, y_list, gap: pd.Timedelta):
    xi = pd.to_datetime(pd.Series(x), errors="coerce")
    order = np.argsort(xi.values)
    xi = xi.iloc[order].reset_index(drop=True)
    ys = [pd.Series(y).iloc[order].reset_index(drop=True) for y in y_list]
    dt = xi.diff()
    breaks = [i for i in range(1, len(xi)) if pd.notna(dt.iloc[i]) and dt.iloc[i] >= gap]
    cuts = [0] + breaks + [len(xi)]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b - a <= 0:
            continue
        xseg = xi.iloc[a:b].tolist()
        ysegs = [yy.iloc[a:b].tolist() for yy in ys]
        out.append((xseg, *ysegs))
    return out


def add_band(fig, x, ymin, ymax, color, alpha=0.30, yaxis="y"):
    segs = segment_by_gap(x, [ymin, ymax], GAP_BREAK)
    for (xseg, yminseg, ymaxseg) in segs:
        fig.add_trace(
            go.Scatter(
                x=xseg,
                y=yminseg,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                yaxis=yaxis,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xseg,
                y=ymaxseg,
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
    segs = segment_by_gap(x, [y], GAP_BREAK)
    first = True
    for (xseg, yseg) in segs:
        fig.add_trace(
            go.Scatter(
                x=xseg,
                y=yseg,
                mode="lines",
                connectgaps=False,
                line=dict(color=rgba_from_color(color, alpha) if alpha < 1.0 else color, width=width, dash=dash),
                name=name,
                showlegend=True if first else False,
                yaxis=yaxis,
            )
        )
        first = False

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


@st.cache_data(show_spinner="データ読み込み中...", ttl=600)
def load_raw(csv_path: Path, _hash_val: str):
    df = pd.read_csv(str(csv_path), encoding=ENCODING)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # ★追加（これが本質）
    df[DATE_COL] = df[DATE_COL].dt.round("30min")

    df = df.dropna(subset=[DATE_COL]).copy()

    df["1m_avg"] = safe_row_mean(df, ["1m(UML)", "1m(Tele)"])
    df["2m_avg"] = safe_row_mean(df, ["2m(UML)", "2m(Tele)"])
    df["3m_avg"] = safe_row_mean(df, ["3m(UML)", "3m(Tele)"])
    df[METRIC] = df[["1m_avg", "2m_avg", "3m_avg"]].mean(axis=1, skipna=True)

    df["Year"] = df[DATE_COL].dt.year
    df["Month"] = df[DATE_COL].dt.month
    df["Day"] = df[DATE_COL].dt.day
    return df

def reindex_full_time(df: pd.DataFrame, freq: str = "30min") -> pd.DataFrame:
    if df is None or df.empty or DATE_COL not in df.columns:
        return df
    d = df.copy()
    for c in ["Year", "Month", "Day"]:
        if c in d.columns:
            d = d.drop(columns=c)

    d = d.sort_values(DATE_COL)
    num_cols = d.select_dtypes(include=["number"]).columns.tolist()

    agg = {}
    for c in d.columns:
        if c == DATE_COL:
            continue
        agg[c] = "mean" if c in num_cols else "first"

    d = d.groupby(DATE_COL, as_index=False).agg(agg)
    d[DATE_COL] = pd.to_datetime(d[DATE_COL], errors="coerce")
    d = d.dropna(subset=[DATE_COL]).sort_values(DATE_COL)
    d = d.set_index(DATE_COL)

    full_idx = pd.date_range(start=d.index.min(), end=d.index.max(), freq=freq)
    d = d.reindex(full_idx)

    d[DATE_COL] = d.index
    d["Year"] = d.index.year
    d["Month"] = d.index.month
    d["Day"] = d.index.day

    return d.reset_index(drop=True)


@st.cache_data(show_spinner="CMEMデータ読み込み中...", ttl=600)
def load_cmem_thetao(csv_path: Path, _hash_val: str):
    df = pd.read_csv(str(csv_path))
    if 'Date' not in df.columns or 'Temp' not in df.columns:
        return pd.DataFrame(columns=['Date', 'Date_JST', 'Depth', 'Temp'])
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
    df = df.dropna(subset=['Date']).copy()
    df['Date_JST'] = df['Date'].dt.tz_convert(JST_TZ).dt.tz_localize(None)
    if 'Depth' not in df.columns:
        df['Depth'] = np.nan
    return df

@st.cache_data(show_spinner=False, ttl=600)
def load_test_data(csv_path: Path, _hash_val: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(str(csv_path), encoding="utf-8")
    except Exception:
        df = pd.read_csv(str(csv_path), encoding="utf-8-sig")
    if "Date" not in df.columns:
        return pd.DataFrame()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).copy()
    df["Date"] = df["Date"].dt.normalize()
    return df

@st.cache_data(show_spinner=False, ttl=600)
def load_growth_data(csv_path: Path, _hash_val: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(str(csv_path), encoding="utf-8")
    except Exception:
        df = pd.read_csv(str(csv_path), encoding="utf-8-sig")

    if ("Date" not in df.columns) and ("Year" not in df.columns):
        return pd.DataFrame()

    d1 = pd.to_datetime(df["Date"], errors="coerce") if "Date" in df.columns else None
    d2 = pd.to_datetime(df["Year"], errors="coerce") if "Year" in df.columns else None

    if d1 is None:
        df["Date"] = d2
    elif d2 is None:
        df["Date"] = d1
    else:
        df["Date"] = d1.fillna(d2)

    df = df.dropna(subset=["Date"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Date"]).copy()
    df["Year"] = df["Date"].dt.year

    for c in ["FL", "BW", "GW", "GSI"]:
        if c in df.columns:
            s = df[c].astype(str).str.replace(",", "", regex=False)
            s = s.replace({"ND": None, "欠測": None, "nan": None, "": None})
            df[c] = pd.to_numeric(s, errors="coerce")

    if "Remark" in df.columns:
        r = df["Remark"].astype(str).str.strip()
        df["Remark"] = r.replace({"": np.nan, "nan": np.nan, "None": np.nan})

    return df
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

    if point_latlon is None:
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
            "Hmax": df[v_h].resample(pd.Timedelta(hours=3)).max(),
            "Tp_mean": df[v_t].resample(pd.Timedelta(hours=3)).mean(),
            "Dir_mean": df[v_d].resample(pd.Timedelta(hours=3)).mean(),
        }
    ).dropna()

    ds.close()

    if daily.empty:
        return daily, {}

    if DROP_LEAPDAY:
        md = daily.index.strftime("%m-%d")
        daily = daily[md != "02-29"].copy()

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

    d_eff = np.clip(daily["D_idx"].astype(float), 0, 1) ** DIR_EXP_Q
    score_base = daily["H_idx"].astype(float) * daily["T_idx"].astype(float) * d_eff

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


@st.cache_data(show_spinner=False, ttl=600)
def load_env(csv_path: Path, _hash_val: str) -> pd.DataFrame:
    df = pd.read_csv(str(csv_path), encoding=ENV_ENCODING)

    if "日付" not in df.columns:
        return pd.DataFrame()

    df["日付"] = pd.to_datetime(df["日付"], errors="coerce")
    df = df.dropna(subset=["日付"]).copy()

    for c in ["時化", "流木等", "濁り"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["DateKey"] = df["日付"].dt.normalize()
    df = df.set_index("DateKey").sort_index()
    return df

def plot_wave(daily: pd.DataFrame, title: str):
    d = daily.sort_index().copy()
    fig = go.Figure()

    if "kind" in d.columns:
        for k, g in d.groupby("kind"):
            fig.add_trace(go.Scatter(
                x=g.index, y=g["score"], mode="lines+markers",
                name=f"スコア({k})", line=dict(width=2), marker=dict(size=5),
            ))
            fig.add_trace(go.Scatter(
                x=g.index, y=g["Hmax"], mode="lines",
                name=f"波高(m)({k})", yaxis="y2",
                line=dict(width=2, dash="dot")
            ))
    else:
        fig.add_trace(go.Scatter(
            x=d.index, y=d["score"], mode="lines+markers",
            name="スコア", line=dict(width=2), marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=d.index, y=d["Hmax"], mode="lines",
            name="波高(m)", yaxis="y2",
            line=dict(width=2, dash="dot")
        ))

    try:
        if ENV_CSV_PATH.exists():
            b = ENV_CSV_PATH.read_bytes()
            env_hash = f"{hashlib.sha1(b).hexdigest()}_{len(b)}"
            env = load_env(ENV_CSV_PATH, env_hash)

            if (env is not None) and (not env.empty) and ("時化" in env.columns):
                idx = pd.DatetimeIndex(pd.to_datetime(d.index))
                tz = idx.tz

                def _x(dt_naive_jst):
                    if tz is None:
                        return dt_naive_jst
                    if isinstance(dt_naive_jst, pd.DatetimeIndex):
                        return dt_naive_jst.tz_localize(JST_TZ).tz_convert(tz)
                    return pd.to_datetime(dt_naive_jst).tz_localize(JST_TZ).tz_convert(tz)

                if tz is None:
                    idx_jst_naive = idx
                else:
                    idx_jst = idx.tz_convert(JST_TZ)
                    idx_jst_naive = idx_jst.tz_localize(None)

                days = pd.DatetimeIndex(sorted(set(idx_jst_naive.normalize())))
                present_days = set(pd.DatetimeIndex(env.index).normalize())

                for day in days:
                    if day not in present_days:
                        x0 = _x(day)
                        x1 = _x(day + pd.Timedelta(days=1))
                        fig.add_vrect(
                            x0=x0,
                            x1=x1,
                            fillcolor="#BDBDBD",
                            opacity=0.22,
                            line_width=0,
                            layer="below",
                        )

                s = env.reindex(days)["時化"].dropna()

                if not s.empty:
                    colmap = {1: "#FFEB3B", 2: "#FF9800", 3: "#F44336"}

                    for day, val in s.items():
                        if not np.isfinite(val):
                            continue
                        v = int(val)
                        x0 = _x(day)
                        x1 = _x(day + pd.Timedelta(days=1))
                        fig.add_vrect(
                            x0=x0,
                            x1=x1,
                            fillcolor=colmap.get(v, "#999999"),
                            opacity=0.06 + 0.06 * v,
                            line_width=0,
                            layer="below",
                        )

                    x = s.index + pd.Timedelta(hours=12)
                    if tz is not None:
                        x = _x(x)
                    y = (s.values - 1.0) / 2.0
                    colors = [colmap.get(int(v), "#999999") for v in s.values]

                    fig.add_trace(go.Scatter(
                        x=x, y=y,
                        mode="markers",
                        name="時化(作業日誌)",
                        marker=dict(
                            symbol="square", size=10, color=colors,
                            line=dict(width=0.5, color="rgba(0,0,0,0.4)")
                        ),
                        customdata=s.values,
                        hovertemplate="日付=%{x|%Y-%m-%d}<br>時化=%{customdata}<extra></extra>",
                    ))
    except Exception:
        pass

    fig.update_layout(
        template="plotly_white",
        height=320,
        hovermode="x unified",
        title=title,
        yaxis=dict(title="スコア (0–1)", range=[0, 1.1]),
        yaxis2=dict(title="波高(m)", overlaying="y", side="right", showgrid=False),
        margin=dict(l=40, r=40, t=40, b=30),
        legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
    )
    return fig


df_raw = None
years = []
CURRENT_YEAR = None

if CSV_PATH.exists():
    csv_bytes = CSV_PATH.read_bytes()
    file_hash = f"{hashlib.sha1(csv_bytes).hexdigest()}_{len(csv_bytes)}"
    df_raw = load_raw(CSV_PATH, file_hash)

    # ===== 追加：欠測日時補完（行が無い欠測） =====
    if FILL_MISSING_TIME and df_raw is not None and not df_raw.empty:
        df_raw = reindex_full_time(df_raw, freq=FILL_FREQ)

    years = sorted(df_raw["Year"].dropna().unique().tolist())
    CURRENT_YEAR = max(years) if years else None

mode = pill_toggle(["水温", "うねり", "グラフ", "飼育"], default="水温", key="mode")

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
        start_def = (end_def - pd.Timedelta(days=10)).normalize()
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
        fn = None
        avail = None
    elif hit_anfc:
        use_kind = "ANFC"
        fn = str(FN_ANFC)
        avail = anfc_range
    elif hit_my:
        use_kind = "MY"
        fn = str(FN_MY)
        avail = my_range
    else:
        st.error("選択した期間がデータ範囲外です。開始日・終了日を見直してください。")
        st.stop()

    if (fn is not None) and (not Path(fn).exists()):
        st.error("ファイルがありません（data/ に .nc を置いてください）")
        st.stop()

    if (use_kind != "BOTH") and (avail is not None):
        s_clamp = max(start_ts, avail[0].normalize())
        e_clamp = min(end_ts, avail[1].normalize())
        if (s_clamp != start_ts) or (e_clamp != end_ts):
            st.info(f"期間をデータ範囲に合わせて補正しました：{s_clamp.date()} – {e_clamp.date()}")
            start_ts, end_ts = s_clamp, e_clamp

    with st.spinner("波浪を計算中..."):

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

            else:
                s_my = start_ts
                e_my = min(end_ts, my_range[1].normalize())
                daily_my, _ = load_wave_daily(str(FN_MY), POINT_MY,
                                              (s_my.strftime("%Y-%m-%d"), e_my.strftime("%Y-%m-%d")),
                                              ref_quantiles=ref_q)
                if daily_my is not None and not daily_my.empty:
                    daily_my["kind"] = "MY"

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


elif mode == "飼育":
    if not TEST_DATA_PATH.exists():
        st.error(f"test_data.csv が見つかりません: {TEST_DATA_PATH}（repoの data/ に置いてください）")
        st.stop()

    b = TEST_DATA_PATH.read_bytes()
    h = f"{hashlib.sha1(b).hexdigest()}_{len(b)}"
    df_test = load_test_data(TEST_DATA_PATH, h)

    if df_test is None or df_test.empty:
        st.warning("test_data.csv を読み込めません（Date列/欠測の可能性）。")
        st.stop()

    cages = sorted({int(m.group(1)) for c in df_test.columns for m in [re.match(r"^(\d+)_food$", c)] if m})
    if not cages:
        st.warning("飼育列（*_food）が見つかりません。列名をご確認ください。")
        st.stop()

    d0 = df_test.copy()
    d0["Year"] = d0["Date"].dt.year

    years_avail = set(d0["Year"].dropna().unique().tolist())
    try:
        if (df_raw is not None) and (not df_raw.empty) and ("Year" in df_raw.columns):
            years_avail |= set(df_raw["Year"].dropna().unique().tolist())
    except Exception:
        pass

    try:
        if GROWTH_DATA_PATH.exists():
            gb = GROWTH_DATA_PATH.read_bytes()
            gh = f"{hashlib.sha1(gb).hexdigest()}_{len(gb)}"
            gdf_all = load_growth_data(GROWTH_DATA_PATH, gh)
            if (gdf_all is not None) and (not gdf_all.empty) and ("Year" in gdf_all.columns):
                years_avail |= set(gdf_all["Year"].dropna().unique().tolist())
        else:
            gdf_all = pd.DataFrame()
    except Exception:
        gdf_all = pd.DataFrame()

    years_avail = sorted([int(y) for y in years_avail if pd.notna(y)])
    if not years_avail:
        st.warning("年情報を作れませんでした。データをご確認ください。")
        st.stop()

    default_years = years_avail[-2:] if len(years_avail) >= 2 else years_avail

    # UI（モバイル向け：飼育選択は非表示で全選択固定）
    sel_cages = cages  # 全選択固定
    sel_years = st.multiselect("年（比較）", options=years_avail, default=default_years)

    # 固定設定（UI非表示）
    show_cage_lines = False
    show_temp = True
    show_bw = True
    bw_tol_days = 5

    if not sel_years:
        st.stop()
    def _align_md(dtindex: pd.DatetimeIndex) -> pd.DatetimeIndex:
        md = pd.to_datetime(dtindex)
        mask = ~((md.month == 2) & (md.day == 29))
        md = md[mask]
        return pd.to_datetime(dict(year=2001, month=md.month, day=md.day))

    def _align_doy_from_date(dt: pd.Series) -> pd.Series:
        d = pd.to_datetime(dt)
        mask = ~((d.dt.month == 2) & (d.dt.day == 29))
        d = d.where(mask)
        return d.dt.dayofyear

    def _bin_center_doy(doy: pd.Series, tol_days: int) -> pd.Series:
        w = int(2 * tol_days + 1)
        if w <= 1:
            return doy
        bin_id = ((doy - 1) // w).astype('Int64')
        center = (bin_id * w + (w // 2) + 1).astype('Int64')
        return center

    def _doy_to_align_date(center_doy: pd.Series) -> pd.Series:
        return pd.to_datetime('2001-01-01') + pd.to_timedelta(center_doy.astype(float) - 1, unit='D')

    def build_year_series(year: int):
        dY = d0[d0["Year"] == year].copy()
        if dY.empty:
            return None

        per_cage = {}
        all_dates = None
        for i in sel_cages:
            fcol = f"{i}_food"
            ncol = f"{i}_number"
            if fcol not in dY.columns:
                continue
            cols = ["Date", fcol]
            if ncol in dY.columns:
                cols.append(ncol)
            tmp = dY[cols].copy()
            agg = {fcol: "sum"}
            if ncol in tmp.columns:
                agg[ncol] = "mean"
            tmp = tmp.groupby("Date", as_index=True).agg(agg).sort_index()

            if ncol in tmp.columns:
                tmp["perfish_day"] = tmp[fcol] / tmp[ncol]
            else:
                tmp["perfish_day"] = np.nan

            tmp["perfish_cum"] = tmp["perfish_day"].cumsum()
            per_cage[i] = tmp
            all_dates = tmp.index if all_dates is None else all_dates.union(tmp.index)

        if not per_cage:
            return None

        all_dates = all_dates.sort_values()

        food_sum = None
        n_sum = None
        for i, tmp in per_cage.items():
            tmp2 = tmp.reindex(all_dates)
            fcol = f"{i}_food"
            ncol = f"{i}_number"
            f = tmp2[fcol].astype(float).fillna(0.0)
            if ncol in tmp2.columns:
                n = tmp2[ncol].astype(float).fillna(0.0)
            else:
                n = pd.Series(0.0, index=all_dates)
            food_sum = f if food_sum is None else (food_sum + f)
            n_sum = n if n_sum is None else (n_sum + n)

        perfish_wday = (food_sum / n_sum.replace(0, np.nan)).astype(float)
        perfish_wcum = perfish_wday.cumsum().ffill().fillna(0.0)

        return {"all_dates": all_dates, "per_cage": per_cage, "wmean_cum": perfish_wcum}

    series_by_year = {}
    for y in sel_years:
        y = int(y)
        out = build_year_series(y)
        if out is not None:
            series_by_year[y] = out

    tab_per, tab_food = st.tabs(["給餌量/尾（累計）", "給餌量（参考）"])

    with tab_per:
        fig = go.Figure()

        palette = px.colors.qualitative.Set2
        years_sorted = sorted([int(y) for y in sel_years])
        colmap = {y: palette[i % len(palette)] for i, y in enumerate(years_sorted)}

        for y in years_sorted:
            if y not in series_by_year:
                continue
            d = series_by_year[y]
            x = _align_md(pd.DatetimeIndex(d["all_dates"]))
            yv = d["wmean_cum"].reindex(d["all_dates"]).astype(float).values
            fig.add_trace(go.Scatter(
                x=x,
                y=yv,
                mode="lines",
                name=f"{y} 加重平均",
                line=dict(color=colmap[y], width=3.0),
            ))

        if show_cage_lines:
            for y in years_sorted:
                if y not in series_by_year:
                    continue
                d = series_by_year[y]
                x = _align_md(pd.DatetimeIndex(d["all_dates"]))
                for i, tmp in d["per_cage"].items():
                    yy = tmp.reindex(d["all_dates"])["perfish_cum"].astype(float).ffill().fillna(0.0).values
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=yy,
                        mode="lines",
                        showlegend=False,
                        line=dict(color=colmap[y], width=1.0),
                        opacity=0.10,
                    ))

        if show_temp and (df_raw is not None) and (not df_raw.empty):
            dtemp = df_raw[[DATE_COL, METRIC]].dropna().copy()
            dtemp["DateD"] = pd.to_datetime(dtemp[DATE_COL]).dt.floor("D")
            dtemp["Year"] = dtemp["DateD"].dt.year
            dtemp["Month"] = dtemp["DateD"].dt.month
            dtemp["Day"] = dtemp["DateD"].dt.day
            dtemp = dtemp[~((dtemp["Month"] == 2) & (dtemp["Day"] == 29))]
            dtemp = dtemp.groupby(["Year", "Month", "Day"], as_index=False)[METRIC].mean()
            dtemp["AlignX"] = pd.to_datetime(dict(year=2001, month=dtemp["Month"], day=dtemp["Day"]))

            full_idx = pd.date_range("2001-01-01", "2001-12-31", freq="D")
            full_idx = full_idx[~((full_idx.month == 2) & (full_idx.day == 29))]

            for y in years_sorted:
                tt = dtemp[dtemp["Year"] == y].set_index("AlignX")[METRIC].sort_index()
                tt = tt.reindex(full_idx)
                fig.add_trace(go.Scatter(
                    x=full_idx,
                    y=tt.values,
                    mode="lines",
                    name=f"{y} 水温",
                    yaxis="y2",
                    line=dict(color=rgba_from_color(colmap[y], 0.30), width=1.5, dash="dot"),
                    connectgaps=False,
                ))

        if show_bw and (gdf_all is not None) and (not gdf_all.empty) and ("BW" in gdf_all.columns):
            gdf = gdf_all.dropna(subset=["BW", "Date", "Year"]).copy()
            if "Remark" in gdf.columns:
                gdf["Remark"] = gdf["Remark"].astype(str).str.strip()
                gdf["Remark"] = gdf["Remark"].replace({"": np.nan, "nan": np.nan, "None": np.nan})
            else:
                gdf["Remark"] = np.nan

            # 半月単位は維持しつつ、Remarkがある半月はRemark別、ない半月は合計で表示
            all_remark_vals = sorted([str(v) for v in gdf["Remark"].dropna().unique().tolist()])
            remark_order = [v for v in all_remark_vals if v not in ["", "nan", "None"]]
            if not remark_order:
                remark_order = ["合計"]

            nR = max(1, len(remark_order))
            total_span_days = 0.72
            offset_step_days = total_span_days / nR
            day_offsets = {lab: (i - (nR - 1) / 2.0) * offset_step_days for i, lab in   enumerate(remark_order)}

            # 箱の幅は従来寄りに固定し、分割時は位置だけ少しずらす
            width_ms = int((15 * 24 * 3600 * 1000) * (0.80 / max(1, len(years_sorted))))


            for y in years_sorted:
                gy = gdf[gdf["Year"] == y]
                if gy.empty:
                    continue

                gy2 = gy.copy()
                gy2["Month"] = gy2["Date"].dt.month
                gy2["Day"] = gy2["Date"].dt.day
                gy2 = gy2[~((gy2["Month"] == 2) & (gy2["Day"] == 29))].copy()
                if gy2.empty:
                    continue

                half = (gy2["Day"] >= 16).astype(int)  # 0=前半, 1=後半
                center_day = np.where(half.values == 0, 8, 23)
                gy2["XBase"] = pd.to_datetime(dict(year=2001, month=gy2["Month"], day=center_day))
                gy2 = gy2.dropna(subset=["XBase", "BW"]).copy()
                if gy2.empty:
                    continue

                # 半月グループごとに、RemarkがあればRemark別のみ、なければ合計のみ
                pieces = []
                for x0, ghalf in gy2.groupby("XBase", sort=True):
                    nonnull = sorted([str(v) for v in ghalf["Remark"].dropna().unique().tolist()])
                    if nonnull:
                        for lab in nonnull:
                            gsub = ghalf[ghalf["Remark"] == lab].copy()
                            if gsub.empty:
                                continue
                            gsub["RemarkGroup"] = lab
                            gsub["X"] = x0 + pd.to_timedelta(day_offsets.get(lab, 0.0), unit="D")
                            pieces.append(gsub)
                    else:
                        gsub = ghalf.copy()
                        gsub["RemarkGroup"] = "合計"
                        gsub["X"] = x0
                        pieces.append(gsub)

                if not pieces:
                    continue
                gy3 = pd.concat(pieces, axis=0, ignore_index=False)

                # BW箱ひげ本体（凡例は年だけにしたいので非表示）
                for lab, gsub in gy3.groupby("RemarkGroup", sort=False):
                    if gsub.empty:
                        continue
                    fill_alpha = 0.10 if lab == "合計" else 0.18
                    fig.add_trace(go.Box(
                        x=gsub["X"],
                        y=gsub["BW"].astype(float).values,
                        name=f"{y} BW",
                        yaxis="y3",
                        showlegend=False,
                        hoverinfo="skip",
                        boxpoints=False,
                        width=width_ms,
                        marker=dict(color=rgba_from_color(colmap[y], fill_alpha)),
                        line=dict(color="#777777", width=1.4),
                    ))

                    # 中央値線（年色）：箱の幅に合わせた水平線
                    med_df = gsub.groupby("X")["BW"].median().reset_index().sort_values("X")
                    dt_half = pd.to_timedelta(width_ms * 0.45, unit='ms')
                    _xm, _ym = [], []
                    for _x0, _m in zip(med_df["X"].tolist(), med_df["BW"].astype(float).tolist()):
                        _xm += [(_x0 - dt_half), (_x0 + dt_half), None]
                        _ym += [_m, _m, None]
                    fig.add_trace(go.Scatter(
                        x=_xm,
                        y=_ym,
                        mode="lines",
                        yaxis="y3",
                        showlegend=False,
                        hoverinfo="skip",
                        line=dict(color=colmap[y], width=2.2),
                    ))

                    # Hoverは半月位置の箱ごとに min/median/max + Remark
                    gq = gsub.groupby("X")["BW"].agg(lo="min", med="median", hi="max").reset_index().sort_values("X")
                    x = gq["X"].tolist()
                    _xh, _yh, _cd = [], [], []
                    for _x0, _lo, _med, _hi in zip(
                        x,
                        gq["lo"].astype(float).tolist(),
                        gq["med"].astype(float).tolist(),
                        gq["hi"].astype(float).tolist(),
                    ):
                        _xh += [_x0, _x0, None]
                        _yh += [_lo, _hi, None]
                        _cd += [[lab, _lo, _med, _hi], [lab, _lo, _med, _hi], [None, None, None, None]]
                    fig.add_trace(go.Scatter(
                        x=_xh,
                        y=_yh,
                        mode="lines",
                        yaxis="y3",
                        showlegend=False,
                        line=dict(width=44, color="rgba(0,0,0,0)"),
                        customdata=_cd,
                        hovertemplate=(
                            f"{y} BW<br>"
                            + "区分=%{customdata[0]}<br>"
                            + "%{x|%m/%d}<br>"
                            + "min=%{customdata[1]:.0f}  median=%{customdata[2]:.0f}  max=%{customdata[3]:.0f}"
                            + "<extra></extra>"
                        ),
                    ))
        fig.update_layout(
            template="plotly_white",
            height=660,
            hovermode="x unified",
            margin=dict(l=40, r=40, t=40, b=90),
            legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
            xaxis=dict(title="（月日）", tickformat="%m/%d", range=["2001-05-01","2001-12-31"]),
            yaxis=dict(title="給餌量（g/尾）"),
            yaxis2=dict(title="水温 (℃)", overlaying="y", side="right", position=0.92, showgrid=False),
            yaxis3=dict(title="BW", overlaying="y", side="right", position=0.985, showgrid=False),
        )
        fig.update_traces(hoverinfo="skip", selector=dict(type="box"))
        st.plotly_chart(fig, use_container_width=True)

    with tab_food:
        st.info("参考表示")
        fig = go.Figure()
        palette = px.colors.qualitative.Set2
        years_sorted = sorted([int(y) for y in sel_years])
        colmap = {y: palette[i % len(palette)] for i, y in enumerate(years_sorted)}

        for y in years_sorted:
            if y not in series_by_year:
                continue
            d = series_by_year[y]
            x = _align_md(pd.DatetimeIndex(d["all_dates"]))
            food_sum = None
            for i, tmp in d["per_cage"].items():
                fcol = f"{i}_food"
                f = tmp.reindex(d["all_dates"])[fcol].astype(float).fillna(0.0)
                food_sum = f if food_sum is None else (food_sum + f)
            yv = food_sum.cumsum().values
            fig.add_trace(go.Scatter(x=x, y=yv, mode="lines", name=f"{y} 合計給餌(累積)",
                                     line=dict(color=colmap[y], width=2.2)))

        fig.update_layout(
            template="plotly_white",
            height=520,
            hovermode="x unified",
            margin=dict(l=40, r=40, t=40, b=90),
            legend=dict(orientation="h", yanchor="top", y=-0.22, xanchor="left", x=0),
            xaxis=dict(title="（月日）", tickformat="%m/%d", range=["2001-05-01","2001-12-31"]),
            yaxis=dict(title="累積給餌量 (g)"),
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    if df_raw is None:
        st.error(f"CSV が見つかりません: {CSV_PATH}（repoの data/ に置いてください）")
        st.stop()

    df_cmem = None
    if CMEM_THETAO_CSV_PATH.exists():
        cmem_bytes = CMEM_THETAO_CSV_PATH.read_bytes()
        cmem_hash = f"{hashlib.sha1(cmem_bytes).hexdigest()}_{len(cmem_bytes)}"
        df_cmem = load_cmem_thetao(CMEM_THETAO_CSV_PATH, cmem_hash)

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

    def build_timeseries_stats_sec(df, metric, bin_hours=SEC_BIN_HOURS):
        d = df[["Year", DATE_COL, metric]].dropna().copy()
        if agg_mode == "daily":
            d["X"] = d[DATE_COL].dt.floor("D")
        else:
            bh = int(bin_hours)
            if bh <= 0:
                bh = 1
            d["X"] = d[DATE_COL].dt.floor(f"{bh}h")
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
        cmem_ts_stats = None
        if df_cmem is not None and (not df_cmem.empty):
            dcm = df_cmem[["Date_JST", "Temp"]].dropna().copy()
            dcm["Year"] = dcm["Date_JST"].dt.year
            dcm = dcm[dcm["Year"].isin(selected_years)]
            if agg_mode == "daily":
                dcm["X"] = dcm["Date_JST"].dt.floor("D")
            else:
                dcm["X"] = dcm["Date_JST"]
            cmem_ts_stats = dcm.groupby(["Year", "X"])["Temp"].agg(["mean", "min", "max"]).reset_index()

        for y in selected_years:
            d = ts_stats[ts_stats["Year"] == y]
            if d.empty:
                continue
            add_band(fig, d["X"], d["min"], d["max"], colors[y], alpha=0.25, yaxis="y")
            add_line(fig, d["X"], d["mean"], colors[y], f"{y} 水温", yaxis="y", width=2.4)

        if cmem_ts_stats is not None and (not cmem_ts_stats.empty):
            for y in selected_years:
                gcm = cmem_ts_stats[cmem_ts_stats["Year"] == y]
                if gcm.empty:
                    continue
                add_band(fig, gcm["X"], gcm["min"], gcm["max"], colors[y], alpha=0.10, yaxis="y")
                add_line(fig, gcm["X"], gcm["mean"], colors[y], f"{y} CMEM(モデル)", yaxis="y", width=1.2, dash="dot", alpha=0.60)

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
        cmem_md_stats = None
        if df_cmem is not None and (not df_cmem.empty):
            dcm = df_cmem[["Date_JST", "Temp"]].dropna().copy()
            dcm["Year"] = dcm["Date_JST"].dt.year
            dcm = dcm[dcm["Year"].isin(selected_years)]
            dcm["Month"] = dcm["Date_JST"].dt.month
            dcm["Day"] = dcm["Date_JST"].dt.day
            dcm = dcm[~((dcm["Month"] == 2) & (dcm["Day"] == 29))]
            if agg_mode == "daily":
                dcm["AlignX"] = pd.to_datetime(dict(year=2001, month=dcm["Month"], day=dcm["Day"]))
            else:
                t = dcm["Date_JST"].dt
                dcm["AlignX"] = pd.to_datetime(dict(year=2001, month=dcm["Month"], day=dcm["Day"], hour=t.hour, minute=t.minute, second=t.second))
            cmem_md_stats = dcm.groupby(["Year", "AlignX"])["Temp"].agg(["mean", "min", "max"]).reset_index()

        for y in selected_years:
            d = md_stats[md_stats["Year"] == y]
            if d.empty:
                continue
            add_band(fig, d["AlignX"], d["min"], d["max"], colors[y], alpha=0.25, yaxis="y")
            add_line(fig, d["AlignX"], d["mean"], colors[y], f"{y} 水温", yaxis="y", width=2.4)

        if cmem_md_stats is not None and (not cmem_md_stats.empty):
            for y in selected_years:
                gcm = cmem_md_stats[cmem_md_stats["Year"] == y]
                if gcm.empty:
                    continue
                add_band(fig, gcm["AlignX"], gcm["min"], gcm["max"], colors[y], alpha=0.10, yaxis="y")
                add_line(fig, gcm["AlignX"], gcm["mean"], colors[y], f"{y} CMEM(モデル)", yaxis="y", width=1.2, dash="dot", alpha=0.60)

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
