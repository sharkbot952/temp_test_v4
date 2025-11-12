# -*- coding: utf-8 -*-
import os
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

# 水温グラフ用（Plotly）
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# =========================================
# 設定
# =========================================
base_dir = "./data"
def pjoin(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))

MODE_FIXED = "予測カレンダー"

# サイドバー：キャッシュクリア
if st.sidebar.button("補正キャッシュをクリア"):
    st.cache_data.clear()
    st.success("補正キャッシュをクリアしました。再計算します。")

# --- コメント用 定数（カレンダー側で使用） ---
LOOKBACK_DAYS = 7
HIGH_TEMP_TH = 22.0
DAY_BINS = [("朝", 4, 6), ("昼", 11, 13), ("夕", 16, 18)]

# =========================================
# 表示モードのボタン（セグメント／フォールバック）
# =========================================
try:
    view_mode = st.segmented_control(
        "表示モード", options=["予測カレンダー", "水温グラフ"], default="予測カレンダー"
    )
except Exception:
    view_mode = st.radio("表示モード", ["予測カレンダー", "水温グラフ"], index=0, horizontal=True)

# =========================================
# 共通ユーティリティ
# =========================================
HEAD_LENGTH_RATIO = 0.55
HEAD_HALF_HEIGHT_RATIO = 0.35
SHAFT_WIDTH_PX = 4.0
def get_arrow_svg(direction_deg, speed_mps):
    if pd.isna(speed_mps) or pd.isna(direction_deg):
        return ""
    css_angle = (direction_deg - 90) % 360
    def _style(s):
        if np.isnan(s): return 18, "#CCCCCC"
        speed_kt = s * 1.94384
        if speed_kt < 1.0: return 18, "#0000FF"
        elif speed_kt < 2.0: return 22, "#FFC107"
        else: return 26, "#FF0000"
    size, color = _style(speed_mps)
    head_length = size * HEAD_LENGTH_RATIO
    head_half_h = size * HEAD_HALF_HEIGHT_RATIO
    line_end = size - head_length
    return f"""
<svg width="{size}" height="{size}" style="display:block;margin:0 auto;transform:rotate({css_angle}deg);">
  <line x1="4" y1="{size/2}" x2="{line_end}" y2="{size/2}"
        stroke="{color}" stroke-width="{SHAFT_WIDTH_PX}" stroke-linecap="round"/>
  <polygon points="{line_end},{size/2 - head_half_h} {size},{size/2} {line_end},{size/2 + head_half_h}"
           fill="{color}"/>
</svg>
""".strip()

def get_color(temp: float, t_min: float = 5, t_max: float = 25) -> str:
    if pd.isna(temp): return "rgba(220,220,220,0.4)"
    ratio = (float(temp) - t_min) / (t_max - t_min)
    ratio = max(0, min(1, ratio))
    if ratio < 0.5:
        r = int(240 * ratio * 2); g = int(240 * ratio * 2); b = 240
    else:
        r = 240; g = int(240 * (1 - (ratio - 0.5) * 2)); b = int(240 * (1 - (ratio - 0.5) * 2))
    return f"rgba({r},{g},{b},0.4)"

def utc_to_jst_naive(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    dt = dt.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return dt

def jst_to_naive(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return dt

def safe_merge_asof_by_depth(
    left: pd.DataFrame,
    right: pd.DataFrame,
    tolerance: pd.Timedelta,
    right_value_cols: List[str],
    suffixes: Tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame:
    out_list = []
    common_depths = sorted(
        set(left["depth_m"].dropna().unique()).intersection(set(right["depth_m"].dropna().unique()))
    )
    for d in common_depths:
        l = left[left["depth_m"] == d].sort_values("datetime")
        r = right[right["depth_m"] == d].sort_values("datetime")[
            ["datetime", "depth_m"] + right_value_cols
        ]
        if l.empty or r.empty: continue
        merged = pd.merge_asof(
            l, r, on="datetime", by="depth_m",
            tolerance=tolerance, direction="nearest", suffixes=suffixes
        )
        out_list.append(merged)
    if not out_list:
        return pd.DataFrame(columns=list(left.columns) + right_value_cols)
    return pd.concat(out_list, ignore_index=True)

@st.cache_data(show_spinner=False)
def load_dr_single_file(base_dir: str, filename: str) -> pd.DataFrame:
    path = pjoin(base_dir, "pred", filename)
    if not os.path.exists(path):
        st.error(f"ファイルが見つかりません: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        st.error(f"読み込み失敗: {path} ({e})")
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = utc_to_jst_naive(df.get("Date"))
    df["depth_m"] = pd.to_numeric(df.get("Depth"), errors="coerce").round(0).astype("Int64")
    df = df.rename(columns={"Temp": "pred_temp", "Salinity": "pred_sal"})
    df = df.dropna(subset=["datetime", "depth_m"]).copy()
    if ("U" in df.columns) and ("V" in df.columns):
        df["U"] = pd.to_numeric(df["U"], errors="coerce")
        df["V"] = pd.to_numeric(df["V"], errors="coerce")
        df["Speed"] = np.sqrt(np.square(df["U"]) + np.square(df["V"]))
        df["Direction_deg"] = (np.degrees(np.arctan2(df["U"], df["V"])) + 360.0) % 360.0
    df["date_day"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    return df

TEMP_MIN, TEMP_MAX = -2.0, 40.0

@st.cache_data(show_spinner=False)
def compute_depthwise_regression(
    base_dir: str,
    train_filename: str,
    tolerance_min: int = 30,
    start_dt: Optional[pd.Timestamp] = None,
    end_dt: Optional[pd.Timestamp] = None,
    min_pairs: int = 10,
) -> Tuple[Optional[Dict[int, Tuple[float, float]]], Optional[Dict[int, int]]]:
    dr_path = pjoin(base_dir, "pred", train_filename)
    obs_path = pjoin(base_dir, "obs", train_filename)
    if not (os.path.exists(dr_path) and os.path.exists(obs_path)):
        return None, None
    try:
        pred = pd.read_csv(dr_path)
        obs = pd.read_csv(obs_path)
    except Exception as e:
        st.warning(f"補正用ファイルの読み込みに失敗しました: {e}")
        return None, None

    pred["datetime"] = utc_to_jst_naive(pred.get("Date"))
    obs["datetime"] = jst_to_naive(obs.get("Date"))
    pred["depth_m"] = pd.to_numeric(pred.get("Depth"), errors="coerce").round(0).astype("Int64")
    obs["depth_m"] = pd.to_numeric(obs.get("Depth"), errors="coerce").round(0).astype("Int64")
    pred = pred.dropna(subset=["datetime", "depth_m"]).copy()
    obs = obs.dropna(subset=["datetime", "depth_m"]).copy()
    pred = pred.rename(columns={"Temp": "pred_temp"})
    obs  = obs.rename(columns={"Temp": "obs_temp"})
    if "pred_temp" not in pred.columns or "obs_temp" not in obs.columns:
        return None, None

    if start_dt is not None:
        pred = pred[pred["datetime"] >= start_dt]
        obs  = obs[obs["datetime"]  >= start_dt]
    if end_dt is not None:
        pred = pred[pred["datetime"] <= end_dt]
        obs  = obs[obs["datetime"]  <= end_dt]
    if pred.empty or obs.empty:
        return None, None

    tol = pd.Timedelta(minutes=int(tolerance_min))
    merged = safe_merge_asof_by_depth(
        pred.sort_values(["depth_m", "datetime"]),
        obs .sort_values(["depth_m", "datetime"]),
        tol, right_value_cols=["obs_temp"], suffixes=("", "")
    )
    pair = merged.dropna(subset=["pred_temp", "obs_temp", "depth_m"]).copy()
    if pair.empty: return None, None

    reg_depth: Dict[int, Tuple[float, float]] = {}
    n_depth: Dict[int, int] = {}
    for d, g in pair.groupby("depth_m"):
        X = g["pred_temp"].astype(float).values
        y = g["obs_temp"].astype(float).values
        mask = np.isfinite(X) & np.isfinite(y)
        X, y = X[mask], y[mask]
        n = len(X)
        n_depth[int(d)] = int(n)
        if n >= min_pairs:
            A = np.vstack([X, np.ones_like(X)]).T
            beta, alpha = np.linalg.lstsq(A, y, rcond=None)[0]
            reg_depth[int(d)] = (float(alpha), float(beta))
    return (reg_depth if reg_depth else None,
            n_depth    if n_depth    else None)

# 8方位（度→方位）
def dir_to_8pt_jp(deg: float) -> str:
    if pd.isna(deg): return ""
    dirs = ["北", "北東", "東", "南東", "南", "南西", "西", "北西"]
    idx = int(((float(deg) + 22.5) % 360) // 45)
    return dirs[idx]

# 流速クラス（m/s → 穏 / やや速 / 速い）
def speed_class_from_mps(v_mps: Optional[float]) -> str:
    if v_mps is None or pd.isna(v_mps): return ""
    kt = float(v_mps) * 1.94384
    if kt >= 1.5: return "速"
    if kt >= 0.8: return "やや速"
    return "穏"

# 回帰なしでも使える簡易トレンド（℃/日）
def slope_c_per_day(ts: pd.Series, idx: pd.Series) -> float:
    if ts.empty or ts.isna().all(): return np.nan
    x = pd.to_datetime(idx).astype("int64") / 1e9
    x = (x - x.min()) / 86400.0
    y = ts.astype(float)
    if len(y) < 3: return np.nan
    var = np.var(x)
    if var < 1e-12: return np.nan
    cov = np.cov(x, y, ddof=0)[0, 1]
    return float(cov / var)

# 水深を3層へ（表層・中層・底層）
def make_layer_groups(depths: List[int]) -> Dict[str, List[int]]:
    if not depths: return {"表層": [], "中層": [], "底層": []}
    d_sorted = sorted(depths); n = len(d_sorted)
    if n <= 3:
        top = d_sorted[:1]
        mid = d_sorted[1:2] if n >= 2 else []
        bot = d_sorted[2:]  if n >= 3 else (d_sorted[-1:] if n >= 1 else [])
    elif n in (4, 5):
        top = d_sorted[:2]; mid = d_sorted[2:3]; bot = d_sorted[3:]
    else:
        top = d_sorted[:2]; bot = d_sorted[-2:]
        mid = [d for d in d_sorted if d not in top + bot]
        if len(mid) >= 3:
            c = len(mid) // 2
            mid = mid[c-1:c+1]
    return {"表層": top, "中層": mid, "底層": bot}

# 週間コメント（昼ごろ・水温のみ）
def summarize_weekly_layer_temp(
    layer_name: str,
    layer_depths: List[int],
    df_period: pd.DataFrame,
    df_all: pd.DataFrame,
    selected_day,
    use_correction: bool = False,
    reg_depthwise: Optional[Dict[int, Tuple[float, float]]] = None,
    stable_eps: float = 0.2,
    outlier_th: Optional[float] = None,
) -> Optional[str]:
    if not layer_depths or df_period is None or df_period.empty: return None
    PHYS_MIN, PHYS_MAX = -1.5, 35.0
    msgs: List[str] = []
    for depth in layer_depths:
        g = df_period[df_period["depth_m"] == depth].copy()
        if g.empty or "pred_temp" not in g.columns: continue
        g = g.sort_values("datetime")
        temps_pred = pd.to_numeric(g["pred_temp"], errors="coerce")
        if use_correction:
            if not (reg_depthwise and int(depth) in reg_depthwise): continue
            alpha, beta = reg_depthwise[int(depth)]
            temps_corr_raw   = alpha + beta * temps_pred
            temps_corr_clip  = np.clip(temps_corr_raw, TEMP_MIN, TEMP_MAX)
            mask_valid = (
                pd.notna(temps_pred) &
                pd.notna(temps_corr_raw) &
                (temps_corr_raw  > PHYS_MIN) & (temps_corr_raw  < PHYS_MAX) &
                (temps_corr_clip > TEMP_MIN) & (temps_corr_clip < TEMP_MAX)
            )
            if outlier_th is not None:
                diff = (temps_corr_raw - temps_pred).abs()
                mask_valid &= (diff < float(outlier_th))
            temps = pd.Series(temps_corr_raw)[mask_valid]
        else:
            temps = temps_pred[(pd.notna(temps_pred)) & (temps_pred > PHYS_MIN) & (temps_pred < PHYS_MAX)]
        temps = pd.to_numeric(temps, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        n = int(len(temps))
        if n == 0: continue
        t_min = float(temps.min()); t_max = float(temps.max())
        if t_max >= HIGH_TEMP_TH:
            tag = f"高水温注意（{t_min:.1f}℃～{t_max:.1f}℃）"
        else:
            if n >= 2:
                t_start = float(temps.iloc[0]); t_end = float(temps.iloc[-1])
                delta = t_end - t_start
                if   delta >  stable_eps: tag = f"上昇（{t_start:.1f}℃→{t_end:.1f}℃）"
                elif delta < -stable_eps: tag = f"下降（{t_start:.1f}℃→{t_end:.1f}℃）"
                else:                     tag = f"安定（{t_start:.1f}℃）"
            else:
                t = float(temps.iloc[0]); tag = f"安定（{t:.1f}℃）"
        msgs.append(f"{depth}m{tag}")
    if not msgs: return None
    return f"**{layer_name}**： " + "／".join(msgs)

# 選択日コメント（レンジ非表示）
def summarize_daily_layer_flow(
    layer_name: str,
    layer_depths: List[int],
    df_day: pd.DataFrame,
    use_short_labels: bool = True,
    merge_same_segments: bool = False
) -> Optional[str]:
    if not layer_depths: return None
    order = {"朝": 0, "昼": 1, "夕": 2}
    rows: List[Tuple[str, str, str]] = []
    for label, h0, h1 in DAY_BINS:
        g = df_day[(df_day["depth_m"].isin(layer_depths)) & (df_day["datetime"].dt.hour.between(h0, h1))]
        if g.empty: continue
        U_mean = g["U"].mean() if "U" in g.columns else np.nan
        V_mean = g["V"].mean() if "V" in g.columns else np.nan
        if pd.notna(U_mean) and pd.notna(V_mean):
            speed_mean = float(np.sqrt(U_mean**2 + V_mean**2))
            dir_deg_mean = (np.degrees(np.arctan2(U_mean, V_mean)) + 360.0) % 360.0
        else:
            D = g["Direction_deg"].dropna() if "Direction_deg" in g.columns else pd.Series(dtype=float)
            if D.empty: continue
            rad = np.deg2rad(D.values)
            C = np.cos(rad).mean(); S = np.sin(rad).mean()
            dir_deg_mean = (np.degrees(np.arctan2(S, C)) + 360.0) % 360.0
            speed_mean = g["Speed"].mean() if "Speed" in g.columns else np.nan
        d_txt = dir_to_8pt_jp(dir_deg_mean) if pd.notna(dir_deg_mean) else ""
        v_cls = speed_class_from_mps(speed_mean) if pd.notna(speed_mean) else ""
        if use_short_labels and v_cls:
            v_map = {"穏やか": "穏", "中程度": "やや速", "速い": "速"}
            v_cls = v_map.get(v_cls, v_cls)
        if d_txt or v_cls:
            rows.append((label, d_txt, v_cls))
    if not rows: return None
    segments: List[str] = []
    if merge_same_segments:
        bucket: Dict[Tuple[str, str], List[str]] = {}
        for lbl, d, v in rows: bucket.setdefault((d, v), []).append(lbl)
        for (d, v), lbls in bucket.items():
            lbls_sorted = sorted(lbls, key=lambda x: order.get(x, 99))
            inner = "・".join([x for x in [d, v] if x])
            segments.append(f"{'・'.join(lbls_sorted)}（{inner}）")
    else:
        rows_sorted = sorted(rows, key=lambda r: order.get(r[0], 99))
        for lbl, d, v in rows_sorted:
            inner = "・".join([x for x in [d, v] if x])
            segments.append(f"{lbl}（{inner}）")
    return f"**{layer_name}**： " + "／".join(segments)

def normalize_generic_uploaded_csv(file) -> pd.DataFrame:
    """
    任意CSV（UTF-8/JST/Depth任意）を右軸重ね表示用に正規化。
    必須: Date 列（JST/naive 前提）
    任意: Depth 列（なければ 0m）
    数値列のみを描画候補にする（Date/Depth/source は除外）
    """
    try:
        df = pd.read_csv(file, encoding="utf-8")
    except Exception as e:
        st.warning(f"アップロードCSV読み込み失敗: {getattr(file, 'name', 'uploaded')} ({e})")
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    if "Date" not in df.columns:
        st.warning(f"'{getattr(file, 'name', 'uploaded')}' に Date 列がありません。")
        return pd.DataFrame()
    df["datetime"] = jst_to_naive(df.get("Date"))
    if "Depth" in df.columns:
        df["depth_m"] = pd.to_numeric(df.get("Depth"), errors="coerce").round(0).astype("Int64")
    else:
        df["depth_m"] = 0
    df = df.dropna(subset=["datetime"]).copy()
    for c in df.columns:
        if c in ["Date", "datetime", "Depth", "depth_m"]:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["source"] = getattr(file, "name", "uploaded_generic")
    return df

# =========================================
# HTML/CSS レンダラ（カレンダー）
# =========================================
def get_calendar_css(max_h_vh: int) -> str:
    return f"""
<style>
.calendar-scroll-container {{
  overflow: auto;
  max-height: {max_h_vh}vh;
  max-width: 100%;
  -webkit-overflow-scrolling: touch;
  border: 1px solid #e5e5e5;
  border-radius: 8px;
  background:
    linear-gradient(to right, rgba(0,0,0,0.06), rgba(0,0,0,0)) left/16px 100% no-repeat,
    linear-gradient(to left, rgba(0,0,0,0.06), rgba(0,0,0,0)) right/16px 100% no-repeat;
  background-attachment: local, local;
}}
.calendar-table {{
  border-collapse: collapse;
  width: max-content;
  min-width: 640px;
  font-size: 14px;
  font-family: 'Noto Sans JP', 'Roboto', sans-serif;
}}
.calendar-table th, .calendar-table td {{
  padding: 6px 10px;
  border-bottom: 1px solid #eee;
  vertical-align: top;
  font-family: 'Noto Sans JP', 'Roboto', sans-serif;
  line-height: 1.2;
  text-align: center;
}}
.calendar-table thead th {{
  position: sticky;
  top: 0;
  background: #fafafa;
  z-index: 2;
  white-space: nowrap;
}}
.calendar-table td.depth-cell,
.calendar-table thead th:first-child {{
  position: sticky;
  left: 0;
  background: #f7f7f7;
  z-index: 3;
  min-width: 40px;
  font-weight: 600;
}}
.calendar-table .pred-small {{
  font-size: 12px; color: #555;
}}
@media (max-width: 480px) {{
  .calendar-table {{ font-size: 13px; }}
  .calendar-table td.depth-cell {{ min-width: 30px; }}
}}
</style>
""".strip()

def render_cell_html(
    temp: Optional[float],
    speed_mps: Optional[float],
    dir_deg: Optional[float],
    use_correction: bool,
    corr_temp_raw: Optional[float],
    corr_temp_clip: Optional[float],
    bg_basis: str,
    hide_outlier_cells: bool,
    is_invalid: bool,
) -> str:
    if is_invalid:
        if hide_outlier_cells:
            return "<td></td>"
        else:
            bg_color = "rgba(220,220,220,0.6)"
            return f"<td style='background:{bg_color}'><div style='text-align:center;'>-</div></td>"

    if (bg_basis == "補正に連動") and (corr_temp_clip is not None):
        bg_color = get_color(float(corr_temp_clip))
    else:
        bg_color = get_color(float(temp)) if (temp is not None and not pd.isna(temp)) else "rgba(220,220,220,0.6)"

    pred_label = f"{float(temp):.1f}°C" if (temp is not None and not pd.isna(temp)) else "NaN"
    pred_html = f"<span class='pred-small'>{pred_label}</span>" if use_correction else f"<span>{pred_label}</span>"

    speed_html, arrow_html = "", ""
    if (speed_mps is not None and not pd.isna(speed_mps)) and (dir_deg is not None and not pd.isna(dir_deg)):
        speed_kt = float(speed_mps) * 1.94384
        speed_html = f"<span style='font-size:12px;color:#444;'>{speed_kt:.1f} kt</span>"
        arrow_html = f"<span style='display:block;line-height:1;margin:0;padding:0;'>{get_arrow_svg(float(dir_deg), float(speed_mps))}</span>"

    corr_html = ""
    if use_correction and (corr_temp_raw is not None):
        corr_html = f"<span style='color:#D32F2F;font-weight:700;font-size:14px;margin:0;padding:0;'>{corr_temp_raw:.1f}°C</span>"

    content = (
        "<div style='display:flex;flex-direction:column;align-items:center;gap:2px;margin:0;padding:0;'>"
        + pred_html + speed_html + arrow_html + corr_html + "</div>"
    )
    return f"<td style='background:{bg_color}'>{content}</td>"

def build_weekly_table_html(
    df_period: pd.DataFrame,
    day_list: List[pd.Timestamp],
    depths_for_table: List[int],
    use_correction: bool,
    reg_depthwise: Optional[Dict[int, Tuple[float, float]]],
    bg_basis: str,
    hide_outlier_cells: bool,
    outlier_th: float,
) -> str:
    PHYS_MIN, PHYS_MAX = -1.5, 35.0
    times = [d.strftime('%m/%d') for d in day_list]
    html = (
        '<div class="calendar-scroll-container"><table class="calendar-table">'
        "<thead><tr><th>水深</th>" + "".join([f"<th>{t}</th>" for t in times]) + "</tr></thead><tbody>"
    )
    for depth in depths_for_table:
        html += f"<tr><td class='depth-cell'>{depth}m</td>"
        for day in day_list:
            g = df_period[(df_period["date_day"] == day.date()) & (df_period["depth_m"] == depth)]
            if not g.empty:
                target_dt = pd.Timestamp(day.date()) + pd.Timedelta(hours=12)
                g2 = g.assign(_diff=(g["datetime"] - target_dt).abs()).sort_values("_diff")
                row = g2.iloc[[0]].drop(columns=["_diff"])
                temp = float(row["pred_temp"].values[0]) if "pred_temp" in row.columns else np.nan
                speed_val = float(row["Speed"].values[0]) if "Speed" in row.columns else np.nan
                dir_val   = float(row["Direction_deg"].values[0]) if "Direction_deg" in row.columns else np.nan
                corr_raw, corr_clip = None, None
                if use_correction and (reg_depthwise is not None) and (int(depth) in reg_depthwise) and not pd.isna(temp):
                    alpha, beta = reg_depthwise[int(depth)]
                    corr_raw  = float(alpha + beta * temp)
                    corr_clip = float(np.clip(corr_raw, TEMP_MIN, TEMP_MAX))
                # 無効判定
                is_invalid = False
                if use_correction and (corr_raw is not None):
                    if (corr_raw <= PHYS_MIN) or (corr_raw >= PHYS_MAX): is_invalid = True
                if (not is_invalid) and (corr_clip is not None):
                    if (corr_clip <= TEMP_MIN) or (corr_clip >= TEMP_MAX): is_invalid = True
                if (not is_invalid) and (corr_raw is not None) and (outlier_th is not None) and (not pd.isna(temp)):
                    if abs(float(corr_raw) - float(temp)) >= float(outlier_th): is_invalid = True

                html += render_cell_html(
                    temp=temp, speed_mps=speed_val, dir_deg=dir_val,
                    use_correction=use_correction, corr_temp_raw=corr_raw, corr_temp_clip=corr_clip,
                    bg_basis=bg_basis, hide_outlier_cells=hide_outlier_cells, is_invalid=is_invalid,
                )
            else:
                html += "<td>-</td>"
        html += "</tr>\n"
    html += "</tbody></table></div>"
    return html

def build_daily_table_html(
    df_day: pd.DataFrame,
    hours_list: List[pd.Timestamp],
    times_hr: List[str],
    depths_for_table: List[int],
    use_correction: bool,
    reg_depthwise: Optional[Dict[int, Tuple[float, float]]],
    bg_basis: str,
    hide_outlier_cells: bool,
    outlier_th: float,
) -> str:
    PHYS_MIN, PHYS_MAX = -1.5, 35.0
    html = (
        '<div class="calendar-scroll-container"><table class="calendar-table">'
        "<thead><tr><th>水深</th>" + "".join([f"<th>{t}</th>" for t in times_hr]) + "</tr></thead><tbody>"
    )
    for depth in depths_for_table:
        html += f"<tr><td class='depth-cell'>{depth}m</td>"
        for t_obj in hours_list:
            row = df_day[(df_day["datetime"].dt.floor("h") == t_obj) & (df_day["depth_m"] == depth)]
            if not row.empty:
                temp = float(row["pred_temp"].values[0]) if "pred_temp" in row.columns else np.nan
                speed_val = float(row["Speed"].values[0]) if "Speed" in row.columns else np.nan
                dir_val   = float(row["Direction_deg"].values[0]) if "Direction_deg" in row.columns else np.nan
                corr_raw, corr_clip = None, None
                if use_correction and pd.notna(temp) and (reg_depthwise is not None) and (int(depth) in reg_depthwise):
                    alpha, beta = reg_depthwise[int(depth)]
                    corr_raw  = float(alpha + beta * float(temp))
                    corr_clip = float(np.clip(corr_raw, TEMP_MIN, TEMP_MAX))
                # 無効判定
                is_invalid = False
                if use_correction and (corr_raw is not None):
                    if (corr_raw <= PHYS_MIN) or (corr_raw >= PHYS_MAX): is_invalid = True
                if (not is_invalid) and (corr_clip is not None):
                    if (corr_clip <= TEMP_MIN) or (corr_clip >= TEMP_MAX): is_invalid = True
                if (not is_invalid) and (corr_raw is not None) and (outlier_th is not None) and (not pd.isna(temp)):
                    if abs(float(corr_raw) - float(temp)) >= float(outlier_th): is_invalid = True

                html += render_cell_html(
                    temp=temp, speed_mps=speed_val, dir_deg=dir_val,
                    use_correction=use_correction, corr_temp_raw=corr_raw, corr_temp_clip=corr_clip,
                    bg_basis=bg_basis, hide_outlier_cells=hide_outlier_cells, is_invalid=is_invalid,
                )
            else:
                html += "<td>-</td>"
        html += "</tr>\n"
    html += "</tbody></table></div>"
    return html

# =========================================
# メイン処理：ボタン切替
# =========================================

# -----------------------------------------
# 予測カレンダー（既存ロジック）
# -----------------------------------------
if view_mode == "予測カレンダー":
    parent_folder_dr = pjoin(base_dir, "pred")
    if not os.path.exists(parent_folder_dr):
        st.error(f"フォルダが見つかりません: {parent_folder_dr}")
        st.stop()
    dr_files = [f for f in os.listdir(parent_folder_dr) if f.endswith(".csv")]
    if not dr_files:
        st.warning("pred に CSV がありません")
        st.stop()

    selected_file = st.selectbox("解析対象ファイルを選択", sorted(dr_files))
    use_correction = st.sidebar.checkbox("実測ベース補正", value=False)
    tolerance_min  = st.sidebar.slider("時刻差の許容範囲（分）", 5, 120, 35, step=5)
    train_days     = st.sidebar.slider("補正学習期間（日数）", 7, 90, 30, step=1)
    bg_basis       = st.sidebar.radio("セル背景の基準", ["予測に連動", "補正に連動"], index=0)
    max_h_vh       = st.sidebar.slider("表の最大高さ (vh)", 40, 80, 65, step=5)
    recent_days    = st.sidebar.slider("表示日数（直近）", 7, 10, 8, step=1)
    outlier_th     = st.sidebar.slider("補正と予測の乖離で無効化（℃）", 0.0, 10.0, 7.0, step=0.5)
    hide_outlier_cells     = st.sidebar.checkbox("無効セルを非表示（週間）", value=False)
    hide_outlier_cells_day = st.sidebar.checkbox("無効セルを非表示（選択日）", value=False)

    df_dr = load_dr_single_file(base_dir, selected_file)
    if df_dr.empty:
        st.warning("DRデータが読み込めませんでした")
        st.stop()

    latest_dt = df_dr["datetime"].max()
    latest_day = latest_dt.date()
    depths_all = sorted([int(d) for d in df_dr["depth_m"].dropna().unique()])

    mode_view  = st.radio("表示期間", ["週間予測（表示値は昼頃）", "選択日（1時間毎）"])
    styles = get_calendar_css(max_h_vh)

    # 週間予測
    if mode_view == "週間予測（表示値は昼頃）":
        available_days = sorted(df_dr["date_day"].unique())
        min_day = min(available_days) if available_days else latest_day
        max_day = max(available_days) if available_days else latest_day
        selected_day = st.date_input("週間予測の基準日", value=max_day, min_value=min_day, max_value=max_day)

        start_day = pd.Timestamp(selected_day) - pd.Timedelta(days=recent_days - 1)
        end_day   = pd.Timestamp(selected_day)
        day_list  = list(pd.date_range(start_day, end_day, freq="D"))
        df_period = df_dr[df_dr["date_day"].isin([d.date() for d in day_list])].copy()

        reg_depthwise, n_match_reg = None, None
        if use_correction:
            train_start_dt = pd.Timestamp(selected_day) - pd.Timedelta(days=train_days)
            train_end_dt   = pd.Timestamp(selected_day) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            with st.spinner(
                f"回帰補正パラメータ算出中（{selected_file}, 終端={selected_day:%Y-%m-%d}, 遡り{train_days}日）..."
            ):
                reg_depthwise, n_match_reg = compute_depthwise_regression(
                    base_dir, selected_file, tolerance_min,
                    start_dt=train_start_dt, end_dt=train_end_dt, min_pairs=5
                )
        if use_correction and (reg_depthwise is None):
            st.warning("回帰係数の算出に失敗（一致データ不足など）。補正なしで表示します。")
            use_correction = False

        st.markdown(f"**{start_day:%m/%d}～{end_day:%m/%d}までの推移（期間の最初と最後の値）**")
        layers = make_layer_groups(depths_all)
        any_line = False
        for lname, ldepths in layers.items():
            line = summarize_weekly_layer_temp(
                lname, ldepths, df_period, df_dr, selected_day,
                use_correction=use_correction, reg_depthwise=reg_depthwise,
                stable_eps=0.4, outlier_th=outlier_th,
            )
            if line:
                any_line = True
                st.markdown(line)
        if not any_line:
            st.caption("（特筆すべき変化はありません）")

        depths_for_table = list(depths_all)
        if use_correction and (reg_depthwise is not None):
            depths_for_table = [d for d in depths_for_table if int(d) in reg_depthwise]
        if not depths_for_table:
            st.caption("（補正係数が算出できた水深がありませんでした）")

        table_html = build_weekly_table_html(
            df_period=df_period, day_list=day_list, depths_for_table=depths_for_table,
            use_correction=use_correction, reg_depthwise=reg_depthwise,
            bg_basis=bg_basis, hide_outlier_cells=hide_outlier_cells, outlier_th=outlier_th,
        )
        full_html = f"<!doctype html><html><head><meta charset='utf-8'>{styles}</head><body>{table_html}</body></html>"
        iframe_height = int(max(400, min(1100, max_h_vh * 10)))
        st_html(full_html, height=iframe_height, scrolling=True)

    # 選択日
    else:
        available_days = sorted(df_dr["date_day"].unique())
        min_day = min(available_days) if available_days else latest_day
        max_day = max(available_days) if available_days else latest_day
        selected_day = st.date_input("表示日（1時間毎）", value=max_day, min_value=min_day, max_value=max_day)

        df_day     = df_dr[df_dr["date_day"] == selected_day].copy()
        hours_list = sorted(df_day["datetime"].dt.floor("h").unique())
        times_hr   = [t.strftime('%H:%M') for t in hours_list]

        st.markdown("**朝(4～6時)、昼(11～13時)、夕(16～18時)**")
        layers = make_layer_groups(depths_all)
        any_line = False
        for lname, ldepths in layers.items():
            line = summarize_daily_layer_flow(lname, ldepths, df_day)
            if line:
                any_line = True
                st.markdown(line)
        if not any_line:
            st.caption("（特筆すべき変化はありません）")

        reg_depthwise, n_match_reg = None, None
        if use_correction:
            sel_train_end_dt   = pd.Timestamp(selected_day) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            sel_train_start_dt = pd.Timestamp(selected_day) - pd.Timedelta(days=train_days)
            with st.spinner(
                f"回帰補正パラメータ算出中（{selected_file}, 終端={selected_day:%Y-%m-%d}, 遡り{train_days}日）..."
            ):
                reg_depthwise_sel, n_match_reg_sel = compute_depthwise_regression(
                    base_dir, selected_file, tolerance_min,
                    start_dt=sel_train_start_dt, end_dt=sel_train_end_dt, min_pairs=5
                )
            if reg_depthwise_sel is None:
                st.warning("選択日終端での回帰係数算出に失敗（一致データ不足など）。補正なしで表示します。")
                use_correction = False
            else:
                reg_depthwise = reg_depthwise_sel
                n_match_reg   = n_match_reg_sel
                total_pairs = sum((n_match_reg or {}).values())
                st.caption(f"補正に使用したデータ数（選択日終端の一致ペア合計）: {total_pairs} 件")

        if use_correction and (reg_depthwise is not None):
            depths_for_table_sel = [d for d in depths_all if int(d) in reg_depthwise]
            if not depths_for_table_sel:
                st.caption("（補正係数が算出できた水深がありませんでした）")
        else:
            depths_for_table_sel = list(depths_all)

        table_html = build_daily_table_html(
            df_day=df_day, hours_list=hours_list, times_hr=times_hr, depths_for_table=depths_for_table_sel,
            use_correction=use_correction, reg_depthwise=reg_depthwise,
            bg_basis=bg_basis, hide_outlier_cells=hide_outlier_cells_day, outlier_th=outlier_th,
        )
        styles = get_calendar_css(max_h_vh)
        full_html = f"<!doctype html><html><head><meta charset='utf-8'>{styles}</head><body>{table_html}</body></html>"
        iframe_height = int(max(400, min(1100, max_h_vh * 10)))
        st_html(full_html, height=iframe_height, scrolling=True)

# -----------------------------------------
# 水温グラフ（コメントなし）
#  - 補正ON：実測がある水深のみ選択・描画（係数がある水深のみ補正線）
#  - 補正OFFでも OBS表示ON なら実測点を重ねて表示
#  - 任意CSVアップロード＆列選択で右軸・透明度0.65で重ね表示
#  - 「任意期間」選択時のスライダーをラジオ直下に表示（本修正点）
# -----------------------------------------
elif view_mode == "水温グラフ":
    # --- DRファイル選択など基本UI ---
    parent_folder_dr = pjoin(base_dir, "pred")
    if not os.path.exists(parent_folder_dr):
        st.error(f"フォルダが見つかりません: {parent_folder_dr}")
        st.stop()
    dr_files = [f for f in os.listdir(parent_folder_dr) if f.endswith(".csv")]
    if not dr_files:
        st.warning("pred に CSV がありません"); st.stop()

    selected_file = st.selectbox("解析対象ファイルを選択", sorted(dr_files), key="sel_dr_file")
    use_correction = st.sidebar.checkbox("実測ベース補正（この版では線のみ表示）", value=False, key="corr_on")
    tolerance_min  = st.sidebar.slider("時刻差の許容範囲（分）", 5, 120, 35, step=5, key="tol_min")

    # --- 表示期間ラジオ & 任意期間スライダー（直下） ---
    period_choice = st.sidebar.radio("表示期間", ["直近2週間", "任意期間"], index=0, key="period_choice")

    # DR 読み込み（期間算出に必要なため先に行う）
    df_dr = load_dr_single_file(base_dir, selected_file)
    if df_dr.empty:
        st.warning("DRデータが読み込めませんでした"); st.stop()

    latest_dt_dr = df_dr["datetime"].max()
    available_days = sorted(df_dr["date_day"].unique()) if "date_day" in df_dr.columns else []
    if available_days:
        min_day = min(available_days)
        max_day = max(available_days)
    else:
        min_day = latest_dt_dr.date()
        max_day = latest_dt_dr.date()

    if period_choice == "直近2週間":
        end_day   = latest_dt_dr.date()
        start_day = (latest_dt_dr - pd.Timedelta(days=13)).date()
        title_suffix = "（直近2週間・時間別）"
        # 補正の学習期間（直近2週間モード用）
        train_days = st.sidebar.slider("補正学習期間（日数）", 7, 90, 30, step=1, key="train_days_recent")
        train_start_dt = pd.Timestamp(end_day) - pd.Timedelta(days=train_days)
        train_end_dt   = pd.Timestamp(end_day) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        start_default = max(min_day, max_day - pd.Timedelta(days=13))
        start_day, end_day = st.sidebar.slider(
            "期間（任意・バーで指定）",
            min_value=min_day, max_value=max_day, value=(start_default, max_day),
            help="開始日〜終了日をバーで調整してください",
            key="custom_period_slider"
        )
        title_suffix   = f"（{start_day:%Y-%m-%d}〜{end_day:%Y-%m-%d}・時間別）"
        train_start_dt = pd.Timestamp(start_day)
        train_end_dt   = pd.Timestamp(end_day) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        st.sidebar.caption("※ 補正の学習は『選択した任意期間全体』で実施します。")

    # --- この下に任意CSVやサイズCSVのエクスパンダ（ラジオ直下にスライダーが来るよう並べ替え） ---
    with st.sidebar.expander("任意データの追加（csv）", expanded=False):
        uploaded_generic = st.file_uploader("CSVを追加", type=["csv"], accept_multiple_files=True, key="gen_csv")

    with st.sidebar.expander("任意列の表示方法", expanded=False):
        generic_plot_type = st.selectbox(
            "表示タイプ（任意列）",
            ["日別棒(平均)", "日別箱ひげ", "日別バイオリン", "時間別線(1H)"],
            index=0, key="generic_plot_type"
        )

    # --- サイズCSV（2行ヘッダー：1行目=サイズ、2行目=系列） ---
    def load_size_csv_with_two_headers(file) -> pd.DataFrame:
        """
        1行目: [Date, 0.5, 1.0, 2.0, ...] ← サイズ値（数値/文字）
        2行目: [Date, w_temp, other, ...] ← 系列名
        出力: columns=[datetime, size, series, value, date_day]
        """
        df_raw = pd.read_csv(file, header=[0, 1], encoding="utf-8")
        df_raw.columns = pd.MultiIndex.from_tuples([
            (str(a).strip() if a is not None else "", str(b).strip() if b is not None else "")
            for a, b in df_raw.columns.to_list()
        ])
        # Date 列の検出
        date_col = None
        for a, b in df_raw.columns:
            if (str(a).lower() == "date") or (str(b).lower() == "date"):
                date_col = (a, b); break
        if date_col is None:
            raise ValueError("Date 列が見つかりません（ヘッダーのどちらかに 'Date' を含めてください）")

        # Date → JST naive
        dt = pd.to_datetime(df_raw[date_col], errors="coerce", utc=False)
        try:
            if getattr(dt.dt, "tz", None) is not None:
                dt = dt.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
        except Exception:
            pass

        def _parse_size(x):
            try:   return float(x)
            except: return str(x)

        rec = []
        for (top, bottom) in df_raw.columns:
            if (top, bottom) == date_col: continue
            size   = _parse_size(top)
            series = str(bottom)
            vals   = pd.to_numeric(df_raw[(top, bottom)], errors="coerce")
            for i in range(len(df_raw)):
                v = vals.iloc[i]
                if pd.isna(v): continue
                rec.append({"datetime": dt.iloc[i], "size": size, "series": series, "value": float(v)})
        out = pd.DataFrame.from_records(rec)
        if out.empty: return out
        out["date_day"] = out["datetime"].dt.date
        return out

    def add_size_horizontal_bars_binned20(
        fig,
        df_long: pd.DataFrame,
        start_day,
        end_day,
        series_filter: str = "",
        bar_max_fraction: float = 1.0,  # 1日幅に対する最大長（比）
        vmax_full_day: float = 1.0,     # 期間全体スケール
        max_span_days: int = 2,         # 最大スパン（日）
        row: int = 2,
        col: int = 1
    ):
        SIZE_BIN = 20.0
        # 期間・series フィルタ
        sub = df_long[(df_long["date_day"] >= start_day) & (df_long["date_day"] <= end_day)].copy()
        if series_filter.strip():
            sub = sub[sub["series"].astype(str) == series_filter.strip()]
        if sub.empty: return

        # サイズ→20刻みビン
        sub["size_num"] = pd.to_numeric(sub["size"], errors="coerce")
        sub = sub.dropna(subset=["size_num"]).copy()
        sub["size_bin"] = np.floor(sub["size_num"] / SIZE_BIN) * SIZE_BIN

        # 日付×サイズビンごと合算
        agg = (sub.groupby(["date_day", "size_bin"], as_index=False)["value"]
                  .sum().rename(columns={"value": "value_sum"}))

        # 下段Y軸（サイズ帯）
        bins_sorted = np.sort(agg["size_bin"].unique())
        def _fmt_label(x: float) -> str:
            return f"{int(x)}" if float(x).is_integer() else f"{x:g}"
        fig.update_yaxes(
            row=row, col=col, secondary_y=False,
            type="linear",
            tickmode="array",
            tickvals=list(bins_sorted),
            ticktext=[_fmt_label(v) for v in bins_sorted],
            title_text="殻長（20㎛刻み）"
        )

        # 合算値→横棒の長さ
        one_day = pd.Timedelta(days=1)
        max_span_days = max(1, int(max_span_days))
        max_len = float(bar_max_fraction) * one_day * max_span_days
        vmax = max(float(vmax_full_day), 1e-9)  # 0除算回避
        agg["length"] = (pd.to_numeric(agg["value_sum"], errors="coerce") / vmax) * max_len
        agg["length"] = agg["length"].clip(lower=pd.Timedelta(0), upper=max_len)
        agg["x_start"] = pd.to_datetime(agg["date_day"])  # 左揃え
        agg["x_end"]   = agg["x_start"] + agg["length"]

        # 着色（200以上＝赤、それ未満＝灰）
        fill_gray = "rgba(160,160,160,0.65)"
        fill_red  = "rgba(200,60,60,0.65)"
        border_black = "rgba(0,0,0,1.0)"

        for _, r in agg.iterrows():
            x0 = r["x_start"]; x1 = r["x_end"]
            b  = float(r["size_bin"])
            y0 = b; y1 = b + SIZE_BIN
            fillcolor = fill_red if b >= 200.0 else fill_gray
            fig.add_shape(
                type="rect",
                x0=x0, x1=x1, y0=y0, y1=y1,
                layer="below",
                row=row, col=col,
                line=dict(color=border_black, width=0.8),
                fillcolor=fillcolor
            )
            # Hover用に不可視マーカー
            vtxt = f"{float(r['value_sum']):.3f}"
            fig.add_trace(go.Scatter(
                x=[x1], y=[(y0 + y1)/2.0],
                mode="markers",
                marker=dict(size=1, color="rgba(0,0,0,0)"),
                showlegend=False,
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>size={_fmt_label(b)}–{_fmt_label(b + SIZE_BIN)}<br>合計値: {vtxt}<extra></extra>"
            ), row=row, col=col, secondary_y=False)

    # --- サイズデータ用エクスパンダ ---
    with st.sidebar.expander("幼生データの追加（csv）", expanded=True):
        uploaded_size_files = st.file_uploader("CSVを追加", type=["csv"], accept_multiple_files=True, key="size_csv")
        size_series_filter  = st.text_input("seriesフィルタ（空=全て、例：w_temp）", value="", key="size_series_filter")
        bar_max_fraction    = st.slider("横棒の最大長（1日比）", 0.1, 1.0, 1.0, step=0.05, key="bar_max_fraction")
        max_span_days       = st.slider("横棒の最大スパン（日数）", 1, 3, 2, step=1, key="max_span_days")

    # --- DR：期間抽出＆1H整形 ---
    df_period = df_dr[(df_dr["date_day"] >= start_day) & (df_dr["date_day"] <= end_day)].copy()
    df_period = df_period.sort_values("datetime")
    if "pred_temp" in df_period.columns and not df_period.empty:
        df_period = (df_period.groupby(["depth_m", "datetime"], as_index=False).agg({"pred_temp": "median"}))
    if not df_period.empty:
        df_period = (
            df_period.sort_values("datetime")
                     .groupby("depth_m", group_keys=False)
                     .apply(lambda g: (
                         g.drop(columns=["depth_m"])
                          .set_index("datetime")
                          .resample("1H").median(numeric_only=True)
                          .interpolate(method="time", limit=2)
                          .reset_index()
                          .assign(depth_m=int(g["depth_m"].iloc[0]))
                     ))
        )
    if "depth_m" in df_period.columns:
        df_period["depth_m"] = pd.to_numeric(df_period["depth_m"], errors="coerce").round(0).astype("Int64")

    # --- OBS 読み込み（補正ON または 実測表示ONのとき） ---
    parent_folder_obs = pjoin(base_dir, "obs")
    show_obs_points   = st.sidebar.checkbox("実測（OBS）を上段に重ねて表示", value=True, key="show_obs")
    df_obs_period = pd.DataFrame()
    if show_obs_points:
        obs_path = pjoin(parent_folder_obs, selected_file)
        if os.path.exists(obs_path):
            try:
                df_obs = pd.read_csv(obs_path)
                df_obs["datetime"] = jst_to_naive(df_obs.get("Date"))
                df_obs["depth_m"]  = pd.to_numeric(df_obs.get("Depth"), errors="coerce").round(0).astype("Int64")
                df_obs = df_obs.rename(columns={"Temp":"obs_temp"})
                df_obs = df_obs.dropna(subset=["datetime","depth_m"]).copy()
                df_obs["date_day"] = df_obs["datetime"].dt.date
                df_obs_period = df_obs[(df_obs["date_day"] >= start_day) & (df_obs["date_day"] <= end_day)].copy()
            except Exception as e:
                st.warning(f"OBSの読み込みに失敗しました: {obs_path} ({e})")
                df_obs_period = pd.DataFrame()
        else:
            st.info("obs フォルダに同名CSVがありません。実測点は表示されません。")

    # --- 実測点重ね用 asof マージ ---
    merged_for_points = pd.DataFrame()
    if show_obs_points and (not df_period.empty) and (not df_obs_period.empty):
        tol = pd.Timedelta(minutes=int(tolerance_min))
        left  = df_period.sort_values(["depth_m","datetime"]).copy()
        right = df_obs_period.sort_values(["depth_m","datetime"])[["datetime","depth_m","obs_temp"]].copy()
        left["depth_m"]  = pd.to_numeric(left["depth_m"],  errors="coerce").round(0).astype("Int64")
        right["depth_m"] = pd.to_numeric(right["depth_m"], errors="coerce").round(0).astype("Int64")
        left  = left.dropna(subset=["depth_m"])
        right = right.dropna(subset=["depth_m"])
        merged_for_points = safe_merge_asof_by_depth(
            left=left, right=right, tolerance=tol, right_value_cols=["obs_temp"], suffixes=("","")
        ).dropna(subset=["obs_temp"])

    # --- 表示水深の候補・選択 ---
    depths_plot = sorted(set(df_period["depth_m"].dropna().astype(int).tolist())) if not df_period.empty else []
    default_depths = depths_plot[:min(3, len(depths_plot))]
    selected_depths = st.multiselect("表示する水深（複数選択可）", depths_plot, default=default_depths, key="plot_depths")

    # --- 回帰係数の学習（補正ON時のみ） ---
    reg_depthwise = None
    if use_correction:
        with st.spinner(f"水深別回帰係数を学習中（{selected_file}）..."):
            reg_depthwise, _ = compute_depthwise_regression(
                base_dir, selected_file, tolerance_min,
                start_dt=train_start_dt, end_dt=train_end_dt, min_pairs=5
            )
        if reg_depthwise is None:
            st.warning("回帰係数の算出に失敗（一致ペア不足など）。未補正で表示します。")
            use_correction = False

    # --- グラフ作成（2段：上段=温度＆任意列、下段=横棒） ---
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        row_heights=[0.6, 0.4]
    )
    has_size_data = False

    # ==== 下段：サイズ横棒 ====
    if uploaded_size_files:
        size_long_list = []
        for f in uploaded_size_files:
            try:
                df_long = load_size_csv_with_two_headers(f)
            except Exception as e:
                st.warning(f"{getattr(f,'name','uploaded')} の読み込みに失敗: {e}")
                continue
            df_long = df_long[(df_long["date_day"] >= start_day) & (df_long["date_day"] <= end_day)].copy()
            if size_series_filter.strip():
                df_long = df_long[df_long["series"].astype(str) == size_series_filter.strip()]
            if df_long.empty:
                continue
            size_long_list.append(df_long)

        if size_long_list:
            df_sz_all = pd.concat(size_long_list, ignore_index=True)
            has_size_data = not df_sz_all.empty

            # 期間内の20刻み合算で最大値（デフォルトVmax）
            tmp = df_sz_all.copy()
            tmp["size_num"] = pd.to_numeric(tmp["size"], errors="coerce")
            tmp = tmp.dropna(subset=["size_num"]).copy()
            SIZE_BIN = 20.0
            tmp["size_bin"] = np.floor(tmp["size_num"] / SIZE_BIN) * SIZE_BIN
            default_vmax = float(tmp.groupby(["date_day","size_bin"], as_index=False)["value"].sum()["value"].max())
            vmax_full_day = st.sidebar.number_input(
                "横棒スケール最大値（期間全体・20刻み合算での最大長）—この値で1日幅×最大スパン",
                min_value=0.0001, value=default_vmax if default_vmax > 0 else 1.0, step=0.1, key="vmax_full_day"
            )

            add_size_horizontal_bars_binned20(
                fig, df_sz_all, start_day, end_day,
                series_filter=size_series_filter.strip(),
                bar_max_fraction=float(bar_max_fraction),
                vmax_full_day=float(vmax_full_day),
                max_span_days=int(max_span_days),
                row=2, col=1
            )
        else:
            st.info("期間内にサイズデータがありません。")

    # ==== 上段：水温・OBS・任意列 ====
    base_colors   = px.colors.qualitative.Dark24
    obs_colors    = px.colors.qualitative.Safe
    marker_symbols= ["circle","square","diamond","triangle-up","triangle-down","x","cross","star","hexagon"]
    color_map     = {int(d): base_colors[i % len(base_colors)] for i, d in enumerate(selected_depths)}
    obs_color_map = {int(d): obs_colors [i % len(obs_colors )] for i, d in enumerate(selected_depths)}
    symbol_map    = {int(d): marker_symbols[i % len(marker_symbols)] for i, d in enumerate(selected_depths)}

    if not df_period.empty and selected_depths:
        for d in selected_depths:
            g = df_period[df_period["depth_m"] == d]
            if g.empty: continue
            x = g["datetime"]; y_raw = g["pred_temp"].astype(float)
            # 予測線
            fig.add_trace(go.Scatter(
                x=x, y=y_raw, mode="lines", name=f"{d}m 予測",
                line=dict(color=color_map[int(d)], width=2),
                hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>水温: %{y:.2f} °C"
            ), row=1, col=1, secondary_y=False)

            # 補正線
            if use_correction and (reg_depthwise is not None) and (int(d) in reg_depthwise):
                alpha, beta = reg_depthwise[int(d)]
                y_corr = (alpha + beta * y_raw).clip(lower=TEMP_MIN, upper=TEMP_MAX)
                fig.add_trace(go.Scatter(
                    x=x, y=y_corr, mode="lines", name=f"{d}m 回帰補正",
                    line=dict(color="orange", dash="dot", width=2),
                    hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>補正水温: %{y:.2f} °C<extra></extra>"
                ), row=1, col=1, secondary_y=False)

    # 実測点
    if show_obs_points and not merged_for_points.empty and selected_depths:
        for d in selected_depths:
            sub_obs = merged_for_points[merged_for_points["depth_m"] == d]
            if sub_obs.empty: continue
            fig.add_trace(go.Scatter(
                x=sub_obs["datetime"], y=sub_obs["obs_temp"], mode="markers",
                name=f"{d}m 実測",
                marker=dict(size=6, color=obs_color_map[int(d)], symbol=symbol_map[int(d)],
                            line=dict(color="black", width=0.1)),
                opacity=0.70,
                hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>実測水温: %{y:.2f} °C<extra></extra>"
            ), row=1, col=1, secondary_y=False)

    # 任意列（右軸）
    colors = px.colors.qualitative.Dark24
    color_idx = 0
    if uploaded_generic:
        st.markdown("#### 任意列（上段右軸）を選択")
        gen_list: List[pd.DataFrame] = []
        for f in uploaded_generic:
            g = normalize_generic_uploaded_csv(f)
            if not g.empty: gen_list.append(g)

        gen_all: List[Dict[str, object]] = []
        for g in gen_list:
            g_period = g[
                (g["datetime"] >= pd.to_datetime(start_day)) &
                (g["datetime"] <= pd.to_datetime(end_day) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
            ].copy()
            if g_period.empty: continue
            value_cols = [
                c for c in g_period.columns
                if c not in ["Date","datetime","Depth","depth_m","source"]
                and pd.api.types.is_numeric_dtype(g_period[c])
            ]
            if not value_cols: continue
            if "depth_m" not in g_period.columns:
                g_period = g_period.assign(depth_m=0)
            g_res = g_period.sort_values("datetime")
            gen_all.append({"df": g_res, "source": g_res["source"].iloc[0], "value_cols": value_cols})

        for item in gen_all:
            g_res = item["df"]; src = item["source"]
            candidates = [c for c in item["value_cols"] if c in g_res.columns]
            if not candidates: continue
            with st.expander(f"{src} の列選択", expanded=False):
                cols = st.multiselect("描画列（複数可）", candidates, default=candidates[:1], key=f"gen_cols_{src}")
            if not cols: continue

            g_res = g_res.copy()
            g_res["date_day"] = g_res["datetime"].dt.date
            for col in cols:
                color = colors[color_idx % len(colors)]; color_idx += 1
                if generic_plot_type == "日別棒(平均)":
                    daily = (g_res.groupby("date_day", as_index=False)[col].mean().dropna(subset=[col]))
                    if daily.empty: continue
                    fig.add_trace(go.Bar(
                        x=daily["date_day"], y=daily[col], name=f"{src} {col}",
                        marker=dict(color=color, line=dict(color="rgba(0,0,0,0.35)", width=0.5)),
                        opacity=0.65,
                        hovertemplate="%{x}<br>" + f"{col}: " + "%{y:.3f}<extra></extra>"
                    ), row=1, col=1, secondary_y=True)

                elif generic_plot_type == "日別箱ひげ":
                    if g_res[col].notna().any():
                        fig.add_trace(go.Box(
                            x=g_res["date_day"], y=g_res[col],
                            name=f"{src} {col}",
                            line=dict(color=color), fillcolor=color,
                            boxmean=True, opacity=0.65,
                            hovertemplate="%{x}<br>" + f"{col}: " + "%{y:.3f}<extra></extra>"
                        ), row=1, col=1, secondary_y=True)
                        fig.update_layout(boxmode="group")

                elif generic_plot_type == "日別バイオリン":
                    if g_res[col].notna().any():
                        fig.add_trace(go.Violin(
                            x=g_res["date_day"], y=g_res[col],
                            name=f"{src} {col}",
                            line_color=color, fillcolor=color,
                            opacity=0.65, meanline_visible=True, points="outliers",
                            hovertemplate="%{x}<br>" + f"{col}: " + "%{y:.3f}<extra></extra>"
                        ), row=1, col=1, secondary_y=True)
                        fig.update_layout(violinmode="group")

                elif generic_plot_type == "時間別線(1H)":
                    sub_line = (g_res.sort_values("datetime").set_index("datetime")[col]
                               .resample("1H").mean().dropna().reset_index())
                    if sub_line.empty: continue
                    fig.add_trace(go.Scatter(
                        x=sub_line["datetime"], y=sub_line[col], mode="lines",
                        name=f"{src} {col}", line=dict(color=color, width=2), opacity=0.65,
                        hovertemplate="%{x}<br>" + f"{col}: " + "%{y:.3f}<extra></extra>"
                    ), row=1, col=1, secondary_y=True)

    # --- レイアウト ---
    show_legend = st.checkbox("凡例を表示", value=True, key="show_legend_temp")
    legend_cfg = dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1, font=dict(size=12), itemsizing="constant")
    fig.update_layout(
        title={"text": f"{selected_file} 水温{title_suffix}", "y": 0.98, "x": 0.01, "xanchor": "left", "font": {"size": 16}, "pad": {"t": 8}},
        margin=dict(l=10, r=10, t=50, b=10),
        height=700, template="plotly_white",
        showlegend=bool(show_legend), legend=legend_cfg if show_legend else dict()
    )
    # X軸：サイズの有無で表示段切替
    x_range = [pd.Timestamp(start_day), pd.Timestamp(end_day) + pd.Timedelta(days=1)]
    fig.update_xaxes(type="date", range=x_range)
    if has_size_data:
        fig.update_xaxes(row=2, col=1, showticklabels=True, ticks="outside", title_text="日時（JST）")
        fig.update_xaxes(row=1, col=1, showticklabels=False)
    else:
        fig.update_xaxes(row=1, col=1, showticklabels=True, ticks="outside", title_text="日時（JST）")
        fig.update_xaxes(row=2, col=1, showticklabels=False)

    fig.update_xaxes(tickfont=dict(size=11))
    fig.update_yaxes(title_text="水温 (℃)", secondary_y=False, tickfont=dict(size=11), row=1, col=1)
    fig.update_yaxes(title_text="任意列",  secondary_y=True,  tickfont=dict(size=11), row=1, col=1)

    # 描画
    st.plotly_chart(fig, use_container_width=True)