import os
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

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

# --- コメント用 定数 ---
LOOKBACK_DAYS = 7     # 週間コメントの基準期間
HIGH_TEMP_TH = 22.0   # 高水温注意
DAY_BINS = [          # 選択日コメントの時間帯
    ("朝", 4, 6),
    ("昼", 11, 13),
    ("夕", 16, 18),
]

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
        if np.isnan(s):
            return 18, "#CCCCCC"
        speed_kt = s * 1.94384
        if speed_kt < 1.0:
            return 18, "#0000FF"
        elif speed_kt < 2.0:
            return 22, "#FFC107"
        else:
            return 26, "#FF0000"
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
    if pd.isna(temp):
        return "rgba(220,220,220,0.4)"
    ratio = (float(temp) - t_min) / (t_max - t_min)
    ratio = max(0, min(1, ratio))
    if ratio < 0.5:
        r = int(240 * ratio * 2)
        g = int(240 * ratio * 2)
        b = 240
    else:
        r = 240
        g = int(240 * (1 - (ratio - 0.5) * 2))
        b = int(240 * (1 - (ratio - 0.5) * 2))
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
        set(left["depth_m"].dropna().unique()).intersection(
            set(right["depth_m"].dropna().unique())
        )
    )
    for d in common_depths:
        l = left[left["depth_m"] == d].sort_values("datetime")
        r = right[right["depth_m"] == d].sort_values("datetime")[
            ["datetime", "depth_m"] + right_value_cols
        ]
        if l.empty or r.empty:
            continue
        merged = pd.merge_asof(
            l, r, on="datetime", by="depth_m", tolerance=tolerance,
            direction="nearest", suffixes=suffixes
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
    obs = obs.rename(columns={"Temp": "obs_temp"})
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
    pred_sorted = pred.sort_values(["depth_m", "datetime"]).copy()
    obs_sorted  = obs.sort_values(["depth_m", "datetime"]).copy()
    merged = safe_merge_asof_by_depth(
        pred_sorted, obs_sorted, tol, right_value_cols=["obs_temp"], suffixes=("", "")
    )
    pair = merged.dropna(subset=["pred_temp", "obs_temp", "depth_m"]).copy()
    if pair.empty:
        return None, None
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
            n_depth if n_depth else None)

# 8方位（度→方位）
def dir_to_8pt_jp(deg: float) -> str:
    if pd.isna(deg):
        return ""
    dirs = ["北", "北東", "東", "南東", "南", "南西", "西", "北西"]
    idx = int(((float(deg) + 22.5) % 360) // 45)
    return dirs[idx]

# 流速クラス（m/s → 穏 / やや速 / 速い）
def speed_class_from_mps(v_mps: Optional[float]) -> str:
    if v_mps is None or pd.isna(v_mps):
        return ""
    kt = float(v_mps) * 1.94384
    if kt >= 1.5:
        return "速"
    if kt >= 0.8:
        return "やや速"
    return "穏"

# 回帰なしでも使える簡易トレンド（℃/日）
def slope_c_per_day(ts: pd.Series, idx: pd.Series) -> float:
    if ts.empty or ts.isna().all():
        return np.nan
    x = pd.to_datetime(idx).astype("int64") / 1e9  # 秒
    x = (x - x.min()) / 86400.0                   # 日
    y = ts.astype(float)
    if len(y) < 3:
        return np.nan
    var = np.var(x)
    if var < 1e-12:
        return np.nan
    cov = np.cov(x, y, ddof=0)[0, 1]
    return float(cov / var)

# 水深を3層へ（表層・中層・底層）
def make_layer_groups(depths: List[int]) -> Dict[str, List[int]]:
    if not depths:
        return {"表層": [], "中層": [], "底層": []}
    d_sorted = sorted(depths)  # 浅→深
    n = len(d_sorted)
    if n <= 3:
        top = d_sorted[:1]
        mid = d_sorted[1:2] if n >= 2 else []
        bot = d_sorted[2:] if n >= 3 else (d_sorted[-1:] if n >= 1 else [])
    elif n == 4:
        top = d_sorted[:2]; mid = d_sorted[2:3]; bot = d_sorted[3:]
    elif n == 5:
        top = d_sorted[:2]; mid = d_sorted[2:3]; bot = d_sorted[3:]
    else:
        top = d_sorted[:2]; bot = d_sorted[-2:]
        mid = [d for d in d_sorted if d not in top + bot]
        if len(mid) >= 3:
            c = len(mid) // 2
            mid = mid[c-1:c+1]  # 中央2
    return {"表層": top, "中層": mid, "底層": bot}

# 週間コメント（昼ごろ・水温のみ）
def summarize_weekly_layer_temp(
    layer_name: str,
    layer_depths: List[int],
    df_period: pd.DataFrame,  # 表示期間（selected_day を含む直近 days）
    df_all: pd.DataFrame,     # 未使用（将来拡張用に残す）
    selected_day,             # date or pd.Timestamp
    use_correction: bool = False,
    reg_depthwise: Optional[Dict[int, Tuple[float, float]]] = None,
    stable_eps: float = 0.2,  # 安定判定の閾値（℃）
    outlier_th: Optional[float] = None,  # 例：7.0
) -> Optional[str]:
    if not layer_depths or df_period is None or df_period.empty:
        return None
    PHYS_MIN, PHYS_MAX = -1.5, 35.0
    msgs: List[str] = []
    for depth in layer_depths:
        g = df_period[df_period["depth_m"] == depth].copy()
        if g.empty or "pred_temp" not in g.columns:
            continue
        g = g.sort_values("datetime")
        temps_pred = pd.to_numeric(g["pred_temp"], errors="coerce")
        if use_correction:
            if not (reg_depthwise and int(depth) in reg_depthwise):
                continue
            alpha, beta = reg_depthwise[int(depth)]
            temps_corr_raw = alpha + beta * temps_pred
            temps_corr_clipped = np.clip(temps_corr_raw, TEMP_MIN, TEMP_MAX)
            mask_valid = (
                pd.notna(temps_pred) &
                pd.notna(temps_corr_raw) &
                (temps_corr_raw > PHYS_MIN) &
                (temps_corr_raw < PHYS_MAX) &
                (temps_corr_clipped > TEMP_MIN) &
                (temps_corr_clipped < TEMP_MAX)
            )
            if outlier_th is not None:
                diff = (temps_corr_raw - temps_pred).abs()
                mask_valid &= (diff < float(outlier_th))
            temps = pd.Series(temps_corr_raw)[mask_valid]
        else:
            temps = temps_pred[
                pd.notna(temps_pred) &
                (temps_pred > PHYS_MIN) &
                (temps_pred < PHYS_MAX)
            ]
        temps = pd.to_numeric(temps, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        n = int(len(temps))
        if n == 0:
            continue
        t_min = float(temps.min())
        t_max = float(temps.max())
        if t_max >= HIGH_TEMP_TH:
            tag = f"高水温注意（{t_min:.1f}℃～{t_max:.1f}℃）"
        else:
            if n >= 2:
                t_start = float(temps.iloc[0])
                t_end = float(temps.iloc[-1])
                delta = t_end - t_start
                if delta > stable_eps:
                    tag = f"上昇（{t_start:.1f}℃→{t_end:.1f}℃）"
                elif delta < -stable_eps:
                    tag = f"下降（{t_start:.1f}℃→{t_end:.1f}℃）"
                else:
                    tag = f"安定（{t_start:.1f}℃）"
            else:
                t = float(temps.iloc[0])
                tag = f"安定（{t:.1f}℃）"
        msgs.append(f"{depth}m{tag}")
    if not msgs:
        return None
    return f"**{layer_name}**： " + "／".join(msgs)

# 選択日コメント（レンジ非表示：各時間帯は「朝（北・穏）」の形で出す）

def summarize_daily_layer_flow(
    layer_name: str,
    layer_depths: List[int],
    df_day: pd.DataFrame,
    use_short_labels: bool = True,    # 速度の短縮表記（穏/中/速）
    merge_same_segments: bool = False # Trueなら「朝・日中（北東・穏）」のように結合
) -> Optional[str]:
    """
    出力例：
    **表層**： 朝（北・穏）／ 日中（北東・穏）／ 夕（北東・穏）
    - 方向は U・V のベクトル平均から再計算（単純な度数平均は使わない）。
    - 速度は平均ベクトルの大きさからクラス分け。
    """
    if not layer_depths:
        return None

    order = {"朝": 0, "昼": 1, "夕": 2}
    rows: List[Tuple[str, str, str]] = []  # (label, dir_txt, speed_txt)

    for label, h0, h1 in DAY_BINS:
        g = df_day[
            (df_day["depth_m"].isin(layer_depths)) &
            (df_day["datetime"].dt.hour.between(h0, h1))
        ]
        if g.empty:
            continue

        # --- ベクトル平均（推奨） ---
        U_mean = g["U"].mean() if "U" in g.columns else np.nan
        V_mean = g["V"].mean() if "V" in g.columns else np.nan

        if pd.notna(U_mean) and pd.notna(V_mean):
            speed_mean = float(np.sqrt(U_mean**2 + V_mean**2))
            # 既存の方位計算に合わせる（Uをx、Vをyにした atan2）
            dir_deg_mean = (np.degrees(np.arctan2(U_mean, V_mean)) + 360.0) % 360.0
        else:
            # U/V が無い場合のフォールバック：円平均
            D = g["Direction_deg"].dropna() if "Direction_deg" in g.columns else pd.Series(dtype=float)
            if D.empty:
                continue
            rad = np.deg2rad(D.values)
            C = np.cos(rad).mean()
            S = np.sin(rad).mean()
            dir_deg_mean = (np.degrees(np.arctan2(S, C)) + 360.0) % 360.0
            speed_mean = g["Speed"].mean() if "Speed" in g.columns else np.nan

        d_txt = dir_to_8pt_jp(dir_deg_mean) if pd.notna(dir_deg_mean) else ""
        v_cls = speed_class_from_mps(speed_mean) if pd.notna(speed_mean) else ""

        # 速度は短縮表記（穏/中/速）
        if use_short_labels and v_cls:
            v_map = {"穏やか": "穏", "中程度": "やや速", "速い": "速"}
            v_cls = v_map.get(v_cls, v_cls)

        if d_txt or v_cls:
            rows.append((label, d_txt, v_cls))

    if not rows:
        return None

    segments: List[str] = []

    if merge_same_segments:
        bucket: Dict[Tuple[str, str], List[str]] = {}
        for lbl, d, v in rows:
            bucket.setdefault((d, v), []).append(lbl)
        for (d, v), lbls in bucket.items():
            lbls_sorted = sorted(lbls, key=lambda x: order.get(x, 99))
            inner = "・".join([x for x in [d, v] if x])  # "北東・穏" / "北東" / "穏"
            segments.append(f"{'・'.join(lbls_sorted)}（{inner}）")
    else:
        rows_sorted = sorted(rows, key=lambda r: order.get(r[0], 99))
        for lbl, d, v in rows_sorted:
            inner = "・".join([x for x in [d, v] if x])
            segments.append(f"{lbl}（{inner}）")

    return f"**{layer_name}**： " + "／".join(segments)

# =========================================
# HTML/CSS レンダラ（共通化）
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
        bg_color = get_color(float(temp)) if (temp is not None and not pd.isna(temp)) \
                   else "rgba(220,220,220,0.6)"

    pred_label = f"{float(temp):.1f}°C" if (temp is not None and not pd.isna(temp)) else "NaN"
    pred_html = f"<span class='pred-small'>{pred_label}</span>" if use_correction \
                else f"<span>{pred_label}</span>"

    speed_html, arrow_html = "", ""
    if (speed_mps is not None and not pd.isna(speed_mps)) and (dir_deg is not None and not pd.isna(dir_deg)):
        speed_kt = float(speed_mps) * 1.94384
        speed_html = f"<span style='font-size:12px;color:#444;'>{speed_kt:.1f} kt</span>"
        arrow_html = f"<span style='display:block;line-height:1;margin:0;padding:0;'>{get_arrow_svg(float(dir_deg), float(speed_mps))}</span>"

    corr_html = ""
    if use_correction and (corr_temp_raw is not None):
        corr_html = f"<span style='color:#D32F2F;font-weight:700;font-size:14px;margin:0;padding:0;'>{corr_temp_raw:.1f}°C</span>"

    cell_content = (
        "<div style='display:flex;flex-direction:column;align-items:center;gap:2px;margin:0;padding:0;'>"
        + pred_html + speed_html + arrow_html + corr_html +
        "</div>"
    )
    return f"<td style='background:{bg_color}'>{cell_content}</td>"

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
        '<div class="calendar-scroll-container">'
        '<table class="calendar-table">'
        "<thead><tr><th>水深</th>"
        + "".join([f"<th>{t}</th>" for t in times])
        + "</tr></thead><tbody>"
    )
    for depth in depths_for_table:
        html += f"<tr><td class='depth-cell'>{depth}m</td>"
        for day in day_list:
            g = df_period[
                (df_period["date_day"] == day.date()) &
                (df_period["depth_m"] == depth)
            ]
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

                is_invalid = False
                if use_correction and (corr_raw is not None):
                    if (corr_raw <= PHYS_MIN) or (corr_raw >= PHYS_MAX):
                        is_invalid = True
                    if not is_invalid:
                        if (corr_clip <= TEMP_MIN) or (corr_clip >= TEMP_MAX):
                            is_invalid = True
                    if not is_invalid and not pd.isna(temp):
                        if abs(float(corr_raw) - float(temp)) >= float(outlier_th):
                            is_invalid = True

                html += render_cell_html(
                    temp=temp,
                    speed_mps=speed_val,
                    dir_deg=dir_val,
                    use_correction=use_correction,
                    corr_temp_raw=corr_raw,
                    corr_temp_clip=corr_clip,
                    bg_basis=bg_basis,
                    hide_outlier_cells=hide_outlier_cells,
                    is_invalid=is_invalid,
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
    """
    選択日（1時間毎）のテーブルでも、週間と同じ無効判定を適用する。
    ・物理レンジ外（PHYS_MIN/PHYS_MAX）／クリップ端（TEMP_MIN/TEMP_MAX）／乖離 outlier_th 以上
    ・hide_outlier_cells=True の場合は空セルにする、False の場合はグレー背景に '-'
    """
    PHYS_MIN, PHYS_MAX = -1.5, 35.0
    html = (
        '<div class="calendar-scroll-container">'
        '<table class="calendar-table">'
        "<thead><tr><th>水深</th>"
        + "".join([f"<th>{t}</th>" for t in times_hr])
        + "</tr></thead><tbody>"
    )
    for depth in depths_for_table:
        html += f"<tr><td class='depth-cell'>{depth}m</td>"
        for t_obj in hours_list:
            row = df_day[
                (df_day["datetime"].dt.floor("h") == t_obj) &
                (df_day["depth_m"] == depth)
            ]
            if not row.empty:
                temp = float(row["pred_temp"].values[0]) if "pred_temp" in row.columns else np.nan
                speed_val = float(row["Speed"].values[0]) if "Speed" in row.columns else np.nan
                dir_val   = float(row["Direction_deg"].values[0]) if "Direction_deg" in row.columns else np.nan

                corr_raw, corr_clip = None, None
                if use_correction and pd.notna(temp) and (reg_depthwise is not None) and (int(depth) in reg_depthwise):
                    alpha, beta = reg_depthwise[int(depth)]
                    corr_raw = float(alpha + beta * float(temp))
                    corr_clip = float(np.clip(corr_raw, TEMP_MIN, TEMP_MAX))

                # === ここが追加（週間と同じ無効判定） ===
                is_invalid = False
                if use_correction and (corr_raw is not None):
                    if (corr_raw <= PHYS_MIN) or (corr_raw >= PHYS_MAX):
                        is_invalid = True
                    if not is_invalid:
                        if (corr_clip <= TEMP_MIN) or (corr_clip >= TEMP_MAX):
                            is_invalid = True
                    if not is_invalid and not pd.isna(temp):
                        if abs(float(corr_raw) - float(temp)) >= float(outlier_th):
                            is_invalid = True
                # =====================================

                html += render_cell_html(
                    temp=temp,
                    speed_mps=speed_val,
                    dir_deg=dir_val,
                    use_correction=use_correction,
                    corr_temp_raw=corr_raw,
                    corr_temp_clip=corr_clip,
                    bg_basis=bg_basis,
                    hide_outlier_cells=hide_outlier_cells,
                    is_invalid=is_invalid,
                )
            else:
                html += "<td>-</td>"
        html += "</tr>\n"
    html += "</tbody></table></div>"
    return html


# =========================================
# メイン処理
# =========================================
if MODE_FIXED == "予測カレンダー":
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
    tolerance_min = st.sidebar.slider("時刻差の許容範囲（分）", 5, 120, 35, step=5)
    train_days = st.sidebar.slider("補正学習期間（日数）", 7, 90, 30, step=1)
    bg_basis = st.sidebar.radio("セル背景の基準", ["予測に連動", "補正に連動"], index=0)
    max_h_vh = st.sidebar.slider("表の最大高さ (vh)", 40, 80, 65, step=5)
    recent_days = st.sidebar.slider("表示日数（直近）", 7, 10, 8, step=1)
    outlier_th = st.sidebar.slider("補正と予測の乖離で無効化（℃）", 0.0, 10.0, 7.0, step=0.5)
    hide_outlier_cells = st.sidebar.checkbox("無効セルを非表示（週間）", value=False)
    # ★ 追加：選択日にも非表示オプションを用意（週間と同動作）
    hide_outlier_cells_day = st.sidebar.checkbox("無効セルを非表示（選択日）", value=False)

    df_dr = load_dr_single_file(base_dir, selected_file)
    if df_dr.empty:
        st.warning("DRデータが読み込めませんでした")
        st.stop()

    latest_dt = df_dr["datetime"].max()
    latest_day = latest_dt.date()
    depths_all = sorted([int(d) for d in df_dr["depth_m"].dropna().unique()])

    mode_view = st.radio("表示期間", ["週間予測（表示値は昼頃）", "選択日（1時間毎）"])

    # CSS
    styles = get_calendar_css(max_h_vh)

    # ----------------------------
    # 週間予測（昼頃）
    # ----------------------------
    if mode_view == "週間予測（表示値は昼頃）":
        available_days = sorted(df_dr["date_day"].unique())
        min_day = min(available_days) if available_days else latest_day
        max_day = max(available_days) if available_days else latest_day
        selected_day = st.date_input(
            "週間予測の基準日", value=max_day, min_value=min_day, max_value=max_day
        )
        start_day = pd.Timestamp(selected_day) - pd.Timedelta(days=recent_days - 1)
        end_day = pd.Timestamp(selected_day)
        day_list = list(pd.date_range(start_day, end_day, freq="D"))

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
            if reg_depthwise is None:
                st.warning("回帰係数の算出に失敗（一致データ不足など）。補正なしで表示します。")
                use_correction = False

        period_label = f"**{start_day:%m/%d}～{end_day:%m/%d}までの推移（期間内の全時間）**"
        st.markdown(period_label)

        layers = make_layer_groups(depths_all)
        any_line = False
        for lname, ldepths in layers.items():
            line = summarize_weekly_layer_temp(
                lname, ldepths, df_period, df_dr, selected_day,
                use_correction=use_correction,
                reg_depthwise=reg_depthwise,
                stable_eps=0.4,
                outlier_th=outlier_th,
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
            df_period=df_period,
            day_list=day_list,
            depths_for_table=depths_for_table,
            use_correction=use_correction,
            reg_depthwise=reg_depthwise,
            bg_basis=bg_basis,
            hide_outlier_cells=hide_outlier_cells,
            outlier_th=outlier_th,
        )
        full_html = f"<!doctype html><html><head><meta charset='utf-8'>{styles}</head><body>{table_html}</body></html>"
        iframe_height = int(max(400, min(1100, max_h_vh * 10)))
        st_html(full_html, height=iframe_height, scrolling=True)

    # ----------------------------
    # 選択日（1時間毎）
    # ----------------------------
    else:
        available_days = sorted(df_dr["date_day"].unique())
        min_day = min(available_days) if available_days else latest_day
        max_day = max(available_days) if available_days else latest_day
        selected_day = st.date_input(
            "表示日（1時間毎）", value=max_day, min_value=min_day, max_value=max_day
        )
        df_day = df_dr[df_dr["date_day"] == selected_day].copy()
        hours_list = sorted(df_day["datetime"].dt.floor("h").unique())
        times_hr = [t.strftime('%H:%M') for t in hours_list]

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

        # ★ ここで週間と同じ基準の無効化を適用
        table_html = build_daily_table_html(
            df_day=df_day,
            hours_list=hours_list,
            times_hr=times_hr,
            depths_for_table=depths_for_table_sel,
            use_correction=use_correction,
            reg_depthwise=reg_depthwise,
            bg_basis=bg_basis,
            hide_outlier_cells=hide_outlier_cells_day,  # ← 選択日用の非表示フラグ
            outlier_th=outlier_th,                      # ← 週間と同じ閾値を使用
        )
        full_html = f"<!doctype html><html><head><meta charset='utf-8'>{styles}</head><body>{table_html}</body></html>"
        iframe_height = int(max(400, min(1100, max_h_vh * 10)))
        st_html(full_html, height=iframe_height, scrolling=True)