# wt_test.py
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
        obs = obs[obs["datetime"] >= start_dt]
    if end_dt is not None:
        pred = pred[pred["datetime"] <= end_dt]
        obs = obs[obs["datetime"] <= end_dt]
    if pred.empty or obs.empty:
        return None, None

    tol = pd.Timedelta(minutes=int(tolerance_min))

    pred_sorted = pred.sort_values(["depth_m", "datetime"]).copy()
    obs_sorted = obs.sort_values(["depth_m", "datetime"]).copy()
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

    df_dr = load_dr_single_file(base_dir, selected_file)
    if df_dr.empty:
        st.warning("DRデータが読み込めませんでした")
        st.stop()

    latest_dt = df_dr["datetime"].max()
    latest_day = latest_dt.date()
    depths_all = sorted([int(d) for d in df_dr["depth_m"].dropna().unique()])

    mode_view = st.radio("表示期間", ["週間予測（昼頃）", "選択日（1時間毎）"])

    styles = f"""
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
    linear-gradient(to left,  rgba(0,0,0,0.06), rgba(0,0,0,0)) right/16px 100% no-repeat;
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
.calendar-scroll-container::-webkit-scrollbar {{ height: 10px; width: 10px; }}
.calendar-scroll-container::-webkit-scrollbar-thumb {{ background: #c8c8c8; border-radius: 6px; }}
.pred-small {{ font-size: 12px; color: #555; }}

@media (max-width: 480px) {{
  .calendar-table {{ font-size: 13px; }}
  .calendar-table td.depth-cell {{ min-width: 30px; }}
}}
</style>
""".strip()

    # -------------------------
    # 週間予測（昼頃）
    # -------------------------
    if mode_view == "週間予測（昼頃）":
        available_days = sorted(df_dr["date_day"].unique())
        min_day = min(available_days) if available_days else latest_day
        max_day = max(available_days) if available_days else latest_day
        selected_day = st.date_input("週間予測の基準日", value=max_day,
                                     min_value=min_day, max_value=max_day)

        start_day = pd.Timestamp(selected_day) - pd.Timedelta(days=recent_days - 1)
        end_day = pd.Timestamp(selected_day)
        day_list = list(pd.date_range(start_day, end_day, freq="D"))

        df_period = df_dr[df_dr["date_day"].isin([d.date() for d in day_list])].copy()

        reg_depthwise, n_match_reg = None, None
        if use_correction:
            train_start_dt = pd.Timestamp(selected_day) - pd.Timedelta(days=train_days)
            train_end_dt = pd.Timestamp(selected_day) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
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
            else:
                total_pairs = sum((n_match_reg or {}).values())
                st.caption(f"補正に使用したデータ数（学習期間内の一致ペア合計）: {total_pairs} 件")

        depths_for_table = list(depths_all)
        if use_correction and (reg_depthwise is not None):
            depths_for_table = [d for d in depths_for_table if int(d) in reg_depthwise]

        times = [d.strftime('%m/%d') for d in day_list]

        table_html = (
            '<div class="calendar-scroll-container">'
            '<table class="calendar-table">'
            "<thead><tr><th>水深</th>"
            + "".join([f"<th>{t}</th>" for t in times])
            + "</tr></thead><tbody>"
        )

        for depth in depths_for_table:
            table_html += f"<tr><td class='depth-cell'>{depth}m</td>"
            for idx in range(len(times)):
                day = day_list[idx]
                g = df_period[
                    (df_period["date_day"] == day.date()) &
                    (df_period["depth_m"] == depth)
                ]
                if not g.empty:
                    target_dt = pd.Timestamp(day.date()) + pd.Timedelta(hours=12)
                    g2 = g.assign(_diff=(g["datetime"] - target_dt).abs()).sort_values("_diff")
                    row = g2.iloc[[0]].drop(columns=["_diff"])
                else:
                    row = pd.DataFrame()

                if not row.empty:
                    temp = row["pred_temp"].astype(float).values[0] if "pred_temp" in row.columns else np.nan
                    pred_label = f"{temp:.1f}°C" if pd.notna(temp) else "NaN"
                    pred_html = f"<span class='pred-small'>{pred_label}</span>" if use_correction else f"<span>{pred_label}</span>"

                    speed_kt_label, arrow_svg = "", ""
                    if "Speed" in row.columns and "Direction_deg" in row.columns:
                        v_speed = row["Speed"].values[0]
                        v_dir = row["Direction_deg"].values[0]
                        if pd.notna(v_speed) and pd.notna(v_dir):
                            speed_kt = float(v_speed) * 1.94384
                            speed_kt_label = f"{speed_kt:.1f} kt"
                            arrow_svg = get_arrow_svg(float(v_dir), float(v_speed))

                    corr_temp, corr_value_html = None, ""
                    if use_correction:
                        if pd.notna(temp) and (reg_depthwise is not None) and (int(depth) in reg_depthwise):
                            alpha, beta = reg_depthwise[int(depth)]
                            corr_temp = float(np.clip(alpha + beta * float(temp), TEMP_MIN, TEMP_MAX))
                            corr_value_html = f"{corr_temp:.1f}°C"
                        else:
                            corr_value_html = "NaN"

                    # 背景色
                    if (bg_basis == "補正に連動") and (corr_temp is not None):
                        bg_color = get_color(float(corr_temp))
                    else:
                        bg_color = get_color(float(temp)) if pd.notna(temp) else "rgba(220,220,220,0.6)"

                    # 可変パーツを事前に作成（f-string の内側ネストを避ける）
                    speed_html = f"<span style='font-size:12px;color:#444;'>{speed_kt_label}</span>" if speed_kt_label else ""
                    arrow_html = f"<span style='display:block;line-height:1;margin:0;padding:0;'>{arrow_svg}</span>" if arrow_svg else ""
                    corr_html = f"<span style='color:#D32F2F;font-weight:700;font-size:14px;margin:0;padding:0;'>{corr_value_html}</span>" if (use_correction and corr_value_html) else ""

                    cell_content = (
                        "<div style='display:flex;flex-direction:column;align-items:center;gap:2px;margin:0;padding:0;'>"
                        + pred_html
                        + speed_html
                        + arrow_html
                        + corr_html
                        + "</div>"
                    )
                    cell = f"<td style='background:{bg_color}'>{cell_content}</td>"
                else:
                    cell = "<td>-</td>"

                table_html += cell
            table_html += "</tr>\n"

        table_html += "</tbody></table></div>"

        full_html = f"<!doctype html><html><head><meta charset='utf-8'>{styles}</head><body>{table_html}</body></html>"
        iframe_height = int(max(400, min(1100, max_h_vh * 10)))
        st_html(full_html, height=iframe_height, scrolling=True)

    # -------------------------
    # 選択日（1時間毎）
    # -------------------------
    else:
        available_days = sorted(df_dr["date_day"].unique())
        min_day = min(available_days) if available_days else latest_day
        max_day = max(available_days) if available_days else latest_day
        selected_day = st.date_input("表示日（1時間毎）", value=max_day,
                                     min_value=min_day, max_value=max_day)

        df_day = df_dr[df_dr["date_day"] == selected_day].copy()
        hours_list = sorted(df_day["datetime"].dt.floor("h").unique())
        times_hr = [t.strftime('%H:%M') for t in hours_list]

        reg_depthwise, n_match_reg = None, None
        if use_correction:
            sel_train_end_dt = pd.Timestamp(selected_day) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
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
                n_match_reg = n_match_reg_sel
                total_pairs = sum((n_match_reg or {}).values())
                st.caption(f"補正に使用したデータ数（選択日終端の一致ペア合計）: {total_pairs} 件")

        table_html = (
            '<div class="calendar-scroll-container">'
            '<table class="calendar-table">'
            "<thead><tr><th>水深</th>"
            + "".join([f"<th>{t}</th>" for t in times_hr])
            + "</tr></thead><tbody>"
        )

        for depth in depths_all:
            table_html += f"<tr><td class='depth-cell'>{depth}m</td>"
            for idx in range(len(times_hr)):
                time_obj = hours_list[idx]
                row = df_day[
                    (df_day["datetime"].dt.floor("h") == time_obj) &
                    (df_day["depth_m"] == depth)
                ]

                if not row.empty:
                    temp = row["pred_temp"].astype(float).values[0] if "pred_temp" in row.columns else np.nan
                    pred_label = f"{temp:.1f}°C" if pd.notna(temp) else "NaN"
                    pred_html = f"<span class='pred-small'>{pred_label}</span>" if use_correction else f"<span>{pred_label}</span>"

                    speed_kt_label, arrow_svg = "", ""
                    if "Speed" in row.columns and "Direction_deg" in row.columns:
                        v_speed = row["Speed"].values[0]
                        v_dir = row["Direction_deg"].values[0]
                        if pd.notna(v_speed) and pd.notna(v_dir):
                            speed_kt = float(v_speed) * 1.94384
                            speed_kt_label = f"{speed_kt:.1f} kt"
                            arrow_svg = get_arrow_svg(float(v_dir), float(v_speed))

                    corr_temp, corr_value_html = None, ""
                    if use_correction:
                        if pd.notna(temp) and (reg_depthwise is not None) and (int(depth) in reg_depthwise):
                            alpha, beta = reg_depthwise[int(depth)]
                            corr_temp = float(np.clip(alpha + beta * float(temp), TEMP_MIN, TEMP_MAX))
                            corr_value_html = f"{corr_temp:.1f}°C"
                        else:
                            corr_value_html = "NaN"

                    if (bg_basis == "補正に連動") and (corr_temp is not None):
                        bg_color = get_color(float(corr_temp))
                    else:
                        bg_color = get_color(float(temp)) if pd.notna(temp) else "rgba(220,220,220,0.6)"

                    # 可変パーツを事前に作成
                    speed_html = f"<span style='font-size:12px;color:#444;'>{speed_kt_label}</span>" if speed_kt_label else ""
                    arrow_html = f"<span style='display:block;line-height:1;margin:0;padding:0;'>{arrow_svg}</span>" if arrow_svg else ""
                    corr_html = f"<span style='color:#D32F2F;font-weight:700;font-size:14px;margin:0;padding:0;'>{corr_value_html}</span>" if (use_correction and corr_value_html) else ""

                    cell_content = (
                        "<div style='display:flex;flex-direction:column;align-items:center;gap:2px;margin:0;padding:0;'>"
                        + pred_html
                        + speed_html
                        + arrow_html
                        + corr_html
                        + "</div>"
                    )
                    cell = f"<td style='background:{bg_color}'>{cell_content}</td>"
                else:
                    cell = "<td>-</td>"

                table_html += cell
            table_html += "</tr>\n"

        table_html += "</tbody></table></div>"

        full_html = f"<!doctype html><html><head><meta charset='utf-8'>{styles}</head><body>{table_html}</body></html>"
        iframe_height = int(max(400, min(1100, max_h_vh * 10)))
        st_html(full_html, height=iframe_height, scrolling=True)