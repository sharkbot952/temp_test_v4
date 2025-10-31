import os
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from streamlit.components.v1 import html as st_html

# =========================================
# 設定
# =========================================
base_dir = "./data"

def pjoin(*parts: str) -> str:
    """join + normpath をまとめて行う（UNC/混在セパレータ対策）"""
    return os.path.normpath(os.path.join(*parts))

# 表示モード
mode = st.sidebar.radio("表示モード", ["予測カレンダー", "水温グラフ", "実測比（PC用）"])

# キャッシュクリア（検証用）
if st.sidebar.button("補正キャッシュをクリア"):
    st.cache_data.clear()
    st.success("補正キャッシュをクリアしました。再計算します。")

# =========================================
# 共通ユーティリティ（矢印・色・TZ）
# =========================================
HEAD_LENGTH_RATIO = 0.55
HEAD_HALF_HEIGHT_RATIO = 0.35
SHAFT_WIDTH_PX = 4.0

def get_arrow_svg(direction_deg, speed_mps):
    """速度と方位からSVG矢印を返す（欠損は空文字）"""
    if pd.isna(speed_mps) or pd.isna(direction_deg):
        return ""
    # SVGのx軸右向き基準に変換
    css_angle = (direction_deg - 90) % 360
    # 速度帯で色・サイズを決定
    def _style(speed_mps):
        if np.isnan(speed_mps):
            return 18, "#CCCCCC"
        speed_kt = speed_mps * 1.94384
        if speed_kt < 1.0:
            return 18, "#0000FF"   # 低速: 青
        elif speed_kt < 2.0:
            return 22, "#FFC107"   # 中速: アンバー
        else:
            return 26, "#FF0000"   # 高速: 赤
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
"""

def get_color(temp: float, t_min: float = 5, t_max: float = 25) -> str:
    """水温の背景色（連続スケール）"""
    if pd.isna(temp):
        return "rgba(220,220,220,0.4)"
    ratio = (float(temp) - t_min) / (t_max - t_min)
    ratio = max(0, min(1, ratio))
    if ratio < 0.5:
        # 青から白へ
        r = int(240 * ratio * 2)
        g = int(240 * ratio * 2)
        b = 240
    else:
        # 白から赤へ
        r = 240
        g = int(240 * (1 - (ratio - 0.5) * 2))
        b = int(240 * (1 - (ratio - 0.5) * 2))
    return f"rgba({r},{g},{b},0.4)"

def utc_to_jst_naive(s: pd.Series) -> pd.Series:
    """UTC文字列/時刻 → JST（naive）に変換"""
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    dt = dt.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return dt

def jst_to_naive(s: pd.Series) -> pd.Series:
    """JST文字列/時刻 → JST（naive）に統一（tzあれば外す）"""
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return dt

# =========================================
# pred/obs asof 結合（深さごと）
# =========================================
def safe_merge_asof_by_depth(
    left: pd.DataFrame,
    right: pd.DataFrame,
    tolerance: pd.Timedelta,
    right_value_cols: List[str],
    suffixes: Tuple[str, str] = ("_x", "_y")
) -> pd.DataFrame:
    """深さ単位で日時昇順にソートし、asof結合（最近傍）。rightは指定列のみ持ち込む。"""
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
            l, r, on="datetime", by="depth_m",
            tolerance=tolerance, direction="nearest",
            suffixes=suffixes
        )
        out_list.append(merged)
    if not out_list:
        return pd.DataFrame(columns=list(left.columns) + right_value_cols)
    return pd.concat(out_list, ignore_index=True)

# =========================================
# pred単一CSVローダ（Temp/Salinity/Depth, 任意で u,v）
# =========================================
@st.cache_data(show_spinner=False)
def load_dr_single_file(base_dir: str, filename: str) -> pd.DataFrame:
    """pred/<filename>.csv を読み込み、日時・深さ整形。u,vあれば速度・方位も付与。"""
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
    # u,v があれば速度・方位
    if ("U" in df.columns) and ("V" in df.columns):
        df["U"] = pd.to_numeric(df["U"], errors="coerce")
        df["V"] = pd.to_numeric(df["V"], errors="coerce")
        df["Speed"] = np.sqrt(np.square(df["U"]) + np.square(df["V"]))
        df["Direction_deg"] = (np.degrees(np.arctan2(df["U"], df["V"])) + 360.0) % 360.0
    # 表示補助
    df["date_day"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    return df

# =========================================
# 補正：水深別回帰（obs ≈ α + β*pred）※水温（Temp）用
# =========================================
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
    """各水深ごとに obs ≈ alpha + beta * pred を学習。係数辞書と一致ペア数辞書を返す。"""
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

    # 期間フィルタ
    if start_dt is not None:
        pred = pred[pred["datetime"] >= start_dt]
        obs = obs[obs["datetime"] >= start_dt]
    if end_dt is not None:
        pred = pred[pred["datetime"] <= end_dt]
        obs = obs[obs["datetime"] <= end_dt]
    if pred.empty or obs.empty:
        return None, None

    # asof一致（深さごと）
    tol = pd.Timedelta(minutes=int(tolerance_min))
    pred_sorted = pred.sort_values(["depth_m", "datetime"]).copy()
    obs_sorted = obs.sort_values(["depth_m", "datetime"]).copy()
    merged = safe_merge_asof_by_depth(
        pred_sorted, obs_sorted, tol,
        right_value_cols=["obs_temp"], suffixes=("", "")
    )
    pair = merged.dropna(subset=["pred_temp", "obs_temp", "depth_m"]).copy()
    if pair.empty:
        return None, None

    # 水深別学習
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
            A = np.vstack([X, np.ones_like(X)]).T  # y ≈ beta*X + alpha
            beta, alpha = np.linalg.lstsq(A, y, rcond=None)[0]
            reg_depth[int(d)] = (float(alpha), float(beta))
    return (reg_depth if reg_depth else None, n_depth if n_depth else None)

# =========================================
# 予測カレンダー（pred参照・回帰補正のみ）— components.html で描画
# =========================================
if mode == "予測カレンダー":
    parent_folder_dr = pjoin(base_dir, "pred")
    if not os.path.exists(parent_folder_dr):
        st.error(f"フォルダが見つかりません: {parent_folder_dr}")
        st.stop()
    dr_files = [f for f in os.listdir(parent_folder_dr) if f.endswith(".csv")]
    if not dr_files:
        st.warning("pred に CSV がありません")
        st.stop()

    selected_file = st.selectbox("解析対象ファイルを選択", sorted(dr_files))

    # 補正UI（回帰のみ）
    use_correction = st.sidebar.checkbox("実測ベース補正", value=False)
    tolerance_min = st.sidebar.slider("時刻差の許容範囲（分）", 5, 120, 35, step=5)
    train_days = st.sidebar.slider("補正学習期間（日数）", 7, 90, 30, step=1)

    # 背景色の基準：予測 or 補正後
    bg_basis = st.sidebar.radio("セル背景の基準", ["予測水温に連動", "補正後水温に連動"], index=0)

    # スクロール高さ
    max_h_vh = st.sidebar.slider("表の最大高さ (vh)", 40, 80, 65, step=5)

    # 直近日数（既定8日）
    recent_days = st.sidebar.slider("表示日数（直近）", 7, 10, 8, step=1)
    exclude_latest = st.sidebar.checkbox("最新日を除外する", value=False)

    # 補正値が無い水深は非表示（補正ON時のみ意味あり）
    hide_uncorrected_depths = st.sidebar.checkbox("補正値が無い水深は非表示", value=False)

    # DRロード
    df_dr = load_dr_single_file(base_dir, selected_file)
    if df_dr.empty:
        st.warning("DRデータが読み込めませんでした")
        st.stop()

    # 最新日と深さ
    latest_dt = df_dr["datetime"].max()
    latest_day = latest_dt.date()
    depths_all = sorted([int(d) for d in df_dr["depth_m"].dropna().unique()])

    # 直近期間の決定
    if exclude_latest:
        start_day = latest_day - pd.Timedelta(days=recent_days)
        end_day = latest_day - pd.Timedelta(days=1)
    else:
        start_day = latest_day - pd.Timedelta(days=(recent_days - 1))
        end_day = latest_day

    # 期間抽出
    day_list = list(pd.date_range(start_day, end_day, freq="D"))
    df_period = df_dr[df_dr["date_day"].isin([d.date() for d in day_list])].copy()

    # --- 回帰補正の学習（必要時）— 終端を最新日に揃える（週表示用の既定） ---
    reg_depthwise, n_match_reg = None, None
    if use_correction:
        train_start_dt = pd.Timestamp(latest_day) - pd.Timedelta(days=train_days)
        train_end_dt   = pd.Timestamp(latest_day) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        with st.spinner(f"回帰補正パラメータ算出中（{selected_file}, 遡り{train_days}日）..."):
            reg_depthwise, n_match_reg = compute_depthwise_regression(
                base_dir, selected_file, tolerance_min,
                start_dt=train_start_dt, end_dt=train_end_dt, min_pairs=5
            )
        if reg_depthwise is None:
            st.warning("回帰係数の算出に失敗（一致データ不足など）。補正なしで表示します。")
            use_correction = False
        else:
            with st.expander(f"補正の学習情報（最新日から{train_days}日遡り）", expanded=False):
                st.write(f"期間: {train_start_dt:%Y-%m-%d} 〜 {train_end_dt:%Y-%m-%d}")
                st.write("一致ペア数:", {int(k): int(v) for k, v in (n_match_reg or {}).items()})
                st.write("学習済み水深（α, β）:",
                         {int(k): (round(v[0], 4), round(v[1], 4))
                          for k, v in (reg_depthwise or {}).items()})

    # 深さの決定（補正ON＋係数がある水深のみ表示にする場合）
    depths_for_table = list(depths_all)
    if use_correction and (reg_depthwise is not None) and hide_uncorrected_depths:
        depths_for_table = [d for d in depths_for_table if int(d) in reg_depthwise]

    # 表示期間
    mode_view = st.radio("表示期間", ["週間予測（昼頃）", "選択日（1時間毎）"])

    # ヘッダー（日付のみ）
    times = [d.strftime('%m/%d') for d in day_list]

    # ===== スクロール対応CSS（stickyヘッダー＆先頭列） =====
    styles = f"""
    <style>
    .calendar-scroll-container {{
      overflow: auto;
      max-height: {max_h_vh}vh;
      max-width: 100%;
      -webkit-overflow-scrolling: touch;
      border: 1px solid #e5e5e5;
      border-radius: 8px;
      background: #fff;
    }}
    .calendar-table {{
      border-collapse: collapse;
      width: max-content;
      min-width: 640px;
      font-size: 14px;
      font-family:'Noto Sans JP', 'Roboto', sans-serif;
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
    .calendar-scroll-container {{
      background:
        linear-gradient(to right, rgba(0,0,0,0.06), rgba(0,0,0,0)) left/16px 100% no-repeat,
        linear-gradient(to left, rgba(0,0,0,0.06), rgba(0,0,0,0)) right/16px 100% no-repeat;
      background-attachment: local, local;
    }}
    .pred-small {{ font-size: 12px; color: #555; }}
    @media (max-width: 480px) {{
      .calendar-table {{ font-size: 13px; }}
      .calendar-table td.depth-cell {{ min-width: 30px; }}
    }}
    </style>
    """

    # ===== テーブルHTML開始 =====
    table_html = """
    <div class="calendar-scroll-container">
    <table class="calendar-table">
    <thead>
    <tr>
      <th>水深</th>
    """ + "".join([f"<th>{t}</th>" for t in (times if mode_view == "週間予測（昼頃）" else [])]) + "</tr></thead><tbody>"

    if mode_view == "週間予測（昼頃）":
        # 週表示：各日 12:00 が無い場合は最寄り時刻を採用（既存ロジック）
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
                    g = g.assign(_diff=(g["datetime"] - target_dt).abs()).sort_values("_diff")
                    row = g.iloc[[0]].drop(columns=["_diff"])
                else:
                    row = pd.DataFrame()

                if not row.empty:
                    # 予測値
                    temp = row["pred_temp"].astype(float).values[0] if "pred_temp" in row.columns else np.nan
                    pred_label = f"{temp:.1f}°C" if pd.notna(temp) else "NaN"
                    pred_html = f"<span class='pred-small'>{pred_label}</span>" if use_correction else f"<span>{pred_label}</span>"

                    # 速度・方位（u,v がある場合のみ）
                    speed_kt_label, arrow_svg = "", ""
                    if "Speed" in row.columns and "Direction_deg" in row.columns:
                        v_speed = row["Speed"].values[0]
                        v_dir = row["Direction_deg"].values[0]
                        if pd.notna(v_speed) and pd.notna(v_dir):
                            speed_kt = float(v_speed) * 1.94384
                            speed_kt_label = f"{speed_kt:.1f} kt"
                            arrow_svg = get_arrow_svg(float(v_dir), float(v_speed))

                    # 回帰補正（あれば）
                    corr_temp, corr_value_html = None, ""
                    if use_correction:
                        if pd.notna(temp) and (reg_depthwise is not None) and (int(depth) in reg_depthwise):
                            alpha, beta = reg_depthwise[int(depth)]
                            corr_temp = float(np.clip(alpha + beta * float(temp), TEMP_MIN, TEMP_MAX))
                            corr_value_html = f"{corr_temp:.1f}°C"
                        else:
                            corr_value_html = "NaN"

                    # 背景色
                    if (bg_basis == "補正後水温に連動") and (corr_temp is not None):
                        bg_color = get_color(float(corr_temp))
                    else:
                        bg_color = get_color(float(temp)) if pd.notna(temp) else "rgba(220,220,220,0.6)"

                    # セル内容
                    cell_content = f"""
                    <div style='display:flex;flex-direction:column;align-items:center;gap:2px;margin:0;padding:0;'>
                      {pred_html}
                      {f"<span style='font-size:12px;color:#444;'>{speed_kt_label}</span>" if speed_kt_label else ""}
                      {f"<span style='display:block;line-height:1;margin:0;padding:0;'>{arrow_svg}</span>" if arrow_svg else ""}
                      {f"<span style='color:#D32F2F;font-weight:700;font-size:14px;margin:0;padding:0;'>{corr_value_html}</span>" if (use_correction and corr_value_html) else ""}
                    </div>
                    """
                    cell = f"<td style='background:{bg_color}'>{cell_content}</td>"
                else:
                    cell = "<td>-</td>"

                table_html += cell
            table_html += "</tr>\n"

    else:
        # ★ 修正：選択日（1時間毎）— 選択日を終端にして係数を再学習
        available_days = sorted(df_dr["date_day"].unique())
        min_day = min(available_days) if available_days else latest_day
        max_day = max(available_days) if available_days else latest_day
        selected_day = st.date_input(
            "表示日（1時間毎）", value=max_day, min_value=min_day, max_value=max_day
        )

        df_day = df_dr[df_dr["date_day"] == selected_day].copy()
        hours_list = sorted(df_day["datetime"].dt.floor("h").unique())
        times_hr = [t.strftime('%H:%M') for t in hours_list]

        # 選択日終端で再学習
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
                st.warning("選択日終端での回帰係数算出に失敗（一致データ不足など）。既存係数（最新日終端）を使用します。")
                # reg_depthwise は週表示で学習済みのもの、または None を維持
            else:
                reg_depthwise = reg_depthwise_sel
                n_match_reg   = n_match_reg_sel
                with st.expander(f"補正の学習情報（終端={selected_day:%Y-%m-%d}・遡り{train_days}日）", expanded=False):
                    st.write(f"期間: {sel_train_start_dt:%Y-%m-%d} 〜 {sel_train_end_dt:%Y-%m-%d}")
                    st.write("一致ペア数:", {int(k): int(v) for k, v in (n_match_reg or {}).items()})
                    st.write("学習済み水深（α, β）:",
                             {int(k): (round(v[0], 4), round(v[1], 4))
                              for k, v in (reg_depthwise or {}).items()})

        # ヘッダーを時間表示に差し替え
        table_html = """
        <div class="calendar-scroll-container">
        <table class="calendar-table">
        <thead>
        <tr>
          <th>水深</th>
        """ + "".join([f"<th>{t}</th>" for t in times_hr]) + "</tr></thead><tbody>"

        for depth in depths_for_table:
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

                    if (bg_basis == "補正後水温に連動") and (corr_temp is not None):
                        bg_color = get_color(float(corr_temp))
                    else:
                        bg_color = get_color(float(temp)) if pd.notna(temp) else "rgba(220,220,220,0.6)"

                    cell_content = f"""
                    <div style='display:flex;flex-direction:column;align-items:center;gap:2px;margin:0;padding:0;'>
                      {pred_html}
                      {f"<span style='font-size:12px;color:#444;'>{speed_kt_label}</span>" if speed_kt_label else ""}
                      {f"<span style='display:block;line-height:1;margin:0;padding:0;'>{arrow_svg}</span>" if arrow_svg else ""}
                      {f"<span style='color:#D32F2F;font-weight:700;font-size:14px;margin:0;padding:0;'>{corr_value_html}</span>" if (use_correction and corr_value_html) else ""}
                    </div>
                    """
                    cell = f"<td style='background:{bg_color}'>{cell_content}</td>"
                else:
                    cell = "<td>-</td>"

                table_html += cell
            table_html += "</tr>\n"

        table_html += "</tbody></table></div>"

    # ===== テーブルHTML終了・描画 =====
    full_html = f"""<!doctype html>
    <html><head><meta charset="utf-8">{styles}</head><body>
    {table_html}
    </body></html>"""

    iframe_height = int(max(400, min(1100, max_h_vh * 10)))
    st_html(full_html, height=iframe_height, scrolling=True)

# =========================================
# 水温グラフ
# =========================================
elif mode == "水温グラフ":
    parent_folder_dr = pjoin(base_dir, "pred")
    if not os.path.exists(parent_folder_dr):
        st.error(f"フォルダが見つかりません: {parent_folder_dr}")
        st.stop()
    dr_files = [f for f in os.listdir(parent_folder_dr) if f.endswith(".csv")]
    if not dr_files:
        st.warning("pred に CSV がありません")
        st.stop()

    selected_file = st.selectbox("解析対象ファイルを選択", sorted(dr_files))
    # 補正UI（回帰のみ）
    use_correction = st.sidebar.checkbox("実測ベース補正", value=False)
    tolerance_min = st.sidebar.slider("時刻差の許容範囲（分）", 5, 120, 35, step=5)
    train_days = st.sidebar.slider("補正学習期間（日数）", 7, 90, 30, step=1)

    # 表示期間（直近2週間 or 直近60日）
    period_choice = st.sidebar.radio("表示期間", ["直近2週間", "直近60日"], index=0)

    # データ取得
    df_dr = load_dr_single_file(base_dir, selected_file)
    if df_dr.empty:
        st.warning("DRデータが読み込めませんでした")
        st.stop()

    # 期間決定
    latest_dt = df_dr["datetime"].max()
    end_day = latest_dt.date()
    if period_choice == "直近2週間":
        start_day = (latest_dt - pd.Timedelta(days=13)).date()
        title_suffix = "（直近2週間・時間別）"
    else:
        start_day = (latest_dt - pd.Timedelta(days=59)).date()
        title_suffix = "（直近60日・時間別）"

    # 期間抽出 & 整形
    df_period = df_dr[(df_dr["date_day"] >= start_day) & (df_dr["date_day"] <= end_day)].copy()
    df_period = df_period.sort_values("datetime")

    # 重複 (depth_m, datetime) は中央値で集約
    if "pred_temp" in df_period.columns:
        df_period = (
            df_period.groupby(["depth_m", "datetime"], as_index=False)
                     .agg({"pred_temp": "median"})
        )

    # 1時間リサンプリング＋時間補間（連続2本まで）
    df_period = (
        df_period.sort_values("datetime")
                 .groupby("depth_m", group_keys=False)
                 .apply(lambda g: (
                     g.set_index("datetime")
                      .resample("1H")
                      .median()
                      .interpolate(method="time", limit=2)
                      .reset_index()
                 ))
    )

    # 利用可能な水深
    depth_options = sorted([int(d) for d in df_period["depth_m"].dropna().unique()])
    default_depths = depth_options[:min(3, len(depth_options))] if depth_options else []

    # 深さフィルタ
    selected_depths = st.multiselect(
        "表示する水深（複数選択可）", depth_options, default=default_depths,
        help="項目数を絞ると凡例もコンパクトになります"
    )
    if not selected_depths:
        st.info("水深が未選択です。少なくとも1つ選んでください。")
        st.stop()

    # 回帰補正の学習（必要時）
    reg_depthwise = None
    if use_correction:
        train_start_dt = pd.Timestamp(end_day) - pd.Timedelta(days=train_days)
        train_end_dt = pd.Timestamp(end_day) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        with st.spinner(f"水深別回帰係数を学習中（{selected_file}, 遡り{train_days}日）..."):
            reg_depthwise, _ = compute_depthwise_regression(
                base_dir, selected_file, tolerance_min,
                start_dt=train_start_dt, end_dt=train_end_dt, min_pairs=5
            )
        if reg_depthwise is None:
            st.warning("回帰係数の算出に失敗（一致ペア不足など）。未補正で表示します。")
            use_correction = False

    # ---- グラフ本体：スマホ見やすさ優先 ----
    colors = px.colors.qualitative.Plotly
    fig = go.Figure()
    color_idx = 0
    for d in selected_depths:
        df_d = df_period[df_period["depth_m"] == d]
        if df_d.empty:
            continue
        x = df_d["datetime"]
        y_raw = df_d["pred_temp"].astype(float)
        color = colors[color_idx % len(colors)]
        color_idx += 1

        # 補正ON：係数がある水深のみ補正線を描画
        if use_correction and (reg_depthwise is not None) and (int(d) in reg_depthwise):
            alpha, beta = reg_depthwise[int(d)]
            y_corr = (alpha + beta * y_raw).clip(lower=TEMP_MIN, upper=TEMP_MAX)
            fig.add_trace(go.Scatter(
                x=x, y=y_corr, mode="lines",
                name=f"{d}m 補正", line=dict(color=color, dash="dot", width=3),
                hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>補正後水温: %{y:.2f} °C"
            ))
            fig.add_trace(go.Scatter(
                x=x, y=y_raw, mode="lines",
                name=f"{d}m 予測", line=dict(color=color, width=1), opacity=0.5,
                hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>水温: %{y:.2f} °C"
            ))

        else:
        # 元予測
            fig.add_trace(go.Scatter(
                x=x, y=y_raw, mode="lines",
                name=f"{d}m 予測", line=dict(color=color),
                hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>水温: %{y:.2f} °C"
            ))

    # 凡例設定
    show_legend = st.checkbox("凡例を表示", value=False)
    legend_cfg = dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="left", x=0,
        font=dict(size=12),
        itemsizing="constant"
    )
    fig.update_layout(
        title={
            "text": f"{selected_file} 水温{title_suffix}",
            "y": 0.92, "x": 0.01, "xanchor": "left",
            "font": {"size": 16}, "pad": {"t": 18}
        },
        xaxis_title="日時（JST）", yaxis_title="水温 (℃)",
        margin=dict(l=10, r=10, t=60, b=10),
        height=550, template="plotly_white",
        showlegend=bool(show_legend),
        legend=legend_cfg if show_legend else dict()
    )
    fig.update_xaxes(tickfont=dict(size=11))
    fig.update_yaxes(tickfont=dict(size=11))

    # モードバー（モバイル簡易）
    is_mobile_view = st.checkbox("モバイル簡易表示", value=True)
    if is_mobile_view:
        fig.update_layout(margin=dict(t=72))
    st.plotly_chart(fig, use_container_width=True)

# =========================================
# 実測比（塩分対応＋回帰補正のみ／回帰を全予測時刻へ適用）
# =========================================
elif mode == "実測比（PC用）":
    parent_folder_dr = pjoin(base_dir, "pred")
    parent_folder_obs = pjoin(base_dir, "obs")
    if not os.path.exists(parent_folder_dr) or not os.path.exists(parent_folder_obs):
        st.warning("実測比モードのデータフォルダが見つかりません")
    else:
        dr_files = [f for f in os.listdir(parent_folder_dr) if f.endswith(".csv")]
        obs_files = [f for f in os.listdir(parent_folder_obs) if f.endswith(".csv")]
        common_files = sorted(set(dr_files) & set(obs_files))
        if not common_files:
            st.warning("DRとOBSで共通のファイル名がありません")
        else:
            selected_file = st.sidebar.selectbox("解析対象ファイルを選択", common_files)

            @st.cache_data(show_spinner=False)
            def _read_csv_cached(path: str) -> pd.DataFrame:
                try:
                    return pd.read_csv(path)
                except Exception as e:
                    st.error(f"読み込み失敗: {path} ({e})")
                    return pd.DataFrame()

            dr_path = pjoin(parent_folder_dr, selected_file)
            obs_path = pjoin(parent_folder_obs, selected_file)
            pred = _read_csv_cached(dr_path)
            obs = _read_csv_cached(obs_path)
            if pred.empty or obs.empty:
                st.warning("ファイルが空か、読み込みに失敗しました")
                st.stop()

            # 時刻の正規化：DR=UTC→JST、OBS=JST→naive
            pred["datetime"] = utc_to_jst_naive(pred.get("Date"))
            obs["datetime"] = jst_to_naive(obs.get("Date"))

            # 水深の整数化
            pred["depth_m"] = pd.to_numeric(pred.get("Depth"), errors="coerce").round(0).astype("Int64")
            obs["depth_m"] = pd.to_numeric(obs.get("Depth"), errors="coerce").round(0).astype("Int64")
            pred = pred.dropna(subset=["datetime", "depth_m"]).copy()
            obs = obs.dropna(subset=["datetime", "depth_m"]).copy()
            pred["depth_m"] = pred["depth_m"].astype("int64")
            obs["depth_m"] = obs["depth_m"].astype("int64")

            # 温度・塩分の列名統一
            pred = pred.rename(columns={"Temp": "pred_temp", "Salinity": "pred_sal"})
            obs = obs.rename(columns={"Temp": "obs_temp", "Salinity": "obs_sal"})

            target = st.sidebar.radio("比較対象", ["水温", "塩分"])
            if target == "水温":
                col_pred, col_obs = "pred_temp", "obs_temp"
                y_label = "水温 (℃)"; unit = "℃"
                temp_clip = (TEMP_MIN, TEMP_MAX)
            else:
                col_pred, col_obs = "pred_sal", "obs_sal"
                y_label = "塩分 (psu)"; unit = "psu"
                temp_clip = (None, None)  # 塩分はクリップ不要

            if col_obs not in obs.columns:
                st.warning(f"OBSデータに {col_obs} 列がありません。塩分データがない可能性があります。")
                st.stop()


            # マージ（日時＋水深、最近傍）— 右は対象列のみ
            tol_min = st.sidebar.slider("予測と実測の時刻一致 許容（分）", 5, 120, 35, step=5)
            tolerance = pd.Timedelta(minutes=tol_min)
            pred_sorted = pred.sort_values(["depth_m", "datetime"]).copy()
            obs_sorted = obs.sort_values(["depth_m", "datetime"]).copy()
            merged = safe_merge_asof_by_depth(
                pred_sorted, obs_sorted, tolerance,
                right_value_cols=[col_obs], suffixes=("_dr", "_obs")
            )
            if merged.empty or not merged["datetime"].notna().any():
                st.warning("日時のマージに失敗しました（共通範囲がないか、許容時間が狭すぎる可能性）")
                st.stop()

            # 期間・深さの選択（JST基準）
            min_date = merged["datetime"].min().date()
            max_date = merged["datetime"].max().date()
            date_range = st.sidebar.slider(
                "期間を選択", min_value=min_date, max_value=max_date, value=(min_date, max_date)
            )
            depth_options = sorted(merged["depth_m"].dropna().unique().tolist())
            default_depths = depth_options[:1] if depth_options else []
            selected_depths = st.sidebar.multiselect(
                "水深を選択（複数可）", depth_options, default=default_depths
            )
            if not selected_depths:
                st.info("水深が未選択です。少なくとも1つ選んでください。")
                st.stop()

            sub_merged = merged[
                (merged["datetime"].dt.date >= date_range[0]) &
                (merged["datetime"].dt.date <= date_range[1]) &
                (merged["depth_m"].isin(selected_depths))
            ].copy()

            # 指標計算（元予測）
            def pair_df_for_depth(df: pd.DataFrame, d: int) -> pd.DataFrame:
                return df[(df["depth_m"] == d)].dropna(subset=[col_pred, col_obs])

            mae_dict, bias_dict_view, corr_dict, n_dict = {}, {}, {}, {}
            for d in selected_depths:
                sub = pair_df_for_depth(sub_merged, d)
                n = len(sub)
                if n == 0:
                    mae_dict[d] = np.nan
                    bias_dict_view[d] = np.nan
                    corr_dict[d] = np.nan
                    n_dict[d] = 0
                    continue
                mae = float(np.mean(np.abs(sub[col_pred] - sub[col_obs])))
                bias = float(np.mean(sub[col_pred] - sub[col_obs]))  # 参考表示
                corr = float(sub[col_pred].corr(sub[col_obs])) if n >= 3 else np.nan
                mae_dict[d], bias_dict_view[d], corr_dict[d], n_dict[d] = mae, bias, corr, n

            # バイアス補正列は無効化（表示しない）
            sub_merged["bias_corrected_pred"] = np.nan

            # 回帰補正：学習は「実測ありの一致ペア」で、適用は「全予測時刻」
            sub_merged["reg_corrected_pred"] = np.nan
            reg_coeffs: Dict[int, Tuple[float, float]] = {}
            MIN_PAIRS = 5  # 実測比ではやや緩め
            for d in selected_depths:
                # 学習データ（実測ありの一致ペア）
                sub_pair = pair_df_for_depth(sub_merged, d)
                if len(sub_pair) >= MIN_PAIRS:
                    X = sub_pair[col_pred].astype(float).values
                    y = sub_pair[col_obs].astype(float).values
                    mask = np.isfinite(X) & np.isfinite(y)
                    X, y = X[mask], y[mask]
                    if len(X) >= MIN_PAIRS:
                        A = np.vstack([X, np.ones_like(X)]).T  # y ≈ beta*X + alpha
                        beta, alpha = np.linalg.lstsq(A, y, rcond=None)[0]
                        reg_coeffs[int(d)] = (float(alpha), float(beta))
                # 適用対象：この水深 d の「全予測行」
                sub_all = sub_merged[sub_merged["depth_m"] == d]
                if int(d) in reg_coeffs:
                    alpha, beta = reg_coeffs[int(d)]
                    corr_all = alpha + beta * sub_all[col_pred].astype(float)
                    if temp_clip[0] is not None:
                        corr_all = np.clip(corr_all, temp_clip[0], temp_clip[1])
                    sub_merged.loc[sub_all.index, "reg_corrected_pred"] = corr_all

            # 補正後MAE（回帰のみ：評価は実測ありの行で）
            mae_reg_dict = {}
            for d in selected_depths:
                sub_pair = pair_df_for_depth(sub_merged, d)
                if sub_pair.empty:
                    mae_reg_dict[d] = np.nan
                else:
                    mae_reg = float(np.mean(np.abs(sub_pair["reg_corrected_pred"] - sub_pair[col_obs])))
                    mae_reg_dict[d] = mae_reg

            # 指標テーブル
            idx = pd.Index(selected_depths, name="水深(m)")
            df_metrics = pd.DataFrame(index=idx)
            df_metrics["N(一致ペア数)"] = df_metrics.index.map(n_dict).astype("Int64")
            df_metrics["MAE(元予測)"] = df_metrics.index.map(mae_dict)
            df_metrics["MAE(回帰補正)"] = df_metrics.index.map(mae_reg_dict)
            df_metrics["平均偏差(Bias)"] = df_metrics.index.map(bias_dict_view)
            df_metrics["相関係数(Corr)"] = df_metrics.index.map(corr_dict)
            df_metrics = df_metrics.round(3)

            st.markdown("### 誤差指標（回帰補正効果）")
            st.dataframe(df_metrics, use_container_width=True)
            with st.expander("回帰補正モデル（水深別：obs ≈ α + β*pred）", expanded=False):
                if reg_coeffs:
                    st.write({int(k): (round(v[0], 4), round(v[1], 4)) for k, v in reg_coeffs.items()})
                else:
                    st.write("学習データ不足のため回帰係数は算出されていません。")
            csv = df_metrics.to_csv(index=True).encode("utf-8-sig")
            st.download_button(
                "指標をCSVでダウンロード",
                data=csv, file_name=f"metrics_{selected_file}.csv", mime="text/csv"
            )

            # グラフ表示（回帰補正追加。回帰線は全予測時刻に描画）
            fig = make_subplots(
                rows=len(selected_depths), cols=1, shared_xaxes=True, vertical_spacing=0.05,
                subplot_titles=[f"水深 {int(d)}m" for d in selected_depths],
            )
            for i, d in enumerate(selected_depths, start=1):
                sub = sub_merged[sub_merged["depth_m"] == d]
                if sub.empty:
                    continue
                lg = f"depth{int(d)}"
                fig.add_trace(
                    go.Scatter(
                        x=sub["datetime"], y=sub[col_pred], mode="lines",
                        name="予測値", legendgroup=lg, showlegend=(i==1),
                        line=dict(color="blue"),
                        hovertemplate="%{x}<br>予測: %{y:.3f} " + unit
                    ),
                    row=i, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=sub["datetime"], y=sub[col_obs], mode="markers",
                        name="実測値", legendgroup=lg, showlegend=(i==1),
                        marker=dict(size=3, color="red"),
                        hovertemplate="%{x}<br>実測: %{y:.3f} " + unit
                    ),
                    row=i, col=1
                )
                fig.add_trace(
                    go.Scatter(
                        x=sub["datetime"], y=sub["reg_corrected_pred"], mode="lines",
                        name="回帰補正", legendgroup=lg, showlegend=(i==1),
                        line=dict(dash="dot", color="orange"),
                        hovertemplate="%{x}<br>補正(回帰): %{y:.3f} " + unit
                    ),
                    row=i, col=1
                )
            fig.update_layout(
                title=f"{selected_file} 比較（{target}＋回帰補正：全予測時刻に適用）",
                height=400 * max(1, len(selected_depths)),
                showlegend=True, margin=dict(l=10, r=10, t=60, b=10)
            )
            fig.update_yaxes(title_text=y_label)
            st.plotly_chart(fig, use_container_width=True)