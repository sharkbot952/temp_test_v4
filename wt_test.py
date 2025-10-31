# ------------------------------------------------------------
# 予測カレンダー（スマホ対応統合版）
# ------------------------------------------------------------

import os
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

# =========================================
# ページ設定
# =========================================
st.set_page_config(page_title="予測カレンダー", layout="wide")

# =========================================
# 便利関数
# =========================================
base_dir = "./data"

def pjoin(*parts: str) -> str:
    """join + normpath をまとめて行う（UNC/混在セパレータ対策）"""
    return os.path.normpath(os.path.join(*parts))

# =========================================
# 共通ユーティリティ（矢印・色・TZ）
# =========================================
HEAD_LENGTH_RATIO = 0.55
HEAD_HALF_HEIGHT_RATIO = 0.35
SHAFT_WIDTH_PX = 4.0

def get_arrow_svg(direction_deg: float, speed_mps: float) -> str:
    """速度と方位からSVG矢印を返す（欠損は空文字）"""
    if pd.isna(speed_mps) or pd.isna(direction_deg):
        return ""
    # SVGのx軸右向き基準に変換（CSSで回転）
    css_angle = (direction_deg - 90) % 360

    # 速度帯で色・サイズを決定
    def _style(v_mps: float):
        if np.isnan(v_mps):
            return 18, "#CCCCCC"
        speed_kt = v_mps * 1.94384
        if speed_kt < 1.0:
            return 18, "#0000FF"  # 低速: 青
        elif speed_kt < 2.0:
            return 22, "#FFC107"  # 中速: アンバー
        else:
            return 26, "#FF0000"  # 高速: 赤

    size, color = _style(speed_mps)
    head_len = size * HEAD_LENGTH_RATIO
    head_half_h = size * HEAD_HALF_HEIGHT_RATIO
    line_end = size - head_len

    # 中心線（シャフト）と三角ヘッドを描く
    # ベースは左→右向きの矢印、CSSで rotate(css_angle) する
    svg = f"""
<svg class="arrow-svg" xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="transform: rotate({css_angle}deg);">
  <g fill="none" stroke="{color}" stroke-width="{SHAFT_WIDTH_PX}">
    <!-- シャフト -->
    <line x1="2" y1="{size/2:.2f}" x2="{line_end:.2f}" y2="{size/2:.2f}" />
    <!-- ヘッド（三角） -->
    <polygon fill="{color}" stroke="{color}" points="
      {line_end:.2f},{size/2:.2f}
      {line_end + head_len:.2f},{size/2 - head_half_h:.2f}
      {line_end + head_len:.2f},{size/2 + head_half_h:.2f}
    " />
  </g>
</svg>
"""
    return svg

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
    suffixes: Tuple[str, str] = ("_x", "_y"),
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
            l,
            r,
            on="datetime",
            by="depth_m",
            tolerance=tolerance,
            direction="nearest",
            suffixes=suffixes,
        )
        out_list.append(merged)
    if not out_list:
        return pd.DataFrame(columns=list(left.columns) + right_value_cols)
    return pd.concat(out_list, ignore_index=True)

# =========================================
# pred単一CSVローダ（Temp/Salinity/Depth, 任意で U,V）
# =========================================
@st.cache_data(show_spinner=False)
def load_dr_single_file(base_dir: str, filename: str) -> pd.DataFrame:
    """pred/.csv を読み込み、日時・深さ整形。U,Vあれば速度・方位も付与。"""
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

    # U,V があれば速度・方位
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
        pred_sorted,
        obs_sorted,
        tol,
        right_value_cols=["obs_temp"],
        suffixes=("", ""),
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
# サイドバー（表示モード）
# =========================================
mode = st.sidebar.radio("表示モード", ["予測カレンダー"])

# キャッシュクリア（検証用）
if st.sidebar.button("補正キャッシュをクリア"):
    st.cache_data.clear()
    st.success("補正キャッシュをクリアしました。再計算します。")

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

    # --- 回帰補正の学習（必要時）— 終端を最新日に揃える（週表示用の既定）
    reg_depthwise, n_match_reg = None, None
    if use_correction:
        train_start_dt = pd.Timestamp(latest_day) - pd.Timedelta(days=train_days)
        train_end_dt = pd.Timestamp(latest_day) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
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
                st.write(
                    "学習済み水深（α, β）:",
                    {int(k): (round(v[0], 4), round(v[1], 4)) for k, v in (reg_depthwise or {}).items()}
                )

    # 深さの決定（補正ON＋係数がある水深のみ表示にする場合）
    depths_for_table = list(depths_all)
    if use_correction and (reg_depthwise is not None) and hide_uncorrected_depths:
        depths_for_table = [d for d in depths_for_table if int(d) in reg_depthwise]

    # 表示期間モード
    mode_view = st.radio("表示期間", ["週間予測（昼頃）", "選択日（1時間毎）"])

    # ===== スクロール対応CSS（stickyヘッダー＆先頭列） =====
    styles = """
<style>
  /* ラッパー：横スクロール許可 */
  .table-wrap {
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    background: #fff;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    min-width: 680px; /* 下限幅 */
  }
  th, td {
    padding: 6px 8px;
    border-bottom: 1px solid #eee;
    text-align: center;
    white-space: nowrap;
    font-size: 14px;
    line-height: 1.3;
  }
  th {
    background: #fafafa;
    position: sticky;
    top: 0;
    z-index: 2;
  }
  /* 先頭列（深さ）を固定 */
  td:first-child, th:first-child {
    position: sticky;
    left: 0;
    background: #fff;
    z-index: 3;
    font-weight: 600;
  }

  /* 補助クラス（速度や矢印） */
  .kv-small {
    font-size: 12px;
    color: #666;
    margin-left: 4px;
  }
  .arrow-svg {
    display: inline-block;
    vertical-align: middle;
    margin-left: 4px;
  }

  /* スマホ対応（768px以下） */
  @media (max-width: 768px) {
    th, td {
      padding: 4px 6px;
      font-size: 12px;
    }
    /* スマホでは速度・矢印を省略して情報量を減らす */
    .kv-small, .arrow-svg {
      display: none !important;
    }
    table {
      min-width: 600px;
    }
  }

  /* さらに狭い幅（480px以下）では数字を優先表示 */
  @media (max-width: 480px) {
    th, td {
      padding: 3px 5px;
      font-size: 11px;
    }
  }
</style>
"""

    # ===== テーブルHTMLの構築 =====
    def build_week_table() -> str:
        # ヘッダー（日付のみ）
        times = [d.strftime("%m/%d") for d in day_list]
        html = """
<div class="table-wrap">
<table>
<tr>
  <th>水深</th>
"""
        html += "".join([f"  <th>{t}</th>\n" for t in times])
        html += "</tr>\n"

        for depth in depths_for_table:
            html += f"<tr>\n<td>{depth}m</td>\n"
            for day in day_list:
                g = df_period[(df_period["date_day"] == day.date()) & (df_period["depth_m"] == depth)]
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
                    pred_html = f"{pred_label}"  # 値はそのまま

                    # 速度・方位（u,v がある場合のみ）
                    speed_html, arrow_svg = "", ""
                    if "Speed" in row.columns and "Direction_deg" in row.columns:
                        v_speed = row["Speed"].values[0]
                        v_dir = row["Direction_deg"].values[0]
                        if pd.notna(v_speed) and pd.notna(v_dir):
                            speed_kt = float(v_speed) * 1.94384
                            speed_kt_label = f"{speed_kt:.1f} kt"
                            speed_html = f'<span class="kv-small">{speed_kt_label}</span>'
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

                    cell_content = f"""
    {pred_html}
    {speed_html if speed_html else ""}
    {arrow_svg if arrow_svg else ""}
    {corr_value_html if (use_correction and corr_value_html) else ""}
"""
                    cell = f'<td style="background-color:{bg_color};">{cell_content}</td>\n'
                else:
                    cell = '<td style="background-color:rgba(220,220,220,0.6);">-</td>\n'

                html += cell
            html += "</tr>\n"
        html += "</table>\n</div>\n"
        return html

    def build_day_table() -> str:
        # 選択日を終端にして係数を再学習（必要時）
        available_days = sorted(df_dr["date_day"].unique())
        min_day = min(available_days) if available_days else latest_day
        max_day = max(available_days) if available_days else latest_day

        selected_day = st.date_input(
            "表示日（1時間毎）",
            value=max_day, min_value=min_day, max_value=max_day
        )
        df_day = df_dr[df_dr["date_day"] == selected_day].copy()
        hours_list = sorted(df_day["datetime"].dt.floor("h").unique())
        times_hr = [t.strftime("%H:%M") for t in hours_list]

        # 選択日終端で再学習
        nonlocal reg_depthwise, n_match_reg
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
            else:
                reg_depthwise = reg_depthwise_sel
                n_match_reg = n_match_reg_sel
                with st.expander(f"補正の学習情報（終端={selected_day:%Y-%m-%d}・遡り{train_days}日）", expanded=False):
                    st.write(f"期間: {sel_train_start_dt:%Y-%m-%d} 〜 {sel_train_end_dt:%Y-%m-%d}")
                    st.write("一致ペア数:", {int(k): int(v) for k, v in (n_match_reg or {}).items()})
                    st.write(
                        "学習済み水深（α, β）:",
                        {int(k): (round(v[0], 4), round(v[1], 4)) for k, v in (reg_depthwise or {}).items()}
                    )

        html = """
<div class="table-wrap">
<table>
<tr>
  <th>水深</th>
"""
        html += "".join([f"  <th>{t}</th>\n" for t in times_hr])
        html += "</tr>\n"

        for depth in depths_for_table:
            html += f"<tr>\n<td>{depth}m</td>\n"
            for time_obj in hours_list:
                row = df_day[
                    (df_day["datetime"].dt.floor("h") == time_obj) &
                    (df_day["depth_m"] == depth)
                ]
                if not row.empty:
                    temp = row["pred_temp"].astype(float).values[0] if "pred_temp" in row.columns else np.nan
                    pred_label = f"{temp:.1f}°C" if pd.notna(temp) else "NaN"
                    pred_html = f"{pred_label}"

                    # 速度・方位（u,v がある場合のみ）
                    speed_html, arrow_svg = "", ""
                    if "Speed" in row.columns and "Direction_deg" in row.columns:
                        v_speed = row["Speed"].values[0]
                        v_dir = row["Direction_deg"].values[0]
                        if pd.notna(v_speed) and pd.notna(v_dir):
                            speed_kt = float(v_speed) * 1.94384
                            speed_kt_label = f"{speed_kt:.1f} kt"
                            speed_html = f'<span class="kv-small">{speed_kt_label}</span>'
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

                    cell_content = f"""
    {pred_html}
    {speed_html if speed_html else ""}
    {arrow_svg if arrow_svg else ""}
    {corr_value_html if (use_correction and corr_value_html) else ""}
"""
                    cell = f'<td style="background-color:{bg_color};">{cell_content}</td>\n'
                else:
                    cell = '<td style="background-color:rgba(220,220,220,0.6);">-</td>\n'

                html += cell
            html += "</tr>\n"
        html += "</table>\n</div>\n"
        return html

    # モードに応じてHTMLを構築
    if mode_view == "週間予測（昼頃）":
        table_html = build_week_table()
    else:
        table_html = build_day_table()

    # ===== テーブルHTML終了・描画 =====
    full_html = f"""{styles}{table_html}"""
    iframe_height = int(max(400, min(1100, max_h_vh * 10)))
    st_html(full_html, height=iframe_height, scrolling=True)