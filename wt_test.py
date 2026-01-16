# -*- coding: utf-8 -*-
import os
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html
import plotly.graph_objects as go
import plotly.express as px

# =========================================
# 設定（フォルダ固定）
# =========================================
from pathlib import Path

BASE_DIR = str(Path(__file__).parent.joinpath("data").resolve())
PRED_DIR = "pred"
OBS_DIR  = "obs"
CORR_DIR = "corr" 


# 固定パラメータ
RECENT_DAYS = 7  # 直近8日（週間）
OUTLIER_TH = 4.0  # 観測なし時: corr - pred の閾値
OUTLIER_TH_OBS = 2.0  # 観測あり時: corr - obs の閾値
OBS_MATCH_TOL_MIN = 60  # 観測近傍マージ許容（分）
CORR_MATCH_TOL_MIN = 60  # 補正近傍マージ許容（分）
TEMP_MIN, TEMP_MAX = -2.0, 40.0
PHYS_MIN, PHYS_MAX = -1.5, 35.0
HIGH_TEMP_TH = 22.0  # コメント用
RANGE_STABLE = 0.6
DELTA_THRESH = 0.3


def pjoin(*parts: str) -> str:
    return os.path.normpath(os.path.join(*parts))


# =========================================
# ユーティリティ
# =========================================
from pathlib import Path
import os
import numpy as np
import pandas as pd
import streamlit as st
from typing import Optional, Tuple, List


def _pick_series_corr_then_pred(g: pd.DataFrame) -> Optional[pd.Series]:
    """
    corr が列として存在し、かつ有効値が1つ以上あれば corr を採用。
    そうでなければ pred。どちらもダメなら None。
    """
    cand = None
    if "corr_temp" in g.columns:
        c = pd.to_numeric(g["corr_temp"], errors="coerce")
        if c.notna().sum() >= 1:
            cand = c
    if cand is None and "pred_temp" in g.columns:
        p = pd.to_numeric(g["pred_temp"], errors="coerce")
        if p.notna().sum() >= 1:
            cand = p
    return cand

def utc_to_jst_naive(s: pd.Series) -> pd.Series:
    """UTCとして解釈 → JSTへ変換 → タイムゾーン情報を外す（naive）"""
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    dt = dt.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return dt

def jst_to_naive(s: pd.Series) -> pd.Series:
    """ローカル／JST相当の文字列→pandas日時→（もしtz付きなら）JSTへ変換→naive化"""
    dt = pd.to_datetime(s, errors="coerce", utc=False)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    return dt

def safe_merge_asof_by_depth_keep_left(
    left: pd.DataFrame,
    right: pd.DataFrame,
    tolerance: pd.Timedelta,
    right_value_cols: List[str],
    suffixes: Tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame:
    """
    depth_m ごとに nearest で asof マージする。
    右側にデータが無い深さは NaN をパディング、左側の行は保持（keep-left）。
    """
    out_list: List[pd.DataFrame] = []
    left_depths = sorted(set(left["depth_m"].dropna().unique()))

    for d in left_depths:
        l = left[left["depth_m"] == d].sort_values("datetime")
        r = right[right["depth_m"] == d].sort_values("datetime")[["datetime", "depth_m"] + right_value_cols]
        if l.empty:
            continue
        if r.empty:
            pad = l.copy()
            for c in right_value_cols:
                pad[c] = np.nan
            out_list.append(pad)
        else:
            merged = pd.merge_asof(
                l, r, on="datetime", by="depth_m",
                tolerance=tolerance, direction="nearest", suffixes=suffixes
            )
            out_list.append(merged)

    if not out_list:
        out = left.copy()
        for c in right_value_cols:
            if c not in out.columns:
                out[c] = np.nan
        return out
    return pd.concat(out_list, ignore_index=True)

def _detect_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """
    ['corr','temp'] のようなキーワード候補から最も合致する列名を推定する。
    完全一致 → 正規化（_除去・小文字化）包含 の順で探索。
    """
    cols = list(df.columns)
    # 完全一致
    for c in cols:
        if c.lower() in [k.lower() for k in keywords]:
            return c
    # 正規化（_ 除去）
    norm = {c: c.lower().replace("_", "") for c in cols}
    for c, n in norm.items():
        ok = all(k.lower().replace("_", "") in n for k in keywords)
        if ok:
            return c
    return None

def to_rgba(color: str, alpha: float = 0.18) -> str:
    """
    '#rrggbb' / 'rgb(r,g,b)' / 'rgba(r,g,b,a)' を RGBA 文字列に正規化し、alpha を差し替える。
    不正値は緑系のデフォルトを返す。
    """
    if not isinstance(color, str) or not color:
        return f"rgba(0,150,0,{alpha})"
    c = color.strip().lower()

    if c.startswith("rgba(") and c.endswith(")"):
        try:
            nums = c[5:-1].split(",")
            r, g, b = [int(float(x)) for x in nums[:3]]
            return f"rgba({r},{g},{b},{alpha})"
        except Exception:
            return f"rgba(0,150,0,{alpha})"

    if c.startswith("rgb(") and c.endswith(")"):
        try:
            r, g, b = [int(float(x)) for x in c[4:-1].split(",")[:3]]
            return f"rgba({r},{g},{b},{alpha})"
        except Exception:
            return f"rgba(0,150,0,{alpha})"

    if c.startswith("#"):
        h = c.lstrip("#")
        try:
            if len(h) == 3:
                r = int(h[0]*2, 16); g = int(h[1]*2, 16); b = int(h[2]*2, 16)
            elif len(h) == 6:
                r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
            else:
                return f"rgba(0,150,0,{alpha})"
            return f"rgba({r},{g},{b},{alpha})"
        except Exception:
            return f"rgba(0,150,0,{alpha})"

    return c

# ---- ここから：キャッシュ無効化のための「ファイル指紋」ユーティリティ ----
def file_fingerprint(path: str) -> str:
    """
    任意パスの存在/mtime/サイズを文字列化（キャッシュキー用）。
    存在しなければ 'missing'。
    """
    p = Path(path)
    if not p.exists():
        return "missing"
    try:
        st_ = p.stat()
        return f"mtime:{int(st_.st_mtime)}:size:{st_.st_size}"
    except Exception:
        return "exists"

def obs_fingerprint(base_dir: str, obs_dir: str, filename: str) -> str:
    """
    OBS ファイルの指紋。load_obs_for(...) の第2引数 fp に渡すこと。
    """
    path = os.path.normpath(os.path.join(base_dir, obs_dir, filename))
    return file_fingerprint(path)
# ---- ここまで ----

@st.cache_data(show_spinner=False)
def load_pred(filename: str) -> pd.DataFrame:
    """
    予測（pred）CSV を読み込む。
    ※ pred は削除される運用が少ないため、キャッシュキー拡張は現状不要。
       変更検知が必要なら file_fingerprint を第2引数に増やす設計に合わせてください。
    """
    path = pjoin(BASE_DIR, PRED_DIR, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.DataFrame()

    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = utc_to_jst_naive(df.get("Date"))
    df["depth_m"]  = pd.to_numeric(df.get("Depth"), errors="coerce").round(0).astype("Int64")
    df = df.rename(columns={"Temp": "pred_temp"})

    if ("U" in df.columns) and ("V" in df.columns):
        df["U"] = pd.to_numeric(df["U"], errors="coerce")
        df["V"] = pd.to_numeric(df["V"], errors="coerce")
        df["Speed"] = np.sqrt(np.square(df["U"]) + np.square(df["V"]))
        df["Direction_deg"] = (np.degrees(np.arctan2(df["U"], df["V"])) + 360.0) % 360.0

    df = df.dropna(subset=["datetime", "depth_m"]).copy()
    df["date_day"] = df["datetime"].dt.date
    return df


# 1) 既存: file_fingerprint は実装済み。これを pred/corr 読み込みにも使う  ← 既にあります
# def file_fingerprint(path: str) -> str: ...

# 2) pred/corr/obs ローダのシグネチャを「fp」を受ける形に変更し、キーに混ぜる
@st.cache_data(show_spinner=False)
def load_pred(filename: str, fp: str | None = None) -> pd.DataFrame:
    path = pjoin(BASE_DIR, PRED_DIR, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = utc_to_jst_naive(df.get("Date"))
    df["depth_m"] = pd.to_numeric(df.get("Depth"), errors="coerce").round(0).astype("Int64")
    df = df.rename(columns={"Temp": "pred_temp"})
    if ("U" in df.columns) and ("V" in df.columns):
        df["U"] = pd.to_numeric(df["U"], errors="coerce")
        df["V"] = pd.to_numeric(df["V"], errors="coerce")
        df["Speed"] = np.sqrt(np.square(df["U"]) + np.square(df["V"]))
        df["Direction_deg"] = (np.degrees(np.arctan2(df["U"], df["V"])) + 360.0) % 360.0
    df = df.dropna(subset=["datetime", "depth_m"]).copy()
    df["date_day"] = df["datetime"].dt.date
    return df

@st.cache_data(show_spinner=False)
def load_corr_for(filename: str, fp: str | None = None) -> pd.DataFrame:
    name, ext = os.path.splitext(filename)
    corr_filename = f"{name}_corr{ext}"
    path = pjoin(BASE_DIR, CORR_DIR, corr_filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = jst_to_naive(df.get("Date"))
    df["depth_m"] = pd.to_numeric(df.get("Depth"), errors="coerce").round(0).astype("Int64")

    corr_col = _detect_column(df, ["corr", "temp"]) or ("CorrTemp" if "CorrTemp" in df.columns else None)
    if corr_col is None:
        corr_col = "Temp" if "Temp" in df.columns else None
    if corr_col is None:
        return pd.DataFrame()

    low_col = _detect_column(df, ["corr", "low"]) or ("CorrLow" if "CorrLow" in df.columns else None)
    high_col = _detect_column(df, ["corr", "high"]) or ("CorrHigh" if "CorrHigh" in df.columns else None)
    rename_map = {corr_col: "corr_temp"}
    if low_col: rename_map[low_col] = "corr_low"
    if high_col: rename_map[high_col] = "corr_high"
    df = df.rename(columns=rename_map)

    keep = ["datetime", "depth_m", "corr_temp"]
    if "corr_low" in df.columns: keep.append("corr_low")
    if "corr_high" in df.columns: keep.append("corr_high")
    df = df[keep].dropna(subset=["datetime", "depth_m", "corr_temp"]).copy()
    df["date_day"] = df["datetime"].dt.date
    return df

@st.cache_data(show_spinner=False)
def load_obs_for(filename: str, fp: str | None = None) -> pd.DataFrame:
    path = pjoin(BASE_DIR, OBS_DIR, filename)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    df["datetime"] = jst_to_naive(df.get("Date"))
    df["depth_m"] = pd.to_numeric(df.get("Depth"), errors="coerce").round(0).astype("Int64")
    df = df.rename(columns={"Temp": "obs_temp"})
    df = df.dropna(subset=["datetime", "depth_m"]).copy()
    df["date_day"] = df["datetime"].dt.date
    return df


def add_corr(df_pred: pd.DataFrame, df_corr: pd.DataFrame) -> pd.DataFrame:
    """
    pred へ corr を depth_m&datetime で近傍（±CORR_MATCH_TOL_MIN 分）マージし、
    corr_temp（＋あれば corr_low / corr_high）を付加する。
    corr が空なら pred の行をそのまま返し、corr_* 列だけ NaN で補う。
    """
    if df_pred.empty or df_corr.empty:
        out = df_pred.copy()
        if "corr_temp" not in out.columns:
            out["corr_temp"] = np.nan
        if "corr_low" not in out.columns:
            out["corr_low"] = np.nan
        if "corr_high" not in out.columns:
            out["corr_high"] = np.nan
        return out

    tol = pd.Timedelta(minutes=CORR_MATCH_TOL_MIN)

    right_cols = ["corr_temp"]
    if "corr_low"  in df_corr.columns: right_cols.append("corr_low")
    if "corr_high" in df_corr.columns: right_cols.append("corr_high")

    right = (
        df_corr.sort_values(["depth_m", "datetime"])[["datetime", "depth_m"] + right_cols]
    )
    left = df_pred.sort_values(["depth_m", "datetime"]).copy()

    merged = safe_merge_asof_by_depth_keep_left(
        left, right, tol, right_value_cols=right_cols, suffixes=("", "")
    )
    return merged


# ---- 余白を詰めるグローバルCSSを注入（全画面に一度だけ）----

from streamlit.components.v1 import html as st_html

def inject_compact_css():
    compact_css = """
    <style>
      /* ===== 1) ヘッダー/フッター/デプロイバッジを実質削除 ===== */
      /* 新UI: アプリ上部のヘッダー */
      [data-testid="stHeader"], header, .stAppHeader { display: none !important; height: 0 !important; }
      /* フッター（Streamlitの著作権やバッジ） */
      footer, #MainMenu, .viewerBadge_container__1QSob { display: none !important; }

      /* ===== 2) ページ上下パディングを強めに圧縮 ===== */
      .block-container {
        padding-top: 2px !important;
        padding-bottom: 2px !important;
      }

      /* ===== 3) 横並びブロック（segmented_control 周辺）のギャップ/余白を圧縮 ===== */
      div[data-testid="stHorizontalBlock"] {
        gap: 4px !important;
        margin-top: 0 !important;
        margin-bottom: 4px !important;
      }

      /* ===== 4) 主要ウィジェット上/下のマージンをさらに圧縮 ===== */
      div[data-testid="stSegmentedControl"],
      div[data-testid="stRadio"],
      div[data-testid="stSelectbox"],
      div[data-testid="stDateInput"],
      div[data-testid="stMultiSelect"],
      div[data-testid="stSlider"],
      div[data-testid="stNumberInput"] {
        margin-top: 0 !important;
        margin-bottom: 4px !important;
      }

      /* ===== 5) 縦積みブロック（本体/サイドバー）ギャップ縮小 ===== */
      div[data-testid="stVerticalBlock"],
      section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 4px !important;
      }

      /* ===== 6) Markdown（コメント）の行間/段落間をさらに詰める ===== */
      .stMarkdown p { margin: 1px 0 !important; line-height: 1.18 !important; }
      .stMarkdown ul, .stMarkdown ol { margin-top: 1px !important; margin-bottom: 1px !important; }
      .stMarkdown li { margin: 0 0 1px 0 !important; line-height: 1.18 !important; }

      /* ===== 7) あなたのカレンダー表のセル余白を縮める ===== */
      .calendar-table th, .calendar-table td { padding: 3px 6px !important; }

      /* ===== 8) モバイル幅でさらに圧縮 ===== */
      @media (max-width: 480px) {
        .block-container { padding-top: 1px !important; padding-bottom: 1px !important; }
        div[data-testid="stHorizontalBlock"] { gap: 3px !important; margin-top: 0 !important; margin-bottom: 3px !important; }
        div[data-testid="stSegmentedControl"],
        div[data-testid="stRadio"],
        div[data-testid="stSelectbox"],
        div[data-testid="stDateInput"],
        div[data-testid="stMultiSelect"],
        div[data-testid="stSlider"],
        div[data-testid="stNumberInput"] {
          margin-top: 0 !important;
          margin-bottom: 3px !important;
        }
        .stMarkdown p { line-height: 1.14 !important; }
      }
    </style>
    """
    st_html(compact_css, height=0)

# =========================================
# カレンダー表示の部品（補正値＆矢印を表示）
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

def get_color(temp: float, t_min: float = 0.0, t_max: float = 25.0) -> str:
    if pd.isna(temp): return "rgba(220,220,220,0.4)"
    ratio = (float(temp) - t_min) / (t_max - t_min)
    ratio = max(0, min(1, ratio))
    if ratio < 0.5:
        r = int(240 * ratio * 2); g = int(240 * ratio * 2); b = 240
    else:
        r = 240; g = int(240 * (1 - (ratio - 0.5) * 2)); b = int(240 * (1 - (ratio - 0.5) * 2))
    return f"rgba({r},{g},{b},0.4)"

def get_calendar_css(max_h_vh: int = 65) -> str:
    return f"""
    <style>
    .calendar-scroll-container {{
      overflow-x: auto; overflow-y: auto;           /* 横／縦スクロール */
      max-height: {max_h_vh}vh; max-width: 100%;
      -webkit-overflow-scrolling: touch;
      border: 1px solid #e5e5e5; border-radius: 8px;
      isolation: isolate;                           /* z-indexの重なり安定化 */
    }}
    .calendar-table {{
      border-collapse: separate;                    /* ← stickyの安定化（Safari等） */
      border-spacing: 0;
      width: max-content; min-width: 640px; font-size: 14px;
    }}
    .calendar-table th, .calendar-table td {{
      padding: 6px 10px;
      border-bottom: 1px solid #eee;
      text-align: center;
      white-space: nowrap;
    }}

    /* 1行目（ヘッダ行）を上側に固定 */
    .calendar-table thead th {{
      position: sticky; top: 0;
      background: #fafafa; z-index: 2;
    }}

    /* 1列目（水深）を左側に固定（th/td両対応）＋中央＋太字 */
    .calendar-table tbody th.depth-cell,
    .calendar-table tbody td.depth-cell {{
      position: sticky; left: 0;
      background: #f7f7f7;                           /* 背景必須（下のセルと重なるため） */
      z-index: 3;                                    /* ヘッダより前面に */
      min-width: 56px;
      text-align: center;
      font-weight: 700 !important;                   /* 太字を強制 */
    }}

    /* thead の先頭セル（左上の角）も左固定にして重なりを整える */
    .calendar-table thead th:first-child {{
      position: sticky; left: 0; top: 0;
      background: #f0f0f0; z-index: 4;               /* 交点セルは最前面 */
      min-width: 56px; text-align: center;
      font-weight: 700;
    }}

    .calendar-table .pred-small {{ font-size: 12px; color: #555; }}
    </style>
    """.strip()

def correction_effective(
    temp_pred: Optional[float],
    temp_corr: Optional[float],
    temp_obs: Optional[float] = None
) -> bool:
    if temp_pred is None or pd.isna(temp_pred): return False
    if temp_corr is None or pd.isna(temp_corr): return False
    if not (PHYS_MIN < float(temp_corr) < PHYS_MAX): return False
    if not (TEMP_MIN < float(temp_corr) < TEMP_MAX): return False
    if (temp_obs is not None) and (not pd.isna(temp_obs)):
        return abs(float(temp_corr) - float(temp_obs)) < OUTLIER_TH_OBS
    else:
        return abs(float(temp_corr) - float(temp_pred)) < OUTLIER_TH

def render_cell_html(
    temp_pred: Optional[float],
    speed_mps: Optional[float],
    dir_deg: Optional[float],
    temp_corr_raw: Optional[float],
    corr_on: bool,
    temp_obs: Optional[float] = None,
) -> str:
    corr_ok = corr_on and correction_effective(temp_pred, temp_corr_raw, temp_obs=temp_obs)
    bg_value = float(temp_corr_raw) if corr_ok else (float(temp_pred) if temp_pred is not None else np.nan)
    bg_color = get_color(bg_value) if not pd.isna(bg_value) else "rgba(220,220,220,0.6)"

    pred_label = f"{float(temp_pred):.1f}°C" if (temp_pred is not None and not pd.isna(temp_pred)) else "NaN"
    pred_html = f"<span class='pred-small'>{pred_label}</span>"

    speed_html, arrow_html = "", ""
    if (speed_mps is not None and not pd.isna(speed_mps)) and (dir_deg is not None and not pd.isna(dir_deg)):
        speed_kt = float(speed_mps) * 1.94384
        speed_html = f"<span style='font-size:12px;color:#444;'>{speed_kt:.1f} kt</span>"
        arrow_html = f"<span style='display:block;line-height:1;margin:0;padding:0;'>{get_arrow_svg(float(dir_deg), float(speed_mps))}</span>"

    corr_html = ""
    if corr_on and (temp_corr_raw is not None) and not pd.isna(temp_corr_raw) and corr_ok:
        corr_html = f"<span style='color:#D32F2F;font-weight:700;font-size:14px;'>{float(temp_corr_raw):.1f}°C</span>"

    content = (
        "<div style='display:flex;flex-direction:column;align-items:center;gap:2px;'>"
        + pred_html + speed_html + arrow_html + corr_html + "</div>"
    )
    return f"<td style='background:{bg_color}'>{content}</td>"

def build_weekly_table_html(df_period: pd.DataFrame, day_list: List[pd.Timestamp], depths: List[int], corr_on: bool) -> str:
    times = [d.strftime('%m/%d') for d in day_list]
    html = (
        '<div class="calendar-scroll-container"><table class="calendar-table">'
        "<thead><tr><th>水深</th>" + "".join([f"<th>{t}</th>" for t in times]) + "</tr></thead><tbody>"
    )
    for depth in depths:
        html += f"<tr><td class='depth-cell'>{depth}m</td>"
        for day in day_list:
            g = df_period[(df_period["date_day"] == day.date()) & (df_period["depth_m"] == depth)]
            if not g.empty:
                target_dt = pd.Timestamp(day.date()) + pd.Timedelta(hours=12)
                row = g.assign(_diff=(g["datetime"] - target_dt).abs()).sort_values("_diff").iloc[[0]]
                temp_ark = row  # alias
                temp_pred = float(temp_ark["pred_temp"].values[0]) if "pred_temp" in temp_ark.columns else np.nan
                speed_val = float(temp_ark["Speed"].values[0]) if "Speed" in temp_ark.columns else np.nan
                dir_val = float(temp_ark["Direction_deg"].values[0]) if "Direction_deg" in temp_ark.columns else np.nan
                temp_corr = float(temp_ark["corr_temp"].values[0]) if "corr_temp" in temp_ark.columns else None
                temp_obs = float(temp_ark["obs_temp"].values[0]) if ("obs_temp" in temp_ark.columns and not pd.isna(temp_ark["obs_temp"].values[0])) else None
                html += render_cell_html(temp_pred, speed_val, dir_val, temp_corr, corr_on, temp_obs=temp_obs)
            else:
                html += "<td>-</td>"
        html += "</tr>\n"
    html += "</tbody></table></div>"
    return html

def build_daily_table_html(df_day: pd.DataFrame, depths: List[int], corr_on: bool) -> str:
    hours_list = sorted(df_day["datetime"].dt.floor("h").unique())
    times_hr = [t.strftime('%H:%M') for t in hours_list]
    html = (
        '<div class="calendar-scroll-container"><table class="calendar-table">'
        "<thead><tr><th>水深</th>" + "".join([f"<th>{t}</th>" for t in times_hr]) + "</tr></thead><tbody>"
    )
    for depth in depths:
        html += f"<tr><td class='depth-cell'>{depth}m</td>"
        for t_obj in hours_list:
            row = df_day[(df_day["datetime"].dt.floor("h") == t_obj) & (df_day["depth_m"] == depth)]
            if not row.empty:
                temp_pred = float(row["pred_temp"].values[0]) if "pred_temp" in row.columns else np.nan
                speed_val = float(row["Speed"].values[0]) if "Speed" in row.columns else np.nan
                dir_val = float(row["Direction_deg"].values[0]) if "Direction_deg" in row.columns else np.nan
                temp_corr = float(row["corr_temp"].values[0]) if "corr_temp" in row.columns else None
                temp_obs = float(row["obs_temp"].values[0]) if ("obs_temp" in row.columns and not pd.isna(row["obs_temp"].values[0])) else None
                html += render_cell_html(temp_pred, speed_val, dir_val, temp_corr, corr_on, temp_obs=temp_obs)
            else:
                html += "<td>-</td>"
        html += "</tr>\n"
    html += "</tbody></table></div>"
    return html

def make_layer_groups(depths: List[int]) -> Dict[str, List[int]]:
    if not depths: return {"表層": [], "中層": [], "底層": []}
    d_sorted = sorted(depths); n = len(d_sorted)
    if n <= 3:
        top = d_sorted[:1]; mid = d_sorted[1:2] if n >= 2 else []; bot = d_sorted[2:] if n >= 3 else (d_sorted[-1:] if n >= 1 else [])
    elif n in (4, 5):
        top = d_sorted[:2]; mid = d_sorted[2:3]; bot = d_sorted[3:]
    else:
        top = d_sorted[:2]; bot = d_sorted[-2:]
        mid = [d for d in d_sorted if d not in top + bot]
        if len(mid) >= 3:
            c = len(mid) // 2
            mid = mid[c-1:c+1]
    return {"表層": top, "中層": mid, "底層": bot}


def summarize_weekly_for_depth(layer_name: str, target_depth: int, df_period: pd.DataFrame) -> Optional[str]:
    """
    指定した 'target_depth' 1本だけで週間コメントを作る。
    corr_temp があれば優先、無ければ pred_temp を使う。
    """
    if df_period.empty or "depth_m" not in df_period.columns:
        return None

    g = df_period[df_period["depth_m"] == int(target_depth)].sort_values("datetime")
    if g.empty:
        return None

    # ✅ corr優先（有効値が無ければpredへフォールバック）
    series = _pick_series_corr_then_pred(g)
    if series is None:
        return None

    temps = pd.to_numeric(series, errors="coerce")
    temps = temps[(temps > PHYS_MIN) & (temps < PHYS_MAX)].dropna()
    if temps.empty:
        return None

    # 高水温（22℃単独：既存ポリシーに合わせて最小変更）
    t_min, t_max = float(temps.min()), float(temps.max())
    if t_max >= HIGH_TEMP_TH:
        tag = f":red[高水温]（{t_min:.1f}℃～{t_max:.1f}℃）"
        return f"**{layer_name}**： {int(target_depth)}m{tag}"

    # ✅ まずレンジで「安定」を早決
    weekly_range = t_max - t_min
    if weekly_range < RANGE_STABLE:
        # 既存の“安定（xx.x℃）”の表現を維持（代表値は中央値に変更可だが最小差分で開始値を使用）
        t_mid = float(temps.iloc[0])  # ここを temps.median() にするとより頑健、ただし今回は最小差分で据え置き可
        tag = f"安定（{t_mid:.1f}℃）"
        return f"**{layer_name}**： {int(target_depth)}m{tag}"

    # ✅ 前半/後半の平均差で方向判定（Day1–3 vs Day5–7、Day4は緩衝）
    idx_first = [i for i in [0,1,2] if i < len(temps)]
    idx_last  = [i for i in [4,5,6] if i < len(temps)]
    first3 = temps.iloc[idx_first] if idx_first else temps.iloc[:max(1, len(temps)//2)]
    last3  = temps.iloc[idx_last]  if idx_last  else temps.iloc[max(1, len(temps)//2):]

    delta = float(last3.mean() - first3.mean())

    # 基本判定
    if delta > +DELTA_THRESH:
        tag = f"上昇（{float(temps.iloc[0]):.1f}℃→{float(temps.iloc[-1]):.1f}℃）"
    elif delta < -DELTA_THRESH:
        tag = f"下降（{float(temps.iloc[0]):.1f}℃→{float(temps.iloc[-1]):.1f}℃）"
    else:
        # タイブレーク：端点差を最小限で利用
        end_diff = float(temps.iloc[-1] - temps.iloc[0])
        if abs(end_diff) >= DELTA_THRESH:
            tag = f"{'上昇' if end_diff > 0 else '下降'}（{float(temps.iloc[0]):.1f}℃→{float(temps.iloc[-1]):.1f}℃）"
        else:
            tag = f"安定（{float(temps.iloc[0]):.1f}℃）"

    return f"**{layer_name}**： {int(target_depth)}m{tag}"

def pick_shallow_mid_deep_min10_from_depths(depths: List[int]) -> List[int]:
    """
    利用可能な深さの配列から、浅・中・深の3層代表を返す（10m起算）。
    - 浅: 10m以上の最小値。なければ最浅。
    - 深: 最深。
    - 中: 浅と深の中間の順位（偶数は下側）。
    - 候補が2以下なら、その分だけ返す。
    """
    if not depths:
        return []
    xs = sorted(set(int(d) for d in depths))
    n = len(xs)
    if n <= 2:
        return xs

    # 浅…10m以上の最小値を優先、無ければ最浅
    low_idx = 0
    for i, d in enumerate(xs):
        if d >= 10:
            low_idx = i
            break

    high_idx = n - 1
    mid_idx = (low_idx + high_idx) // 2  # 偶数は下側

    chosen = [xs[low_idx], xs[mid_idx], xs[high_idx]]
    return sorted(set(chosen))


def summarize_weekly_layer_temp(layer_name: str, layer_depths: List[int], df_period: pd.DataFrame) -> Optional[str]:

    if not layer_depths or df_period.empty or "depth_m" not in df_period.columns:
        return None

    valid_depths = set(pd.to_numeric(df_period["depth_m"], errors="coerce").dropna().astype(int))
    depths_in_data = sorted(int(d) for d in layer_depths if int(d) in valid_depths)
    if not depths_in_data:
        return None

    # レイヤー集合から「浅・中・深（10m起算）」の候補3本を抽出
    smd = pick_shallow_mid_deep_min10_from_depths(depths_in_data)
    if not smd:
        return None

    # レイヤー名に応じて代表1本だけ選ぶ
    if layer_name == "表層":
        target_depth = smd[0]                    # 浅（10m以上の最小が優先）
    elif layer_name == "中層":
        target_depth = smd[min(1, len(smd)-1)]   # 中（候補が2以下でも破綻しない）
    else:
        target_depth = smd[-1]                   # 底層＝深

    # 代表深度の系列（corr優先：列があり有効値が1つ以上あればcorr、無ければpred）
    g = df_period[df_period["depth_m"] == target_depth].sort_values("datetime")
    if g.empty:
        return None

    series = None
    if "corr_temp" in g.columns:
        c = pd.to_numeric(g["corr_temp"], errors="coerce")
        if c.notna().sum() >= 1:
            series = c
    if series is None and "pred_temp" in g.columns:
        p = pd.to_numeric(g["pred_temp"], errors="coerce")
        if p.notna().sum() >= 1:
            series = p
    if series is None:
        return None

    temps = pd.to_numeric(series, errors="coerce")
    temps = temps[(temps > PHYS_MIN) & (temps < PHYS_MAX)].dropna()
    if temps.empty:
        return None

    # しきい値（グローバル定数があればそれを使い、無ければ既定値でフォールバック）
    try:
        rng_th = float(RANGE_STABLE)  # 例：0.6
    except Exception:
        rng_th = 0.6
    try:
        dlt_th = float(DELTA_THRESH)  # 例：0.3
    except Exception:
        dlt_th = 0.3

    # 高水温（22℃等の絶対閾値のみ。p80併用は別途配線）
    t_min, t_max = float(temps.min()), float(temps.max())
    if t_max >= HIGH_TEMP_TH:
        tag = f":red[高水温]（{t_min:.1f}℃～{t_max:.1f}℃）"
        return f"**{layer_name}**： {target_depth}m{tag}"

    # まず週レンジで「安定」を早決
    weekly_range = t_max - t_min
    if weekly_range < rng_th:
        # 既存表現を踏襲：開始値で表示（中央値にしたい場合は temps.median() に変更可能）
        t_start = float(temps.iloc[0])
        tag = f"安定（{t_start:.1f}℃）"
        return f"**{layer_name}**： {target_depth}m{tag}"

    # 前半/後半の平均差（Day1–3 vs Day5–7、Day4は緩衝。データ不足時は前半/後半でフォールバック）
    n = len(temps)
    idx_first = [i for i in [0, 1, 2] if i < n]
    idx_last  = [i for i in [4, 5, 6] if i < n]

    if idx_first and idx_last:
        first = temps.iloc[idx_first]
        last  = temps.iloc[idx_last]
    else:
        k = max(1, n // 2)
        first = temps.iloc[:k]
        last  = temps.iloc[k:] if k < n else temps.iloc[-1:]

    delta = float(last.mean() - first.mean())

    t_start = float(temps.iloc[0])
    t_end   = float(temps.iloc[-1])

    if delta > +dlt_th:
        tag = f"上昇（{t_start:.1f}℃→{t_end:.1f}℃）"
    elif delta < -dlt_th:
        tag = f"下降（{t_start:.1f}℃→{t_end:.1f}℃）"
    else:
        # タイブレーク：端点差を最小限で利用
        end_diff = t_end - t_start
        if abs(end_diff) >= dlt_th:
            tag = f"{'上昇' if end_diff > 0 else '下降'}（{t_start:.1f}℃→{t_end:.1f}℃）"
        else:
            tag = f"安定（{t_start:.1f}℃）"

    return f"**{layer_name}**： {target_depth}m{tag}"

def dir_to_8pt_jp(deg: float) -> str:
    if pd.isna(deg): return ""
    dirs = ["北", "北東", "東", "南東", "南", "南西", "西", "北西"]
    idx = int(((float(deg) + 22.5) % 360) // 45)
    return dirs[idx]

def speed_class_from_mps(v_mps: Optional[float]) -> str:
    if v_mps is None or pd.isna(v_mps): return ""
    kt = float(v_mps) * 1.94384
    if kt >= 1.5: return "速"
    if kt >= 0.5: return "中"
    return "穏"

def summarize_daily_layer_flow(
    layer_name: str,
    layer_depths: List[int],
    df_day: pd.DataFrame,
    use_short_labels: bool = True,
    merge_same_segments: bool = False
) -> Optional[str]:
    if not layer_depths: return None
    DAY_BINS = [("朝", 4, 6), ("昼", 11, 13), ("夕", 16, 18)]
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
            v_map = {"穏やか": "穏", "中程度": "中", "速い": "速"}
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


inject_compact_css()

# =========================================
# メインUI：表示モード（ラベル非表示）
# =========================================

try:
    view_mode = st.segmented_control(
        "",  # ← ラベル非表示
        options=["予測カレンダー", "水温グラフ"],
        default="予測カレンダー"
    )
except Exception:
    view_mode = st.radio(
        "",  # ← ラベル非表示
        ["予測カレンダー", "水温グラフ"],
        index=0, horizontal=True, label_visibility="collapsed"
    )

# pred ファイル選択（ラベル非表示）
pred_folder = pjoin(BASE_DIR, PRED_DIR)
if not os.path.exists(pred_folder):
    st.error(f"フォルダが見つかりません: {pred_folder}")
    st.stop()
pred_files = [f for f in os.listdir(pred_folder) if f.endswith(".csv")]
if not pred_files:
    st.warning("pred に CSV がありません")
    st.stop()
selected_file = st.selectbox(
    "", sorted(pred_files), key="sel_pred_file", label_visibility="collapsed"  # ← ラベル非表示
)

pred_path = pjoin(BASE_DIR, PRED_DIR, selected_file)
corr_name, ext = os.path.splitext(selected_file)
corr_path = pjoin(BASE_DIR, CORR_DIR, f"{corr_name}_corr{ext}")
obs_path  = pjoin(BASE_DIR, OBS_DIR,  selected_file)

fp_pred = file_fingerprint(pred_path)
fp_corr = file_fingerprint(corr_path)
fp_obs  = file_fingerprint(obs_path)


# =========================================
# 予測カレンダー
# =========================================
if view_mode == "予測カレンダー":
    df_pred = load_pred(selected_file)
    df_corr = load_corr_for(selected_file)
    df_obs = load_obs_for(selected_file)
    corr_available = not df_corr.empty

    if df_pred.empty:
        st.warning("予測データが読み込めませんでした")
        st.stop()

    latest_day = df_pred["date_day"].max()
    available_days = sorted(df_pred["date_day"].unique())
    min_day = min(available_days) if available_days else latest_day
    max_day = max(available_days) if available_days else latest_day

    # 表示期間（ラベル非表示）
    try:
        cal_choice = st.segmented_control(
            "", options=["週間表示（昼頃）", "選択日（1時間毎）"], default="週間表示", key="cal_choice"  # ← ラベル非表示
        )
    except Exception:
        cal_choice = st.radio(
            "", ["週間表示（昼頃）", "選択日（1時間毎）"],
            index=0, horizontal=True, key="cal_choice_radio", label_visibility="collapsed"  # ← ラベル非表示
        )

    if cal_choice == "週間表示（昼頃）":
        selected_day = st.date_input(
            "", value=max_day, min_value=min_day, max_value=max_day, key="week_base_day", label_visibility="collapsed"  # ← ラベル非表示
        )
        start_day = pd.Timestamp(selected_day) - pd.Timedelta(days=RECENT_DAYS - 1)
        end_day = pd.Timestamp(selected_day)
        day_list = list(pd.date_range(start_day, end_day, freq="D"))
        df_period = df_pred[df_pred["date_day"].isin([d.date() for d in day_list])].copy()

        if corr_available:
            df_corr_period = df_corr[df_corr["date_day"].isin([d.date() for d in day_list])].copy()
            df_period = add_corr(df_period, df_corr_period)

        if not df_obs.empty and not df_period.empty:
            df_obs_week = df_obs[df_obs["date_day"].between(day_list[0].date(), day_list[-1].date())].copy()
            tol_obs = pd.Timedelta(minutes=OBS_MATCH_TOL_MIN)
            left = df_period.sort_values(["depth_m", "datetime"]).copy()
            right = df_obs_week.sort_values(["depth_m", "datetime"])[["datetime", "depth_m", "obs_temp"]].copy()
            merged = safe_merge_asof_by_depth_keep_left(left, right, tolerance=tol_obs, right_value_cols=["obs_temp"], suffixes=("", ""))
            if "obs_temp" in merged.columns:
                df_period = merged


        # 週間コメント（各レイヤー＝1本だけ。浅=10m起算・中=中間・深=最深）
        depths_all = sorted([int(d) for d in df_pred["depth_m"].dropna().unique()])
        st.markdown(f"**{start_day:%m/%d}～{end_day:%m/%d}の推移**")

        # 全体の水深集合から「浅・中・深（10m起算）」の3本を先に決める
        reps = pick_shallow_mid_deep_min10_from_depths(depths_all)  # 既存関数を利用

        # reps の並びは [浅, 中, 深]（候補が2以下でも破綻しないように扱う）
        mapping = []
        if len(reps) >= 1:
            mapping.append(("表層", reps[0]))                     # 浅（10m以上の最小が優先）
        if len(reps) >= 2:
            mapping.append(("中層", reps[min(1, len(reps)-1)]))    # 中（要素2なら reps[1]）
        if len(reps) >= 3:
            mapping.append(("底層", reps[-1]))                     # 深（最深）

        any_line = False
        for lname, depth_sel in mapping:
            line = summarize_weekly_for_depth(lname, depth_sel, df_period)  # ← 新関数で1本だけ要約
            if line:
                any_line = True
                st.markdown(line)

        if not any_line:
            st.caption("（特筆すべき変化はありません）")


        table_html = build_weekly_table_html(df_period, day_list, depths_all, corr_on=corr_available)
        styles = get_calendar_css(65)
        full_html = f"<!doctype html><html><head><meta charset='utf-8'>{styles}</head><body>{table_html}</body></html>"
        st_html(full_html, height=650, scrolling=True)

    else:
        selected_day = st.date_input(
            "", value=max_day, min_value=min_day, max_value=max_day, key="day_sel", label_visibility="collapsed"  # ← ラベル非表示
        )
        df_day = df_pred[df_pred["date_day"] == selected_day].copy()

        if corr_available:
            df_corr_sel = df_corr[df_corr["date_day"] == selected_day].copy()
            df_day = add_corr(df_day, df_corr_sel)

        if not df_obs.empty and not df_day.empty:
            df_obs_sel = df_obs[df_obs["date_day"] == selected_day].copy()
            tol_obs = pd.Timedelta(minutes=OBS_MATCH_TOL_MIN)
            left = df_day.sort_values(["depth_m", "datetime"]).copy()
            right = df_obs_sel.sort_values(["depth_m", "datetime"])[["datetime", "depth_m", "obs_temp"]].copy()
            merged = safe_merge_asof_by_depth_keep_left(left, right, tolerance=tol_obs, right_value_cols=["obs_temp"], suffixes=("", ""))
            if "obs_temp" in merged.columns:
                df_day = merged

        depths_all = sorted([int(d) for d in df_pred["depth_m"].dropna().unique()])
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

        table_html = build_daily_table_html(df_day, depths_all, corr_on=corr_available)
        styles = get_calendar_css(65)
        full_html = f"<!doctype html><html><head><meta charset='utf-8'>{styles}</head><body>{table_html}</body></html>"
        st_html(full_html, height=650, scrolling=True)

# =========================================
# 水温グラフ（凡例簡略化：補正・実測のみ表示）
# =========================================
elif view_mode == "水温グラフ":
    df_pred = load_pred(selected_file)
    df_corr = load_corr_for(selected_file)
    df_obs = load_obs_for(selected_file)
    corr_available = not df_corr.empty

    if df_pred.empty:
        st.warning("予測データが読み込めませんでした")
        st.stop()

    try:
        period_mode = st.segmented_control(
            "", options=["直近1か月", "任意期間"], default="直近1か月", key="graph_period_mode"  # ← ラベル非表示
        )
    except Exception:
        period_mode = st.radio(
            "", ["直近1か月", "任意期間"], index=0, horizontal=True, key="graph_period_mode_radio", label_visibility="collapsed"  # ← ラベル非表示
        )

    latest_dt = df_pred["datetime"].max()
    available_days = sorted(df_pred["date_day"].unique()) if "date_day" in df_pred.columns else []
    if available_days:
        min_day = min(available_days); max_day = max(available_days)
    else:
        min_day = latest_dt.date(); max_day = latest_dt.date()

    if period_mode == "直近1か月":
        end_day = latest_dt.date()
        start_day = (latest_dt - pd.Timedelta(days=29)).date()
        title_suffix = "（直近1か月・時間別）"
    else:
        start_default = max(min_day, max_day - pd.Timedelta(days=29))
        start_day, end_day = st.slider(
            "", min_value=min_day, max_value=max_day, value=(start_default, max_day), key="graph_period_slider", label_visibility="collapsed"  # ← ラベル非表示
        )
        title_suffix = f"（{start_day:%Y-%m-%d}〜{end_day:%Y-%m-%d}・時間別）"

    # 1H 整形（予測）
    df_period = df_pred[(df_pred["date_day"] >= start_day) & (df_pred["date_day"] <= end_day)].copy()
    df_period = df_period.sort_values("datetime")
    if "pred_temp" in df_period.columns and not df_period.empty:
        df_period = (
            df_period.groupby(["depth_m", "datetime"], as_index=False).agg({"pred_temp": "median"})
        )
    if not df_period.empty:
        df_period = (
            df_period.sort_values("datetime")
            .groupby("depth_m", group_keys=False)
            .apply(lambda g: (
                g.drop(columns=["depth_m"]).set_index("datetime")
                .resample("1H").median(numeric_only=True)
                .interpolate(method="time", limit=2).reset_index()
                .assign(depth_m=int(g["depth_m"].iloc[0]))
            ))
        )
    if "depth_m" in df_period.columns:
        df_period["depth_m"] = pd.to_numeric(df_period["depth_m"], errors="coerce").round(0).astype("Int64")

    # OBS（近傍点）— 左を保持

    # --- OBS の近傍マージ（期間内に 1 点でもあれば obs_available=True） ---
    merged_for_points = pd.DataFrame(columns=["datetime", "depth_m", "obs_temp"])
    if not df_obs.empty and not df_period.empty:
        df_obs_period = df_obs[(df_obs["date_day"] >= start_day) & (df_obs["date_day"] <= end_day)].copy()
        if not df_obs_period.empty:
            tol = pd.Timedelta(minutes=CORR_MATCH_TOL_MIN)
            left  = df_period.sort_values(["depth_m","datetime"]).copy()
            right = df_obs_period.sort_values(["depth_m","datetime"])[["datetime","depth_m","obs_temp"]].copy()
            merged_for_points = safe_merge_asof_by_depth_keep_left(
                left=left, right=right, tolerance=tol, right_value_cols=["obs_temp"], suffixes=("","")
            )

    # 期間内に実測が 1 点でもあるか
    obs_available = (("obs_temp" in merged_for_points.columns)
                     and (not merged_for_points.dropna(subset=["obs_temp"]).empty))

    # corr を1Hに整形（帯があれば一緒に）
    df_corr_period = pd.DataFrame()
    if corr_available:
        df_corr_period = df_corr[(df_corr["date_day"] >= start_day) & (df_corr["date_day"] <= end_day)].copy()
        if not df_corr_period.empty:
            use_cols = ["corr_temp"]
            if "corr_low" in df_corr_period.columns: use_cols.append("corr_low")
            if "corr_high" in df_corr_period.columns: use_cols.append("corr_high")
            df_corr_period = (
                df_corr_period.sort_values("datetime")
                .groupby("depth_m", group_keys=False)
                .apply(lambda g: (
                    g.drop(columns=["depth_m"])
                    .set_index("datetime")[use_cols]
                    .resample("1H").median().dropna(how="all").reset_index()
                    .assign(depth_m=int(g["depth_m"].iloc[0]))
                ))
            )

    # グラフ（凡例簡略化）
    fig = go.Figure()
    base_colors = px.colors.qualitative.Dark24

    # pred 期間内に描画可能な全水深（整数）

    # --- 3 層の候補集合 ---
    depths_pred_all = sorted(set(df_period["depth_m"].dropna().astype(int).tolist())) if not df_period.empty else []

    depths_with_corr = set()
    if not df_corr_period.empty and "depth_m" in df_corr_period.columns:
        depths_with_corr = set(pd.to_numeric(df_corr_period["depth_m"], errors="coerce").dropna().astype(int).unique())

    depths_with_obs = set()
    if ("depth_m" in merged_for_points.columns) and ("obs_temp" in merged_for_points.columns):
        tmp_obs = merged_for_points.dropna(subset=["obs_temp"])
        if not tmp_obs.empty:
            depths_with_obs = set(pd.to_numeric(tmp_obs["depth_m"], errors="coerce").dropna().astype(int).unique())

    both_corr_obs = sorted(depths_with_corr.intersection(depths_with_obs))

    def pick_shallow_mid_deep_min10(cands: list[int], k: int = 3) -> list[int]:
        if not cands:
            return []
        xs = sorted(set(int(d) for d in cands))
        n = len(xs)
        if n <= 2:
            return xs[:k]
        # 浅＝10m以上の最小（無ければ最浅）
        low_idx = 0
        for i, d in enumerate(xs):
            if d >= 10:
                low_idx = i
                break
        high_idx = n - 1
        mid_idx = (low_idx + high_idx) // 2  # 偶数は下側
        idxs = [low_idx, mid_idx, high_idx]
        chosen = [xs[i] for i in sorted(set(idxs))]
        if len(chosen) < k:
            center = xs[mid_idx]
            rest = [d for d in xs if d not in chosen]
            rest_sorted = sorted(rest, key=lambda d: (abs(d - center), d))
            chosen.extend(rest_sorted[:k - len(chosen)])
        return chosen[:k]

    # --- 優先順：① corr+obs ≥3 → ② corr ≥3 → ③ pred 全体 ---
    if len(both_corr_obs) >= 3:
        default_depths = pick_shallow_mid_deep_min10(both_corr_obs, k=3)
    elif len(depths_with_corr) >= 3:
        default_depths = pick_shallow_mid_deep_min10(sorted(depths_with_corr), k=3)
    else:
        default_depths = pick_shallow_mid_deep_min10(depths_pred_all, k=3)

    if not default_depths:
        default_depths = depths_pred_all[: min(3, len(depths_pred_all))]



    # ユーザー選択 UI（ラベル非表示のまま）
    selected_depths = st.multiselect(
        "", depths_pred_all, default=default_depths, key="graph_depths", label_visibility="collapsed"
    )


    def emphasize_color(hex_color: str) -> str:
        try:
            rr = int(hex_color[1:3], 16); gg = int(hex_color[3:5], 16); bb = int(hex_color[5:7], 16)
            rr = min(255, rr + 25); gg = min(255, gg + 25); bb = min(255, bb + 25)
            return f"#{rr:02x}{gg:02x}{bb:02x}"
        except Exception:
            return hex_color

    for i, d in enumerate(selected_depths):
        base_col = base_colors[i % len(base_colors)]
        corr_col = emphasize_color(base_col)
        lg = f"depth{int(d)}"

        g_pred = df_period[df_period["depth_m"] == d]
        g_corr = df_corr_period[df_corr_period["depth_m"] == d] if not df_corr_period.empty else pd.DataFrame()
        g_obs = merged_for_points[merged_for_points["depth_m"] == d] if ("depth_m" in merged_for_points.columns) else pd.DataFrame()

        if not g_corr.empty:
            # 信頼帯 → 凡例非表示（fill は残す）
            if ("corr_low" in g_corr.columns) and ("corr_high" in g_corr.columns):
                fig.add_trace(go.Scatter(
                    x=g_corr["datetime"], y=g_corr["corr_low"].clip(lower=TEMP_MIN, upper=TEMP_MAX),
                    mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", name=f"{d}m 帯(下)"
                ))
                fig.add_trace(go.Scatter(
                    x=g_corr["datetime"], y=g_corr["corr_high"].clip(lower=TEMP_MIN, upper=TEMP_MAX),
                    mode="lines", line=dict(width=0),
                    fill='tonexty', fillcolor=to_rgba(corr_col, 0.18),
                    name=f"{d}m 信頼帯", legendgroup=lg, showlegend=False, hoverinfo="skip"
                ))
            # 補正 → 凡例表示
            y_corr = g_corr["corr_temp"].clip(lower=TEMP_MIN, upper=TEMP_MAX)
            fig.add_trace(go.Scatter(
                x=g_corr["datetime"], y=y_corr, mode="lines",
                name=f"{d}m 補正", legendgroup=lg, showlegend=True,
                line=dict(color=corr_col, width=3.0), opacity=1.0,
                hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>補正水温: %{y:.2f} °C<extra></extra>"
            ))
            # 予測（薄線）→ 凡例非表示
            if not g_pred.empty:
                y_pred = g_pred["pred_temp"].astype(float).clip(lower=TEMP_MIN, upper=TEMP_MAX)
                fig.add_trace(go.Scatter(
                    x=g_pred["datetime"], y=y_pred, mode="lines",
                    name=f"{d}m 予測", legendgroup=lg, showlegend=False,
                    line=dict(color=base_col, width=1.2, dash="dot"), opacity=0.35,
                    hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>予測水温: %{y:.2f} °C<extra></extra>"
                ))
            # 実測 → 凡例表示
            if not g_obs.empty:
                fig.add_trace(go.Scatter(
                    x=g_obs["datetime"], y=g_obs["obs_temp"], mode="markers",
                    name=f"{d}m 実測", legendgroup=lg, showlegend=True,
                    marker=dict(size=6, color=emphasize_color(base_col), line=dict(color="black", width=0.1)),
                    opacity=0.80,
                    hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>実測水温: %{y:.2f} °C<extra></extra>"
                ))
        else:
            # 補正が無い場合：予測（凡例表示）+ 実測（凡例表示）
            if not g_pred.empty:
                x = g_pred["datetime"]; y_pred = g_pred["pred_temp"].astype(float)
                fig.add_trace(go.Scatter(
                    x=x, y=y_pred, mode="lines",
                    name=f"{d}m 予測", legendgroup=lg, showlegend=True,
                    line=dict(color=base_col, width=2.0), opacity=1.0,
                    hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>水温: %{y:.2f} °C"
                ))
            if not g_obs.empty:
                fig.add_trace(go.Scatter(
                    x=g_obs["datetime"], y=g_obs["obs_temp"], mode="markers",
                    name=f"{d}m 実測", legendgroup=lg, showlegend=True,
                    marker=dict(size=4, color=emphasize_color(base_col), line=dict(color="black", width=0.1)),
                    opacity=0.40,
                    hovertemplate="%{x}<br>水深: " + f"{d}m" + "<br>実測水温: %{y:.2f} °C<extra></extra>"
                ))

    fig.update_layout(
        title={"text": f"{selected_file} 水温{title_suffix}", "y": 0.98, "x": 0.01, "xanchor": "left", "font": {"size": 16}},
        margin=dict(l=10, r=10, t=50, b=10),
        height=550, template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=1.02, xanchor="right", x=1, font=dict(size=12))
    )
    x_range = [pd.Timestamp(start_day), pd.Timestamp(end_day) + pd.Timedelta(days=1)]
    fig.update_xaxes(type="date", range=x_range, title_text="日時（JST）", tickfont=dict(size=11))
    fig.update_yaxes(title_text="水温 (℃)", tickfont=dict(size=11))
    st.plotly_chart(fig, use_container_width=True)