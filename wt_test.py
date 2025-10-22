import os
import glob
import re
import pandas as pd
import numpy as np
import streamlit as st

# =========================================
# データ読み込み関数
# =========================================
def load_uv_ts(folder):
    files_uv = glob.glob(os.path.join(folder, "u_v_*.csv"))
    files_ts = glob.glob(os.path.join(folder, "t_s_*.csv"))

    def extract_depth(fname):
        m = re.search(r"Lv(\d+(?:\.\d+)?)", os.path.basename(fname))
        return float(m.group(1)) if m else np.nan

    uv_list, ts_list = [], []
    for f in files_uv:
        depth = extract_depth(f)
        try:
            df = pd.read_csv(f)
        except:
            continue
        df = df.rename(columns=lambda x: x.strip())
        if "Date" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Depth"] = depth
        uv_list.append(df[["Date","Depth","u","v"]])

    for f in files_ts:
        depth = extract_depth(f)
        try:
            df = pd.read_csv(f)
        except:
            continue
        df = df.rename(columns=lambda x: x.strip())
        if "Date" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Depth"] = depth
        ts_list.append(df[["Date","Depth","t","s"]])

    uv_all = pd.concat(uv_list, ignore_index=True) if uv_list else pd.DataFrame()
    ts_all = pd.concat(ts_list, ignore_index=True) if ts_list else pd.DataFrame()

    if uv_all.empty and ts_all.empty:
        return pd.DataFrame()

    if uv_all.empty:
        merged = ts_all.copy()
        merged["u"] = np.nan
        merged["v"] = np.nan
    elif ts_all.empty:
        merged = uv_all.copy()
        merged["t"] = np.nan
        merged["s"] = np.nan
    else:
        merged = pd.merge(uv_all, ts_all, on=["Date","Depth"], how="outer")

    merged["Date_JST"] = merged["Date"] + pd.Timedelta(hours=9)
    merged = merged.sort_values(["Date_JST","Depth"]).reset_index(drop=True)
    merged["Speed"] = np.sqrt(merged["u"].fillna(0)**2 + merged["v"].fillna(0)**2)
    merged["Direction_deg"] = (np.degrees(np.arctan2(merged["u"].fillna(0), merged["v"].fillna(0))) + 360) % 360
    return merged

# =========================================
# SVG矢印生成関数
# =========================================
arrow_half_height = 6

def get_arrow_svg(direction, speed):
    if np.isnan(speed) or np.isnan(direction):
        return ""
    css_angle = (direction - 90) % 360
    size, color = get_arrow_style(speed)
    line_end = size * 0.55
    return f"""
    <svg width="{size}" height="{size}" style="transform: rotate({css_angle}deg);">
        <line x1="4" y1="{size/2}" x2="{line_end}" y2="{size/2}" stroke="{color}" stroke-width="2"/>
        <polygon points="{line_end},{size/2 - arrow_half_height} {size},{size/2} {line_end},{size/2 + arrow_half_height}" fill="{color}"/>
    </svg>
    """

def get_arrow_style(speed):
    if speed < 1.0:
        return 18, "blue"
    elif speed < 1.3:
        return 22, "green"
    else:
        return 26, "red"

def get_color(temp, t_min=5, t_max=25):
    if np.isnan(temp):
        return "rgba(220,220,220,0.6)"
    ratio = (temp - t_min) / (t_max - t_min)
    ratio = max(0, min(1, ratio))
    r = int(200 * ratio + 55)
    g = int(180 * (1 - abs(ratio - 0.5) * 2))
    b = int(200 * (1 - ratio) + 55)
    return f"rgba({r},{g},{b},0.6)"

# =========================================
# CSS（スマホ対応）
# =========================================
st.markdown("""
<style>
.block-container {padding-top: 0.5rem !important; padding-bottom: 0.5rem !important;}
table, th, td {font-family: 'Noto Sans JP', 'Roboto', sans-serif;}
table {border-collapse: collapse; width: 100%; font-size: 10px;}
th, td {text-align: center; padding: 2px; line-height: 1.0;}
th {background-color: #f0f0f0; position: sticky; top: 0; z-index: 3; font-size: 12px; font-weight: bold;}
th.sticky-col, td.sticky-col {position: sticky; left: 0; background-color: #fff; z-index: 4; width: 80px; min-width: 80px; max-width: 80px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 12px; font-weight: bold;}
.current-data {font-size: 12px; font-weight: bold;}
.prev-data {font-size: 10px; color: #555;}
@media (max-width: 768px) {
    table {font-size: 8px;}
    th, td {padding: 1px;}
}
/* ▼ Streamlitフォーム要素の余白削除＋ラベル非表示＋高さ縮小 */
div[data-baseweb="radio"], div[data-baseweb="select"], div[data-baseweb="input"] {
    margin: 0 !important;
    padding: 0 !important;
    height: auto !important;
}
div[data-baseweb="radio"] label, div[data-baseweb="select"] label, div[data-baseweb="input"] label {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# UI
# =========================================
parent_folder = "data"
if not os.path.exists(parent_folder):
    st.error(f"フォルダが見つかりません: {parent_folder}")
else:
    subfolders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
    selected_subfolder = st.selectbox("フォルダ", options=subfolders)

    mode_view = st.radio("", ["週間（12時）", "1時間ピッチ"])
    show_prev_year = st.checkbox("前年データ")

    folder_now = os.path.join(parent_folder, selected_subfolder, "2025_now")
    merged_now = load_uv_ts(folder_now)

    if merged_now.empty:
        st.warning("データがありません")
    else:
        merged_now["Date_day"] = merged_now["Date_JST"].dt.date
        depths = sorted(merged_now["Depth"].unique())
        dates = sorted(merged_now["Date_day"].unique())
        selected_date = st.date_input("", value=dates[0], min_value=dates[0], max_value=dates[-1])

        df_prev = pd.DataFrame()
        if show_prev_year:
            folder_prev = os.path.join(parent_folder, selected_subfolder, "2024")
            merged_prev = load_uv_ts(folder_prev)
            if not merged_prev.empty:
                df_prev = merged_prev

        table_html = "<div style='max-height:80vh; overflow-x:auto; overflow-y:auto;'><table>"
        if mode_view == "週間（12時）":
            start_date = selected_date
            end_date = start_date + pd.Timedelta(days=6)
            df_period = merged_now[(merged_now["Date_day"] >= start_date) & (merged_now["Date_day"] <= end_date)]
            times = [f"{d.strftime('%m/%d')} 12:00" for d in pd.date_range(start_date, end_date)]
        else:
            df_period = merged_now[merged_now["Date_day"] == selected_date]
            times = [t.strftime('%H:%M') for t in sorted(df_period["Date_JST"].dt.floor("h").unique())]

        table_html += "<tr><th class='sticky-col'>水深</th>"
        for t in times:
            table_html += f"<th>{t}</th>"
        table_html += "</tr>"

        for depth in depths:
            table_html += f"<tr><td class='sticky-col'>{depth}m</td>"
            for idx, t in enumerate(times):
                if mode_view == "週間（12時）":
                    day = selected_date + pd.Timedelta(days=idx)
                    row = df_period[(df_period["Date_day"] == day) & (df_period["Date_JST"].dt.hour == 12) & (df_period["Depth"] == depth)]
                    prev_row = df_prev[
                        (df_prev["Date_JST"].dt.month == day.month) &
                        (df_prev["Date_JST"].dt.day == day.day) &
                        (df_prev["Date_JST"].dt.hour == 12) &
                        (df_prev["Depth"] == depth)
                    ] if not df_prev.empty else pd.DataFrame()
                else:
                    time_obj = sorted(df_period["Date_JST"].dt.floor("h").unique())[idx]
                    row = df_period[(df_period["Date_JST"].dt.floor("h") == time_obj) & (df_period["Depth"] == depth)]
                    prev_row = df_prev[
                        (df_prev["Date_JST"].dt.month == time_obj.month) &
                        (df_prev["Date_JST"].dt.day == time_obj.day) &
                        (df_prev["Date_JST"].dt.hour == time_obj.hour) &
                        (df_prev["Depth"] == depth)
                    ] if not df_prev.empty else pd.DataFrame()

                if not row.empty:
                    temp = row["t"].values[0]
                    speed = row["Speed"].values[0]
                    speed_kt = f"{speed * 1.94384:.1f} kt" if not np.isnan(speed) else "-"
                    direction = row["Direction_deg"].values[0]
                    arrow_svg = get_arrow_svg(direction, speed)
                    color = get_color(temp)
                    cell_content = f"<div class='current-data'>{temp:.1f}°C<br>{speed_kt}<br>{arrow_svg}</div>"
                    if not prev_row.empty:
                        prev_temp = prev_row.iloc[0]["t"]
                        cell_content += f"<div class='prev-data'>({prev_temp:.1f}°C)</div>"
                    cell = f"<td style='background-color:{color};'>{cell_content}</td>"
                else:
                    cell = "<td>-</td>"
                table_html += cell
            table_html += "</tr>"

        table_html += "</table></div>"
        st.markdown(table_html, unsafe_allow_html=True)