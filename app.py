"""
Advanced ML Object Tracking Dashboard
======================================
A Streamlit analytics UI for reviewing tracked video assets and their
per-frame tracking CSV outputs.

Run with:
    streamlit run app.py
"""

import os
import glob
import traceback
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Object Tracking Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = "data"        # folder the app scans for CSVs, e.g. data/*.csv
VIDEO_DIR = "videos"     # folder the app scans for matching .mp4 files
REQUIRED_COLUMNS = ["Frame", "Tracking_ID", "Class"]

# Recognized vehicle classes the KPI row cares about. Anything else in the
# 'Class' column still shows up in the charts, it's just not pinned to a card.
KPI_CLASSES = {
    "Total Cars": "car",
    "Total Trucks": "truck",
    "Total Motorcycles": "motorcycle",
    "Total Buses": "bus",
}

CUSTOM_CSS = """
<style>
    .status-dot {
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .status-connected { background-color: #2ecc71; }
    .status-waiting { background-color: #e74c3c; }
    .kpi-card {
        background-color: #12161f;
        border: 1px solid #232838;
        border-radius: 10px;
        padding: 18px 16px;
        text-align: center;
    }
    .kpi-value {
        font-size: 30px;
        font-weight: 700;
        color: #ffffff;
        margin: 4px 0 0 0;
    }
    .kpi-label {
        font-size: 12px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: #8a93a6;
        margin: 0;
    }
    .panel-title {
        font-size: 15px;
        font-weight: 600;
        color: #e6e8ef;
        margin-bottom: 6px;
    }
    .log-line {
        font-size: 13px;
        color: #c4c9d4;
        line-height: 1.55;
        margin: 0 0 4px 0;
    }
    .log-tag {
        font-weight: 600;
        margin-right: 6px;
    }
    .tag-ok { color: #2ecc71; }
    .tag-warn { color: #f39c12; }
    .tag-info { color: #3498db; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# CORE DATA ENGINE — THE CONNECTOR PIPELINE
# --------------------------------------------------------------------------
# This section is intentionally defensive. Every possible failure point in
# "pick a file -> read it -> validate it -> hand it to the UI" is caught and
# reported instead of crashing the app.

def discover_local_csv_files(folder: str) -> list:
    """Return a sorted list of CSV filenames found in `folder`.

    Never raises: a missing folder just yields an empty list.
    """
    if not os.path.isdir(folder):
        return []
    try:
        paths = glob.glob(os.path.join(folder, "*.csv"))
        return sorted(os.path.basename(p) for p in paths)
    except OSError:
        return []


def discover_local_videos(folder: str) -> dict:
    """Map a dataset's base name -> matching .mp4 path, if one exists."""
    if not os.path.isdir(folder):
        return {}
    try:
        mapping = {}
        for path in glob.glob(os.path.join(folder, "*.mp4")):
            base = os.path.splitext(os.path.basename(path))[0]
            mapping[base] = path
        return mapping
    except OSError:
        return {}


def validate_schema(df: pd.DataFrame) -> list:
    """Check the dataframe has the columns the rest of the app depends on.

    Returns a list of human-readable problems. Empty list = valid.
    """
    problems = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"Missing required column(s): {', '.join(missing)}.")
        return problems  # no point checking further, columns aren't there

    if df.empty:
        problems.append("The file parsed correctly but contains zero rows.")

    if "Frame" in df.columns and not pd.api.types.is_numeric_dtype(df["Frame"]):
        problems.append("Column 'Frame' contains non-numeric values.")

    if "Tracking_ID" in df.columns and df["Tracking_ID"].isna().all():
        problems.append("Column 'Tracking_ID' is entirely empty.")

    return problems


@st.cache_data(show_spinner=False)
def load_tracking_csv(file_source, source_label: str):
    """
    THE CRITICAL CONNECTOR: takes either an uploaded file object or a local
    file path, safely parses it with pd.read_csv, validates the schema,
    and returns (dataframe_or_None, list_of_error_or_warning_strings).

    This function never lets an exception escape — every failure mode
    (missing file, corrupt bytes, wrong encoding, empty file, bad schema)
    is converted into a clean message the UI can render.
    """
    errors = []

    # Step 1: does the source even exist / is it readable?
    if file_source is None:
        return None, [f"No file was provided for '{source_label}'."]

    if isinstance(file_source, str):
        if not os.path.isfile(file_source):
            return None, [f"File not found on disk: {file_source}"]
        if os.path.getsize(file_source) == 0:
            return None, [f"File '{source_label}' is empty (0 bytes)."]

    # Step 2: attempt the parse, trying a couple of encodings before giving up.
    df = None
    parse_errors = []
    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(file_source, encoding=encoding)
            break
        except UnicodeDecodeError as e:
            parse_errors.append(f"Encoding '{encoding}' failed: {e}")
            # if file_source is an uploaded file object, reset the pointer
            if hasattr(file_source, "seek"):
                file_source.seek(0)
            continue
        except pd.errors.EmptyDataError:
            return None, [f"'{source_label}' has no columns to parse (empty CSV)."]
        except pd.errors.ParserError as e:
            return None, [f"'{source_label}' could not be parsed as CSV: {e}"]
        except Exception as e:  # noqa: BLE001 - deliberately broad, this is the safety net
            return None, [
                f"Unexpected error while reading '{source_label}': {e}",
                traceback.format_exc(limit=1),
            ]

    if df is None:
        return None, parse_errors or [f"Could not decode '{source_label}' with any supported encoding."]

    # Step 3: validate the resulting dataframe against the expected schema.
    schema_problems = validate_schema(df)
    if schema_problems:
        return None, [f"Schema validation failed for '{source_label}':"] + schema_problems

    # Step 4: light cleanup so downstream charts/KPIs don't choke on dirty data.
    df["Class"] = df["Class"].astype(str).str.strip().str.lower()
    df["Frame"] = pd.to_numeric(df["Frame"], errors="coerce")
    df = df.dropna(subset=["Frame"])
    df["Frame"] = df["Frame"].astype(int)

    return df, errors


# --------------------------------------------------------------------------
# ANALYTICS HELPERS
# --------------------------------------------------------------------------

def compute_kpis(df: pd.DataFrame) -> dict:
    """Unique tracked object count per vehicle class, plus a grand total."""
    counts = {}
    for label, class_name in KPI_CLASSES.items():
        subset = df[df["Class"] == class_name]
        counts[label] = subset["Tracking_ID"].nunique()

    counts["Total Units"] = df["Tracking_ID"].nunique()
    return counts


def class_distribution_chart(df: pd.DataFrame):
    dist = (
        df.groupby("Class")["Tracking_ID"]
        .nunique()
        .reset_index(name="Unique Objects")
        .sort_values("Unique Objects", ascending=False)
    )
    fig = px.bar(
        dist,
        x="Class",
        y="Unique Objects",
        color="Class",
        title="Vehicle distribution by class",
        template="plotly_dark",
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
    )
    return fig


def objects_over_time_chart(df: pd.DataFrame):
    timeline = (
        df.groupby("Frame")["Tracking_ID"]
        .nunique()
        .reset_index(name="Active Objects")
        .sort_values("Frame")
    )
    fig = px.line(
        timeline,
        x="Frame",
        y="Active Objects",
        title="Tracked objects per frame",
        template="plotly_dark",
    )
    fig.update_traces(line_color="#3498db")
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
    )
    return fig


# --------------------------------------------------------------------------
# AI PROJECT INTELLIGENCE PANEL (mock assistant)
# --------------------------------------------------------------------------

def generate_traffic_report(df: pd.DataFrame) -> list:
    """
    Simulated project assistant. Reads the current dataset state and returns
    a list of (tag, message) tuples for the chat-style log panel.

    This is intentionally a rules-based mock, not a live model call — it
    demonstrates the panel contract so the ML team can later swap this out
    for a real inference/report-generation call without touching the UI.
    """
    log = []
    now = datetime.now().strftime("%H:%M:%S")

    total_ids = df["Tracking_ID"].nunique()
    total_frames = df["Frame"].nunique()
    classes_seen = df["Class"].nunique()
    frame_span = df["Frame"].max() - df["Frame"].min() if not df.empty else 0

    log.append(("ok", f"[{now}] Pipeline check complete — dataset parsed and validated."))
    log.append(("info", f"Ingested {total_frames:,} frames spanning {frame_span:,} frame indices."))
    log.append(("info", f"Detected {total_ids:,} unique tracked objects across {classes_seen} class types."))

    # naive duplicate/gap detection to simulate a "diagnostics" pass
    dup_rows = df.duplicated(subset=["Frame", "Tracking_ID"]).sum()
    if dup_rows > 0:
        log.append(("warn", f"Found {dup_rows} duplicate (Frame, Tracking_ID) rows — possible re-detection noise."))
    else:
        log.append(("ok", "No duplicate tracking rows detected. Association quality looks stable."))

    # per-class density note
    top_class = df["Class"].value_counts().idxmax()
    log.append(("info", f"Most frequent class in this stream: '{top_class}'."))

    # audit-style recommendation
    if total_ids > 0 and total_frames > 0:
        density = total_ids / total_frames
        if density > 1.5:
            log.append(("warn", "High object density per frame — consider reviewing occlusion handling."))
        else:
            log.append(("ok", "Object density per frame is within a normal operating range."))

    log.append(("info", "Recommendation: cross-check KPI totals against the raw video for a spot audit."))
    return log


def render_chat_panel(df: pd.DataFrame | None):
    st.markdown('<p class="panel-title">🤖 AI project intelligence</p>', unsafe_allow_html=True)

    with st.container(height=300, border=True):
        if df is None:
            with st.chat_message("assistant"):
                st.write("Awaiting a validated dataset. Load a CSV from the sidebar to generate a report.")
        else:
            with st.chat_message("assistant"):
                st.write("Dataset loaded. Here is the automated audit:")
                for tag, message in generate_traffic_report(df):
                    css_class = {"ok": "tag-ok", "warn": "tag-warn", "info": "tag-info"}[tag]
                    label = {"ok": "OK", "warn": "WARN", "info": "INFO"}[tag]
                    st.markdown(
                        f'<p class="log-line"><span class="log-tag {css_class}">[{label}]</span>{message}</p>',
                        unsafe_allow_html=True,
                    )

    prompt = st.chat_input("Ask the assistant about this dataset...", disabled=(df is None))
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            st.write(
                "This is a mock assistant wired up for the demo UI. "
                "Swap `generate_traffic_report` for a real model call to answer free-form questions like this one."
            )


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------

def render_sidebar():
    st.sidebar.title("🛰️ Tracking console")
    st.sidebar.markdown("Navigation")
    st.sidebar.radio(
        "View",
        ["Dashboard", "Dataset explorer", "About"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()

    st.sidebar.subheader("Data source")
    source_mode = st.sidebar.radio(
        "Source mode",
        ["Upload file", "Select from data/ folder"],
        label_visibility="collapsed",
    )

    file_source = None
    source_label = None

    if source_mode == "Upload file":
        uploaded = st.sidebar.file_uploader("Upload tracking CSV", type=["csv"])
        if uploaded is not None:
            file_source = uploaded
            source_label = uploaded.name
    else:
        local_files = discover_local_csv_files(DATA_DIR)
        if not local_files:
            st.sidebar.info(f"No CSV files found in '{DATA_DIR}/'. Add files there or switch to upload mode.")
        else:
            selected = st.sidebar.selectbox("Choose a dataset", local_files)
            file_source = os.path.join(DATA_DIR, selected)
            source_label = selected

    return file_source, source_label


def render_status_indicator(is_connected: bool):
    dot_class = "status-connected" if is_connected else "status-waiting"
    label = "Connected" if is_connected else "Awaiting data"
    st.sidebar.markdown(
        f'<div style="margin-top:8px;"><span class="status-dot {dot_class}"></span>{label}</div>',
        unsafe_allow_html=True,
    )


def render_class_filter(df: pd.DataFrame | None):
    if df is None:
        return None
    classes = sorted(df["Class"].dropna().unique().tolist())
    st.sidebar.divider()
    st.sidebar.subheader("Metrics filter")
    selected = st.sidebar.multiselect("Filter by class", classes, default=classes)
    return selected


# --------------------------------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------------------------------

def render_video_player(source_label: str | None):
    st.markdown('<p class="panel-title">🎬 Tracking stream playback</p>', unsafe_allow_html=True)
    if source_label is None:
        st.info("No dataset selected yet — the matching video preview will appear here once one is loaded.")
        return

    base_name = os.path.splitext(source_label)[0]
    video_map = discover_local_videos(VIDEO_DIR)

    if base_name in video_map:
        st.video(video_map[base_name])
    else:
        st.warning(
            f"No matching video found for '{source_label}' in '{VIDEO_DIR}/'. "
            "Add a file named "
            f"'{base_name}.mp4' to that folder, or upload a video below."
        )
        manual_video = st.file_uploader("Or upload the tracking video manually", type=["mp4"], key="manual_video")
        if manual_video is not None:
            st.video(manual_video)


def render_kpi_row(df: pd.DataFrame):
    kpis = compute_kpis(df)
    order = ["Total Units", "Total Cars", "Total Trucks", "Total Motorcycles", "Total Buses"]
    cols = st.columns(len(order))
    for col, label in zip(cols, order):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <p class="kpi-label">{label}</p>
                    <p class="kpi-value">{kpis[label]:,}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_empty_state():
    st.markdown("### Awaiting data")
    st.write(
        "Use the sidebar to upload a tracking CSV or select one from the `data/` folder. "
        "Expected columns: `Frame`, `Tracking_ID`, `Class`."
    )


def render_error_state(errors: list):
    st.error("The selected file could not be loaded.")
    for e in errors:
        st.write(f"- {e}")


def main():
    file_source, source_label = render_sidebar()

    df = None
    errors = []
    if file_source is not None:
        df, errors = load_tracking_csv(file_source, source_label)

    render_status_indicator(is_connected=df is not None)
    selected_classes = render_class_filter(df)

    st.title("Object tracking intelligence dashboard")
    st.caption("Live review of ML-generated tracking streams and per-frame analytics.")

    render_video_player(source_label)
    st.divider()

    if df is None:
        if errors:
            render_error_state(errors)
        else:
            render_empty_state()
        return

    filtered_df = df[df["Class"].isin(selected_classes)] if selected_classes else df

    render_kpi_row(filtered_df)
    st.divider()

    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(class_distribution_chart(filtered_df), use_container_width=True)
        st.plotly_chart(objects_over_time_chart(filtered_df), use_container_width=True)
    with right:
        render_chat_panel(filtered_df)


if __name__ == "__main__":
    main()
