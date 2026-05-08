import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import cv2
import numpy as np
import tempfile
import json
from pathlib import Path
from PIL import Image

st.set_page_config(
    page_title="EV Smart Intelligence System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Global */
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #1a1d2e; border-right: 1px solid #2d2f3e; }

/* Cards */
.ev-card {
    background: linear-gradient(135deg, #1a3a2a, #1f4d35);
    border: 1px solid #2ecc71;
    border-left: 4px solid #2ecc71;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
}
.nonev-card {
    background: linear-gradient(135deg, #3a1a1a, #4d1f1f);
    border: 1px solid #e74c3c;
    border-left: 4px solid #e74c3c;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
}
.warn-card {
    background: linear-gradient(135deg, #3a2d1a, #4d3d1f);
    border: 1px solid #f39c12;
    border-left: 4px solid #f39c12;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
}
.card-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 10px;
    letter-spacing: 0.5px;
}
.card-row {
    display: flex;
    gap: 20px;
    margin-top: 8px;
}
.card-item {
    flex: 1;
}
.card-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    opacity: 0.7;
    margin-bottom: 2px;
}
.card-value {
    font-size: 16px;
    font-weight: 600;
}

/* Stat boxes */
.stat-box {
    background: #1e2130;
    border-radius: 10px;
    padding: 15px;
    text-align: center;
    border: 1px solid #2d2f3e;
    margin-bottom: 10px;
}
.stat-number {
    font-size: 32px;
    font-weight: 700;
    line-height: 1;
}
.stat-label {
    font-size: 12px;
    opacity: 0.6;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Alert banner */
.alert-banner {
    background: #e74c3c;
    color: white;
    padding: 10px 16px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 13px;
    text-align: center;
    margin-bottom: 10px;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.8; }
    100% { opacity: 1; }
}

/* Log items */
.log-item {
    background: #1e2130;
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 6px;
    font-size: 12px;
    border-left: 3px solid #2d2f3e;
}
.log-ev { border-left-color: #2ecc71; }
.log-nonev { border-left-color: #e74c3c; }
.log-warn { border-left-color: #f39c12; }

/* Header */
.main-header {
    background: linear-gradient(135deg, #1a1d2e, #2d2f4e);
    border-radius: 12px;
    padding: 24px 30px;
    margin-bottom: 24px;
    border: 1px solid #3d3f5e;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    from core.pipeline import EVDetectionPipeline
    return EVDetectionPipeline()


# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size:28px;">⚡ EV Smart Intelligence System</h1>
    <p style="margin:6px 0 0; opacity:0.6; font-size:14px;">
        Proof of Concept — Real-time EV / Non-EV Detection at Charging Bays
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    input_mode = st.radio("Input Mode", ["Image", "Video"], horizontal=True)
    show_debug = st.checkbox("Show debug info", value=False)

    st.markdown("---")

    # Load stats from log
    log_dir   = Path("data/logs")
    log_files = sorted(log_dir.glob("*.jsonl"), reverse=True) if log_dir.exists() else []
    events    = []

    if log_files:
        with open(log_files[0]) as f:
            for line in f:
                try:
                    events.append(json.loads(line))
                except:
                    pass

    total    = len(events)
    ev_cnt   = sum(1 for e in events if "ev" in e.get("decision","") and "non" not in e.get("decision",""))
    non_ev   = sum(1 for e in events if e.get("decision") == "non_ev")
    uncertain = sum(1 for e in events if e.get("decision") == "uncertain")

    st.markdown("### 📊 Session Stats")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number" style="color:#2ecc71">{ev_cnt}</div>
            <div class="stat-label">EV Detected</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number" style="color:#e74c3c">{non_ev}</div>
            <div class="stat-label">Non-EV Alert</div>
        </div>""", unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number" style="color:#f39c12">{uncertain}</div>
            <div class="stat-label">Uncertain</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number" style="color:#3498db">{total}</div>
            <div class="stat-label">Total</div>
        </div>""", unsafe_allow_html=True)

    # Recent events
    if events:
        st.markdown("### 🕐 Recent Events")
        for e in events[-6:][::-1]:
            decision = e.get("decision", "")
            reg      = e.get("registration_number") or "N/A"
            time     = e.get("timestamp", "")[-8:-3]
            color    = e.get("plate_color", "")

            if "ev" in decision and "non" not in decision:
                css = "log-ev"
                icon = "✅"
            elif decision == "non_ev":
                css = "log-nonev"
                icon = "🚨"
            else:
                css = "log-warn"
                icon = "⚠️"

            st.markdown(f"""
            <div class="log-item {css}">
                {icon} <b>{reg}</b><br>
                <span style="opacity:0.6">{decision} · {time}</span>
            </div>""", unsafe_allow_html=True)

    if st.button("🗑️ Clear Log", use_container_width=True):
        for f in log_dir.glob("*.jsonl"):
            f.unlink()
        st.rerun()

# ── Main layout ───────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📤 Input")

    if input_mode == "Image":
        uploaded = st.file_uploader(
            "Upload vehicle image",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption="Input image", use_column_width=True)

            if st.button("🔍  Detect Vehicle", type="primary", use_container_width=True):
                with st.spinner("Analyzing vehicle..."):
                    pipeline = load_pipeline()
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        img.save(tmp.name)
                        annotated_frame, results = pipeline.process_image(tmp.name)
                    st.session_state["annotated"] = annotated_frame
                    st.session_state["results"]   = results
                    st.rerun()

    else:
        uploaded = st.file_uploader(
            "Upload vehicle video",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed"
        )
        if uploaded:
            st.video(uploaded)

            if st.button("🔍  Process Video", type="primary", use_container_width=True):
                pipeline = load_pipeline()
                pipeline.reset_tracks()

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                    tmp.write(uploaded.read())
                    video_path = tmp.name

                out_path    = video_path.replace(".mp4", "_out.mp4")
                progress    = st.progress(0, text="Processing frames...")
                all_results = []

                frame_list = list(pipeline.process_video(video_path, out_path))
                for i, (frame, res) in enumerate(frame_list):
                    all_results.extend(res)
                    progress.progress((i + 1) / max(len(frame_list), 1))

                st.session_state["results"] = all_results
                st.success(f"✅ Processed {len(frame_list)} frames")
                st.rerun()

with col_right:
    st.markdown("### 📊 Detection Results")

    if "annotated" in st.session_state:
        ann     = st.session_state["annotated"]
        ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
        st.image(ann_rgb, use_column_width=True)

    if "results" in st.session_state and st.session_state["results"]:
        results = st.session_state["results"]

        for i, result in enumerate(results):
            decision    = result.decision.value
            plate_color = result.plate_color.value.upper() if result.plate_color else "N/A"
            reg         = result.registration_number or "Not extracted"
            conf        = result.color_confidence

            # OCR confidence indicator
            if result.registration_number:
                ocr_icon = "🟢" if conf > 0.6 else "🟡" if conf > 0.3 else "🔴"
            else:
                ocr_icon = "⚫"

            if result.alert_level == "none":
                st.markdown(f"""
                <div class="ev-card">
                    <div class="card-title" style="color:#2ecc71">
                        ✅ EV DETECTED — Charging Allowed
                    </div>
                    <div class="card-row">
                        <div class="card-item">
                            <div class="card-label">Plate Color</div>
                            <div class="card-value">🟢 {plate_color}</div>
                        </div>
                        <div class="card-item">
                            <div class="card-label">Reg Number</div>
                            <div class="card-value">{ocr_icon} {reg}</div>
                        </div>
                        <div class="card-item">
                            <div class="card-label">Confidence</div>
                            <div class="card-value">{result.score}/3</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            elif result.alert_level == "critical":
                st.markdown(f"""
                <div class="alert-banner">🚨 ALERT — NON-EV VEHICLE DETECTED AT CHARGING BAY</div>
                <div class="nonev-card">
                    <div class="card-title" style="color:#e74c3c">
                        🚫 NON-EV VEHICLE — Bay Blocked
                    </div>
                    <div class="card-row">
                        <div class="card-item">
                            <div class="card-label">Plate Color</div>
                            <div class="card-value">⚪ {plate_color}</div>
                        </div>
                        <div class="card-item">
                            <div class="card-label">Reg Number</div>
                            <div class="card-value">{ocr_icon} {reg}</div>
                        </div>
                        <div class="card-item">
                            <div class="card-label">Action Required</div>
                            <div class="card-value">Remove Vehicle</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="warn-card">
                    <div class="card-title" style="color:#f39c12">
                        ⚠️ UNCLEAR — Manual Check Required
                    </div>
                    <div class="card-row">
                        <div class="card-item">
                            <div class="card-label">Plate Color</div>
                            <div class="card-value">❓ {plate_color}</div>
                        </div>
                        <div class="card-item">
                            <div class="card-label">Reg Number</div>
                            <div class="card-value">{ocr_icon} {reg}</div>
                        </div>
                        <div class="card-item">
                            <div class="card-label">Reason</div>
                            <div class="card-value">Low visibility</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if show_debug:
                with st.expander(f"Debug info — Detection #{i+1}"):
                    st.json({
                        "decision": decision,
                        "score": result.score,
                        "plate_color": plate_color,
                        "color_confidence": round(conf, 3),
                        "registration_number": reg,
                        "flags": result.flags,
                        "alert_level": result.alert_level
                    })

    elif "results" in st.session_state and not st.session_state["results"]:
        st.warning("No vehicles detected in this image. Try a clearer image with the full vehicle visible.")
    else:
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; opacity:0.4;">
            <div style="font-size:48px;">🚗</div>
            <div style="font-size:16px; margin-top:12px;">
                Upload an image or video to begin detection
            </div>
        </div>
        """, unsafe_allow_html=True)