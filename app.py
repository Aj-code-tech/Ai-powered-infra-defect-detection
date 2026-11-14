import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. Page Configuration & Custom Dashboard Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="InfraGuard AI | Road Defect & Maintenance Prioritization",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1e222b;
        border: 1px solid #303642;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .priority-high {
        background-color: rgba(239, 68, 68, 0.15);
        border-left: 5px solid #ef4444;
        padding: 14px;
        border-radius: 6px;
        color: #fca5a5;
    }
    .priority-medium {
        background-color: rgba(245, 158, 11, 0.15);
        border-left: 5px solid #f59e0b;
        padding: 14px;
        border-radius: 6px;
        color: #fcd34d;
    }
    .priority-low {
        background-color: rgba(59, 130, 246, 0.15);
        border-left: 5px solid #3b82f6;
        padding: 14px;
        border-radius: 6px;
        color: #93c5fd;
    }
    .priority-clean {
        background-color: rgba(16, 185, 129, 0.15);
        border-left: 5px solid #10b981;
        padding: 14px;
        border-radius: 6px;
        color: #6ee7b7;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Civil Engineering Metadata & Cached Model Loader
# ---------------------------------------------------------
DEFECT_META = {
    0: {"code": "D00", "name": "Longitudinal Crack", "weight": 1.0, "color": (255, 255, 0)},   # Cyan (BGR)
    1: {"code": "D10", "name": "Transverse Crack",   "weight": 1.5, "color": (0, 165, 255)},   # Orange (BGR)
    2: {"code": "D20", "name": "Alligator Crack",    "weight": 2.5, "color": (255, 0, 255)},   # Magenta (BGR)
    3: {"code": "D40", "name": "Pothole",            "weight": 3.5, "color": (50, 50, 255)},   # Red (BGR)
}

@st.cache_resource
def load_yolo_model(weights_path="best_road_defect.pt"):
    return YOLO(weights_path)

# ---------------------------------------------------------
# 3. Inference & Maintenance Prioritization Engine
# ---------------------------------------------------------
def analyze_image(model, pil_image, filename, conf_thresh, roi_cutoff, show_roi_line=True):
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h_img, w_img, _ = img_bgr.shape
    total_area = float(h_img * w_img)

    results = model.predict(source=img_bgr, conf=conf_thresh, verbose=False)[0]

    annotated = img_bgr.copy()
    roi_y_px = int(h_img * roi_cutoff)

    if show_roi_line and roi_cutoff > 0:
        cv2.line(annotated, (0, roi_y_px), (w_img, roi_y_px), (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(
            annotated, f"Road ROI Cutoff ({int(roi_cutoff*100)}%)",
            (12, max(roi_y_px - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2
        )

    detections = []
    weighted_severity_sum = 0.0
    total_defect_area_ratio = 0.0
    class_counts = {"D00": 0, "D10": 0, "D20": 0, "D40": 0}

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])

        # Filter out sky / overhead false positives above ROI cutoff
        y_center_norm = ((y1 + y2) / 2.0) / h_img
        if y_center_norm < roi_cutoff:
            continue

        meta = DEFECT_META[cls_id]
        box_area_ratio = ((x2 - x1) * (y2 - y1)) / total_area

        impact = meta["weight"] * (0.6 + 0.4 * conf) * (1.0 + 15.0 * box_area_ratio)
        weighted_severity_sum += impact
        total_defect_area_ratio += box_area_ratio
        class_counts[meta["code"]] += 1

        # Draw bounding box and label
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), meta["color"], 3)
        label = f"{meta['code']} {meta['name']} ({int(conf*100)}%)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(annotated, (ix1, max(iy1 - th - 10, 0)), (ix1 + tw + 8, iy1), meta["color"], -1)
        cv2.putText(
            annotated, label, (ix1 + 4, max(iy1 - 5, th)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2
        )

        detections.append({
            "Code": meta["code"],
            "Defect Type": meta["name"],
            "Confidence": f"{conf*100:.1f}%",
            "Area (%)": round(box_area_ratio * 100, 2)
        })

    defect_count = len(detections)

    if defect_count == 0:
        priority_score = 0.0
        priority_tier = "CLEAN"
        action = "No Pavement Distress Detected — Continue Routine Patrol"
    else:
        raw_score = (weighted_severity_sum * 1.4) + (total_defect_area_ratio * 25.0) + (defect_count * 0.45)
        priority_score = round(min(10.0, raw_score), 1)

        if priority_score >= 7.0:
            priority_tier = "HIGH"
            action = "URGENT: Emergency Pothole Patching / Full-Depth Repair (Within 48 Hours)"
        elif priority_score >= 3.5:
            priority_tier = "MEDIUM"
            action = "SCHEDULED: Crack Sealing / Bituminous Surface Overlay (Within 2 Weeks)"
        else:
            priority_tier = "LOW"
            action = "MONITOR: Preventive Maintenance on Next Quarterly Inspection Cycle"

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return {
        "filename": filename,
        "annotated_image": annotated_rgb,
        "detections": detections,
        "defect_count": defect_count,
        "class_counts": class_counts,
        "affected_area_pct": round(total_defect_area_ratio * 100, 2),
        "priority_score": priority_score,
        "priority_tier": priority_tier,
        "recommended_action": action
    }

# ---------------------------------------------------------
# 4. Sidebar Controls & File Upload
# ---------------------------------------------------------
with st.sidebar:
    st.title("🛣️ InfraGuard AI")
    st.caption("YOLOv8s Road Distress Detection & IRC Maintenance Prioritization")
    st.divider()

    uploaded_files = st.file_uploader(
        "Upload Road Inspection Images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload single or multiple road/dashcam images for automated defect ranking."
    )

    st.subheader("Detection & ROI Parameters")
    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.10, max_value=0.80, value=0.22, step=0.02,
        help="Lower values catch faint potholes; higher values reduce false alarms."
    )
    roi_cutoff = st.slider(
        "Sky / Overhead ROI Cutoff (%)",
        min_value=0, max_value=50, value=35, step=5,
        help="Ignores detections in the top percentage of the frame (sky, trees, power lines)."
    ) / 100.0
    show_roi_line = st.checkbox("Show ROI Boundary Line on Image", value=True)

    st.divider()
    st.markdown("**Distress Severity Weights:**")
    st.markdown("- 🔴 **D40 Pothole:** `3.5x` (Critical)\n- 🟣 **D20 Alligator Crack:** `2.5x` (High)\n- 🟠 **D10 Transverse Crack:** `1.5x` (Moderate)\n- 🔵 **D00 Longitudinal Crack:** `1.0x` (Base)")

# ---------------------------------------------------------
# 5. Main Workspace
# ---------------------------------------------------------
st.title("Infrastructure Defect Detection & Maintenance Prioritization")

if not os.path.exists("best_road_defect.pt"):
    st.error("Model weights `best_road_defect.pt` not found in the root directory. Please place it next to `app.py`.")
    st.stop()

model = load_yolo_model("best_road_defect.pt")

if not uploaded_files:
    st.info("👈 Upload one or more Indian road inspection images from the left sidebar to begin automated defect detection and priority ranking.")
    st.stop()

# Analyze all uploaded images
results_list = [
    analyze_image(model, Image.open(f), f.name, conf_thresh, roi_cutoff, show_roi_line)
    for f in uploaded_files
]

# Sort by Priority Score descending
results_list.sort(key=lambda x: x["priority_score"], reverse=True)

# Top KPI Summary Row
total_defects_all = sum(r["defect_count"] for r in results_list)
high_priority_count = sum(1 for r in results_list if r["priority_tier"] == "HIGH")
avg_priority = round(np.mean([r["priority_score"] for r in results_list]), 1)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Road Segments Inspected", len(results_list))
kpi2.metric("Total Defects Detected", total_defects_all)
kpi3.metric("High-Priority Segments", high_priority_count)
kpi4.metric("Average Severity Index", f"{avg_priority} / 10")

st.divider()

# Interactive Segment Selector & Main Inspection View
col_nav, col_img, col_stats = st.columns([1.1, 2.4, 1.5])

with col_nav:
    st.subheader("📋 Inspection Queue")
    options = [
        f"#{idx+1} [{r['priority_tier']}] {r['filename']} ({r['priority_score']}/10)"
        for idx, r in enumerate(results_list)
    ]
    selected_label = st.radio("Ranked by Urgency:", options, label_visibility="collapsed")
    selected_idx = options.index(selected_label)
    active = results_list[selected_idx]

with col_img:
    st.subheader(f"🔍 Active Inspection: {active['filename']}")
    st.image(active["annotated_image"], use_container_width=True)

with col_stats:
    st.subheader("📊 System Insights")

    tier = active["priority_tier"]
    css_class = f"priority-{tier.lower()}"
    st.markdown(
        f"""
        <div class="{css_class}">
            <strong>PRIORITY TIER: {tier} ({active['priority_score']} / 10)</strong><br/>
            <small>{active['recommended_action']}</small>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.metric("Defects in Segment", active["defect_count"], f"{active['affected_area_pct']}% Road Area")

    c1, c2 = st.columns(2)
    c1.metric("D40 Potholes", active["class_counts"]["D40"])
    c2.metric("D20 Alligator", active["class_counts"]["D20"])
    c3, c4 = st.columns(2)
    c3.metric("D10 Transverse", active["class_counts"]["D10"])
    c4.metric("D00 Longitudinal", active["class_counts"]["D00"])

    if active["detections"]:
        st.markdown("**Detected Bounding Boxes:**")
        st.dataframe(pd.DataFrame(active["detections"]), hide_index=True, use_container_width=True)

# ---------------------------------------------------------
# 6. Batch Maintenance Prioritization Schedule & CSV Export
# ---------------------------------------------------------
st.divider()
st.subheader("📑 Automated Maintenance Prioritization Schedule")

schedule_df = pd.DataFrame([
    {
        "Repair Rank": idx + 1,
        "Road Segment / Image": r["filename"],
        "Priority Score (0-10)": r["priority_score"],
        "Priority Tier": r["priority_tier"],
        "Total Defects": r["defect_count"],
        "Potholes (D40)": r["class_counts"]["D40"],
        "Alligator (D20)": r["class_counts"]["D20"],
        "Affected Area (%)": r["affected_area_pct"],
        "Recommended Action": r["recommended_action"]
    }
    for idx, r in enumerate(results_list)
])

st.dataframe(schedule_df, hide_index=True, use_container_width=True)

csv_bytes = schedule_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download Maintenance Priority Schedule (CSV)",
    data=csv_bytes,
    file_name="maintenance_priority_schedule.csv",
    mime="text/csv"
)