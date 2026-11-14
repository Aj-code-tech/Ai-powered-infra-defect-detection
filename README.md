# 🛣️ InfraGuard AI: Infrastructure Defect Detection & Maintenance Prioritization


live test here - https://ai-powered-infra-defect-detection-cwemh2mqqybjz7ktjnlsr6.streamlit.app/


An end-to-end Computer Vision and Civil Engineering decision-support system that detects Indian pavement distresses using **YOLOv8s** and ranks road segments for repair using an **IRC-aligned Severity & Prioritization Engine**.

### Key Features
- **Multi-Class Distress Detection:** Trained on 16,777 images from the **RDD2022 India** dataset across 4 standard classes: `D00` (Longitudinal Crack), `D10` (Transverse Crack), `D20` (Alligator Crack), and `D40` (Pothole).
- **Spatial ROI Filtering:** Filters out upper-frame sky, tree, and overhead wire false positives using a configurable Region-of-Interest cutoff.
- **Automated Maintenance Prioritization:** Computes a `0–10` Severity Score combining defect hazard weights, bounding box area ratio, and defect density to classify segments into **HIGH**, **MEDIUM**, **LOW**, or **CLEAN** tiers with CSV schedule export.

### Tech Stack
- **Model & Vision:** PyTorch, Ultralytics YOLOv8s, OpenCV
- **Dashboard & Backend:** Python, Streamlit, Pandas, NumPy
