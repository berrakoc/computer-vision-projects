
<img width="1193" height="1056" alt="tumor" src="https://github.com/user-attachments/assets/a2c6e083-c5a8-4519-bc44-75627826bea8" />

# Brain MRI Tumor Detection (Streamlit + Detectron2)

A simple web app that lets you upload a brain MRI image and runs an object-detection model to draw bounding boxes around suspected tumor regions. The UI is built with Streamlit; detections are visualized with Plotly and can be toggled on/off.

## 👀 What this app does
- Upload a `.png`, `.jpg`, or `.jpeg` MRI image
- Run a Detectron2 **RetinaNet (R-101-FPN)** model (custom weights) on **CPU**
- Draw bounding boxes for detections above a confidence threshold (default **0.5**)
- Toggle between **Original** and **Detections** views
- Optional custom background image for the app

## 🧰 Tech Stack
- **Python** (3.10 recommended)
- **Streamlit** for the web UI
- **Detectron2** (RetinaNet R101 FPN config)
- **PyTorch** (CPU)
- **Pillow**, **NumPy**
- **Plotly** for interactive visualization

## 🧪 How It Works (Pipeline)
1. User uploads an image → converted to RGB (Pillow)
2. Converted to a contiguous **uint8** NumPy array (Detectron2 expects HWC)
3. Detectron2 predictor returns instances (classes, scores, boxes)
4. Boxes with **score > threshold** are kept
5. Plotly draws rectangles over the original image with a toggle
