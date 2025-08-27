# Automatic Number Plate Recognition (ANPR) with YOLOv8, SORT, and EasyOCR

Detect vehicles, find their license plates, read plate text, and render an annotated video with per‑vehicle plate crops and IDs. The pipeline uses **YOLOv8** (Ultralytics) for detection, **SORT** for tracking, and **EasyOCR** for text recognition. Outputs are written to CSV and then visualized into a new MP4.

## Demo

<p align="center">
  <img src="assets/demo.png" alt="Demo Output" width="600"/>
</p>


---

## Features

* ✅ Vehicle detection using COCO‑pretrained **YOLOv8n**
* ✅ License‑plate detection using a custom **YOLOv8n** model (`models/license_plate_detector.pt`)
* ✅ Multi‑object tracking with **SORT** (stable IDs per car across frames)
* ✅ OCR with **EasyOCR** (format filtering + character post‑mapping for typical 7‑char plates: `AA00AAA`)
* ✅ CSV export per frame (`test.csv`) and **gap interpolation** for smoother tracks (`test_interpolated.csv`)
* ✅ High‑quality video overlay and export to `out.mp4`
* ✅ Apple Silicon support via **PyTorch MPS** (Metal) with `device="mps"`

---

## Project Structure

```
.
├── main.py                     # run detection + tracking + OCR → writes test.csv
├── util.py                     # helpers: OCR, plate formatting, CSV writer, car↔plate association
├── add_missing_data.py         # interpolates missing frames → writes test_interpolated.csv
├── visualize.py                # renders annotated video out.mp4
├── models/
│   └── license_plate_detector.pt   # custom YOLOv8n plate detector (Roboflow data)
├── sort/                       # SORT tracker (Simple Online and Realtime Tracking)
│   └── sort.py
├── sample.mp4                  # input video
└── README.md                   # this file
```

---

## Requirements

* Python 3.10+ (tested on 3.11)
* macOS / Linux / Windows
* For Apple Silicon acceleration: PyTorch with **MPS** support (macOS 12.3+)

### Python Packages

* ultralytics
* torch
* opencv-python
* easyocr
* numpy, pandas, scipy

---

## Models & Data

* **Vehicle detector**: `YOLOv8n` pretrained on **COCO** (IDs used: `car=2`, `motorcycle=3`, `bus=5`, `truck=7`).
* **Plate detector**: custom `YOLOv8n` trained on a Roboflow dataset. Place the weight file at `models/license_plate_detector.pt`.
* **Video**: put your input at `./sample.mp4` (or change the path in `main.py`).

> To retrain the plate detector, train in Ultralytics and export a `.pt` file, then replace `models/license_plate_detector.pt`.

---

## How It Works

1. **Detection** (`main.py`)

   * Detect vehicles with COCO YOLOv8n.
   * Track vehicles with **SORT** to maintain `car_id` across frames.
   * Detect license plates.
   * **Assign** each plate to the enclosing tracked vehicle.
   * **OCR**: grayscale → binary threshold → EasyOCR; apply format check (`AA00AAA`) and character mapping (e.g., `O→0`, `1→I`).  
    > Note: Surya OCR could provide higher accuracy, but it is heavier. For now I prioritized runtime speed, so EasyOCR’s slightly lower accuracy is acceptable.

   * Write rows to `test.csv` with per‑frame results.

2. **Interpolation** (`add_missing_data.py`)

   * Fills gaps in trajectories by linear interpolation of bounding boxes, creating `test_interpolated.csv` for smoother playback.

3. **Visualization** (`visualize.py`)

   * Chooses the **best OCR** (max confidence) per `car_id` and uses that label across the full track.
   * Draws a **green bracket** around cars and a **red rectangle** on plates; shows a **plate crop** and the recognized text in a white banner.
   * Writes `out.mp4` and displays a rescaled preview window.

---

## Usage

From the project root:

```bash
# 1) Run detection + OCR → creates test.csv
python main.py

# 2) Densify tracks → creates test_interpolated.csv
python add_missing_data.py

# 3) Render annotated video → creates out.mp4
python visualize.py
```

Press `q` to close the preview while visualizing.

---

## CSV Schema

`test.csv` (written by `util.write_csv`) has:

```
frame_nmr,car_id,car_bbox,license_plate_bbox,license_plate_bbox_score,license_number,license_number_score
```

* `car_bbox` and `license_plate_bbox` are `[x1 y1 x2 y2]` in image pixels.
* `license_number` is **only** written when format check passes; otherwise a gap is left to be filled later.

`test_interpolated.csv` has the same columns; missing frames are filled with linearly interpolated boxes. For interpolated rows, scores/text default to `0`.


---

## Acknowledgements

* **Ultralytics YOLOv8** for detection.
* **SORT** (Simple Online and Realtime Tracking) for multi‑object tracking.
* **EasyOCR** for optical character recognition.
* **Roboflow** for dataset tooling.
---

## Roadmap (Nice‑to‑have)

* Replace EasyOCR with a more robust OCR (e.g., surya) and add language toggles.
* Train a higher‑capacity plate detector (e.g., `YOLOv8s/m`) and support multi‑region formats.
