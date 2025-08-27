# Automatic Number Plate Recognition (ANPR) with YOLOv8, SORT, and EasyOCR

Detect vehicles, find their license plates, read plate text, and render an annotated video with per‑vehicle plate crops and IDs. The pipeline uses **YOLOv8** (Ultralytics) for detection, **SORT** for tracking, and **EasyOCR** for text recognition. Outputs are written to CSV and then visualized into a new MP4.

> **Demo**: The visualization overlays a green bracket around the tracked vehicle, a red box on the plate, and a white banner showing the best OCR result for that vehicle. (See `out.mp4` after running.)

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
│   └── license_plate_detector.pt   # custom YOLOv8n plate detector (Roboflow‑trained)
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

**Optional**: If your OpenCV build lacks H.264 writers, change the codec or install FFmpeg.

You can create a virtual environment and install:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install ultralytics torch torchvision torchaudio opencv-python easyocr numpy pandas scipy
```

> **Apple Silicon (M‑series)**: Using `device="mps"` is enabled by default in `main.py`. If `MPS available: False` prints, set `device="cpu"` (or install a PyTorch build with MPS).

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

## Configuration

Key spots to adjust:

* **Paths**: video (`sample.mp4`), plate model (`models/license_plate_detector.pt`).
* **Vehicle classes** (COCO IDs) in `main.py` → `vehicles = [2, 3, 5, 7]`.
* **OCR threshold** in `main.py` (`cv2.threshold(..., 64, 255, THRESH_BINARY_INV)`), tune per lighting.
* **Plate format** in `util.license_complies_format` and **mappings** in `dict_char_to_int` / `dict_int_to_char` for your country.
* **Codec** in `visualize.py` (`mp4v`); change to `XVID`, `MJPG`, or `avc1` if needed.

---

## Troubleshooting

* **`FileNotFoundError: test.csv`** when running `add_missing_data.py` → run `main.py` first, or check working directory.
* **No preview window / black video** → ensure `cap.read()` returns `True`, and that `sample.mp4` path & codec are valid. In headless envs, skip `imshow` and open `out.mp4`.
* **`MPS available: False`** on Apple Silicon → install a PyTorch build with MPS; otherwise set `device="cpu"`.
* **OpenCV writer fails** (e.g., H.264 not found) → switch FourCC (`'XVID'`, `'MJPG'`) and use `.avi`, or install FFmpeg.
* **OCR quality is poor** → experiment with thresholds, try non‑inverted binary, denoise/blur, or upgrade the plate detector quality.

---

## Acknowledgements

* **Ultralytics YOLOv8** for detection.
* **SORT** (Simple Online and Realtime Tracking) for multi‑object tracking.
* **EasyOCR** for optical character recognition.
* **Roboflow** for dataset tooling.

---

## License

This repository is for educational/research use. Datasets and model weights may have their own licenses—please review and comply with those terms before redistribution or commercial use.

---

## Roadmap (Nice‑to‑have)

* Replace EasyOCR with a more robust OCR (e.g., Transformer‑based) and add language toggles.
* Train a higher‑capacity plate detector (e.g., `YOLOv8s/m`) and support multi‑region formats.
* Confidence‑aware smoothing of OCR (temporal voting instead of argmax).
* Export per‑track JSON and plug‑in metrics (per‑plate recall/precision).
