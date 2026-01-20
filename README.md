# Colour Corrector for underwater footage/images

This repository contains two Python entrypoints:

- **Backend (CLI)**: `app_backend_segment_speedflags.py`
  - Corrects a single **image** or **video**.
  - Supports optional **segment-only** processing (`--start-sec`, `--duration-sec`).
  - Emits progress lines (`PROGRESS ANALYZE …` / `PROGRESS PROCESS …`) that the GUI can parse.

- **GUI (PySide6)**: `app_corrector_gui_segment_speedflags.py`
  - Desktop UI for selecting input/output, tuning parameters, previewing, and running the backend.

## Quick start

### 1) Create a virtual environment

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate
```

### 2) Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3) Install FFmpeg (required)

The backend uses `ffmpeg` for trimming, encoding and audio muxing. Install FFmpeg and ensure it is available on your `PATH`.

Verification:

```bash
ffmpeg -version
```

## Usage

### Backend (CLI)

**Video:**

```bash
python app_backend_segment_speedflags.py video input.mp4 output.mp4 \
  --start-sec 10 --duration-sec 30 \
  --downsample 1 \
  --fast-hs --fast-hs-map-scale 0.10
```

**Image:**

```bash
python app_backend_segment_speedflags.py image input.jpg output.jpg
```

Run `-h` for all parameters:

```bash
python app_backend_segment_speedflags.py -h
python app_backend_segment_speedflags.py video -h
```

### GUI

```bash
python app_corrector_gui_segment_speedflags.py

##To build app yourself, bundle ffmpeg and script

pyinstaller --noconfirm --clean --onedir --windowed --name ColourCorrector   --add-data "app_backend_segment_speedflags.py:."   --add-binary "ffmpeg:."   --hidden-import numpy   --hidden-import numpy.core._multiarray_umath   --hidden-import cv2   app_corrector_gui_segment_speedflags.py

```

The GUI defaults to using the backend script in the same folder. If you move files around, update the "Script path" field in the GUI.

## Notes

- For the best experience, use Python 3.10+.
- If OpenCV installation is problematic on your system, consider using a platform-specific wheel or a conda environment.

## Repository layout

```
.
├─ app_backend_segment_speedflags.py
├─ app_corrector_gui_segment_speedflags.py
├─ requirements.txt
└─ .gitignore
```

## App preview
![Preview](https://github.com/user-attachments/assets/ba5204cb-417b-4bb8-8afd-e5bd88af3a2e)

## Inspiration
This repo was inspired by https://github.com/nikolajbech/underwater-image-color-correction and https://github.com/bornfree/dive-color-corrector
