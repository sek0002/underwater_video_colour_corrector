# Contributing

## Development setup

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Install FFmpeg and confirm `ffmpeg -version` works.

## Running

- CLI backend:

  ```bash
  python app_backend_segment_speedflags.py video input.mp4 output.mp4
  ```

- GUI:

  ```bash
  python app_corrector_gui_segment_speedflags.py
  ```

## Code style

This project currently uses plain scripts. If you want to modernize it, a good next step is to move code into a `src/` package and add formatting/linting (ruff/black).
