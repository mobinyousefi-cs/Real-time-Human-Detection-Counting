# Real-time Human Detection & Counting

Real-time human detection and counting system using Python, OpenCV, and the HOG + SVM people detector.

The application can:

* Use your **webcam** or a **video file** as input
* Detect people in each frame using **HOG + SVM** (OpenCV built-in people detector)
* Draw bounding boxes and show the **number of detected humans per frame**
* Optionally save the annotated output video

> Author: [Mobin Yousefi](https://github.com/mobinyousefi-cs)

---

## Features

* Real-time detection and counting through webcam
* Support for:

  * Webcam stream (default)
  * Video file (e.g., `.mp4`, `.avi`)
  * Single image (for quick testing)
* HOG + SVM-based detection (no need to train from scratch)
* Configurable detection parameters (hit threshold, stride, padding, scale)
* Simple CLI interface using `argparse`
* Modular, testable code under `src/`

---

## Project Structure

```text
real-time-human-detection-counting/
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ .gitignore
├─ .editorconfig
├─ src/
│  └─ human_detection_counter/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ detector.py
│     ├─ visualization.py
│     ├─ video_io.py
│     └─ cli.py
├─ scripts/
│  └─ run_webcam.py
└─ tests/
   └─ test_imports.py
```

---

## Installation

It is recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
# .venv\Scripts\activate   # Windows
```

Install the project in editable mode:

```bash
pip install -e .
```

Or install using `pyproject.toml` with `pip`:

```bash
pip install .
```

---

## Dependencies

Main dependencies:

* `opencv-python`
* `imutils`
* `numpy`

They are declared in `pyproject.toml`.

---

## Usage

### 1. Webcam mode (default)

```bash
human-detection-counter
```

This opens the default webcam, runs human detection, shows a window with bounding boxes and the current count.

### 2. Webcam with options

```bash
human-detection-counter \
  --source webcam \
  --resize-width 640 \
  --display \
  --hit-threshold 0.0 \
  --scale 1.05
```

### 3. Video file as input

```bash
human-detection-counter \
  --source path/to/video.mp4 \
  --output path/to/output_annotated.mp4 \
  --resize-width 800
```

### 4. Single image

For single images, the CLI will process and show the result once:

```bash
human-detection-counter \
  --source path/to/image.jpg \
  --image
```

> Note: The `--image` flag forces single-frame processing and prevents video loop logic.

---

## Command-line Options

```text
--source          Input source: "webcam", integer webcam index (e.g., "0"), video file path, or image path.
--image           Treat the source as a single image instead of a video stream.
--resize-width    Optional width to resize frames before detection (maintains aspect ratio).
--output          Optional path to save annotated video. Ignored in image mode.
--no-display      Disable GUI display (useful in headless environments).
--hit-threshold   HOG SVM hit threshold (float). Higher -> fewer detections, lower -> more.
--win-stride      HOG sliding window stride, two ints, e.g. "8 8".
--padding         HOG padding, two ints, e.g. "8 8".
--scale           Image pyramid scale factor, e.g. 1.05.
--max-frames      Maximum number of frames to process (useful for testing).
```

Example with custom stride and padding:

```bash
human-detection-counter \
  --source webcam \
  --win-stride 8 8 \
  --padding 8 8 \
  --scale 1.03
```

---

## HOG + SVM People Detector

This project uses OpenCV’s built-in HOG descriptor with a pre-trained SVM for person (pedestrian) detection:

* `cv2.HOGDescriptor()` with `setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())`
* Detection is performed via `detectMultiScale`

This avoids the need for training, making it ideal for a compact, demonstration-ready project.

---

## Development & Testing

Run the basic tests (using `pytest`):

```bash
pip install pytest
pytest
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
