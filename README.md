# Indoor Object Detection with YOLOv10 + Speech Output + Distance Estimation

## Overview

This project is a practical experiment in combining computer vision and speech. The idea is simple:

* Detect everyday **indoor objects** using a YOLOv10 model.
* Estimate how far those objects are from the camera with a lightweight distance calculation.
* Announce the results out loud using text-to-speech (TTS).

It works with single images or a live webcam feed. The long-term goal is to run the pipeline efficiently on a **mobile device** so it can be used in smart home or accessibility applications.

---

## Features

* Object detection powered by **YOLOv10**
* **Distance estimation** using a simple calibration method
* Multiple **TTS options** (offline and online) so the voice can sound natural, not robotic
* Works in Jupyter for demos, or as standalone Python scripts

---

## Installation

Clone the repository:

```bash
git clone https://github.com/MitsuPishi/HiddenEye
cd HiddenEye
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Detect objects in an image

```python
from src.Detection import run_image_mode
run_image_mode("path_to_image")
```

### Run webcam mode

```python
from src.Detection import run_webcam_mode
```

### Calibrate for distance

```python
from src.Detection import calibrate_camera
focal_length = calibrate_camera(
    known_distance=50, 
    known_width=14, 
    image_path="data/calibration.jpg"
)
```
---

### Command-Line Usage

The app supports a simple CLI for running detection:

**Run on an image:**

```bash
python indoor_object_detection.py --mode image --path data/test/room.jpg
```

**Run on an image with distance calibration:**

```bash
python indoor_object_detection.py --mode image --path data/calibration.jpg --calibrate
```

**Run live webcam detection:**

```bash
python indoor_object_detection.py --mode webcam
```

**Output:**

* Annotated image window (desktop) or live webcam feed
* Spoken detection results (object name, distance, and direction)

> Notes:
>
> * Calibration is optional but recommended for accurate distance estimation.
> * The webcam mode does not require a path.

---

### What each argument does:

1. **`--mode`**

   * Accepts `"image"` or `"webcam"`
   * Default is `"image"`
   * Determines whether the script runs on a **single image** or **live webcam feed**

2. **`--path`**

   * The **path to the input image** if you are in image mode
   * Ignored in webcam mode

3. **`--calibrate`**

   * Optional flag
   * If provided, runs the **camera calibration function** first (to compute focal length for distance estimation)

---

## Dataset and Classes

The model is not yet tuned for indoor use. Current labels include standard coco dataset labels.

The list can would be extended to include more furniture.

---

## Roadmap

- [ ] Fine-tune YOLOv10 on a custom indoor object dataset for higher accuracy in home/office settings
- [ ] Deploy on mobile devices (Android/iOS) using ONNX Runtime Mobile or TensorFlow Lite
- [ ] More natural TTS voices (Coqui-TTS, Edge-TTS, ElevenLabs) with configurable voice settings
- [ ] Smarter distance estimation (depth estimation via monocular depth models, stereo cameras, or LiDAR integration)
- [ ] Multi-language support for speech output
- [ ] Voice interaction (ask the system “what’s in front of me?” and get spoken answers)

---

## Contributing

If you’re interested in indoor-focused detection, accessibility tools, or speech interfaces, feel free to open issues or submit PRs.

---

## License

MIT License.
