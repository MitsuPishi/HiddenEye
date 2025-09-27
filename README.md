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
