"""
indoor_object_detection_tts.py

- Detect indoor objects using YOLOv10
- Estimate distance using focal length calibration
- Speak the object name, distance, and direction
- Works with images or live webcam
"""

import os
import cv2
import pyttsx3
from ultralytics import YOLO
from typing import List, Tuple, Optional


# Resolve model path relative to this file, and fix case to match repository
_HERE = os.path.dirname(__file__)
DEFAULT_MODEL_PATH = os.path.join(_HERE, "yolov10x.pt")   # replace with yolov10s/10m/10l if needed
KNOWN_WIDTHS = {
    "person": 0.5,
    "cup": 0.08,
    "chair": 0.45,
    "laptop": 0.30,
    "bottle": 0.07,
}
FOCAL_LENGTH = 707.94


def tts(names: List[str], engine=None):
    if not names:
        return
    seen = set()
    unique = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    sentence = "I see " + (" and ".join(unique))
    try:
        if engine is None:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            voices = engine.getProperty('voices')
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)
        engine.say(sentence)
        engine.runAndWait()
    except Exception as e:
        print("TTS error:", e)


class Detector:
    """Reusable detector decoupled from UI/TTS for mobile integration."""

    def __init__(self, model_path: Optional[str] = None, score_threshold: float = 0.5):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.score_threshold = score_threshold
        self.model = YOLO(self.model_path)

    def infer(self, image_bgr):
        return self.model(image_bgr)

    def draw_boxes(self, image_bgr, results) -> Tuple[List[str], List[Tuple[str, float, str]]]:
        return draw_boxes(image_bgr, results, self.score_threshold)

    def extract_names(self, image_bgr, results) -> List[str]:
        return extract_names(image_bgr, results, self.score_threshold)

def calibrate_camera(known_distance: float, known_width: float, image_path: str) -> float:
    """
    Calibrate focal length using a reference object of known size and distance.
    """
    
    # Real world width of your calibration object (meters)
    KNOWN_WIDTH = 0.0856  # credit card width
    # Distance from camera to object during calibration (meters)
    KNOWN_DISTANCE = 0.3  
    
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' when done selecting ROI...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("Calibration - place object at known distance", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Select ROI (draw bounding box around object)
            roi = cv2.selectROI("Calibration", frame, False, False)
            x, y, w, h = roi
            pixel_width = w
            focal_length = (pixel_width * KNOWN_DISTANCE) / KNOWN_WIDTH
            print(f"Estimated focal length: {focal_length:.2f} pixels")
            break

    cap.release()
    cv2.destroyAllWindows()


def estimate_distance(pixel_width: float, known_width: float, focal_length: float) -> float:
    """Estimate distance from camera in cm."""
    if focal_length is None:
        return -1
    return (known_width * focal_length) / pixel_width


# ---------------------- Direction ----------------------
def get_direction(frame_width: int, x_center: int) -> str:  
    """
    Determine the horizontal direction of the detected object.
    Splits the frame into 5 zones: Far Left, Slightly Left, Center, Slightly Right, Far Right.
    """   
    relative_pos = float(x_center / frame_width)
    
    if relative_pos < 0.2:
        return "far left"
    elif relative_pos < 0.4:
        return "slightly left"
    elif relative_pos < 0.6:
        return "center"
    elif relative_pos < 0.8:
        return "slightly right"
    else:
        return "far right"


# ---------------------- Drawing + Extraction ----------------------
def draw_boxes(image_bgr, results, score_threshold=0.5) -> Tuple[List[str], List[Tuple[str, float, str]]]:
    """Draw boxes, return detected names + (name, distance, direction)."""
    out = image_bgr.copy()
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < score_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = r.names[cls_id]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{cls}: {conf:.2f}"
            ((text_w, text_h), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(out, (x1, y1 - text_h - 6), (x1 + text_w, y1), (0, 255, 0), -1)
            cv2.putText(out, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def extract_names(image_bgr, results, score_threshold=0.5):
    names = []
    out = image_bgr.copy()
    
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = r.names[cls_id]
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            x_center = (x1 + x2) // 2

            if conf < score_threshold:
                continue
            cls_id = int(box.cls[0])
            distance = 0
            if label in KNOWN_WIDTHS:
                distance = estimate_distance(w , KNOWN_WIDTHS[label] , FOCAL_LENGTH)

            direction = get_direction(out.shape[1] , x_center)
            if distance > 0:
                names.append(f"{r.names[cls_id]} to {direction}, about {distance:.1f} meters away")
            else:
                names.append(f"{r.names[cls_id]} to {direction}")
    return names

# ---------------------- Modes ----------------------
def run_image_mode(input_path: str, score: float = 0.5, output_path: str = None, model_path: Optional[str] = None):
    detector = Detector(model_path=model_path, score_threshold=score)
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: could not read {input_path}")
        return
    results = detector.infer(img)
    drawn = detector.draw_boxes(img, results)
    if output_path:
        cv2.imwrite(output_path, drawn)
        print(f"Wrote annotated image to {output_path}")
    else:
        names = detector.extract_names(img, results)
        tts(names)
        cv2.imshow('detected', drawn)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_webcam_mode(score: float = 0.5, frame_resize: float = 0.6, frame_skip: int = 3, model_path: Optional[str] = None):
    detector = Detector(model_path=model_path, score_threshold=score)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return
    # Initialize TTS engine once and reuse
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    voices = engine.getProperty('voices')
    if len(voices) > 1:
        engine.setProperty('voice', voices[1].id)
    last_spoken = []
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                cv2.imshow('webcam', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            if frame_resize != 1.0:
                small = cv2.resize(frame, (0, 0), fx=frame_resize, fy=frame_resize)
            else:
                small = frame
            results = detector.infer(small)
            out_small = detector.draw_boxes(small, results)
            names = detector.extract_names(small, results)

            if names and names != last_spoken:
                tts(names, engine)
                last_spoken = names
            cv2.imshow('webcam', out_small)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print('Interrupted by user')
    finally:
        cap.release()
        cv2.destroyAllWindows()


# ---------------------- Main ----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Indoor Object Detection with YOLOv10")
    parser.add_argument("--mode", choices=["image", "webcam"], default="image",
                    help="Run detection on a single image or webcam stream")
    parser.add_argument("--path", type=str,
                    help="Path to input image if mode=image (required for image mode)")
    parser.add_argument("--model", type=str, default=None, help="Path to model file (optional)")
    parser.add_argument("--calibrate", action="store_true",
                    help="Run calibration before detection (for distance estimation)")
    args = parser.parse_args()

    if args.calibrate and args.path:
        FOCAL_LENGTH = calibrate_camera(known_distance=50, known_width=KNOWN_WIDTH, image_path=args.path)

    if args.mode == "image" and args.path:
        run_image_mode(args.path, model_path=args.model)
    else:
        run_webcam_mode(model_path=args.model)
