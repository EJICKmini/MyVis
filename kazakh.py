import time
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
from picamera2 import Picamera2
from ultralytics import YOLO  # YOLOv8
from gtts import gTTS
import os

# ======================
# Settings
# ======================
WEIGHTS = "/home/sst/projects/CIS/yolov8n.pt"
IMG_SIZE = 640
CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.35
TTS_FILENAME = "tts_temp.mp3"

# ======================
# Initialize YOLOv8 model
# ======================
device = 'cuda'  # If Pi has CUDA (Jetson), otherwise 'cpu'
model = YOLO(WEIGHTS)

# ======================
# COCO class names in English
# ======================
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "gloves", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ======================
# Draw text using PIL
# ======================
def draw_text(frame, text, position, font_path="arial.ttf", font_size=18, color=(0, 255, 0)):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# ======================
# TTS function using gTTS
# ======================
def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(TTS_FILENAME)
        os.system(f"mpg123 {TTS_FILENAME} >/dev/null 2>&1")  # plays audio silently
        os.remove(TTS_FILENAME)
    except Exception as e:
        print(f"TTS error: {e}")

# ======================
# Detect objects
# ======================
def detect_objects(frame):
    results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)[0]
    detected_objects = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        detected_objects.append(class_name)

        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        frame = draw_text(frame, label, (int(x1), int(y1) - 25), font_size=20)

    return frame, detected_objects

# ======================
# Main loop
# ======================
def main():
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(preview_config)
    picam2.start()

    last_detected = ""
    last_time = time.time()

    try:
        while True:
            frame = picam2.capture_array()
            frame, detected_objects = detect_objects(frame)

            if detected_objects:
                current_objects = ", ".join(set(detected_objects))
                if current_objects != last_detected or time.time() - last_time > 5:
                    print(f"Detected: {current_objects}")
                    speak(f"I see {current_objects}")  # TTS
                    last_detected = current_objects
                    last_time = time.time()

            cv2.imshow("MyVision", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import time
import numpy as np
import cv2
from PIL import ImageFont, ImageDraw, Image
from picamera2 import Picamera2
from ultralytics import YOLO  # YOLOv8
from gtts import gTTS
import os

# ======================
# Settings
# ======================
WEIGHTS = "/home/sst/projects/CIS/yolov8n.pt"
IMG_SIZE = 640
CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.35
TTS_FILENAME = "tts_temp.mp3"

# ======================
# Initialize YOLOv8 model
# ======================
device = 'cuda'  # If Pi has CUDA (Jetson), otherwise 'cpu'
model = YOLO(WEIGHTS)

# ======================
# COCO class names in English
# ======================
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "gloves", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ======================
# Draw text using PIL
# ======================
def draw_text(frame, text, position, font_path="arial.ttf", font_size=18, color=(0, 255, 0)):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# ======================
# TTS function using gTTS
# ======================
def speak(text):
    try:
        tts = gTTS(text=text, lang='en')
        tts.save(TTS_FILENAME)
        os.system(f"mpg123 {TTS_FILENAME} >/dev/null 2>&1")  # plays audio silently
        os.remove(TTS_FILENAME)
    except Exception as e:
        print(f"TTS error: {e}")

# ======================
# Detect objects
# ======================
def detect_objects(frame):
    results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)[0]
    detected_objects = []

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
        detected_objects.append(class_name)

        label = f"{class_name} {conf:.2f}"
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        frame = draw_text(frame, label, (int(x1), int(y1) - 25), font_size=20)

    return frame, detected_objects

# ======================
# Main loop
# ======================
def main():
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
    picam2.configure(preview_config)
    picam2.start()

    last_detected = ""
    last_time = time.time()

    try:
        while True:
            frame = picam2.capture_array()
            frame, detected_objects = detect_objects(frame)

            if detected_objects:
                current_objects = ", ".join(set(detected_objects))
                if current_objects != last_detected or time.time() - last_time > 5:
                    print(f"Detected: {current_objects}")
                    speak(f"I see {current_objects}")  # TTS
                    last_detected = current_objects
                    last_time = time.time()

            cv2.imshow("MyVision", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
