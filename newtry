import cv2
import time
import numpy as np
import pyttsx3
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO

# Path to your YOLOv8 model
MODEL_PATH = "MyVis/yolo11n.pt"  # Update if needed

# Settings
IMG_SIZE = 640
CONF_THRESHOLD = 0.5

# Class names (Russian labels)
CLASS_NAMES = [
    "человек", "велосипед", "машина", "мотоцикл", "самолет", "автобус", "поезд",
    "грузовик", "лодка", "светофор", "пожарный гидрант", "знак стоп", "паркометр",
    "скамейка", "птица", "кошка", "собака", "лошадь", "овца", "корова", "слон",
    "медведь", "зебра", "жираф", "рюкзак", "зонт", "дамская сумка", "галстук",
    "чемодан", "фрисби", "лыжи", "сноуборд", "мяч", "воздушный змей", "бита",
    "перчатка", "скейтборд", "серфборд", "теннисная ракетка", "бутылка",
    "бокал", "чашка", "вилка", "нож", "ложка", "миска", "банан", "яблоко",
    "сэндвич", "апельсин", "брокколи", "морковь", "хот-дог", "пицца", "пончик",
    "торт", "стул", "диван", "растение", "кровать", "обеденный стол", "туалет",
    "телевизор", "ноутбук", "мышь", "пульт", "клавиатура", "телефон", "микроволновка",
    "духовка", "тостер", "раковина", "холодильник", "книга", "часы", "ваза",
    "ножницы", "медвежонок", "фен", "зубная щетка"
]

# Initialize text-to-speech engine (Russian voice)
engine = pyttsx3.init()
for voice in engine.getProperty('voices'):
    if "ru" in voice.languages or "Russian" in voice.name:
        engine.setProperty('voice', voice.id)
        break

def speak(text):
    engine.say(text)
    engine.runAndWait()

def draw_russian_text(frame, text, position, font_path="arial.ttf", font_size=18, color=(0, 255, 0)):
    # Convert OpenCV image (BGR) to PIL image
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

def detect_objects(frame):
    results = model(frame)[0]

    detected_objects = []

    for box in results.boxes:
        conf = box.conf.item()
        if conf < CONF_THRESHOLD:
            continue
        class_id = int(box.cls.item())
        class_name = CLASS_NAMES[class_id]
        detected_objects.append(class_name)

        xyxy = box.xyxy[0].cpu().numpy().astype(int)  # [xmin, ymin, xmax, ymax]
        label = f"{class_name} {conf:.2f}"

        # Draw bounding box
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)

        # Draw label with Russian text
        frame = draw_russian_text(frame, label, (xyxy[0], max(xyxy[1] - 25, 0)), font_size=20, color=(0, 255, 0))

    return frame, detected_objects

def main():
    cap = cv2.VideoCapture(0)
    last_spoken = ""
    last_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera.")
            break

        frame, detected_objects = detect_objects(frame)

        if detected_objects:
            current_objects = ", ".join(sorted(set(detected_objects)))
            # Speak if new objects or 5+ seconds passed
            if current_objects != last_spoken or time.time() - last_time > 5:
                print(f"Обнаружено: {current_objects}")
                speak(f"Я вижу: {current_objects}")
                last_spoken = current_objects
                last_time = time.time()

        cv2.imshow("MyVision", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
