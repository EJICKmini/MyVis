import cv2
import torch
import time
import numpy as np
import pyttsx3
import sys
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image
import os

YOLOV7_PATH = "yolov7"
sys.path.append(YOLOV7_PATH)
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

#Settings
WEIGHTS = "yolov7.pt"
IMG_SIZE = 640
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

device = select_device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load("yolov7/yolov7.pt", map_location=device)
model.eval()


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

#TTS setup
engine = pyttsx3.init()
for voice in engine.getProperty('voices'):
    if "ru" in voice.languages or "Russian" in voice.name:
        engine.setProperty('voice', voice.id)
        break

def speak(text):
    engine.say(text)
    engine.runAndWait()

#Function for drawing different languages text using PIL
def draw_russian_text(frame, text, position, font_path="arial.ttf", font_size=18, color=(0, 255, 0)):
    # convert to PIL Image
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

#Detection
def detect_objects(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(img)[0]
    pred = non_max_suppression(pred, CONF_THRESHOLD, IOU_THRESHOLD)[0]

    detected_objects = []

    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], frame.shape).round()

        for *xyxy, conf, cls in pred:
            class_id = int(cls)
            class_name = CLASS_NAMES[class_id]
            detected_objects.append(class_name)

            label = f"{class_name} {conf:.2f}"
            # draw rectangle
            cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])),
                          (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            #Replace cv2.putText with PIL-based text rendering
            frame = draw_russian_text(
                frame,
                label,
                (int(xyxy[0]), int(xyxy[1]) - 25),
                font_path="arial.ttf",  #check font file !
                font_size=20,
                color=(0, 255, 0)
            )

    return frame, detected_objects

#Main loop
def main():
    cap = cv2.VideoCapture(0)
    last_spoken = ""
    last_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, detected_objects = detect_objects(frame)

        if detected_objects:
            current_objects = ", ".join(set(detected_objects))
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
