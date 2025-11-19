import time
import threading
import numpy as np
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
from gtts import gTTS
import os

WEIGHTS = "/home/sst/projects/CIS/yolov8n.pt"
CONF_THRESHOLD = 0.30
IOU_THRESHOLD = 0.45
TTS_FILENAME = "tts_temp.mp3"
TTS_LOCK = threading.Lock()
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

model = YOLO(WEIGHTS)


CLASS_NAMES = [
    "человек", "велосипед", "машина", "мотоцикл", "самолет", "автобус",
    "поезд", "грузовик", "лодка", "светофор", "пожарный гидрант",
    "знак стоп", "паркометр", "скамейка", "птица", "кошка", "собака",
    "лошадь", "овца", "корова", "слон", "медведь", "зебра", "жираф",
    "рюкзак", "зонт", "дамская сумка", "галстук", "чемодан", "фрисби",
    "лыжи", "сноуборд", "мяч", "воздушный змей", "бита", "перчатка",
    "скейтборд", "серфборд", "теннисная ракетка", "бутылка", "бокал",
    "чашка", "вилка", "нож", "ложка", "миска", "банан", "яблоко",
    "сэндвич", "апельсин", "брокколи", "морковь", "хот-дог", "пицца",
    "пончик", "торт", "стул", "диван", "растение", "кровать",
    "обеденный стол", "туалет", "телевизор", "ноутбук", "мышь",
    "пульт", "клавиатура", "телефон", "микроволновка", "духовка",
    "тостер", "раковина", "холодильник", "книга", "часы", "ваза",
    "ножницы", "медвежонок", "фен", "зубная щетка"
]


def draw_label(frame, text, x, y):
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2, cv2.LINE_AA)



def speak_async(text):
    def worker():
        with TTS_LOCK:
            try:
                tts = gTTS(text=text, lang='ru')
                tts.save(TTS_FILENAME)
                os.system(f"mpg123 {TTS_FILENAME} >/dev/null 2>&1")
                os.remove(TTS_FILENAME)
            except Exception as e:
                print("TTS error:", e)

    threading.Thread(target=worker, daemon=True).start()



def main():
    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}
    )
    picam2.configure(config)
    picam2.start()

    last_detected = ""
    last_time = time.time()

    try:
        while True:
            frame = picam2.capture_array()

            results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]

            detected_classes = []

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls < len(CLASS_NAMES):
                    name = CLASS_NAMES[cls]
                else:
                    name = str(cls)

                detected_classes.append(name)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                draw_label(frame, f"{name} {conf:.2f}", x1, y1 - 10)

            if detected_classes:
                joined = ", ".join(sorted(set(detected_classes)))
                if joined != last_detected or time.time() - last_time > 1:
                    speak_async(f"Я вижу {joined}")
                    last_detected = joined
                    last_time = time.time()

            cv2.imshow("MyVision", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
