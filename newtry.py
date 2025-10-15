import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# New imports for TTS and Russian text rendering
import pyttsx3
from PIL import ImageFont, ImageDraw, Image

# ------------------ Russian TTS Setup ------------------
engine = pyttsx3.init()
for voice in engine.getProperty('voices'):
    if "ru" in voice.languages or "Russian" in voice.name:
        engine.setProperty('voice', voice.id)
        break

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ------------------ Russian Label Drawing ------------------
def draw_russian_text(frame, text, position, font_path="arial.ttf", font_size=18, color=(0, 255, 0)):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

# ------------------ Parse Args ------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--thresh', default=0.5)
parser.add_argument('--resolution', default=None)
parser.add_argument('--record', action='store_true')
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

if not os.path.exists(model_path):
    print('ERROR: Model path is invalid.')
    sys.exit(0)

# ------------------ Load Model & Russian Labels ------------------
model = YOLO(model_path, task='detect')

# Russian COCO classes
labels = [
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

# ------------------ Source Type Detection ------------------
img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid.')
    sys.exit(0)

# ------------------ Resolution & Recorder Setup ------------------
resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

if record:
    if source_type not in ['video', 'usb']:
        print('Recording only works for video and camera sources.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    recorder = cv2.VideoWriter('demo1.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (resW, resH))

# ------------------ Input Source Setup ------------------
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1] in img_ext_list]
elif source_type in ['video', 'usb']:
    cap = cv2.VideoCapture(img_source if source_type == 'video' else usb_idx)
    if resize:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# ------------------ Misc Setup ------------------
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
               (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0
last_spoken = ""
last_time = time.time()

# ------------------ Main Inference Loop ------------------
while True:
    t_start = time.perf_counter()

    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('All images processed. Exiting.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type in ['video', 'usb']:
        ret, frame = cap.read()
        if not ret:
            print('No more frames or camera error.')
            break
    elif source_type == 'picamera':
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0
    detected_labels = []

    for i in range(len(detections)):
        xyxy_tensor = detections[i].xyxy.cpu()
        xyxy = xyxy_tensor.numpy().squeeze()
        xmin, ymin, xmax, ymax = xyxy.astype(int)

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            label_ymin = max(ymin, 25)

            # Draw label with PIL
            frame = draw_russian_text(frame, label, (xmin, label_ymin - 25))

            object_count += 1
            detected_labels.append(classname)

    # Speak if necessary
    if detected_labels:
        current_objects = ", ".join(set(detected_labels))
        if current_objects != last_spoken or time.time() - last_time > 5:
            print(f"Обнаружено: {current_objects}")
            speak(f"Я вижу: {current_objects}")
            last_spoken = current_objects
            last_time = time.time()

    # FPS
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLO + Russian TTS", frame)

        if record:
            recorder.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        cv2.imshow("YOLO + Russian TTS", frame)
        cv2.waitKey(0)
        break

# ------------------ Cleanup ------------------
if source_type in ['video', 'usb']:
    cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()
