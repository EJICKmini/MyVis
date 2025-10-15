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
    "чемодан", "фрисби", "лыжи", "сноуб
