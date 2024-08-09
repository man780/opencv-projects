import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import h5py
import json
import tensorflow as tf
from tensorflow.keras.models import model_from_json


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
# classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Загрузка конфигурации модели
with h5py.File("Model/keras_model.h5", "r") as f:
    model_config = f.attrs.get("model_config")

# Десериализация конфигурации модели (если это строка)
if isinstance(model_config, bytes):
    model_config = model_config.decode('utf-8')

model_config = json.loads(model_config)

# Изменение конфигурации для удаления 'groups' из слоев DepthwiseConv2D
for layer in model_config['config']['layers']:
    if layer['class_name'] == 'DepthwiseConv2D':
        if 'groups' in layer['config']:
            del layer['config']['groups']

# Сериализация измененной конфигурации
model_config = json.dumps(model_config)

# Сохранение измененной модели
with h5py.File("Model/modified_keras_model.h5", "w") as f:
    f.attrs["model_config"] = model_config

# Загрузка измененной модели
with h5py.File("Model/modified_keras_model.h5", "r") as f:
    modified_model_config = f.attrs.get("model_config")
    if isinstance(modified_model_config, bytes):
        modified_model_config = modified_model_config.decode('utf-8')
    modified_model = model_from_json(modified_model_config)
    modified_model.load_weights("Model/keras_model.h5")

# Теперь можно использовать измененную модель
classifier = modified_model

offset = 20
imgSize = 300
labels = ["A", "B", "C"]

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            # print(prediction, index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
