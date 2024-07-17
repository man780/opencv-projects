from mtcnn import MTCNN
import cv2
import numpy as np


"""
@description: we used https://github.com/ipazc/mtcnn repository
"""


def blur_apply(image, depth):
    detector = MTCNN()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image)
    for face in faces:
        x, y, w, h = face['box']
        face = image[y:y + h, x:x + w]
        blurred_face = cv2.GaussianBlur(face, (47, 47), depth)
        mask = np.zeros_like(face)
        cv2.circle(mask, (w // 2, h // 2), min(w, h) // 2, (255, 255, 255), -1)
        face[np.where(mask)] = blurred_face[np.where(mask)]
        image[y:y + h, x:x + w] = face

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)