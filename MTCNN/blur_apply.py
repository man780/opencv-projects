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


def detect_face(image):
    detector = MTCNN()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image)
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def detect_face_2(image):
    detector = MTCNN()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detector.detect_faces(image)

    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    bounding_box = result[0]['box']
    keypoints = result[0]['keypoints']

    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0, 155, 255),
                  2)

    cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
    cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

    cv2.imwrite(image, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    print(result)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
