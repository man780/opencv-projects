import tensorflow as tf

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from io import BytesIO

import numpy as np
import cv2
from MTCNN.blur_apply import blur_apply, detect_face, detect_face_2

"""
* Optional
* This part is set to be processed only by the CPU. 
* If you are using a supported graphics card and will do the 
* processing via this graphics card, you can remove this part.
"""
tf.config.set_visible_devices([], 'GPU')

"""
#ReadMe 
* repositories : ['mtcnn':'face detection', 'tensorflow':'core face detection']
"""
app = FastAPI(
    title="OpenCV Detection",
    description="Detection APIs",
    version="1.0.0"
)


@app.post("/face/blur", tags=["Face"])
async def image_upload(
        file: UploadFile = File(..., description="Uploaded File"),
        depth: int = Form(30, title="Depth", description="Blur Depth", ge=0, le=100)
):
    content_type = file.content_type
    if content_type not in ["image/jpeg"]:
        raise HTTPException(status_code=422, detail={"error": str("Invalid content type")})

    file_content = await file.read()
    i_buffer = np.frombuffer(file_content, np.uint8)
    image = cv2.imdecode(i_buffer, cv2.IMREAD_COLOR)
    processed_image = blur_apply(image, depth)

    _, img_encoded = cv2.imencode('.jpg', processed_image)
    return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/jpeg")


@app.post("/face/detect", tags=["Face"])
async def face_detect(
        file: UploadFile = File(..., description="Uploaded File"),
):
    content_type = file.content_type
    if content_type not in ["image/jpeg"]:
        raise HTTPException(status_code=422, detail={"error": str("Invalid content type")})
    file_content = await file.read()
    i_buffer = np.frombuffer(file_content, np.uint8)
    image = cv2.imdecode(i_buffer, cv2.IMREAD_COLOR)
    processed_image = detect_face(image)

    _, img_encoded = cv2.imencode('.jpg', processed_image)
    return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/jpeg")


@app.post("/face/detect2", tags=["Face"])
async def face_detect2(
        file: UploadFile = File(..., description="Uploaded File"),
):
    content_type = file.content_type
    if content_type not in ["image/jpeg"]:
        raise HTTPException(status_code=422, detail={"error": str("Invalid content type")})
    file_content = await file.read()
    i_buffer = np.frombuffer(file_content, np.uint8)
    image = cv2.imdecode(i_buffer, cv2.IMREAD_COLOR)
    processed_image = detect_face_2(image)

    _, img_encoded = cv2.imencode('.jpg', processed_image)
    return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/jpeg")
