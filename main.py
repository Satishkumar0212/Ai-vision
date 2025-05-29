from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load the trained YOLOv8 model
model = YOLO("C:/Users/satis_asxzdl9/Ai-vision/runs/train/exp/weights/best.pt")  # Update if path differs

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    # Run inference
    results = model.predict(image_np, conf=0.25)

    # Process detections
    detections = []
    for result in results:
        for box in result.boxes:
            label = result.names[int(box.cls)]
            confidence = float(box.conf)
            x, y, w, h = box.xywh[0].tolist()  # x_center, y_center, width, height
            detections.append({
                "label": label,
                "confidence": confidence,
                "bbox": [x, y, w, h]
            })

    return {"detections": detections}
results = model.predict(image_np, conf=0.25, device="cpu")