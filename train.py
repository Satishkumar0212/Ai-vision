from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Train
model.train(data="C:/Users/satis_asxzdl9/Ai-vision/data.yaml", epochs=100, imgsz=640, device=0)

import sys
print(sys.path)