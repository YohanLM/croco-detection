import kagglehub
from ultralytics import YOLO

# Download latest version
path = kagglehub.dataset_download("andrewmvd/pothole-detection")

# Load the YOLO model
model = YOLO("yolo11n.pt")

results = model.train(
    data="potholes_50.yaml", 
    epochs=50, 
    imgsz=640, 
    device="mps"
)