from ultralytics import YOLO
import os

# Absolute path to dataset
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_YAML = os.path.join(BASE_DIR, 'Vehicle-Registration-Plates-1', 'data.yaml')

print(f"Using dataset: {DATA_YAML}")
print(f"Dataset exists: {os.path.exists(DATA_YAML)}")

model = YOLO('yolov8n.pt')

results = model.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,
    batch=8,
    name='indian_plate_detector',
    patience=10,
    save=True,
    exist_ok=True
)

print("Training complete!")
print("Best model at: runs/detect/indian_plate_detector/weights/best.pt")