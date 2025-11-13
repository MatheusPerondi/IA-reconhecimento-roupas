from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Treinar
model.train(
    data="C:\Trbalho rede neural\modelo\data.yaml",  
    epochs=80,                      
    imgsz=640,                       
    batch=16                         
)
