'''
实例分割训练
'''
from ultralytics import YOLO

#train
model = YOLO('./ultralytics/cfg/models/v8/yolov8-seg.yaml').load('yolov8n-seg.pt')

# Train the model
model.train(data='./ultralytics/cfg/datasets/human_body-seg.yaml', epochs=300, imgsz=640, batch=8, workers=0)
