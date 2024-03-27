from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.yaml').load('/Users/jadeja19/Downloads/yolov8n-seg.pt')  # build from YAML and transfer weights


# Train the model
results = model.train(data='/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/Segmentation/iktd.v1i.yolov8/data.yaml', epochs=2, imgsz=640)