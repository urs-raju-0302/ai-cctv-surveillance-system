import torch

# Load YOLOv5 nano model (lightweight)
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

# Test image from internet
results = model('https://ultralytics.com/images/zidane.jpg')

# Show results
results.show()