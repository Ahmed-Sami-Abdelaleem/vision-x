from ultralytics import YOLO
import shutil
model = YOLO("yolo11x.pt");

dataset = "football-players-detection.v2i.yolov11";