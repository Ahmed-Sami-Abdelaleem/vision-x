from ultralytics import YOLO

if __name__ == '__main__':
    # Load YOLO model
    model = YOLO("models/yolo11x.pt")

    # Train the model
    model.train(
        data=r"D:\development\pytorch\grade-project\football-players-detection.v2i.yolov11\data.yaml",
        epochs=100,
        imgsz=640,
        device="0",
        batch=16,  # Adjust batch size if needed
        workers=8,  # Adjust number of workers if needed
    )