from ultralytics import YOLO



model = YOLO("models/best.pt");


res= model.predict("input_videos/test.mp4",save=True);



