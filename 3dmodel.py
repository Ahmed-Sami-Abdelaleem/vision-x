import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO

# Load YOLOv8 Model
model = YOLO("yolov11x.pt")  # Use "yolov8m.pt" for better accuracy

# Video Capture
cap = cv2.VideoCapture("videos/video.mp4")  # Change to your video file

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Video Writer Setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

# Define goal line coordinates (adjust as per field dimensions)
goal_line_left = (100, 500)
goal_line_right = (1000, 500)

# Player Tracking
player_tracks = {}  # Dictionary to store player positions over time
prev_time = time.time()

# Function to calculate speed
def calculate_speed(prev_pos, curr_pos, time_interval):
    if prev_pos is None or curr_pos is None:
        return 0
    dist = np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))  # Euclidean distance
    speed = dist / time_interval  # Pixels per second
    return round(speed * 0.1, 2)  # Convert to estimated real-world m/s

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit when video ends

    # Get time difference
    curr_time = time.time()
    time_interval = curr_time - prev_time
    prev_time = curr_time

    # Run YOLO on the frame
    results = model(frame)

    # Ball and goal detection
    ball_position = None
    goal_scored = False

    # Process detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            obj_x, obj_y = (x1 + x2) // 2, (y1 + y2) // 2  # Get center of object

            if label == "sports ball":
                ball_position = (obj_x, obj_y)

                # Check if ball crosses goal line
                if goal_line_left[0] < obj_x < goal_line_right[0] and obj_y >= goal_line_left[1]:
                    goal_scored = True
                color = (0, 0, 255)  # Red for ball

            elif label == "person":
                # Crop player's jersey region
                player_img = frame[y1:y1 + (y2 - y1) // 2, x1:x2]
                avg_color = np.mean(player_img, axis=(0, 1))  # Average color

                # Classify players by jersey color
                if avg_color[2] > avg_color[1] and avg_color[2] > avg_color[0]:  # Red-dominant
                    team = "Team A"
                    color = (0, 0, 255)  # Red for Team A
                elif avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:  # Blue-dominant
                    team = "Team B"
                    color = (255, 0, 0)  # Blue for Team B
                else:
                    team = "Referee"
                    color = (255, 255, 0)  # Yellow for referee

                # Track player movements
                player_id = f"{obj_x}_{obj_y}"
                if player_id not in player_tracks:
                    player_tracks[player_id] = []
                player_tracks[player_id].append((obj_x, obj_y))

                # Calculate speed
                prev_pos = player_tracks[player_id][-2] if len(player_tracks[player_id]) > 1 else None
                speed = calculate_speed(prev_pos, (obj_x, obj_y), time_interval)

                # Display player info
                cv2.putText(frame, f"{team} | {speed} m/s", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Display goal message
    if goal_scored:
        cv2.putText(frame, "GOAL!", (400, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    # Write frame to output video
    out.write(frame)

    # Display frame (optional)
    cv2.imshow("Football Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to exit

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Video processing complete! Saved as output.mp4")
