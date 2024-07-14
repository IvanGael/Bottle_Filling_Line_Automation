import cv2
import numpy as np
from ultralytics import YOLO
import time

# Load the YOLOv8 model
model = YOLO('yolov8n-seg')

# Open the video file
video_path = "video.mp4"
cap = cv2.VideoCapture(video_path)

# Get the video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create VideoWriter object
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Calculate the position of the vertical bar (80% of the width)
bar_position = int(frame_width * 0.8)

# Initialize persistent counter for right objects
total_right_objects = 0
previous_right_objects = set()

# Initialize variables for additional features
start_time = time.time()
bottles_per_second = 0
defective_bottles = 0
efficiency = 100.0

while True:
    # Read a frame from the video
    success, frame = cap.read()

    if not success:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame, classes=39)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Draw the vertical bar
    cv2.line(annotated_frame, (bar_position, 0), (bar_position, frame_height), (204, 0, 204), 2)

    # Count objects
    total_objects = len(results[0])
    left_objects = sum(1 for box in results[0].boxes.xyxy if ((box[0] + box[2]) / 2) < bar_position)
    right_objects = total_objects - left_objects

    # Update persistent counter for right objects
    current_right_objects = set()
    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = [int(v) for v in box[:4].tolist()]
        x_center = (x1 + x2) // 2
        if x_center >= bar_position:
            object_id = f"{x_center}_{y1}_{y2}" 
            if total_right_objects <= total_objects:
                current_right_objects.add(object_id)
                if object_id not in previous_right_objects:
                    total_right_objects += right_objects
    
    previous_right_objects = current_right_objects

    # Calculate bottles per second
    elapsed_time = time.time() - start_time
    bottles_per_second = int((len(results[0])) / elapsed_time) + len(results[0])

    # Simulate defective bottle detection (random for demonstration)
    if np.random.random() < 0.05:  # 5% chance of defective bottle
        defective_bottles = 0

    # Calculate efficiency
    if total_right_objects > 0:
        efficiency_eval = ((len(results[0]) - 2) / total_right_objects) * 100
        if efficiency_eval <= 100:
            efficiency = int(((len(results[0]) - 2) / total_right_objects) * 100)
        else:
            efficiency = 100

    # Draw count circles and text
    cv2.circle(annotated_frame, (50, 50), 40, (204, 0, 102), -1)
    cv2.putText(annotated_frame, str(left_objects), (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(annotated_frame, "Unfilled", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.circle(annotated_frame, (frame_width - 50, 50), 40, (0, 255, 0), -1)
    cv2.putText(annotated_frame, str(total_right_objects), (frame_width - 65, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(annotated_frame, "Filled", (frame_width - 80, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display additional information
    cv2.putText(annotated_frame, f"Bottles/sec: {bottles_per_second:.2f}", (10, frame_height - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Defective: {defective_bottles}", (10, frame_height - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(annotated_frame, f"Efficiency: {efficiency:.2f}%", (10, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Simulate production line speed control based on bottles per second
    if bottles_per_second < 0.8:  # Assuming 0.8 bottles/sec is the lower threshold
        cv2.putText(annotated_frame, "Speed: Decrease", (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    elif bottles_per_second > 1.7:  # Assuming 1.7 bottles/sec is the upper threshold
        cv2.putText(annotated_frame, "Speed: Increase", (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        cv2.putText(annotated_frame, "Speed: Optimal", (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Write the frame into the output file
    out.write(annotated_frame)

    # Display the annotated frame
    cv2.imshow("Bottle Filling Line Automation", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()