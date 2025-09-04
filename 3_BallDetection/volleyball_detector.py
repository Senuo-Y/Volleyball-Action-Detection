from ultralytics import YOLO
import cv2
import time

# Model Loading. Options: "models/yolo11n_vb.pt", "models/yolo11s_vb.pt"
model = YOLO("models/yolo11n_vb.pt")

# Input
input_video_path = "videos/PeopleVideo.mp4"
cap = cv2.VideoCapture(input_video_path) # 0 instead of filename for webcam use

if not cap.isOpened():
    print("Error: Could not open input.")
    exit()

print(f"Reading video from {input_video_path}.")

# FPS tracking
frame_count = 0
fps_list = []
prev_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video file or failed to read frame.")
        break

    start_time = time.time()

    # Detect people
    results = model.predict(frame, verbose=False)

    # Draw detections and FPS on the frame
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Volleyball: {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    fps_list.append(fps)
    frame_count += 1

    # Show FPS
    fps_text = f"FPS: {fps:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (15, 15), (15 + text_w, 15 + text_h + baseline), (255, 255, 255), -1)
    cv2.putText(frame, fps_text, (15, 15 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Person Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Average FPS
if fps_list:
    avg_fps = sum(fps_list) / len(fps_list)
    print(f"Processed {frame_count} frames.")
    print(f"Average FPS: {avg_fps:.2f}")
else:
    print("No frames processed.")