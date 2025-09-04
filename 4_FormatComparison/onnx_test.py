import cv2
import time
from ultralytics import YOLO

# Model Loading
person_model = YOLO("models/yolo11n.onnx")
volleyball_model = YOLO("models/yolo11n_vb.onnx")

# Input
video_path = "Volleyball_Video.mp4"
cap = cv2.VideoCapture(video_path)  # 0 instead of filename for webcam use

if not cap.isOpened():
    print("Error: Could not open input.")
    exit()

# FPS tracking
frame_count = 0
fps_list = []
start_time = time.time()  # track global runtime

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_inference_time = time.time()  # track inference runtime

    # Detect people and volleyballs
    person_results = person_model.predict(frame, classes=0, verbose=False)
    volleyball_results = volleyball_model.predict(frame, verbose=False)

    # Draw person detections
    for box in person_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw volleyball detections
    for box in volleyball_results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Volleyball {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Calculate FPS (instantaneous, per-frame)
    end_inference_time = time.time()
    fps = 1 / (end_inference_time - start_inference_time)
    fps_list.append(fps)
    frame_count += 1

    # Show FPS on frame
    fps_text = f"FPS: {fps:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (15, 15), (15 + text_w, 15 + text_h + baseline), (255, 255, 255), -1)
    cv2.putText(frame, fps_text, (15, 15 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("ONNX Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

end_time = time.time()

cap.release()
cv2.destroyAllWindows()

# Results
if fps_list:
    total_time = end_time - start_time
    avg_fps_global = frame_count / total_time
    print(f"Processed {frame_count} frames.")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average FPS (overall): {avg_fps_global:.2f}")
else:
    print("No frames processed.")
