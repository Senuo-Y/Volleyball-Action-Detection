import mediapipe as mp
import cv2
from ultralytics import YOLO
import time

# --- Pad frame to square and return padding info ---
def pad_frame_to_square(frame):
    h, w, _ = frame.shape
    if h == w:
        return frame, 0, 0
    elif h > w:
        padding = h - w
        pad_left = padding // 2
        pad_right = padding - pad_left
        padded = cv2.copyMakeBorder(frame, 0, 0, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded, pad_left, 0
    else:
        padding = w - h
        pad_top = padding // 2
        pad_bottom = padding - pad_top
        padded = cv2.copyMakeBorder(frame, pad_top, pad_bottom, 0, 0,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded, 0, pad_top

# --- Translate landmarks back to original coordinates ---
def translate_landmarks(landmarks, bbox, padded_shape, original_shape, pad_x, pad_y):
    person_x_min, person_y_min, person_x_max, person_y_max = bbox
    roi_w = person_x_max - person_x_min
    roi_h = person_y_max - person_y_min
    padded_w, padded_h = padded_shape[1], padded_shape[0]
    H, W = original_shape[0], original_shape[1]

    for lm in landmarks.landmark:
        # Scale from padded square ROI to original ROI
        x_rel = (lm.x * padded_w - pad_x) / roi_w
        y_rel = (lm.y * padded_h - pad_y) / roi_h

        # Translate to full-frame coords (normalize to [0,1])
        lm.x = (person_x_min + x_rel * roi_w) / W
        lm.y = (person_y_min + y_rel * roi_h) / H

    return landmarks

# MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Model Loading
person_model = YOLO("yolo11n.pt")

# Input. Options: "videos/OnePersonVideo.mp4", "videos/ThreePeopleVideo.mp4", "videos/ManyPeopleVideo.mp4"
input_video_path = "videos/ManyPeopleVideo.mp4"
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Error: Could not open input.")
    exit()

# FPS tracking
frame_count = 0
fps_list = []
start_time = time.time() # track global runtime

print("--- Starting processing loop ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to grab frame.")
        break

    start_time_inference = time.time()  # track inference runtime

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_frame_shape = frame.shape

    # Detect people with YOLO
    results = person_model(frame, classes=0, verbose=False)

    box = results[0].boxes[0] # Choose only the first person
    person_x_min, person_y_min, person_x_max, person_y_max = map(int, box.xyxy[0])
    conf = float(box.conf[0])

    # Draw detection box
    cv2.rectangle(frame, (person_x_min, person_y_min), (person_x_max, person_y_max), (0, 255, 0), 2)
    cv2.putText(frame, f"Person {conf:.2f}", (person_x_min, person_y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Clip box to frame size
    person_x_min = max(0, min(person_x_min, original_frame_shape[1] - 1))
    person_y_min = max(0, min(person_y_min, original_frame_shape[0] - 1))
    person_x_max = max(0, min(person_x_max, original_frame_shape[1]))
    person_y_max = max(0, min(person_y_max, original_frame_shape[0]))

    if person_x_min < person_x_max and person_y_min < person_y_max:
        roi = frame_rgb[person_y_min:person_y_max, person_x_min:person_x_max]

        if roi.size != 0:
            square_roi, pad_x, pad_y = pad_frame_to_square(roi)
            pose_results = pose.process(square_roi)

            if pose_results.pose_landmarks:
                translated = translate_landmarks(
                    pose_results.pose_landmarks,
                    (person_x_min, person_y_min, person_x_max, person_y_max),
                    square_roi.shape, original_frame_shape, pad_x, pad_y
                )

                # Draw translated landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    translated,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

    # Calculate FPS (instantaneous, per-frame)
    end_time_inference = time.time()
    fps = 1 / (end_time_inference - start_time_inference)
    fps_list.append(fps)
    frame_count += 1

    # Show FPS on frame
    fps_text = f"FPS: {fps:.2f}"
    (text_w, text_h), baseline = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (15, 15), (15 + text_w, 15 + text_h + baseline), (255, 255, 255), -1)
    cv2.putText(frame, fps_text, (15, 15 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("People Detection + Pose", frame)
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
