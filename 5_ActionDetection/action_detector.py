import os
import cv2
import pickle
import mediapipe as mp
import numpy as np
import onnxruntime
import time
from frame_utilities import *

providers = ["DmlExecutionProvider", "CPUExecutionProvider"]

# Load YOLO model
session_coco = onnxruntime.InferenceSession("yolo11n.onnx", providers=providers)
input_name_coco = session_coco.get_inputs()[0].name
output_name_coco = session_coco.get_outputs()[0].name
print(f"1st ONNX model loaded successfully using providers: {session_coco.get_providers()}.")

# Load volleyball YOLO model
session_volleyball = onnxruntime.InferenceSession("yolo11n_vb.onnx", providers=providers)
input_name_volleyball = session_volleyball.get_inputs()[0].name
output_name_volleyball = session_volleyball.get_outputs()[0].name
print(f"2nd ONNX model loaded successfully using providers: {session_volleyball.get_providers()}.")

# Load the trained RandomForestClassifier model
model_dict = pickle.load(open("model.p", "rb"))
model = model_dict["model"]
print("RandomForestClassifier model loaded.")

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
print("MediaPipe Pose model initialized.")

cap = cv2.VideoCapture("../2_PoseEstimation/videos/ThreePeopleVideo.mp4")
if not cap.isOpened():
    print("Error: Could not open input.")
    exit()

frame_count = 0
last_action = "NONE"
fps_list = []
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
thickness = 2
start_time = time.time() # track global runtime

while cap.isOpened():

    data = []
    last_action = "NONE"

    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_frame_shape = frame.shape

    start_time_inference = time.time()  # track inference runtime

    # Detect volleyballs
    input_ball = preprocess_yolo_input(frame_rgb)
    ball_outs = session_volleyball.run([output_name_volleyball], {input_name_volleyball: input_ball})
    ball_boxes, ball_scores, _ = postprocess_yolo_output(ball_outs[0], original_frame_shape, conf_threshold=0.5)

    if len(ball_boxes) > 0:

        # Detect people
        input_coco = preprocess_yolo_input(frame_rgb)
        coco_outs = session_coco.run([output_name_coco], {input_name_coco: input_coco})
        coco_boxes, coco_scores, coco_class_ids = postprocess_yolo_output(coco_outs[0], original_frame_shape, conf_threshold=0.5)

        person_boxes = []
        person_scores = []

        for i, class_id in enumerate(coco_class_ids):
            if class_id == 0:  # COCO class 0 = person
                person_boxes.append(coco_boxes[i])
                person_scores.append(coco_scores[i])

        if len(person_boxes) > 0:

            # Take the most confidently detected ball
            ball_box_index = np.argmax(ball_scores)
            ball_box = ball_boxes[ball_box_index]
            ball_x_min, ball_y_min, ball_x_max, ball_y_max = ball_box

            closest_person_box = person_boxes[0]
            min_distance = get_distance_person_ball_np(closest_person_box, ball_box)
            for person_box in person_boxes:
                distance = get_distance_person_ball_np(person_box, ball_box)
                if distance < min_distance:
                    closest_person_box = person_box
                    min_distance = distance

            person_x_min, person_y_min, person_x_max, person_y_max = closest_person_box

            # Clip coordinates to be within frame bounds (important for cropping)
            person_x_min = max(0, min(person_x_min, original_frame_shape[1] - 1))
            person_y_min = max(0, min(person_y_min, original_frame_shape[0] - 1))
            person_x_max = max(0, min(person_x_max, original_frame_shape[1]))
            person_y_max = max(0, min(person_y_max, original_frame_shape[0]))

            # Ensure valid crop dimensions before proceeding
            if person_x_min <= person_x_max and person_y_min <= person_y_max:

                person_frame_roi = frame_rgb[person_y_min:person_y_max, person_x_min:person_x_max]

                # Check if ROI is not empty after clipping
                if person_frame_roi.size > 0:

                    square_person_frame, pad_left, pad_top = pad_frame_to_square(person_frame_roi)
                    pose_results = pose.process(square_person_frame)

                    if pose_results.pose_landmarks:

                        # Selecting relevant landmarks and getting absolute coords
                        relevant_landmarks = pose_results.pose_landmarks.landmark[11:25]

                        # Get min/max coordinates of selected pose landmarks for normalization
                        pose_x_coords = [landmark.x for landmark in relevant_landmarks]
                        pose_y_coords = [landmark.y for landmark in relevant_landmarks]

                        pose_x_min, pose_x_max = min(pose_x_coords), max(pose_x_coords)
                        pose_y_min, pose_y_max = min(pose_y_coords), max(pose_y_coords)

                        # Calculate ranges for normalization, add small epsilon to avoid division by zero
                        x_range = pose_x_max - pose_x_min
                        y_range = pose_y_max - pose_y_min
                        if x_range == 0: x_range = 1e-6
                        if y_range == 0: y_range = 1e-6

                        # Normalize pose landmark coordinates and add pose to data_aux
                        for landmark in relevant_landmarks:
                            pose_x_normalized = (landmark.x - pose_x_min) / x_range
                            pose_y_normalized = (landmark.y - pose_y_min) / y_range
                            data.append(pose_x_normalized)
                            data.append(pose_y_normalized)

                        # Make ball coords relative to pose bounding box
                        ball_x_min_relative = (ball_x_min - pose_x_min) / x_range
                        ball_y_min_relative = (ball_y_min - pose_y_min) / y_range
                        ball_size_x = (ball_x_max - ball_x_min) / x_range
                        ball_size_y = (ball_y_max - ball_y_min) / y_range
                        ball_diameter = max(ball_size_x, ball_size_y)

                        # Add ball data to data_aux
                        data.append(ball_x_min_relative)
                        data.append(ball_y_min_relative)
                        data.append(ball_diameter)

                        # Ensure data_aux has the correct number of features for your model (2 * 14 + 3 = 31)
                        if len(data) == 31:
                            prediction = model.predict([np.asarray(data)])
                            last_action = str(prediction[0])

                            if last_action != "NONE":
                                action_text = f"Action: {last_action}"
                                (text_w, text_h), baseline = cv2.getTextSize(action_text, font, scale, thickness)
                                x, y = 15, 100
                                cv2.rectangle(frame, (x - 5, y - text_h - 5), (x + text_w + 5, y + baseline + 5), (255, 255, 255), -1)
                                cv2.putText(frame, action_text, (x, y), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

                                x1, y1, x2, y2 = map(int, closest_person_box)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"Person", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                                x1, y1, x2, y2 = map(int, ball_box)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(frame, f"Volleyball", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Calculate FPS (instantaneous, per-frame)
    end_time_inference = time.time()
    fps = 1 / (end_time_inference - start_time_inference)
    fps_list.append(fps)
    frame_count += 1

    # Show FPS on frame
    fps_text = f"FPS: {int(fps)}"
    (text_w, text_h), baseline = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (10, 10), (15 + text_w, 15 + text_h + baseline), (255, 255, 255), -1)
    cv2.putText(frame, fps_text, (15, 15 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Volleyball Action Detection", frame)
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