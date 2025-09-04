import subprocess
import time
import streamlit as st
import cv2
import tempfile
import os
import onnxruntime
import pickle
import mediapipe as mp
from frame_utilities import *  # your helpers

providers = ["DmlExecutionProvider", "CPUExecutionProvider"]

# Person detection YOLO
session_coco = onnxruntime.InferenceSession("yolo11n.onnx", providers=providers)
input_name_coco = session_coco.get_inputs()[0].name
output_name_coco = session_coco.get_outputs()[0].name

# Volleyball detection YOLO
session_volleyball = onnxruntime.InferenceSession("yolo11n_vb.onnx", providers=providers)
input_name_volleyball = session_volleyball.get_inputs()[0].name
output_name_volleyball = session_volleyball.get_outputs()[0].name

# RandomForest for actions
model_dict = pickle.load(open("model.p", "rb"))
rf_model = model_dict["model"]

# MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

st.set_page_config(page_title="Real-Time Volleyball Action Detection", page_icon="🏐", layout="centered")
st.markdown("<h1 style='text-align: center;'>🏐 Real-Time Volleyball 🏐<br>Action Detection</h1>", unsafe_allow_html=True)

# Select task
task = st.radio("Choose a task:", ["Person Detection", "Volleyball Detection", "Action Detection"])

# Select input type
input_type = st.radio("Choose input type:", ["Video", "Image"])

if input_type == "Image":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image and st.button("Run Detection on Image"):
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)  # decode as OpenCV image (BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        start_time_inference = time.time()  # track inference runtime

        # ---- Run selected task ----
        if task == "Person Detection":
            # Detect people
            input_coco = preprocess_yolo_input(frame_rgb)
            coco_outs = session_coco.run([output_name_coco], {input_name_coco: input_coco})
            coco_boxes, coco_scores, coco_class_ids = postprocess_yolo_output(coco_outs[0], frame.shape,
                                                                              conf_threshold=0.5)

            # Draw person detections
            for box, score, cls in zip(coco_boxes, coco_scores, coco_class_ids):
                if cls == 0:  # "Person" class in COCO
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        elif task == "Volleyball Detection":
            # Detect volleyballs
            input_ball = preprocess_yolo_input(frame_rgb)
            ball_outs = session_volleyball.run([output_name_volleyball], {input_name_volleyball: input_ball})
            ball_boxes, ball_scores, _ = postprocess_yolo_output(ball_outs[0], frame.shape, conf_threshold=0.5)

            for box in ball_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Volleyball", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        elif task == "Action Detection":
            data = []
            original_frame_shape = frame.shape
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1
            thickness = 2
            action = "NONE"

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
                                    prediction = rf_model.predict([np.asarray(data)])
                                    action = str(prediction[0])

                                    if action != "NONE":

                                        x1, y1, x2, y2 = map(int, closest_person_box)
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        cv2.putText(frame, f"Person", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                                    (0, 255, 0), 2)

                                        x1, y1, x2, y2 = map(int, ball_box)
                                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                        cv2.putText(frame, f"Volleyball", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                                    (0, 0, 255), 2)

            action_text = f"Action: {action}"
            (text_w, text_h), baseline = cv2.getTextSize(action_text, font, scale,
                                                         thickness)
            x, y = 15, 35
            cv2.rectangle(frame, (x - 5, y - text_h - 5),
                          (x + text_w + 5, y + baseline + 5), (255, 255, 255), -1)
            cv2.putText(frame, action_text, (x, y), font, scale, (0, 0, 0), thickness,
                        cv2.LINE_AA)

        end_time_inference = time.time()
        total_time = end_time_inference - start_time_inference
        st.success(f"Total Processing Time: {total_time:.2f} seconds ✅")
        # Show result
        st.image(frame, channels="BGR")

        # Allow download of annotated image
        result_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
        cv2.imwrite(result_path, frame)
        with open(result_path, "rb") as f:
            st.download_button("⬇️ Download Processed Image", f, "processed_image.jpg", "image/jpeg")

elif input_type == "Video":
    # Upload video
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

    # Button to run
    if uploaded_video and st.button("Run Detection"):
        # Save video to temp file so OpenCV can read it
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        tfile.close()  # 🔑 Important for Windows, unlocks the file

        cap = cv2.VideoCapture(tfile.name)

        frame_count = 0
        fps_list = []
        start_time = time.time()

        # Prepare output video file
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # temporary codec
        temp_out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(temp_out_path, fourcc, fps, (width, height))

        stframe = st.empty()  # placeholder to stream frames in the app

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inference_time_start = time.time()

            # ----------------------
            # Person detection
            # ----------------------
            if task == "Person Detection":
                # Detect people
                input_coco = preprocess_yolo_input(frame_rgb)
                coco_outs = session_coco.run([output_name_coco], {input_name_coco: input_coco})
                coco_boxes, coco_scores, coco_class_ids = postprocess_yolo_output(coco_outs[0], frame.shape, conf_threshold=0.5)

                # Draw person detections
                for box, score, cls in zip(coco_boxes, coco_scores, coco_class_ids):
                    if cls == 0:  # "Person" class in COCO
                        x1, y1, x2, y2 = box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"Person {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # ----------------------
            # Volleyball detection
            # ----------------------
            elif task == "Volleyball Detection":
                # Detect volleyballs
                input_ball = preprocess_yolo_input(frame_rgb)
                ball_outs = session_volleyball.run([output_name_volleyball], {input_name_volleyball: input_ball})
                ball_boxes, ball_scores, _ = postprocess_yolo_output(ball_outs[0], frame.shape, conf_threshold=0.5)

                for box in ball_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Volleyball", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ----------------------
            # Action detection
            # ----------------------
            elif task == "Action Detection":
                data = []
                frame_count = 0
                last_action = "NONE"
                fps_list = []
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1
                thickness = 2

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
                    coco_boxes, coco_scores, coco_class_ids = postprocess_yolo_output(coco_outs[0], original_frame_shape,
                                                                                      conf_threshold=0.5)

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
                                        prediction = rf_model.predict([np.asarray(data)])
                                        last_action = str(prediction[0])

                                        if last_action != "NONE":
                                            action_text = f"Action: {last_action}"
                                            (text_w, text_h), baseline = cv2.getTextSize(action_text, font, scale,
                                                                                         thickness)
                                            x, y = 15, 100
                                            cv2.rectangle(frame, (x - 5, y - text_h - 5),
                                                          (x + text_w + 5, y + baseline + 5), (255, 255, 255), -1)
                                            cv2.putText(frame, action_text, (x, y), font, scale, (0, 0, 0), thickness,
                                                        cv2.LINE_AA)

                                            x1, y1, x2, y2 = map(int, closest_person_box)
                                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            cv2.putText(frame, f"Person", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                                        (0, 255, 0), 2)

                                            x1, y1, x2, y2 = map(int, ball_box)
                                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                            cv2.putText(frame, f"Volleyball", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                                        (0, 0, 255), 2)

            inference_time_end = time.time()
            fps = 1 / (inference_time_end - inference_time_start)
            fps_list.append(fps)
            frame_count += 1

            # Show FPS on frame
            fps_text = f"FPS: {int(fps)}"
            (text_w, text_h), baseline = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (10, 10), (15 + text_w, 15 + text_h + baseline), (255, 255, 255), -1)
            cv2.putText(frame, fps_text, (15, 15 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # Show processed frame and save it to output video
            stframe.image(frame, channels="BGR")
            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()
        stframe.empty()
        os.remove(tfile.name)

        end_time = time.time()
        total_time = end_time - start_time
        avg_fps_global = frame_count / total_time
        st.success(f"Processed {frame_count} frames ✅\n\n"
                   f"Total Processing Time: {total_time:.2f} seconds ✅\n\n"
                   f"Average FPS: {avg_fps_global:.2f} ✅")

        # ----------------------
        # Convert to H264 using ffmpeg
        # ----------------------
        h264_out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        cmd = [
            "ffmpeg", "-y", "-i", temp_out_path,
            "-vcodec", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            h264_out_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(temp_out_path)

        # Display video
        st.video(h264_out_path)

        # Allow download
        with open(h264_out_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Processed Video",
                data=f,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
