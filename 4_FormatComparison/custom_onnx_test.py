import cv2
import time
import onnxruntime
import numpy as np

def preprocess_yolo_input(image_rgb, input_size=(640, 640)):
    resized = cv2.resize(image_rgb, input_size)
    input_data = resized.astype(np.float32) / 255.0
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def postprocess_yolo_output(output, original_img_shape, input_size=(640, 640),
                            conf_threshold=0.25, nms_threshold=0.45):
    output = np.squeeze(output)

    if output.shape[0] < output.shape[1]:
        output = output.T

    num_features = output.shape[1]

    if num_features == 5:  # ball model (single class)
        boxes_raw = output[:, :4]
        scores = output[:, 4]
        class_ids = np.zeros(len(scores), dtype=int)
    elif num_features == 84:  # COCO person model
        boxes_raw = output[:, :4]
        class_scores = output[:, 4:]
        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
    else:
        return np.array([]).reshape(0,4), np.array([]), np.array([])

    valid_mask = scores > conf_threshold
    boxes_filtered = boxes_raw[valid_mask]
    scores_filtered = scores[valid_mask]
    class_ids_filtered = class_ids[valid_mask]

    if len(boxes_filtered) == 0:
        return np.array([]).reshape(0,4), np.array([]), np.array([])

    img_h, img_w = original_img_shape[:2]
    input_h, input_w = input_size

    scale_x = img_w / input_w
    scale_y = img_h / input_h

    x_center, y_center, width, height = boxes_filtered[:, 0], boxes_filtered[:, 1], boxes_filtered[:, 2], boxes_filtered[:, 3]

    x1 = (x_center - width / 2) * scale_x
    y1 = (y_center - height / 2) * scale_y
    x2 = (x_center + width / 2) * scale_x
    y2 = (y_center + height / 2) * scale_y

    boxes_final = np.clip(np.stack([x1, y1, x2, y2], axis=1), 0, [img_w, img_h, img_w, img_h]).astype(int)

    # NMS
    boxes_nms_input = np.array([[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes_final])
    indices = cv2.dnn.NMSBoxes(boxes_nms_input.tolist(), scores_filtered.tolist(), conf_threshold, nms_threshold)
    if len(indices) == 0:
        return np.array([]).reshape(0,4), np.array([]), np.array([])

    indices = indices.flatten()
    return boxes_final[indices], scores_filtered[indices], class_ids_filtered[indices]

# --- Load ONNX models ---
providers = ["DmlExecutionProvider", "CPUExecutionProvider"]

session_person = onnxruntime.InferenceSession("models/yolo11n.onnx", providers=providers)
input_name_person = session_person.get_inputs()[0].name
output_name_person = session_person.get_outputs()[0].name

session_ball = onnxruntime.InferenceSession("models/yolo11n_vb.onnx", providers=providers)
input_name_ball = session_ball.get_inputs()[0].name
output_name_ball = session_ball.get_outputs()[0].name

print(f"Person ONNX model providers: {session_person.get_providers()}")
print(f"Volleyball ONNX model providers: {session_ball.get_providers()}")
# -------------------------

# Input
video_path = "Volleyball_Video.mp4"
cap = cv2.VideoCapture(video_path) # 0 instead of filename for webcam use
if not cap.isOpened():
    print("Error: Could not open input.")
    exit()

# FPS tracking
frame_count = 0
fps_list = []
start_time = time.time() # track global runtime

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_shape = frame.shape

    start_inference_time = time.time() # track inference runtime

    # Detect people
    input_person = preprocess_yolo_input(frame_rgb)
    person_outs = session_person.run([output_name_person], {input_name_person: input_person})
    person_boxes, person_scores, person_class_ids = postprocess_yolo_output(person_outs[0], original_shape, conf_threshold=0.5)

    # Detect volleyballs
    input_ball = preprocess_yolo_input(frame_rgb)
    ball_outs = session_ball.run([output_name_ball], {input_name_ball: input_ball})
    ball_boxes, ball_scores, _ = postprocess_yolo_output(ball_outs[0], original_shape, conf_threshold=0.5)

    # Draw person detections
    for box, score, cls in zip(person_boxes, person_scores, person_class_ids):
        if cls == 0:  # "Person" class in COCO
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Draw volleyball detections
    for box, score in zip(ball_boxes, ball_scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Volleyball {score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

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

    cv2.imshow("Custom ONNX Detection", frame)
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
