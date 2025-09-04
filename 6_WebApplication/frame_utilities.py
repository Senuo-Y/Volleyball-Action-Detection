import numpy as np
import cv2

def get_distance_person_ball_np(person_box_np, ball_box_np):
    person_center_x = (person_box_np[0] + person_box_np[2]) / 2
    person_center_y = (person_box_np[1] + person_box_np[3]) / 2
    ball_center_x = (ball_box_np[0] + ball_box_np[2]) / 2
    ball_center_y = (ball_box_np[1] + ball_box_np[3]) / 2
    return np.sqrt((person_center_x - ball_center_x) ** 2 + (person_center_y - ball_center_y) ** 2)

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