from ultralytics import YOLO

data_yaml = "./val_dataset/data.yaml"

# Model Loading. Options: "models/yolo11n.pt", "models/yolov8n.pt"
model = YOLO("models/yolo11n.pt")

# Run Validation on Volleyball dataset, only for class "Person"
results = model.val(data=data_yaml, split="val", imgsz=640, classes=[0])

# Extract metrics
precision = results.results_dict["metrics/precision(B)"]
recall = results.results_dict["metrics/recall(B)"]
map50 = results.results_dict["metrics/mAP50(B)"]
map5095 = results.results_dict["metrics/mAP50-95(B)"]

print("\n=== YOLO11-nano Validation Results on Volleyball ===")
print(f"Precision:      {precision:.3f}")
print(f"Recall:         {recall:.3f}")
print(f"mAP@0.5:        {map50:.3f}")
print(f"mAP@0.5:0.95:   {map5095:.3f}")