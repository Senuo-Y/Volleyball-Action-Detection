from ultralytics import YOLO

yolo_person_model = YOLO("yolov11n.pt")
yolo_vb_model = YOLO("yolov11n_vb.pt")

yolo_person_model.export(format="openvino", half=True)
yolo_vb_model.export(format="openvino", half=True)