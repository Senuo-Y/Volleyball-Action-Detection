# Real-Time Volleyball Action Detection Using Machine Learning

This project presents a system for **real-time volleyball action detection** using **computer vision** and **machine learning** techniques. The goal is to automatically detect and classify key volleyball actions such as **setting, bumping, and attacking** from video footage with **low latency** and **high accuracy**, making it suitable for practical use during training sessions and games.  

---

## ✨ Features
- **Person detection** using YOLO11 and YOLOv8 models  
- **Pose estimation** with MediaPipe for detailed player movement analysis  
- **Volleyball detection** via a custom YOLO11 model trained on curated datasets  
- **Action classification** (Set, Bump, Attack, None) using classical ML models  
- **Model optimization** with ONNX and OpenVINO for real-time performance  
- **Interactive web application** built with Streamlit to process video and image inputs  

---

## 📊 Datasets
- **Player detection**: COCO dataset (person class)  
- **Volleyball detection**: Custom dataset (6 merged Roboflow datasets, 6207 images)  
- **Action detection**: Custom dataset recorded and annotated with volleyball actions  

---

## 🛠️ Tech Stack
- **Programming Language**: Python
- **Models**: YOLO11, YOLOv8, YOLO-NAS, RT-DETR
- **Pose Estimation**: MediaPipe
- **Deployment**: PyTorch, ONNX, OpenVINO
- **Interface**: Streamlit
- **Other Tools**: Roboflow Universe, Git, DroidCam, Figma

---
