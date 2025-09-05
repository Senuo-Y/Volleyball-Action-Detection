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

## 🚀 Running the Application

The application can be executed by any user who wishes to test the available detection types.

### 1. Install Dependencies
Before running the app, install the required libraries (OpenCV, Mediapipe, Ultralytics, etc.):
```bash
pip install -r requirements.txt
```

### 2. Navigate to the Application Folder
Go to the folder containing the app and required files:
```bash
cd 6_WebApplication
```

### 3. Launch the Streamlit Interface
Start the application:
```bash
streamlit run demo.py
```
This command initializes the Streamlit server and should automatically open the application in your default web browser.
If the browser does not open automatically, copy and paste the local URL displayed in the terminal (typically http://localhost:8501).

### 4. Using the Interface
Once the application is launched, the interface allows you to:
- **Select a detection option**: choose between person detection, volleyball detection, or action detection.  
- **Upload media**: upload images or videos to test the models.  
- **View results**: see annotated frames in real time, including bounding boxes and predicted actions.  
- **Download output**: save the processed images or videos for further analysis.

### 5. Stopping the Application
To stop the application, return to the terminal where Streamlit is running and press `Ctrl + C`.

---