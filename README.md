# cse445-group07-object-tracking
CSE445 Group 07 project: Develop a machine learning model to detect and track moving objects in real time using videos collected from online sources.

##TRACKED  VIDEOS  : link - https://drive.google.com/file/d/1DRJLhJYYd7Ji883IwJIJ98UFBXvLRSu6/view?usp=sharing

saved videos: https://drive.google.com/drive/folders/1Q-iCJkx8oJXgLWACZ8jltS_BczDSbDts?usp=sharing
   
##models saved at HUGGING FACE : https://huggingface.co/RaihanGG2026/cse445-hardhat-tracker/tree/main 
    https://huggingface.co/RaihanGG2026/yolov8-drone-footage

# 🚗 SmartVision: Real-Time Traffic Object Tracking & Analytics Control Center

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?logo=opencv&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Model%20Registry-Hugging%20Face-yellow?logo=huggingface&logoColor=white)

An enterprise-grade, data-driven computer vision application developed for CSE445 (Machine Learning). This project implements an end-to-end MLOps pipeline—from custom dataset tuning and video object tracking to a unified interactive web dashboard complete with real-time analytics and an AI project assistant.


# Live Demonstration & Portfolio
You can view our core processed traffic tracking video directly via cloud streaming:
* **[Watch Traffic Tracking Demo (Google Drive)](https://drive.google.com/file/d/1DRJLhJYYd7Ji883IwJIJ98UFBXvLRSu6/view?usp=sharing)**
  * *Description: High-density mixed traffic analysis utilizing YOLOv8 and ByteTrack, demonstrating persistent ID assignment across 330 verified traffic units.*

---

### 📊 Project Highlights & Verified Metrics
FROM 1st video: 
Total Tracked Traffic Units:** 330 verified entities.
* Vehicle Breakdown: Cars: 185
  * Trucks: 93
  * Motorcycles: 63
  * Buses: 41
  Inference Efficiency:Operating at 8.6ms per frame (~115 FPS) on GPU acceleration, proving real-time feasibility.


### 🛠️ Technical Architecture & Stack
* **Core ML Engine:** Ultralytics YOLOv8 (Pre-trained COCO weights + fine-tuned custom weights for safety gear).
* **Tracking Algorithm:** ByteTrack (persistent multi-object association across frames).
* **Model Registry:** Hosted centrally on **Hugging Face** (`RaihanGG2026/cse445-hardhat-tracker`) for cloud-based weight fetching.
* **Data Pipeline:** OpenCV for frame manipulation, structured into enterprise-ready `traffic_analytics.csv` files.
* **Frontend Dashboard:** Built using Next.Js and **Plotly Express** for dynamic data visualization and AI chatbot auditing.


### 👥 Team Contributions & Roles
#Raihan Mahmud-2222642642— Data & ML Lead: Managed dataset sourcing (Roboflow & Pexels), core YOLOv8 model training (`best.pt`), video inference pipeline, and Hugging Face MLOps model registry deployment.
#MD Emran Hossain — Frontend UI Lead: Designed and structured the core layout, Frontend, theme styling, Plotly visual charts, and sidebar repository integrations.



### 🚀 Getting Started Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/RaihanMD992/cse445-group07-object-tracking.git](https://github.com/RaihanMD992/cse445-group07-object-tracking.git)
   cd cse445-group07-object-tracking
