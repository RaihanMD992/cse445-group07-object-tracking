# cse445-group07-object-tracking
CSE445 Group 07 project: Develop a machine learning model to detect and track moving objects in real time using videos collected from online sources.


#  SmartVision: Real-Time Object Tracking & Analytics Control Center

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange?logo=opencv&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Model%20Registry-Hugging%20Face-yellow?logo=huggingface&logoColor=white)

This project implements an end-to-end MLOps pipeline—from custom dataset tuning and video object tracking to a unified interactive web dashboard complete with real-time analytics and an AI project assistant.

SmartVision: Real-Time Multi-Domain Object Tracking & Telemetry Analytics Engine
SmartVision is an end-to-end computer vision and telemetry analytics platform that pairs high-speed single-stage object detection (YOLOv8) with multi-object association (ByteTrack). The platform ingests raw multi-domain video streams (urban traffic, drone feeds, sports clips), performs real-time frame-by-frame spatial inference and tracking, re-encodes annotated video via server-side FFmpeg for web streaming, and extracts structured time-series telemetry for downstream AI analysis.

Key System Features:
Multi-Domain Detection Engine: Utilizes fine-tuned YOLOv8 CNN backbones to detect multi-scale targets (vehicles, pedestrians, players) with anchor-free spatial heads.

Occlusion-Resilient Motion Tracking: Integrated ByteTrack association algorithm leveraging Kalman Filter state prediction and Hungarian bipartite matching to retain object identities during severe occlusions.

Server-Side FFmpeg Transcoding Pipeline: Converts processed video frames into web-compatible H.264 (AVC) video containers (yuv420p pixel format) to prevent browser streaming errors.

Structured Telemetry Data Extraction: Exports object frame data, spatial coordinates, velocity proxies, confidence scores, and bounding box dimensions directly into standardized CSV and JSON schemas (traffic_analytics.csv).

Natural Language Telemetry Assistant (Gemini LLM): An integrated LLM agent that allows operators to query numerical telemetry data using natural language (e.g., "How many heavy trucks passed between frames 100 and 300?").

Full-Stack Dashboard: Built with Next.js (frontend) and FastAPI (backend) for real-time KPI metrics, streaming video player controls, and dynamic telemetry chart filters.

Our project directly maps core concepts from the CSE445 Machine Learning curriculum into a production scaled system :
**1.Supervised Learning & Dataset Pipeline:**
Framework: Frame inputs ($X$) map to annotated targets ($Y$) representing multi-class categories and spatial bounding box bounds.
Data Splits & Optimization: Datasets are structured into Training (weight updates), Validation (hyperparameter tuning), and Test sets (out-of-sample evaluation). Data augmentations (Mosaic, MixUp) mitigate overfitting and lower model variance.
**2.Multi-Class Classification vs. Continuous Bounding Box Regression:**
Classification Head: Predicts discrete probability distributions over classes (car, bus, truck, person, player) using cross-entropy evaluation.
Regression Head: Predicts continuous bounding box bounds $[x_{\text{center}}, y_{\text{center}}, \text{width}, \text{height}]$ using Complete IoU (CIoU) Loss and Distribution Focal Loss (DFL).
**3. Deep Learning, Backpropagation & Feature Scaling:**
Architecture: Uses CSPDarknet backbone feature extraction and Path Aggregation Network (PANet) neck structures for multi-scale feature maps.
Preprocessing: Input RGB frames ($0\text{--}255$) undergo intensity normalization to $[0, 1]$ to stabilize backpropagation gradients during optimization.

# Live Demonstration & Portfolio
You can view our core processed traffic tracking video directly via cloud streaming:
* **[Watch Traffic Tracking Demo (Google Drive)](https://drive.google.com/file/d/1DRJLhJYYd7Ji883IwJIJ98UFBXvLRSu6/view?usp=sharing)**
  * *Description: High-density mixed traffic analysis utilizing YOLOv8 and ByteTrack, demonstrating persistent ID assignment across 330 verified traffic units.*

---

### Project Highlights & Verified Metrics
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


###  Team Contributions & Roles
#Raihan Mahmud-2222642642— Data & ML Lead: Managed dataset sourcing (Roboflow & Pexels), core YOLOv8 model training (`best.pt`), video inference pipeline, and Hugging Face MLOps model registry deployment.

#MD Emran Hossain — Frontend UI Lead: Designed and structured the core layout, Frontend, theme styling, Plotly visual charts, and sidebar repository integrations.


##TRACKED  VIDEOS  : link - https://drive.google.com/file/d/1DRJLhJYYd7Ji883IwJIJ98UFBXvLRSu6/view?usp=sharing

Saved best.pt yolov8 model: https://drive.google.com/file/d/1TI7VH8rlAJEGj4ARQqqtSOnlTpry1NBM/view?usp=sharing

saved videos: https://drive.google.com/drive/folders/1Q-iCJkx8oJXgLWACZ8jltS_BczDSbDts?usp=sharing
   
##models saved at HUGGING FACE : https://huggingface.co/RaihanGG2026/cse445-hardhat-tracker/tree/main 
    https://huggingface.co/RaihanGG2026/yolov8-drone-footage

