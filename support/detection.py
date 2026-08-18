import os
import torch
import cv2
import pandas as pd
from ultralytics import YOLO

def load_tracking_model(model_path="yolov8n.pt"):
    """Loads YOLOv8 architecture or fine-tuned weights."""
    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    return model

def process_video_tracking(model, video_path, output_video_path, tracker_type="bytetrack.yaml", conf_thresh=0.25, iou_thresh=0.5):
    """
    Runs persistent multi-object tracking on an input video using ByteTrack,
    renders tracked bounding boxes, and records telemetry data.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or torch.isnan(torch.tensor(fps)):
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize output video stream
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    telemetry_data = []
    unique_ids = set()
    frame_idx = 0

    print(f"Processing tracking on: {os.path.basename(video_path)} ({total_frames} frames)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        
        # Run YOLO + ByteTrack
        results = model.track(
            source=frame,
            persist=True,
            tracker=tracker_type,
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False
        )

        annotated_frame = frame.copy()

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
                cls_ids = boxes.cls.int().cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                xyxys = boxes.xyxy.cpu().tolist()

                for track_id, cls_id, conf, bbox in zip(track_ids, cls_ids, confs, xyxys):
                    unique_ids.add(track_id)
                    x1, y1, x2, y2 = bbox
                    class_name = model.names[cls_id]

                    # Extract Telemetry Record
                    telemetry_data.append({
                        "frame": frame_idx,
                        "timestamp_sec": round(frame_idx / fps, 2),
                        "object_id": int(track_id),
                        "class_id": int(cls_id),
                        "class_name": class_name,
                        "confidence": round(float(conf), 4),
                        "bbox_x1": round(float(x1), 2),
                        "bbox_y1": round(float(y1), 2),
                        "bbox_x2": round(float(x2), 2),
                        "bbox_y2": round(float(y2), 2)
                    })

                    # Draw Bounding Box and Track ID Label
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    label = f"ID:{track_id} {class_name} {conf:.2f}"
                    cv2.putText(annotated_frame, label, (int(x1), max(20, int(y1) - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(annotated_frame)

        if frame_idx % 50 == 0 or frame_idx == total_frames:
            print(f"  ➜ Frame {frame_idx}/{total_frames} | Unique Tracks: {len(unique_ids)}")

    cap.release()
    out.release()

    print(f"Video processing finished. Rendered output saved to: {output_video_path}")
    return pd.DataFrame(telemetry_data)