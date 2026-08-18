import os
import sys

# Connect support folder to python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from support.detection import load_tracking_model, process_video_tracking
from support.utils import save_telemetry_csv, print_project_summary

def main():
    print("\n========================================================")
    print(" CSE445: Multi-Domain Real-Time Object Tracker Engine ")
    print("========================================================\n")

    # Define paths according to required repo structure
    root_dir = os.path.dirname(os.path.abspath(__file__))
    input_video = os.path.join(root_dir, "data", "sample_video.mp4")
    output_video = os.path.join(root_dir, "data", "output_tracked.avi")
    output_csv = os.path.join(root_dir, "data", "traffic_analytics.csv")
    
    # Check if fine-tuned weights exist in support/, otherwise fallback to yolov8n.pt
    custom_weights = os.path.join(root_dir, "support", "best.pt")
    model_weights = custom_weights if os.path.exists(custom_weights) else "yolov8n.pt"

    # Input validation
    if not os.path.exists(input_video):
        print(f"Warning: Input video not found at '{input_video}'.")
        print("Please place a sample video file inside 'data/' named 'sample_video.mp4'.")
        return

    # 1. Load Model Architecture
    model = load_tracking_model(model_weights)

    # 2. Run Tracking & Extract Telemetry
    df_telemetry = process_video_tracking(
        model=model,
        video_path=input_video,
        output_video_path=output_video,
        tracker_type="bytetrack.yaml",
        conf_thresh=0.25,
        iou_thresh=0.5
    )

    # 3. Save Normalized Telemetry CSV
    save_telemetry_csv(df_telemetry, output_csv)

    # 4. Display Execution Summary
    print_project_summary(df_telemetry)

    print("Project execution finished successfully!")

if __name__ == "__main__":
    main()