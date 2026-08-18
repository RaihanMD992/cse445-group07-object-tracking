import os
import pandas as pd

def save_telemetry_csv(df_telemetry, csv_output_path):
    """Saves telemetry dataframe to CSV format."""
    os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
    df_telemetry.to_csv(csv_output_path, index=False)
    print(f"Telemetry analytics exported to: {csv_output_path}")

def print_project_summary(df_telemetry):
    """Prints execution summary metrics to standard output."""
    print("\n" + "="*60)
    print(" TRACKING ANALYTICS SUMMARY")
    print("="*60)
    if not df_telemetry.empty:
        print(f" Total Frame Logs:      {len(df_telemetry)}")
        print(f" Unique Objects Tracked: {df_telemetry['object_id'].nunique()}")
        print(f" Classes Detected:       {', '.join(df_telemetry['class_name'].unique().tolist())}")
    else:
        print("No objects were tracked in this sequence.")
    print("="*60 + "\n")