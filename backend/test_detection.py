"""
PlexIE Detection Test
=====================
Based on Mohamed's working YOLO implementation.

This script:
1. Shows video with detection LIVE in a window
2. Saves output to output_detected.avi

Usage:
    python test_detection.py

Press 'q' to quit early.
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os

# ============================================
# CONFIGURATION - EDIT THESE IF NEEDED
# ============================================

# Find video - looks in same folder first, then public/datasets
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Try to find a video
VIDEO_PATH = None
search_paths = [
    os.path.join(SCRIPT_DIR, "test.mp4"),
    os.path.join(SCRIPT_DIR, "cam1.mp4"),
    os.path.join(PROJECT_DIR, "public", "datasets", "cam1.mp4"),
    os.path.join(PROJECT_DIR, "public", "datasets", "cam2.mp4"),
    os.path.join(PROJECT_DIR, "public", "datasets", "test.mp4"),
]

for p in search_paths:
    if os.path.exists(p):
        VIDEO_PATH = p
        break

if VIDEO_PATH is None:
    print("=" * 60)
    print("ERROR: No video file found!")
    print("=" * 60)
    print("\nPlease put a video file in one of these locations:")
    print(f"  - {SCRIPT_DIR}/test.mp4")
    print(f"  - {PROJECT_DIR}/public/datasets/cam1.mp4")
    print("\nOr edit VIDEO_PATH in this script.")
    exit(1)

# Output file (AVI works better than MP4 for OpenCV)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "output_detected.avi")

# Model - yolov8n is small/fast, yolov8m is more accurate
MODEL_NAME = "yolov8n.pt"
CONFIDENCE = 0.3

# Grid for density calculation
GRID_ROWS = 4
GRID_COLS = 4

# Real-world area (estimate)
REAL_WIDTH_METERS = 10.0
REAL_HEIGHT_METERS = 8.0

# ============================================
# MAIN SCRIPT
# ============================================

print("=" * 60)
print("PlexIE Detection Test")
print("=" * 60)
print(f"Video: {VIDEO_PATH}")
print(f"Output: {OUTPUT_PATH}")
print(f"Model: {MODEL_NAME}")
print("=" * 60)

# Load model
print("\nLoading YOLO model (may download on first run)...")
model = YOLO(MODEL_NAME)
print("Model loaded!")

# Open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Could not open video: {VIDEO_PATH}")
    exit(1)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo: {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames")

# Grid cell dimensions
cell_width = frame_width // GRID_COLS
cell_height = frame_height // GRID_ROWS
cell_area_m2 = (REAL_WIDTH_METERS / GRID_COLS) * (REAL_HEIGHT_METERS / GRID_ROWS)

print(f"Grid: {GRID_ROWS}x{GRID_COLS}, each cell = {cell_area_m2:.2f} m²")

# Setup output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (frame_width, frame_height))

# Make window resizable
cv2.namedWindow('PlexIE Detection Test', cv2.WINDOW_NORMAL)
cv2.resizeWindow('PlexIE Detection Test', 1280, 720)

# Track statistics
global_max_density = 0
global_max_count_in_cell = 0
global_max_frame = 0
frame_number = 0

print(f"\nProcessing... Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video finished!")
        break
    
    frame_number += 1
    
    # Run YOLO detection
    results = model(frame, classes=[0], verbose=False, conf=CONFIDENCE)
    
    # Get boxes and draw manually (no labels)
    boxes = results[0].boxes
    annotated_frame = frame.copy()
    
    # Initialize grid
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    
    # Process detections
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw GREEN box (no label)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Calculate center for grid
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Which grid cell?
        col = min(int(center_x // cell_width), GRID_COLS - 1)
        row = min(int(center_y // cell_height), GRID_ROWS - 1)
        grid[row][col] += 1
    
    # Find hotspot (densest cell)
    max_density_this_frame = 0
    max_cell_count = 0
    max_cell_location = (0, 0)
    
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            people_in_cell = grid[row][col]
            density = people_in_cell / cell_area_m2
            
            if density > max_density_this_frame:
                max_density_this_frame = density
                max_cell_count = people_in_cell
                max_cell_location = (row, col)
    
    # Update global max
    if max_density_this_frame > global_max_density:
        global_max_density = max_density_this_frame
        global_max_count_in_cell = max_cell_count
        global_max_frame = frame_number
    
    # Draw grid lines (gray)
    for i in range(1, GRID_ROWS):
        cv2.line(annotated_frame, (0, i * cell_height), 
                (frame_width, i * cell_height), (100, 100, 100), 1)
    for j in range(1, GRID_COLS):
        cv2.line(annotated_frame, (j * cell_width, 0), 
                (j * cell_width, frame_height), (100, 100, 100), 1)
    
    # Highlight hotspot with RED rectangle
    if max_cell_count > 0:
        row, col = max_cell_location
        x1 = col * cell_width
        y1 = row * cell_height
        x2 = x1 + cell_width
        y2 = y1 + cell_height
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    
    # Add text overlay
    total_people = len(boxes)
    cv2.putText(annotated_frame, f'Total People: {total_people}', 
                (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(annotated_frame, f'Hotspot Density: {max_density_this_frame:.2f} p/m2', 
                (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated_frame, f'People in Hotspot: {max_cell_count}', 
                (10, 75), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated_frame, f'Global Max: {global_max_density:.2f} p/m2', 
                (10, 100), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
    
    # Show frame LIVE
    cv2.imshow('PlexIE Detection Test', annotated_frame)
    
    # Write to output file
    out.write(annotated_frame)
    
    # Progress every 30 frames
    if frame_number % 30 == 0:
        pct = (frame_number / total_frames) * 100
        print(f"  Frame {frame_number}/{total_frames} ({pct:.0f}%) - {total_people} people")
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quit by user")
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Final report
print("\n" + "=" * 60)
print("COMPLETE!")
print("=" * 60)
print(f"Output saved: {OUTPUT_PATH}")
print(f"\nStatistics:")
print(f"  Max density: {global_max_density:.2f} people/m²")
print(f"  People in hotspot: {global_max_count_in_cell}")
print(f"  At frame: {global_max_frame}")
print(f"  Total frames: {frame_number}")
print("=" * 60)
