"""
P2PNet Video Test Script
Based on working implementation from amindehnavi/Crowd-Counting-P2PNet

This script processes a video and displays crowd counting with grid density.
Press 'q' to quit.
"""

import os
import sys
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from timeit import default_timer as timer

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from p2pnet import build_model

# Configuration
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'weights', 'SHTechA.pth')
DEFAULT_VIDEO = os.path.join(os.path.dirname(__file__), '..', 'public', 'datasets', 'cam1.mp4')

# Grid configuration
GRID_ROWS = 4
GRID_COLS = 4

# Real-world dimensions (meters) - adjust for your venue
REAL_WIDTH_M = 20.0
REAL_HEIGHT_M = 15.0

# Confidence threshold - try lower value first
CONFIDENCE_THRESHOLD = 0.3

# Preprocessing transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def download_weights():
    """Download P2PNet weights if not present."""
    if os.path.exists(WEIGHTS_PATH):
        print(f"Weights found: {WEIGHTS_PATH}")
        return True
    
    print("Downloading P2PNet weights...")
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    
    url = "https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/raw/main/weights/SHTechA.pth"
    
    try:
        import urllib.request
        urllib.request.urlretrieve(url, WEIGHTS_PATH)
        print(f"Downloaded to: {WEIGHTS_PATH}")
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nPlease download manually from:")
        print("https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet/tree/main/weights")
        return False


def load_model(device):
    """Load the P2PNet model with weights."""
    print("Loading P2PNet model...")
    
    # Build model without pretrained backbone (we'll load full weights)
    model = build_model(row=2, line=2, pretrained_backbone=False)
    
    # Load checkpoint
    print(f"Loading weights from: {WEIGHTS_PATH}")
    checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=False)
    
    # Check what's in the checkpoint
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Try to load weights
    try:
        model.load_state_dict(state_dict, strict=True)
        print("Weights loaded successfully (strict mode)")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        print("Trying with strict=False...")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {len(missing)}")
        print(f"Unexpected keys: {len(unexpected)}")
        if missing:
            print(f"First 5 missing: {missing[:5]}")
        if unexpected:
            print(f"First 5 unexpected: {unexpected[:5]}")
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded on: {device}")
    return model


def process_video(video_path, model, device):
    """Process video with P2PNet crowd counting."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo: {video_path}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    
    # Display size - scale up if video is small
    display_width = max(frame_width, 960)
    display_height = int(display_width * frame_height / frame_width)
    
    # P2PNet input size
    INPUT_W, INPUT_H = 640, 480
    
    # Calculate grid cell dimensions (on display size)
    cell_width = display_width // GRID_COLS
    cell_height = display_height // GRID_ROWS
    cell_area_m2 = (REAL_WIDTH_M / GRID_COLS) * (REAL_HEIGHT_M / GRID_ROWS)
    
    print(f"Display size: {display_width}x{display_height}")
    print(f"Grid: {GRID_ROWS}x{GRID_COLS}, Cell area: {cell_area_m2:.2f} m²")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print("\nProcessing... Press 'q' to quit\n")
    
    # Create resizable window
    cv2.namedWindow('P2PNet Crowd Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('P2PNet Crowd Detection', display_width, display_height)
    
    # Tracking variables
    global_max_density = 0.0
    global_max_count = 0
    max_frame = 0
    frame_number = 0
    fps_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_number += 1
        start_time = timer()
        
        # Preprocess: BGR -> RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to P2PNet input size
        img_resized = cv2.resize(img_rgb, (INPUT_W, INPUT_H))
        
        # Transform to tensor
        img_tensor = transform(img_resized).unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(img_tensor)
        
        # Get predictions
        pred_logits = outputs['pred_logits'][0]  # (N, 2)
        pred_points = outputs['pred_points'][0]  # (N, 2)
        
        # Apply softmax to get probabilities for person class (class 1)
        probs = torch.softmax(pred_logits, dim=-1)
        scores = probs[:, 1]  # Person class probability
        
        # Debug: print score statistics on first frame
        if frame_number == 1:
            print(f"Debug - Score stats: min={scores.min():.4f}, max={scores.max():.4f}, "
                  f"mean={scores.mean():.4f}, >0.5: {(scores > 0.5).sum()}, >0.3: {(scores > 0.3).sum()}")
        
        # Filter by confidence threshold
        mask = scores > CONFIDENCE_THRESHOLD
        valid_points = pred_points[mask].cpu().numpy()
        valid_scores = scores[mask].cpu().numpy()
        
        # Resize frame to display size
        display_frame = cv2.resize(frame, (display_width, display_height))
        
        # Scale factors from P2PNet input to display
        scale_x = display_width / INPUT_W
        scale_y = display_height / INPUT_H
        
        # Count people per grid cell
        grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
        
        for point in valid_points:
            # Points are in INPUT coordinates (640x480)
            x = int(point[0] * scale_x)
            y = int(point[1] * scale_y)
            
            # Clamp to display bounds
            x = max(0, min(x, display_width - 1))
            y = max(0, min(y, display_height - 1))
            
            col = min(x // cell_width, GRID_COLS - 1)
            row = min(y // cell_height, GRID_ROWS - 1)
            grid[row][col] += 1
            
            # Draw detection point (green dot)
            cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
        
        # Calculate densities
        density_grid = grid / cell_area_m2
        max_density_this_frame = np.max(density_grid)
        max_count_this_frame = np.max(grid)
        max_cell = np.unravel_index(np.argmax(grid), grid.shape) if grid.sum() > 0 else (0, 0)
        
        # Update global maximum
        if max_density_this_frame > global_max_density:
            global_max_density = max_density_this_frame
            global_max_count = max_count_this_frame
            max_frame = frame_number
        
        # Draw grid lines
        for i in range(1, GRID_ROWS):
            cv2.line(display_frame, (0, i * cell_height), (display_width, i * cell_height), (100, 100, 100), 1)
        for j in range(1, GRID_COLS):
            cv2.line(display_frame, (j * cell_width, 0), (j * cell_width, display_height), (100, 100, 100), 1)
        
        # Highlight densest cell with red rectangle
        if max_count_this_frame > 0:
            r, c = max_cell
            x1, y1 = c * cell_width, r * cell_height
            x2, y2 = x1 + cell_width, y1 + cell_height
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Calculate FPS
        elapsed = timer() - start_time
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        fps_list.append(current_fps)
        
        # Draw text overlay (larger for visibility)
        total_people = len(valid_points)
        cv2.putText(display_frame, f'Total People: {total_people}', (10, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(display_frame, f'Hotspot: {max_density_this_frame:.2f} p/m2 ({max_count_this_frame})', (10, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(display_frame, f'Global Max: {global_max_density:.2f} p/m2', (10, 105),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 0, 255), 2)
        cv2.putText(display_frame, f'FPS: {current_fps:.1f}', (10, 140),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 0), 2)
        cv2.putText(display_frame, f'Frame: {frame_number}/{total_frames}', (10, 175),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('P2PNet Crowd Detection', display_frame)
        
        # Print progress every 30 frames
        if frame_number % 30 == 0:
            print(f"Frame {frame_number}/{total_frames} - People: {total_people}, "
                  f"Hotspot: {max_density_this_frame:.2f} p/m², FPS: {current_fps:.1f}")
        
        # Check for quit key (wait 1ms)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nQuitting...")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    avg_fps = np.mean(fps_list) if fps_list else 0
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Frames processed: {frame_number}")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"\nMAXIMUM DENSITY DETECTED:")
    print(f"  Density: {global_max_density:.2f} people/m²")
    print(f"  People in hotspot: {global_max_count}")
    print(f"  Occurred at frame: {max_frame}")
    print("=" * 60)


def main():
    print("=" * 60)
    print("P2PNet Crowd Counting Test")
    print("=" * 60)
    
    # Check for weights
    if not download_weights():
        return
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU (GPU not available)")
    
    # Load model
    try:
        model = load_model(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Find video
    video_path = None
    
    # Check command line argument
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    # Check default locations
    if not video_path or not os.path.exists(video_path):
        search_paths = [
            DEFAULT_VIDEO,
            os.path.join(os.path.dirname(__file__), '..', 'public', 'datasets', 'cam1.mp4'),
            os.path.join(os.path.dirname(__file__), 'cam1.mp4'),
            os.path.join(os.path.dirname(__file__), 'test_video.mp4'),
        ]
        for path in search_paths:
            if os.path.exists(path):
                video_path = path
                break
    
    if not video_path or not os.path.exists(video_path):
        print("\nNo video found. Please provide a video path:")
        print(f"  python test_p2pnet.py <video_path>")
        print("\nOr place a video in one of these locations:")
        print(f"  {DEFAULT_VIDEO}")
        return
    
    # Process video
    process_video(video_path, model, device)


if __name__ == '__main__':
    main()
