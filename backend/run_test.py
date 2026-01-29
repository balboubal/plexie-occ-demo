"""
PlexIE P2PNet Crowd Counting - Optimized for Live Preview
==========================================================
Features:
- Frame skipping (process every Nth frame)
- Resolution limiting for faster inference
- Auto GPU detection (NVIDIA CUDA / AMD DirectML / CPU fallback)
- 3-second startup delay for window positioning
- Live preview with density hotspot detection
"""

import argparse
import datetime
import random
from timeit import default_timer
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as standard_transforms
import numpy as np
from scipy.ndimage import filters
from scipy.stats import gaussian_kde
import scipy
from PIL import Image
import cv2
# from crowd_datasets import build_dataset  # Not needed for video mode
from engine import *
from models import build_model
import os
import warnings
import time
from timeit import default_timer
warnings.filterwarnings('ignore')


# ============================================
# GPU DETECTION AND SETUP
# ============================================
def setup_device(gpu_id=0):
    """Auto-detect best available device: CUDA > DirectML > CPU"""
    
    # Try NVIDIA CUDA first
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"[GPU] NVIDIA CUDA detected: {gpu_name}")
        return device, "cuda"
    
    # Try AMD DirectML (Windows)
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"[GPU] AMD DirectML detected")
        return device, "directml"
    except ImportError:
        pass
    except Exception as e:
        print(f"[GPU] DirectML error: {e}")
    
    # Fallback to CPU
    print("[CPU] No GPU acceleration available, using CPU")
    print("      For AMD GPUs, install: pip install torch-directml")
    return torch.device('cpu'), "cpu"


# ============================================
# VIDEO OUTPUT CREATION
# ============================================
def make_video(args, output_frames):
    """Create video from processed frames"""
    if len(output_frames) == 0:
        print("No frames to create video from")
        return
        
    # Get original video FPS
    cap = cv2.VideoCapture(args.video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # Use reasonable FPS for output
    target_fps = min(30, max(10, int(original_fps / args.frame_skip)))
    
    print(f'\n[Video] Creating output at {target_fps} FPS from {len(output_frames)} frames...')
    
    # Get frame size from first frame
    h, w = output_frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_path = os.path.join(args.output_dir, 'output.avi')
    video = cv2.VideoWriter(video_path, fourcc, target_fps, (w, h))
    
    for frame in output_frames:
        video.write(frame)
    
    video.release()
    print(f'[Video] Saved: {video_path}')


# ============================================
# DENSITY MAP GENERATION
# ============================================
def gaussian_filter_density(gt):
    """Generate gaussian density map from point annotations"""
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 100
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.
        if gt_count > 3:
            distances, locations = tree.query(pts, k=4)
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        elif gt_count == 3:
            distances, locations = tree.query(pts, k=3)
            sigma = (distances[i][1]+distances[i][2])*0.1
        elif gt_count == 2:
            distances, locations = tree.query(pts, k=2)
            sigma = (distances[i][1])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2.
        density += filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density


# ============================================
# ARGUMENT PARSER
# ============================================
def get_args_parser():
    parser = argparse.ArgumentParser(
        'P2PNet Optimized Evaluation', add_help=False)

    # Model settings
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Backbone: vgg16_bn or vgg16")
    parser.add_argument('--row', default=2, type=int,
                        help="Row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="Line number of anchor points")
    
    # Paths
    parser.add_argument('--output_dir', default='./output',
                        help='Output directory for frames and video')
    parser.add_argument('--weight_path', default='./weights/SHTechA.pth',
                        help='Path to P2PNet weights')
    parser.add_argument('--density_path', default='./density',
                        help='Density map output directory')
    
    # Device settings
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='GPU ID to use')
    parser.add_argument('--device', default='auto', type=str,
                        help='Device: auto, cuda, directml, or cpu')
    
    # Detection settings
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='Detection confidence threshold')
    parser.add_argument('--shape', default=[640, 480], nargs='+', type=int,
                        help='Input shape [width, height]')
    
    # Optimization settings
    parser.add_argument('--frame_skip', default=3, type=int,
                        help='Process every Nth frame (1=all, 3=every 3rd)')
    parser.add_argument('--max_resolution', default=480, type=int,
                        help='Max resolution (longer edge). 0=no limit')
    parser.add_argument('--startup_delay', default=3, type=int,
                        help='Seconds to wait before processing (for window positioning)')
    
    # Mode settings
    parser.add_argument('--video', action='store_true',
                        help='Video mode')
    parser.add_argument('--video_path', default='', 
                        help='Path to video file')
    parser.add_argument('--images', action='store_true',
                        help='Images mode')
    parser.add_argument('--images_dir', default='./Dataset',
                        help='Path to images directory')
    parser.add_argument('--density_map', action='store_true',
                        help='Generate density maps')
    parser.add_argument('--add_density_to_img', action='store_true',
                        help='Overlay density map on image')
    parser.add_argument('--save_frames', action='store_true', default=True,
                        help='Save individual frames')

    return parser


# ============================================
# MAIN PROCESSING
# ============================================
def main(args):
    print("\n" + "="*60)
    print("PlexIE P2PNet Crowd Counting - Optimized")
    print("="*60)
    
    # ===== SETUP DEVICE =====
    if args.device == 'auto':
        device, device_type = setup_device(args.gpu_id)
    elif args.device == 'cuda':
        device = torch.device(f'cuda:{args.gpu_id}')
        device_type = 'cuda'
    elif args.device == 'directml':
        import torch_directml
        device = torch_directml.device()
        device_type = 'directml'
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
    
    print(f"[Device] Using: {device_type.upper()}")
    
    # ===== CREATE DIRECTORIES =====
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.density_path, exist_ok=True)

    # ===== BUILD MODEL =====
    print(f"[Model] Loading P2PNet with {args.backbone} backbone...")
    model = build_model(args)
    model.to(device)
    
    # Load weights
    if args.weight_path and os.path.exists(args.weight_path):
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        print(f"[Model] Weights loaded: {args.weight_path}")
    else:
        print(f"[ERROR] Weights not found: {args.weight_path}")
        return
    
    model.eval()
    
    # ===== PREPROCESSING TRANSFORM =====
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # ===== GRID CONFIG FOR DENSITY =====
    GRID_ROWS = 4
    GRID_COLS = 4
    real_width_meters = 5.0
    real_height_meters = 12.0
    cell_area_m2 = (real_width_meters / GRID_COLS) * (real_height_meters / GRID_ROWS)
    
    # ===== OPEN VIDEO =====
    if not args.video:
        print("[ERROR] Only video mode supported in this version")
        return
        
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[Video] {args.video_path}")
    print(f"        {orig_width}x{orig_height} @ {original_fps:.1f} FPS, {total_frames} frames")
    
    # ===== CALCULATE PROCESSING RESOLUTION =====
    proc_width, proc_height = args.shape[0], args.shape[1]
    
    # Apply max resolution limit
    if args.max_resolution > 0:
        max_dim = max(proc_width, proc_height)
        if max_dim > args.max_resolution:
            scale = args.max_resolution / max_dim
            proc_width = int(proc_width * scale)
            proc_height = int(proc_height * scale)
            # Ensure dimensions are divisible by 16 (for VGG)
            proc_width = (proc_width // 16) * 16
            proc_height = (proc_height // 16) * 16
    
    print(f"[Process] Resolution: {proc_width}x{proc_height}")
    print(f"[Process] Frame skip: every {args.frame_skip} frame(s)")
    print(f"[Process] Threshold: {args.threshold}")
    
    # Grid cell dimensions
    cell_width = proc_width // GRID_COLS
    cell_height = proc_height // GRID_ROWS
    
    # ===== SETUP DISPLAY WINDOW =====
    window_name = 'PlexIE P2PNet Live - Press Q to quit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    # ===== STARTUP DELAY =====
    if args.startup_delay > 0:
        print(f"\n[Ready] Starting in {args.startup_delay} seconds...")
        print("        Position the preview window now!")
        
        # Show blank frame with countdown
        for i in range(args.startup_delay, 0, -1):
            blank = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(blank, f"Starting in {i}...", (500, 350), 
                       cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            cv2.putText(blank, "Position this window now", (400, 420),
                       cv2.FONT_HERSHEY_DUPLEX, 1, (150, 150, 150), 1)
            cv2.putText(blank, "Press Q to quit", (520, 480),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (100, 100, 100), 1)
            cv2.imshow(window_name, blank)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                print("[Cancelled]")
                cv2.destroyAllWindows()
                return
    
    print("\n[Processing] Started...")
    print("-" * 60)
    
    # ===== TRACKING VARIABLES =====
    frame_count = 0
    processed_count = 0
    output_frames = []
    FPSs = []
    
    global_max_density = 0
    global_max_count_in_cell = 0
    global_max_frame = 0
    global_max_location = (0, 0)
    
    last_detections = []  # Store last detection for skipped frames
    last_count = 0
    last_max_density = 0
    last_max_cell_count = 0
    last_max_location = (0, 0)
    
    # ===== MAIN PROCESSING LOOP =====
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = default_timer()
        
        # Resize for processing
        frame_resized = cv2.resize(frame, (proc_width, proc_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # ===== PROCESS OR REUSE =====
        if frame_count % args.frame_skip == 1 or args.frame_skip == 1:
            # Actually run detection
            processed_count += 1
            
            # Preprocess
            img_tensor = transform(frame_rgb)
            samples = img_tensor.unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(samples)
            
            # Get predictions
            outputs_scores = torch.nn.functional.softmax(
                outputs['pred_logits'], -1)[:, :, 1][0]
            outputs_points = outputs['pred_points'][0]
            
            # Filter by threshold
            mask = outputs_scores > args.threshold
            points = outputs_points[mask].detach().cpu().numpy().tolist()
            predict_cnt = len(points)
            
            # Grid density calculation
            grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
            for p in points:
                x, y = int(p[0]), int(p[1])
                col = min(x // cell_width, GRID_COLS - 1)
                row = min(y // cell_height, GRID_ROWS - 1)
                grid[row][col] += 1
            
            # Find hotspot
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
                global_max_frame = frame_count
                global_max_location = max_cell_location
            
            # Store for skipped frames
            last_detections = points
            last_count = predict_cnt
            last_max_density = max_density_this_frame
            last_max_cell_count = max_cell_count
            last_max_location = max_cell_location
            
        else:
            # Reuse last detection
            points = last_detections
            predict_cnt = last_count
            max_density_this_frame = last_max_density
            max_cell_count = last_max_cell_count
            max_cell_location = last_max_location
        
        # ===== DRAW VISUALIZATION =====
        img_to_draw = frame_resized.copy()
        
        # Draw grid lines
        for i in range(1, GRID_ROWS):
            cv2.line(img_to_draw, (0, i * cell_height), 
                    (proc_width, i * cell_height), (80, 80, 80), 1)
        for j in range(1, GRID_COLS):
            cv2.line(img_to_draw, (j * cell_width, 0), 
                    (j * cell_width, proc_height), (80, 80, 80), 1)
        
        # Highlight hotspot with red rectangle
        if max_cell_count > 0:
            row, col = max_cell_location
            x1, y1 = col * cell_width, row * cell_height
            x2, y2 = x1 + cell_width, y1 + cell_height
            cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Draw detection points (green dots)
        for p in points:
            cv2.circle(img_to_draw, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
        
        # Calculate FPS
        elapsed = default_timer() - start_time
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        FPSs.append(current_fps)
        
        # Draw text overlay (with background for readability)
        overlay_texts = [
            f'Total People: {predict_cnt}',
            f'Hotspot: {max_density_this_frame:.2f} p/m2 ({max_cell_count} ppl)',
            f'Global Max: {global_max_density:.2f} p/m2',
            f'Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f}',
            f'Device: {device_type.upper()}' + (f' | Skip: {args.frame_skip}x' if args.frame_skip > 1 else ''),
        ]
        
        y_offset = 25
        for text in overlay_texts:
            # Draw background rectangle
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)
            cv2.rectangle(img_to_draw, (8, y_offset - 15), (15 + tw, y_offset + 5), (0, 0, 0), -1)
            # Draw text
            color = (0, 255, 0) if 'Total' in text else (0, 200, 255) if 'Hotspot' in text else (200, 200, 200)
            cv2.putText(img_to_draw, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1, cv2.LINE_AA)
            y_offset += 22
        
        # ===== DISPLAY =====
        display_frame = cv2.resize(img_to_draw, (1280, 720))
        cv2.imshow(window_name, display_frame)
        
        # Save frame
        if args.save_frames:
            output_frames.append(img_to_draw)
            cv2.imwrite(os.path.join(args.output_dir, f'pred_{frame_count:05d}.jpg'), img_to_draw)
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[User] Stopped early")
            break
        
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            avg_fps = np.mean(FPSs[-30:]) if len(FPSs) >= 30 else np.mean(FPSs)
            print(f"[Progress] {progress:.1f}% | Frame {frame_count}/{total_frames} | "
                  f"Detected: {predict_cnt} | FPS: {avg_fps:.1f}")
    
    # ===== CLEANUP =====
    cap.release()
    cv2.destroyAllWindows()
    
    # ===== CREATE OUTPUT VIDEO =====
    if len(output_frames) > 0:
        make_video(args, output_frames)
    
    # ===== FINAL REPORT =====
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total frames: {frame_count}")
    print(f"Processed frames: {processed_count} (skipped {frame_count - processed_count})")
    print(f"Average FPS: {np.mean(FPSs):.2f}")
    print(f"\nMAXIMUM DENSITY DETECTED:")
    print(f"  Density: {global_max_density:.2f} people/m²")
    print(f"  People in hotspot: {global_max_count_in_cell}")
    print(f"  Frame: {global_max_frame}")
    print(f"  Location: Row {global_max_location[0]+1}, Col {global_max_location[1]+1}")
    print(f"\nOutput saved to: {args.output_dir}/")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet Optimized', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
