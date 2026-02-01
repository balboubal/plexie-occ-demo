"""
PlexIE Video Processor - JSON Detection Output
===============================================
Processes video with P2PNet and outputs:
1. Detection data as JSON (frame-indexed)
2. Optionally: annotated video for reference

Usage:
  python process_video_json.py --video_path input.mp4 --output_dir ./output

Output JSON format:
{
  "video_info": { "fps": 30, "total_frames": 900, "width": 640, "height": 480 },
  "grid_config": { "rows": 4, "cols": 4, "cell_width": 160, "cell_height": 120 },
  "frames": {
    "1": { "count": 45, "points": [[x,y], ...], "grid": [[3,2,1,0], ...], "max_density": 2.5 },
    "2": { ... },
    ...
  }
}
"""

import argparse
import json
import torch
import torchvision.transforms as standard_transforms
import numpy as np
import cv2
from models import build_model
import os
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Grid configuration
GRID_ROWS = 4
GRID_COLS = 4
ASSUMED_CELL_AREA_M2 = 4.0  # Assumed area per grid cell in m²


def setup_device(gpu_id=0):
    """Auto-detect best available device"""
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        print(f"[GPU] NVIDIA CUDA: {torch.cuda.get_device_name(gpu_id)}")
        return device, "cuda"
    
    try:
        import torch_directml
        device = torch_directml.device()
        print(f"[GPU] AMD DirectML detected")
        return device, "directml"
    except:
        pass
    
    print("[CPU] Using CPU (install torch-directml for AMD GPU support)")
    return torch.device('cpu'), "cpu"


def get_args_parser():
    parser = argparse.ArgumentParser('PlexIE Video Processor')
    
    parser.add_argument('--video_path', required=True, help='Input video path')
    parser.add_argument('--output_dir', default='./output', help='Output directory')
    parser.add_argument('--output_name', default=None, help='Output filename (without extension)')
    parser.add_argument('--weight_path', default='./weights/SHTechA.pth', help='P2PNet weights')
    parser.add_argument('--backbone', default='vgg16_bn', type=str)
    parser.add_argument('--row', default=2, type=int)
    parser.add_argument('--line', default=2, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--threshold', default=0.5, type=float, help='Detection confidence threshold')
    parser.add_argument('--frame_skip', default=1, type=int, help='Process every Nth frame (1=all)')
    parser.add_argument('--max_width', default=640, type=int, help='Max processing width')
    parser.add_argument('--save_video', action='store_true', help='Also save annotated video')
    parser.add_argument('--no_display', action='store_true', help='Disable live preview')
    
    return parser


def main(args):
    print("\n" + "=" * 60)
    print("PlexIE Video Processor - JSON Output")
    print("=" * 60)
    
    # Setup device
    device, device_type = setup_device(args.gpu_id)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine output filename
    if args.output_name:
        output_base = args.output_name
    else:
        output_base = Path(args.video_path).stem
    
    json_path = os.path.join(args.output_dir, f"{output_base}_detections.json")
    video_path = os.path.join(args.output_dir, f"{output_base}_annotated.mp4")
    
    # Load model
    print(f"\n[Model] Loading P2PNet weights from {args.weight_path}")
    model = build_model(args)
    
    if not os.path.exists(args.weight_path):
        print(f"[ERROR] Weight file not found: {args.weight_path}")
        return
    
    checkpoint = torch.load(args.weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print("[Model] Loaded successfully")
    
    # Transform for preprocessing
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Open video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video_path}")
        return
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate processing size
    scale = min(1.0, args.max_width / original_width)
    proc_width = int(original_width * scale)
    proc_height = int(original_height * scale)
    
    # Make height divisible by 16 for model
    proc_height = (proc_height // 16) * 16
    proc_width = (proc_width // 16) * 16
    
    # Grid cell dimensions
    cell_width = proc_width // GRID_COLS
    cell_height = proc_height // GRID_ROWS
    
    print(f"\n[Video] {args.video_path}")
    print(f"  Original: {original_width}x{original_height} @ {original_fps:.1f} FPS")
    print(f"  Processing: {proc_width}x{proc_height}")
    print(f"  Frames: {total_frames} (processing every {args.frame_skip})")
    print(f"  Grid: {GRID_ROWS}x{GRID_COLS} cells ({cell_width}x{cell_height} px each)")
    
    # Initialize output structures
    detection_data = {
        "video_info": {
            "source": args.video_path,
            "fps": original_fps,
            "total_frames": total_frames,
            "original_width": original_width,
            "original_height": original_height,
            "process_width": proc_width,
            "process_height": proc_height,
            "frame_skip": args.frame_skip
        },
        "grid_config": {
            "rows": GRID_ROWS,
            "cols": GRID_COLS,
            "cell_width": cell_width,
            "cell_height": cell_height,
            "assumed_cell_area_m2": ASSUMED_CELL_AREA_M2
        },
        "detection_config": {
            "threshold": args.threshold,
            "model": "P2PNet",
            "weights": args.weight_path
        },
        "frames": {}
    }
    
    # Video writer for annotated output
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, original_fps, (proc_width, proc_height))
    
    # Processing loop
    print("\n[Processing] Starting...")
    print("-" * 60)
    
    frame_idx = 0
    processed_count = 0
    last_detection = {"count": 0, "points": [], "grid": [[0]*GRID_COLS for _ in range(GRID_ROWS)], "max_density": 0, "max_cell": [0, 0]}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Resize for processing
        frame_resized = cv2.resize(frame, (proc_width, proc_height))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Process or reuse previous detection
        if frame_idx % args.frame_skip == 1 or args.frame_skip == 1:
            processed_count += 1
            
            # Preprocess
            img_tensor = transform(frame_rgb)
            samples = img_tensor.unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(samples)
            
            # Get predictions
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
            outputs_points = outputs['pred_points'][0]
            
            # Filter by threshold
            mask = outputs_scores > args.threshold
            points = outputs_points[mask].detach().cpu().numpy().tolist()
            
            # Calculate grid density
            grid = [[0]*GRID_COLS for _ in range(GRID_ROWS)]
            for p in points:
                x, y = int(p[0]), int(p[1])
                col = min(x // cell_width, GRID_COLS - 1)
                row = min(y // cell_height, GRID_ROWS - 1)
                grid[row][col] += 1
            
            # Find max density cell
            max_density = 0
            max_cell = [0, 0]
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    density = grid[r][c] / ASSUMED_CELL_AREA_M2
                    if density > max_density:
                        max_density = density
                        max_cell = [r, c]
            
            # Store detection
            last_detection = {
                "count": len(points),
                "points": [[round(p[0], 1), round(p[1], 1)] for p in points],
                "grid": grid,
                "max_density": round(max_density, 2),
                "max_cell": max_cell
            }
        
        # Save frame detection data
        detection_data["frames"][str(frame_idx)] = last_detection.copy()
        
        # Draw annotated frame if saving video or displaying
        if args.save_video or not args.no_display:
            img_draw = frame_resized.copy()
            
            # Draw grid
            for i in range(1, GRID_ROWS):
                cv2.line(img_draw, (0, i * cell_height), (proc_width, i * cell_height), (80, 80, 80), 1)
            for j in range(1, GRID_COLS):
                cv2.line(img_draw, (j * cell_width, 0), (j * cell_width, proc_height), (80, 80, 80), 1)
            
            # Highlight max density cell
            if last_detection["count"] > 0:
                r, c = last_detection["max_cell"]
                x1, y1 = c * cell_width, r * cell_height
                x2, y2 = x1 + cell_width, y1 + cell_height
                # Color based on density (green -> yellow -> red)
                density = last_detection["max_density"]
                if density > 5:
                    color = (0, 0, 255)  # Red - dangerous
                elif density > 3:
                    color = (0, 165, 255)  # Orange - warning
                else:
                    color = (0, 255, 255)  # Yellow - caution
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            
            # Draw detection points
            for p in last_detection["points"]:
                cv2.circle(img_draw, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
            
            # Draw info overlay
            texts = [
                f"Count: {last_detection['count']}",
                f"Max Density: {last_detection['max_density']:.1f} p/m2",
                f"Frame: {frame_idx}/{total_frames}"
            ]
            y = 20
            for text in texts:
                cv2.putText(img_draw, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y += 20
            
            if args.save_video:
                video_writer.write(img_draw)
            
            if not args.no_display:
                cv2.imshow("PlexIE Processing", img_draw)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[User] Stopped early")
                    break
        
        # Progress
        if frame_idx % 100 == 0:
            progress = frame_idx / total_frames * 100
            print(f"[Progress] {progress:.1f}% - Frame {frame_idx}/{total_frames} - Count: {last_detection['count']}")
    
    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    # Save JSON
    print(f"\n[Saving] Writing detection data to {json_path}")
    with open(json_path, 'w') as f:
        json.dump(detection_data, f)
    
    # Calculate file size
    json_size = os.path.getsize(json_path) / 1024 / 1024
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Frames processed: {processed_count}/{total_frames}")
    print(f"JSON output: {json_path} ({json_size:.1f} MB)")
    if args.save_video:
        video_size = os.path.getsize(video_path) / 1024 / 1024
        print(f"Video output: {video_path} ({video_size:.1f} MB)")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('PlexIE Video Processor', parents=[get_args_parser()], add_help=False)
    args = parser.parse_args()
    main(args)
