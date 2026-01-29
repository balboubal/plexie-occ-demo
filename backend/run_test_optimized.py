"""
PlexIE P2PNet Crowd Counting - Maximum Performance Edition
===========================================================
Optimizations:
- Frame skipping (process every Nth frame)
- Resolution limiting for faster inference
- Auto GPU detection (NVIDIA CUDA / AMD DirectML / CPU)
- FP16 half-precision inference (2x faster on GPU)
- torch.compile() JIT compilation (PyTorch 2.0+)
- Async frame reading (read ahead while processing)
- Threaded frame saving (non-blocking I/O)
- Pre-allocated tensors (reduced memory churn)
- Optional frame saving (faster if disabled)
"""

import argparse
from timeit import default_timer
import torch
import torchvision.transforms as standard_transforms
import numpy as np
from PIL import Image
import cv2
from models import build_model
import os
import warnings
import time
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import gc

warnings.filterwarnings('ignore')


# ============================================
# GPU DETECTION AND SETUP
# ============================================
def setup_device(gpu_id=0, use_fp16=True):
    """Auto-detect best available device: CUDA > DirectML > CPU"""
    
    device_type = "cpu"
    device = torch.device('cpu')
    supports_fp16 = False
    
    # Try NVIDIA CUDA first
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        device_type = "cuda"
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"[GPU] NVIDIA CUDA detected: {gpu_name}")
        
        # Check FP16 support (most modern GPUs support it)
        supports_fp16 = use_fp16 and torch.cuda.get_device_capability(gpu_id)[0] >= 7
        if supports_fp16:
            print(f"[GPU] FP16 half-precision: ENABLED")
        return device, device_type, supports_fp16
    
    # Try AMD DirectML (Windows)
    try:
        import torch_directml
        device = torch_directml.device()
        device_type = "directml"
        print(f"[GPU] AMD DirectML detected")
        # DirectML FP16 support varies - disable for safety
        supports_fp16 = False
        return device, device_type, supports_fp16
    except ImportError:
        pass
    except Exception as e:
        print(f"[GPU] DirectML error: {e}")
    
    # Fallback to CPU
    print("[CPU] No GPU acceleration available, using CPU")
    print("      For AMD GPUs: pip install torch-directml")
    return device, device_type, False


# ============================================
# ASYNC FRAME READER
# ============================================
class AsyncVideoReader:
    """Read frames in a background thread for zero-latency frame access"""
    
    def __init__(self, video_path, buffer_size=5):
        self.cap = cv2.VideoCapture(video_path)
        self.buffer = Queue(maxsize=buffer_size)
        self.stopped = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
        
        # Video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def _reader(self):
        """Background thread that reads frames"""
        while not self.stopped:
            if not self.buffer.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.buffer.put(frame)
            else:
                time.sleep(0.001)  # Small sleep to prevent CPU spinning
    
    def read(self):
        """Get next frame (blocks if buffer empty)"""
        try:
            return True, self.buffer.get(timeout=1.0)
        except Empty:
            return False, None
    
    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


# ============================================
# ASYNC FRAME SAVER
# ============================================
class AsyncFrameSaver:
    """Save frames in background threads"""
    
    def __init__(self, max_workers=2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []
    
    def save(self, frame, path):
        """Queue frame for async saving"""
        future = self.executor.submit(cv2.imwrite, path, frame)
        self.futures.append(future)
        
        # Clean up completed futures periodically
        if len(self.futures) > 100:
            self.futures = [f for f in self.futures if not f.done()]
    
    def wait_all(self):
        """Wait for all saves to complete"""
        for f in self.futures:
            f.result()
        self.futures = []
    
    def shutdown(self):
        self.executor.shutdown(wait=True)


# ============================================
# VIDEO OUTPUT CREATION
# ============================================
def make_video(output_dir, output_frames, original_fps, frame_skip):
    """Create video from processed frames"""
    if len(output_frames) == 0:
        print("No frames to create video from")
        return
    
    # Use reasonable FPS for output
    target_fps = min(30, max(10, int(original_fps / frame_skip)))
    
    print(f'\n[Video] Creating output at {target_fps} FPS from {len(output_frames)} frames...')
    
    # Get frame size from first frame
    h, w = output_frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_path = os.path.join(output_dir, 'output.avi')
    video = cv2.VideoWriter(video_path, fourcc, target_fps, (w, h))
    
    for frame in output_frames:
        video.write(frame)
    
    video.release()
    print(f'[Video] Saved: {video_path}')


# ============================================
# ARGUMENT PARSER
# ============================================
def get_args_parser():
    parser = argparse.ArgumentParser('P2PNet Maximum Performance', add_help=False)

    # Model settings
    parser.add_argument('--backbone', default='vgg16_bn', type=str)
    parser.add_argument('--row', default=2, type=int)
    parser.add_argument('--line', default=2, type=int)
    
    # Paths
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--weight_path', default='./weights/SHTechA.pth')
    
    # Device settings
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--device', default='auto', type=str,
                        help='auto, cuda, directml, or cpu')
    
    # Detection settings
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--shape', default=[640, 480], nargs='+', type=int)
    
    # Optimization settings
    parser.add_argument('--frame_skip', default=3, type=int,
                        help='Process every Nth frame')
    parser.add_argument('--max_resolution', default=480, type=int,
                        help='Max resolution (longer edge)')
    parser.add_argument('--startup_delay', default=3, type=int,
                        help='Seconds before starting')
    
    # Performance flags
    parser.add_argument('--use_fp16', action='store_true', default=True,
                        help='Use FP16 half precision (faster on GPU)')
    parser.add_argument('--use_compile', action='store_true', default=False,
                        help='Use torch.compile (PyTorch 2.0+, slower startup)')
    parser.add_argument('--async_read', action='store_true', default=True,
                        help='Async frame reading')
    parser.add_argument('--async_save', action='store_true', default=True,
                        help='Async frame saving')
    parser.add_argument('--save_frames', action='store_true', default=True,
                        help='Save individual frames (disable for speed)')
    parser.add_argument('--save_video', action='store_true', default=True,
                        help='Create output video')
    parser.add_argument('--display_skip', default=1, type=int,
                        help='Update display every N frames')
    
    # Mode settings
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--video_path', default='')

    return parser


# ============================================
# MAIN PROCESSING
# ============================================
def main(args):
    print("\n" + "="*60)
    print("PlexIE P2PNet - Maximum Performance Edition")
    print("="*60)
    
    # ===== SETUP DEVICE =====
    if args.device == 'auto':
        device, device_type, use_fp16 = setup_device(args.gpu_id, args.use_fp16)
    elif args.device == 'cuda':
        device = torch.device(f'cuda:{args.gpu_id}')
        device_type = 'cuda'
        use_fp16 = args.use_fp16
    elif args.device == 'directml':
        import torch_directml
        device = torch_directml.device()
        device_type = 'directml'
        use_fp16 = False
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
        use_fp16 = False
    
    print(f"[Device] Using: {device_type.upper()}")
    
    # ===== CREATE DIRECTORIES =====
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Clear old output files
    for f in os.listdir(args.output_dir):
        if f.startswith('pred_') and f.endswith('.jpg'):
            os.remove(os.path.join(args.output_dir, f))

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
    
    # ===== APPLY OPTIMIZATIONS =====
    
    # FP16 half precision
    if use_fp16 and device_type == 'cuda':
        model = model.half()
        print("[Optim] FP16 half precision: ENABLED")
    
    # torch.compile (PyTorch 2.0+)
    if args.use_compile:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("[Optim] torch.compile: ENABLED (first inference will be slow)")
        except Exception as e:
            print(f"[Optim] torch.compile failed: {e}")
    
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
        print("[ERROR] Only video mode supported")
        return
    
    # Use async reader if enabled
    if args.async_read:
        reader = AsyncVideoReader(args.video_path, buffer_size=30)
        total_frames = reader.total_frames
        original_fps = reader.fps
        orig_width = reader.width
        orig_height = reader.height
        print("[Optim] Async frame reading: ENABLED (30 frame buffer)")
    else:
        reader = cv2.VideoCapture(args.video_path)
        total_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = reader.get(cv2.CAP_PROP_FPS)
        orig_width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[Video] {args.video_path}")
    print(f"        {orig_width}x{orig_height} @ {original_fps:.1f} FPS, {total_frames} frames")
    
    # ===== CALCULATE PROCESSING RESOLUTION =====
    proc_width, proc_height = args.shape[0], args.shape[1]
    
    if args.max_resolution > 0:
        max_dim = max(proc_width, proc_height)
        if max_dim > args.max_resolution:
            scale = args.max_resolution / max_dim
            proc_width = int(proc_width * scale)
            proc_height = int(proc_height * scale)
            proc_width = (proc_width // 16) * 16
            proc_height = (proc_height // 16) * 16
    
    print(f"[Process] Resolution: {proc_width}x{proc_height}")
    print(f"[Process] Frame skip: {args.frame_skip}x")
    
    cell_width = proc_width // GRID_COLS
    cell_height = proc_height // GRID_ROWS
    
    # ===== SETUP ASYNC SAVER =====
    saver = None
    if args.async_save and args.save_frames:
        saver = AsyncFrameSaver(max_workers=2)
        print("[Optim] Async frame saving: ENABLED")
    
    # ===== SETUP DISPLAY WINDOW =====
    window_name = 'PlexIE P2PNet Live - Press Q to quit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    # ===== PRE-ALLOCATE TENSORS =====
    dummy_input = torch.zeros(1, 3, proc_height, proc_width, device=device)
    if use_fp16 and device_type == 'cuda':
        dummy_input = dummy_input.half()
    
    # ================================================================
    # STARTUP: Warmup model + Pre-buffer frames DURING countdown
    # ================================================================
    print(f"\n[Startup] Warming up model and buffering frames...")
    
    warmup_done = False
    def warmup_model():
        nonlocal warmup_done
        # Run multiple warmup iterations
        for _ in range(3):
            with torch.inference_mode():
                _ = model(dummy_input)
        warmup_done = True
    
    warmup_thread = threading.Thread(target=warmup_model, daemon=True)
    warmup_thread.start()
    
    # Countdown while buffering and warming up
    if args.startup_delay > 0:
        for i in range(args.startup_delay, 0, -1):
            buffer_size = reader.buffer.qsize() if hasattr(reader, 'buffer') else 0
            buffer_status = f"Buffer: {buffer_size} frames"
            warmup_status = "Model: Ready" if warmup_done else "Model: Warming up..."
            
            blank = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(blank, f"Starting in {i}...", (500, 300), 
                       cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            cv2.putText(blank, f"Device: {device_type.upper()}", (480, 360),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (150, 150, 150), 1)
            cv2.putText(blank, buffer_status, (520, 420),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0) if buffer_size >= 10 else (255, 255, 0), 1)
            cv2.putText(blank, warmup_status, (520, 460),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0) if warmup_done else (255, 255, 0), 1)
            cv2.imshow(window_name, blank)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                print("[Cancelled]")
                cv2.destroyAllWindows()
                if hasattr(reader, 'release'):
                    reader.release()
                return
    
    # Wait for warmup to complete
    warmup_thread.join(timeout=5)
    if warmup_done:
        print("[Warmup] Complete")
    else:
        print("[Warmup] Still running, first frames may be slow...")
    
    # Ensure buffer has frames
    if hasattr(reader, 'buffer'):
        wait_start = time.time()
        while reader.buffer.qsize() < 5 and time.time() - wait_start < 2:
            time.sleep(0.1)
        print(f"[Buffer] {reader.buffer.qsize()} frames ready")
    
    # Clear GPU cache before starting
    if device_type == 'cuda':
        torch.cuda.empty_cache()
    
    print("\n[Processing] Started...")
    print("-" * 60)
    
    # ===== TRACKING VARIABLES =====
    frame_count = 0
    processed_count = 0
    output_frames = []
    FPSs = []
    inference_times = []
    
    global_max_density = 0
    global_max_count_in_cell = 0
    global_max_frame = 0
    global_max_location = (0, 0)
    
    last_detections = []
    last_count = 0
    last_max_density = 0
    last_max_cell_count = 0
    last_max_location = (0, 0)
    
    total_start = default_timer()
    
    # ===== MAIN PROCESSING LOOP =====
    while True:
        ret, frame = reader.read()
        if not ret:
            break
        
        frame_count += 1
        start_time = default_timer()
        
        # Resize for processing
        frame_resized = cv2.resize(frame, (proc_width, proc_height))
        
        # ===== PROCESS OR REUSE =====
        should_process = (frame_count % args.frame_skip == 1) or (args.frame_skip == 1)
        
        if should_process:
            processed_count += 1
            inference_start = default_timer()
            
            # Convert to RGB and preprocess
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_tensor = transform(frame_rgb).unsqueeze(0).to(device)
            
            # FP16 conversion
            if use_fp16 and device_type == 'cuda':
                img_tensor = img_tensor.half()
            
            # Inference
            with torch.inference_mode():
                outputs = model(img_tensor)
            
            inference_times.append(default_timer() - inference_start)
            
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
        
        # Highlight hotspot
        if max_cell_count > 0:
            row, col = max_cell_location
            x1, y1 = col * cell_width, row * cell_height
            x2, y2 = x1 + cell_width, y1 + cell_height
            cv2.rectangle(img_to_draw, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Draw detection points
        for p in points:
            cv2.circle(img_to_draw, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)
        
        # Calculate FPS
        elapsed = default_timer() - start_time
        current_fps = 1.0 / elapsed if elapsed > 0 else 0
        FPSs.append(current_fps)
        
        # Draw text overlay
        avg_inference = np.mean(inference_times[-10:]) * 1000 if inference_times else 0
        overlay_texts = [
            f'People: {predict_cnt} | Hotspot: {max_density_this_frame:.1f} p/m2',
            f'Frame: {frame_count}/{total_frames} ({100*frame_count/total_frames:.0f}%)',
            f'FPS: {current_fps:.1f} | Inference: {avg_inference:.0f}ms',
            f'{device_type.upper()}' + (' FP16' if use_fp16 else '') + f' | Skip {args.frame_skip}x',
        ]
        
        y_offset = 22
        for text in overlay_texts:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_to_draw, (8, y_offset - 15), (15 + tw, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(img_to_draw, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            y_offset += 20
        
        # ===== DISPLAY (skip some frames for speed) =====
        if frame_count % args.display_skip == 0:
            display_frame = cv2.resize(img_to_draw, (1280, 720))
            cv2.imshow(window_name, display_frame)
        
        # ===== SAVE FRAME =====
        if args.save_frames:
            frame_path = os.path.join(args.output_dir, f'pred_{frame_count:05d}.jpg')
            if saver:
                saver.save(img_to_draw, frame_path)
            else:
                cv2.imwrite(frame_path, img_to_draw)
        
        if args.save_video:
            output_frames.append(img_to_draw.copy())
        
        # Check for quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[User] Stopped early")
            break
        
        # Progress update
        if frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            avg_fps = np.mean(FPSs[-50:]) if len(FPSs) >= 50 else np.mean(FPSs)
            eta = (total_frames - frame_count) / avg_fps if avg_fps > 0 else 0
            print(f"[{progress:5.1f}%] Frame {frame_count}/{total_frames} | "
                  f"People: {predict_cnt:3d} | FPS: {avg_fps:5.1f} | ETA: {eta:.0f}s")
        
        # Periodic GPU cache clear
        if device_type == 'cuda' and frame_count % 500 == 0:
            torch.cuda.empty_cache()
    
    total_time = default_timer() - total_start
    
    # ===== CLEANUP =====
    reader.release()
    cv2.destroyAllWindows()
    
    # Wait for async saves
    if saver:
        print("[Saving] Waiting for frame saves to complete...")
        saver.wait_all()
        saver.shutdown()
    
    # ===== CREATE OUTPUT VIDEO =====
    if args.save_video and len(output_frames) > 0:
        make_video(args.output_dir, output_frames, original_fps, args.frame_skip)
    
    # ===== FINAL REPORT =====
    avg_inference = np.mean(inference_times) * 1000 if inference_times else 0
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total time: {total_time:.1f}s")
    print(f"Total frames: {frame_count}")
    print(f"Processed frames: {processed_count} ({100*processed_count/frame_count:.0f}%)")
    print(f"Average FPS: {np.mean(FPSs):.1f}")
    print(f"Average inference: {avg_inference:.1f}ms")
    print(f"\nMAXIMUM DENSITY:")
    print(f"  {global_max_density:.2f} people/m² ({global_max_count_in_cell} people)")
    print(f"  Frame {global_max_frame}, Cell ({global_max_location[0]+1}, {global_max_location[1]+1})")
    print(f"\nOutput: {args.output_dir}/")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet Max Performance', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
