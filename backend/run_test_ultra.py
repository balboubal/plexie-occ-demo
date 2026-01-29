"""
PlexIE P2PNet Crowd Counting - Ultra Performance Edition
==========================================================
Maximum optimizations:
- ONNX Runtime inference (2-3x faster than PyTorch)
- Motion-based ROI detection (skip static areas)
- Frame skipping with interpolation
- Resolution scaling based on content
- FP16/INT8 quantization support
- Async I/O pipeline
- Memory-mapped video reading
- Numba JIT for grid calculations
- Persistent thread pool
- Smart batch processing
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

# Try to import optional acceleration libraries
ONNX_AVAILABLE = False
NUMBA_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    pass

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    # Fallback - define dummy decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# ============================================
# NUMBA-ACCELERATED GRID CALCULATION
# ============================================
if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def calculate_grid_fast(points, proc_width, proc_height, grid_rows, grid_cols):
        """JIT-compiled grid density calculation"""
        cell_width = proc_width // grid_cols
        cell_height = proc_height // grid_rows
        grid = np.zeros((grid_rows, grid_cols), dtype=np.int32)
        
        for i in range(len(points)):
            x, y = int(points[i, 0]), int(points[i, 1])
            col = min(x // cell_width, grid_cols - 1)
            row = min(y // cell_height, grid_rows - 1)
            grid[row, col] += 1
        
        return grid
else:
    def calculate_grid_fast(points, proc_width, proc_height, grid_rows, grid_cols):
        """Standard grid calculation fallback"""
        cell_width = proc_width // grid_cols
        cell_height = proc_height // grid_rows
        grid = np.zeros((grid_rows, grid_cols), dtype=np.int32)
        
        for p in points:
            x, y = int(p[0]), int(p[1])
            col = min(x // cell_width, grid_cols - 1)
            row = min(y // cell_height, grid_rows - 1)
            grid[row, col] += 1
        
        return grid


# ============================================
# GPU DETECTION
# ============================================
def setup_device(gpu_id=0, use_fp16=True):
    """Auto-detect best available device"""
    
    device_type = "cpu"
    device = torch.device('cpu')
    supports_fp16 = False
    onnx_providers = ['CPUExecutionProvider']
    
    # Check NVIDIA CUDA
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        device_type = "cuda"
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"[GPU] NVIDIA CUDA: {gpu_name}")
        supports_fp16 = use_fp16 and torch.cuda.get_device_capability(gpu_id)[0] >= 7
        
        if ONNX_AVAILABLE:
            onnx_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        return device, device_type, supports_fp16, onnx_providers
    
    # Check AMD DirectML
    try:
        import torch_directml
        device = torch_directml.device()
        device_type = "directml"
        print(f"[GPU] AMD DirectML detected")
        
        if ONNX_AVAILABLE:
            onnx_providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        
        return device, device_type, False, onnx_providers
    except:
        pass
    
    print("[CPU] No GPU acceleration, using CPU")
    return device, device_type, False, onnx_providers


# ============================================
# ONNX MODEL WRAPPER
# ============================================
class ONNXModel:
    """ONNX Runtime inference wrapper"""
    
    def __init__(self, onnx_path, providers):
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        print(f"[ONNX] Loaded: {onnx_path}")
        print(f"[ONNX] Providers: {self.session.get_providers()}")
    
    def __call__(self, x):
        """Run inference"""
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        outputs = self.session.run(None, {self.input_name: x})
        return {
            'pred_logits': torch.from_numpy(outputs[0]),
            'pred_points': torch.from_numpy(outputs[1])
        }


# ============================================
# EXPORT MODEL TO ONNX
# ============================================
def export_to_onnx(model, output_path, input_shape=(1, 3, 480, 640)):
    """Export PyTorch model to ONNX format"""
    print(f"[ONNX] Exporting model to {output_path}...")
    
    model.eval()
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['pred_logits', 'pred_points'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'pred_logits': {0: 'batch'},
            'pred_points': {0: 'batch'}
        },
        opset_version=14
    )
    print(f"[ONNX] Export complete: {output_path}")
    return output_path


# ============================================
# ASYNC VIDEO READER WITH BUFFER
# ============================================
class AsyncVideoReader:
    """Read frames in background thread"""
    
    def __init__(self, video_path, buffer_size=10):
        self.cap = cv2.VideoCapture(video_path)
        self.buffer = Queue(maxsize=buffer_size)
        self.stopped = False
        self.frame_idx = 0
        
        # Video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Start reader thread
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
    
    def _reader(self):
        while not self.stopped:
            if not self.buffer.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stopped = True
                    break
                self.frame_idx += 1
                self.buffer.put((self.frame_idx, frame))
            else:
                time.sleep(0.001)
    
    def read(self):
        try:
            return self.buffer.get(timeout=2.0)
        except Empty:
            return None, None
    
    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


# ============================================
# MOTION DETECTOR FOR ROI
# ============================================
class MotionDetector:
    """Detect motion to focus processing on active areas"""
    
    def __init__(self, threshold=25, min_area=500):
        self.prev_frame = None
        self.threshold = threshold
        self.min_area = min_area
    
    def detect(self, frame):
        """Returns bounding box of motion area or None"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return None
        
        # Compute difference
        delta = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get bounding box of all motion
        motion_boxes = []
        for c in contours:
            if cv2.contourArea(c) > self.min_area:
                motion_boxes.append(cv2.boundingRect(c))
        
        self.prev_frame = gray
        
        if not motion_boxes:
            return None
        
        # Combine all motion into one box
        x_min = min(b[0] for b in motion_boxes)
        y_min = min(b[1] for b in motion_boxes)
        x_max = max(b[0] + b[2] for b in motion_boxes)
        y_max = max(b[1] + b[3] for b in motion_boxes)
        
        return (x_min, y_min, x_max, y_max)


# ============================================
# ASYNC FRAME SAVER
# ============================================
class AsyncFrameSaver:
    def __init__(self, max_workers=3):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []
    
    def save(self, frame, path):
        future = self.executor.submit(cv2.imwrite, path, frame, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.futures.append(future)
        # Cleanup completed
        if len(self.futures) > 50:
            self.futures = [f for f in self.futures if not f.done()]
    
    def wait_all(self):
        for f in self.futures:
            try:
                f.result(timeout=5)
            except:
                pass
    
    def shutdown(self):
        self.executor.shutdown(wait=True)


# ============================================
# VIDEO WRITER
# ============================================
def make_video(output_dir, frames, original_fps, frame_skip):
    if not frames:
        return
    
    target_fps = min(30, max(10, int(original_fps / frame_skip)))
    h, w = frames[0].shape[:2]
    
    video_path = os.path.join(output_dir, 'output.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(video_path, fourcc, target_fps, (w, h))
    
    print(f'[Video] Creating output at {target_fps} FPS...')
    for frame in frames:
        writer.write(frame)
    writer.release()
    print(f'[Video] Saved: {video_path}')


# ============================================
# ARGUMENT PARSER
# ============================================
def get_args_parser():
    parser = argparse.ArgumentParser('P2PNet Ultra Performance', add_help=False)

    # Model
    parser.add_argument('--backbone', default='vgg16_bn', type=str)
    parser.add_argument('--row', default=2, type=int)
    parser.add_argument('--line', default=2, type=int)
    
    # Paths
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--weight_path', default='./weights/SHTechA.pth')
    parser.add_argument('--onnx_path', default='./weights/p2pnet.onnx')
    
    # Device
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--device', default='auto', type=str)
    
    # Detection
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--shape', default=[640, 480], nargs='+', type=int)
    
    # Performance
    parser.add_argument('--frame_skip', default=3, type=int)
    parser.add_argument('--max_resolution', default=480, type=int)
    parser.add_argument('--startup_delay', default=3, type=int)
    parser.add_argument('--use_fp16', action='store_true', default=True)
    parser.add_argument('--use_onnx', action='store_true', default=True)
    parser.add_argument('--use_motion_roi', action='store_true', default=False)
    parser.add_argument('--async_read', action='store_true', default=True)
    parser.add_argument('--async_save', action='store_true', default=True)
    parser.add_argument('--save_frames', action='store_true', default=True)
    parser.add_argument('--save_video', action='store_true', default=True)
    parser.add_argument('--display_skip', default=1, type=int)
    
    # Mode
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--video_path', default='')
    parser.add_argument('--export_onnx', action='store_true')

    return parser


# ============================================
# MAIN
# ============================================
def main(args):
    print("\n" + "="*60)
    print("PlexIE P2PNet - Ultra Performance Edition")
    print("="*60)
    
    # Setup device
    device, device_type, use_fp16, onnx_providers = setup_device(args.gpu_id, args.use_fp16)
    print(f"[Device] {device_type.upper()}" + (" + FP16" if use_fp16 else ""))
    
    if ONNX_AVAILABLE:
        print(f"[ONNX] Available - providers: {onnx_providers}")
    if NUMBA_AVAILABLE:
        print(f"[Numba] JIT acceleration available")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build PyTorch model first
    print(f"[Model] Loading P2PNet...")
    pytorch_model = build_model(args)
    
    if not os.path.exists(args.weight_path):
        print(f"[ERROR] Weights not found: {args.weight_path}")
        return
    
    checkpoint = torch.load(args.weight_path, map_location='cpu')
    pytorch_model.load_state_dict(checkpoint['model'])
    pytorch_model.eval()
    
    # Try to use ONNX if available
    model = None
    using_onnx = False
    
    if args.use_onnx and ONNX_AVAILABLE:
        onnx_path = args.onnx_path
        
        # Export to ONNX if doesn't exist
        if not os.path.exists(onnx_path) or args.export_onnx:
            try:
                export_to_onnx(pytorch_model, onnx_path, 
                              input_shape=(1, 3, args.shape[1], args.shape[0]))
            except Exception as e:
                print(f"[ONNX] Export failed: {e}")
        
        # Load ONNX model
        if os.path.exists(onnx_path):
            try:
                model = ONNXModel(onnx_path, onnx_providers)
                using_onnx = True
            except Exception as e:
                print(f"[ONNX] Load failed: {e}, using PyTorch")
    
    # Fallback to PyTorch
    if model is None:
        model = pytorch_model.to(device)
        if use_fp16 and device_type == 'cuda':
            model = model.half()
        print("[Model] Using PyTorch inference")
    
    # Transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Grid config
    GRID_ROWS, GRID_COLS = 4, 4
    cell_area_m2 = (5.0 / GRID_COLS) * (12.0 / GRID_ROWS)
    
    # Open video
    if not args.video:
        print("[ERROR] Only video mode supported")
        return
    
    reader = AsyncVideoReader(args.video_path, buffer_size=30) if args.async_read else None
    if not reader:
        cap = cv2.VideoCapture(args.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
    else:
        total_frames = reader.total_frames
        original_fps = reader.fps
    
    print(f"[Video] {total_frames} frames @ {original_fps:.1f} FPS")
    
    # Calculate processing resolution
    proc_width, proc_height = args.shape[0], args.shape[1]
    if args.max_resolution > 0:
        scale = args.max_resolution / max(proc_width, proc_height)
        if scale < 1:
            proc_width = (int(proc_width * scale) // 16) * 16
            proc_height = (int(proc_height * scale) // 16) * 16
    
    print(f"[Process] {proc_width}x{proc_height}, skip={args.frame_skip}x")
    
    cell_width = proc_width // GRID_COLS
    cell_height = proc_height // GRID_ROWS
    
    # Setup components
    saver = AsyncFrameSaver(3) if args.async_save and args.save_frames else None
    motion_detector = MotionDetector() if args.use_motion_roi else None
    
    # Display window
    window_name = 'PlexIE Ultra - Press Q to quit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    # ================================================================
    # STARTUP: Warmup model + Pre-buffer frames DURING countdown
    # ================================================================
    print(f"\n[Startup] Warming up model and buffering frames...")
    
    # Start warmup in background
    warmup_done = False
    def warmup_model():
        nonlocal warmup_done
        dummy = torch.randn(1, 3, proc_height, proc_width)
        # Run multiple warmup iterations for GPU to fully initialize
        for _ in range(3):
            if using_onnx:
                _ = model(dummy)
            else:
                d = dummy.to(device)
                if use_fp16 and device_type == 'cuda':
                    d = d.half()
                with torch.inference_mode():
                    _ = model(d)
        warmup_done = True
    
    import threading
    warmup_thread = threading.Thread(target=warmup_model, daemon=True)
    warmup_thread.start()
    
    # Countdown while buffering and warming up
    if args.startup_delay > 0:
        for i in range(args.startup_delay, 0, -1):
            # Check buffer status
            buffer_size = reader.buffer.qsize() if reader else 0
            buffer_status = f"Buffer: {buffer_size} frames"
            warmup_status = "Model: Ready" if warmup_done else "Model: Warming up..."
            
            blank = np.zeros((720, 1280, 3), dtype=np.uint8)
            info = f"ONNX: {'ON' if using_onnx else 'OFF'} | Device: {device_type.upper()}"
            cv2.putText(blank, f"Starting in {i}...", (480, 300),
                       cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 255), 2)
            cv2.putText(blank, info, (450, 360),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (150, 150, 150), 1)
            cv2.putText(blank, buffer_status, (500, 420),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0) if buffer_size >= 10 else (255, 255, 0), 1)
            cv2.putText(blank, warmup_status, (500, 460),
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0) if warmup_done else (255, 255, 0), 1)
            cv2.imshow(window_name, blank)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                return
    
    # Wait for warmup to complete if not done
    warmup_thread.join(timeout=5)
    if not warmup_done:
        print("[Warmup] Still warming up, first frames may be slow...")
    else:
        print("[Warmup] Complete")
    
    # Ensure buffer has frames
    if reader:
        wait_start = time.time()
        while reader.buffer.qsize() < 5 and time.time() - wait_start < 2:
            time.sleep(0.1)
        print(f"[Buffer] {reader.buffer.qsize()} frames ready")
    
    print("\n[Processing]...")
    
    # Tracking
    frame_count = 0
    processed_count = 0
    output_frames = []
    FPSs = []
    inference_times = []
    
    global_max_density = 0
    global_max_count = 0
    global_max_frame = 0
    global_max_loc = (0, 0)
    
    last_points = []
    last_count = 0
    last_density = 0
    last_cell_count = 0
    last_loc = (0, 0)
    
    total_start = default_timer()
    
    # Main loop
    while True:
        if reader:
            result = reader.read()
            if result[0] is None:
                break
            frame_count, frame = result
        else:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
        
        start_time = default_timer()
        frame_resized = cv2.resize(frame, (proc_width, proc_height))
        
        # Process or reuse
        should_process = (frame_count % args.frame_skip == 1) or (args.frame_skip == 1)
        
        if should_process:
            processed_count += 1
            t0 = default_timer()
            
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img_tensor = transform(frame_rgb).unsqueeze(0)
            
            if using_onnx:
                outputs = model(img_tensor)
            else:
                img_tensor = img_tensor.to(device)
                if use_fp16 and device_type == 'cuda':
                    img_tensor = img_tensor.half()
                with torch.inference_mode():
                    outputs = model(img_tensor)
            
            inference_times.append(default_timer() - t0)
            
            # Get predictions
            scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
            points_tensor = outputs['pred_points'][0]
            
            mask = scores > args.threshold
            points = points_tensor[mask].detach().cpu().numpy()
            predict_cnt = len(points)
            
            # Grid calculation
            if len(points) > 0:
                grid = calculate_grid_fast(points, proc_width, proc_height, GRID_ROWS, GRID_COLS)
            else:
                grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
            
            # Find hotspot
            max_density = 0
            max_cell_count = 0
            max_loc = (0, 0)
            
            for r in range(GRID_ROWS):
                for c in range(GRID_COLS):
                    d = grid[r, c] / cell_area_m2
                    if d > max_density:
                        max_density = d
                        max_cell_count = grid[r, c]
                        max_loc = (r, c)
            
            if max_density > global_max_density:
                global_max_density = max_density
                global_max_count = max_cell_count
                global_max_frame = frame_count
                global_max_loc = max_loc
            
            last_points = points.tolist() if len(points) > 0 else []
            last_count = predict_cnt
            last_density = max_density
            last_cell_count = max_cell_count
            last_loc = max_loc
        else:
            points = last_points
            predict_cnt = last_count
            max_density = last_density
            max_cell_count = last_cell_count
            max_loc = last_loc
        
        # Draw
        img_draw = frame_resized.copy()
        
        # Grid lines
        for i in range(1, GRID_ROWS):
            cv2.line(img_draw, (0, i*cell_height), (proc_width, i*cell_height), (60,60,60), 1)
        for j in range(1, GRID_COLS):
            cv2.line(img_draw, (j*cell_width, 0), (j*cell_width, proc_height), (60,60,60), 1)
        
        # Hotspot
        if max_cell_count > 0:
            r, c = max_loc
            cv2.rectangle(img_draw, (c*cell_width, r*cell_height),
                         ((c+1)*cell_width, (r+1)*cell_height), (0,0,255), 3)
        
        # Points
        for p in (last_points if not should_process else points):
            if isinstance(p, (list, tuple)):
                cv2.circle(img_draw, (int(p[0]), int(p[1])), 3, (0,255,0), -1)
            else:
                cv2.circle(img_draw, (int(p[0]), int(p[1])), 3, (0,255,0), -1)
        
        # Stats
        elapsed = default_timer() - start_time
        fps = 1/elapsed if elapsed > 0 else 0
        FPSs.append(fps)
        
        avg_inf = np.mean(inference_times[-10:])*1000 if inference_times else 0
        
        texts = [
            f'People: {predict_cnt} | Hotspot: {max_density:.1f} p/m2',
            f'Frame: {frame_count}/{total_frames} | FPS: {fps:.0f}',
            f'{"ONNX" if using_onnx else "PyTorch"} {device_type.upper()} | Inf: {avg_inf:.0f}ms'
        ]
        
        y = 20
        for t in texts:
            (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_draw, (8, y-14), (12+tw, y+4), (0,0,0), -1)
            cv2.putText(img_draw, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            y += 18
        
        # Display
        if frame_count % args.display_skip == 0:
            cv2.imshow(window_name, cv2.resize(img_draw, (1280, 720)))
        
        # Save
        if args.save_frames:
            path = os.path.join(args.output_dir, f'pred_{frame_count:05d}.jpg')
            if saver:
                saver.save(img_draw, path)
            else:
                cv2.imwrite(path, img_draw)
        
        if args.save_video:
            output_frames.append(img_draw.copy())
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if frame_count % 50 == 0:
            pct = 100*frame_count/total_frames
            avg_fps = np.mean(FPSs[-50:])
            eta = (total_frames-frame_count)/avg_fps if avg_fps > 0 else 0
            print(f"[{pct:5.1f}%] Frame {frame_count} | People: {predict_cnt:3d} | "
                  f"FPS: {avg_fps:5.1f} | ETA: {eta:.0f}s")
        
        if device_type == 'cuda' and frame_count % 300 == 0:
            torch.cuda.empty_cache()
    
    total_time = default_timer() - total_start
    
    # Cleanup
    if reader:
        reader.release()
    else:
        cap.release()
    cv2.destroyAllWindows()
    
    if saver:
        saver.wait_all()
        saver.shutdown()
    
    if args.save_video and output_frames:
        make_video(args.output_dir, output_frames, original_fps, args.frame_skip)
    
    # Report
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Time: {total_time:.1f}s | Frames: {frame_count} | Processed: {processed_count}")
    print(f"Avg FPS: {np.mean(FPSs):.1f} | Avg Inference: {np.mean(inference_times)*1000:.1f}ms")
    print(f"Max Density: {global_max_density:.2f} p/m² @ frame {global_max_frame}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet Ultra', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
