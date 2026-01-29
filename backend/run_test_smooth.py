"""
PlexIE P2PNet Crowd Counting - Smooth Edition v3
=================================================
TRUE smooth playback with async inference:
- Display runs at video FPS (never blocked)
- AI inference runs in BACKGROUND THREAD
- Detection results update asynchronously
- GPU acceleration (DirectML for AMD, CUDA for NVIDIA)
"""

import argparse
from timeit import default_timer
import torch
import torchvision.transforms as standard_transforms
import numpy as np
import cv2
from models import build_model
import os
import warnings
import time
import threading
from queue import Queue, Empty, Full
from collections import deque
import gc

warnings.filterwarnings('ignore')

# Try optional accelerators
ONNX_AVAILABLE = False
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    pass


# ============================================
# GPU DETECTION - VERBOSE
# ============================================
def setup_device(gpu_id=0):
    """Auto-detect best available device with detailed logging"""
    
    print("\n[GPU Detection]")
    
    # Check NVIDIA CUDA
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{gpu_id}')
        gpu_name = torch.cuda.get_device_name(gpu_id)
        print(f"  → Using NVIDIA CUDA: {gpu_name}")
        supports_fp16 = torch.cuda.get_device_capability(gpu_id)[0] >= 7
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ONNX_AVAILABLE else []
        return device, "cuda", supports_fp16, providers
    
    # Check AMD DirectML
    print("  Checking DirectML...")
    try:
        import torch_directml
        print(f"  torch_directml imported OK")
        print(f"  Device count: {torch_directml.device_count()}")
        
        if torch_directml.device_count() > 0:
            dml_device = torch_directml.device(gpu_id)
            print(f"  Device object: {dml_device}")
            
            # Test tensor operation
            test_tensor = torch.tensor([1.0, 2.0, 3.0])
            test_tensor = test_tensor.to(dml_device)
            result = test_tensor * 2
            print(f"  Test computation: {result.cpu().numpy()}")
            
            print(f"  → Using AMD DirectML!")
            providers = ['DmlExecutionProvider', 'CPUExecutionProvider'] if ONNX_AVAILABLE else []
            return dml_device, "directml", False, providers
        else:
            print("  No DirectML devices found")
    except ImportError as e:
        print(f"  DirectML import failed: {e}")
    except Exception as e:
        print(f"  DirectML error: {type(e).__name__}: {e}")
    
    print("  → Falling back to CPU")
    providers = ['CPUExecutionProvider'] if ONNX_AVAILABLE else []
    return torch.device('cpu'), "cpu", False, providers


# ============================================
# ASYNC INFERENCE THREAD
# ============================================
class AsyncInference:
    """Run AI inference in background thread - never blocks display"""
    
    def __init__(self, model, device, device_type, use_fp16, transform, threshold, 
                 proc_w, proc_h, grid_rows, grid_cols, cell_w, cell_h, cell_area):
        self.model = model
        self.device = device
        self.device_type = device_type
        self.use_fp16 = use_fp16
        self.transform = transform
        self.threshold = threshold
        self.proc_w = proc_w
        self.proc_h = proc_h
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.cell_area = cell_area
        
        # Input queue (frames to process)
        self.input_queue = Queue(maxsize=2)
        
        # Shared results (thread-safe)
        self.lock = threading.Lock()
        self.points = []
        self.count = 0
        self.density = 0.0
        self.cell_count = 0
        self.hotspot_loc = (0, 0)
        self.max_density = 0.0
        self.inference_time_ms = 0
        self.processed_count = 0
        
        self.stopped = False
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
    
    def submit(self, frame):
        """Submit frame for processing (non-blocking, drops if busy)"""
        try:
            self.input_queue.put_nowait(frame)
        except Full:
            pass  # Skip if inference is busy
    
    def get_results(self):
        """Get latest detection results (thread-safe)"""
        with self.lock:
            return {
                'points': self.points.copy(),
                'count': self.count,
                'density': self.density,
                'cell_count': self.cell_count,
                'hotspot_loc': self.hotspot_loc,
                'max_density': self.max_density,
                'inference_ms': self.inference_time_ms,
                'processed': self.processed_count
            }
    
    def _inference_loop(self):
        """Background inference loop"""
        while not self.stopped:
            try:
                frame = self.input_queue.get(timeout=0.1)
            except Empty:
                continue
            
            start = default_timer()
            
            # Preprocess
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0)
            
            # Run inference
            if self.device_type != "cpu":
                tensor = tensor.to(self.device)
            if self.use_fp16:
                tensor = tensor.half()
            
            with torch.inference_mode():
                out = self.model(tensor)
            
            # Postprocess
            scores = torch.softmax(out['pred_logits'], -1)[:, :, 1][0]
            pts = out['pred_points'][0]
            
            mask = scores > self.threshold
            points = pts[mask].detach().cpu().numpy()
            count = len(points)
            
            # Grid calculation
            grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.int32)
            for p in points:
                c = min(int(p[0]) // self.cell_w, self.grid_cols - 1)
                r = min(int(p[1]) // self.cell_h, self.grid_rows - 1)
                grid[r, c] += 1
            
            max_idx = np.unravel_index(grid.argmax(), grid.shape)
            max_cell = grid[max_idx]
            density = max_cell / self.cell_area
            
            inference_time = (default_timer() - start) * 1000
            
            # Update shared results
            with self.lock:
                self.points = points.tolist()
                self.count = count
                self.density = density
                self.cell_count = max_cell
                self.hotspot_loc = max_idx
                if density > self.max_density:
                    self.max_density = density
                self.inference_time_ms = inference_time
                self.processed_count += 1
    
    def stop(self):
        self.stopped = True


# ============================================
# ASYNC VIDEO READER
# ============================================
class AsyncVideoReader:
    """Async video reader with large buffer"""
    
    def __init__(self, video_path, buffer_size=120):
        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        self.buffer = Queue(maxsize=buffer_size)
        self.stopped = False
        self.frame_idx = 0
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()
    
    def _reader(self):
        while not self.stopped:
            if self.buffer.full():
                time.sleep(0.001)
                continue
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                break
            self.frame_idx += 1
            try:
                self.buffer.put_nowait((self.frame_idx, frame))
            except Full:
                pass
    
    def read(self):
        try:
            return self.buffer.get(timeout=0.5)
        except Empty:
            return None, None
    
    def release(self):
        self.stopped = True
        self.cap.release()


# ============================================
# ARGUMENTS
# ============================================
def get_args_parser():
    parser = argparse.ArgumentParser('P2PNet Smooth', add_help=False)
    parser.add_argument('--backbone', default='vgg16_bn')
    parser.add_argument('--row', default=2, type=int)
    parser.add_argument('--line', default=2, type=int)
    parser.add_argument('--output_dir', default='./output')
    parser.add_argument('--weight_path', default='./weights/SHTechA.pth')
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--shape', default=[640, 480], nargs='+', type=int)
    parser.add_argument('--frame_skip', default=6, type=int)
    parser.add_argument('--max_resolution', default=480, type=int)
    parser.add_argument('--startup_delay', default=3, type=int)
    parser.add_argument('--save_video', action='store_true', default=True)
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--video_path', default='')
    parser.add_argument('--device', default='auto', help='Device: auto, cuda, directml, cpu')
    return parser


# ============================================
# MAIN
# ============================================
def main(args):
    print("\n" + "="*60)
    print("PlexIE P2PNet - Smooth Edition v3 (Async Inference)")
    print("="*60)
    
    # Setup device
    device, device_type, use_fp16, onnx_providers = setup_device(args.gpu_id)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("\n[Model] Loading P2PNet...")
    model = build_model(args)
    
    if not os.path.exists(args.weight_path):
        print(f"[ERROR] Weights not found: {args.weight_path}")
        return
    
    checkpoint = torch.load(args.weight_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Move model to device
    if device_type != "cpu":
        model = model.to(device)
    print(f"[Model] Loaded on {device_type.upper()}")
    
    # Video setup
    if not args.video or not args.video_path:
        print("[ERROR] Provide --video --video_path")
        return
    
    reader = AsyncVideoReader(args.video_path, buffer_size=120)
    total_frames = reader.total_frames
    original_fps = reader.fps
    
    print(f"[Video] {total_frames} frames @ {original_fps:.1f} FPS")
    
    # Resolution
    proc_w, proc_h = args.shape
    if args.max_resolution > 0:
        scale = args.max_resolution / max(proc_w, proc_h)
        if scale < 1:
            proc_w = (int(proc_w * scale) // 16) * 16
            proc_h = (int(proc_h * scale) // 16) * 16
    
    print(f"[Process] {proc_w}x{proc_h}, inference every {args.frame_skip} frames")
    
    # Transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Grid config
    GRID_ROWS, GRID_COLS = 4, 4
    cell_w = proc_w // GRID_COLS
    cell_h = proc_h // GRID_ROWS
    cell_area = (5.0 / GRID_COLS) * (12.0 / GRID_ROWS)
    
    # Create async inference engine
    inference = AsyncInference(
        model=model,
        device=device,
        device_type=device_type,
        use_fp16=use_fp16,
        transform=transform,
        threshold=args.threshold,
        proc_w=proc_w,
        proc_h=proc_h,
        grid_rows=GRID_ROWS,
        grid_cols=GRID_COLS,
        cell_w=cell_w,
        cell_h=cell_h,
        cell_area=cell_area
    )
    
    # Window setup
    window_name = 'PlexIE Smooth v3 - Press Q to quit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    # Warmup
    print("\n[Warmup] Running test inference...")
    dummy_frame = np.zeros((proc_h, proc_w, 3), dtype=np.uint8)
    for _ in range(3):
        inference.submit(dummy_frame)
        time.sleep(0.1)
    
    # Wait for buffer to fill
    print("[Buffer] Filling...")
    for i in range(args.startup_delay, 0, -1):
        buf = reader.buffer.qsize()
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(blank, f"Starting in {i}...", (450, 300), cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255), 2)
        cv2.putText(blank, f"Buffer: {buf}/120 frames", (470, 380), cv2.FONT_HERSHEY_DUPLEX, 0.8, (150,150,150), 1)
        cv2.putText(blank, f"Device: {device_type.upper()}", (510, 420), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0), 1)
        cv2.imshow(window_name, blank)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            reader.release()
            inference.stop()
            cv2.destroyAllWindows()
            return
    
    while reader.buffer.qsize() < 60 and not reader.stopped:
        time.sleep(0.05)
    print(f"[Buffer] {reader.buffer.qsize()} frames ready")
    
    # Disable GC
    gc.disable()
    
    print("\n[Playing] Smooth 30 FPS with async detection...")
    
    # State
    frame_count = 0
    output_frames = [] if args.save_video else None
    
    start_time = default_timer()
    frame_delay_ms = int(1000 / original_fps)
    
    # Main display loop - NEVER BLOCKED by inference!
    while True:
        loop_start = default_timer()
        
        # Get frame
        result = reader.read()
        if result[0] is None:
            break
        
        frame_idx, frame = result
        frame_count = frame_idx
        
        # Resize for processing
        frame_resized = cv2.resize(frame, (proc_w, proc_h))
        
        # Submit every Nth frame for inference (non-blocking!)
        if frame_count % args.frame_skip == 1:
            inference.submit(frame_resized.copy())
        
        # Get latest detection results (non-blocking!)
        det = inference.get_results()
        
        # Draw visualization
        vis = frame_resized.copy()
        
        # Grid lines
        for i in range(1, GRID_ROWS):
            cv2.line(vis, (0, i*cell_h), (proc_w, i*cell_h), (50,50,50), 1)
        for j in range(1, GRID_COLS):
            cv2.line(vis, (j*cell_w, 0), (j*cell_w, proc_h), (50,50,50), 1)
        
        # Hotspot
        if det['cell_count'] > 0:
            r, c = det['hotspot_loc']
            cv2.rectangle(vis, (c*cell_w, r*cell_h), ((c+1)*cell_w, (r+1)*cell_h), (0,0,255), 2)
        
        # Points
        for p in det['points']:
            cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0,255,0), -1)
        
        # Stats overlay
        cv2.putText(vis, f"People: {det['count']}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(vis, f"Density: {det['density']:.1f} p/m2", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.putText(vis, f"Inference: {det['inference_ms']:.0f}ms ({device_type})", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(vis, f"Frame: {frame_count}/{total_frames}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
        
        # Display
        display = cv2.resize(vis, (1280, 720))
        cv2.imshow(window_name, display)
        
        # Save for video
        if output_frames is not None:
            output_frames.append(vis.copy())
        
        # Frame rate control - wait to match video FPS
        processing_time_ms = int((default_timer() - loop_start) * 1000)
        wait_time = max(1, frame_delay_ms - processing_time_ms)
        
        key = cv2.waitKey(wait_time) & 0xFF
        if key == ord('q'):
            print("\n[Interrupted by user]")
            break
        
        # Progress
        if frame_count % 100 == 0:
            pct = 100 * frame_count / total_frames
            print(f"[{pct:5.1f}%] Frame {frame_count} | People: {det['count']} | Inf: {det['inference_ms']:.0f}ms | Buf: {reader.buffer.qsize()}")
    
    total_time = default_timer() - start_time
    
    # Cleanup
    gc.enable()
    inference.stop()
    reader.release()
    cv2.destroyAllWindows()
    
    # Get final stats
    final = inference.get_results()
    
    # Save video
    if output_frames and len(output_frames) > 0:
        print(f"\n[Video] Saving {len(output_frames)} frames...")
        out_fps = int(original_fps)
        h, w = output_frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(
            os.path.join(args.output_dir, 'output.avi'),
            fourcc, out_fps, (w, h)
        )
        for f in output_frames:
            writer.write(f)
        writer.release()
        print(f"[Video] Saved: output.avi")
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Time: {total_time:.1f}s | Frames: {frame_count}")
    print(f"Inferences: {final['processed']} | Avg: {final['inference_ms']:.0f}ms")
    print(f"Max density: {final['max_density']:.2f} p/m²")
    print(f"Device: {device_type.upper()}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet Smooth', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
