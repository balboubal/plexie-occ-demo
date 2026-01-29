"""
PlexIE OCC Multi-Camera Backend v7.3
=====================================
Frame-based detection: Every N frames, run AI detection.

How it works:
1. Each camera reads video at native FPS
2. Every N frames (default 30 = ~1 FPS at 30fps video), detection runs
3. Detection is triggered from the camera's read loop (no polling)
4. Results sent to frontend via WebSocket
5. Frontend shows video, overlays update every ~1 second
"""

import asyncio
import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# Import P2PNet model
import sys
sys.path.insert(0, os.path.dirname(__file__))
from models import build_model

# ============================================
# Configuration
# ============================================
MAX_CAMERAS = 5
DETECTION_FRAME_SKIP = 30  # Process every N frames (30 = ~1 sec at 30fps)
PROCESS_WIDTH = 480
PROCESS_HEIGHT = 352       # Must be divisible by 16
GRID_ROWS = 4
GRID_COLS = 4
DETECTION_THRESHOLD = 0.5

# ============================================
# Data Classes
# ============================================
@dataclass
class Detection:
    camera_id: int
    timestamp: float
    frame_idx: int  # Which frame this detection is for
    count: int
    density: float
    points: List[List[float]]
    hotspot: Dict
    grid_counts: List[List[int]]
    inference_ms: float

# ============================================
# GPU Detection
# ============================================
def setup_device():
    """Auto-detect best device with bulletproof fallback to CPU"""
    print("\n[Device Detection]")
    device = torch.device('cpu')
    device_type = "cpu"
    
    # Check CUDA first (NVIDIA)
    try:
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            device_type = "cuda"
            print(f"  OK NVIDIA CUDA: {torch.cuda.get_device_name(0)}")
            return device, device_type
        else:
            print("  CUDA: not available")
    except Exception as e:
        print(f"  CUDA check error: {e}")
    
    # Try DirectML for AMD GPUs (Windows) - with extreme caution
    print("  Checking DirectML...")
    try:
        # Import in a try block
        torch_directml = None
        try:
            import torch_directml as tdml
            torch_directml = tdml
            print("    torch_directml module loaded")
        except ImportError:
            print("    torch_directml not installed")
        except Exception as e:
            print(f"    torch_directml import error: {e}")
        
        if torch_directml is not None:
            dml_device = None
            
            # Try multiple approaches to get device
            methods = [
                ("device()", lambda: torch_directml.device()),
                ("device(0)", lambda: torch_directml.device(0)),
            ]
            
            for name, method in methods:
                if dml_device is not None:
                    break
                try:
                    dml_device = method()
                    print(f"    {name} succeeded")
                except Exception as e:
                    print(f"    {name} failed: {type(e).__name__}")
            
            # Test device if we got one
            if dml_device is not None:
                try:
                    test = torch.tensor([1.0, 2.0]).to(dml_device)
                    _ = (test * 2).cpu()
                    device = dml_device
                    device_type = "directml"
                    print(f"  OK Using AMD DirectML")
                    return device, device_type
                except Exception as e:
                    print(f"    DirectML test failed: {e}")
                    
    except Exception as e:
        print(f"  DirectML error: {type(e).__name__}: {e}")
    
    # Fallback to CPU (guaranteed to work)
    print("  -> Using CPU (this will be slower but works)")
    return device, device_type

# ============================================
# Inference Engine (Thread-Safe)
# ============================================
class InferenceEngine:
    """Shared P2PNet model with thread-safe inference"""
    
    def __init__(self, weight_path: str, device, device_type: str):
        self.device = device
        self.device_type = device_type
        self.lock = threading.Lock()  # For thread-safe inference
        
        # Build model
        class Args:
            backbone = 'vgg16_bn'
            row = 2
            line = 2
        
        print("\n[Model] Loading P2PNet...")
        self.model = build_model(Args())
        # Load weights (handle different PyTorch versions)
        try:
            checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
        except TypeError:
            # Older PyTorch versions don't have weights_only parameter
            checkpoint = torch.load(weight_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        
        if device_type != "cpu":
            self.model = self.model.to(device)
        
        # Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        # Grid config
        self.cell_w = PROCESS_WIDTH // GRID_COLS
        self.cell_h = PROCESS_HEIGHT // GRID_ROWS
        self.cell_area = (5.0 / GRID_COLS) * (12.0 / GRID_ROWS)
        
        # Warmup
        print("[Model] Warming up...")
        dummy = torch.randn(1, 3, PROCESS_HEIGHT, PROCESS_WIDTH)
        if device_type != "cpu":
            dummy = dummy.to(device)
        with torch.inference_mode():
            for _ in range(3):
                self.model(dummy)
        print("[Model] Ready!\n")
    
    def process(self, camera_id: int, frame: np.ndarray, frame_idx: int) -> Detection:
        """Run inference on a frame (thread-safe)"""
        start = time.time()
        
        # Resize
        frame_resized = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        
        # Preprocess
        rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0)
        
        # Thread-safe inference (model can only process one at a time on GPU)
        with self.lock:
            if self.device_type != "cpu":
                tensor = tensor.to(self.device)
            
            with torch.inference_mode():
                out = self.model(tensor)
            
            # Get results while still holding lock
            scores = torch.softmax(out['pred_logits'], -1)[:, :, 1][0]
            pts = out['pred_points'][0]
            
            mask = scores > DETECTION_THRESHOLD
            points = pts[mask].detach().cpu().numpy()
        
        count = len(points)
        
        # Grid density (doesn't need GPU)
        grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
        for p in points:
            c = min(int(p[0]) // self.cell_w, GRID_COLS - 1)
            r = min(int(p[1]) // self.cell_h, GRID_ROWS - 1)
            grid[r, c] += 1
        
        max_idx = np.unravel_index(grid.argmax(), grid.shape)
        max_cell = int(grid[max_idx])
        density = max_cell / self.cell_area
        
        inference_ms = (time.time() - start) * 1000
        
        return Detection(
            camera_id=camera_id,
            timestamp=time.time(),
            frame_idx=frame_idx,
            count=count,
            density=round(density, 2),
            points=[[round(float(p[0]), 1), round(float(p[1]), 1)] for p in points],
            hotspot={
                'row': int(max_idx[0]),
                'col': int(max_idx[1]),
                'count': max_cell,
                'density': round(density, 2)
            },
            grid_counts=grid.tolist(),
            inference_ms=round(inference_ms, 1)
        )

# ============================================
# Simple Camera with frame-based detection
# ============================================
class SimpleCamera:
    """Camera that triggers detection every N frames"""
    
    def __init__(self, camera_id: int, video_path: str, on_detection_frame=None):
        self.camera_id = camera_id
        self.video_path = video_path
        self.on_detection_frame = on_detection_frame  # Callback when detection should run
        self.cap = cv2.VideoCapture(video_path)
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Frame tracking
        self.frame_idx = 0
        self.active = True
        self.lock = threading.Lock()
        
        # Frame skip (detect every N frames)
        self.frame_skip = DETECTION_FRAME_SKIP
        
        # Start reader thread
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        
        print(f"[Camera {camera_id}] Started: {Path(video_path).name} "
              f"({self.fps:.1f} fps, detect every {self.frame_skip} frames)")
    
    def _read_loop(self):
        """Read frames at video FPS, trigger detection every N frames"""
        frame_interval = 1.0 / self.fps
        
        while self.active:
            loop_start = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                with self.lock:
                    self.frame_idx = 0
                continue
            
            with self.lock:
                self.frame_idx += 1
                current_frame = self.frame_idx
            
            # Trigger detection every N frames
            if current_frame % self.frame_skip == 0 and self.on_detection_frame:
                self.on_detection_frame(self.camera_id, frame.copy(), current_frame)
            
            # Maintain video FPS timing
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def reset(self):
        """Reset camera to beginning"""
        with self.lock:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_idx = 0
    
    def get_video_time(self) -> float:
        """Get current playback position in seconds"""
        with self.lock:
            return self.frame_idx / self.fps
    
    def stop(self):
        self.active = False
        self.cap.release()

# ============================================
# Camera Manager
# ============================================
class CameraManager:
    """Manages cameras with callback-based detection"""
    
    def __init__(self, video_dir: str, weight_path: str):
        self.video_dir = Path(video_dir)
        self.cameras: Dict[int, SimpleCamera] = {}
        self.latest_detections: Dict[int, Detection] = {}
        
        # Setup device and model
        self.device, self.device_type = setup_device()
        self.engine = InferenceEngine(weight_path, self.device, self.device_type)
        
        # WebSocket clients
        self.clients: List[WebSocket] = []
        
        # Event loop for async broadcasts
        self.loop = None
    
    def get_videos(self) -> List[str]:
        """Get available video files"""
        videos = []
        for ext in ['*.mp4', '*.avi', '*.mkv', '*.mov']:
            videos.extend(self.video_dir.glob(ext))
        return sorted([v.name for v in videos])
    
    def _on_detection_frame(self, camera_id: int, frame: np.ndarray, frame_idx: int):
        """Callback when camera has a frame ready for detection"""
        try:
            detection = self.engine.process(camera_id, frame, frame_idx)
            if detection:
                self.latest_detections[camera_id] = detection
                self._broadcast_sync(detection)
        except Exception as e:
            print(f"[Camera {camera_id}] Detection error: {e}")
    
    def start_camera(self, camera_id: int, video_name: str) -> bool:
        """Start a camera"""
        if camera_id in self.cameras:
            self.stop_camera(camera_id)
        
        video_path = self.video_dir / video_name
        if not video_path.exists():
            return False
        
        # Pass detection callback to camera
        self.cameras[camera_id] = SimpleCamera(
            camera_id, 
            str(video_path), 
            on_detection_frame=self._on_detection_frame
        )
        return True
    
    def stop_camera(self, camera_id: int):
        """Stop a camera"""
        if camera_id in self.cameras:
            self.cameras[camera_id].stop()
            del self.cameras[camera_id]
            if camera_id in self.latest_detections:
                del self.latest_detections[camera_id]
            print(f"[Camera {camera_id}] Stopped")
    
    def reset_all(self):
        """Reset all cameras and detections"""
        for cam in self.cameras.values():
            cam.reset()
        self.latest_detections.clear()
        print("[Pipeline] Reset all cameras")
    
    def _broadcast_sync(self, detection: Detection):
        """Synchronously broadcast detection to WebSocket clients"""
        if not self.clients:
            return
        
        message = json.dumps({
            'type': 'detection',
            'data': asdict(detection)
        })
        
        # Use asyncio to send but handle from sync context
        loop = None
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        async def send_all():
            disconnected = []
            for client in self.clients:
                try:
                    await client.send_text(message)
                except:
                    disconnected.append(client)
            for client in disconnected:
                if client in self.clients:
                    self.clients.remove(client)
        
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(send_all(), loop)
        else:
            loop.run_until_complete(send_all())
    
    def get_status(self) -> Dict:
        """Get system status"""
        cameras = []
        for cam_id, cam in self.cameras.items():
            cameras.append({
                'camera_id': cam_id,
                'video': Path(cam.video_path).name,
                'fps': cam.fps,
                'total_frames': cam.total_frames,
                'current_frame': cam.frame_idx,
                'video_time': round(cam.get_video_time(), 1)
            })
        
        return {
            'device': self.device_type,
            'processing': self.running,
            'frame_skip': DETECTION_FRAME_SKIP,
            'cameras': cameras,
            'latest_detections': {str(k): asdict(v) for k, v in self.latest_detections.items()}
        }

# ============================================
# FastAPI App
# ============================================
app = FastAPI(title="PlexIE OCC Backend v7.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

manager: Optional[CameraManager] = None

@app.on_event("startup")
async def startup():
    global manager
    
    print("\n" + "="*60)
    print("  PlexIE OCC Multi-Camera Backend v7.3")
    print(f"  Frame-based detection: Every {DETECTION_FRAME_SKIP} frames")
    print("="*60)
    
    video_dir = Path(__file__).parent.parent / "public" / "datasets"
    weight_path = Path(__file__).parent / "weights" / "SHTechA.pth"
    
    if not weight_path.exists():
        print(f"\n[ERROR] Weights not found: {weight_path}")
        print("  Please ensure weights/SHTechA.pth exists")
        return
    
    if not video_dir.exists():
        print(f"\n[ERROR] Video directory not found: {video_dir}")
        return
    
    try:
        manager = CameraManager(str(video_dir), str(weight_path))
        print(f"\n[Server] Ready! Detection every {DETECTION_FRAME_SKIP} frames")
        print("="*60 + "\n")
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\n[Server] Running in degraded mode - no AI processing")
        manager = None

@app.on_event("shutdown")
async def shutdown():
    global manager
    if manager:
        for cam_id in list(manager.cameras.keys()):
            manager.stop_camera(cam_id)

# Global video directory for fallback
_video_dir = Path(__file__).parent.parent / "public" / "datasets"

@app.get("/api/videos")
async def get_videos():
    if manager:
        return {"videos": manager.get_videos()}
    # Fallback: scan directory directly
    try:
        videos = []
        for ext in ['*.mp4', '*.avi', '*.mkv', '*.mov']:
            videos.extend(_video_dir.glob(ext))
        return {"videos": sorted([v.name for v in videos])}
    except Exception as e:
        return {"videos": [], "error": str(e)}

@app.get("/api/status")
async def get_status():
    if manager:
        return manager.get_status()
    # Fallback status
    return {
        'device': 'none',
        'processing': False,
        'frame_skip': DETECTION_FRAME_SKIP,
        'cameras': [],
        'latest_detections': {},
        'error': 'Backend not fully initialized - check console for errors'
    }

@app.get("/api/config")
async def get_config():
    """Get timing configuration for frontend sync"""
    return {
        'frame_skip': DETECTION_FRAME_SKIP,
        'process_width': PROCESS_WIDTH,
        'process_height': PROCESS_HEIGHT,
        'grid_rows': GRID_ROWS,
        'grid_cols': GRID_COLS
    }

@app.post("/api/reset")
async def reset_all():
    """Reset all cameras to beginning"""
    if not manager:
        return JSONResponse({"error": "Not initialized"}, status_code=500)
    manager.reset_all()
    return {"status": "reset"}

@app.post("/api/camera/{camera_id}/start")
async def start_camera(camera_id: int, video: str):
    if not manager:
        return JSONResponse({"error": "Not initialized"}, status_code=500)
    if camera_id < 0 or camera_id >= MAX_CAMERAS:
        return JSONResponse({"error": f"Invalid camera_id (0-{MAX_CAMERAS-1})"}, status_code=400)
    
    success = manager.start_camera(camera_id, video)
    if success:
        return {"status": "started", "camera_id": camera_id, "video": video}
    return JSONResponse({"error": "Failed to start camera"}, status_code=400)

@app.post("/api/camera/{camera_id}/stop")
async def stop_camera(camera_id: int):
    if not manager:
        return JSONResponse({"error": "Not initialized"}, status_code=500)
    manager.stop_camera(camera_id)
    return {"status": "stopped", "camera_id": camera_id}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    if manager:
        manager.clients.append(websocket)
        print(f"[WebSocket] Client connected ({len(manager.clients)} total)")
        
        # Send current config
        await websocket.send_json({
            'type': 'config',
            'data': {
                'frame_skip': DETECTION_FRAME_SKIP
            }
        })
    
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        if manager and websocket in manager.clients:
            manager.clients.remove(websocket)
            print(f"[WebSocket] Client disconnected ({len(manager.clients)} total)")

@app.get("/videos/{video_name}")
async def serve_video(video_name: str):
    video_path = Path(__file__).parent.parent / "public" / "datasets" / video_name
    if video_path.exists():
        return FileResponse(video_path, media_type="video/mp4")
    return JSONResponse({"error": "Video not found"}, status_code=404)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
