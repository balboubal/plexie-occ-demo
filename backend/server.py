#!/usr/bin/env python3
"""
PlexIE AI Backend v4.0
======================
Uses point-based detection for accurate crowd counting (like P2PNet).
Extracts head center points from YOLO detections.

Key features:
1. Point-based head detection (green dots, like P2PNet)
2. Grid-based density calculation with hotspot detection
3. Only runs when client sends "start" command
4. No simulation - AI only
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import threading

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("WARNING: ultralytics not installed. Run: pip install ultralytics")
    YOLO_AVAILABLE = False

# ============================================
# CONFIGURATION  
# ============================================
CONFIG = {
    "port": 8000,
    "host": "0.0.0.0",
    "model": "yolo11n.pt",  # YOLOv11 nano
    "confidence_threshold": 0.3,
    "detection_fps": 8,
    "grid_rows": 6,
    "grid_cols": 8,
    "datasets_dir": Path(__file__).parent.parent / "public" / "datasets",
    # For density calculation (assumed venue dimensions in meters)
    "venue_width_m": 50.0,
    "venue_height_m": 30.0,
}

# ============================================
# FASTAPI APP
# ============================================
app = FastAPI(title="PlexIE AI Backend", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# CROWD DETECTOR (YOLO with head point extraction)
# ============================================
class CrowdDetector:
    """
    Detects people and extracts head points.
    Uses YOLO for detection, then extracts head center from each bbox.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance.model = None
                    cls._instance.loaded = False
        return cls._instance
    
    def load(self):
        if self.loaded:
            return True
        if not YOLO_AVAILABLE:
            print("[Model] YOLO not available - using simulation")
            return False
        
        print(f"[Model] Loading {CONFIG['model']}...")
        try:
            self.model = YOLO(CONFIG["model"])
            self.loaded = True
            print(f"[Model] Loaded successfully")
            return True
        except Exception as e:
            print(f"[Model] Load failed: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> tuple:
        """
        Detect people and return:
        - points: head center points (like P2PNet)
        - boxes: bounding boxes for visualization
        """
        if not self.loaded or self.model is None:
            return [], []  # No simulation - return empty if no model
        
        try:
            # Run YOLO detection
            results = self.model(
                frame,
                verbose=False,
                classes=[0],  # Person class only
                conf=CONFIG["confidence_threshold"],
                iou=0.4,
            )
            
            points = []
            boxes = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    # Store bounding box (for green box visualization)
                    boxes.append({
                        "x": float(x1),
                        "y": float(y1),
                        "w": float(x2 - x1),
                        "h": float(y2 - y1),
                        "conf": round(conf, 2),
                    })
                    
                    # Extract HEAD position - center point for density calc
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    
                    # Head center X: center of bbox
                    head_x = x1 + bbox_width / 2
                    # Head center Y: near top of bbox (for grid density)
                    head_y = y1 + bbox_height * 0.15
                    
                    points.append({
                        "x": float(head_x),
                        "y": float(head_y),
                        "conf": round(conf, 2),
                    })
            
            return points, boxes
            
        except Exception as e:
            print(f"[Model] Detection error: {e}")
            return [], []

detector = CrowdDetector()

# ============================================
# VIDEO PROCESSOR
# ============================================
class VideoProcessor:
    def __init__(self, video_path: Path, camera_id: str):
        self.video_path = video_path
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
        self.fps = 30
        self.width = 640
        self.height = 480
        self.total_frames = 0
        self.duration = 0
        
    def open(self) -> bool:
        if not self.video_path.exists():
            print(f"[{self.camera_id}] Video not found: {self.video_path}")
            return False
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            return False
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        print(f"[{self.camera_id}] Opened: {self.width}x{self.height} @ {self.fps:.1f}fps")
        return True
    
    def seek_to_time(self, time_sec: float):
        if self.cap:
            frame_num = int(time_sec * self.fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame_num, self.total_frames - 1)))
    
    def read_frame(self) -> Optional[np.ndarray]:
        if not self.cap:
            return None
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return frame if ret else None
    
    def get_current_time(self) -> float:
        if not self.cap:
            return 0
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES) / self.fps if self.fps > 0 else 0
    
    def close(self):
        if self.cap:
            self.cap.release()
            self.cap = None

# ============================================
# DENSITY CALCULATION (P2PNet style grid)
# ============================================
def calculate_grid_density(points: List[Dict], width: int, height: int) -> tuple:
    """
    Calculate density grid from head points (like P2PNet).
    Returns (density_grid, max_density, max_cell, max_count)
    """
    rows, cols = CONFIG["grid_rows"], CONFIG["grid_cols"]
    cell_w = width / cols
    cell_h = height / rows
    cell_area_m2 = (CONFIG["venue_width_m"] / cols) * (CONFIG["venue_height_m"] / rows)
    
    # Count people in each grid cell
    grid = [[0] * cols for _ in range(rows)]
    for p in points:
        col = min(int(p["x"] / cell_w), cols - 1)
        row = min(int(p["y"] / cell_h), rows - 1)
        if 0 <= row < rows and 0 <= col < cols:
            grid[row][col] += 1
    
    # Calculate density for each cell
    density_grid = [[0.0] * cols for _ in range(rows)]
    max_density = 0.0
    max_cell = (0, 0)
    max_count = 0
    
    for r in range(rows):
        for c in range(cols):
            count = grid[r][c]
            density = count / cell_area_m2
            density_grid[r][c] = round(density, 2)
            if density > max_density:
                max_density = density
                max_cell = (r, c)
                max_count = count
    
    return density_grid, max_density, max_cell, max_count

def calculate_metrics(points: List[Dict], density_grid: List[List[float]], max_density: float) -> Dict:
    count = len(points)
    all_densities = [d for row in density_grid for d in row]
    avg_density = sum(all_densities) / len(all_densities) if all_densities else 0
    
    # Simulated environmental
    base_temp = 26 + (count / 50)
    temp = round(base_temp + np.sin(time.time() * 0.1) * 1.5, 1)
    co2 = int(420 + count * 3 + np.random.normal(0, 5))
    
    return {
        "count": count,
        "density": round(avg_density, 2),
        "maxDensity": round(max_density, 2),
        "temp": temp,
        "co2": co2,
    }

# ============================================
# API ROUTES
# ============================================
@app.on_event("startup")
async def startup():
    detector.load()

@app.get("/api/status")
async def get_status():
    return {
        "status": "running",
        "version": "4.0.0",
        "model_loaded": detector.loaded,
        "model_name": CONFIG["model"],
        "yolo_available": YOLO_AVAILABLE,
    }

@app.get("/api/cameras")
async def get_cameras():
    cameras = []
    for i in range(1, 6):
        cam_id = f"cam{i}"
        video_path = CONFIG["datasets_dir"] / f"{cam_id}.mp4"
        cameras.append({
            "id": cam_id,
            "name": f"Camera {i}",
            "video": f"/datasets/{cam_id}.mp4",
            "available": video_path.exists(),
        })
    return {"cameras": cameras}

@app.websocket("/ws/stream/{camera_id}")
async def websocket_stream(websocket: WebSocket, camera_id: str):
    await websocket.accept()
    print(f"[{camera_id}] Client connected")
    
    video_path = CONFIG["datasets_dir"] / f"{camera_id}.mp4"
    processor = VideoProcessor(video_path, camera_id)
    
    if not processor.open():
        await websocket.send_json({"type": "error", "error": f"Video not found"})
        await websocket.close()
        return
    
    await websocket.send_json({
        "type": "init",
        "camera": camera_id,
        "width": processor.width,
        "height": processor.height,
        "fps": processor.fps,
        "duration": processor.duration,
        "modelLoaded": detector.loaded,
    })
    
    running = False
    detection_interval = 1.0 / CONFIG["detection_fps"]
    last_detection = 0
    
    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_json(), timeout=0.05)
                cmd = msg.get("command", "")
                if cmd == "start":
                    running = True
                    processor.seek_to_time(0)
                    print(f"[{camera_id}] Started")
                elif cmd == "stop":
                    running = False
                    print(f"[{camera_id}] Stopped")
            except asyncio.TimeoutError:
                pass
            
            if not running:
                await asyncio.sleep(0.1)
                continue
            
            current_time = time.time()
            if current_time - last_detection < detection_interval:
                await asyncio.sleep(0.02)
                continue
            
            last_detection = current_time
            frame = processor.read_frame()
            if frame is None:
                continue
            
            # Detect - returns (points, boxes)
            points, boxes = detector.detect(frame)
            
            # Calculate density using head points
            density_grid, max_density, max_cell, max_count = calculate_grid_density(
                points, processor.width, processor.height
            )
            metrics = calculate_metrics(points, density_grid, max_density)
            
            await websocket.send_json({
                "type": "detection",
                "camera": camera_id,
                "timestamp": current_time,
                "videoTime": processor.get_current_time(),
                "width": processor.width,
                "height": processor.height,
                "points": points,  # Head center points (for dots)
                "boxes": boxes,    # Bounding boxes (for green rectangles)
                "densityGrid": density_grid,
                "hotspot": {
                    "row": max_cell[0],
                    "col": max_cell[1],
                    "density": round(max_density, 2),
                    "count": max_count,
                },
                "fromAI": detector.loaded,
                **metrics,
            })
            
    except WebSocketDisconnect:
        print(f"[{camera_id}] Disconnected")
    except Exception as e:
        print(f"[{camera_id}] Error: {e}")
    finally:
        processor.close()

if __name__ == "__main__":
    print("=" * 50)
    print("PlexIE AI Backend v4.0")
    print("=" * 50)
    print(f"Model: {CONFIG['model']}")
    print(f"YOLO: {YOLO_AVAILABLE}")
    print(f"Grid: {CONFIG['grid_rows']}x{CONFIG['grid_cols']}")
    print(f"Server: http://localhost:{CONFIG['port']}")
    print("=" * 50)
    
    uvicorn.run(app, host=CONFIG["host"], port=CONFIG["port"], log_level="info")
