import { useState, useEffect, useRef, useCallback } from 'react'

// Configuration - must match backend
const API_URL = 'http://localhost:8000'
const WS_URL = 'ws://localhost:8000/ws'
const GRID_ROWS = 4
const GRID_COLS = 4
const PROCESS_WIDTH = 480
const PROCESS_HEIGHT = 352

// ============================================
// Camera Feed Component - Simple approach
// ============================================
function CameraFeed({ cameraId, videoName, detection, onStop }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [isPlaying, setIsPlaying] = useState(false)

  // Handle video ready to play
  const handleCanPlay = () => {
    const video = videoRef.current
    if (!video || isPlaying) return
    video.play()
    setIsPlaying(true)
  }

  // Draw detection overlay
  useEffect(() => {
    const canvas = canvasRef.current
    const video = videoRef.current
    if (!canvas || !video) return

    const ctx = canvas.getContext('2d')
    const rect = video.getBoundingClientRect()
    
    canvas.width = rect.width
    canvas.height = rect.height
    
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    if (!detection || !isPlaying) {
      // Show waiting message
      ctx.fillStyle = 'rgba(0,0,0,0.5)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.fillStyle = '#00ff00'
      ctx.font = 'bold 14px monospace'
      ctx.textAlign = 'center'
      ctx.fillText(isPlaying ? 'Waiting for detection...' : 'Loading...', canvas.width/2, canvas.height/2)
      return
    }
    
    // Scale factors
    const scaleX = canvas.width / PROCESS_WIDTH
    const scaleY = canvas.height / PROCESS_HEIGHT
    const cellW = canvas.width / GRID_COLS
    const cellH = canvas.height / GRID_ROWS
    
    // Draw grid (subtle)
    ctx.strokeStyle = 'rgba(0,255,0,0.2)'
    ctx.lineWidth = 1
    for (let i = 1; i < GRID_ROWS; i++) {
      ctx.beginPath()
      ctx.moveTo(0, i * cellH)
      ctx.lineTo(canvas.width, i * cellH)
      ctx.stroke()
    }
    for (let j = 1; j < GRID_COLS; j++) {
      ctx.beginPath()
      ctx.moveTo(j * cellW, 0)
      ctx.lineTo(j * cellW, canvas.height)
      ctx.stroke()
    }
    
    // Draw density heatmap
    if (detection.grid_counts) {
      const maxCount = Math.max(...detection.grid_counts.flat(), 1)
      for (let r = 0; r < GRID_ROWS; r++) {
        for (let c = 0; c < GRID_COLS; c++) {
          const count = detection.grid_counts[r][c]
          if (count > 0) {
            const intensity = count / maxCount
            // Green to yellow to red gradient
            const r_color = Math.min(255, intensity * 2 * 255)
            const g_color = Math.min(255, (1 - intensity) * 2 * 255)
            ctx.fillStyle = `rgba(${r_color},${g_color},0,${intensity * 0.3})`
            ctx.fillRect(c * cellW + 1, r * cellH + 1, cellW - 2, cellH - 2)
          }
        }
      }
    }
    
    // Draw hotspot highlight
    if (detection.hotspot && detection.hotspot.count > 0) {
      const { row, col, density } = detection.hotspot
      ctx.strokeStyle = density > 5 ? '#ff0000' : density > 3 ? '#ffaa00' : '#00ff00'
      ctx.lineWidth = 3
      ctx.strokeRect(col * cellW + 2, row * cellH + 2, cellW - 4, cellH - 4)
    }
    
    // Draw detection points (green dots)
    ctx.fillStyle = '#00ff00'
    ctx.shadowColor = '#00ff00'
    ctx.shadowBlur = 8
    for (const pt of detection.points || []) {
      const x = pt[0] * scaleX
      const y = pt[1] * scaleY
      ctx.beginPath()
      ctx.arc(x, y, 5, 0, Math.PI * 2)
      ctx.fill()
    }
    ctx.shadowBlur = 0
    
  }, [detection, isPlaying])

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (canvasRef.current && videoRef.current) {
        const rect = videoRef.current.getBoundingClientRect()
        canvasRef.current.width = rect.width
        canvasRef.current.height = rect.height
      }
    }
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  const densityLevel = detection?.density > 5 ? 'critical' : 
                       detection?.density > 3 ? 'warning' : 'normal'
  const densityColor = {
    critical: 'text-red-500 animate-pulse',
    warning: 'text-yellow-400',
    normal: 'text-green-400'
  }[densityLevel]

  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden shadow-xl border border-gray-700">
      {/* Header */}
      <div className="bg-gray-800 px-3 py-2 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isPlaying ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'}`} />
          <span className="text-white font-bold">CAM {cameraId + 1}</span>
          <span className="text-gray-500 text-xs">{videoName}</span>
        </div>
        {detection && (
          <div className="flex gap-3 text-xs font-mono">
            <span className="text-green-400">{detection.count} <span className="text-gray-500">ppl</span></span>
            <span className={densityColor}>{detection.density} <span className="text-gray-500">p/m2</span></span>
            <span className="text-blue-400">{detection.inference_ms}<span className="text-gray-500">ms</span></span>
          </div>
        )}
      </div>
      
      {/* Video + Canvas */}
      <div className="relative aspect-video bg-black">
        {videoName ? (
          <>
            <video
              ref={videoRef}
              src={`${API_URL}/videos/${videoName}`}
              className="w-full h-full object-cover"
              loop
              muted
              playsInline
              onCanPlay={handleCanPlay}
            />
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none"
            />
          </>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500">
            No video assigned
          </div>
        )}
      </div>
      
      {/* Footer */}
      <div className="bg-gray-800 px-3 py-1.5 flex justify-between items-center text-xs">
        <div className="text-gray-500">
          {isPlaying ? 'Live' : 'Loading...'}
          {detection && ` | Frame ${detection.frame_idx}`}
        </div>
        {videoName && (
          <button 
            onClick={() => onStop(cameraId)}
            className="text-red-400 hover:text-red-300 font-medium"
          >
            Stop
          </button>
        )}
      </div>
    </div>
  )
}

// ============================================
// Stats Panel
// ============================================
function StatsPanel({ detections, deviceType, isConnected }) {
  const detectionsArray = Object.values(detections)
  const totalPeople = detectionsArray.reduce((sum, d) => sum + (d?.count || 0), 0)
  const maxDensity = Math.max(...detectionsArray.map(d => d?.density || 0), 0)
  const avgInference = detectionsArray.length > 0 
    ? detectionsArray.reduce((sum, d) => sum + (d?.inference_ms || 0), 0) / detectionsArray.length
    : 0
  const activeCameras = detectionsArray.filter(d => d).length

  // Find camera with highest density
  const hotCamera = detectionsArray.reduce((max, d) => {
    if (!d) return max
    if (!max || d.density > max.density) return d
    return max
  }, null)

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-bold text-white flex items-center gap-2">
          <span className="text-green-500">●</span> PlexIE OCC Dashboard
        </h2>
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-500 bg-gray-700 px-2 py-1 rounded">
            1 FPS detection
          </span>
          <div className={`px-2 py-1 rounded text-xs font-bold ${isConnected ? 'bg-green-900 text-green-400' : 'bg-red-900 text-red-400'}`}>
            {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
          </div>
        </div>
      </div>
      
      <div className="grid grid-cols-5 gap-3 mb-4">
        <div className="bg-gray-900 rounded-lg p-3 text-center border border-gray-700">
          <div className="text-3xl font-bold text-green-400">{totalPeople}</div>
          <div className="text-xs text-gray-500 mt-1">Total People</div>
        </div>
        <div className="bg-gray-900 rounded-lg p-3 text-center border border-gray-700">
          <div className={`text-3xl font-bold ${maxDensity > 5 ? 'text-red-500 animate-pulse' : maxDensity > 3 ? 'text-yellow-400' : 'text-green-400'}`}>
            {maxDensity.toFixed(1)}
          </div>
          <div className="text-xs text-gray-500 mt-1">Max Density</div>
        </div>
        <div className="bg-gray-900 rounded-lg p-3 text-center border border-gray-700">
          <div className="text-3xl font-bold text-blue-400">{avgInference.toFixed(0)}</div>
          <div className="text-xs text-gray-500 mt-1">Avg Latency (ms)</div>
        </div>
        <div className="bg-gray-900 rounded-lg p-3 text-center border border-gray-700">
          <div className="text-3xl font-bold text-purple-400">{activeCameras}/5</div>
          <div className="text-xs text-gray-500 mt-1">Active Cams</div>
        </div>
        <div className="bg-gray-900 rounded-lg p-3 text-center border border-gray-700">
          <div className="text-2xl font-bold text-cyan-400 uppercase">{deviceType || '...'}</div>
          <div className="text-xs text-gray-500 mt-1">AI Device</div>
        </div>
      </div>
      
      {/* Alert for high density */}
      {maxDensity > 4 && hotCamera && (
        <div className={`p-3 rounded-lg border ${maxDensity > 5 ? 'bg-red-900/40 border-red-500' : 'bg-yellow-900/40 border-yellow-500'}`}>
          <div className="flex items-center gap-3">
            <span className="text-3xl">{maxDensity > 5 ? '🚨' : '⚠️'}</span>
            <div>
              <div className={`font-bold text-lg ${maxDensity > 5 ? 'text-red-400' : 'text-yellow-400'}`}>
                {maxDensity > 5 ? 'CRITICAL: Crowd Crush Risk!' : 'Warning: Elevated Density'}
              </div>
              <div className="text-sm text-gray-300">
                Camera {hotCamera.camera_id + 1}: {hotCamera.density} p/m² | {hotCamera.count} people detected
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================
// Video Selector Modal
// ============================================
function VideoSelector({ videos, onSelect, onClose }) {
  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 w-96 max-h-[80vh] overflow-auto border border-gray-600">
        <h3 className="text-xl font-bold text-white mb-4">Select Video Feed</h3>
        <div className="space-y-2">
          {videos.map(video => (
            <button
              key={video}
              onClick={() => onSelect(video)}
              className="w-full text-left px-4 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg text-white transition-colors"
            >
              📹 {video}
            </button>
          ))}
        </div>
        <button 
          onClick={onClose}
          className="mt-4 w-full py-2 bg-gray-700 hover:bg-gray-600 rounded text-gray-300"
        >
          Cancel
        </button>
      </div>
    </div>
  )
}

// ============================================
// Main App
// ============================================
function App() {
  const [connected, setConnected] = useState(false)
  const [deviceType, setDeviceType] = useState(null)
  const [videos, setVideos] = useState([])
  const [cameras, setCameras] = useState({})
  const [detections, setDetections] = useState({})
  const [selectingFor, setSelectingFor] = useState(null)
  const [resetKey, setResetKey] = useState(0)
  const wsRef = useRef(null)

  // Fetch videos
  const fetchVideos = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/videos`)
      const data = await res.json()
      setVideos(data.videos || [])
    } catch (e) {
      console.error('Failed to fetch videos:', e)
    }
  }, [])

  // Fetch status
  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/status`)
      const data = await res.json()
      setDeviceType(data.device)
      
      // Sync camera states
      const cams = {}
      for (const cam of data.cameras || []) {
        cams[cam.camera_id] = cam.video
      }
      setCameras(cams)
    } catch (e) {
      console.error('Failed to fetch status:', e)
    }
  }, [])

  // WebSocket connection
  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    
    const ws = new WebSocket(WS_URL)
    
    ws.onopen = () => {
      console.log('[WS] Connected')
      setConnected(true)
    }
    
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'detection' && msg.data) {
          setDetections(prev => ({
            ...prev,
            [msg.data.camera_id]: msg.data
          }))
        }
      } catch (err) {
        console.error('[WS] Parse error:', err)
      }
    }
    
    ws.onclose = () => {
      console.log('[WS] Disconnected')
      setConnected(false)
      wsRef.current = null
      setTimeout(connectWS, 2000)
    }
    
    ws.onerror = (e) => {
      console.error('[WS] Error:', e)
      ws.close()
    }
    
    wsRef.current = ws
  }, [])

  // Start camera
  const startCamera = async (cameraId, videoName) => {
    try {
      const res = await fetch(`${API_URL}/api/camera/${cameraId}/start?video=${videoName}`, {
        method: 'POST'
      })
      if (res.ok) {
        setCameras(prev => ({ ...prev, [cameraId]: videoName }))
      }
    } catch (e) {
      console.error('Failed to start camera:', e)
    }
  }

  // Stop camera
  const stopCamera = async (cameraId) => {
    try {
      await fetch(`${API_URL}/api/camera/${cameraId}/stop`, { method: 'POST' })
      setCameras(prev => {
        const next = { ...prev }
        delete next[cameraId]
        return next
      })
      setDetections(prev => {
        const next = { ...prev }
        delete next[cameraId]
        return next
      })
    } catch (e) {
      console.error('Failed to stop camera:', e)
    }
  }

  // Auto-start all cameras
  const autoStartAll = async () => {
    for (let i = 0; i < Math.min(5, videos.length); i++) {
      await startCamera(i, videos[i])
      // Small delay between starts
      await new Promise(r => setTimeout(r, 200))
    }
  }

  // Reset all cameras
  const resetAll = async () => {
    try {
      await fetch(`${API_URL}/api/reset`, { method: 'POST' })
      setDetections({})
      setResetKey(prev => prev + 1)  // Force video remount
    } catch (e) {
      console.error('Failed to reset:', e)
    }
  }

  // Initialize
  useEffect(() => {
    fetchVideos()
    fetchStatus()
    connectWS()
    
    return () => wsRef.current?.close()
  }, [fetchVideos, fetchStatus, connectWS])

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 bg-green-600 rounded-lg flex items-center justify-center font-bold text-2xl shadow-lg shadow-green-600/30">
              P
            </div>
            <div>
              <h1 className="text-2xl font-bold">PlexIE OCC</h1>
              <p className="text-xs text-gray-400">Operations Command Center v7.3</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <button
              onClick={resetAll}
              disabled={Object.keys(cameras).length === 0}
              className="px-4 py-2.5 bg-yellow-600 hover:bg-yellow-500 disabled:bg-gray-600 rounded-lg text-sm font-bold transition-colors shadow-lg"
            >
              ↻ Reset
            </button>
            <button
              onClick={autoStartAll}
              disabled={videos.length === 0}
              className="px-5 py-2.5 bg-green-600 hover:bg-green-500 disabled:bg-gray-600 rounded-lg text-sm font-bold transition-colors shadow-lg"
            >
              ▶ Start All Cameras
            </button>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="p-4 pb-16">
        {/* Stats */}
        <div className="mb-4">
          <StatsPanel 
            detections={detections} 
            deviceType={deviceType}
            isConnected={connected}
          />
        </div>
        
        {/* Camera Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[0, 1, 2, 3, 4].map(cameraId => (
            <CameraFeed
              key={`${cameraId}-${resetKey}`}
              cameraId={cameraId}
              videoName={cameras[cameraId]}
              detection={detections[cameraId]}
              onStop={stopCamera}
            />
          ))}
          
          {/* Add camera button for empty slots */}
          {Object.keys(cameras).length < 5 && (
            <div 
              onClick={() => setSelectingFor(Object.keys(cameras).length)}
              className="bg-gray-800 rounded-lg border-2 border-dashed border-gray-600 hover:border-green-500 cursor-pointer transition-colors flex items-center justify-center aspect-video"
            >
              <div className="text-center">
                <div className="text-4xl text-gray-500 mb-2">+</div>
                <div className="text-gray-400">Add Camera</div>
              </div>
            </div>
          )}
        </div>
        
        {/* Backend not connected */}
        {!connected && !deviceType && (
          <div className="mt-6 bg-red-900/30 border border-red-500 rounded-lg p-4">
            <h3 className="text-red-400 font-bold mb-2">⚠️ Backend Not Connected</h3>
            <p className="text-gray-300 text-sm mb-2">Start the backend server:</p>
            <code className="bg-gray-800 px-3 py-2 rounded text-green-400 text-sm block font-mono">
              cd backend && python server_multicam.py
            </code>
          </div>
        )}
      </main>
      
      {/* Footer */}
      <footer className="fixed bottom-0 left-0 right-0 bg-gray-800 border-t border-gray-700 px-4 py-2">
        <div className="flex justify-between items-center text-xs text-gray-500">
          <span>PlexIE OCC v7.3 - SSIG Stadium Safety System</span>
          <span>
            Cameras: {Object.keys(cameras).length}/5 | 
            Device: {deviceType?.toUpperCase() || '...'} | 
            Pipeline: 1 FPS detection
          </span>
        </div>
      </footer>
      
      {/* Video Selector */}
      {selectingFor !== null && (
        <VideoSelector
          videos={videos}
          onSelect={(video) => {
            startCamera(selectingFor, video)
            setSelectingFor(null)
          }}
          onClose={() => setSelectingFor(null)}
        />
      )}
    </div>
  )
}

export default App
