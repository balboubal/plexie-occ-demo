import { useState, useEffect, useRef } from 'react'

// Video configuration - list your videos here
const AVAILABLE_VIDEOS = [
  'video1.mp4',
  'video2.mp4',
  'video3.mp4',
  'video4.mp4',
  'video5.mp4'
]

// Detection grid configuration
const GRID_ROWS = 4
const GRID_COLS = 4

// ============================================
// Camera Feed Component
// ============================================
function CameraFeed({ cameraId, videoName, plexieEnabled, onStop }) {
  const annotatedVideoRef = useRef(null)
  const cleanVideoRef = useRef(null)
  const canvasRef = useRef(null)
  const [isPlaying, setIsPlaying] = useState(false)

  // Start both videos when component mounts or video changes
  useEffect(() => {
    if (!videoName) return

    const annotatedVideo = annotatedVideoRef.current
    const cleanVideo = cleanVideoRef.current

    if (annotatedVideo && cleanVideo) {
      // Play both videos
      Promise.all([
        annotatedVideo.play().catch(e => console.log('Annotated play error:', e)),
        cleanVideo.play().catch(e => console.log('Clean play error:', e))
      ]).then(() => {
        setIsPlaying(true)
      })
    }
  }, [videoName])

  // Draw overlay on canvas
  useEffect(() => {
    const canvas = canvasRef.current
    const video = plexieEnabled ? annotatedVideoRef.current : cleanVideoRef.current
    if (!canvas || !video) return

    const ctx = canvas.getContext('2d')
    const rect = video.getBoundingClientRect()
    
    canvas.width = rect.width
    canvas.height = rect.height
    
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    if (!isPlaying) {
      // Show loading message
      ctx.fillStyle = 'rgba(0,0,0,0.5)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.fillStyle = '#00ff00'
      ctx.font = 'bold 14px monospace'
      ctx.textAlign = 'center'
      ctx.fillText('Loading...', canvas.width/2, canvas.height/2)
      return
    }
    
    // Only show overlay when PlexIE is ON
    if (plexieEnabled) {
      const cellW = canvas.width / GRID_COLS
      const cellH = canvas.height / GRID_ROWS
      
      // Draw subtle grid
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
      
      // Draw sample detection info
      ctx.fillStyle = 'rgba(0,255,0,0.8)'
      ctx.font = 'bold 12px monospace'
      ctx.textAlign = 'left'
      ctx.fillText('PlexIE Active', 10, 25)
    }
    
  }, [isPlaying, plexieEnabled])

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      if (canvasRef.current) {
        const video = plexieEnabled ? annotatedVideoRef.current : cleanVideoRef.current
        if (video) {
          const rect = video.getBoundingClientRect()
          canvasRef.current.width = rect.width
          canvasRef.current.height = rect.height
        }
      }
    }
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [plexieEnabled])

  // Construct video paths
  const annotatedPath = videoName ? `/videos/annotated/${videoName}` : null
  const cleanPath = videoName ? `/videos/clean/${videoName}` : null

  return (
    <div className="bg-gray-900 rounded-lg overflow-hidden shadow-xl border border-gray-700">
      {/* Header */}
      <div className="bg-gray-800 px-3 py-2 flex justify-between items-center">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${isPlaying ? 'bg-green-500 animate-pulse' : 'bg-yellow-500'}`} />
          <span className="text-white font-bold">CAM {cameraId + 1}</span>
          <span className="text-gray-500 text-xs">{videoName || 'No video'}</span>
        </div>
        <div className="flex gap-3 text-xs font-mono">
          <span className={`${plexieEnabled ? 'text-green-400' : 'text-gray-500'}`}>
            {plexieEnabled ? 'PlexIE ON' : 'PlexIE OFF'}
          </span>
        </div>
      </div>
      
      {/* Video Container with both videos */}
      <div className="relative bg-black" style={{ aspectRatio: '16/10' }}>
        {annotatedPath && cleanPath ? (
          <>
            {/* Annotated Video - visible when PlexIE is ON */}
            <video
              ref={annotatedVideoRef}
              src={annotatedPath}
              className={`w-full h-full object-contain absolute inset-0 transition-opacity duration-300 ${
                plexieEnabled ? 'opacity-100 z-10' : 'opacity-0 z-0'
              }`}
              loop
              muted
              playsInline
            />
            
            {/* Clean Video - visible when PlexIE is OFF */}
            <video
              ref={cleanVideoRef}
              src={cleanPath}
              className={`w-full h-full object-contain absolute inset-0 transition-opacity duration-300 ${
                !plexieEnabled ? 'opacity-100 z-10' : 'opacity-0 z-0'
              }`}
              loop
              muted
              playsInline
            />
            
            {/* Canvas overlay */}
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full pointer-events-none z-20"
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
          {isPlaying ? 'Playing' : 'Stopped'}
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
// Video Selector Modal
// ============================================
function VideoSelector({ videos, onSelect, onClose }) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full border border-gray-700">
        <h3 className="text-xl font-bold text-white mb-4">Select Video</h3>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {videos.map(video => (
            <button
              key={video}
              onClick={() => onSelect(video)}
              className="w-full px-4 py-3 bg-gray-700 hover:bg-gray-600 rounded text-white text-left transition-colors"
            >
              {video}
            </button>
          ))}
        </div>
        <button
          onClick={onClose}
          className="mt-4 w-full px-4 py-2 bg-red-600 hover:bg-red-500 rounded text-white font-bold"
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
  const [cameras, setCameras] = useState({}) // { cameraId: videoName }
  const [plexieEnabled, setPlexieEnabled] = useState(true)
  const [selectingFor, setSelectingFor] = useState(null)
  const [resetKey, setResetKey] = useState(0)

  // Start camera with video
  const startCamera = (cameraId, videoName) => {
    setCameras(prev => ({ ...prev, [cameraId]: videoName }))
  }

  // Stop camera
  const stopCamera = (cameraId) => {
    setCameras(prev => {
      const next = { ...prev }
      delete next[cameraId]
      return next
    })
  }

  // Auto-start all cameras with available videos
  const autoStartAll = () => {
    const newCameras = {}
    for (let i = 0; i < Math.min(5, AVAILABLE_VIDEOS.length); i++) {
      newCameras[i] = AVAILABLE_VIDEOS[i]
    }
    setCameras(newCameras)
  }

  // Reset all cameras
  const resetAll = () => {
    setCameras({})
    setResetKey(prev => prev + 1)  // Force video remount
  }

  const activeCameraCount = Object.keys(cameras).length

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
              <p className="text-xs text-gray-400">Operations Command Center v8.0</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <button
              onClick={() => setPlexieEnabled(!plexieEnabled)}
              className={`px-4 py-2.5 rounded-lg text-sm font-bold transition-all shadow-lg flex items-center gap-2 ${
                plexieEnabled 
                  ? 'bg-green-600 hover:bg-green-500' 
                  : 'bg-gray-600 hover:bg-gray-500'
              }`}
            >
              <span className={`w-3 h-3 rounded-full ${plexieEnabled ? 'bg-green-300' : 'bg-gray-400'}`} />
              PlexIE {plexieEnabled ? 'ON' : 'OFF'}
            </button>
            <button
              onClick={resetAll}
              disabled={activeCameraCount === 0}
              className="px-4 py-2.5 bg-yellow-600 hover:bg-yellow-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg text-sm font-bold transition-colors shadow-lg"
            >
              ↻ Reset
            </button>
            <button
              onClick={autoStartAll}
              disabled={AVAILABLE_VIDEOS.length === 0}
              className="px-5 py-2.5 bg-green-600 hover:bg-green-500 disabled:bg-gray-700 disabled:text-gray-500 rounded-lg text-sm font-bold transition-colors shadow-lg"
            >
              ▶ Start All Cameras
            </button>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="p-4 pb-16">
        {/* Stats Panel */}
        <div className="mb-4 bg-gray-800 rounded-lg p-4 border border-gray-700">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-bold text-white flex items-center gap-2">
              <span className="text-green-500">●</span> PlexIE OCC Dashboard
            </h2>
            <div className="flex items-center gap-3">
              <span className="text-xs text-gray-500 bg-gray-700 px-2 py-1 rounded">
                Frontend Only Mode
              </span>
              <div className={`px-2 py-1 rounded text-xs font-bold ${plexieEnabled ? 'bg-green-900 text-green-400' : 'bg-gray-700 text-gray-400'}`}>
                {plexieEnabled ? 'PlexIE ACTIVE' : 'PlexIE INACTIVE'}
              </div>
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-3">
            <div className="bg-gray-900 rounded-lg p-3 text-center border border-gray-700">
              <div className="text-3xl font-bold text-green-400">{activeCameraCount}</div>
              <div className="text-xs text-gray-500 mt-1">Active Cameras</div>
            </div>
            <div className="bg-gray-900 rounded-lg p-3 text-center border border-gray-700">
              <div className="text-3xl font-bold text-blue-400">{AVAILABLE_VIDEOS.length}</div>
              <div className="text-xs text-gray-500 mt-1">Available Videos</div>
            </div>
            <div className="bg-gray-900 rounded-lg p-3 text-center border border-gray-700">
              <div className={`text-3xl font-bold ${plexieEnabled ? 'text-green-400' : 'text-gray-500'}`}>
                {plexieEnabled ? 'ON' : 'OFF'}
              </div>
              <div className="text-xs text-gray-500 mt-1">PlexIE Status</div>
            </div>
          </div>
        </div>
        
        {/* Camera Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[0, 1, 2, 3, 4].map(cameraId => (
            <CameraFeed
              key={`${cameraId}-${resetKey}`}
              cameraId={cameraId}
              videoName={cameras[cameraId]}
              plexieEnabled={plexieEnabled}
              onStop={stopCamera}
            />
          ))}
          
          {/* Add camera button for empty slots */}
          {activeCameraCount < 5 && (
            <div 
              onClick={() => {
                // Find first available camera slot
                const nextSlot = [0,1,2,3,4].find(id => !cameras[id])
                if (nextSlot !== undefined) setSelectingFor(nextSlot)
              }}
              className="bg-gray-800 rounded-lg border-2 border-dashed border-gray-600 hover:border-green-500 cursor-pointer transition-colors flex items-center justify-center"
              style={{ aspectRatio: '16/10' }}
            >
              <div className="text-center">
                <div className="text-4xl text-gray-500 mb-2">+</div>
                <div className="text-gray-400">Add Camera</div>
              </div>
            </div>
          )}
        </div>
        
        {/* Instructions */}
        {activeCameraCount === 0 && (
          <div className="mt-6 bg-blue-900/30 border border-blue-500 rounded-lg p-4">
            <h3 className="text-blue-400 font-bold mb-2">🔹 Getting Started</h3>
            <p className="text-gray-300 text-sm mb-2">
              Click <strong>"Start All Cameras"</strong> to load all available videos, or click <strong>"+ Add Camera"</strong> to select individual videos.
            </p>
            <p className="text-gray-300 text-sm">
              Toggle <strong>"PlexIE ON/OFF"</strong> to switch between annotated and clean video versions.
            </p>
          </div>
        )}
      </main>
      
      {/* Footer */}
      <footer className="fixed bottom-0 left-0 right-0 bg-gray-800 border-t border-gray-700 px-4 py-2">
        <div className="flex justify-between items-center text-xs text-gray-500">
          <span>PlexIE OCC v8.0 - Frontend Only Mode</span>
          <span>
            Active: {activeCameraCount}/5 | 
            Mode: {plexieEnabled ? 'Annotated' : 'Clean'} Videos
          </span>
        </div>
      </footer>
      
      {/* Video Selector */}
      {selectingFor !== null && (
        <VideoSelector
          videos={AVAILABLE_VIDEOS}
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
