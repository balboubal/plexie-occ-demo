import { useState, useEffect, useRef, useCallback } from 'react'

const VIDEOS = ['video1.mp4', 'video2.mp4', 'video3.mp4', 'video4.mp4', 'video5.mp4']

const ZONES = ['North Stand', 'North East', 'South Stand', 'East Wing', 'West Wing']

const HVAC_UNITS = [
  { name: 'North', baseTemp: 32 },
  { name: 'South', baseTemp: 33 },
  { name: 'East', baseTemp: 31 },
  { name: 'West', baseTemp: 31 },
]

const GATES = ['North Entry', 'North Exit', 'South Entry', 'South Exit', 'East Emergency', 'West Emergency']

const SEVERITY_RANK = {
  'NORMAL': 1,
  'ELEVATED': 2,
  'HIGH': 3,
  'CRITICAL': 4
}

function getSeverity(density) {
  if (density >= 6) return { label: 'CRITICAL', color: 'red' }
  if (density >= 5) return { label: 'HIGH', color: 'orange' }
  if (density >= 4) return { label: 'ELEVATED', color: 'yellow' }
  return { label: 'NORMAL', color: 'blue' }
}

/* --------- Safe play helper (defensive) ---------
   Wraps video.play() and returns a promise. This avoids unhandled rejections
   (autoplay blocks etc.). We use it wherever we call play().
*/
const safePlay = (videoEl) => {
  if (!videoEl) return Promise.reject(new Error('no video element'))
  try {
    const p = videoEl.play()
    if (p && typeof p.then === 'function') {
      return p.catch((err) => {
        // bubble rejection to caller
        return Promise.reject(err)
      })
    }
    return Promise.resolve()
  } catch (err) {
    return Promise.reject(err)
  }
}

/* --------- Camera component (unchanged structure; small guards + safePlay) --------- */
function Camera({ id, videoName, detectionData, plexieEnabled, onData, paused }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const [data, setData] = useState(null)
  const lastUpdate = useRef(0)

  useEffect(() => {
    if (!videoName || !detectionData?.frames) return
    const video = videoRef.current
    if (!video) return
    const fps = detectionData.video_info?.fps || 30
    let animId
    const loop = () => {
      if (video && !video.paused) {
        const frame = Math.floor(video.currentTime * fps) + 1
        let fd = detectionData.frames[String(frame)]
        if (!fd) {
          const keys = Object.keys(detectionData.frames).map(Number)
          if (keys.length) {
            const closest = keys.reduce((p, c) => Math.abs(c - frame) < Math.abs(p - frame) ? c : p)
            fd = detectionData.frames[String(closest)]
          }
        }
        if (fd) {
          const now = Date.now()
          if (now - lastUpdate.current >= 500) {
            lastUpdate.current = now
            setData(fd)
            try { onData(id, fd) } catch (e) { /* defensive: don't let onData break rendering */ }
          }
          try { drawOverlay(fd) } catch (e) { /* swallow overlay errors to avoid app crash */ }
        }
      }
      animId = requestAnimationFrame(loop)
    }
    animId = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(animId)
  }, [videoName, detectionData, id, onData, plexieEnabled])

  const drawOverlay = (det) => {
    const canvas = canvasRef.current, video = videoRef.current
    // Defensive guards: required metadata must exist
    if (!canvas || !video || !plexieEnabled || !detectionData?.grid_config) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const rect = video.getBoundingClientRect()
    canvas.width = rect.width; canvas.height = rect.height
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    const gc = detectionData.grid_config || {}
    const videoInfo = detectionData.video_info || {}
    const vw = video.videoWidth || videoInfo.process_width || rect.width
    const vh = video.videoHeight || videoInfo.process_height || rect.height
    if (!vw || !vh) return
    const va = vw / vh, ca = rect.width / rect.height
    let rw, rh, ox, oy
    if (ca > va) { rh = rect.height; rw = rh * va; ox = (rect.width - rw) / 2; oy = 0 }
    else { rw = rect.width; rh = rw / va; ox = 0; oy = (rect.height - rh) / 2 }
    const sx = rw / (videoInfo.process_width || vw), sy = rh / (videoInfo.process_height || vh)
    const cw = (gc.cell_width || 1) * sx, ch = (gc.cell_height || 1) * sy

    if (det.grid && gc.rows && gc.cols && gc.assumed_cell_area_m2) {
      for (let r = 0; r < gc.rows; r++) {
        for (let c = 0; c < gc.cols; c++) {
          const d = (det.grid?.[r]?.[c] || 0) / (gc.assumed_cell_area_m2 || 1)
          if (d >= 6) ctx.fillStyle = 'rgba(255,0,0,0.5)'
          else if (d >= 4) ctx.fillStyle = 'rgba(255,165,0,0.4)'
          else if (d >= 2) ctx.fillStyle = 'rgba(255,255,0,0.3)'
          else continue
          ctx.fillRect(ox + c * cw, oy + r * ch, cw, ch)
        }
      }
    }
    if (det.points) {
      ctx.fillStyle = '#0f0'
      det.points.forEach(p => { 
        if (p && p.length >= 2) ctx.beginPath(), ctx.arc(ox + p[0] * sx, oy + p[1] * sy, 3, 0, Math.PI * 2), ctx.fill()
      })
    }
    // small info box
    ctx.fillStyle = 'rgba(0,0,0,0.8)'; ctx.fillRect(ox + 5, oy + 5, 105, 38)
    ctx.font = 'bold 12px monospace'; ctx.fillStyle = '#0f0'; ctx.fillText('Count: ' + (det.count ?? 0), ox + 10, oy + 20)
    ctx.fillStyle = (det.max_density >= 6) ? '#f00' : (det.max_density >= 4 ? '#fa0' : '#0f0')
    ctx.fillText('Density: ' + (det.max_density ?? 0), ox + 10, oy + 36)
  }

  // Play when videoName changes (muted videos are usually allowed to autoplay)
  useEffect(() => { 
    const v = videoRef.current
    if (v && videoName) {
      safePlay(v).catch(() => { /* ignore; user will see paused video */ })
    }
  }, [videoName])

  // Respond to paused prop safely
  useEffect(() => {
    const v = videoRef.current
    if (!v || !videoName) return
    if (paused) {
      try { v.pause() } catch (e) { /* ignore */ }
    } else {
      safePlay(v).catch(() => { /* ignore */ })
    }
  }, [paused, videoName])

  let border = 'border border-gray-700'
  if (data && plexieEnabled && data.max_density >= 3) {
    if (data.max_density >= 6) border = 'border-4 border-red-500'
    else if (data.max_density >= 5) border = 'border-4 border-orange-500'
    else if (data.max_density >= 4) border = 'border-3 border-yellow-500'
    else border = 'border-2 border-blue-500'
  }

  const zoneName = ZONES[id] || `Camera ${id + 1}`
  
  return (
    <div className={`bg-gray-900 rounded-lg overflow-hidden ${border}`}>
      <div className="bg-gray-800 px-2 py-1 flex justify-between">
        <span className="text-white font-bold text-sm">{zoneName}</span>
        {data && <span className="text-xs text-gray-300">{data.count} ppl | {data.max_density} p/m²</span>}
      </div>
      <div className="relative bg-black" style={{ aspectRatio: '16/10' }}>
        {videoName ? (
          <>
            <video
              ref={videoRef}
              src={'/videos/clean/' + videoName}
              className="w-full h-full object-contain"
              loop
              muted
              playsInline
            />
            {plexieEnabled && <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />}
          </>
        ) : <div className="flex items-center justify-center h-full text-gray-600">No Feed</div>}
      </div>
    </div>
  )
}

/* -------------------- App -------------------- */
function App() {
  const [cameras, setCameras] = useState({})
  const [detectionData, setDetectionData] = useState({})
  const [plexieEnabled, setPlexieEnabled] = useState(true)
  const [loading, setLoading] = useState(false)
  const [resetKey, setResetKey] = useState(0)
  const [alerts, setAlerts] = useState([])
  const [hvac, setHvac] = useState(() => HVAC_UNITS.map(u => ({ temp: u.baseTemp, boost: 0, target: 0 })))
  const [aiHvac, setAiHvac] = useState(true)
  const [gates, setGates] = useState(() => GATES.map(() => false))
  const [paused, setPaused] = useState(false)
  const [started, setStarted] = useState(false)
  const [isMobile, setIsMobile] = useState(false)
  
  // Store latest camera data
  const camDataRef = useRef({})
  // Track last escalation time for each alert (1.5s cooldown before de-escalation)
  const alertCooldowns = useRef({})

  // Detect mobile screen size
  useEffect(() => {
    const mediaQuery = window.matchMedia('(max-width: 767px)')
    setIsMobile(mediaQuery.matches)
    
    const handler = (e) => setIsMobile(e.matches)
    mediaQuery.addEventListener('change', handler)
    return () => mediaQuery.removeEventListener('change', handler)
  }, [])

  // Generate smart actions based on conditions
  const generateActions = (camId, density, count, temp) => {
    const actions = []
    const zone = ZONES[camId]
    
    if (density >= 6) {
      actions.push({ label: '🚨 Emergency crowd control - ' + zone, done: false })
      actions.push({ label: '🚪 Open all emergency exits - ' + zone, done: false })
      actions.push({ label: '📢 Urgent dispersal announcement', done: false })
      actions.push({ label: '🏥 Medical team on standby', done: false })
    } else if (density >= 5) {
      actions.push({ label: '👮 Position security at ' + zone, done: false })
      actions.push({ label: '🚪 Pre-open auxiliary exits', done: false })
      actions.push({ label: '📢 Flow guidance announcement', done: false })
    } else if (density >= 4) {
      actions.push({ label: '👁️ Enhanced monitoring - ' + zone, done: false })
      actions.push({ label: '📢 Gentle crowd guidance', done: false })
    } else {
      actions.push({ label: '👁️ Continue monitoring ' + zone, done: false })
    }
    
    if (temp >= 30) {
      actions.push({ label: '❄️ Emergency cooling +30%', done: false, hvac: camId % 4, boost: 30 })
      actions.push({ label: '💧 Deploy hydration stations', done: false })
    } else if (temp >= 28) {
      actions.push({ label: '❄️ Increase cooling +20%', done: false, hvac: camId % 4, boost: 20 })
    } else if (temp >= 26 && density >= 3) {
      actions.push({ label: '❄️ Boost ventilation +15%', done: false, hvac: camId % 4, boost: 15 })
    }
    
    if (count > 35 && density >= 4) {
      actions.push({ label: '⚠️ FORECAST: May reach critical in ~3 min', done: false, info: true })
    }
    
    return actions
  }

  // Generate alert directly when camera sends data
  const handleCameraData = useCallback((camId, data) => {
    camDataRef.current[camId] = data
    
    // Don't update alerts when paused or AI is off
    if (paused || !plexieEnabled) return
    
    const hvacIdx = camId % 4
    
    if (data.max_density >= 2) {
      setAlerts(prev => {
        const temp = hvac[hvacIdx]?.temp || 28
        
        const idx = prev.findIndex(a => a.camId === camId)
        if (idx >= 0) {
          const updated = [...prev]
          const oldSeverity = updated[idx].severity.label
          const newSeverity = getSeverity(data.max_density)
          
          let severityChanged = oldSeverity !== newSeverity.label
          
          if (severityChanged) {
            const oldRank = SEVERITY_RANK[oldSeverity] || 0
            const newRank = SEVERITY_RANK[newSeverity.label] || 0
            const isDeEscalating = newRank < oldRank
            
            if (isDeEscalating) {
              const now = Date.now()
              const lastEscalation = alertCooldowns.current[camId] || 0
              const timeSinceEscalation = now - lastEscalation
              
              if (timeSinceEscalation < 2500) {
                severityChanged = false
              }
            } else {
              alertCooldowns.current[camId] = Date.now()
            }
          }
          
          updated[idx] = { 
            ...updated[idx], 
            density: data.max_density, 
            count: data.count,
            temp,
            severity: severityChanged ? newSeverity : updated[idx].severity,
            actions: severityChanged ? generateActions(camId, data.max_density, data.count, temp) : updated[idx].actions
          }
          return updated
        }
        if (prev.length >= 5) return prev
        return [...prev, {
          id: Date.now() + camId,
          camId,
          zone: ZONES[camId] || 'Zone ' + (camId + 1),
          density: data.max_density,
          count: data.count,
          temp,
          severity: getSeverity(data.max_density),
          actions: generateActions(camId, data.max_density, data.count, temp)
        }]
      })
    }
  }, [hvac, paused, plexieEnabled])

  useEffect(() => {
    if (!plexieEnabled) {
      setAlerts([])
      alertCooldowns.current = {}
    }
  }, [plexieEnabled])

  // HVAC simulation
  useEffect(() => {
    if (!started) return
    const interval = setInterval(() => {
      if (paused) return
      setHvac(prev => prev.map((h, i) => {
        let newBoost = h.boost
        if (h.boost < h.target) newBoost = Math.min(h.target, h.boost + 1)
        else if (h.boost > h.target) newBoost = Math.max(h.target, h.boost - 1)
        const cooling = newBoost * 0.18
        const newTemp = h.temp + (HVAC_UNITS[i].baseTemp - cooling - h.temp) * 0.08
        return { ...h, boost: newBoost, temp: newTemp }
      }))
    }, 300)
    return () => clearInterval(interval)
  }, [started, paused])

  // AI HVAC
  useEffect(() => {
    if (!aiHvac || !started) return
    const interval = setInterval(() => {
      if (paused) return
      setHvac(prev => prev.map((h, i) => {
        const camIds = i === 0 ? [0, 1] : i === 1 ? [2] : i === 2 ? [3] : [4]
        let maxD = 0
        camIds.forEach(c => { if (camDataRef.current[c]) maxD = Math.max(maxD, camDataRef.current[c].max_density) })
        let target = 10
        if (h.temp >= 30 || maxD >= 6) target = 50
        else if (h.temp >= 28 || maxD >= 5) target = 40
        else if (h.temp >= 26 || maxD >= 4) target = 30
        else if (maxD >= 3) target = 25
        return { ...h, target }
      }))
    }, 1500)
    return () => clearInterval(interval)
  }, [aiHvac, started, paused])

  const loadDet = async (name) => {
    try {
      const r = await fetch('/videos/detections/' + name.replace('.mp4', '') + '_detections.json')
      if (r.ok) { const d = await r.json(); setDetectionData(p => ({ ...p, [name]: d })); return true }
    } catch (e) {}
    return false
  }

  const startAll = async () => {
    setLoading(true)
    setStarted(true) // Mark system as started
    for (let i = 0; i < VIDEOS.length; i++) {
      setCameras(p => ({ ...p, [i]: VIDEOS[i] }))
      if (!detectionData[VIDEOS[i]]) await loadDet(VIDEOS[i])
    }
    setLoading(false)
  }

  const reset = () => {
    setCameras({}); camDataRef.current = {}; setAlerts([])
    alertCooldowns.current = {} // Clear cooldown timestamps
    setHvac(HVAC_UNITS.map(u => ({ temp: u.baseTemp, boost: 0, target: 0 })))
    setGates(GATES.map(() => false)); setResetKey(k => k + 1)
    setStarted(false) // Mark system as stopped
  }

  const approve = (alertId, actIdx) => {
    setAlerts(prev => prev.map(a => {
      if (a.id !== alertId) return a
      const acts = [...a.actions]
      const act = acts[actIdx]
      if (act.info) return a // Info items can't be approved
      acts[actIdx] = { ...act, done: true }
      if (act.hvac !== undefined) {
        const boostAmt = act.boost || 20
        setHvac(h => h.map((x, i) => i === act.hvac ? { ...x, target: Math.min(50, x.target + boostAmt) } : x))
      }
      return { ...a, actions: acts }
    }))
  }

  const approveAll = (alertId) => {
    setAlerts(prev => prev.map(a => {
      if (a.id !== alertId) return a
      const acts = a.actions.map(act => {
        if (act.done || act.info) return act
        if (act.hvac !== undefined) {
          const boostAmt = act.boost || 20
          setHvac(h => h.map((x, i) => i === act.hvac ? { ...x, target: Math.min(50, x.target + boostAmt) } : x))
        }
        return { ...act, done: true }
      })
      return { ...a, actions: acts }
    }))
  }

  const dismiss = (id) => {
    setAlerts(p => {
      const alert = p.find(a => a.id === id)
      if (alert) {
        delete alertCooldowns.current[alert.camId]
      }
      return p.filter(a => a.id !== id)
    })
  }

  const crit = alerts.filter(a => a.severity.label === 'CRITICAL').length

  /* Small CSS inserted to preserve desktop alerts-panel height but allow mobile flow.
     We add bottom padding to root to avoid the fixed footer overlapping content on md+.
  */
  return (
    <div className="min-h-screen bg-gray-900 text-white pb-16">
      <style>{`
        /* keep alerts panel full-height on desktop, flow on mobile */
        @media (min-width: 768px) {
          .alerts-panel { height: calc(100vh - 100px); }
          .alerts-panel .alerts-scroll { height: calc(100% - 44px); overflow-y: auto; }
        }
        @media (max-width: 767px) {
          .alerts-panel { height: auto; }
          .alerts-panel .alerts-scroll { height: auto; overflow: visible; }
        }
      `}</style>

      <header className="bg-gray-800 border-b border-gray-700 px-4 py-2 flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-green-600 rounded flex items-center justify-center font-bold">P</div>
          <div><h1 className="text-lg font-bold">PlexIE OCC</h1><p className="text-xs text-gray-400">v8.9</p></div>
        </div>
        <div className="flex items-center gap-2">
          {crit > 0 && <div className="px-3 py-1 bg-red-600 rounded text-sm font-bold animate-pulse">🚨 {crit}</div>}
          <button onClick={() => setPlexieEnabled(!plexieEnabled)} className={`px-3 py-1 rounded text-sm font-bold ${plexieEnabled ? 'bg-green-600' : 'bg-gray-600'}`}>AI {plexieEnabled ? 'ON' : 'OFF'}</button>
          <button onClick={() => setPaused(!paused)} className={`px-3 py-1 rounded text-sm font-bold ${paused ? 'bg-orange-600' : 'bg-purple-600'}`}>{paused ? '▶ Resume' : '⏸ Pause'}</button>
          <button onClick={reset} className="px-3 py-1 bg-yellow-600 rounded text-sm font-bold">Reset</button>
          <button onClick={startAll} disabled={loading} className="px-3 py-1 bg-blue-600 rounded text-sm font-bold">{loading ? '...' : '▶ Start'}</button>
        </div>
      </header>

      {/* Main layout: stack vertically on small screens, horizontal on md+ (keeps desktop exact layout) */}
      <main className="p-3 flex flex-col md:flex-row gap-3">
        <div className="flex-1 space-y-3">
          {/* camera grid: responsive columns (mobile 1 column, desktop 3) */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {[0,1,2,3,4].map(id => {
              // Skip North Stand (0) and West Wing (4) on mobile
              if (isMobile && (id === 0 || id === 4)) return null
              return (
                <Camera
                  key={`${id}-${resetKey}`}
                  id={id}
                  videoName={cameras[id]}
                  detectionData={cameras[id] ? detectionData[cameras[id]] : null}
                  plexieEnabled={plexieEnabled}
                  onData={handleCameraData}
                  paused={paused}
                />
              )
            })}
            <div className="bg-gray-800 rounded-lg border border-gray-700">
              <div className="bg-gray-700 px-3 py-2 flex justify-between items-center">
                <span className="font-bold text-sm">🌡️ HVAC</span>
                <button onClick={() => setAiHvac(!aiHvac)} className={`px-2 py-1 rounded text-xs font-bold ${aiHvac ? 'bg-green-600' : 'bg-gray-600'}`}>AI {aiHvac ? 'ON' : 'OFF'}</button>
              </div>
              <div className="p-2 space-y-2">
                {HVAC_UNITS.map((u, i) => {
                  // Skip West HVAC (index 3) on mobile
                  if (isMobile && i === 3) return null
                  const h = hvac[i], moving = Math.abs(h.boost - h.target) > 0.5
                  const tc = h.temp >= 29 ? 'text-red-400' : h.temp >= 26 ? 'text-yellow-400' : 'text-green-400'
                  return (
                    <div key={i} className="bg-gray-900 rounded p-2">
                      <div className="flex justify-between mb-1"><span className="text-xs font-bold">{u.name}</span><span className={`text-sm font-bold ${tc}`}>{h.temp.toFixed(1)}°C</span></div>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 relative h-3 bg-gray-700 rounded overflow-hidden">
                          <div className="absolute left-0 top-0 h-full bg-cyan-600 transition-all" style={{ width: `${h.boost * 2}%` }} />
                          {aiHvac && moving && <div className="absolute top-0 h-full w-1 bg-green-400" style={{ left: `${h.target * 2}%` }} />}
                        </div>
                        <span className="text-xs text-cyan-400 w-8">{Math.round(h.boost)}%</span>
                      </div>
                      {aiHvac && moving && <div className="text-xs text-green-400 mt-1">→ {h.target}%</div>}
                    </div>
                  )
                })}
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg border border-gray-700 p-3">
            <h3 className="font-bold text-sm mb-2">🚪 Gates</h3>
            <div className="grid grid-cols-3 gap-2">
              {GATES.map((n, i) => <button key={i} onClick={() => setGates(g => g.map((v, j) => j === i ? !v : v))} className={`px-2 py-2 rounded text-xs font-bold ${gates[i] ? 'bg-green-600' : 'bg-gray-700'}`}>{n}</button>)}
            </div>
          </div>
        </div>

        {/* ALERTS - make it full-width on mobile, fixed-width on desktop */}
        <div className="w-full md:w-[340px]">
          <div className="bg-gray-800 rounded-lg border border-gray-700 alerts-panel">
            <div className="bg-gray-700 px-3 py-2 flex justify-between items-center">
              <span className="font-bold text-sm">⚠️ Alerts ({alerts.length}/5)</span>
              {alerts.length > 0 && <button onClick={() => setAlerts([])} className="text-xs text-gray-400">Clear</button>}
            </div>
            <div className="p-2 alerts-scroll">
              {alerts.length === 0 ? (
                <div className="text-center text-gray-500 py-8"><div className="text-3xl mb-2">✓</div>No alerts<div className="text-xs mt-1">Click Start</div></div>
              ) : alerts
                .filter(a => !isMobile || (a.zone !== 'North Stand' && a.zone !== 'West Wing'))
                .map(a => {
                const bg = { red: 'bg-red-950', orange: 'bg-orange-950', yellow: 'bg-yellow-950', blue: 'bg-blue-950' }[a.severity.color]
                const bd = { red: 'border-red-500', orange: 'border-orange-500', yellow: 'border-yellow-500', blue: 'border-blue-500' }[a.severity.color]
                const badge = { red: 'bg-red-600', orange: 'bg-orange-600', yellow: 'bg-yellow-600', blue: 'bg-blue-600' }[a.severity.color]
                const pendingCount = a.actions.filter(x => !x.done && !x.info).length
                return (
                  <div key={a.id} className={`${bg} ${bd} border-2 rounded-lg mb-3`}>
                    <div className="p-3 border-b border-white/10 flex justify-between">
                      <div>
                        <span className={`${badge} text-white text-xs font-bold px-2 py-0.5 rounded mr-2`}>{a.severity.label}</span>
                        <span className="text-white font-bold text-lg">{a.zone}</span>
                        <div className="text-gray-300 text-sm mt-1">{a.density} p/m² • {a.count} people • {a.temp?.toFixed(1)}°C</div>
                      </div>
                      <button onClick={() => dismiss(a.id)} className="text-gray-400 hover:text-white text-xl">×</button>
                    </div>
                    <div className="p-3">
                      {a.actions.map((act, i) => (
                        <div key={i} className={`flex justify-between items-center py-2 px-3 mb-2 rounded ${
                          act.info ? 'bg-yellow-900/30 text-yellow-300 border border-yellow-600/50' :
                          act.done ? 'bg-green-900/50 text-green-400' : 'bg-black/30'
                        }`}>
                          <span className="text-sm">{act.label}</span>
                          {!act.done && !act.info && <button onClick={() => approve(a.id, i)} className="px-3 py-1 bg-blue-600 rounded text-sm font-bold">OK</button>}
                          {act.done && <span className="text-green-400">✓</span>}
                        </div>
                      ))}
                      {pendingCount > 0 && (
                        <button onClick={() => approveAll(a.id)} className="w-full mt-2 py-2 bg-green-600 hover:bg-green-500 rounded font-bold text-sm">
                          ✓ Approve All ({pendingCount})
                        </button>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        </div>
      </main>

      {/* Footer: fixed only on md+ so mobile can scroll vertically */}
      <footer className="md:fixed bottom-0 left-0 right-0 bg-gray-800 border-t border-gray-700 px-4 py-1 flex justify-between text-xs text-gray-500">
        <span>v8.9 • HVAC: {aiHvac ? 'AUTO' : 'MANUAL'}</span>
        <span>Alerts: {alerts.length}</span>
      </footer>
    </div>
  )
}

export default App
