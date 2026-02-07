# PlexIE Operations Command Center (OCC) Demo

**AI-Powered Venue Safety Platform Demo for SSIG Investor Presentations**

Version 3.0 | February 2026

## ⚠️ Important: Video Files Required

This demo requires video files to function properly. Place MP4 video files in:
```
public/datasets/
├── cam1.mp4
├── cam2.mp4
├── cam3.mp4
├── cam4.mp4
└── cam5.mp4
```

**Video Sources:**
- [Pexels Crowd Videos](https://www.pexels.com/search/videos/crowd/) (Free)
- [MOT17 Dataset](https://motchallenge.net/data/MOT17/) (Research)
- Any surveillance-style footage of crowds

---

## 🎯 What This Demo Shows

This interactive demonstration showcases PlexIE's core value proposition for Saudi Arabia's Vision 2030 mega-venues:

1. **Unified Command Picture** - All cameras, sensors, and alerts in one interface
2. **AI-Powered Detection** - Real-time crowd density, collapse detection, heat monitoring
3. **Intelligent Recommendations** - Context-aware action suggestions
4. **Human-in-the-Loop** - AI recommends, operators approve
5. **Staff Dispatch** - Visual response team routing with ETA tracking
6. **Environmental Control** - HVAC integration and optimization
7. **Post-Event Analytics** - ROI and performance metrics
8. **Live Narrator** - Auto-commentary explaining what's happening

---

## 🚀 Quick Start

### Option 1: Use the Launcher (Recommended)
```bash
# Windows
PlexIE-Launcher.bat

# macOS/Linux
./PlexIE-Launcher.sh
```
The launcher will:
- Check dependencies and install if needed
- Verify video files are present
- Start the development server
- Provide diagnostics if there are issues

### Option 2: Manual Start
```bash
cd plexie-occ-demo
npm install
npm run dev
```
Open http://localhost:5173

### Option 3: With AI Backend (Real YOLOv11 Detection)
```bash
# Terminal 1: Start AI Backend
cd plexie-occ-demo/backend
pip install -r requirements.txt
python server.py

# Terminal 2: Start Frontend
cd plexie-occ-demo
npm install
npm run dev
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play/Pause |
| `R` | Reset demo |
| `P` | Toggle PlexIE AI |
| `F` | Focus mode (fullscreen camera) |
| `H` | Toggle heatmap |
| `B` | Toggle bounding boxes |
| `1-5` | Switch camera feed |

---

## 🎮 Demo Guide

### Basic Controls
- **Start Demo** - Begin the simulation timeline
- **PlexIE AI: ON/OFF** - Toggle AI features (key demo contrast!)
- **Heatmap/Detections** - Toggle visualization overlays
- **Focus** - Fullscreen mode for clean presentations
- **Analytics** - View post-event performance summary
- **Reset** - Return to initial state

### Demo Scenarios
Click scenario buttons to trigger different situations:

| Scenario | What Happens | Key Takeaway |
|----------|-------------|--------------|
| **Normal Ops** | Baseline monitoring | System watches so operators don't have to |
| **Heat Emergency** | Zone F1 temperature rises | AI detects 10 min before collapse |
| **Medical Incident** | Collapse in Zone B2 | 2 min response vs 14 min without PlexIE |
| **Evacuation** | Exit flow optimization | Smart routing prevents bottlenecks |

### AI Assistant
Click "AI Assistant" to open the Claude-powered chat:
- Type "status" for full system overview
- Ask about crowd conditions
- Get analysis of alerts
- Request recommendations
- Learn about evacuation protocols

### Analytics Dashboard
Click "Analytics" to view post-event metrics:
- Demo statistics (duration, alerts, incidents)
- PlexIE impact analysis
- ROI summary for 15k venues

---

## 📊 Key Demo Points for Investors

### The Problem (Show with PlexIE OFF)
- Operators can only monitor 6-9 feeds effectively
- 14+ minute response times to incidents
- Fragmented systems = dangerous blind spots
- Manual coordination = confusion and delays
- ⚠️ **Warning banner shows when PlexIE is OFF during scenarios**

### The Solution (Show with PlexIE ON)
- AI monitors ALL feeds simultaneously
- 2-minute response times (within golden window)
- Unified command picture = complete awareness
- Smart recommendations = confident decisions
- Response Time widget shows 85% improvement

### The Numbers
- **70%** of stadium collapses are heat-related
- **7-10%** survival drop per minute of cardiac delay
- **6+ p/m²** = crush risk threshold
- **SAR 3-6M/year** value per 15k venue

---

## 🎬 Recommended Demo Flow (7 minutes)

1. **(0:00-0:30)** Start with PlexIE **OFF**, Normal scenario - baseline
2. **(0:30-1:30)** Switch to **Heat Emergency** - watch temp rise, no alerts
3. **(1:30-2:00)** Enable **PlexIE AI** - alerts appear, HVAC auto-adjusts
4. **(2:00-3:00)** Point out **Response Time widget** (14min → 2min)
5. **(3:00-4:00)** Switch to **Medical Incident** - collapse detected, dispatch team
6. **(4:00-5:00)** Open **AI Assistant** → type "status" for overview
7. **(5:00-5:30)** Show **Focus Mode** (F key) for clean presentation
8. **(5:30-6:30)** Open **Analytics** → show ROI and impact metrics
9. **(6:30-7:00)** Demo **Evacuation** scenario for exit optimization

---

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (5173)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ CCTV 1   │  │ CCTV 2   │  │ CCTV 3   │  │ CCTV 4   │    │
│  │ +Overlay │  │ +Overlay │  │ +Overlay │  │ +Overlay │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       └─────────────┴──────┬──────┴─────────────┘           │
│                            │                                 │
│  ┌─────────────────────────▼─────────────────────────────┐  │
│  │                  Alert Engine                          │  │
│  │   Density + Temp + CO₂ + AI Detection → Alerts        │  │
│  └────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │ WebSocket
┌───────────────────────────▼─────────────────────────────────┐
│                 Python AI Backend (8000)                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ FastAPI     │  │ YOLOv11     │  │ Density     │         │
│  │ WebSocket   │──│ Detection   │──│ Grid Calc   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

### Core Features
- ✅ 5 CCTV camera feeds with AI overlays
- ✅ Real-time density heatmaps
- ✅ Person detection bounding boxes
- ✅ Collapse/fall detection indicators
- ✅ Temperature and CO₂ monitoring
- ✅ HVAC status and control simulation
- ✅ Staff dispatch system with ETA
- ✅ Interactive venue mini-map
- ✅ AI-powered alert system with audio
- ✅ Claude AI Assistant integration
- ✅ Demo scenario presets
- ✅ Response time comparison widget
- ✅ Incident timeline
- ✅ Post-event analytics dashboard
- ✅ Keyboard shortcuts
- ✅ Focus/fullscreen mode
- ✅ PlexIE OFF warning banner
- ✅ **NEW** Live narrator mode with auto-commentary
- ✅ **NEW** Toast notifications for events
- ✅ **NEW** Live clock in header
- ✅ **NEW** Scenario progress bar

### PlexIE ON vs OFF Contrast
| Feature | PlexIE OFF | PlexIE ON |
|---------|-----------|-----------|
| Alerts | None | Intelligent, prioritized |
| Density | Uncontrolled | 30% reduction |
| Temperature | Manual monitoring | Auto HVAC adjust |
| Response | 14+ minutes | 2 minutes |
| Recommendations | None | Context-aware |
| Audio Alerts | None | Critical beeps |

---

## 📁 Project Structure

```
plexie-occ-demo/
├── src/
│   ├── App.jsx          # Main application (1900+ lines)
│   ├── main.jsx         # Entry point
│   └── index.css        # Tailwind styles
├── backend/
│   ├── server.py        # FastAPI + YOLOv11 server
│   └── requirements.txt # Python dependencies
├── public/
│   └── datasets/        # Place videos here (cam1.mp4, etc.)
└── README.md
```

---

## 🎥 Video Setup

Place MP4 files in `public/datasets/`:
- `cam1.mp4` - Gate A / North Entry
- `cam2.mp4` - Gate B / South Exit
- `cam3.mp4` - Main Concourse
- `cam4.mp4` - West Stand (heat scenario zone)
- `cam5.mp4` - East Stand (collapse scenario zone)

**Recommended sources:**
- [Pexels Crowd Videos](https://www.pexels.com/search/videos/crowd/)
- [MOT17 Dataset](https://motchallenge.net/data/MOT17/)
- Stadium/venue surveillance footage

---

## 🔧 Troubleshooting

**Videos not loading?**
- Check file names (cam1.mp4, lowercase)
- Verify files are in public/datasets/
- Try Ctrl+Shift+R hard refresh

**Backend not connecting?**
- Ensure python server.py is running
- Check http://localhost:8000/api/status
- Verify port 8000 is available

**Build fails?**
```bash
rm -rf node_modules
npm cache clean --force
npm install
```

---

## 📝 License

Proprietary - Saudi Smart Infrastructure Group (SSIG)
For investor demonstration purposes only.

---

## 👥 SSIG Team

- **Akram Z. Awan** - CEO
- **Abdurrahman Foudhaily** - COO
- **Mohamed A. Abdelaziz** - CTO
- **Sultan R. Makanati** - CCO
- **Aisha Baothman** - CDO
