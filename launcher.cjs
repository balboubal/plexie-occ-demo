#!/usr/bin/env node

/**
 * PlexIE OCC Demo - Unified Launcher v7.3
 * ========================================
 * Features:
 * - Robust dependency detection with verification
 * - Auto-installs with fallback options
 * - Settings menu for tweaking AI parameters
 * - Installation verification with helpful tips
 */

const { spawn, execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const readline = require('readline');
const http = require('http');

// ============================================
// CONFIGURATION
// ============================================
const CONFIG = {
  frontendPort: 5173,
  backendPort: 8000,
  requiredVideos: ['cam1.mp4', 'cam2.mp4', 'cam3.mp4', 'cam4.mp4', 'cam5.mp4'],
  videoDir: 'public/datasets',
  backendDir: 'backend',
  settingsFile: 'plexie-settings.json',
};

// Default AI Settings (can be modified via settings menu)
const DEFAULT_SETTINGS = {
  DETECTION_FRAME_SKIP: 30,    // Detect every N frames (30 = ~1 FPS at 30fps)
  DETECTION_THRESHOLD: 0.5,    // Confidence threshold (0.0 - 1.0)
  PROCESS_WIDTH: 480,          // Processing resolution width
  PROCESS_HEIGHT: 352,         // Processing resolution height (must be divisible by 16)
  GRID_ROWS: 4,                // Density grid rows
  GRID_COLS: 4,                // Density grid columns
  MAX_CAMERAS: 5,              // Maximum concurrent cameras
};

// Load or create settings
function loadSettings() {
  const settingsPath = path.join(__dirname, CONFIG.settingsFile);
  try {
    if (fs.existsSync(settingsPath)) {
      const data = JSON.parse(fs.readFileSync(settingsPath, 'utf8'));
      return { ...DEFAULT_SETTINGS, ...data };
    }
  } catch (e) {
    // Ignore errors, use defaults
  }
  return { ...DEFAULT_SETTINGS };
}

function saveSettings(settings) {
  const settingsPath = path.join(__dirname, CONFIG.settingsFile);
  try {
    fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2));
    return true;
  } catch (e) {
    return false;
  }
}

let currentSettings = loadSettings();

// Colors for terminal
const c = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  white: '\x1b[37m',
};

// Logging helpers
const log = {
  info: (m) => console.log(`  ${c.blue}ℹ${c.reset} ${m}`),
  ok: (m) => console.log(`  ${c.green}✓${c.reset} ${m}`),
  warn: (m) => console.log(`  ${c.yellow}⚠${c.reset} ${m}`),
  err: (m) => console.log(`  ${c.red}✗${c.reset} ${m}`),
  section: (m) => console.log(`\n${c.cyan}${c.bold}▶ ${m}${c.reset}`),
};

// ============================================
// UTILITY FUNCTIONS
// ============================================
const fileExists = (p) => {
  try { fs.accessSync(p); return true; } catch { return false; }
};

const getFileSize = (p) => {
  try { return fs.statSync(p).size; } catch { return 0; }
};

const formatBytes = (bytes) => {
  if (bytes === 0) return '0 B';
  const k = 1024, sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
};

const clearScreen = () => {
  process.stdout.write('\x1b[2J\x1b[H');
};

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// Run command and return success/fail
const runCmd = (cmd, opts = {}) => {
  try {
    execSync(cmd, { 
      stdio: opts.silent ? 'pipe' : 'inherit', 
      cwd: opts.cwd || __dirname, 
      shell: true,
      timeout: opts.timeout || 300000, // 5 min default
    });
    return true;
  } catch { return false; }
};

// Run command and return output
const getOutput = (cmd) => {
  try {
    return execSync(cmd, { 
      encoding: 'utf8', 
      shell: true, 
      stdio: ['pipe', 'pipe', 'pipe'],
      timeout: 10000,
    }).trim();
  } catch { return null; }
};

// Check if port is in use
const isPortInUse = async (port) => {
  return new Promise((resolve) => {
    const req = http.get(`http://localhost:${port}`, () => resolve(true));
    req.on('error', () => resolve(false));
    req.setTimeout(500, () => { req.destroy(); resolve(false); });
  });
};

// Global process handles
let frontendProcess = null;
let backendProcess = null;

// ============================================
// DEPENDENCY DETECTION
// ============================================
function getNodeVersion() {
  const v = getOutput('node --version');
  return v ? v.replace('v', '') : null;
}

function getNpmVersion() {
  return getOutput('npm --version');
}

function getPythonCmd() {
  // Try different Python commands
  for (const cmd of ['python3', 'python', 'py']) {
    const v = getOutput(`${cmd} --version`);
    if (v && v.includes('Python 3')) {
      return cmd;
    }
  }
  return null;
}

function getPythonVersion() {
  const cmd = getPythonCmd();
  if (!cmd) return null;
  const v = getOutput(`${cmd} --version`);
  return v ? v.replace('Python ', '') : null;
}

function getPipCmd() {
  const pyCmd = getPythonCmd();
  if (!pyCmd) return null;
  
  // Try using python -m pip first (most reliable)
  if (getOutput(`${pyCmd} -m pip --version`)) {
    return `${pyCmd} -m pip`;
  }
  
  // Fallback to pip3/pip
  for (const cmd of ['pip3', 'pip']) {
    if (getOutput(`${cmd} --version`)) return cmd;
  }
  return null;
}

function checkPythonPackage(pkg) {
  const pyCmd = getPythonCmd();
  if (!pyCmd) return { installed: false, version: null };
  
  // First try import
  try {
    const result = execSync(
      `${pyCmd} -c "import ${pkg}; print(getattr(${pkg}, '__version__', 'installed'))"`,
      { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'], timeout: 10000 }
    ).trim();
    return { installed: true, version: result };
  } catch {
    // Fallback: try pip show (handles packages with different import names)
    try {
      // Convert import name to package name (e.g., torch_directml -> torch-directml)
      const pkgName = pkg.replace(/_/g, '-');
      const pipCmd = getPipCmd();
      if (pipCmd) {
        const pipResult = execSync(
          `${pipCmd} show ${pkgName}`,
          { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'], timeout: 10000 }
        );
        // Extract version from pip show output
        const versionMatch = pipResult.match(/Version:\s*(.+)/);
        if (versionMatch) {
          return { installed: true, version: versionMatch[1].trim() };
        }
      }
    } catch {
      // Also try the original package name with pip show
      try {
        const pipCmd = getPipCmd();
        if (pipCmd) {
          const pipResult = execSync(
            `${pipCmd} show ${pkg}`,
            { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'], timeout: 10000 }
          );
          const versionMatch = pipResult.match(/Version:\s*(.+)/);
          if (versionMatch) {
            return { installed: true, version: versionMatch[1].trim() };
          }
        }
      } catch {
        // Package truly not installed
      }
    }
    return { installed: false, version: null };
  }
}

// ============================================
// COMPREHENSIVE DEPENDENCY CHECK
// ============================================
function checkAllDependencies() {
  log.section('Checking Dependencies');
  console.log();
  
  const deps = {
    node: { ok: false, version: null, required: '16.0.0' },
    npm: { ok: false, version: null },
    python: { ok: false, version: null, required: '3.8.0', cmd: null },
    pip: { ok: false, cmd: null },
    ultralytics: { ok: false, version: null },
    opencv: { ok: false, version: null },
    fastapi: { ok: false, version: null },
    uvicorn: { ok: false, version: null },
    npmModules: { ok: false },
    videos: { ok: false, count: 0 },
  };
  
  // Node.js
  deps.node.version = getNodeVersion();
  deps.node.ok = deps.node.version !== null;
  if (deps.node.ok) {
    log.ok(`Node.js: ${c.green}v${deps.node.version}${c.reset}`);
  } else {
    log.err(`Node.js: ${c.red}NOT FOUND${c.reset} (required: v${deps.node.required}+)`);
  }
  
  // npm
  deps.npm.version = getNpmVersion();
  deps.npm.ok = deps.npm.version !== null;
  if (deps.npm.ok) {
    log.ok(`npm: ${c.green}v${deps.npm.version}${c.reset}`);
  } else {
    log.err(`npm: ${c.red}NOT FOUND${c.reset}`);
  }
  
  // Python
  deps.python.cmd = getPythonCmd();
  deps.python.version = getPythonVersion();
  deps.python.ok = deps.python.version !== null;
  if (deps.python.ok) {
    log.ok(`Python: ${c.green}v${deps.python.version}${c.reset} (${deps.python.cmd})`);
  } else {
    log.err(`Python: ${c.red}NOT FOUND${c.reset} (required: v${deps.python.required}+)`);
  }
  
  // pip
  deps.pip.cmd = getPipCmd();
  deps.pip.ok = deps.pip.cmd !== null;
  if (deps.pip.ok) {
    log.ok(`pip: ${c.green}available${c.reset} (${deps.pip.cmd})`);
  } else {
    log.err(`pip: ${c.red}NOT FOUND${c.reset}`);
  }
  
  // Python packages (only if Python found)
  let missingRequired = [];
  let missingOptional = [];
  
  if (deps.python.ok) {
    console.log();
    log.info('Python packages (required):');
    
    // Required packages
    const requiredPackages = [
      { name: 'torch', importName: 'torch', desc: 'PyTorch (AI framework)' },
      { name: 'torchvision', importName: 'torchvision', desc: 'Vision models' },
      { name: 'opencv-python', importName: 'cv2', desc: 'Computer vision' },
      { name: 'numpy', importName: 'numpy', desc: 'Numerical computing' },
      { name: 'scipy', importName: 'scipy', desc: 'Scientific computing' },
      { name: 'matplotlib', importName: 'matplotlib', desc: 'Plotting' },
      { name: 'pillow', importName: 'PIL', desc: 'Image processing' },
      { name: 'fastapi', importName: 'fastapi', desc: 'API server' },
      { name: 'uvicorn', importName: 'uvicorn', desc: 'ASGI server' },
    ];
    
    for (const pkg of requiredPackages) {
      const check = checkPythonPackage(pkg.importName);
      deps[pkg.name] = { ok: check.installed, version: check.version };
      if (check.installed) {
        log.ok(`  ${pkg.name}: ${c.green}${check.version}${c.reset}`);
      } else {
        log.err(`  ${pkg.name}: ${c.red}MISSING${c.reset} - ${pkg.desc}`);
        missingRequired.push(pkg.name);
      }
    }
    
    // Recommended packages (for performance)
    console.log();
    log.info('Python packages (recommended for performance):');
    
    const recommendedPackages = [
      { name: 'onnxruntime', importName: 'onnxruntime', desc: '2-3x faster inference' },
      { name: 'numba', importName: 'numba', desc: 'JIT compilation' },
    ];
    
    for (const pkg of recommendedPackages) {
      const check = checkPythonPackage(pkg.importName);
      deps[pkg.name] = { ok: check.installed, version: check.version };
      if (check.installed) {
        log.ok(`  ${pkg.name}: ${c.green}${check.version}${c.reset}`);
      } else {
        log.warn(`  ${pkg.name}: ${c.yellow}not installed${c.reset} - ${pkg.desc}`);
        missingOptional.push(pkg.name);
      }
    }
    
    // Optional GPU packages
    console.log();
    log.info('Python packages (GPU acceleration):');
    
    // DirectML for AMD
    const directml = checkPythonPackage('torch_directml');
    deps.directml = { ok: directml.installed, version: directml.version };
    if (directml.installed) {
      log.ok(`  torch-directml: ${c.green}${directml.version}${c.reset} (AMD GPU)`);
    } else {
      log.warn(`  torch-directml: ${c.yellow}not installed${c.reset} (for AMD GPUs)`);
      missingOptional.push('torch-directml');
    }
    
    // Check if onnxruntime has GPU
    const onnxGpu = checkPythonPackage('onnxruntime');
    if (onnxGpu.installed) {
      // Check if it's GPU version by looking for CUDA provider
      const hasGpu = getOutput(`${deps.python.cmd} -c "import onnxruntime; print('CUDAExecutionProvider' in onnxruntime.get_available_providers())"`) === 'True';
      if (hasGpu) {
        log.ok(`  onnxruntime-gpu: ${c.green}${onnxGpu.version}${c.reset} (NVIDIA CUDA)`);
      } else {
        log.info(`  onnxruntime: ${c.dim}${onnxGpu.version} CPU version${c.reset} (for NVIDIA GPU: pip install onnxruntime-gpu)`);
      }
    }
  }
  
  // npm modules
  console.log();
  const nodeModulesPath = path.join(__dirname, 'node_modules');
  deps.npmModules.ok = fileExists(nodeModulesPath) && fileExists(path.join(nodeModulesPath, 'vite'));
  if (deps.npmModules.ok) {
    log.ok(`npm modules: ${c.green}installed${c.reset}`);
  } else {
    log.warn(`npm modules: ${c.yellow}not installed${c.reset}`);
    missingRequired.push('npm modules');
  }
  
  // Videos
  const videoDir = path.join(__dirname, CONFIG.videoDir);
  let videoCount = 0;
  for (const v of CONFIG.requiredVideos) {
    if (fileExists(path.join(videoDir, v)) && getFileSize(path.join(videoDir, v)) > 10000) {
      videoCount++;
    }
  }
  deps.videos.count = videoCount;
  deps.videos.ok = videoCount > 0;
  if (videoCount > 0) {
    log.ok(`Videos: ${c.green}${videoCount}/${CONFIG.requiredVideos.length}${c.reset} in ${CONFIG.videoDir}/`);
  } else {
    log.warn(`Videos: ${c.yellow}none found${c.reset} in ${CONFIG.videoDir}/`);
  }
  
  // ===== SUMMARY =====
  console.log();
  console.log(`${c.cyan}${'─'.repeat(54)}${c.reset}`);
  
  if (missingRequired.length > 0) {
    log.err(`Missing REQUIRED packages (${missingRequired.length}):`);
    console.log(`  ${c.red}${missingRequired.join(', ')}${c.reset}`);
    console.log();
    log.info(`Run option ${c.bold}2${c.reset} to install all dependencies`);
  } else if (missingOptional.length > 0) {
    log.ok('All required packages installed!');
    console.log();
    log.info(`Optional packages for better performance: ${c.yellow}${missingOptional.join(', ')}${c.reset}`);
    log.info(`Install with: ${c.dim}pip install ${missingOptional.join(' ')}${c.reset}`);
  } else {
    log.ok('All packages installed! Ready to run.');
  }
  
  // Check for model weights
  const weightsPath = path.join(__dirname, 'backend', 'weights', 'SHTechA.pth');
  const vggPath = path.join(__dirname, 'backend', 'checkpoints', 'vgg16_bn-6c64b313.pth');
  
  if (!fileExists(weightsPath) || !fileExists(vggPath)) {
    console.log();
    log.warn('Model weights missing:');
    if (!fileExists(weightsPath)) {
      log.info(`  ${c.yellow}•${c.reset} SHTechA.pth → backend/weights/`);
    }
    if (!fileExists(vggPath)) {
      log.info(`  ${c.yellow}•${c.reset} vgg16_bn-6c64b313.pth → backend/checkpoints/`);
    }
  }
  
  return deps;
}

// ============================================
// INSTALL DEPENDENCIES WITH VERIFICATION
// ============================================
async function installAllDependencies() {
  log.section('Installing All Dependencies');
  console.log();
  
  const installResults = {
    npm: { attempted: false, success: false, verified: false },
    python: { attempted: false, success: false, packages: {} },
  };
  
  // ===== NPM PACKAGES =====
  log.info('Installing npm packages...');
  installResults.npm.attempted = true;
  
  if (runCmd('npm install')) {
    installResults.npm.success = true;
    
    // Verify installation
    const nodeModulesPath = path.join(__dirname, 'node_modules');
    const criticalModules = ['vite', 'react', 'tailwindcss'];
    let allVerified = true;
    
    for (const mod of criticalModules) {
      const modPath = path.join(nodeModulesPath, mod);
      if (fileExists(modPath)) {
        log.ok(`  Verified: ${mod}`);
      } else {
        log.err(`  Missing: ${mod}`);
        allVerified = false;
      }
    }
    
    installResults.npm.verified = allVerified;
    
    if (allVerified) {
      log.ok('npm packages installed and verified');
    } else {
      log.warn('Some npm packages may not have installed correctly');
      console.log();
      log.info(`${c.yellow}TIP:${c.reset} Try these steps:`);
      log.info(`  1. Delete node_modules folder: ${c.dim}rm -rf node_modules${c.reset}`);
      log.info(`  2. Clear npm cache: ${c.dim}npm cache clean --force${c.reset}`);
      log.info(`  3. Run install again: ${c.dim}npm install${c.reset}`);
    }
  } else {
    log.err('Failed to install npm packages');
    console.log();
    log.info(`${c.yellow}TIPS:${c.reset}`);
    log.info(`  - Check Node.js is installed: ${c.dim}node --version${c.reset}`);
    log.info(`  - Check npm is installed: ${c.dim}npm --version${c.reset}`);
    log.info(`  - Try running as administrator (Windows) or with sudo (Linux)`);
    log.info(`  - Check internet connection`);
    return installResults;
  }
  
  // ===== PYTHON PACKAGES =====
  console.log();
  const pipCmd = getPipCmd();
  const pyCmd = getPythonCmd();
  
  if (!pipCmd || !pyCmd) {
    log.err('pip not found - cannot install Python packages');
    console.log();
    log.info(`${c.yellow}TIPS:${c.reset}`);
    log.info(`  - Install Python 3.8+ from ${c.cyan}https://python.org${c.reset}`);
    log.info(`  - On Windows, check "Add Python to PATH" during installation`);
    log.info(`  - On Linux: ${c.dim}sudo apt install python3 python3-pip${c.reset}`);
    log.info(`  - On macOS: ${c.dim}brew install python3${c.reset}`);
    return installResults;
  }
  
  installResults.python.attempted = true;
  
  log.info('Installing Python packages (this may take several minutes)...');
  log.info(`${c.dim}Core: torch, torchvision, opencv-python, numpy, scipy${c.reset}`);
  log.info(`${c.dim}Server: fastapi, uvicorn, websockets${c.reset}`);
  console.log();
  
  const reqPath = path.join(__dirname, CONFIG.backendDir, 'requirements.txt');
  
  // Determine install command based on platform
  let installCmd;
  if (process.platform === 'linux') {
    // Try with --break-system-packages first (for externally managed Python)
    installCmd = `${pipCmd} install -r "${reqPath}" --break-system-packages`;
    if (!runCmd(installCmd, { silent: true })) {
      installCmd = `${pipCmd} install -r "${reqPath}"`;
    }
  } else {
    installCmd = `${pipCmd} install -r "${reqPath}"`;
  }
  
  if (runCmd(installCmd)) {
    installResults.python.success = true;
    console.log();
    
    // Verify critical Python packages
    log.info('Verifying Python packages...');
    const criticalPkgs = [
      { name: 'torch', import: 'torch' },
      { name: 'torchvision', import: 'torchvision' },
      { name: 'opencv', import: 'cv2' },
      { name: 'numpy', import: 'numpy' },
      { name: 'fastapi', import: 'fastapi' },
      { name: 'uvicorn', import: 'uvicorn' },
    ];
    
    let allPyVerified = true;
    for (const pkg of criticalPkgs) {
      const result = checkPythonPackage(pkg.import);
      installResults.python.packages[pkg.name] = result;
      
      if (result.installed) {
        log.ok(`  ${pkg.name}: ${c.green}${result.version || 'OK'}${c.reset}`);
      } else {
        log.err(`  ${pkg.name}: ${c.red}NOT FOUND${c.reset}`);
        allPyVerified = false;
      }
    }
    
    if (!allPyVerified) {
      console.log();
      log.warn('Some Python packages failed to install');
      log.info(`${c.yellow}TIPS:${c.reset}`);
      log.info(`  - Try installing individually: ${c.dim}${pipCmd} install torch torchvision${c.reset}`);
      log.info(`  - For torch issues, visit: ${c.cyan}https://pytorch.org/get-started/locally/${c.reset}`);
      log.info(`  - On Windows, you may need Visual C++ Build Tools`);
      log.info(`  - Try upgrading pip: ${c.dim}${pipCmd} install --upgrade pip${c.reset}`);
    } else {
      log.ok('Python packages installed and verified');
    }
    
    // GPU Acceleration
    console.log();
    log.info('Checking GPU for acceleration...');
    
    const hasNvidia = getOutput('nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo ""');
    
    let hasAmd = false;
    if (process.platform === 'win32') {
      const dxdiag = getOutput('wmic path win32_VideoController get name 2>nul || echo ""');
      hasAmd = dxdiag && (dxdiag.toLowerCase().includes('amd') || dxdiag.toLowerCase().includes('radeon'));
    }
    
    if (hasNvidia && hasNvidia.trim()) {
      log.info(`Detected NVIDIA GPU: ${c.green}${hasNvidia.trim()}${c.reset}`);
      log.info('Installing onnxruntime-gpu for CUDA acceleration...');
      
      runCmd(`${pipCmd} uninstall onnxruntime -y`, { silent: true });
      
      if (runCmd(`${pipCmd} install onnxruntime-gpu`, { silent: true })) {
        // Verify
        const onnx = checkPythonPackage('onnxruntime');
        if (onnx.installed) {
          log.ok(`onnxruntime-gpu installed: ${onnx.version}`);
        } else {
          log.warn('onnxruntime-gpu install may have failed');
          log.info(`${c.yellow}TIP:${c.reset} For CUDA 12.x, try: ${c.dim}${pipCmd} install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/${c.reset}`);
        }
      } else {
        log.warn('Could not install onnxruntime-gpu');
        log.info(`${c.yellow}TIP:${c.reset} Ensure CUDA toolkit is installed from ${c.cyan}https://developer.nvidia.com/cuda-downloads${c.reset}`);
      }
    } else if (hasAmd) {
      log.info(`Detected AMD GPU`);
      log.info('Installing torch-directml for AMD GPU acceleration...');
      
      if (runCmd(`${pipCmd} install torch-directml`, { silent: true })) {
        const dml = checkPythonPackage('torch_directml');
        if (dml.installed) {
          log.ok(`torch-directml installed: ${dml.version}`);
        } else {
          log.warn('torch-directml may not have installed correctly');
          log.info(`${c.yellow}TIP:${c.reset} torch-directml requires Windows 10/11 with DirectX 12`);
        }
      } else {
        log.warn('Could not install torch-directml');
      }
    } else {
      log.info('No dedicated GPU detected - using CPU (optimized with numba)');
    }
  } else {
    console.log();
    log.err('Failed to install Python packages');
    log.info(`${c.yellow}TIPS:${c.reset}`);
    log.info(`  - Try manually: ${c.dim}${pipCmd} install -r backend/requirements.txt${c.reset}`);
    log.info(`  - Check for error messages above`);
    log.info(`  - Ensure you have enough disk space`);
    log.info(`  - Try upgrading pip: ${c.dim}${pipCmd} install --upgrade pip${c.reset}`);
  }
  
  // Summary
  console.log();
  log.section('Installation Summary');
  console.log();
  log.info(`npm packages: ${installResults.npm.verified ? `${c.green}OK${c.reset}` : `${c.yellow}Issues${c.reset}`}`);
  log.info(`Python packages: ${installResults.python.success ? `${c.green}OK${c.reset}` : `${c.yellow}Issues${c.reset}`}`);
  
  return installResults;
}

// ============================================
// START BACKEND SERVER
// ============================================
async function startBackend() {
  const pyCmd = getPythonCmd();
  
  if (!pyCmd) {
    log.err('Python not found!');
    log.info('Install Python 3.8+ from https://python.org');
    return null;
  }
  
  if (await isPortInUse(CONFIG.backendPort)) {
    log.ok(`Backend already running on port ${CONFIG.backendPort}`);
    return 'already_running';
  }
  
  // Check if deps installed
  const torch = checkPythonPackage('torch');
  const cv2 = checkPythonPackage('cv2');
  const fastapi = checkPythonPackage('fastapi');
  
  if (!torch.installed || !cv2.installed || !fastapi.installed) {
    log.warn('Python dependencies missing - installing...');
    console.log();
    await installAllDependencies();
    console.log();
  }
  
  // Use server_multicam.py (P2PNet-based, 5 cameras)
  const serverPath = path.join(__dirname, CONFIG.backendDir, 'server_multicam.py');
  if (!fileExists(serverPath)) {
    log.err('backend/server_multicam.py not found!');
    return null;
  }
  
  log.info('Starting Multi-Camera AI Backend (P2PNet)...');
  log.info('This will process 5 camera feeds with crowd detection');
  console.log();
  
  return new Promise((resolve) => {
    backendProcess = spawn(pyCmd, [serverPath], {
      cwd: path.join(__dirname, CONFIG.backendDir),
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: false,
    });
    
    let started = false;
    
    const checkStarted = (str) => {
      if (!started && (str.includes('Uvicorn running') || str.includes('Application startup complete'))) {
        started = true;
        console.log();
        log.ok(`AI Backend running on http://localhost:${CONFIG.backendPort}`);
        resolve(backendProcess);
      }
    };
    
    backendProcess.stdout.on('data', (data) => {
      const str = data.toString();
      process.stdout.write(`  ${c.dim}[backend] ${str}${c.reset}`);
      checkStarted(str);
    });
    
    backendProcess.stderr.on('data', (data) => {
      const str = data.toString();
      if (str.includes('INFO') || str.includes('Uvicorn') || str.includes('Started')) {
        process.stdout.write(`  ${c.dim}[backend] ${str}${c.reset}`);
      }
      checkStarted(str);
    });
    
    backendProcess.on('error', (err) => {
      log.err(`Failed to start backend: ${err.message}`);
      resolve(null);
    });
    
    backendProcess.on('close', (code) => {
      if (!started) {
        log.err(`Backend exited with code ${code}`);
        resolve(null);
      }
      backendProcess = null;
    });
    
    // Timeout - model download can take time
    setTimeout(() => {
      if (!started) {
        log.warn('Backend still loading (may be downloading YOLO model)...');
        resolve(backendProcess);
      }
    }, 60000);
  });
}

// ============================================
// START FRONTEND SERVER
// ============================================
async function startFrontend() {
  if (await isPortInUse(CONFIG.frontendPort)) {
    log.ok(`Frontend already running on port ${CONFIG.frontendPort}`);
    return 'already_running';
  }
  
  // Check npm modules
  if (!fileExists(path.join(__dirname, 'node_modules', 'vite'))) {
    log.info('Installing npm packages...');
    if (!runCmd('npm install')) {
      log.err('Failed to install npm packages');
      return null;
    }
    log.ok('npm packages installed');
  }
  
  log.info('Starting frontend server...');
  
  const vitePath = path.join(__dirname, 'node_modules', '.bin', 'vite');
  const viteCmd = process.platform === 'win32' ? `"${vitePath}.cmd"` : vitePath;
  
  return new Promise((resolve) => {
    frontendProcess = spawn(viteCmd, ['--host'], {
      cwd: __dirname,
      stdio: ['ignore', 'pipe', 'pipe'],
      shell: true,
    });
    
    let started = false;
    
    frontendProcess.stdout.on('data', (data) => {
      const str = data.toString();
      process.stdout.write(str);
      if (!started && str.includes('Local:')) {
        started = true;
        resolve(frontendProcess);
      }
    });
    
    frontendProcess.stderr.on('data', (data) => {
      process.stderr.write(data);
    });
    
    frontendProcess.on('error', (err) => {
      log.err(`Failed to start frontend: ${err.message}`);
      resolve(null);
    });
    
    frontendProcess.on('close', () => {
      frontendProcess = null;
    });
  });
}

// ============================================
// STOP SERVERS
// ============================================
function stopServers() {
  if (frontendProcess) {
    frontendProcess.kill();
    frontendProcess = null;
  }
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
  }
}

// ============================================
// OPEN BROWSER
// ============================================
function openBrowser() {
  const url = `http://localhost:${CONFIG.frontendPort}`;
  const cmd = process.platform === 'win32' ? `start ${url}` :
              process.platform === 'darwin' ? `open ${url}` : `xdg-open ${url}`;
  runCmd(cmd, { silent: true });
  log.ok(`Opened ${url}`);
}

// ============================================
// TEST P2PNET (CROWD COUNTING)
// ============================================
async function runP2PNetTest() {
  log.section('P2PNet Crowd Counting Test');
  console.log();
  log.info('P2PNet is better for dense crowds (stadiums, concerts)');
  log.info('It places dots on heads rather than bounding boxes');
  console.log();
  
  // Check Python
  const pyCmd = getPythonCmd();
  if (!pyCmd) {
    log.err('Python not found!');
    log.info('Install Python 3.8+ from https://python.org');
    return;
  }
  log.ok(`Python: ${pyCmd}`);
  
  // Check video
  const videoDir = path.join(__dirname, CONFIG.videoDir);
  let videoPath = null;
  for (const v of CONFIG.requiredVideos) {
    const vp = path.join(videoDir, v);
    if (fileExists(vp) && getFileSize(vp) > 10000) {
      videoPath = vp;
      log.ok(`Video: ${v} (${formatBytes(getFileSize(vp))})`);
      break;
    }
  }
  
  if (!videoPath) {
    log.err('No video found!');
    log.info(`Add a video file to: ${CONFIG.videoDir}/`);
    log.info('Name it cam1.mp4 (or cam2.mp4, etc.)');
    return;
  }
  
  // Check/install Python deps
  const torch = checkPythonPackage('torch');
  const cv2 = checkPythonPackage('cv2');
  const scipy = checkPythonPackage('scipy');
  const matplotlib = checkPythonPackage('matplotlib');
  
  if (!torch.installed || !cv2.installed || !scipy.installed || !matplotlib.installed) {
    log.warn('Required packages not installed');
    console.log();
    
    const pipCmd = getPipCmd();
    if (!pipCmd) {
      log.err('pip not found - cannot install packages');
      return;
    }
    
    log.info('Installing torch, torchvision, opencv-python, scipy, matplotlib...');
    log.info('This may take a few minutes...');
    console.log();
    
    // Install torch (this can take a while)
    if (!runCmd(`${pipCmd} install torch torchvision pillow opencv-python numpy scipy matplotlib`)) {
      log.err('Failed to install packages');
      log.info('Try installing manually:');
      log.info(`  ${pipCmd} install torch torchvision pillow opencv-python numpy scipy matplotlib`);
      return;
    }
    log.ok('Packages installed');
  } else {
    log.ok(`torch: ${torch.version}`);
    log.ok(`opencv: ${cv2.version}`);
  }
  
  // Check for P2PNet weights
  const weightsPath = path.join(__dirname, 'backend', 'weights', 'SHTechA.pth');
  if (!fileExists(weightsPath)) {
    console.log();
    log.warn('P2PNet weights not found!');
    log.info('Download SHTechA.pth from:');
    log.info('  https://drive.google.com/file/d/1hhbj_3vPp4hSqhC9WPTKdv6u1gJKdaXc/view');
    log.info(`Save to: backend/weights/SHTechA.pth`);
    return;
  } else {
    log.ok(`P2PNet Weights: SHTechA.pth (${formatBytes(getFileSize(weightsPath))})`);
  }
  
  // Check for VGG backbone weights
  const vggWeightsPath = path.join(__dirname, 'backend', 'checkpoints', 'vgg16_bn-6c64b313.pth');
  if (!fileExists(vggWeightsPath)) {
    console.log();
    log.warn('VGG16-BN backbone weights not found!');
    log.info('Download vgg16_bn-6c64b313.pth from:');
    log.info('  https://download.pytorch.org/models/vgg16_bn-6c64b313.pth');
    log.info(`Save to: backend/checkpoints/vgg16_bn-6c64b313.pth`);
    return;
  } else {
    log.ok(`VGG Backbone: vgg16_bn-6c64b313.pth (${formatBytes(getFileSize(vggWeightsPath))})`);
  }
  
  // Create output directory
  const outputDir = path.join(__dirname, 'backend', 'output');
  if (!fileExists(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  // Check for scripts in order of preference: smooth > ultra > optimized > regular
  let testScript = path.join(__dirname, 'backend', 'run_test_smooth.py');
  let scriptType = 'smooth';
  
  if (!fileExists(testScript)) {
    testScript = path.join(__dirname, 'backend', 'run_test_ultra.py');
    scriptType = 'ultra';
  }
  if (!fileExists(testScript)) {
    testScript = path.join(__dirname, 'backend', 'run_test_optimized.py');
    scriptType = 'optimized';
  }
  if (!fileExists(testScript)) {
    testScript = path.join(__dirname, 'backend', 'run_test.py');
    scriptType = 'basic';
  }
  
  if (!fileExists(testScript)) {
    log.err('No detection script found!');
    return;
  }
  
  console.log();
  log.info('Starting P2PNet crowd counting...');
  log.info(`Using: ${scriptType} performance mode`);
  log.info('A live preview window will open with 3s delay');
  log.info('Press Q to quit early');
  console.log();
  
  // Show optimizations based on script type
  console.log(`  ${c.cyan}Performance Optimizations:${c.reset}`);
  console.log(`    ${c.dim}- Frame skip: Every 6th frame (6x faster)${c.reset}`);
  console.log(`    ${c.dim}- Resolution: 480p max (2x faster)${c.reset}`);
  console.log(`    ${c.dim}- GPU: CUDA (NVIDIA) / DirectML (AMD) / CPU${c.reset}`);
  
  if (scriptType === 'smooth') {
    console.log(`    ${c.dim}- Decoupled display thread (30 FPS target)${c.reset}`);
    console.log(`    ${c.dim}- 60-frame read-ahead buffer${c.reset}`);
    console.log(`    ${c.dim}- Pre-allocated memory buffers${c.reset}`);
    console.log(`    ${c.dim}- Disabled GC during processing${c.reset}`);
  }
  if (scriptType === 'smooth' || scriptType === 'ultra' || scriptType === 'optimized') {
    console.log(`    ${c.dim}- FP16: Half precision on GPU (2x faster)${c.reset}`);
    console.log(`    ${c.dim}- Async I/O: Background frame read/save${c.reset}`);
  }
  if (scriptType === 'smooth' || scriptType === 'ultra') {
    console.log(`    ${c.dim}- ONNX Runtime: Optimized inference (2-3x faster)${c.reset}`);
  }
  console.log();
  
  console.log(`  ${c.cyan}Display Features:${c.reset}`);
  console.log(`    ${c.dim}- Green dots: Detected people${c.reset}`);
  console.log(`    ${c.dim}- Red rectangle: Hotspot (densest area)${c.reset}`);
  console.log(`    ${c.dim}- Grid overlay: 4x4 density zones${c.reset}`);
  console.log(`    ${c.dim}- Real-time stats & ETA${c.reset}`);
  console.log();
  
  // Check for optional acceleration packages
  const hasDirectML = checkPythonPackage('torch_directml');
  const hasOnnx = checkPythonPackage('onnxruntime');
  const hasNumba = checkPythonPackage('numba');
  
  console.log(`  ${c.cyan}Optional Accelerators:${c.reset}`);
  if (hasDirectML.installed) {
    console.log(`    ${c.green}✓${c.reset} torch-directml (AMD GPU)`);
  } else {
    console.log(`    ${c.dim}○ torch-directml: pip install torch-directml${c.reset}`);
  }
  if (hasOnnx.installed) {
    console.log(`    ${c.green}✓${c.reset} onnxruntime ${hasOnnx.version}`);
  } else {
    console.log(`    ${c.dim}○ onnxruntime: pip install onnxruntime${c.reset}`);
  }
  if (hasNumba.installed) {
    console.log(`    ${c.green}✓${c.reset} numba ${hasNumba.version} (JIT)`);
  } else {
    console.log(`    ${c.dim}○ numba: pip install numba${c.reset}`);
  }
  console.log();
  
  console.log(`  ${c.yellow}Processing... please wait${c.reset}`);
  console.log();
  
  // Run Python script with live output
  try {
    const cmdArgs = [
      `"${testScript}"`,
      '--video',
      `--video_path "${videoPath}"`,
      `--weight_path "${weightsPath}"`,
      `--output_dir "${outputDir}"`,
      '--shape 640 480',
      '--threshold 0.5',
      '--frame_skip 6',
      '--max_resolution 480',
      '--startup_delay 3',
      '--device auto',
    ];
    
    // Add script-specific flags
    if (scriptType === 'smooth') {
      cmdArgs.push('--save_video');
      // Note: smooth version handles ONNX internally, no flag needed
    } else if (scriptType === 'ultra') {
      cmdArgs.push('--use_fp16');
      cmdArgs.push('--use_onnx');
      cmdArgs.push('--async_read');
      cmdArgs.push('--async_save');
      cmdArgs.push('--save_video');
    } else if (scriptType === 'optimized') {
      cmdArgs.push('--use_fp16');
      cmdArgs.push('--async_read');
      cmdArgs.push('--async_save');
      cmdArgs.push('--save_video');
    }
    
    execSync(`${pyCmd} ${cmdArgs.join(' ')}`, {
      stdio: 'inherit',
      cwd: path.join(__dirname, 'backend'),
      shell: true,
    });
  } catch (e) {
    // User may have quit with Q, which is fine
  }
  
  const outputAvi = path.join(outputDir, 'output.avi');
  console.log();
  if (fileExists(outputAvi)) {
    log.ok(`Output video saved: backend/output/output.avi`);
    log.info(`Size: ${formatBytes(getFileSize(outputAvi))}`);
    
    // Count output frames
    try {
      const frames = fs.readdirSync(outputDir).filter(f => f.startsWith('pred_') && f.endsWith('.jpg'));
      log.info(`Frames saved: ${frames.length} images in backend/output/`);
    } catch {}
  }
}

// ============================================
// TEST YOLO (BOUNDING BOXES)
// ============================================
async function runDetectionTest() {
  log.section('YOLO Detection Test');
  console.log();
  log.info('YOLO uses bounding boxes - simpler but less accurate for dense crowds');
  console.log();
  
  // Check Python
  const pyCmd = getPythonCmd();
  if (!pyCmd) {
    log.err('Python not found!');
    log.info('Install Python 3.8+ from https://python.org');
    return;
  }
  log.ok(`Python: ${pyCmd}`);
  
  // Check video
  const videoDir = path.join(__dirname, CONFIG.videoDir);
  let videoPath = null;
  for (const v of CONFIG.requiredVideos) {
    const vp = path.join(videoDir, v);
    if (fileExists(vp) && getFileSize(vp) > 10000) {
      videoPath = vp;
      log.ok(`Video: ${v} (${formatBytes(getFileSize(vp))})`);
      break;
    }
  }
  
  if (!videoPath) {
    log.err('No video found!');
    log.info(`Add a video file to: ${CONFIG.videoDir}/`);
    log.info('Name it cam1.mp4 (or cam2.mp4, etc.)');
    return;
  }
  
  // Check/install Python deps
  const ultra = checkPythonPackage('ultralytics');
  const cv2 = checkPythonPackage('cv2');
  
  if (!ultra.installed || !cv2.installed) {
    log.warn('Required packages not installed');
    console.log();
    
    const pipCmd = getPipCmd();
    if (!pipCmd) {
      log.err('pip not found - cannot install packages');
      return;
    }
    
    log.info('Installing ultralytics and opencv-python...');
    if (!runCmd(`${pipCmd} install ultralytics opencv-python numpy`)) {
      log.err('Failed to install packages');
      return;
    }
    log.ok('Packages installed');
  } else {
    log.ok(`ultralytics: ${ultra.version}`);
    log.ok(`opencv: ${cv2.version}`);
  }
  
  // Run test
  const testScript = path.join(__dirname, 'backend', 'test_detection.py');
  if (!fileExists(testScript)) {
    log.err('backend/test_detection.py not found!');
    return;
  }
  
  console.log();
  log.info('Starting detection test...');
  log.info('A window will open showing the video with detections');
  log.info('Press Q to quit early');
  console.log();
  console.log(`  ${c.yellow}Processing video... please wait${c.reset}`);
  console.log();
  
  // Run Python script with live output
  try {
    execSync(`${pyCmd} "${testScript}"`, {
      stdio: 'inherit',
      cwd: path.join(__dirname, 'backend'),
      shell: true,
    });
  } catch (e) {
    // User may have quit with Q, which is fine
  }
  
  const outputPath = path.join(__dirname, 'backend', 'output_detected.avi');
  console.log();
  if (fileExists(outputPath)) {
    log.ok(`Output saved: backend/output_detected.avi`);
    log.info(`Size: ${formatBytes(getFileSize(outputPath))}`);
  }
}

// ============================================
// CREATE DISTRIBUTION ZIP
// ============================================
async function createDistributionZip() {
  log.section('Create Distribution ZIP');
  console.log();
  
  log.info('This will create a clean ZIP package without:');
  console.log(`    ${c.dim}- node_modules/ (npm dependencies)${c.reset}`);
  console.log(`    ${c.dim}- backend/weights/, checkpoints/ (AI model weights)${c.reset}`);
  console.log(`    ${c.dim}- backend/density/, output/ (generated images)${c.reset}`);
  console.log(`    ${c.dim}- backend/.git/ (git repository data)${c.reset}`);
  console.log(`    ${c.dim}- public/datasets/*.mp4 (video files)${c.reset}`);
  console.log(`    ${c.dim}- *.mp4, *.avi, *.pt, *.pth (media/model files)${c.reset}`);
  console.log(`    ${c.dim}- __pycache__/, *.pyc (Python cache)${c.reset}`);
  console.log();
  
  // Generate zip filename with timestamp
  const timestamp = new Date().toISOString().slice(0, 10).replace(/-/g, '');
  const zipName = `plexie-occ-demo-${timestamp}.zip`;
  const zipPath = path.join(__dirname, zipName);
  
  // Remove old zip if exists
  if (fileExists(zipPath)) {
    try { fs.unlinkSync(zipPath); } catch {}
  }
  
  log.info(`Creating: ${c.cyan}${zipName}${c.reset}`);
  console.log();
  
  // Use Node.js to create zip manually - most reliable cross-platform
  log.info('Scanning files...');
  
  // Files/folders to EXCLUDE (patterns)
  const excludePatterns = [
    'node_modules',
    '.git',
    '__pycache__',
    '_dist_temp',
    'backend/weights',
    'backend/checkpoints',
    'backend/density',
    'backend/output',
    'backend/crowd_datasets',
    'backend/vis',
    '.pyc',
    '.pt',
    '.pth',
    '.mp4',
    '.avi',
    '.mov',
    '.zip',
    '.ipynb',
    'structure.txt',
    'plexie-settings.json',
  ];
  
  const shouldExclude = (filePath) => {
    const rel = path.relative(__dirname, filePath).replace(/\\/g, '/');
    for (const pattern of excludePatterns) {
      if (rel.includes(pattern) || rel.endsWith(pattern)) {
        return true;
      }
    }
    return false;
  };
  
  // Collect files to include
  const filesToInclude = [];
  
  const scanDir = (dir, prefix = '') => {
    try {
      const items = fs.readdirSync(dir, { withFileTypes: true });
      for (const item of items) {
        const fullPath = path.join(dir, item.name);
        const relPath = path.join(prefix, item.name);
        
        if (shouldExclude(fullPath)) continue;
        
        if (item.isDirectory()) {
          scanDir(fullPath, relPath);
        } else {
          filesToInclude.push({ full: fullPath, rel: relPath });
        }
      }
    } catch {}
  };
  
  scanDir(__dirname);
  
  log.info(`Found ${filesToInclude.length} files to include`);
  
  // Calculate total size
  let totalSize = 0;
  for (const f of filesToInclude) {
    totalSize += getFileSize(f.full);
  }
  log.info(`Total size: ${formatBytes(totalSize)}`);
  console.log();
  
  // Try to use system zip command first (faster)
  let success = false;
  
  if (process.platform !== 'win32') {
    // Unix - try zip command
    const hasZip = getOutput('zip --version') !== null;
    if (hasZip) {
      log.info('Using system zip...');
      const excludeArgs = excludePatterns.map(p => `-x '${p}' -x '*/${p}' -x '*/${p}/*' -x '*.${p}'`).join(' ');
      try {
        execSync(`zip -r "${zipName}" . -x 'node_modules/*' -x '.git/*' -x '*__pycache__*' -x 'backend/weights/*' -x 'backend/checkpoints/*' -x 'backend/density/*' -x 'backend/output/*' -x 'backend/crowd_datasets/*' -x 'backend/vis/*' -x 'backend/util/*' -x '*.pyc' -x '*.pt' -x '*.pth' -x '*.mp4' -x '*.avi' -x '*.mov' -x '*.zip' -x '*.ipynb' -x 'structure.txt' -x '_dist_temp/*' -x 'backend/.git/*'`, {
          cwd: __dirname,
          stdio: 'pipe',
        });
        success = true;
      } catch (e) {
        log.warn('System zip failed, trying PowerShell...');
      }
    }
  }
  
  if (!success && process.platform === 'win32') {
    // Windows - use PowerShell Compress-Archive
    log.info('Using PowerShell...');
    
    const tempDir = path.join(__dirname, '_dist_temp');
    
    // Clean temp dir
    try {
      if (fileExists(tempDir)) {
        fs.rmSync(tempDir, { recursive: true, force: true });
      }
      fs.mkdirSync(tempDir, { recursive: true });
    } catch {}
    
    // Copy files using Node.js (more reliable than PowerShell copy)
    log.info('Copying files...');
    let copied = 0;
    for (const f of filesToInclude) {
      try {
        const destPath = path.join(tempDir, f.rel);
        const destDir = path.dirname(destPath);
        if (!fileExists(destDir)) {
          fs.mkdirSync(destDir, { recursive: true });
        }
        fs.copyFileSync(f.full, destPath);
        copied++;
      } catch {}
    }
    log.info(`Copied ${copied} files`);
    
    // Create zip with PowerShell
    try {
      execSync(`powershell -Command "Compress-Archive -Path '${tempDir}\\*' -DestinationPath '${zipPath}' -Force"`, {
        cwd: __dirname,
        stdio: 'pipe',
      });
      success = true;
    } catch (e) {
      log.err('PowerShell Compress-Archive failed');
    }
    
    // Cleanup temp
    try {
      fs.rmSync(tempDir, { recursive: true, force: true });
    } catch {}
  }
  
  if (success && fileExists(zipPath)) {
    const zipSize = getFileSize(zipPath);
    console.log();
    log.ok(`Distribution ZIP created successfully!`);
    log.ok(`File: ${c.cyan}${zipName}${c.reset}`);
    log.ok(`Size: ${c.green}${formatBytes(zipSize)}${c.reset}`);
    console.log();
    log.info('To use this package:');
    console.log(`    ${c.dim}1. Extract the ZIP${c.reset}`);
    console.log(`    ${c.dim}2. Run: npm install${c.reset}`);
    console.log(`    ${c.dim}3. Run: pip install -r backend/requirements.txt${c.reset}`);
    console.log(`    ${c.dim}4. Download SHTechA.pth weights to backend/weights/${c.reset}`);
    console.log(`    ${c.dim}5. Add video files to public/datasets/${c.reset}`);
    console.log(`    ${c.dim}6. Run: node launcher.cjs${c.reset}`);
  } else {
    log.err('Failed to create ZIP');
  }
}

// ============================================
// APPLY SETTINGS TO BACKEND
// ============================================
function applySettingsToBackend() {
  const serverPath = path.join(__dirname, CONFIG.backendDir, 'server_multicam.py');
  if (!fileExists(serverPath)) {
    return false;
  }
  
  try {
    let content = fs.readFileSync(serverPath, 'utf8');
    
    // Update settings in the Python file
    const replacements = [
      { pattern: /DETECTION_FRAME_SKIP\s*=\s*\d+/, replacement: `DETECTION_FRAME_SKIP = ${currentSettings.DETECTION_FRAME_SKIP}` },
      { pattern: /DETECTION_THRESHOLD\s*=\s*[\d.]+/, replacement: `DETECTION_THRESHOLD = ${currentSettings.DETECTION_THRESHOLD}` },
      { pattern: /PROCESS_WIDTH\s*=\s*\d+/, replacement: `PROCESS_WIDTH = ${currentSettings.PROCESS_WIDTH}` },
      { pattern: /PROCESS_HEIGHT\s*=\s*\d+/, replacement: `PROCESS_HEIGHT = ${currentSettings.PROCESS_HEIGHT}` },
      { pattern: /GRID_ROWS\s*=\s*\d+/, replacement: `GRID_ROWS = ${currentSettings.GRID_ROWS}` },
      { pattern: /GRID_COLS\s*=\s*\d+/, replacement: `GRID_COLS = ${currentSettings.GRID_COLS}` },
      { pattern: /MAX_CAMERAS\s*=\s*\d+/, replacement: `MAX_CAMERAS = ${currentSettings.MAX_CAMERAS}` },
    ];
    
    for (const { pattern, replacement } of replacements) {
      content = content.replace(pattern, replacement);
    }
    
    fs.writeFileSync(serverPath, content);
    return true;
  } catch (e) {
    return false;
  }
}

// ============================================
// SETTINGS MENU
// ============================================
async function showSettingsMenu() {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const ask = (q) => new Promise(r => rl.question(q, r));
  
  let inSettings = true;
  
  while (inSettings) {
    clearScreen();
    console.log(`
${c.cyan}╔══════════════════════════════════════════════════════╗
║           ${c.bold}${c.white}PlexIE OCC Settings${c.reset}${c.cyan}                        ║
╚══════════════════════════════════════════════════════╝${c.reset}

  ${c.bold}Detection Settings:${c.reset}
    ${c.cyan}[1]${c.reset} Frame Skip: ${c.green}${currentSettings.DETECTION_FRAME_SKIP}${c.reset} ${c.dim}(detect every N frames, 30=~1fps at 30fps video)${c.reset}
    ${c.cyan}[2]${c.reset} Detection Threshold: ${c.green}${currentSettings.DETECTION_THRESHOLD}${c.reset} ${c.dim}(confidence 0.0-1.0)${c.reset}

  ${c.bold}Processing Settings:${c.reset}
    ${c.cyan}[3]${c.reset} Process Width: ${c.green}${currentSettings.PROCESS_WIDTH}${c.reset} ${c.dim}(resize width for AI)${c.reset}
    ${c.cyan}[4]${c.reset} Process Height: ${c.green}${currentSettings.PROCESS_HEIGHT}${c.reset} ${c.dim}(must be divisible by 16)${c.reset}

  ${c.bold}Grid Settings:${c.reset}
    ${c.cyan}[5]${c.reset} Grid Rows: ${c.green}${currentSettings.GRID_ROWS}${c.reset} ${c.dim}(density heatmap rows)${c.reset}
    ${c.cyan}[6]${c.reset} Grid Columns: ${c.green}${currentSettings.GRID_COLS}${c.reset} ${c.dim}(density heatmap columns)${c.reset}

  ${c.bold}System Settings:${c.reset}
    ${c.cyan}[7]${c.reset} Max Cameras: ${c.green}${currentSettings.MAX_CAMERAS}${c.reset} ${c.dim}(maximum concurrent feeds)${c.reset}

  ${c.bold}Actions:${c.reset}
    ${c.cyan}[R]${c.reset} Reset to Defaults
    ${c.cyan}[S]${c.reset} Save & Apply Settings
    ${c.cyan}[0]${c.reset} Back to Main Menu
`);
    
    const choice = await ask(`  ${c.bold}Enter choice:${c.reset} `);
    console.log();
    
    switch (choice.trim().toLowerCase()) {
      case '1': {
        const val = await ask(`  Enter Frame Skip (current: ${currentSettings.DETECTION_FRAME_SKIP}): `);
        const num = parseInt(val);
        if (!isNaN(num) && num >= 1 && num <= 120) {
          currentSettings.DETECTION_FRAME_SKIP = num;
          log.ok(`Frame Skip set to ${num}`);
        } else {
          log.warn('Invalid value. Must be 1-120');
        }
        await sleep(1000);
        break;
      }
      
      case '2': {
        const val = await ask(`  Enter Detection Threshold (current: ${currentSettings.DETECTION_THRESHOLD}): `);
        const num = parseFloat(val);
        if (!isNaN(num) && num >= 0.1 && num <= 1.0) {
          currentSettings.DETECTION_THRESHOLD = num;
          log.ok(`Detection Threshold set to ${num}`);
        } else {
          log.warn('Invalid value. Must be 0.1-1.0');
        }
        await sleep(1000);
        break;
      }
      
      case '3': {
        const val = await ask(`  Enter Process Width (current: ${currentSettings.PROCESS_WIDTH}): `);
        const num = parseInt(val);
        if (!isNaN(num) && num >= 320 && num <= 1920) {
          currentSettings.PROCESS_WIDTH = num;
          log.ok(`Process Width set to ${num}`);
        } else {
          log.warn('Invalid value. Must be 320-1920');
        }
        await sleep(1000);
        break;
      }
      
      case '4': {
        const val = await ask(`  Enter Process Height (current: ${currentSettings.PROCESS_HEIGHT}): `);
        const num = parseInt(val);
        if (!isNaN(num) && num >= 192 && num <= 1080 && num % 16 === 0) {
          currentSettings.PROCESS_HEIGHT = num;
          log.ok(`Process Height set to ${num}`);
        } else {
          log.warn('Invalid value. Must be 192-1080 and divisible by 16');
        }
        await sleep(1000);
        break;
      }
      
      case '5': {
        const val = await ask(`  Enter Grid Rows (current: ${currentSettings.GRID_ROWS}): `);
        const num = parseInt(val);
        if (!isNaN(num) && num >= 2 && num <= 16) {
          currentSettings.GRID_ROWS = num;
          log.ok(`Grid Rows set to ${num}`);
        } else {
          log.warn('Invalid value. Must be 2-16');
        }
        await sleep(1000);
        break;
      }
      
      case '6': {
        const val = await ask(`  Enter Grid Columns (current: ${currentSettings.GRID_COLS}): `);
        const num = parseInt(val);
        if (!isNaN(num) && num >= 2 && num <= 16) {
          currentSettings.GRID_COLS = num;
          log.ok(`Grid Columns set to ${num}`);
        } else {
          log.warn('Invalid value. Must be 2-16');
        }
        await sleep(1000);
        break;
      }
      
      case '7': {
        const val = await ask(`  Enter Max Cameras (current: ${currentSettings.MAX_CAMERAS}): `);
        const num = parseInt(val);
        if (!isNaN(num) && num >= 1 && num <= 10) {
          currentSettings.MAX_CAMERAS = num;
          log.ok(`Max Cameras set to ${num}`);
        } else {
          log.warn('Invalid value. Must be 1-10');
        }
        await sleep(1000);
        break;
      }
      
      case 'r': {
        currentSettings = { ...DEFAULT_SETTINGS };
        log.ok('Settings reset to defaults');
        await sleep(1000);
        break;
      }
      
      case 's': {
        if (saveSettings(currentSettings)) {
          log.ok('Settings saved to plexie-settings.json');
        } else {
          log.warn('Could not save settings file');
        }
        
        if (applySettingsToBackend()) {
          log.ok('Settings applied to backend');
        } else {
          log.warn('Could not apply settings to backend (file not found or error)');
        }
        await sleep(1500);
        break;
      }
      
      case '0':
      case 'q':
        inSettings = false;
        break;
        
      default:
        if (choice.trim()) {
          log.warn('Invalid choice');
          await sleep(500);
        }
    }
  }
  
  rl.close();
}

// ============================================
// MAIN MENU
// ============================================
async function showMenu() {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  const ask = (q) => new Promise(r => rl.question(q, r));
  
  let running = true;
  while (running) {
    const feUp = await isPortInUse(CONFIG.frontendPort);
    const beUp = await isPortInUse(CONFIG.backendPort);
    
    console.log(`
${c.cyan}╔══════════════════════════════════════════════════════╗
║      ${c.bold}${c.white}PlexIE OCC Demo Launcher v7.3${c.reset}${c.cyan}                 ║
║         SSIG Stadium Safety - Frame-Based             ║
╚══════════════════════════════════════════════════════╝${c.reset}

  ${c.bold}Status:${c.reset}
    Frontend:   ${feUp ? `${c.green}● Running${c.reset}` : `${c.dim}○ Stopped${c.reset}`}
    AI Backend: ${beUp ? `${c.green}● Running${c.reset}` : `${c.dim}○ Stopped${c.reset}`}
    Frame Skip: ${c.cyan}${currentSettings.DETECTION_FRAME_SKIP}${c.reset} ${c.dim}(~${Math.round(30/currentSettings.DETECTION_FRAME_SKIP)} FPS detection)${c.reset}

  ${c.bold}Setup:${c.reset}
    ${c.cyan}[1]${c.reset} Check Dependencies
    ${c.cyan}[2]${c.reset} Install All Dependencies

  ${c.bold}Run:${c.reset}
    ${c.cyan}[3]${c.reset} ${c.green}Start Full OCC System${c.reset} ${c.dim}(5-Camera AI Dashboard)${c.reset}
    ${c.cyan}[4]${c.reset} Start Frontend Only
    ${c.cyan}[5]${c.reset} Start AI Backend Only

  ${c.bold}Test:${c.reset}
    ${c.cyan}[6]${c.reset} Test P2PNet ${c.dim}(single video)${c.reset}
    ${c.cyan}[7]${c.reset} Test YOLO ${c.dim}(bounding boxes)${c.reset}

  ${c.bold}Tools:${c.reset}
    ${c.cyan}[8]${c.reset} Open in Browser
    ${c.cyan}[9]${c.reset} ${c.yellow}Settings${c.reset} ${c.dim}(tweak detection parameters)${c.reset}
    ${c.cyan}[Z]${c.reset} Create Distribution ZIP

    ${c.cyan}[0]${c.reset} Exit
`);
    
    const choice = await ask(`  ${c.bold}Enter choice:${c.reset} `);
    console.log();
    
    switch (choice.trim()) {
      case '1':
        checkAllDependencies();
        console.log();
        await ask(`  Press Enter to continue...`);
        clearScreen();
        break;
        
      case '2':
        await installAllDependencies();
        console.log();
        await ask(`  Press Enter to continue...`);
        clearScreen();
        break;
        
      case '3':
        rl.close();
        log.section('Starting Full System');
        console.log();
        await startBackend();
        console.log();
        await startFrontend();
        console.log();
        log.ok('System running!');
        console.log(`\n  ${c.green}Open: http://localhost:${CONFIG.frontendPort}${c.reset}`);
        console.log(`  ${c.yellow}Press Ctrl+C to stop${c.reset}\n`);
        await new Promise(() => {}); // Wait forever
        break;
        
      case '4':
        rl.close();
        log.section('Starting Frontend');
        console.log();
        await startFrontend();
        console.log(`\n  ${c.yellow}Press Ctrl+C to stop${c.reset}\n`);
        await new Promise(() => {});
        break;
        
      case '5':
        rl.close();
        log.section('Starting AI Backend');
        console.log();
        await startBackend();
        console.log(`\n  ${c.yellow}Press Ctrl+C to stop${c.reset}\n`);
        await new Promise(() => {});
        break;
        
      case '6':
        await runP2PNetTest();
        console.log();
        await ask(`  Press Enter to continue...`);
        clearScreen();
        break;
        
      case '7':
        await runDetectionTest();
        console.log();
        await ask(`  Press Enter to continue...`);
        clearScreen();
        break;
        
      case '8':
        if (feUp) {
          openBrowser();
        } else {
          log.warn('Frontend not running - start it first');
        }
        await ask(`  Press Enter to continue...`);
        clearScreen();
        break;
      
      case '9':
        await showSettingsMenu();
        clearScreen();
        break;
      
      case 'z':
      case 'Z':
        await createDistributionZip();
        console.log();
        await ask(`  Press Enter to continue...`);
        clearScreen();
        break;
        
      case '0':
      case 'q':
      case 'Q':
        running = false;
        break;
        
      default:
        if (choice.trim()) {
          log.warn('Invalid choice');
          await sleep(1000);
        }
        clearScreen();
    }
  }
  
  rl.close();
  console.log(`\n  ${c.dim}Goodbye!${c.reset}\n`);
}

// ============================================
// CLEANUP
// ============================================
process.on('SIGINT', () => {
  console.log('\n');
  stopServers();
  process.exit(0);
});

process.on('SIGTERM', () => {
  stopServers();
  process.exit(0);
});

// ============================================
// MAIN ENTRY
// ============================================
async function main() {
  clearScreen();
  
  console.log(`
${c.cyan}${c.bold}╔══════════════════════════════════════════════════════╗
║      PlexIE OCC Demo Launcher v4.7                   ║
║      SSIG Stadium Safety System                      ║
╚══════════════════════════════════════════════════════╝${c.reset}
`);
  
  log.section('System Check');
  console.log();
  
  // Node.js check
  const nodeVer = getNodeVersion();
  if (!nodeVer) {
    log.err('Node.js not found!');
    log.info('Install from https://nodejs.org/');
    console.log();
    console.log('  Press any key to exit...');
    await new Promise(r => process.stdin.once('data', r));
    process.exit(1);
  }
  log.ok(`Node.js v${nodeVer}`);
  
  // npm check
  const npmVer = getNpmVersion();
  if (npmVer) {
    log.ok(`npm v${npmVer}`);
  } else {
    log.warn('npm not found');
  }
  
  // Python check  
  const pyCmd = getPythonCmd();
  const pyVer = getPythonVersion();
  if (pyVer) {
    log.ok(`Python v${pyVer} (${pyCmd})`);
  } else {
    log.warn('Python not found (needed for AI detection)');
    log.info('Install from https://python.org/');
  }
  
  // pip check
  const pipCmd = getPipCmd();
  if (pipCmd) {
    log.ok(`pip available (${pipCmd})`);
  } else if (pyCmd) {
    log.warn('pip not found');
  }
  
  // Check for videos
  console.log();
  const videoDir = path.join(__dirname, CONFIG.videoDir);
  let videoCount = 0;
  let firstVideo = null;
  for (const v of CONFIG.requiredVideos) {
    const vp = path.join(videoDir, v);
    if (fileExists(vp) && getFileSize(vp) > 10000) {
      videoCount++;
      if (!firstVideo) firstVideo = v;
    }
  }
  
  if (videoCount > 0) {
    log.ok(`Videos: ${videoCount} found in ${CONFIG.videoDir}/`);
    log.info(`  First video: ${firstVideo}`);
  } else {
    log.warn('No videos found!');
    log.info(`Add video files to: ${CONFIG.videoDir}/`);
    log.info('Name them: cam1.mp4, cam2.mp4, etc.');
  }
  
  // Check npm modules
  const hasNpmModules = fileExists(path.join(__dirname, 'node_modules', 'vite'));
  if (!hasNpmModules) {
    console.log();
    log.warn('npm modules not installed');
    log.info('Select option [2] to install dependencies');
  }
  
  // Check Python packages
  if (pyCmd) {
    const ultra = checkPythonPackage('ultralytics');
    const cv2 = checkPythonPackage('cv2');
    if (!ultra.installed || !cv2.installed) {
      console.log();
      log.warn('Python AI packages not installed');
      log.info('Select option [2] to install dependencies');
    }
  }
  
  await showMenu();
}

// Run
main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
