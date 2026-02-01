#!/usr/bin/env node

/**
 * PlexIE OCC Demo - Unified Launcher v8.0
 * ========================================
 * Features:
 * - Automatic video processing with JSON output
 * - Frame-synced detection data for alerts
 * - No manual Python commands needed
 */

const { spawn, execSync, spawnSync } = require('child_process');
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
  cleanVideoDir: 'public/videos/clean',
  annotatedVideoDir: 'public/videos/annotated',
  detectionsDir: 'public/videos/detections',
  backendDir: 'backend',
  settingsFile: 'plexie-settings.json',
};

// Default AI Settings
const DEFAULT_SETTINGS = {
  DETECTION_THRESHOLD: 0.5,
  PROCESS_WIDTH: 640,
  FRAME_SKIP: 1,  // Process every frame for JSON
};

// Load settings
function loadSettings() {
  const settingsPath = path.join(__dirname, CONFIG.settingsFile);
  try {
    if (fs.existsSync(settingsPath)) {
      return { ...DEFAULT_SETTINGS, ...JSON.parse(fs.readFileSync(settingsPath, 'utf8')) };
    }
  } catch (e) {}
  return { ...DEFAULT_SETTINGS };
}

function saveSettings(settings) {
  try {
    fs.writeFileSync(path.join(__dirname, CONFIG.settingsFile), JSON.stringify(settings, null, 2));
    return true;
  } catch (e) { return false; }
}

let currentSettings = loadSettings();

// Colors
const c = {
  reset: '\x1b[0m', bold: '\x1b[1m', dim: '\x1b[2m',
  red: '\x1b[31m', green: '\x1b[32m', yellow: '\x1b[33m',
  blue: '\x1b[34m', cyan: '\x1b[36m', white: '\x1b[37m',
};

const log = {
  info: (m) => console.log(`  ${c.blue}ℹ${c.reset} ${m}`),
  ok: (m) => console.log(`  ${c.green}✓${c.reset} ${m}`),
  warn: (m) => console.log(`  ${c.yellow}⚠${c.reset} ${m}`),
  err: (m) => console.log(`  ${c.red}✗${c.reset} ${m}`),
  section: (m) => console.log(`\n${c.cyan}${c.bold}▶ ${m}${c.reset}`),
};

// Utilities
const fileExists = (p) => { try { fs.accessSync(p); return true; } catch { return false; } };
const getFileSize = (p) => { try { return fs.statSync(p).size; } catch { return 0; } };
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const clearScreen = () => { process.stdout.write('\x1b[2J\x1b[H'); };

// ============================================
// SYSTEM CHECKS
// ============================================
function getPythonCmd() {
  for (const cmd of ['python', 'python3', 'py']) {
    try {
      const result = spawnSync(cmd, ['--version'], { encoding: 'utf8', timeout: 5000 });
      if (result.status === 0) return cmd;
    } catch {}
  }
  return null;
}

function getPythonVersion() {
  const cmd = getPythonCmd();
  if (!cmd) return null;
  try {
    const result = spawnSync(cmd, ['--version'], { encoding: 'utf8' });
    const match = result.stdout?.match(/(\d+\.\d+\.\d+)/) || result.stderr?.match(/(\d+\.\d+\.\d+)/);
    return match ? match[1] : null;
  } catch { return null; }
}

function getNodeVersion() {
  try {
    const match = process.version.match(/v(\d+\.\d+\.\d+)/);
    return match ? match[1] : null;
  } catch { return null; }
}

function getNpmVersion() {
  try {
    return execSync('npm --version', { encoding: 'utf8', timeout: 5000 }).trim();
  } catch { return null; }
}

function checkPythonPackage(pkg) {
  const pyCmd = getPythonCmd();
  if (!pyCmd) return { installed: false };
  try {
    const result = spawnSync(pyCmd, ['-c', `import ${pkg}; print('ok')`], { encoding: 'utf8', timeout: 10000 });
    return { installed: result.status === 0 };
  } catch { return { installed: false }; }
}

async function isPortInUse(port) {
  return new Promise((resolve) => {
    const server = http.createServer();
    server.once('error', () => resolve(true));
    server.once('listening', () => { server.close(); resolve(false); });
    server.listen(port);
  });
}

// ============================================
// VIDEO DISCOVERY
// ============================================
function getCleanVideos() {
  const dir = path.join(__dirname, CONFIG.cleanVideoDir);
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).filter(f => f.endsWith('.mp4')).sort();
}

function getDetectionFiles() {
  const dir = path.join(__dirname, CONFIG.detectionsDir);
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir).filter(f => f.endsWith('_detections.json')).sort();
}

function getVideoStatus() {
  const cleanVideos = getCleanVideos();
  const detectionFiles = getDetectionFiles();
  
  const status = [];
  for (const video of cleanVideos) {
    const baseName = video.replace('.mp4', '');
    const jsonFile = `${baseName}_detections.json`;
    const hasJson = detectionFiles.includes(jsonFile);
    
    // Check JSON file size to ensure it's valid
    let jsonValid = false;
    if (hasJson) {
      const jsonPath = path.join(__dirname, CONFIG.detectionsDir, jsonFile);
      const size = getFileSize(jsonPath);
      jsonValid = size > 1000; // At least 1KB
    }
    
    status.push({
      video,
      baseName,
      hasJson: hasJson && jsonValid,
      jsonFile
    });
  }
  return status;
}

// ============================================
// VIDEO PROCESSING
// ============================================
async function processVideo(videoName, showProgress = true) {
  const pyCmd = getPythonCmd();
  if (!pyCmd) {
    log.err('Python not found!');
    return false;
  }
  
  const baseName = videoName.replace('.mp4', '');
  const inputPath = path.join(__dirname, CONFIG.cleanVideoDir, videoName);
  const outputDir = path.join(__dirname, CONFIG.detectionsDir);
  
  // Create output directory
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  if (showProgress) {
    log.info(`Processing ${videoName}...`);
  }
  
  return new Promise((resolve) => {
    const args = [
      'process_video_json.py',
      '--video_path', inputPath,
      '--output_dir', outputDir,
      '--output_name', baseName,
      '--threshold', String(currentSettings.DETECTION_THRESHOLD),
      '--frame_skip', String(currentSettings.FRAME_SKIP || 1),
      '--no_display'
    ];
    
    const proc = spawn(pyCmd, args, {
      cwd: path.join(__dirname, CONFIG.backendDir),
      stdio: showProgress ? 'inherit' : 'pipe'
    });
    
    proc.on('close', (code) => {
      if (code === 0) {
        if (showProgress) log.ok(`Processed ${videoName}`);
        resolve(true);
      } else {
        if (showProgress) log.err(`Failed to process ${videoName}`);
        resolve(false);
      }
    });
    
    proc.on('error', (err) => {
      log.err(`Error: ${err.message}`);
      resolve(false);
    });
  });
}

async function processAllVideos() {
  log.section('Processing Videos');
  console.log();
  
  const status = getVideoStatus();
  const toProcess = status.filter(s => !s.hasJson);
  
  if (toProcess.length === 0) {
    log.ok('All videos already processed!');
    return true;
  }
  
  log.info(`Found ${toProcess.length} videos to process`);
  console.log();
  
  let success = 0;
  for (const item of toProcess) {
    const result = await processVideo(item.video);
    if (result) success++;
    console.log();
  }
  
  log.info(`Processed ${success}/${toProcess.length} videos`);
  return success === toProcess.length;
}

// ============================================
// DEPENDENCY INSTALLATION
// ============================================
async function installAllDependencies() {
  log.section('Installing Dependencies');
  console.log();
  
  // npm install
  log.info('Installing npm packages...');
  try {
    execSync('npm install', { cwd: __dirname, stdio: 'inherit', timeout: 300000 });
    log.ok('npm packages installed');
  } catch (e) {
    log.err('npm install failed');
  }
  console.log();
  
  // pip install
  const pyCmd = getPythonCmd();
  if (pyCmd) {
    log.info('Installing Python packages...');
    const reqPath = path.join(__dirname, CONFIG.backendDir, 'requirements.txt');
    if (fileExists(reqPath)) {
      try {
        execSync(`${pyCmd} -m pip install -r "${reqPath}"`, { cwd: __dirname, stdio: 'inherit', timeout: 600000 });
        log.ok('Python packages installed');
      } catch (e) {
        log.err('pip install failed');
      }
    }
  }
}

function checkAllDependencies() {
  log.section('Checking Dependencies');
  console.log();
  
  // Node
  const nodeVer = getNodeVersion();
  if (nodeVer) log.ok(`Node.js v${nodeVer}`);
  else log.err('Node.js not found');
  
  // npm
  const npmVer = getNpmVersion();
  if (npmVer) log.ok(`npm v${npmVer}`);
  else log.warn('npm not found');
  
  // Python
  const pyCmd = getPythonCmd();
  const pyVer = getPythonVersion();
  if (pyVer) log.ok(`Python v${pyVer} (${pyCmd})`);
  else log.warn('Python not found');
  
  // Python packages
  if (pyCmd) {
    const torch = checkPythonPackage('torch');
    const cv2 = checkPythonPackage('cv2');
    if (torch.installed) log.ok('PyTorch installed');
    else log.warn('PyTorch not installed');
    if (cv2.installed) log.ok('OpenCV installed');
    else log.warn('OpenCV not installed');
  }
  
  // npm modules
  const hasVite = fileExists(path.join(__dirname, 'node_modules', 'vite'));
  if (hasVite) log.ok('npm modules installed');
  else log.warn('npm modules not installed - run option [2]');
  
  // Videos
  console.log();
  const status = getVideoStatus();
  log.info(`Videos found: ${status.length}`);
  const processed = status.filter(s => s.hasJson).length;
  log.info(`Processed: ${processed}/${status.length}`);
  if (processed < status.length) {
    log.warn(`${status.length - processed} videos need processing - run option [P]`);
  }
}

// ============================================
// SERVERS
// ============================================
let frontendProcess = null;

async function startFrontend() {
  if (frontendProcess) {
    log.warn('Frontend already running');
    return;
  }
  
  log.info('Starting frontend...');
  
  const npmCmd = process.platform === 'win32' ? 'npm.cmd' : 'npm';
  frontendProcess = spawn(npmCmd, ['run', 'dev'], {
    cwd: __dirname,
    stdio: 'inherit',
    shell: true
  });
  
  frontendProcess.on('close', () => { frontendProcess = null; });
  
  // Wait for server
  for (let i = 0; i < 30; i++) {
    await sleep(500);
    if (await isPortInUse(CONFIG.frontendPort)) {
      log.ok(`Frontend running at http://localhost:${CONFIG.frontendPort}`);
      return;
    }
  }
  log.warn('Frontend may still be starting...');
}

function stopServers() {
  if (frontendProcess) {
    frontendProcess.kill();
    frontendProcess = null;
  }
}

function openBrowser() {
  const url = `http://localhost:${CONFIG.frontendPort}`;
  const cmd = process.platform === 'win32' ? 'start' : process.platform === 'darwin' ? 'open' : 'xdg-open';
  try {
    execSync(`${cmd} ${url}`, { stdio: 'ignore' });
    log.ok('Opened browser');
  } catch {
    log.info(`Open manually: ${url}`);
  }
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
    const status = getVideoStatus();
    const processed = status.filter(s => s.hasJson).length;
    const needsProcessing = status.length - processed;
    
    console.log(`
${c.cyan}╔══════════════════════════════════════════════════════╗
║      ${c.bold}${c.white}PlexIE OCC Demo Launcher v8.0${c.reset}${c.cyan}                 ║
║         SSIG Stadium Safety System                    ║
╚══════════════════════════════════════════════════════╝${c.reset}

  ${c.bold}Status:${c.reset}
    Frontend: ${feUp ? `${c.green}● Running${c.reset}` : `${c.dim}○ Stopped${c.reset}`}
    Videos:   ${c.cyan}${status.length}${c.reset} found, ${c.green}${processed}${c.reset} processed${needsProcessing > 0 ? `, ${c.yellow}${needsProcessing} pending${c.reset}` : ''}

  ${c.bold}Setup:${c.reset}
    ${c.cyan}[1]${c.reset} Check Dependencies
    ${c.cyan}[2]${c.reset} Install All Dependencies

  ${c.bold}Video Processing:${c.reset}
    ${c.cyan}[P]${c.reset} ${c.yellow}Process Videos${c.reset} ${c.dim}(generate detection JSON)${c.reset}${needsProcessing > 0 ? ` ${c.yellow}← ${needsProcessing} pending${c.reset}` : ''}

  ${c.bold}Run:${c.reset}
    ${c.cyan}[3]${c.reset} ${c.green}Start Demo${c.reset} ${c.dim}(frontend + auto-process if needed)${c.reset}
    ${c.cyan}[4]${c.reset} Start Frontend Only

  ${c.bold}Tools:${c.reset}
    ${c.cyan}[8]${c.reset} Open in Browser
    ${c.cyan}[9]${c.reset} Video Status

    ${c.cyan}[0]${c.reset} Exit
`);
    
    const choice = await ask(`  ${c.bold}Enter choice:${c.reset} `);
    console.log();
    
    switch (choice.trim().toLowerCase()) {
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
        
      case 'p':
        await processAllVideos();
        console.log();
        await ask(`  Press Enter to continue...`);
        clearScreen();
        break;
        
      case '3':
        // Auto-process if needed
        const needProcess = getVideoStatus().filter(s => !s.hasJson);
        if (needProcess.length > 0) {
          log.info(`Processing ${needProcess.length} videos first...`);
          console.log();
          await processAllVideos();
          console.log();
        }
        
        rl.close();
        log.section('Starting Demo');
        console.log();
        await startFrontend();
        console.log();
        log.ok('System running!');
        console.log(`\n  ${c.green}Open: http://localhost:${CONFIG.frontendPort}${c.reset}`);
        console.log(`  ${c.yellow}Press Ctrl+C to stop${c.reset}\n`);
        
        // Auto-open browser
        await sleep(1000);
        openBrowser();
        
        await new Promise(() => {});
        break;
        
      case '4':
        rl.close();
        log.section('Starting Frontend');
        console.log();
        await startFrontend();
        console.log(`\n  ${c.yellow}Press Ctrl+C to stop${c.reset}\n`);
        await new Promise(() => {});
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
        log.section('Video Status');
        console.log();
        const vStatus = getVideoStatus();
        for (const v of vStatus) {
          const icon = v.hasJson ? `${c.green}✓${c.reset}` : `${c.yellow}○${c.reset}`;
          console.log(`  ${icon} ${v.video} ${v.hasJson ? c.dim + '(JSON ready)' + c.reset : c.yellow + '(needs processing)' + c.reset}`);
        }
        if (vStatus.length === 0) {
          log.warn(`No videos found in ${CONFIG.cleanVideoDir}/`);
        }
        console.log();
        await ask(`  Press Enter to continue...`);
        clearScreen();
        break;
        
      case '0':
      case 'q':
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
// MAIN
// ============================================
async function main() {
  clearScreen();
  
  console.log(`
${c.cyan}${c.bold}╔══════════════════════════════════════════════════════╗
║      PlexIE OCC Demo Launcher v8.0                   ║
║      SSIG Stadium Safety System                      ║
╚══════════════════════════════════════════════════════╝${c.reset}
`);
  
  log.section('System Check');
  console.log();
  
  // Node check
  const nodeVer = getNodeVersion();
  if (!nodeVer) {
    log.err('Node.js not found!');
    log.info('Install from https://nodejs.org/');
    await new Promise(r => process.stdin.once('data', r));
    process.exit(1);
  }
  log.ok(`Node.js v${nodeVer}`);
  
  // Python check
  const pyCmd = getPythonCmd();
  const pyVer = getPythonVersion();
  if (pyVer) {
    log.ok(`Python v${pyVer} (${pyCmd})`);
  } else {
    log.warn('Python not found (needed for video processing)');
  }
  
  // Videos check
  const status = getVideoStatus();
  if (status.length > 0) {
    const processed = status.filter(s => s.hasJson).length;
    log.ok(`Videos: ${status.length} found, ${processed} processed`);
  } else {
    log.warn(`No videos in ${CONFIG.cleanVideoDir}/`);
  }
  
  // npm modules
  const hasNpm = fileExists(path.join(__dirname, 'node_modules', 'vite'));
  if (!hasNpm) {
    console.log();
    log.warn('npm modules not installed - select [2] first');
  }
  
  await showMenu();
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
