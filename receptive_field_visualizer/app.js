/**
 * Main 3D SubQ Receptive Field Application
 * Uses Three.js to render interactive 3D tokens, connection arcs/pyramids, and animated signal pulses.
 */

(function() {
  // Application State
  const state = {
    seqLen: 96,
    K: 5,
    mode: 'multiscale', // 'multiscale' or 'local'
    isCausal: false,
    maxHops: 4,
    currentHop: 1,
    selectedToken: 48,
    viewMode: 'cone', // 'cone' (temporal stack) or 'arcs' (flat track)
    isPlaying: false,
    playbackSpeed: 1.0,
    showSignals: true
  };

  const HOP_COLORS = [
    new THREE.Color(0x38bdf8), // Hop 0: Sky Blue (raw tokens)
    new THREE.Color(0x00f0ff), // Hop 1: Neon Cyan
    new THREE.Color(0xa855f7), // Hop 2: Vivid Purple
    new THREE.Color(0xf59e0b), // Hop 3: Neon Amber
    new THREE.Color(0xf43f5e), // Hop 4: Electric Rose
    new THREE.Color(0x10b981)  // Hop 5: Mint Green
  ];

  // Three.js Globals
  let scene, camera, renderer, controls;
  let particleSystem;
  let graphEngine;
  let coneData = null;

  // Visual Objects
  let tokenMeshes = []; // Array of { mesh, idx, tier }
  let edgeGroup = new THREE.Group();
  let tokenGroup = new THREE.Group();
  let gridHelper = null;
  let raycaster, mouse;
  let hoveredToken = null;
  let clock = new THREE.Clock();

  // Spacing constants
  const TOKEN_SPACING = 1.6;
  const TIER_HEIGHT = 6.0;

  // Initialize
  function init() {
    const container = document.getElementById('canvas-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x08090e);
    scene.fog = new THREE.FogExp2(0x08090e, 0.006);

    // Camera
    camera = new THREE.PerspectiveCamera(45, width / height, 0.5, 1000);
    camera.position.set(0, 28, 85);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    container.appendChild(renderer.domElement);

    // OrbitControls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.screenSpacePanning = true; // Essential: allows panning left/right across screen
    controls.maxPolarAngle = Math.PI / 2 + 0.05;
    controls.minDistance = 5;
    controls.maxDistance = 350;
    controls.target.set(0, 8, 0);

    // Default mouse button setup
    controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN
    };
    controls.touches = {
      ONE: THREE.TOUCH.ROTATE,
      TWO: THREE.TOUCH.DOLLY_PAN
    };

    // Lights
    const ambientLight = new THREE.AmbientLight(0x20293a, 1.9);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.3);
    keyLight.position.set(20, 50, 40);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x00d8ff, 0.9);
    fillLight.position.set(-30, 30, -30);
    scene.add(fillLight);

    // Grid Floor
    gridHelper = new THREE.GridHelper(300, 100, 0x1f293d, 0x0e1422);
    gridHelper.position.y = -0.6;
    scene.add(gridHelper);

    // Scene Groups
    scene.add(tokenGroup);
    scene.add(edgeGroup);

    // Particle System for signal pulses
    particleSystem = new SignalParticleSystem(scene);

    // Raycaster for mouse interaction
    raycaster = new THREE.Raycaster();
    mouse = new THREE.Vector2();

    // Graph Engine
    graphEngine = new ReceptiveFieldGraph({
      seqLen: state.seqLen,
      K: state.K,
      mode: state.mode,
      isCausal: state.isCausal,
      maxHops: state.maxHops
    });

    if (state.selectedToken >= state.seqLen) {
      state.selectedToken = Math.floor(state.seqLen / 2);
    }

    // Build scene elements
    rebuildGraphAndTokens();

    // Event Listeners
    window.addEventListener('resize', onWindowResize);
    renderer.domElement.addEventListener('mousemove', onMouseMove);
    renderer.domElement.addEventListener('click', onClick);

    setupUI();

    // Start animation loop
    animate();
  }

  function rebuildGraphAndTokens() {
    // Reconfigure graph engine
    graphEngine.seqLen = state.seqLen;
    graphEngine.K = state.K;
    graphEngine.mode = state.mode;
    graphEngine.isCausal = state.isCausal;
    graphEngine.maxHops = state.maxHops;

    if (state.selectedToken >= state.seqLen) {
      state.selectedToken = Math.floor(state.seqLen / 2);
    }

    // Compute cone for current hop
    coneData = graphEngine.computeCone(state.selectedToken, state.currentHop);

    // Rebuild Token Meshes
    buildTokenMeshes();

    // Update Visualization (edges, particles, highlights)
    updateVisualization();
  }

  function buildTokenMeshes() {
    // Clear old tokens
    while (tokenGroup.children.length > 0) {
      const obj = tokenGroup.children[0];
      tokenGroup.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    }
    tokenMeshes = [];

    const totalTiers = state.viewMode === 'cone' ? state.maxHops + 1 : 1;
    const geometry = new THREE.BoxGeometry(1.15, 0.75, 1.15);

    for (let tier = 0; tier < totalTiers; tier++) {
      const y = state.viewMode === 'cone' ? tier * TIER_HEIGHT : 0;

      for (let i = 0; i < state.seqLen; i++) {
        const x = (i - (state.seqLen - 1) / 2) * TOKEN_SPACING;
        const z = 0;

        const material = new THREE.MeshStandardMaterial({
          color: 0x1e293b,
          roughness: 0.35,
          metalness: 0.2,
          emissive: 0x000000
        });

        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.set(x, y, z);
        mesh.castShadow = true;
        mesh.receiveShadow = true;

        mesh.userData = {
          tokenIdx: i,
          tier: tier,
          isToken: true
        };

        tokenGroup.add(mesh);
        tokenMeshes.push({
          mesh: mesh,
          idx: i,
          tier: tier
        });
      }
    }
  }

  function updateVisualization() {
    if (!coneData) return;

    // Clear old edges
    while (edgeGroup.children.length > 0) {
      const obj = edgeGroup.children[0];
      edgeGroup.remove(obj);
      if (obj.geometry) obj.geometry.dispose();
      if (obj.material) obj.material.dispose();
    }

    const tierActiveSets = {};
    for (let h = 0; h <= state.currentHop; h++) {
      tierActiveSets[h] = new Set(coneData.tierActive[h] || []);
    }
    const rawSet = new Set(coneData.rawCovered);

    // Update Token Meshes materials
    for (const tObj of tokenMeshes) {
      const idx = tObj.idx;
      const tier = tObj.tier;
      const mesh = tObj.mesh;

      const isQuery = (idx === state.selectedToken);

      if (state.viewMode === 'cone') {
        const isTierActive = (tier <= state.currentHop);
        const isActiveInTier = isTierActive && tierActiveSets[tier] && tierActiveSets[tier].has(idx);

        if (isQuery && tier === state.currentHop) {
          // Selected query at current top tier
          mesh.material.color.setHex(0xffffff);
          mesh.material.emissive.setHex(0xfbbf24);
          mesh.material.emissiveIntensity = 1.0;
          mesh.scale.set(1.5, 1.8, 1.5);
        } else if (isActiveInTier) {
          const color = HOP_COLORS[tier] || HOP_COLORS[HOP_COLORS.length - 1];
          mesh.material.color.copy(color);
          mesh.material.emissive.copy(color);
          mesh.material.emissiveIntensity = 0.65;
          mesh.scale.set(1.25, 1.25, 1.25);
        } else {
          // Inactive / unreached
          mesh.material.color.setHex(0x1a2233);
          mesh.material.emissive.setHex(0x000000);
          mesh.material.emissiveIntensity = 0;
          mesh.scale.set(1.0, 1.0, 1.0);
        }
      } else {
        // Arc mode (single layer runway)
        if (isQuery) {
          mesh.material.color.setHex(0xffffff);
          mesh.material.emissive.setHex(0xfbbf24);
          mesh.material.emissiveIntensity = 1.0;
          mesh.scale.set(1.5, 2.2, 1.5);
        } else if (rawSet.has(idx)) {
          // Reached token
          const color = HOP_COLORS[state.currentHop] || HOP_COLORS[HOP_COLORS.length - 1];
          mesh.material.color.copy(color);
          mesh.material.emissive.copy(color);
          mesh.material.emissiveIntensity = 0.7;
          mesh.scale.set(1.25, 1.4, 1.25);
        } else {
          mesh.material.color.setHex(0x1a2233);
          mesh.material.emissive.setHex(0x000000);
          mesh.material.emissiveIntensity = 0;
          mesh.scale.set(1.0, 1.0, 1.0);
        }
      }
    }

    // Build 3D Connection Edges and Particle Paths
    const particlePaths = [];

    if (state.viewMode === 'cone') {
      // Cone mode: Connect from tier h-1 to tier h
      for (const edge of coneData.edges) {
        const fromIdx = edge.from;
        const toIdx = edge.to;
        const fromTier = edge.fromTier;
        const toTier = edge.toTier;
        const hop = edge.hop;

        const x1 = (fromIdx - (state.seqLen - 1) / 2) * TOKEN_SPACING;
        const y1 = fromTier * TIER_HEIGHT;
        const z1 = 0;

        const x2 = (toIdx - (state.seqLen - 1) / 2) * TOKEN_SPACING;
        const y2 = toTier * TIER_HEIGHT;
        const z2 = 0;

        const startPos = new THREE.Vector3(x1, y1 + 0.4, z1);
        const endPos = new THREE.Vector3(x2, y2 - 0.4, z2);

        const hopColor = HOP_COLORS[hop] || HOP_COLORS[HOP_COLORS.length - 1];
        const edgeMaterial = new THREE.LineBasicMaterial({
          color: hopColor,
          transparent: true,
          opacity: 0.65,
          linewidth: 2
        });

        const curve = new THREE.LineCurve3(startPos, endPos);
        const geom = new THREE.BufferGeometry().setFromPoints(curve.getPoints(8));
        const line = new THREE.Line(geom, edgeMaterial);
        edgeGroup.add(line);

        if (state.showSignals) {
          particlePaths.push({
            curve: curve,
            color: hopColor,
            hop: hop,
            targetPos: endPos
          });
        }
      }
    } else {
      // Dynamic 3D Arc overhead mode
      // Connect each reached token in rawCovered to the selected token
      const toX = (state.selectedToken - (state.seqLen - 1) / 2) * TOKEN_SPACING;
      const endPos = new THREE.Vector3(toX, 0.5, 0);

      for (const edge of coneData.edges) {
        const fromIdx = edge.from;
        const toIdx = edge.to;
        const hop = edge.hop;

        const x1 = (fromIdx - (state.seqLen - 1) / 2) * TOKEN_SPACING;
        const x2 = (toIdx - (state.seqLen - 1) / 2) * TOKEN_SPACING;
        const dist = Math.abs(x2 - x1);

        const startPos = new THREE.Vector3(x1, 0.45, 0);
        const targetPos = new THREE.Vector3(x2, 0.45, 0);

        const arcHeight = Math.max(3.2, Math.sqrt(dist) * 2.8);
        const midPos = new THREE.Vector3(
          (x1 + x2) / 2,
          arcHeight,
          Math.sin((x1 + x2) * 0.04) * 3.5
        );

        const curve = new THREE.QuadraticBezierCurve3(startPos, midPos, targetPos);
        const hopColor = HOP_COLORS[hop] || HOP_COLORS[HOP_COLORS.length - 1];

        const edgeMaterial = new THREE.LineBasicMaterial({
          color: hopColor,
          transparent: true,
          opacity: 0.6,
          linewidth: 2
        });

        const geom = new THREE.BufferGeometry().setFromPoints(curve.getPoints(24));
        const line = new THREE.Line(geom, edgeMaterial);
        edgeGroup.add(line);

        if (state.showSignals) {
          particlePaths.push({
            curve: curve,
            color: hopColor,
            hop: hop,
            targetPos: targetPos
          });
        }
      }
    }

    // Set particle paths
    particleSystem.setPaths(particlePaths);

    // Update HUD metrics
    updateHUD();
  }

  function updateHUD() {
    if (!coneData) return;

    const coveredCount = coneData.receptiveFieldCount;
    const totalCount = state.seqLen;
    const coveragePct = coneData.coveragePct.toFixed(1);

    document.getElementById('stat-token-id').textContent = `#${state.selectedToken}`;
    document.getElementById('stat-current-hop').textContent = `${state.currentHop} / ${state.maxHops}`;
    document.getElementById('stat-branching-k').textContent = `${state.K} tokens`;
    document.getElementById('stat-rf-count').textContent = `${coveredCount} / ${totalCount}`;
    document.getElementById('stat-coverage-pct').textContent = `${coveragePct}%`;
    document.getElementById('progress-bar-fill').style.width = `${coveragePct}%`;

    // Mathematical intuition badge
    const theoreticalPaths = Math.pow(state.K, state.currentHop);
    const badgeText = state.mode === 'multiscale'
      ? `SubQ Multi-Scale: K^T = ${state.K}^${state.currentHop} = ${theoreticalPaths.toLocaleString()} paths | Reaching ${coveredCount} tokens!`
      : `Local Window: T × K linear expansion (${coveredCount} tokens reached)`;
    document.getElementById('scaling-badge').textContent = badgeText;

    // Timeline dots highlight
    const stepButtons = document.querySelectorAll('.timeline-step');
    stepButtons.forEach((btn, idx) => {
      if (idx === state.currentHop) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    // Coverage fill color
    const fill = document.getElementById('progress-bar-fill');
    if (coneData.coveragePct >= 99.0) {
      fill.style.backgroundColor = '#10b981';
    } else if (coneData.coveragePct >= 50.0) {
      fill.style.backgroundColor = '#00f0ff';
    } else {
      fill.style.backgroundColor = '#a855f7';
    }
  }

  // Mouse Interaction: Hover & Raycast Selection
  function onMouseMove(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(tokenGroup.children);

    const tooltip = document.getElementById('token-tooltip');

    if (intersects.length > 0) {
      const hit = intersects[0].object;
      if (hit.userData && hit.userData.isToken) {
        hoveredToken = hit.userData.tokenIdx;
        const tier = hit.userData.tier;
        renderer.domElement.style.cursor = 'pointer';

        const offset = hoveredToken - state.selectedToken;
        const dist = Math.abs(offset);
        tooltip.style.display = 'block';
        tooltip.style.left = `${event.clientX + 16}px`;
        tooltip.style.top = `${event.clientY - 30}px`;
        tooltip.innerHTML = `<strong>Token #${hoveredToken}</strong> ${state.viewMode === 'cone' ? `(Tier ${tier})` : ''}<br><span style="color:#94a3b8">Offset: ${offset > 0 ? '+' : ''}${offset} (${dist} tokens away)</span>`;
        return;
      }
    }

    hoveredToken = null;
    renderer.domElement.style.cursor = 'default';
    tooltip.style.display = 'none';
  }

  function onClick(event) {
    const rect = renderer.domElement.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(tokenGroup.children);

    if (intersects.length > 0) {
      const hit = intersects[0].object;
      if (hit.userData && hit.userData.isToken) {
        selectToken(hit.userData.tokenIdx);
      }
    }
  }

  function selectToken(idx) {
    state.selectedToken = idx;
    coneData = graphEngine.computeCone(state.selectedToken, state.currentHop);
    updateVisualization();

    // Trigger ring pulse at target
    const x = (idx - (state.seqLen - 1) / 2) * TOKEN_SPACING;
    const y = state.viewMode === 'cone' ? state.currentHop * TIER_HEIGHT : 0;
    particleSystem.triggerPulse(new THREE.Vector3(x, y + 0.4, 0), 0xfbbf24);
  }

  function setHop(hop) {
    state.currentHop = Math.max(0, Math.min(state.maxHops, hop));
    coneData = graphEngine.computeCone(state.selectedToken, state.currentHop);
    updateVisualization();
  }

  function stepHop(direction) {
    let nextHop = state.currentHop + direction;
    if (nextHop > state.maxHops) nextHop = 0;
    if (nextHop < 0) nextHop = state.maxHops;
    setHop(nextHop);
  }

  function togglePlay() {
    state.isPlaying = !state.isPlaying;
    const playBtn = document.getElementById('btn-play');
    if (state.isPlaying) {
      playBtn.innerHTML = '❚❚ Pause';
      playBtn.classList.add('playing');
    } else {
      playBtn.innerHTML = '▶ Play';
      playBtn.classList.remove('playing');
    }
  }

  function setViewMode(mode) {
    state.viewMode = mode;
    buildTokenMeshes();
    coneData = graphEngine.computeCone(state.selectedToken, state.currentHop);
    updateVisualization();

    // Smooth camera transition
    if (mode === 'cone') {
      controls.target.set(0, (state.maxHops * TIER_HEIGHT) / 2, 0);
      camera.position.set(0, 32, 90);
    } else {
      controls.target.set(0, 4, 0);
      camera.position.set(0, 22, 75);
    }
    controls.update();

    document.getElementById('btn-view-cone').classList.toggle('active', mode === 'cone');
    document.getElementById('btn-view-arcs').classList.toggle('active', mode === 'arcs');
  }

  // Mouse Drag Mode (Rotate vs Pan)
  let mouseDragMode = 'rotate';
  function setMouseMode(mode) {
    mouseDragMode = mode;
    const btnRotate = document.getElementById('btn-mode-rotate');
    const btnPan = document.getElementById('btn-mode-pan');

    if (mode === 'pan') {
      controls.mouseButtons.LEFT = THREE.MOUSE.PAN;
      controls.touches.ONE = THREE.TOUCH.PAN;
      if (btnPan) btnPan.classList.add('active');
      if (btnRotate) btnRotate.classList.remove('active');
      renderer.domElement.style.cursor = 'grab';
    } else {
      controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
      controls.touches.ONE = THREE.TOUCH.ROTATE;
      if (btnRotate) btnRotate.classList.add('active');
      if (btnPan) btnPan.classList.remove('active');
      renderer.domElement.style.cursor = 'default';
    }
  }

  // Sidebar Collapse / Expand
  let isSidebarCollapsed = false;
  function toggleSidebar(forceState) {
    if (forceState !== undefined) {
      isSidebarCollapsed = forceState;
    } else {
      isSidebarCollapsed = !isSidebarCollapsed;
    }
    const sidebar = document.getElementById('controls-sidebar');
    const toggleBtn = document.getElementById('btn-toggle-sidebar');
    const icon = document.getElementById('sidebar-toggle-icon');

    if (sidebar) {
      if (isSidebarCollapsed) {
        sidebar.classList.add('collapsed');
        if (toggleBtn) toggleBtn.classList.remove('active');
        if (icon) icon.textContent = '⚙️';
      } else {
        sidebar.classList.remove('collapsed');
        if (toggleBtn) toggleBtn.classList.add('active');
        if (icon) icon.textContent = '◀';
      }
    }
  }

  // Camera Pan & Focus Controls
  function panCamera(dx) {
    controls.target.x += dx;
    camera.position.x += dx;
    controls.update();
  }

  function focusToken(idx) {
    if (idx === undefined) idx = state.selectedToken;
    const targetX = (idx - (state.seqLen - 1) / 2) * TOKEN_SPACING;
    const dx = targetX - controls.target.x;
    controls.target.x = targetX;
    camera.position.x += dx;
    controls.update();

    // Visual feedback ring
    const y = state.viewMode === 'cone' ? state.currentHop * TIER_HEIGHT : 0;
    particleSystem.triggerPulse(new THREE.Vector3(targetX, y + 0.4, 0), 0xfbbf24);
  }

  function resetCamera() {
    const yCenter = state.viewMode === 'cone' ? (state.maxHops * TIER_HEIGHT) / 2 : 4;
    controls.target.set(0, yCenter, 0);
    if (state.viewMode === 'cone') {
      camera.position.set(0, 32, 90);
    } else {
      camera.position.set(0, 22, 75);
    }
    controls.update();
  }

  function setupUI() {
    // Mouse Mode switchers
    const btnModeRotate = document.getElementById('btn-mode-rotate');
    const btnModePan = document.getElementById('btn-mode-pan');
    if (btnModeRotate) btnModeRotate.addEventListener('click', () => setMouseMode('rotate'));
    if (btnModePan) btnModePan.addEventListener('click', () => setMouseMode('pan'));

    // Sidebar Toggle
    const btnToggleSidebar = document.getElementById('btn-toggle-sidebar');
    const btnCloseSidebar = document.getElementById('btn-close-sidebar');
    if (btnToggleSidebar) btnToggleSidebar.addEventListener('click', () => toggleSidebar());
    if (btnCloseSidebar) btnCloseSidebar.addEventListener('click', () => toggleSidebar(true));

    // Camera Navigation Buttons
    const btnPanLeft = document.getElementById('btn-pan-left');
    const btnPanRight = document.getElementById('btn-pan-right');
    const btnFocus = document.getElementById('btn-focus-token');
    const btnResetCam = document.getElementById('btn-reset-cam');

    if (btnPanLeft) btnPanLeft.addEventListener('click', () => panCamera(-8 * TOKEN_SPACING));
    if (btnPanRight) btnPanRight.addEventListener('click', () => panCamera(8 * TOKEN_SPACING));
    if (btnFocus) btnFocus.addEventListener('click', () => focusToken(state.selectedToken));
    if (btnResetCam) btnResetCam.addEventListener('click', resetCamera);

    // Keyboard Shortcuts
    window.addEventListener('keydown', (e) => {
      if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;

      if (e.key === 'ArrowLeft' || e.key === 'a' || e.key === 'A') {
        panCamera(-6 * TOKEN_SPACING);
      } else if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') {
        panCamera(6 * TOKEN_SPACING);
      } else if (e.key === 'f' || e.key === 'F') {
        focusToken(state.selectedToken);
      } else if (e.key === 'h' || e.key === 'H') {
        toggleSidebar();
      } else if (e.key === ' ' || e.key === 'Spacebar') {
        togglePlay();
        e.preventDefault();
      }
    });

    // Horizontal Mouse Wheel / Trackpad panning
    renderer.domElement.addEventListener('wheel', (e) => {
      if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
        const dx = (e.deltaX !== 0 ? e.deltaX : e.deltaY) * 0.08;
        panCamera(dx);
        e.preventDefault();
      }
    }, { passive: false });

    // Play/Pause & Stepping
    document.getElementById('btn-play').addEventListener('click', togglePlay);
    document.getElementById('btn-prev').addEventListener('click', () => stepHop(-1));
    document.getElementById('btn-next').addEventListener('click', () => stepHop(1));
    document.getElementById('btn-reset').addEventListener('click', () => setHop(0));

    // Timeline Buttons
    const stepContainer = document.getElementById('timeline-steps');
    stepContainer.innerHTML = '';
    for (let h = 0; h <= state.maxHops; h++) {
      const btn = document.createElement('button');
      btn.className = `timeline-step ${h === state.currentHop ? 'active' : ''}`;
      btn.innerHTML = `<span class="hop-num">${h}</span><span class="hop-label">Hop ${h}</span>`;
      btn.addEventListener('click', () => setHop(h));
      stepContainer.appendChild(btn);
    }

    // View Mode buttons
    document.getElementById('btn-view-cone').addEventListener('click', () => setViewMode('cone'));
    document.getElementById('btn-view-arcs').addEventListener('click', () => setViewMode('arcs'));

    // Mode Selector (SubQ Multi-scale vs Local Window)
    const modeSelect = document.getElementById('select-mode');
    modeSelect.value = state.mode;
    modeSelect.addEventListener('change', (e) => {
      state.mode = e.target.value;
      rebuildGraphAndTokens();
    });

    // K Slider
    const kSlider = document.getElementById('slider-k');
    const kVal = document.getElementById('val-k');
    kSlider.value = state.K;
    kVal.textContent = state.K;
    kSlider.addEventListener('input', (e) => {
      state.K = parseInt(e.target.value, 10);
      kVal.textContent = state.K;
      rebuildGraphAndTokens();
    });

    // Sequence Length Slider
    const seqSlider = document.getElementById('slider-seq');
    const seqVal = document.getElementById('val-seq');
    seqSlider.value = state.seqLen;
    seqVal.textContent = state.seqLen;
    seqSlider.addEventListener('input', (e) => {
      state.seqLen = parseInt(e.target.value, 10);
      seqVal.textContent = state.seqLen;
      rebuildGraphAndTokens();
    });

    // Max Hops Slider
    const hopsSlider = document.getElementById('slider-hops');
    const hopsVal = document.getElementById('val-hops');
    hopsSlider.value = state.maxHops;
    hopsVal.textContent = state.maxHops;
    hopsSlider.addEventListener('input', (e) => {
      state.maxHops = parseInt(e.target.value, 10);
      hopsVal.textContent = state.maxHops;
      if (state.currentHop > state.maxHops) state.currentHop = state.maxHops;
      setupUI();
      rebuildGraphAndTokens();
    });

    // Speed Slider
    const speedSlider = document.getElementById('slider-speed');
    speedSlider.value = state.playbackSpeed;
    speedSlider.addEventListener('input', (e) => {
      state.playbackSpeed = parseFloat(e.target.value);
      particleSystem.speed = state.playbackSpeed;
    });

    // Causal Checkbox
    const causalCheck = document.getElementById('check-causal');
    causalCheck.checked = state.isCausal;
    causalCheck.addEventListener('change', (e) => {
      state.isCausal = e.target.checked;
      rebuildGraphAndTokens();
    });

    // Preset buttons
    document.getElementById('preset-subq').addEventListener('click', () => {
      state.mode = 'multiscale';
      state.K = 5;
      modeSelect.value = 'multiscale';
      kSlider.value = 5;
      kVal.textContent = 5;
      rebuildGraphAndTokens();
      setHop(3);
    });

    document.getElementById('preset-local').addEventListener('click', () => {
      state.mode = 'local';
      state.K = 5;
      modeSelect.value = 'local';
      kSlider.value = 5;
      kVal.textContent = 5;
      rebuildGraphAndTokens();
      setHop(3);
    });

    document.getElementById('preset-wide').addEventListener('click', () => {
      state.mode = 'multiscale';
      state.K = 8;
      modeSelect.value = 'multiscale';
      kSlider.value = 8;
      kVal.textContent = 8;
      rebuildGraphAndTokens();
      setHop(3);
    });
  }

  function onWindowResize() {
    const container = document.getElementById('canvas-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  }

  // Playback Auto-Stepper
  let playTimer = 0;
  const PLAY_INTERVAL = 1.6;

  function animate() {
    requestAnimationFrame(animate);

    const delta = clock.getDelta();

    // Auto Playback
    if (state.isPlaying) {
      playTimer += delta * state.playbackSpeed;
      if (playTimer >= PLAY_INTERVAL) {
        playTimer = 0;
        stepHop(1);
      }
    }

    // Update particle signal system
    particleSystem.update(delta);

    // Subtle idle animation on selected token
    for (const tObj of tokenMeshes) {
      if (tObj.idx === state.selectedToken) {
        const pulse = 1.0 + Math.sin(clock.getElapsedTime() * 4.0) * 0.08;
        tObj.mesh.scale.set(1.5 * pulse, (state.viewMode === 'cone' ? 1.8 : 2.2) * pulse, 1.5 * pulse);
      }
    }

    // Orbit controls update
    controls.update();

    // Render scene
    renderer.render(scene, camera);
  }

  // Run on DOM ready
  window.addEventListener('DOMContentLoaded', init);
})();
