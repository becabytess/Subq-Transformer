/**
 * Signal Pulse Animation Engine
 * Creates and updates glowing signal packets traveling along 3D connection lines/arcs
 * to visualize information cascading towards the target token.
 */

class SignalParticleSystem {
  constructor(scene) {
    this.scene = scene;
    this.particles = [];
    this.paths = [];
    this.speed = 1.0;
    this.active = true;

    // Create a particle texture (glowing round circle)
    this.texture = this._createParticleTexture();

    // Reusable particle material
    this.material = new THREE.PointsMaterial({
      size: 1.4,
      map: this.texture,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      vertexColors: true
    });

    // Particle geometry
    this.maxParticles = 1200;
    this.geometry = new THREE.BufferGeometry();
    this.positions = new Float32Array(this.maxParticles * 3);
    this.colors = new Float32Array(this.maxParticles * 3);

    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setAttribute('color', new THREE.BufferAttribute(this.colors, 3));

    this.points = new THREE.Points(this.geometry, this.material);
    this.points.frustumCulled = false;
    this.scene.add(this.points);

    // Shockwave pulse rings
    this.pulses = [];
    this.pulseMaterial = new THREE.MeshBasicMaterial({
      color: 0x00ffff,
      transparent: true,
      opacity: 0.8,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });
    this.ringGeometry = new THREE.RingGeometry(0.2, 0.4, 24);
  }

  _createParticleTexture() {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');

    const gradient = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    gradient.addColorStop(0, 'rgba(255, 255, 255, 1)');
    gradient.addColorStop(0.2, 'rgba(120, 240, 255, 0.9)');
    gradient.addColorStop(0.6, 'rgba(0, 160, 255, 0.4)');
    gradient.addColorStop(1, 'rgba(0, 40, 100, 0)');

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, 64, 64);

    const texture = new THREE.CanvasTexture(canvas);
    return texture;
  }

  /**
   * Set active paths for particles to follow
   * paths: Array of { curve: THREE.Curve, color: THREE.Color, hop: number, targetPos: THREE.Vector3 }
   */
  setPaths(paths) {
    this.paths = paths;
    this.particles = [];

    if (!paths || paths.length === 0) {
      this._clearAllPositions();
      return;
    }

    // Allocate particles per path
    const particlesPerPath = Math.max(1, Math.min(6, Math.floor(this.maxParticles / paths.length)));
    
    for (let i = 0; i < paths.length; i++) {
      const pInfo = paths[i];
      for (let k = 0; k < particlesPerPath; k++) {
        if (this.particles.length >= this.maxParticles) break;
        this.particles.push({
          curve: pInfo.curve,
          targetPos: pInfo.targetPos,
          progress: (k / particlesPerPath) + Math.random() * 0.1,
          speed: 0.25 + Math.random() * 0.15,
          color: pInfo.color || new THREE.Color(0x00ffff),
          hop: pInfo.hop
        });
      }
    }
  }

  _clearAllPositions() {
    for (let i = 0; i < this.maxParticles * 3; i++) {
      this.positions[i] = 999999;
    }
    this.geometry.attributes.position.needsUpdate = true;
  }

  triggerPulse(pos, colorHex = 0x00ffff) {
    const mesh = new THREE.Mesh(this.ringGeometry, this.pulseMaterial.clone());
    mesh.material.color.setHex(colorHex);
    mesh.position.copy(pos);
    mesh.rotation.x = -Math.PI / 2;
    mesh.scale.set(1, 1, 1);
    this.scene.add(mesh);

    this.pulses.push({
      mesh,
      scale: 1,
      opacity: 0.85
    });
  }

  update(delta) {
    if (!this.active || this.particles.length === 0) {
      this._clearAllPositions();
      this._updatePulses(delta);
      return;
    }

    const posAttr = this.geometry.attributes.position;
    const colAttr = this.geometry.attributes.color;
    const effDelta = Math.min(delta, 0.1) * this.speed;

    let activeCount = 0;

    for (let i = 0; i < this.particles.length; i++) {
      const p = this.particles[i];
      p.progress += p.speed * effDelta;

      if (p.progress >= 1.0) {
        p.progress = p.progress % 1.0;
        // Optionally trigger a subtle pulse when arriving at target
        if (p.targetPos && Math.random() < 0.15) {
          this.triggerPulse(p.targetPos, p.color.getHex());
        }
      }

      const point = p.curve.getPoint(p.progress);
      if (point) {
        const idx3 = activeCount * 3;
        this.positions[idx3] = point.x;
        this.positions[idx3 + 1] = point.y;
        this.positions[idx3 + 2] = point.z;

        this.colors[idx3] = p.color.r;
        this.colors[idx3 + 1] = p.color.g;
        this.colors[idx3 + 2] = p.color.b;

        activeCount++;
      }
    }

    // Hide remaining unused slots
    for (let i = activeCount; i < this.maxParticles; i++) {
      const idx3 = i * 3;
      this.positions[idx3] = 999999;
      this.positions[idx3 + 1] = 999999;
      this.positions[idx3 + 2] = 999999;
    }

    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;

    this._updatePulses(delta);
  }

  _updatePulses(delta) {
    for (let i = this.pulses.length - 1; i >= 0; i--) {
      const p = this.pulses[i];
      p.scale += delta * 5.0;
      p.opacity -= delta * 1.8;

      p.mesh.scale.set(p.scale, p.scale, p.scale);
      p.mesh.material.opacity = Math.max(0, p.opacity);

      if (p.opacity <= 0) {
        this.scene.remove(p.mesh);
        p.mesh.geometry.dispose();
        p.mesh.material.dispose();
        this.pulses.splice(i, 1);
      }
    }
  }

  clear() {
    this.setPaths([]);
    for (const p of this.pulses) {
      this.scene.remove(p.mesh);
      p.mesh.geometry.dispose();
      p.mesh.material.dispose();
    }
    this.pulses = [];
  }
}

window.SignalParticleSystem = SignalParticleSystem;
