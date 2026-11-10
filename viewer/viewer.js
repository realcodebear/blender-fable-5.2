// 3D viewer for the Blender Effort Test page.
// One viewer per scene section. Each loads a Draco GLB + a small JSON (camera, lights) + a sky panorama
// that tools/export_glb.py wrote from the .blend file.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { Reflector } from 'three/addons/objects/Reflector.js';

const BASE = new URL('../', import.meta.url);
const draco = new DRACOLoader().setDecoderPath(new URL('vendor/three/addons/libs/draco/gltf/', BASE).href);
const gltfLoader = new GLTFLoader().setDRACOLoader(draco);

// Per-scene look. Blender lights are in watts and the web renderer is not a path tracer,
// so these numbers were tuned by eye against the Cycles renders.
export const LOOK = {
  s1_v1: { exposure: 1.3, env: 1.2, light: 1.3, ambient: 0.5, bloom: 0.15, tone: 'Filmic', maxLights: 8, shadows: 2 },
  s1_v2: { exposure: 1.6, env: 1.2, light: 1.3, ambient: 0.5, bloom: 0.2, tone: 'Filmic', maxLights: 8, shadows: 2 },
  s2_v1: { exposure: 1.15, env: 1.3, light: 1.0, bloom: 0.2, maxLights: 8, shadows: 1 },
  s2_v2: { exposure: 1.15, env: 1.0, light: 1.0, bloom: 0.25, maxLights: 10, shadows: 1 },
  s3_v1: { exposure: 1.0, env: 1.0, light: 1.5, ambient: 1.0, bloom: 0.18, bloomRadius: 0.35, bloomThreshold: 1.5, maxLights: 10, shadows: 0, fog: [0x0a1020, 0.0025], wet: { y: 0.02, strength: 0.25 } },
  s3_v2: { exposure: 1.0, env: 1.0, light: 1.0, ambient: 0.22, emissive: 2.5, bloom: 0.3, bloomRadius: 0.35, bloomThreshold: 1.2, maxLights: 12, shadows: 0, fog: [0x0a0c18, 0.006], wet: { y: 0.02, strength: 0.25 } },
};

// Wet-street look: a mirror plane just above the road, added on top of the asphalt.
// It is a flat, even reflection, not the path-traced puddles from the Cycles render.
const WetShader = {
  name: 'WetShader',
  uniforms: { color: { value: null }, tDiffuse: { value: null }, textureMatrix: { value: null }, strength: { value: 0.3 } },
  vertexShader: `
    uniform mat4 textureMatrix; varying vec4 vUv; varying vec3 vW;
    void main() { vUv = textureMatrix * vec4(position, 1.0); vW = (modelMatrix * vec4(position, 1.0)).xyz; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `
    uniform sampler2D tDiffuse; uniform float strength; varying vec4 vUv; varying vec3 vW;
    void main() {
      vec3 base = texture2DProj(tDiffuse, vUv).rgb;
      float f = pow(1.0 - clamp(normalize(cameraPosition - vW).y, 0.0, 1.0), 3.0);
      gl_FragColor = vec4(min(base, vec3(6.0)) * strength * mix(0.25, 1.0, f), 1.0);
    }`,
};
function wetRoad(look, small) {
  const px = small ? 512 : 1024;
  const m = new Reflector(new THREE.PlaneGeometry(look.wet.size || 600, look.wet.size || 600), { textureWidth: px, textureHeight: px, shader: WetShader, multisample: 0 });
  m.material.uniforms.strength.value = look.wet.strength;
  m.material.transparent = true;
  m.material.blending = THREE.AdditiveBlending;
  m.material.depthWrite = false;
  m.material.polygonOffset = true; m.material.polygonOffsetFactor = -2; m.material.polygonOffsetUnits = -2;
  m.rotation.x = -Math.PI / 2;
  m.position.y = look.wet.y;
  m.renderOrder = 1;
  return m;
}

const TONE = { AgX: THREE.AgXToneMapping, Filmic: THREE.ACESFilmicToneMapping, Standard: THREE.LinearToneMapping, 'Khronos PBR Neutral': THREE.NeutralToneMapping };
const v3 = (a) => new THREE.Vector3(a[0], a[1], a[2]);

function buildLights(meta, look, scene, center, radius) {
  // Narrow, huge spots are searchlight beams that only show up in fog: skip them as real lights.
  const src = meta.lights.filter((L) => !(L.type === 'SPOT' && L.energy > 1e5));
  const small = window.matchMedia('(max-width: 760px), (pointer: coarse)').matches;
  const maxLights = small ? Math.min(look.maxLights, 6) : look.maxLights;
  const score = (L) => {
    if (L.type === 'SUN') return 1e9 * L.energy;
    const d = Math.max(v3(L.pos).distanceTo(center), 1);
    return (L.type === 'AREA' ? 4 : 1) * L.energy / (d * d);
  };
  src.sort((a, b) => score(b) - score(a));
  let shadows = look.shadows;
  const out = [];
  for (const L of src.slice(0, maxLights)) {
    const col = new THREE.Color().setRGB(L.color[0], L.color[1], L.color[2], THREE.LinearSRGBColorSpace);
    let light;
    const pos = v3(L.pos), dir = v3(L.dir);
    if (L.type === 'SUN') {
      light = new THREE.DirectionalLight(col, L.energy * look.light);
      light.position.copy(center).addScaledVector(dir, -radius * 2);
      light.target.position.copy(center);
    } else if (L.type === 'POINT') {
      light = new THREE.PointLight(col, (L.energy / (4 * Math.PI * Math.PI)) * look.light, 0, 2);
      light.position.copy(pos);
    } else {
      const area = L.type === 'AREA';
      const cd = area ? L.energy / (Math.PI * Math.PI) : L.energy / (4 * Math.PI * Math.PI);
      const ang = area ? 1.35 : Math.min(L.angle / 2, 1.5);
      light = new THREE.SpotLight(col, cd * look.light, 0, ang, area ? 1 : Math.max(L.blend, 0.15), 2);
      light.position.copy(pos);
      light.target.position.copy(pos).add(dir);
    }
    if (shadows > 0 && L.type !== 'POINT') {
      shadows--;
      light.castShadow = true;
      light.shadow.mapSize.set(2048, 2048);
      light.shadow.bias = -0.0004;
      light.shadow.normalBias = 0.02;
      light.shadow.radius = 4;
      if (L.type === 'SUN') {
        const c = light.shadow.camera;
        c.left = c.bottom = -radius; c.right = c.top = radius; c.near = 0.1; c.far = radius * 4;
        light.shadow.normalBias = radius * 0.002;
      } else {
        light.shadow.camera.near = 0.05; light.shadow.camera.far = radius * 4;
      }
    }
    scene.add(light);
    if (light.target) scene.add(light.target);
    out.push(light);
  }
  return out;
}

export class SceneViewer {
  constructor(host, opts) {
    this.host = host;              // element that gets the canvas
    this.opts = opts;              // { onProgress(frac, text), onReady(key, meta), onError(err) }
    this.cache = new Map();        // key -> { root, meta, lights, sky, env }
    this.key = null;
    this.visible = true;
    this.needs = true;
    const r = this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    r.setPixelRatio(Math.min(window.devicePixelRatio || 1, window.matchMedia('(pointer: coarse)').matches ? 1.5 : 2));
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFSoftShadowMap;
    r.domElement.className = 'v3d-canvas';
    r.domElement.setAttribute('aria-label', 'Interactive 3D view. Drag to orbit, scroll or pinch to zoom, right-drag or two fingers to pan.');
    host.appendChild(r.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 16 / 9, 0.1, 1000);
    this.controls = new OrbitControls(this.camera, r.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.zoomToCursor = true;
    this.controls.addEventListener('change', () => { this.needs = true; });
    this.pmrem = new THREE.PMREMGenerator(r);
    this.composer = new EffectComposer(r, new THREE.WebGLRenderTarget(4, 4, { type: THREE.HalfFloatType, samples: 4 }));
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.3, 0.55, 1.0);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    new ResizeObserver(() => this.resize()).observe(host);
    new IntersectionObserver((e) => { this.visible = e[0].isIntersecting; this.needs = true; }).observe(host);
    this.resize();
    r.setAnimationLoop(() => this.frame());
  }

  resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.composer.setPixelRatio(this.renderer.getPixelRatio());
    this.composer.setSize(w, h);
    this.camera.aspect = w / h;
    this.fitFov();
    this.needs = true;
  }

  // Blender's FOV is for a 16:9 frame. On a taller (phone) frame keep the same horizontal view.
  fitFov() {
    if (!this.baseFov) { this.camera.updateProjectionMatrix(); return; }
    const a = this.camera.aspect, ref = 16 / 9;
    const v = THREE.MathUtils.degToRad(this.baseFov);
    this.camera.fov = a >= ref ? this.baseFov : THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(v / 2) * ref / a));
    this.camera.updateProjectionMatrix();
  }

  frame() {
    if (!this.visible || !this.key) return;
    this.controls.update();
    if (!this.needs) return;
    this.needs = false;
    this.composer.render();
  }

  async fetchScene(key) {
    if (this.cache.has(key)) return this.cache.get(key);
    const url = (f) => new URL('models/' + f, BASE).href;
    const report = (f, t) => this.opts.onProgress && this.opts.onProgress(f, t);
    report(0, 'Loading…');
    const meta = await fetch(url(key + '.json')).then((r) => { if (!r.ok) throw new Error('scene info ' + r.status); return r.json(); });
    const total = meta.stats.glb_bytes;
    const [gltf, skyTex] = await Promise.all([
      gltfLoader.loadAsync(url(key + '.glb'), (e) => report(Math.min(e.loaded / total, 1) * 0.92, (e.loaded / 1e6).toFixed(1) + ' / ' + (total / 1e6).toFixed(1) + ' MB')),
      new THREE.TextureLoader().loadAsync(url(key + '_sky.jpg')),
    ]);
    report(0.96, 'Building scene…');
    skyTex.mapping = THREE.EquirectangularReflectionMapping;
    skyTex.colorSpace = THREE.SRGBColorSpace;
    const env = this.pmrem.fromEquirectangular(skyTex).texture;
    const look = LOOK[key] || LOOK.s1_v1;
    const root = new THREE.Group();
    root.add(gltf.scene);
    const maxAniso = this.renderer.capabilities.getMaxAnisotropy();
    gltf.scene.traverse((o) => {
      if (!o.isMesh) return;
      const m = o.material;
      if (m.transmission > 0) { m.opacity = Math.min(m.opacity, 0.3); m.transmission = 0; m.transparent = true; }
      if (look.emissive && m.emissiveIntensity) m.emissiveIntensity *= look.emissive;
      if (m.transparent) { m.depthWrite = false; o.renderOrder = 2; } else { o.castShadow = true; }
      o.receiveShadow = true;
      for (const k of ['map', 'emissiveMap', 'roughnessMap', 'metalnessMap']) if (m[k]) m[k].anisotropy = Math.min(8, maxAniso);
    });
    const bmin = v3(meta.bounds.min), bmax = v3(meta.bounds.max);
    const cam = meta.camera, cpos = v3(cam.pos), cdir = v3(cam.dir);
    const boxC = bmin.clone().add(bmax).multiplyScalar(0.5);
    let focus = cam.focus > 0.01 ? cam.focus : Math.max(boxC.clone().sub(cpos).dot(cdir), 1);
    const target = cpos.clone().addScaledVector(cdir, focus);
    const radius = Math.min(bmax.distanceTo(bmin) / 2, focus * 3);
    const lights = buildLights(meta, look, root, target, radius);
    if (look.wet) root.add(wetRoad(look, window.matchMedia('(max-width: 760px), (pointer: coarse)').matches));
    if (look.ambient) root.add(new THREE.AmbientLight(look.ambientColor || 0xffffff, look.ambient));
    const entry = { root, meta, look, env, sky: skyTex, cpos, target, focus, size: bmax.distanceTo(bmin), lights };
    this.cache.set(key, entry);
    return entry;
  }

  async show(key) {
    try {
      const e = await this.fetchScene(key);
      const prev = this.key && this.cache.get(this.key);
      if (prev) this.scene.remove(prev.root);
      this.key = key;
      this.scene.add(e.root);
      this.scene.background = e.sky;
      this.scene.environment = e.env;
      this.scene.environmentIntensity = e.look.env;
      this.scene.fog = e.look.fog ? new THREE.FogExp2(e.look.fog[0], e.look.fogDensity ?? e.look.fog[1]) : null;
      const r = this.renderer;
      r.toneMapping = TONE[e.look.tone || e.meta.view.transform] || THREE.AgXToneMapping;
      r.toneMappingExposure = Math.pow(2, e.meta.view.exposure || 0) * e.look.exposure;
      this.bloom.strength = e.look.bloom;
      this.bloom.radius = e.look.bloomRadius ?? 0.55;
      this.bloom.threshold = e.look.bloomThreshold ?? 1.0;
      this.baseFov = e.meta.camera.vfov;
      this.camera.near = Math.max(e.focus / 200, 0.01);
      this.camera.far = e.size * 1.6 + 50;
      this.controls.minDistance = e.focus * 0.08;
      this.controls.maxDistance = Math.max(e.size * 1.2, e.focus * 4);
      this.resetView();
      this.fitFov();
      this.opts.onProgress && this.opts.onProgress(1, '');
      this.opts.onReady && this.opts.onReady(key, e.meta);
    } catch (err) {
      console.error('[viewer]', key, err);
      this.opts.onError && this.opts.onError(err);
    }
  }

  resetView() {
    const e = this.cache.get(this.key);
    if (!e) return;
    this.camera.position.copy(e.cpos);
    this.controls.target.copy(e.target);
    this.controls.update();
    this.needs = true;
  }
}
