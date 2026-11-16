import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const SEGS = { ffr_unet: "FFR-UNet", cas_net: "CAS-Net", swin_unetr: "Swin UNETR" };
const CASES = [
  { key: "hero", title: "Dice 0.948, still broken", tab: "High Dice, critical, caught",
    story: "By overlap this reconstruction looks excellent. But the LAD and its D2 branch are cut off from the rest of the left tree, starting about 38 mm from the ostium. The detached piece shows up as an extra colour. CoronarySafe ranked it 25th of 160, inside the review budget." },
  { key: "caught", title: "Ranked 5th of 160", tab: "Obvious failure, top of the list",
    story: "The primary test segmenter (FFR-UNet, never seen during development) broke both trees: the LAD, D1, D2 and OM2 come apart from the left main, the RCA is split, and spurious vessel-like pieces appear. CoronarySafe put it near the very top of the review list." },
  { key: "clean", title: "A healthy reconstruction", tab: "Clean case, ranked last",
    story: "Two clean, connected trees (left and right) with nothing critical missing. CoronarySafe ranked it 160th of 160, last in line for review, which is exactly where a good reconstruction belongs." },
  { key: "missed", title: "One it missed", tab: "Honest miss",
    story: "Not every failure is caught. Here the LAD and D2 are disconnected at Dice 0.902, but its overall structure did not stand out to the features, so it ranked 148th and was not flagged. Misses like this are why the paper reports raw counts, not just the headline." },
];
const COLORS = { tree: [0x2563eb, 0x0ea5e9], extra: [0xf59e0b, 0xf97316, 0xeab308, 0xfb923c], fragments: 0xfbbf24, missed: 0xdc2626, spurious: 0xa21caf, ref: 0x94a3b8 };

const host = document.getElementById("viewer");
const loading = document.getElementById("viewer-loading");
let renderer;
try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); }
catch (e) {
  loading.innerHTML = "3D view needs WebGL, which is turned off in this browser.<br>Try another browser or enable hardware acceleration.";
  loading.style.textAlign = "center";
  throw e;
}
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
host.appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.domElement.style.position = "absolute"; labels.domElement.style.inset = "0"; labels.domElement.style.pointerEvents = "none";
host.appendChild(labels.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 0.5, 5000);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; controls.autoRotate = true; controls.autoRotateSpeed = 0.9;
controls.addEventListener("start", () => { controls.autoRotate = false; });
scene.add(new THREE.HemisphereLight(0xffffff, 0x9aa3b2, 1.6));
const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(200, -300, 400); scene.add(key);
const rim = new THREE.DirectionalLight(0xffffff, 0.6); rim.position.set(-300, 200, -100); scene.add(rim);

let group = null, layers = {}, home = null;
const cache = {};

function decode(b64, Type) {
  const bin = atob(b64), buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Type(buf.buffer);
}
function mesh(m, color, opts = {}) {
  if (!m) return null;
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(decode(m.pos, Float32Array), 3));
  g.setIndex(new THREE.BufferAttribute(decode(m.idx, Uint32Array), 1));
  g.computeVertexNormals();
  const mat = new THREE.MeshStandardMaterial({ color, roughness: .42, metalness: .05, side: THREE.DoubleSide,
    transparent: !!opts.opacity, opacity: opts.opacity ?? 1, depthWrite: !opts.opacity });
  const o = new THREE.Mesh(g, mat); o.renderOrder = opts.order ?? 0; return o;
}

function build(data) {
  if (group) { scene.remove(group); group.traverse(o => { o.geometry?.dispose(); o.material?.dispose?.(); }); }
  group = new THREE.Group(); layers = { pred: new THREE.Group(), missed: new THREE.Group(), spurious: new THREE.Group(), ref: new THREE.Group(), markers: new THREE.Group() };
  const M = data.meshes;
  (M.components || []).forEach((c, i) => {
    const col = i < 2 ? COLORS.tree[i] : COLORS.extra[(i - 2) % COLORS.extra.length];
    const o = mesh(c, col); if (o) layers.pred.add(o);
  });
  const fr = mesh(M.fragments, COLORS.fragments); if (fr) layers.pred.add(fr);
  const mi = mesh(M.missed, COLORS.missed, { order: 2 }); if (mi) layers.missed.add(mi);
  const sp = mesh(M.spurious, COLORS.spurious, { opacity: .85, order: 3 }); if (sp) layers.spurious.add(sp);
  const rf = mesh(M.ref, COLORS.ref, { opacity: .16, order: 4 }); if (rf) layers.ref.add(rf);
  layers.ref.visible = false;
  for (const mk of data.markers || []) {
    const s = new THREE.Mesh(new THREE.SphereGeometry(2.2, 20, 14), new THREE.MeshStandardMaterial({ color: 0xb91c1c, emissive: 0x7f1d1d }));
    s.position.set(...mk.pos); layers.markers.add(s);
    const div = document.createElement("div"); div.className = "marker-label";
    div.textContent = `${mk.subtype === "disconnection" ? "break" : mk.flag} · ${mk.segment}`;
    const lbl = new CSS2DObject(div); lbl.position.set(mk.pos[0], mk.pos[1], mk.pos[2] + 6); layers.markers.add(lbl);
  }
  Object.values(layers).forEach(l => group.add(l));
  scene.add(group);
  const box = new THREE.Box3().setFromObject(layers.pred.children.length ? layers.pred : group);
  const c = box.getCenter(new THREE.Vector3()), r = box.getSize(new THREE.Vector3()).length() / 2;
  controls.target.copy(c);
  const dist = r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2)) * 0.95;
  camera.position.set(c.x + dist * 0.15, c.y - dist * 0.98, c.z + dist * 0.25);
  home = { pos: camera.position.clone(), target: c.clone() };
  controls.autoRotate = true;
  renderToggles(data);
}

function renderToggles(data) {
  const T = document.getElementById("toggles"); T.innerHTML = "";
  const items = [
    ["pred", "AI prediction (one colour per piece)", "linear-gradient(90deg,#2563eb 50%,#f59e0b 50%)"],
    ["missed", "Reference the AI missed", "#dc2626"],
    ["spurious", "Spurious extra vessel", "#a21caf"],
    ["ref", "Reference (ghost)", "#94a3b8"],
    ["markers", "Detected breaks", "#b91c1c"],
  ];
  for (const [k, label, col] of items) {
    if (k === "markers" && !(data.markers || []).length) continue;
    if (!layers[k] || !layers[k].children.length) continue;
    const b = document.createElement("div"); b.className = "toggle" + (layers[k].visible ? "" : " off");
    b.innerHTML = `<i style="background:${col}"></i>${label}`;
    b.onclick = () => {
      layers[k].visible = !layers[k].visible; b.classList.toggle("off", !layers[k].visible);
      if (k === "markers") layers.markers.children.forEach(o => { if (o.element) o.element.style.display = layers.markers.visible ? "" : "none"; });
    };
    T.appendChild(b);
  }
}

function card(c, d) {
  const el = document.getElementById("case-card");
  const pct = (d.rank - 0.5) / d.n * 100;
  const reasons = (d.critical_reasons || []).map(r => {
    const [f, segs] = r.split(":"); return `<li>${f === "F2" ? "Disconnected" : f === "F1" ? "Missing run in" : "Leak / bridge in"} <b>${(segs || "").replaceAll("+", ", ")}</b></li>`;
  }).join("");
  el.innerHTML = `
    <div class="badges">
      <span class="badge">${SEGS[d.method]} · patient ${d.scan}</span>
      ${d.critical ? '<span class="badge red">critical failure</span>' : '<span class="badge green">no critical failure</span>'}
      ${d.flagged ? '<span class="badge amber">flagged for review</span>' : '<span class="badge">not flagged</span>'}
    </div>
    <h3>${c.title}</h3>
    <div class="facts">
      <div class="fact"><b>${d.dice.toFixed(3)}</b><span>Dice overlap</span></div>
      <div class="fact"><b>#${d.rank}<small style="font-size:12px;color:#7a808c"> / ${d.n}</small></b><span>review rank</span></div>
    </div>
    <div><div class="rankbar"><i style="left:${pct}%"></i></div><div class="rankcap"><span>reviewed first (top 20%)</span><span>reviewed last</span></div></div>
    <p class="story">${c.story}</p>
    ${reasons ? `<div><b style="font-size:13px">Why it counts as critical</b><ul>${reasons}</ul></div>` : ""}`;
}

async function show(i) {
  const c = CASES[i];
  [...document.getElementById("case-tabs").children].forEach((b, j) => b.classList.toggle("on", i === j));
  loading.style.display = "grid";
  try {
    const d = cache[c.key] || (cache[c.key] = await (await fetch(`data/cases/${c.key}.json`)).json());
    build(d); card(c, d);
  } catch (e) { loading.textContent = "Could not load this case."; return; }
  loading.style.display = "none";
}

const tabs = document.getElementById("case-tabs");
CASES.forEach((c, i) => {
  const b = document.createElement("button"); b.className = "tab";
  b.innerHTML = `<b>${c.tab}</b><span>${c.title}</span>`; b.onclick = () => show(i); tabs.appendChild(b);
});

function resize() {
  const w = host.clientWidth, h = host.clientHeight;
  renderer.setSize(w, h, false); labels.setSize(w, h);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(host);
renderer.domElement.addEventListener("dblclick", () => { if (home) { camera.position.copy(home.pos); controls.target.copy(home.target); } });

let visible = true;
new IntersectionObserver(es => { visible = es[0].isIntersecting; }).observe(host);
renderer.setAnimationLoop(() => {
  if (!visible) return;
  controls.update(); renderer.render(scene, camera); labels.render(scene, camera);
});
resize();
show(0);
