import json
import streamlit as st

AGENTS = ["Search", "Read", "Write", "Grade", "Fact-check"]

_HTML = r"""
<style>
html,body{margin:0;background:transparent;overflow:hidden;font-family:'IBM Plex Sans',system-ui,sans-serif}
#w{position:relative;width:100%;height:__H__px;cursor:grab}
#w:active{cursor:grabbing}
canvas{display:block;width:100%;height:100%}
.lb{position:absolute;transform:translate(-50%,0);font-size:12px;font-weight:500;pointer-events:none;white-space:nowrap;transition:color .3s}
#hint{position:absolute;left:12px;bottom:10px;font-size:11px;color:#5d7598}
#cap{position:absolute;left:12px;top:10px;font-size:13px;color:#C9D3E1}
#cap b{color:#F2B134;font-weight:500}
</style>
<div id="w"><canvas id="c"></canvas><div id="L"></div><div id="cap"></div><div id="hint">Drag to turn</div></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const S = __STATE__;
const NAMES = S.names, ACTIVE = S.active, DONE = S.done, SCORE = S.score;
const w = document.getElementById('w'), cv = document.getElementById('c');
const R = new THREE.WebGLRenderer({canvas: cv, antialias: true, alpha: true});
R.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
const cam = new THREE.PerspectiveCamera(42, 1, .1, 100);
scene.add(new THREE.AmbientLight(0x8fa6c4, .55));
const L1 = new THREE.PointLight(0xffd38a, 1.3, 40); L1.position.set(4, 6, 8); scene.add(L1);
const L2 = new THREE.PointLight(0x5FB3A6, .8, 40); L2.position.set(-6, -3, 4); scene.add(L2);
const AMBER = 0xF2B134, TEAL = 0x5FB3A6, IDLE = 0x34496A;

const world = new THREE.Group(); scene.add(world);

// the "web" — a sphere of source points at the centre
const N = 900, g = new THREE.BufferGeometry(), p = new Float32Array(N * 3);
for (let i = 0; i < N; i++) {
  const u = Math.random() * 2 - 1, t = Math.random() * Math.PI * 2, r = 1.25 + Math.random() * .08;
  const s = Math.sqrt(1 - u * u);
  p[i*3] = r * s * Math.cos(t); p[i*3+1] = r * u; p[i*3+2] = r * s * Math.sin(t);
}
g.setAttribute('position', new THREE.BufferAttribute(p, 3));
const web = new THREE.Points(g, new THREE.PointsMaterial({color: 0x7d93b3, size: .035, transparent: true, opacity: .85}));
world.add(web);
const core = new THREE.Mesh(new THREE.IcosahedronGeometry(.55, 1),
  new THREE.MeshStandardMaterial({color: 0x15263D, emissive: 0x1B3A5C, flatShading: true, roughness: .6}));
world.add(core);

// agents orbit the web
const orbitR = 2.9, nodes = [], labels = [];
const Lb = document.getElementById('L');
NAMES.forEach((n, i) => {
  const a = i / NAMES.length * Math.PI * 2;
  const isA = i === ACTIVE, isD = DONE.includes(i);
  const col = isA ? AMBER : isD ? TEAL : IDLE;
  const m = new THREE.Mesh(new THREE.OctahedronGeometry(isA ? .34 : .26, 0),
    new THREE.MeshStandardMaterial({color: col, emissive: col, emissiveIntensity: isA ? .7 : isD ? .3 : .05, flatShading: true, roughness: .3}));
  m.userData = {a, i};
  world.add(m); nodes.push(m);
  const d = document.createElement('div'); d.className = 'lb'; d.textContent = n;
  d.style.color = isA ? '#F2B134' : isD ? '#9FD3CA' : '#7d8ea8';
  Lb.appendChild(d); labels.push(d);
});
// orbit ring
const ring = new THREE.Mesh(new THREE.TorusGeometry(orbitR, .006, 6, 160),
  new THREE.MeshBasicMaterial({color: 0x2a3d58}));
ring.rotation.x = Math.PI / 2; world.add(ring);

// beams: from active agent to the core, particles travel along them
const beams = [];
if (ACTIVE >= 0) {
  for (let k = 0; k < 14; k++) {
    const b = new THREE.Mesh(new THREE.SphereGeometry(.035, 6, 6), new THREE.MeshBasicMaterial({color: AMBER}));
    world.add(b); beams.push({b, t: k / 14});
  }
}
const halo = new THREE.Mesh(new THREE.RingGeometry(.5, .54, 48),
  new THREE.MeshBasicMaterial({color: AMBER, transparent: true, opacity: ACTIVE >= 0 ? .8 : 0, side: THREE.DoubleSide}));
world.add(halo);

// caption
const cap = document.getElementById('cap');
cap.innerHTML = ACTIVE >= 0 ? `<b>${NAMES[ACTIVE]}</b> is working` :
  SCORE ? `Finished · score <b>${SCORE}/10</b>` : 'Waiting for a topic';

// interaction: drag to rotate, gentle drift otherwise
let rotY = .3, rotX = .35, vy = 0, drag = false, lx = 0, ly = 0, hover = -1;
const ray = new THREE.Raycaster(), mouse = new THREE.Vector2();
w.addEventListener('pointerdown', e => { drag = true; lx = e.clientX; ly = e.clientY; w.setPointerCapture(e.pointerId); });
w.addEventListener('pointerup', () => drag = false);
w.addEventListener('pointermove', e => {
  const r = w.getBoundingClientRect();
  mouse.x = (e.clientX - r.left) / r.width * 2 - 1; mouse.y = -(e.clientY - r.top) / r.height * 2 + 1;
  if (!drag) return;
  vy = (e.clientX - lx) * .006; rotY += vy; rotX = Math.max(-.2, Math.min(1.0, rotX + (e.clientY - ly) * .004));
  lx = e.clientX; ly = e.clientY;
});

function resize() {
  const W = w.clientWidth, H = w.clientHeight;
  R.setSize(W, H, false); cam.aspect = W / H;
  cam.position.set(0, 0, W < 420 ? 10.5 : 8.6); cam.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(w); resize();

const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const t0 = performance.now(), v3 = new THREE.Vector3();
function frame(now) {
  const t = (now - t0) / 1000;
  if (!drag) { vy *= .94; rotY += vy + (reduce ? 0 : .0025); }
  world.rotation.set(rotX, rotY, 0);
  web.rotation.y = -t * .05; core.rotation.set(t * .2, t * .3, 0);
  const pulse = ACTIVE >= 0 ? 1 + Math.sin(t * 4) * .04 : 1;
  web.scale.setScalar(pulse);

  ray.setFromCamera(mouse, cam);
  const hit = ray.intersectObjects(nodes)[0]; hover = hit ? hit.object.userData.i : -1;

  nodes.forEach((m, i) => {
    const a = m.userData.a + (reduce ? 0 : t * .08);
    m.position.set(Math.cos(a) * orbitR, Math.sin(t * 1.2 + i) * .12, Math.sin(a) * orbitR);
    m.rotation.set(t * .6, t * (i === ACTIVE ? 1.6 : .4), 0);
    const sc = (i === hover ? 1.35 : 1); m.scale.lerp(v3.set(sc, sc, sc), .15);
  });
  if (ACTIVE >= 0) {
    const src = nodes[ACTIVE].position;
    halo.position.copy(src); halo.lookAt(cam.position.clone().applyMatrix4(world.matrixWorld.clone().invert()));
    halo.scale.setScalar(1 + Math.sin(t * 3) * .1);
    beams.forEach(o => {
      o.t = (o.t + .012) % 1;
      const k = DONE.length === 0 || ACTIVE === 0 ? 1 - o.t : o.t;   // search pulls in, others push out
      o.b.position.lerpVectors(src, new THREE.Vector3(0, 0, 0), k * .78);
    });
  }
  R.render(scene, cam);
  const W = w.clientWidth, H = w.clientHeight;
  nodes.forEach((m, i) => {
    m.getWorldPosition(v3); v3.y -= .55; v3.project(cam);
    labels[i].style.left = (v3.x + 1) / 2 * W + 'px'; labels[i].style.top = (-v3.y + 1) / 2 * H + 'px';
    labels[i].style.opacity = v3.z > .98 ? .3 : 1;
  });
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>
"""

def scene(active=-1, done=(), score=None, height=380):
    state = {"names": AGENTS, "active": active, "done": list(done), "score": score}
    html = _HTML.replace("__STATE__", json.dumps(state)).replace("__H__", str(height))
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:                                   # older Streamlit
        import streamlit.components.v1 as components
        components.html(html, height=height)