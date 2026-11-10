"""Scene 2 (medium): low-poly floating island at sunset - cabin, windmill, pond, waterfall."""
import os
exec(open(os.path.join(HERE, "common.py")).read())
random.seed(7)

sc = new_scene("S2_FloatingIsland")
setup_cycles(samples=192, exposure=-0.3)

# ---- sunset gradient world
w = world_color((0, 0, 0), 1.0)
nt = w.node_tree
bg = next(n for n in nt.nodes if n.type == "BACKGROUND")
tc = nt.nodes.new("ShaderNodeTexCoord")
sx = nt.nodes.new("ShaderNodeSeparateXYZ")
mr = nt.nodes.new("ShaderNodeMapRange")
mr.inputs["From Min"].default_value = -1; mr.inputs["From Max"].default_value = 1
cr = nt.nodes.new("ShaderNodeValToRGB")
stops = [(0.0, (0.05, 0.06, 0.16)), (0.36, (0.42, 0.22, 0.42)), (0.47, (1.0, 0.50, 0.28)),
         (0.58, (0.80, 0.50, 0.62)), (0.75, (0.22, 0.30, 0.62)), (1.0, (0.05, 0.10, 0.32))]
el = cr.color_ramp.elements
el[0].position, el[0].color = stops[0][0], stops[0][1] + (1,)
el[1].position, el[1].color = stops[-1][0], stops[-1][1] + (1,)
for pos, c in stops[1:-1]:
    e = el.new(pos); e.color = c + (1,)
nt.links.new(tc.outputs["Generated"], sx.inputs[0])
nt.links.new(sx.outputs["Z"], mr.inputs["Value"])
nt.links.new(mr.outputs[0], cr.inputs["Fac"])
nt.links.new(cr.outputs["Color"], bg.inputs["Color"])
bg.inputs["Strength"].default_value = 0.9

# ---- layout
CAB = Vector((-0.8, 1.7)); POND = Vector((2.5, -0.7)); MILL = Vector((-2.9, -2.6))
EDGE_A = math.radians(-38)
WATER = 0.12


def smooth(a, b, x):
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def redge(a):
    return 6.0 * (1 + 0.13 * noise.noise(Vector((math.cos(a) * 1.4, math.sin(a) * 1.4, 0.5))))


STREAM_END = Vector((math.cos(EDGE_A), math.sin(EDGE_A))) * redge(EDGE_A)


def seg_dist(p, a, b):
    ab = b - a
    t = max(0, min(1, (p - a).dot(ab) / ab.length_squared))
    return (p - (a + ab * t)).length


def height(x, y):
    p = Vector((x, y))
    h = 0.32 + 0.42 * noise.noise(Vector((x * 0.25, y * 0.25, 3.1))) + 0.14 * noise.noise(Vector((x * 0.8, y * 0.8, 7.0)))
    h += 1.15 * math.exp(-(p - MILL).length_squared / 2.4)
    h += 0.7 * math.exp(-(p - Vector((1.5, 3.4))).length_squared / 3.0)
    wc = smooth(2.6, 1.5, (p - CAB).length)
    h = h * (1 - wc) + 0.35 * wc
    wp = smooth(1.9, 1.0, (p - POND).length)
    h = h * (1 - wp) + (-0.12) * wp
    ws = smooth(0.75, 0.3, seg_dist(p, POND, STREAM_END))
    h = h * (1 - ws) + 0.02 * ws
    return h


# ---- materials
def terrain_mat():
    m, nt, p = mat_nodes("s2_terrain")
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    sx = nt.nodes.new("ShaderNodeSeparateXYZ")
    cr = nt.nodes.new("ShaderNodeValToRGB")
    cr.color_ramp.interpolation = "CONSTANT"
    e = cr.color_ramp.elements
    e[0].position, e[0].color = 0.0, (0.20, 0.14, 0.11, 1)
    e[1].position, e[1].color = 0.80, (0.22, 0.42, 0.10, 1)
    k = e.new(0.55); k.color = (0.30, 0.20, 0.12, 1)
    tcn = nt.nodes.new("ShaderNodeTexCoord")
    nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 0.6
    mx = nt.nodes.new("ShaderNodeMix"); mx.data_type = "RGBA"; mx.blend_type = "MULTIPLY"
    mx.inputs["Factor"].default_value = 0.55
    L = nt.links.new
    L(geo.outputs["True Normal"], sx.inputs[0]); L(sx.outputs["Z"], cr.inputs["Fac"])
    L(tcn.outputs["Object"], nz.inputs["Vector"])
    L(cr.outputs["Color"], mx.inputs["A"]); L(nz.outputs["Color"], mx.inputs["B"])
    L(mx.outputs["Result"], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.9
    return m


m_terrain = terrain_mat()
m_rock = mat("s2_rock", (0.27, 0.25, 0.25), rough=0.9)
m_trunk = mat("s2_trunk", (0.16, 0.09, 0.05), rough=0.9)
m_pines = [mat(f"s2_pine{i}", c, rough=0.8) for i, c in enumerate([(0.05, 0.22, 0.10), (0.07, 0.30, 0.12), (0.04, 0.17, 0.12)])]
m_autumn = [mat(f"s2_aut{i}", c, rough=0.8) for i, c in enumerate([(0.85, 0.35, 0.05), (0.80, 0.18, 0.06), (0.95, 0.60, 0.10)])]
m_plankA = mat("s2_plankA", (0.36, 0.20, 0.10), rough=0.8)
m_plankB = mat("s2_plankB", (0.29, 0.15, 0.07), rough=0.8)
m_roof = mat("s2_roof", (0.50, 0.10, 0.08), rough=0.7)
m_dark = mat("s2_darkwood", (0.10, 0.06, 0.04), rough=0.8)
m_glow = mat("s2_glow", (1, 0.8, 0.4), emit=(1.0, 0.62, 0.22), emit_str=9)
m_stone = mat("s2_stone", (0.45, 0.43, 0.42), rough=0.9)
m_white = mat("s2_white", (0.85, 0.82, 0.76), rough=0.8)
m_water = mat("s2_water", (0.10, 0.42, 0.55), rough=0.04, trans=0.4, ior=1.33)
m_fall, nt, p = mat_nodes("s2_fall")
tcf = nt.nodes.new("ShaderNodeTexCoord"); mpf = nt.nodes.new("ShaderNodeMapping")
mpf.inputs["Scale"].default_value = (7.0, 7.0, 0.35)
nzf = nt.nodes.new("ShaderNodeTexNoise"); nzf.inputs["Scale"].default_value = 2.0; nzf.inputs["Detail"].default_value = 4.0
crf = nt.nodes.new("ShaderNodeValToRGB")
crf.color_ramp.elements[0].position = 0.42; crf.color_ramp.elements[0].color = (0.16, 0.50, 0.72, 1)
crf.color_ramp.elements[1].position = 0.62; crf.color_ramp.elements[1].color = (0.92, 0.97, 1.0, 1)
nt.links.new(tcf.outputs["Object"], mpf.inputs["Vector"]); nt.links.new(mpf.outputs[0], nzf.inputs["Vector"])
nt.links.new(nzf.outputs["Fac"], crf.inputs["Fac"]); nt.links.new(crf.outputs["Color"], p.inputs["Base Color"])
nt.links.new(crf.outputs["Color"], p.inputs["Emission Color"])
p.inputs["Emission Strength"].default_value = 0.25; p.inputs["Roughness"].default_value = 0.2
m_cloud = mat("s2_cloud", (1, 0.93, 0.9), rough=1.0, emit=(1.0, 0.75, 0.65), emit_str=0.12)
m_smoke = mat("s2_smoke", (0.75, 0.72, 0.72), rough=1.0)
m_sail = mat("s2_sail", (0.93, 0.88, 0.75), rough=0.9)
m_ball = [mat("s2_ballA", (0.90, 0.22, 0.12), rough=0.6), mat("s2_ballB", (0.98, 0.85, 0.55), rough=0.6)]

# ---- island mesh
group("Island")
NR, NS, NB, DEPTH = 22, 60, 10, 7.5
bm = bmesh.new()
top_c = bm.verts.new((0, 0, height(0, 0)))
rings = []
for i in range(1, NR + 1):
    t = i / NR
    ring = []
    for s in range(NS):
        a = 2 * math.pi * s / NS
        r = t * redge(a)
        x, y = r * math.cos(a), r * math.sin(a)
        ring.append(bm.verts.new((x, y, height(x, y) * (1 - 0.7 * t ** 8))))
    rings.append(ring)
for k in range(1, NB):
    s_ = k / NB
    ring = []
    for s in range(NS):
        a = 2 * math.pi * s / NS
        n1 = noise.noise(Vector((math.cos(a) * 2.0, math.sin(a) * 2.0, k * 0.6)))
        r = redge(a) * (1 - s_) ** 0.7 * (1 + 0.22 * n1)
        z = -0.35 - DEPTH * s_ ** 1.25 * (1 + 0.15 * n1)
        ring.append(bm.verts.new((r * math.cos(a), r * math.sin(a), z)))
    rings.append(ring)
tip = bm.verts.new((0.3, -0.2, -DEPTH - 0.9))
for s in range(NS):
    bm.faces.new((top_c, rings[0][s], rings[0][(s + 1) % NS]))
    bm.faces.new((tip, rings[-1][(s + 1) % NS], rings[-1][s]))
for a_, b_ in zip(rings, rings[1:]):
    for s in range(NS):
        j = (s + 1) % NS
        bm.faces.new((a_[s], b_[s], b_[j], a_[j]))
bmesh.ops.triangulate(bm, faces=bm.faces)
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
_finish("Island", bm, m_terrain)

# hanging rocks under the island
for i in range(9):
    a = random.uniform(0, 6.28); r = random.uniform(1.0, 4.0)
    z = -0.6 - DEPTH * (1 - r / 6.0) * 0.75
    ln = random.uniform(1.0, 2.4)
    cyl(f"Stalactite{i}", random.uniform(0.35, 0.7), ln, (r * math.cos(a), r * math.sin(a), z - ln), m_rock,
        segs=6, r2=0.0, smooth=False, rot=(math.pi, 0, 0)).location.z += ln

# ---- water: pond, stream, waterfall
group("Water")
cyl("Pond", 1.75, 0.02, (POND.x, POND.y, WATER - 0.02), m_water, segs=28, smooth=False)
d = (STREAM_END - POND).normalized(); n = Vector((-d.y, d.x))
verts, faces = [], []
pts = []
L = (STREAM_END - POND).length
for i in range(11):
    p = POND + d * (1.6 + (L - 1.35) * i / 10)
    pts.append((p, WATER - 0.03 - 0.05 * i / 10))
edge_p = pts[-1][0]
for i in range(1, 15):          # the fall: arc out, then drop
    t = i / 14
    pts.append((edge_p + d * (0.9 * (1 - (1 - t) ** 2)), 0.05 - 9.0 * t ** 1.6))
for i, (p, z) in enumerate(pts):
    wd = 0.24 if i < 11 else 0.24 + 0.2 * (i - 10) / 14
    verts += [(p.x - n.x * wd, p.y - n.y * wd, z), (p.x + n.x * wd, p.y + n.y * wd, z)]
    if i:
        faces.append((2 * i - 2, 2 * i - 1, 2 * i + 1, 2 * i))
fall = mesh_from("StreamAndFall", verts, faces, m_fall)
md = fall.modifiers.new("Solid", "SOLIDIFY"); md.thickness = 0.06
for i in range(7):   # mist at the bottom of the fall
    q = edge_p + d * 0.9 + Vector((random.uniform(-0.6, 0.6), random.uniform(-0.6, 0.6)))
    sphere(f"Mist{i}", random.uniform(0.35, 0.7), (q.x, q.y, -8.8 + random.uniform(-0.4, 0.5)), m_cloud, ico=1,
           smooth=False)

# ---- cabin
group("Cabin")
gz = 0.33
for i in range(6):
    box(f"Wall{i}", (2.4 + (i % 2) * 0.06, 2.0 + ((i + 1) % 2) * 0.06, 0.22), (CAB.x, CAB.y, gz + i * 0.215),
        m_plankA if i % 2 else m_plankB, base=True)
wall_top = gz + 6 * 0.215
box("Foundation", (2.6, 2.2, 0.12), (CAB.x, CAB.y, gz - 0.08), m_stone, base=True)
rise = 0.80
mesh_from("Gables", [(-1.2, -1.0, 0), (-1.2, 1.0, 0), (-1.2, 0, rise), (1.2, -1.0, 0), (1.2, 1.0, 0), (1.2, 0, rise)],
          [(0, 1, 2), (3, 5, 4), (0, 2, 5, 3), (1, 4, 5, 2), (0, 3, 4, 1)], m_plankA, loc=(CAB.x, CAB.y, wall_top))
ang = math.atan(rise / 1.0)
for sgn in (-1, 1):
    half = 1.35
    box(f"Roof{sgn}", (2.95, half / math.cos(ang), 0.09),
        (CAB.x, CAB.y + sgn * half / 2, wall_top + rise - (half / 2) * rise + 0.06), m_roof, rot=(-sgn * ang, 0, 0))
box("RoofRidge", (3.0, 0.16, 0.1), (CAB.x, CAB.y, wall_top + rise + 0.06), m_dark)
fy = CAB.y - 1.03
box("Door", (0.55, 0.06, 0.95), (CAB.x - 0.35, fy, gz + 0.04), m_dark, base=True)
sphere("DoorKnob", 0.035, (CAB.x - 0.15, fy - 0.04, gz + 0.5), m_glow, ico=1)
box("Step", (0.9, 0.5, 0.1), (CAB.x - 0.35, fy - 0.25, gz - 0.06), m_stone, base=True)


def window(name, loc, along_x=True):
    sz = (0.6, 0.06, 0.5) if along_x else (0.06, 0.6, 0.5)
    box(name, sz, loc, m_glow)
    fr = [((0.7, 0.08, 0.06), (0, 0, 0.27)), ((0.7, 0.08, 0.06), (0, 0, -0.27)), ((0.06, 0.08, 0.6), (0.32, 0, 0)),
          ((0.06, 0.08, 0.6), (-0.32, 0, 0)), ((0.04, 0.08, 0.5), (0, 0, 0)), ((0.6, 0.08, 0.04), (0, 0, 0))]
    for k, (s, o) in enumerate(fr):
        if not along_x:
            s = (s[1], s[0], s[2]); o = (o[1], o[0], o[2])
        box(f"{name}_f{k}", s, Vector(loc) + Vector(o), m_dark)


window("WinFront", (CAB.x + 0.6, fy, gz + 0.72))
window("WinSide", (CAB.x + 1.23, CAB.y + 0.1, gz + 0.72), along_x=False)
light("WinLightF", "POINT", (CAB.x + 0.6, fy - 0.5, gz + 0.7), 40, (1.0, 0.6, 0.25), size=0.3)
light("WinLightS", "POINT", (CAB.x + 1.7, CAB.y + 0.1, gz + 0.7), 40, (1.0, 0.6, 0.25), size=0.3)
ch = Vector((CAB.x - 0.7, CAB.y + 0.45, wall_top + 0.2))
box("Chimney", (0.4, 0.4, 1.2), ch, m_stone, base=True)
box("ChimneyCap", (0.5, 0.5, 0.08), ch + Vector((0, 0, 1.2)), m_rock, base=True)
for i in range(6):
    sphere(f"Smoke{i}", 0.14 + i * 0.07, ch + Vector((0.12 * i * i * 0.35, 0.05 * i, 1.5 + i * 0.45)), m_smoke,
           ico=1, smooth=False)

# lantern post by the door
lp = Vector((CAB.x + 0.55, fy - 1.0, height(CAB.x + 0.55, fy - 1.0)))
cyl("LanternPost", 0.035, 1.1, lp, m_dark, segs=6, smooth=False)
box("LanternArm", (0.3, 0.04, 0.04), lp + Vector((0.12, 0, 1.08)), m_dark)
box("Lantern", (0.13, 0.13, 0.18), lp + Vector((0.25, 0, 0.92)), m_glow)
light("LanternLight", "POINT", lp + Vector((0.25, -0.15, 0.9)), 25, (1.0, 0.62, 0.25), size=0.1)

# stepping stones door -> pond
a0 = Vector((CAB.x - 0.35, fy - 0.8)); b0 = POND + (a0 - POND).normalized() * 1.9
for i in range(7):
    t = i / 6
    p = a0.lerp(b0, t) + Vector((0.18 * math.sin(i * 2.1), 0.12 * math.cos(i * 1.7)))
    cyl(f"PathStone{i}", random.uniform(0.16, 0.24), 0.05, (p.x, p.y, height(p.x, p.y) - 0.01), m_stone, segs=7,
        smooth=False, rot=(0, 0, random.uniform(0, 3)))

# fence arc behind the cabin
prev = None
for i in range(12):
    a = math.radians(55 + i * 9)
    r = redge(a) - 0.75
    p = Vector((r * math.cos(a), r * math.sin(a))); z = height(p.x, p.y) * 0.95
    cyl(f"FencePost{i}", 0.04, 0.5, (p.x, p.y, z - 0.05), m_plankB, segs=5, smooth=False)
    if prev:
        for hh in (0.2, 0.36):
            rod(f"FenceRail{i}_{hh}", (prev[0].x, prev[0].y, prev[1] + hh), (p.x, p.y, z + hh), 0.02, m_plankA, segs=4)
    prev = (p, z)

# ---- windmill
group("Windmill")
mz = height(MILL.x, MILL.y) - 0.1
cyl("MillTower", 0.62, 2.5, (MILL.x, MILL.y, mz), m_white, segs=8, r2=0.38, smooth=False)
cyl("MillBase", 0.72, 0.3, (MILL.x, MILL.y, mz), m_stone, segs=8, smooth=False)
cyl("MillCap", 0.52, 0.6, (MILL.x, MILL.y, mz + 2.5), m_roof, segs=8, r2=0.0, smooth=False)
box("MillDoor", (0.3, 0.06, 0.6), (MILL.x + 0.44, MILL.y - 0.44, mz + 0.3), m_dark, rot=(0, 0, math.radians(45)), base=True)
box("MillWindow", (0.2, 0.06, 0.25), (MILL.x + 0.33, MILL.y - 0.33, mz + 1.55), m_glow, rot=(0, 0, math.radians(45)))
hub = bpy.data.objects.new("MillHub", None)
hub.location = (MILL.x + 0.42, MILL.y - 0.42, mz + 2.25)
hub.rotation_euler = (0, 0, math.radians(-135))
S["col"].objects.link(hub)
axle = cyl("MillAxle", 0.06, 0.3, (0, -0.1, 0), m_dark, segs=8, rot=(-math.pi / 2, 0, 0)); axle.parent = hub
for k in range(4):
    ry = math.radians(k * 90 + 24)
    spar = box(f"Spar{k}", (0.05, 0.05, 1.7), (0, 0.16, 0), m_dark, rot=(0, ry, 0), base=True); spar.parent = hub
    sail = box(f"Sail{k}", (0.42, 0.025, 1.25), (0, 0.17, 0), m_sail, rot=(0, ry, 0), base=True); sail.parent = hub
    for v in sail.data.vertices:
        v.co.x += 0.2; v.co.z += 0.4

# ---- trees & rocks
group("Trees")
placed = []


def free(p, r=0.75):
    if (p - CAB).length < 2.2 or (p - POND).length < 2.1 or (p - MILL).length < 1.2:
        return False
    if p.y < CAB.y + 0.6 and p.x > CAB.x - 1.6 and (p - CAB).length < 4.2:
        return False
    if p.y < MILL.y + 1.0 and p.x > MILL.x - 0.5 and (p - MILL).length < 3.0:
        return False
    if seg_dist(p, POND, STREAM_END) < 0.8 or seg_dist(p, a0, b0) < 0.6:
        return False
    return all((p - q).length > r for q in placed)


def pine(name, p, z, s):
    m = random.choice(m_pines)
    cyl(name + "_trunk", 0.09 * s, 0.6 * s, (p.x, p.y, z - 0.05), m_trunk, segs=6, smooth=False)
    for k in range(4):
        cyl(f"{name}_c{k}", (0.62 - k * 0.13) * s, 0.75 * s, (p.x, p.y, z + (0.4 + k * 0.42) * s), m, segs=7, r2=0.0,
            smooth=False, rot=(0, 0, random.uniform(0, 1)))


def round_tree(name, p, z, s):
    m = random.choice(m_autumn)
    cyl(name + "_trunk", 0.1 * s, 0.9 * s, (p.x, p.y, z - 0.05), m_trunk, segs=6, smooth=False)
    for k in range(4):
        o = Vector((random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3), 1.1 + random.uniform(-0.1, 0.4))) * s
        sphere(f"{name}_l{k}", random.uniform(0.38, 0.55) * s, (p.x + o.x, p.y + o.y, z + o.z), m, ico=1, smooth=False,
               rough=0.15, seed=k)


count = 0
for _ in range(600):
    a = random.uniform(0, 6.283); r = math.sqrt(random.random()) * 5.2
    p = Vector((r * math.cos(a), r * math.sin(a)))
    if r > redge(a) - 0.8 or not free(p):
        continue
    placed.append(p)
    z = height(p.x, p.y)
    if count % 5 == 4:
        round_tree(f"Tree{count}", p, z, random.uniform(0.8, 1.15))
    else:
        pine(f"Pine{count}", p, z, random.uniform(0.7, 1.35))
    count += 1
    if count >= 24:
        break

group("Rocks")
for i in range(16):
    a = random.uniform(0, 6.283); r = random.uniform(0.8, 5.3)
    p = Vector((r * math.cos(a), r * math.sin(a)))
    if (p - CAB).length < 1.9 or (p - MILL).length < 0.9:
        continue
    s = random.uniform(0.12, 0.38)
    sphere(f"Rock{i}", s, (p.x, p.y, height(p.x, p.y) + s * 0.2), m_rock, ico=1, smooth=False, rough=0.3, seed=i,
           scale=(1, random.uniform(0.7, 1.3), 0.7))

# ---- floating islets
group("Islets")
for i, (x, y, z, r) in enumerate([(8.5, 2.5, -1.5, 1.1), (-8.0, -1.0, 1.0, 1.4), (5.5, 7.5, 2.2, 0.9), (-5.5, 6.5, -2.5, 0.8),
                                  (7.5, -3.0, -4.5, 0.6)]):
    isl = sphere(f"Islet{i}", r, (x, y, z), m_terrain, ico=2, smooth=False, rough=0.25, seed=i + 3, scale=(1, 1, 1.5))
    for v in isl.data.vertices:
        if v.co.z > r * 0.25:
            v.co.z = r * 0.25 + (v.co.z - r * 0.25) * 0.12
    pine(f"IsletPine{i}", Vector((x, y)), z + r * 0.3, r * 0.8)

# ---- clouds
group("Clouds")
for i, (x, y, z, s) in enumerate([(-9, 6, -5, 2.2), (9, 8, -3, 2.6), (3, 12, 4, 2.4), (-12, 10, 5, 3.0), (10, -1, -7.5, 1.8),
                                  (-6, -4, -8, 2.0), (-14, 2, -2, 2.2), (14, 12, 1, 2.8), (0, 4, -10, 2.4)]):
    for k in range(6):
        sphere(f"Cloud{i}_{k}", s * random.uniform(0.35, 0.6),
               (x + (k - 2.5) * s * 0.42, y + random.uniform(-0.4, 0.4) * s, z + random.uniform(-0.12, 0.2) * s),
               m_cloud, ico=2, scale=(1.2, 1, 0.62))

# ---- hot-air balloon
group("Balloon")
BL = Vector((-12.0, -3.0, 3.0))
prof = [(0, 1.75)] + [(1.5 * math.sin(math.radians(t)) * (1 if t < 95 else 1 - (t - 95) / 140), 1.7 * math.cos(math.radians(t)))
                      for t in range(15, 166, 15)] + [(0.28, -1.95)]
bal = lathe("Balloon", prof, BL, None, segs=12, smooth=False)
for m in m_ball:
    bal.data.materials.append(m)
for poly in bal.data.polygons:
    ang_ = math.atan2(poly.center.y, poly.center.x) % (2 * math.pi)
    poly.material_index = int(ang_ / (2 * math.pi / 12)) % 2
box("Basket", (0.5, 0.5, 0.4), BL + Vector((0, 0, -2.9)), m_plankB)
for dx in (-0.22, 0.22):
    for dy in (-0.22, 0.22):
        rod(f"Rope{dx}{dy}", BL + Vector((dx, dy, -2.7)), BL + Vector((dx * 1.2, dy * 1.2, -1.93)), 0.012, m_dark, segs=4)
light("Burner", "POINT", BL + Vector((0, 0, -2.2)), 60, (1.0, 0.5, 0.15), size=0.15)

# ---- rig
group("Rig")
light("Sun", "SUN", (14, -5, 5.5), 4.5, (1.0, 0.68, 0.45), size=3.0, target=(0, 0, 0))
light("SkyFill", "AREA", (-10, -12, 10), 1500, (0.45, 0.55, 1.0), size=12, target=(0, 0, 0))
camera("Cam", (14.0, -18.5, 4.0), (0.0, 0.3, -1.5), lens=31, fstop=5.6, focus=(0, 0, 0.5))
result = {"objects": len(sc.objects), "trees": count}
