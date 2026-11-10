"""Scene 1 (simple, not basic): lamp-lit still life on a side table."""
import os
exec(open(os.path.join(HERE, "common.py")).read())
random.seed(11)

sc = new_scene("S1_StillLife")
setup_cycles(samples=192, exposure=0.2)
world_color((0.02, 0.03, 0.04), 1.0)

# ---- materials
m_wall = mat("s1_wall", (0.035, 0.11, 0.12), rough=0.85)
m_wood = wood_mat("s1_wood", (0.20, 0.09, 0.04), (0.42, 0.22, 0.10), scale=2.5)
m_leg = mat("s1_leg", (0.03, 0.03, 0.035), rough=0.35, metal=1.0)
m_brass = mat("s1_brass", (0.83, 0.58, 0.22), rough=0.25, metal=1.0)
m_shade_in = mat("s1_shade_in", (0.95, 0.9, 0.8), rough=0.6)
m_bulb = mat("s1_bulb", (1, 0.8, 0.5), emit=(1.0, 0.62, 0.28), emit_str=40)
m_mug = mat("s1_mug", (0.86, 0.83, 0.76), rough=0.18, coat=0.6)
m_coffee = mat("s1_coffee", (0.05, 0.02, 0.01), rough=0.05)
m_pages = mat("s1_pages", (0.9, 0.86, 0.75), rough=0.9)
m_pot = mat("s1_pot", (0.62, 0.27, 0.14), rough=0.8)
m_soil = mat("s1_soil", (0.04, 0.03, 0.02), rough=1.0)
m_leaf = mat("s1_leaf", (0.05, 0.28, 0.08), rough=0.35, coat=0.3, sss=0.1)
m_stem = mat("s1_stem", (0.12, 0.25, 0.06), rough=0.6)
book_cols = [(0.55, 0.09, 0.07), (0.9, 0.62, 0.12), (0.08, 0.16, 0.38)]

# orange with pore bumps
m_orange, nt, p = mat_nodes("s1_orange")
p.inputs["Base Color"].default_value = (0.95, 0.38, 0.03, 1)
p.inputs["Roughness"].default_value = 0.38
p.inputs["Subsurface Weight"].default_value = 0.15
nz = nt.nodes.new("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 260
bp = nt.nodes.new("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.25
nt.links.new(nz.outputs["Fac"], bp.inputs["Height"]); nt.links.new(bp.outputs[0], p.inputs["Normal"])

# ---- backdrop: floor sweeping up into wall
group("Backdrop")
prof = [(-5.0, 0.0), (1.2, 0.0)]
for i in range(1, 13):
    a = math.radians(-90 + i * 7.5)
    prof.append((1.2 + 1.4 * math.cos(a), 1.4 + 1.4 * math.sin(a)))
prof.append((2.6, 6.0))
verts, faces = [], []
for i, (y, z) in enumerate(prof):
    verts += [(-7, y, z), (7, y, z)]
    if i:
        faces.append((2 * i - 2, 2 * i - 1, 2 * i + 1, 2 * i))
mesh_from("Backdrop", verts, faces, m_wall, smooth=True, sharp=0)

# ---- table
group("Table")
TOP = 0.62
cyl("TableTop", 0.55, 0.04, (0, 0, TOP - 0.04), m_wood, segs=96, bevel=0.008)
cyl("TableRing", 0.50, 0.03, (0, 0, TOP - 0.07), m_leg, segs=96)
for i in range(3):
    a = math.radians(90 + i * 120)
    rod(f"Leg{i}", (0.42 * math.cos(a), 0.42 * math.sin(a), TOP - 0.06),
        (0.52 * math.cos(a), 0.52 * math.sin(a), 0), 0.016, m_leg, r2=0.011)

# ---- books
group("Books")
z = TOP
bx, by = -0.16, 0.06
for i, c in enumerate(book_cols):
    w, d, h = 0.26 - i * 0.02, 0.18 - i * 0.012, 0.034 + (i % 2) * 0.008
    rz = math.radians(random.uniform(-14, 14))
    mc = mat(f"s1_book{i}", c, rough=0.55)
    box(f"BookCover{i}", (w, d, h), (bx, by, z), mc, rot=(0, 0, rz), bevel=0.003, base=True)
    pg = box(f"BookPages{i}", (w - 0.012, d - 0.006, h - 0.008), (bx, by, z + 0.004), m_pages,
             rot=(0, 0, rz), base=True)
    pg.location += Matrix.Rotation(rz, 3, "Z") @ Vector((0.004, -0.006, 0))
    z += h
BOOK_TOP = z

# ---- mug (on the books)
group("Mug")
mx, my = bx + 0.01, by
lathe("Mug", [(0, 0), (0.036, 0), (0.041, 0.004), (0.043, 0.02), (0.043, 0.085), (0.041, 0.088),
              (0.039, 0.085), (0.039, 0.012), (0, 0.008)], (mx, my, BOOK_TOP), m_mug, segs=64)
cyl("Coffee", 0.0392, 0.002, (mx, my, BOOK_TOP + 0.072), m_coffee, segs=64)
h = torus("MugHandle", 0.026, 0.006, (mx + 0.046, my - 0.012, BOOK_TOP + 0.046), m_mug,
          rot=(math.pi / 2, 0, math.radians(-15)))
h.scale = (0.85, 1.15, 1.0)

# ---- desk lamp
group("Lamp")
lx, ly = -0.30, 0.30
cyl("LampBase", 0.085, 0.018, (lx, ly, TOP), m_brass, segs=64, bevel=0.004)
elbow = Vector((lx - 0.03, ly + 0.02, TOP + 0.42))
head = Vector((lx + 0.20, ly - 0.16, TOP + 0.50))
rod("LampArm1", (lx, ly, TOP + 0.015), elbow, 0.008, m_brass)
rod("LampArm2", elbow, head, 0.008, m_brass)
sphere("LampJoint", 0.016, elbow, m_brass)
aim = Vector((bx + 0.05, by - 0.05, BOOK_TOP)) - head
q = Vector((0, 0, -1)).rotation_difference(aim.normalized())
shade = lathe("LampShade", [(0.0, 0.03), (0.03, 0.03), (0.035, 0.0), (0.105, -0.13), (0.108, -0.13),
                            (0.038, 0.004), (0.03, 0.034), (0.0, 0.034)], head, m_brass, segs=64)
inner = lathe("LampShadeInner", [(0.033, -0.002), (0.103, -0.129)], head, m_shade_in, segs=64)
bulb = sphere("Bulb", 0.028, head + q @ Vector((0, 0, -0.055)), m_bulb)
for o in (shade, inner):
    o.rotation_mode = "QUATERNION"; o.rotation_quaternion = q
sp = light("LampSpot", "SPOT", head + q @ Vector((0, 0, -0.09)), 45, (1.0, 0.72, 0.42), size=0.03,
           spot=115, blend=0.5)
sp.rotation_mode = "QUATERNION"; sp.rotation_quaternion = q

# ---- plant
group("Plant")
px, py = 0.24, 0.20
lathe("Pot", [(0, 0), (0.055, 0), (0.075, 0.11), (0.082, 0.11), (0.082, 0.125), (0.068, 0.125),
              (0.066, 0.1), (0, 0.1)], (px, py, TOP), m_pot, segs=48)
cyl("Soil", 0.067, 0.004, (px, py, TOP + 0.098), m_soil, segs=48)


def leaf(name, length, width, droop, fold):
    n = 14
    verts, faces = [], []
    for i in range(n + 1):
        t = i / n
        w = width * (math.sin(math.pi * min(1, t * 1.08)) ** 0.7) * (1 - 0.25 * t)
        x = length * t
        zc = length * 0.9 * t - droop * t * t
        verts += [(x, -w, zc + fold * w), (x, 0, zc), (x, w, zc + fold * w)]
        if i:
            b = 3 * (i - 1)
            faces += [(b, b + 1, b + 4, b + 3), (b + 1, b + 2, b + 5, b + 4)]
    ob = mesh_from(name, verts, faces, m_leaf, smooth=True, sharp=0)
    md = ob.modifiers.new("Solid", "SOLIDIFY"); md.thickness = 0.0015
    md = ob.modifiers.new("Sub", "SUBSURF"); md.levels = 1; md.render_levels = 2
    return ob


for i in range(11):
    a = i * 2.39996 + random.uniform(-0.2, 0.2)
    ln = random.uniform(0.2, 0.34)
    lf = leaf(f"Leaf{i}", ln, ln * 0.2, ln * random.uniform(0.35, 0.95), 0.35)
    base = Vector((px + 0.02 * math.cos(a), py + 0.02 * math.sin(a), TOP + 0.1))
    lf.location = base
    lf.rotation_euler = (random.uniform(-0.3, 0.3), -random.uniform(0.1, 0.45), a)

# ---- orange + slice of peel
group("Fruit")
sphere("Orange", 0.04, (0.06, -0.30, TOP + 0.039), m_orange, scale=(1, 1, 0.94))
cyl("OrangeStem", 0.004, 0.004, (0.06, -0.30, TOP + 0.075), m_stem, segs=8)
sphere("Orange2", 0.037, (0.15, -0.24, TOP + 0.036), m_orange, scale=(1, 1, 0.94))
cyl("Orange2Stem", 0.004, 0.004, (0.15, -0.24, TOP + 0.069), m_stem, segs=8)

# ---- lights + camera
group("Rig")
light("Key", "AREA", (-2.6, -2.2, 2.6), 70, (0.72, 0.84, 1.0), size=2.2, target=(0, 0, TOP))
light("Rim", "AREA", (2.4, 1.6, 1.9), 110, (1.0, 0.6, 0.35), size=1.0, target=(0, 0, TOP + 0.2))
light("WallGlow", "SPOT", (0.4, -0.5, 0.25), 60, (0.2, 0.75, 0.8), size=0.4, target=(0.3, 2.5, 2.0), spot=95,
      blend=1.0)
camera("Cam", (1.45, -2.05, 1.22), (0.0, 0.05, TOP + 0.2), lens=62, fstop=3.2, focus=(mx, my, BOOK_TOP + 0.04))
result = {"objects": len(sc.objects)}
