"""Scene 3 (complex): rainy neon city street at night - procedural buildings, signs, traffic, rail, fog."""
import os
exec(open(os.path.join(HERE, "common.py")).read())
random.seed(42)

sc = new_scene("S3_NeonCity")
setup_cycles(samples=256, exposure=-0.25)
sc.cycles.max_bounces = 6
sc.cycles.volume_bounces = 0
sc.cycles.sample_clamp_indirect = 4.0
world_color((0.006, 0.008, 0.02), 1.0)

ROAD_W, WALK_W = 5.0, 3.0           # half road width, sidewalk width
FRONT = ROAD_W + WALK_W             # building line (|x|)
CROSS = (41.0, 55.0)                # cross street (elevated rail runs here)

NEON = {"magenta": (1, 0.05, 0.45), "cyan": (0.05, 0.75, 1), "yellow": (1, 0.7, 0.08), "green": (0.15, 1, 0.4),
        "red": (1, 0.08, 0.06), "purple": (0.5, 0.18, 1), "orange": (1, 0.35, 0.04), "white": (1, 0.9, 0.75)}
NEON_KEYS = list(NEON)


def neon(key, strength=10):
    return mat(f"s3_neon_{key}_{strength}", NEON[key], rough=0.4, emit=NEON[key], emit_str=strength * 0.45)


# ---------------- materials ----------------
def building_mat(name, base, col_a, col_b, density, win_w, row_h, strength):
    m, nt, p = mat_nodes(name)
    N, L = nt.nodes.new, nt.links.new
    tc = N("ShaderNodeTexCoord"); sx = N("ShaderNodeSeparateXYZ")
    oi = N("ShaderNodeObjectInfo")
    add = N("ShaderNodeMath"); add.operation = "ADD"
    rnd = N("ShaderNodeMath"); rnd.operation = "MULTIPLY"; rnd.inputs[1].default_value = 97.0
    add2 = N("ShaderNodeMath"); add2.operation = "ADD"
    cb = N("ShaderNodeCombineXYZ")
    br = N("ShaderNodeTexBrick")
    br.offset = 0.0
    br.inputs["Color1"].default_value = (0, 0, 0, 1); br.inputs["Color2"].default_value = (1, 1, 1, 1)
    br.inputs["Mortar"].default_value = (0, 0, 0, 1)
    br.inputs["Scale"].default_value = 1.0
    br.inputs["Mortar Size"].default_value = 0.32
    br.inputs["Mortar Smooth"].default_value = 0.0
    br.inputs["Bias"].default_value = 0.0
    br.inputs["Brick Width"].default_value = win_w
    br.inputs["Row Height"].default_value = row_h
    gt = N("ShaderNodeMath"); gt.operation = "GREATER_THAN"; gt.inputs[1].default_value = 1.0 - density
    cr = N("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].position = 1.0 - density; cr.color_ramp.elements[0].color = tuple(col_a) + (1,)
    cr.color_ramp.elements[1].color = tuple(col_b) + (1,)
    geo = N("ShaderNodeNewGeometry"); sn = N("ShaderNodeSeparateXYZ")
    ab = N("ShaderNodeMath"); ab.operation = "ABSOLUTE"
    side = N("ShaderNodeMath"); side.operation = "LESS_THAN"; side.inputs[1].default_value = 0.5
    mul = N("ShaderNodeMath"); mul.operation = "MULTIPLY"
    st = N("ShaderNodeMath"); st.operation = "MULTIPLY"; st.inputs[1].default_value = strength
    rr = N("ShaderNodeMapRange")
    rr.inputs["To Min"].default_value = 0.12; rr.inputs["To Max"].default_value = 0.75
    L(tc.outputs["Object"], sx.inputs[0])
    L(sx.outputs["X"], add.inputs[0]); L(sx.outputs["Y"], add.inputs[1])
    L(oi.outputs["Random"], rnd.inputs[0])
    L(add.outputs[0], add2.inputs[0]); L(rnd.outputs[0], add2.inputs[1])
    L(add2.outputs[0], cb.inputs["X"]); L(sx.outputs["Z"], cb.inputs["Y"])
    L(cb.outputs[0], br.inputs["Vector"])
    L(br.outputs["Color"], gt.inputs[0]); L(br.outputs["Color"], cr.inputs["Fac"])
    L(geo.outputs["True Normal"], sn.inputs[0]); L(sn.outputs["Z"], ab.inputs[0]); L(ab.outputs[0], side.inputs[0])
    dv = N("ShaderNodeMath"); dv.operation = "DIVIDE"; dv.inputs[1].default_value = row_h
    fr = N("ShaderNodeMath"); fr.operation = "FRACT"
    lo = N("ShaderNodeMath"); lo.operation = "GREATER_THAN"; lo.inputs[1].default_value = 0.30
    hi = N("ShaderNodeMath"); hi.operation = "LESS_THAN"; hi.inputs[1].default_value = 0.74
    m2 = N("ShaderNodeMath"); m2.operation = "MULTIPLY"; m3 = N("ShaderNodeMath"); m3.operation = "MULTIPLY"
    m4 = N("ShaderNodeMath"); m4.operation = "MULTIPLY"
    L(sx.outputs["Z"], dv.inputs[0]); L(dv.outputs[0], fr.inputs[0]); L(fr.outputs[0], lo.inputs[0]); L(fr.outputs[0], hi.inputs[0])
    L(lo.outputs[0], m2.inputs[0]); L(hi.outputs[0], m2.inputs[1])
    L(gt.outputs[0], mul.inputs[0]); L(side.outputs[0], mul.inputs[1])
    L(mul.outputs[0], m3.inputs[0]); L(m2.outputs[0], m3.inputs[1])
    L(m3.outputs[0], st.inputs[0])
    L(cr.outputs["Color"], p.inputs["Emission Color"]); L(st.outputs[0], p.inputs["Emission Strength"])
    L(br.outputs["Fac"], rr.inputs["Value"]); L(rr.outputs[0], p.inputs["Roughness"])
    p.inputs["Base Color"].default_value = tuple(base) + (1,)
    return m


B_MATS = [
    building_mat("s3_bld_warm", (0.035, 0.03, 0.03), (1.0, 0.55, 0.2), (1.0, 0.85, 0.6), 0.30, 1.6, 3.0, 1.6),
    building_mat("s3_bld_cool", (0.02, 0.025, 0.035), (0.3, 0.7, 1.0), (0.85, 0.95, 1.0), 0.34, 1.3, 3.2, 1.4),
    building_mat("s3_bld_office", (0.015, 0.02, 0.03), (0.2, 0.9, 0.9), (0.9, 1.0, 0.95), 0.40, 1.9, 3.6, 0.8),
    building_mat("s3_bld_mixed", (0.04, 0.03, 0.045), (1.0, 0.3, 0.6), (1.0, 0.8, 0.5), 0.22, 1.4, 2.8, 1.7),
    building_mat("s3_bld_dark", (0.025, 0.025, 0.03), (1.0, 0.7, 0.35), (1.0, 0.9, 0.8), 0.13, 1.8, 3.0, 1.8),
]


def road_mat():
    m, nt, p = mat_nodes("s3_road")
    N, L = nt.nodes.new, nt.links.new
    tc = N("ShaderNodeTexCoord")
    nz = N("ShaderNodeTexNoise"); nz.inputs["Scale"].default_value = 0.22; nz.inputs["Detail"].default_value = 6.0
    cr = N("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].position = 0.42; cr.color_ramp.elements[0].color = (0.02, 0.02, 0.02, 1)
    cr.color_ramp.elements[1].position = 0.62; cr.color_ramp.elements[1].color = (0.42, 0.42, 0.42, 1)
    n2 = N("ShaderNodeTexNoise"); n2.inputs["Scale"].default_value = 14.0; n2.inputs["Detail"].default_value = 3.0
    bp = N("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.06
    L(tc.outputs["Object"], nz.inputs["Vector"]); L(tc.outputs["Object"], n2.inputs["Vector"])
    L(nz.outputs["Fac"], cr.inputs["Fac"]); L(cr.outputs["Color"], p.inputs["Roughness"])
    L(n2.outputs["Fac"], bp.inputs["Height"]); L(bp.outputs[0], p.inputs["Normal"])
    p.inputs["Base Color"].default_value = (0.012, 0.012, 0.015, 1)
    return m


def walk_mat():
    m, nt, p = mat_nodes("s3_walk")
    N, L = nt.nodes.new, nt.links.new
    tc = N("ShaderNodeTexCoord")
    br = N("ShaderNodeTexBrick"); br.offset = 0.5
    br.inputs["Scale"].default_value = 1.0; br.inputs["Brick Width"].default_value = 1.0
    br.inputs["Row Height"].default_value = 1.0; br.inputs["Mortar Size"].default_value = 0.03
    br.inputs["Color1"].default_value = (0.07, 0.07, 0.075, 1); br.inputs["Color2"].default_value = (0.10, 0.10, 0.11, 1)
    br.inputs["Mortar"].default_value = (0.01, 0.01, 0.01, 1)
    bp = N("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.4; bp.invert = True
    L(tc.outputs["Object"], br.inputs["Vector"]); L(br.outputs["Color"], p.inputs["Base Color"])
    L(br.outputs["Fac"], bp.inputs["Height"]); L(bp.outputs[0], p.inputs["Normal"])
    p.inputs["Roughness"].default_value = 0.28
    return m


def ad_mat(name, tint, scale, strength=5.0):
    m, nt, p = mat_nodes(name)
    N, L = nt.nodes.new, nt.links.new
    tc = N("ShaderNodeTexCoord"); mp = N("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (scale, scale, scale)
    mg = N("ShaderNodeTexMagic"); mg.turbulence_depth = 3
    mg.inputs["Scale"].default_value = 1.0; mg.inputs["Distortion"].default_value = 1.6
    mx = N("ShaderNodeMix"); mx.data_type = "RGBA"; mx.blend_type = "MULTIPLY"; mx.inputs["Factor"].default_value = 0.75
    mx.inputs["B"].default_value = tuple(tint) + (1,)
    L(tc.outputs["Object"], mp.inputs["Vector"]); L(mp.outputs[0], mg.inputs["Vector"])
    L(mg.outputs["Color"], mx.inputs["A"]); L(mx.outputs["Result"], p.inputs["Emission Color"])
    p.inputs["Emission Strength"].default_value = strength
    p.inputs["Base Color"].default_value = (0, 0, 0, 1)
    return m


def shop_mat(name, c1, c2, strength=1.6):
    """Lit shop window: rows of coloured 'shelves' seen through the glass."""
    m, nt, p = mat_nodes(name)
    N, L = nt.nodes.new, nt.links.new
    tc = N("ShaderNodeTexCoord"); sx = N("ShaderNodeSeparateXYZ"); cb = N("ShaderNodeCombineXYZ")
    br = N("ShaderNodeTexBrick"); br.offset = 0.5
    br.inputs["Scale"].default_value = 1.0; br.inputs["Brick Width"].default_value = 0.45
    br.inputs["Row Height"].default_value = 0.42; br.inputs["Mortar Size"].default_value = 0.035
    br.inputs["Color1"].default_value = tuple(c1) + (1,); br.inputs["Color2"].default_value = tuple(c2) + (1,)
    br.inputs["Mortar"].default_value = (0.02, 0.015, 0.01, 1)
    L(tc.outputs["Object"], sx.inputs[0]); L(sx.outputs["Y"], cb.inputs["X"]); L(sx.outputs["Z"], cb.inputs["Y"])
    L(cb.outputs[0], br.inputs["Vector"]); L(br.outputs["Color"], p.inputs["Emission Color"])
    p.inputs["Emission Strength"].default_value = strength
    p.inputs["Base Color"].default_value = (0.01, 0.01, 0.01, 1); p.inputs["Roughness"].default_value = 0.05
    return m


m_road = road_mat(); m_walk = walk_mat()
m_shops = [shop_mat("s3_shop0", (1.0, 0.55, 0.2), (1.0, 0.85, 0.55)), shop_mat("s3_shop1", (0.2, 0.75, 1.0), (0.9, 0.95, 1.0)),
           shop_mat("s3_shop2", (1.0, 0.2, 0.5), (1.0, 0.75, 0.4)), shop_mat("s3_shop3", (0.3, 1.0, 0.5), (0.95, 1.0, 0.7)),
           shop_mat("s3_shop4", (0.6, 0.3, 1.0), (1.0, 0.7, 0.9))]
m_ads = [ad_mat("s3_ad0", (1, 0.3, 0.8), 0.25), ad_mat("s3_ad1", (0.3, 0.9, 1), 0.4), ad_mat("s3_ad2", (1, 0.7, 0.2), 0.3)]
m_metal = mat("s3_metal", (0.05, 0.05, 0.055), rough=0.35, metal=1.0)
m_dark = mat("s3_dark", (0.012, 0.012, 0.015), rough=0.6)
m_conc = mat("s3_concrete", (0.07, 0.07, 0.075), rough=0.7)
m_rust = mat("s3_rust", (0.12, 0.05, 0.03), rough=0.7, metal=0.6)
m_paint = mat("s3_roadpaint", (0.55, 0.5, 0.35), rough=0.5)
m_glass = mat("s3_glass", (0.01, 0.015, 0.02), rough=0.03, metal=0.9)
m_skin = mat("s3_skin", (0.5, 0.35, 0.28), rough=0.6)
m_rain = mat("s3_rain", (0.6, 0.7, 0.8), rough=0.3, emit=(0.6, 0.75, 1.0), emit_str=0.12)
m_beacon = neon("red", 30)

# ---------------- ground ----------------
group("Ground")
plane("Asphalt", 600, 600, (0, 100, 0), m_road)
for side in (-1, 1):
    for y0, y1 in ((-30, CROSS[0]), (CROSS[1], 230)):
        box(f"Sidewalk{side}_{y0}", (WALK_W, y1 - y0, 0.18), (side * (ROAD_W + WALK_W / 2), (y0 + y1) / 2, 0), m_walk, base=True)
        box(f"Curb{side}_{y0}", (0.2, y1 - y0, 0.22), (side * (ROAD_W + 0.1), (y0 + y1) / 2, 0), m_conc, base=True)
y = -24
while y < 220:
    if not (CROSS[0] - 6 < y < CROSS[1] + 2):
        box(f"Dash{int(y)}", (0.16, 2.6, 0.012), (0, y, 0.006), m_paint)
    y += 6.5
for i in range(11):
    box(f"Zebra{i}", (0.55, 3.2, 0.012), (-ROAD_W + 0.7 + i * 0.86, CROSS[0] - 3.5, 0.006), m_paint)
for i in range(4):
    cyl(f"Manhole{i}", 0.45, 0.012, (random.choice((-2.3, 2.3)), 8 + i * 27, 0.002), m_metal, segs=20)

# ---------------- signs ----------------
WORDS = ["RAMEN", "HOTEL", "BAR", "24H", "ARCADE", "SUSHI", "TAXI", "CAFE", "NOODLE", "OPEN", "KARAOKE", "TEA", "BOOKS",
         "LASER", "PIXEL", "CLUB", "SAUNA", "PAWN", "DINER", "VIDEO", "MOTEL", "PIZZA"]
sign_i = [0]


def blade_sign(center, size, key, vertical, side):
    """Sign board hanging off a facade, facing down the street toward the camera."""
    i = sign_i[0]; sign_i[0] += 1
    word = WORDS[i % len(WORDS)]
    if vertical:
        W, H = size * 1.5, len(word) * size * 0.98 + 0.5
    else:
        W, H = 0.74 * size * len(word) + 0.6, size * 1.6
    cx = side * (FRONT - W / 2 - 0.35)
    c = Vector((cx, center[1], center[2]))
    box(f"SignBoard{i}", (W, 0.16, H), c, m_dark)
    fm = neon(key, 14); t = 0.06
    for k, (s, o) in enumerate([((W, 0.2, t), (0, 0, H / 2)), ((W, 0.2, t), (0, 0, -H / 2)), ((t, 0.2, H), (W / 2, 0, 0)),
                                ((t, 0.2, H), (-W / 2, 0, 0))]):
        box(f"SignFrame{i}_{k}", s, c + Vector(o), fm)
    k2 = NEON_KEYS[(NEON_KEYS.index(key) + 3) % len(NEON_KEYS)]
    tx = text(f"SignText{i}", "\n".join(word) if vertical else word, c + Vector((0, -0.1, 0)), neon(k2, 16), size=size,
              extrude=0.03)
    tx.data.align_y = "CENTER"
    if vertical:
        tx.data.space_line = 0.9
    box(f"SignArm{i}", (0.5, 0.08, 0.08), (side * (FRONT - 0.2), center[1], center[2] + H / 2 - 0.2), m_metal)
    box(f"SignArmB{i}", (0.5, 0.08, 0.08), (side * (FRONT - 0.2), center[1], center[2] - H / 2 + 0.2), m_metal)


def shopfront(side, y0, y1, idx):
    w = y1 - y0 - 1.6
    key = NEON_KEYS[idx % len(NEON_KEYS)]
    x = side * (FRONT - 0.06)
    box(f"ShopGlass{idx}", (0.12, w, 2.5), (x, (y0 + y1) / 2, 0.55), m_shops[idx % len(m_shops)], base=True)
    for k in range(int(w // 2.2) + 1):          # mullions
        box(f"ShopMull{idx}_{k}", (0.2, 0.1, 2.6), (x, y0 + 0.8 + k * w / max(1, int(w // 2.2)), 0.5), m_dark, base=True)
    box(f"ShopBase{idx}", (0.3, w + 0.4, 0.4), (x, (y0 + y1) / 2, 0.18), m_conc, base=True)
    box(f"Awning{idx}", (1.5, w + 0.6, 0.08), (side * (FRONT - 0.7), (y0 + y1) / 2, 3.35), m_dark, rot=(0, side * 0.22, 0))
    box(f"AwningNeon{idx}", (0.08, w + 0.6, 0.1), (side * (FRONT - 1.45), (y0 + y1) / 2, 3.18), neon(key, 12))
    word = WORDS[(idx * 7 + 3) % len(WORDS)]
    tx = text(f"ShopText{idx}", word, (side * (FRONT - 0.12), (y0 + y1) / 2, 4.3), neon(NEON_KEYS[(idx + 4) % 8], 14), size=1.1,
              extrude=0.03, rot=(math.pi / 2, 0, -side * math.pi / 2))
    tx.data.align_y = "CENTER"
    # matte board behind the letters, so the glossy wall does not mirror them
    box(f"ShopTextBoard{idx}", (0.1, 0.8 * 1.1 * len(word) + 0.8, 1.6), (side * (FRONT - 0.05), (y0 + y1) / 2, 4.3),
        mat("s3_board", (0.01, 0.01, 0.012), rough=1.0))


# ---------------- buildings ----------------
def rooftop(x, y, d, w, h, idx, near):
    r = random.random()
    if r < 0.55:    # water tank
        tx, ty = x + random.uniform(-d / 4, d / 4), y + random.uniform(-w / 4, w / 4)
        for dx in (-0.7, 0.7):
            for dy in (-0.7, 0.7):
                rod(f"TankLeg{idx}{dx}{dy}", (tx + dx, ty + dy, h), (tx + dx, ty + dy, h + 1.6), 0.06, m_metal, segs=6)
        cyl(f"Tank{idx}", 1.2, 2.2, (tx, ty, h + 1.6), m_rust, segs=14)
        cyl(f"TankTop{idx}", 1.3, 0.7, (tx, ty, h + 3.8), m_rust, segs=14, r2=0.0)
    for k in range(random.randint(1, 4)):   # AC units
        box(f"AC{idx}_{k}", (random.uniform(1, 2), random.uniform(1, 2), random.uniform(0.7, 1.2)),
            (x + random.uniform(-d / 3, d / 3), y + random.uniform(-w / 3, w / 3), h), m_metal, base=True)
    if random.random() < 0.6:               # antenna + beacon
        ax, ay = x + random.uniform(-d / 3, d / 3), y + random.uniform(-w / 3, w / 3)
        ah = random.uniform(4, 11)
        rod(f"Antenna{idx}", (ax, ay, h), (ax, ay, h + ah), 0.07, m_metal, segs=6, r2=0.02)
        sphere(f"Beacon{idx}", 0.16, (ax, ay, h + ah), m_beacon, ico=1)
    box(f"Parapet{idx}", (d + 0.3, w + 0.3, 0.5), (x, y, h - 0.1), m_conc, base=True)
    if near and random.random() < 0.7:      # rooftop billboard facing the camera
        bw, bh = min(d - 1, random.uniform(6, 10)), random.uniform(3.5, 6)
        by = y - w / 2 + 0.6
        box(f"Billboard{idx}", (bw, 0.15, bh), (x, by, h + 2.2 + bh / 2), random.choice(m_ads))
        box(f"BillboardBack{idx}", (bw + 0.4, 0.2, bh + 0.4), (x, by + 0.15, h + 2.2 + bh / 2), m_dark)
        for sx_ in (-bw / 3, bw / 3):
            rod(f"BillLeg{idx}{sx_}", (x + sx_, by + 0.2, h), (x + sx_, by + 0.2, h + 2.2), 0.1, m_metal, segs=6)
            rod(f"BillBrace{idx}{sx_}", (x + sx_, by + 2.4, h), (x + sx_, by + 0.3, h + 2.2 + bh * 0.7), 0.06, m_metal, segs=6)


def fire_escape(side, y, h, idx):
    x = side * (FRONT - 0.55)
    floors = int(min(h, 26) // 3.2)
    for f in range(1, floors):
        z = 4.0 + f * 3.2
        box(f"FE{idx}_{f}", (1.1, 3.4, 0.07), (x, y, z), m_metal)
        for yy in (-1.7, 1.7):
            rod(f"FEpost{idx}_{f}_{yy}", (x - side * 0.5, y + yy, z), (x - side * 0.5, y + yy, z + 1.0), 0.025, m_metal, segs=4)
        rod(f"FErail{idx}_{f}", (x - side * 0.5, y - 1.7, z + 1.0), (x - side * 0.5, y + 1.7, z + 1.0), 0.025, m_metal, segs=4)
        if f < floors - 1:
            s = 1 if f % 2 else -1
            rod(f"FEstair{idx}_{f}", (x, y - s * 1.3, z), (x, y + s * 1.3, z + 3.2), 0.05, m_metal, segs=4)


group("Buildings")
shops = group("Shops"); signs = group("Signs"); roofs = group("Rooftops"); S["col"] = bpy.data.collections["Buildings"]
bcol = bpy.data.collections["Buildings"]
idx = 0
for side in (-1, 1):
    y = -30.0
    while y < 215:
        w = random.uniform(9, 17)
        if y < CROSS[0] < y + w:
            if CROSS[0] - y < 6:
                y = CROSS[1]; continue
            w = CROSS[0] - y
        if CROSS[0] <= y < CROSS[1]:
            y = CROSS[1]; continue
        d = random.uniform(10, 15)
        h = random.uniform(20, 42) + max(0, y) * random.uniform(0.05, 0.3)
        x = side * (FRONT + d / 2)
        yc = y + w / 2
        near = y < 110
        S["col"] = bcol
        box(f"Bld{idx}", (d, w - 0.5, h), (x, yc, 0), random.choice(B_MATS), base=True)
        top = h
        if random.random() < 0.45:
            h2 = h * random.uniform(0.2, 0.45)
            box(f"BldTier{idx}", (d * 0.7, (w - 0.5) * 0.7, h2), (x + side * d * 0.1, yc, h), random.choice(B_MATS), base=True)
            box(f"TierLedge{idx}", (d + 0.4, w - 0.1, 0.4), (x, yc, h - 0.2), m_conc, base=True)
            top = h + h2; dd, ww = d * 0.7, (w - 0.5) * 0.7; xx = x + side * d * 0.1
        else:
            dd, ww, xx = d, w - 0.5, x
        if near:
            box(f"Ledge{idx}", (d + 0.5, w - 0.3, 0.3), (x, yc, 3.6), m_conc, base=True)
            for k in range(random.randint(2, 6)):   # facade AC boxes
                box(f"FacAC{idx}_{k}", (0.6, 0.9, 0.6), (side * (FRONT - 0.3), yc + random.uniform(-w / 2 + 1, w / 2 - 1),
                                                        6 + random.randint(0, int(min(h, 30) // 3) - 2) * 3.0), m_metal)
        S["col"] = roofs
        rooftop(xx, yc, dd, ww, top, idx, near)
        if near:
            S["col"] = shops
            shopfront(side, y, y + w - 0.5, idx)
            S["col"] = signs
            n_s = random.choice((1, 1, 2))
            for k in range(n_s):
                vertical = random.random() < 0.55
                size = random.uniform(0.8, 1.25)
                blade_sign((0, y + 1.5 + (w - 3) * (k + 0.5) / n_s, random.uniform(7, 15) + k * 5), size,
                           random.choice(NEON_KEYS[:7]), vertical, side)
            if random.random() < 0.45 and w > 8:
                S["col"] = roofs
                fire_escape(side, yc + random.uniform(-1, 1), h, idx)
        idx += 1
        y += w

# skyline behind
group("Skyline")
for i in range(46):
    side = random.choice((-1, 1))
    x = side * random.uniform(28, 95); y = random.uniform(-10, 300)
    if CROSS[0] - 4 < y < CROSS[1] + 4:
        continue
    d = random.uniform(12, 22); h = random.uniform(60, 170)
    box(f"Sky{i}", (d, d * random.uniform(0.7, 1.3), h), (x, y, 0), random.choice(B_MATS), base=True)
    if random.random() < 0.6:
        rod(f"SkySpire{i}", (x, y, h), (x, y, h + 14), 0.25, m_metal, segs=6, r2=0.05)
        sphere(f"SkyBeacon{i}", 0.5, (x, y, h + 14), m_beacon, ico=1)

# hero tower at the end of the street
group("HeroTower")
TY = 235.0
box("Hero", (30, 30, 150), (0, TY, 0), B_MATS[2], base=True)
box("HeroTop", (20, 20, 40), (0, TY, 150), B_MATS[1], base=True)
rod("HeroSpire", (0, TY, 190), (0, TY, 240), 1.0, m_metal, segs=8, r2=0.1)
for z in (200, 215, 230, 240):
    sphere(f"HeroBeacon{z}", 0.9, (0, TY, z), m_beacon, ico=1)
torus("HoloRing", 27, 0.7, (0, TY, 128), neon("cyan", 25), rot=(math.radians(14), math.radians(6), 0), seg=64, ring=8)
torus("HoloRing2", 21, 0.45, (0, TY, 172), neon("magenta", 25), rot=(math.radians(-10), math.radians(-8), 0), seg=64, ring=8)
box("HeroScreen", (20, 0.3, 46), (0, TY - 15.3, 78), m_ads[0])
for sx_ in (-15.2, 15.2):
    box(f"HeroStrip{sx_}", (0.5, 0.5, 150), (sx_, TY - 15.2, 75), neon("cyan", 12))

# ---------------- elevated rail + train ----------------
group("Rail")
RY, RZ = 48.0, 10.5
box("RailDeck", (190, 5.2, 0.9), (0, RY, RZ), m_conc, base=True)
for sy in (-2.5, 2.5):
    box(f"RailWall{sy}", (190, 0.2, 1.0), (0, RY + sy, RZ + 0.9), m_metal, base=True)
box("RailUnderLight", (60, 0.15, 0.08), (0, RY - 2.0, RZ - 0.05), neon("cyan", 8))
for px in (-62, -42, -24, -6.6, 6.6, 24, 42, 62):
    box(f"RailPillar{px}", (1.3, 1.6, RZ), (px, RY, 0), m_conc, base=True)
    box(f"RailCap{px}", (2.4, 4.6, 0.8), (px, RY, RZ - 0.8), m_conc, base=True)
for k in range(4):
    tx = -17.0 + k * 13.2
    box(f"TrainCar{k}", (12.6, 3.0, 3.2), (tx, RY, RZ + 1.3), mat("s3_train", (0.5, 0.52, 0.55), rough=0.3, metal=0.9),
        base=True, bevel=0.25)
    box(f"TrainWin{k}", (11.4, 0.06, 1.0), (tx, RY - 1.52, RZ + 3.3), neon("white", 9))
    box(f"TrainStripe{k}", (12.4, 0.06, 0.18), (tx, RY - 1.52, RZ + 2.2), neon("orange", 6))
    for dx in (-3.4, 0, 3.4):
        box(f"TrainDoor{k}{dx}", (0.12, 0.1, 1.9), (tx + dx, RY - 1.54, RZ + 2.8), m_dark)

# ---------------- street furniture ----------------
group("Street")


def cable(name, p1, p2, sag, r=0.025, lights=None):
    p1, p2 = Vector(p1), Vector(p2)
    cu = bpy.data.curves.new(name, "CURVE"); cu.dimensions = "3D"; cu.bevel_depth = r; cu.bevel_resolution = 1
    sp = cu.splines.new("POLY"); n = 14
    sp.points.add(n)
    pts = []
    for i in range(n + 1):
        t = i / n
        p = p1.lerp(p2, t); p.z -= sag * 4 * t * (1 - t)
        sp.points[i].co = (p.x, p.y, p.z, 1); pts.append(p)
    cu.materials.append(m_dark)
    ob = bpy.data.objects.new(name, cu); S["col"].objects.link(ob)
    if lights:
        for i, p in enumerate(pts[1:-1]):
            sphere(f"{name}_b{i}", 0.11, (p.x, p.y, p.z - 0.16), neon(lights[i % len(lights)], 18), ico=1)


def street_lamp(side, y, i):
    x = side * (ROAD_W + 0.6)
    cyl(f"LampPole{i}", 0.09, 7.0, (x, y, 0.18), m_metal, segs=8, r2=0.06)
    rod(f"LampArm{i}", (x, y, 7.1), (x - side * 2.2, y, 7.5), 0.05, m_metal, segs=6)
    box(f"LampHead{i}", (0.9, 0.35, 0.12), (x - side * 2.3, y, 7.45), m_metal)
    box(f"LampGlow{i}", (0.8, 0.28, 0.04), (x - side * 2.3, y, 7.37), neon("orange" if i % 3 else "white", 40))
    if y < 130:
        light(f"LampSpot{i}", "SPOT", (x - side * 2.3, y, 7.3), 900, (1.0, 0.6, 0.3) if i % 3 else (0.85, 0.9, 1.0), size=0.2,
              rot=(0, 0, 0), spot=120, blend=0.6)


i = 0
for y in range(-14, 210, 17):
    if CROSS[0] - 2 < y < CROSS[1] + 2:
        continue
    for side in (-1, 1):
        street_lamp(side, y + (4 if side > 0 else -4), i); i += 1

# traffic light before the crossing
tx_, ty_ = ROAD_W + 0.5, CROSS[0] - 6.0
cyl("TrafficPole", 0.1, 6.2, (tx_, ty_, 0.18), m_metal, segs=8)
rod("TrafficArm", (tx_, ty_, 6.2), (tx_ - 4.6, ty_, 6.2), 0.07, m_metal, segs=6)
for k, ax in enumerate((-2.2, -4.4)):
    box(f"TrafficBox{k}", (0.42, 0.3, 1.2), (tx_ + ax, ty_, 5.5), m_dark)
    for j, key in enumerate(("red", "yellow", "green")):
        sphere(f"TrafficLamp{k}{j}", 0.13, (tx_ + ax, ty_ - 0.14, 5.9 - j * 0.38),
               neon(key, 45) if key == "red" else mat(f"s3_off_{key}", [c * 0.05 for c in NEON[key]], rough=0.3), ico=2)

# overhead cables + string lights
for k, y in enumerate((6, 15, 24, 33, 64, 78, 95)):
    z1, z2 = random.uniform(9, 16), random.uniform(9, 16)
    cable(f"Cable{k}", (-FRONT, y, z1), (FRONT, y + random.uniform(-2, 2), z2), random.uniform(1.0, 2.2),
          lights=(["orange", "red", "yellow"], ["cyan", "magenta"], None)[k % 3])
    cable(f"CableB{k}", (-FRONT, y + 1.5, z1 + 3), (FRONT, y + 3, z2 + 2.5), 1.4)

# vending machines, bins, bollards, hydrant
for k, (side, y, key) in enumerate([(1, 2.0, "cyan"), (1, 3.2, "red"), (-1, 9.0, "green"), (1, 19.5, "magenta"), (-1, 27.0, "cyan")]):
    x = side * (FRONT - 0.6)
    box(f"Vend{k}", (0.8, 1.0, 1.9), (x, y, 0.18), mat(f"s3_vend_{key}", [c * 0.25 for c in NEON[key]], rough=0.35), base=True, bevel=0.03)
    box(f"VendPanel{k}", (0.05, 0.84, 1.1), (x - side * 0.41, y, 1.4), m_shops[k % len(m_shops)])
    box(f"VendStrip{k}", (0.05, 0.84, 0.16), (x - side * 0.41, y, 0.62), neon(key, 14))
for k in range(12):
    side = 1 if k % 2 else -1
    cyl(f"Bollard{k}", 0.09, 0.8, (side * (ROAD_W + 0.45), -4 + k * 3.4, 0.18), m_metal, segs=8)
    box(f"BollardBand{k}", (0.2, 0.2, 0.06), (side * (ROAD_W + 0.45), -4 + k * 3.4, 0.8), neon("yellow", 5))
for k, (side, y) in enumerate([(-1, 4.5), (1, 11.0), (-1, 21.0), (1, 30.0)]):
    cyl(f"Bin{k}", 0.3, 0.9, (side * (ROAD_W + 1.2), y, 0.18), m_metal, segs=10, r2=0.34)
cyl("Hydrant", 0.13, 0.6, (ROAD_W + 0.9, 6.5, 0.18), mat("s3_hyd", (0.5, 0.05, 0.03), rough=0.4), segs=10)
sphere("HydrantTop", 0.14, (ROAD_W + 0.9, 6.5, 0.8), bpy.data.materials["s3_hyd"], ico=2)

# ---------------- cars ----------------
group("Cars")


def car(name, x, y, heading, color, taxi=False, lit=False):
    """heading: +1 drives +Y (we see tail lights), -1 drives -Y (head lights face camera)."""
    body = mat(f"s3_car_{name}", color, rough=0.22, metal=0.7, coat=1.0)
    box(name + "_body", (1.85, 4.4, 0.62), (x, y, 0.32), body, base=True, bevel=0.12)
    box(name + "_cabin", (1.6, 2.2, 0.55), (x, y - heading * 0.25, 0.92), m_glass, base=True, bevel=0.14)
    for dx in (-0.9, 0.9):
        for dy in (-1.35, 1.35):
            cyl(f"{name}_wheel{dx}{dy}", 0.33, 0.24, (x + dx - (0.12 if dx > 0 else -0.12), y + dy, 0.33), m_dark, segs=16,
                rot=(0, math.pi / 2, 0), base=False)
    fy_, by_ = y + heading * 2.21, y - heading * 2.21
    for dx in (-0.62, 0.62):
        box(f"{name}_head{dx}", (0.42, 0.05, 0.16), (x + dx, fy_, 0.68), neon("white", 60))
        box(f"{name}_tail{dx}", (0.5, 0.05, 0.13), (x + dx, by_, 0.72), neon("red", 40))
    box(f"{name}_tailbar", (1.5, 0.04, 0.04), (x, by_, 0.86), neon("red", 25))
    if taxi:
        box(name + "_sign", (0.7, 0.3, 0.2), (x, y, 1.5), neon("yellow", 20), base=True)
    if lit:
        light(name + "_beam", "SPOT", (x, fy_ + heading * 0.2, 0.7), 1200, (0.9, 0.95, 1.0), size=0.15,
              target=(x, fy_ + heading * 14, 0.0), spot=70, blend=0.5)


car("CarA", 2.4, 9.0, 1, (0.35, 0.02, 0.03))
car("CarB", 2.5, 21.5, 1, (0.9, 0.6, 0.05), taxi=True)
car("CarC", -2.4, 15.0, -1, (0.03, 0.05, 0.2), lit=True)
car("CarD", -2.5, 30.0, -1, (0.5, 0.5, 0.52), lit=True)
car("CarE", 2.4, 62.0, 1, (0.05, 0.05, 0.06))
car("CarF", -2.4, 70.0, -1, (0.9, 0.6, 0.05), taxi=True, lit=True)
car("CarG", 2.5, 84.0, 1, (0.1, 0.25, 0.3))
car("CarH", -2.5, 101.0, -1, (0.3, 0.3, 0.32), lit=True)
car("CarI", 2.4, 118.0, 1, (0.4, 0.05, 0.3))
car("CarJ", -2.4, 140.0, -1, (0.2, 0.2, 0.22))

# flying traffic
for k in range(9):
    x = random.uniform(-4.5, 4.5); y = 30 + k * 19 + random.uniform(-5, 5); z = random.uniform(24, 62)
    hd = 1 if k % 2 else -1
    box(f"Flyer{k}", (1.4, 3.6, 0.5), (x, y, z), m_metal, bevel=0.1)
    box(f"FlyerTail{k}", (1.2, 0.08, 0.14), (x, y - hd * 1.85, z), neon("red", 45))
    box(f"FlyerHead{k}", (1.2, 0.08, 0.14), (x, y + hd * 1.85, z), neon("white", 45))
    box(f"FlyerTrail{k}", (0.07, 11, 0.05), (x, y - hd * 7.5, z), neon("red" if hd > 0 else "cyan", 14))
    box(f"FlyerGlow{k}", (1.0, 2.6, 0.04), (x, y, z - 0.27), neon("cyan", 12))

# ---------------- people with umbrellas ----------------
group("People")
COATS = [mat(f"s3_coat{i}", c, rough=0.6) for i, c in enumerate([(0.02, 0.02, 0.025), (0.08, 0.03, 0.03), (0.03, 0.05, 0.09),
                                                                (0.12, 0.1, 0.07), (0.05, 0.07, 0.05)])]
UMB = [mat("s3_umbA", (0.02, 0.02, 0.02), rough=0.3), mat("s3_umbB", (0.5, 0.03, 0.04), rough=0.3),
       mat("s3_umbC", (0.75, 0.7, 0.1), rough=0.3), mat("s3_umbD", (0.7, 0.9, 1.0), rough=0.15, trans=0.8),
       mat("s3_umbE", (0.03, 0.2, 0.35), rough=0.3)]


def person(i, x, y, s=1.0, umbrella=True):
    z = 0.18
    coat = random.choice(COATS)
    for dx in (-0.09, 0.09):
        cyl(f"P{i}_leg{dx}", 0.065 * s, 0.82 * s, (x + dx * s, y + random.uniform(-0.08, 0.08), z), m_dark, segs=8)
    cyl(f"P{i}_torso", 0.21 * s, 0.72 * s, (x, y, z + 0.74 * s), coat, segs=10, r2=0.17 * s)
    sphere(f"P{i}_head", 0.115 * s, (x, y, z + 1.6 * s), m_skin, ico=2)
    sphere(f"P{i}_hair", 0.12 * s, (x, y + 0.015, z + 1.63 * s), m_dark, ico=2, scale=(1, 1, 0.8))
    if umbrella:
        ux = x + 0.16 * s
        rod(f"P{i}_urod", (ux, y, z + 1.05 * s), (ux, y, z + 2.12 * s), 0.012, m_metal, segs=5)
        lathe(f"P{i}_umb", [(0, 0.26), (0.3, 0.2), (0.55, 0.07), (0.64, 0.0)], (ux, y, z + 1.86 * s), random.choice(UMB), segs=8,
              smooth=False)


pi_ = 0
for side, ys in ((1, (4.8, 8.2, 8.9, 13.5, 17.0, 23.5, 26.0, 31.0, 36.5, 60, 66, 73)),
                 (-1, (3.0, 6.4, 11.0, 12.0, 18.5, 22.0, 29.5, 34.0, 37.5, 62, 70, 81))):
    for y in ys:
        person(pi_, side * (ROAD_W + random.uniform(0.9, 2.3)), y, random.uniform(0.92, 1.08), umbrella=random.random() < 0.8)
        pi_ += 1
for k in range(3):      # crossing the zebra
    person(pi_, -2.5 + k * 1.9, CROSS[0] - 3.5 + random.uniform(-0.8, 0.8), 1.0); pi_ += 1

# ---------------- rain + fog ----------------
group("Atmosphere")
bm = bmesh.new()
for _ in range(2600):
    x = random.uniform(-13, 13); y = random.uniform(-3, 42); z = random.uniform(0.2, 16)
    ln = random.uniform(0.25, 0.6); wd = 0.003 + 0.0004 * max(0, y)
    vs = [bm.verts.new(v) for v in ((x - wd, y, z), (x + wd, y, z), (x + wd + 0.04, y, z + ln), (x - wd + 0.04, y, z + ln))]
    bm.faces.new(vs)
_finish("Rain", bm, m_rain)

fog = box("Fog", (260, 400, 260), (0, 150, -1), None, base=True)
fm = bpy.data.materials.new("s3_fog")
try:
    fm.use_nodes = True
except Exception:
    pass
nt = fm.node_tree
for n in list(nt.nodes):
    nt.nodes.remove(n)
out = nt.nodes.new("ShaderNodeOutputMaterial"); vol = nt.nodes.new("ShaderNodeVolumePrincipled")
vol.inputs["Color"].default_value = (0.55, 0.65, 1.0, 1)
vol.inputs["Density"].default_value = 0.0013
vol.inputs["Anisotropy"].default_value = 0.35
nt.links.new(vol.outputs[0], out.inputs["Volume"])
fog.data.materials.append(fm)
fog.display_type = "WIRE"

# searchlights sweeping the fog
for k, (x, y, z, tx2, ty2) in enumerate([(-22, 120, 60, -60, 200), (26, 150, 70, 80, 260), (-16, 90, 45, 30, 150)]):
    light(f"Search{k}", "SPOT", (x, y, z), 4e6, (0.7, 0.85, 1.0), size=0.5, target=(tx2, ty2, z + 220), spot=5, blend=0.2)

# coloured kickers so the wet road picks up the neon
light("KickMagenta", "AREA", (-7.5, 10, 6), 700, (1.0, 0.1, 0.6), size=5, target=(0, 14, 0))
light("KickCyan", "AREA", (7.5, 24, 7), 700, (0.1, 0.7, 1.0), size=5, target=(0, 26, 0))

# ---------------- camera ----------------
group("Rig")
camera("Cam", (1.3, -10.0, 1.6), (-0.3, 60, 14.5), lens=22, fstop=4.0, focus=(2.4, 9.0, 0.8))
result = {"objects": len(sc.objects)}
