"""
Scene 2 v2 - low-poly floating island at sunset.

Self-contained: builds the whole scene from an empty file (no external helpers).

  Blender -b --factory-startup -P scene2_floating_island_v2.py -- --final
        1920x1080 Cycles render -> S2_FloatingIsland_v2.png, and saves scene2_floating_island_v2.blend
  ... -- --res 50 --samples 48 --out _preview.png      quick preview
  ... -- --norender                                    build only + frame report
  ... -- --crop 0.3,0.3,0.7,0.8 --res 100              render only part of the frame

Running it from the GUI text editor also works, but it clears the current file first.
"""
import bpy, bmesh, math, random, sys, os, time, zlib, argparse
from mathutils import Vector, Matrix, Euler, noise, geometry

T0 = time.time()
try:
    HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    HERE = os.getcwd()

_ap = argparse.ArgumentParser()
_ap.add_argument("--final", action="store_true")
_ap.add_argument("--res", type=int, default=50)
_ap.add_argument("--samples", type=int, default=48)
_ap.add_argument("--out", default="")
_ap.add_argument("--norender", action="store_true")
_ap.add_argument("--save", action="store_true")
_ap.add_argument("--crop", default="")
_ap.add_argument("--cam", default="")        # debug camera "x,y,z,tx,ty,tz,lens"
_ap.add_argument("--debuglight", action="store_true")   # flat grey light, to judge shapes and colours
_ap.add_argument("--exec", default="")       # debug: python run just before rendering
_ap.add_argument("--look", default="")       # also save the render with these extra view transforms/looks
A = _ap.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])

FINAL_SAMPLES = 640
EXPOSURE = 0.0
PI, TAU = math.pi, 2 * math.pi


# =====================================================================================
#  small maths / colour helpers
# =====================================================================================
def smooth(a, b, x):
    """smoothstep that also works reversed (a > b)."""
    if a == b:
        return 0.0
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def srgb(h):
    """'#rrggbb' -> scene-linear rgb tuple."""
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c)


def mixc(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def mulc(a, k):
    return (a[0] * k, a[1] * k, a[2] * k) + tuple(a[3:])


def jit(c, amt, rng, hue=0.0):
    k = 1 + rng.uniform(-amt, amt)
    r, g, b = c[0] * k, c[1] * k, c[2] * k
    if hue:
        t = rng.uniform(-hue, hue)
        r *= 1 + t
        b *= 1 - t
    return (r, g, b) + tuple(c[3:])


def angdiff(a, b):
    return (a - b + PI) % TAU - PI


def n3(x, y, z):
    return noise.noise(Vector((x, y, z)))


def catmull(P, sub=6):
    P = [Vector(p) for p in P]
    Q = [P[0] + (P[0] - P[1])] + P + [P[-1] + (P[-1] - P[-2])]
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        for k in range(sub):
            t = k / sub
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1].copy())
    return out


def polylen(P):
    return sum((P[i + 1] - P[i]).length for i in range(len(P) - 1))


def poly_dist(p, P):
    """(distance, normalised arclength) of point p to polyline P."""
    best_d, best_t, run, tot = 1e9, 0.0, 0.0, polylen(P)
    for i in range(len(P) - 1):
        a, b = P[i], P[i + 1]
        ab = b - a
        l2 = ab.length_squared
        t = max(0.0, min(1.0, (p - a).dot(ab) / l2)) if l2 > 1e-12 else 0.0
        d = (p - (a + ab * t)).length
        if d < best_d:
            best_d, best_t = d, (run + t * ab.length) / tot
        run += ab.length
    return best_d, best_t


def poly_at(P, t):
    """point and unit tangent at normalised arclength t."""
    tot = polylen(P)
    want, run = max(0.0, min(1.0, t)) * tot, 0.0
    for i in range(len(P) - 1):
        seg = (P[i + 1] - P[i]).length
        if run + seg >= want or i == len(P) - 2:
            k = (want - run) / seg if seg > 1e-9 else 0.0
            return P[i].lerp(P[i + 1], k), (P[i + 1] - P[i]).normalized()
        run += seg


def TR(loc=(0, 0, 0), rot=(0, 0, 0), scale=1.0):
    if isinstance(scale, (int, float)):
        S = Matrix.Scale(scale, 4)
    else:
        S = Matrix.Diagonal((scale[0], scale[1], scale[2], 1.0))
    return Matrix.Translation(Vector(loc)) @ Euler(rot, "XYZ").to_matrix().to_4x4() @ S


def align_z(d):
    """rotation taking +Z onto direction d."""
    return Vector((0, 0, 1)).rotation_difference(Vector(d).normalized()).to_matrix().to_4x4()


# =====================================================================================
#  camera maths first: a lot of the layout is done in frame space
# =====================================================================================
CAM_LOC = Vector((1.5, -27.0, 6.45))
CAM_TGT = Vector((0.2, 0.0, 0.12))
LENS = 32.0
if A.cam:
    _c = [float(x) for x in A.cam.split(",")]
    CAM_LOC, CAM_TGT, LENS = Vector(_c[0:3]), Vector(_c[3:6]), _c[6]
ASPECT = 16 / 9
_fw = (CAM_TGT - CAM_LOC).normalized()
_rt = _fw.cross(Vector((0, 0, 1))).normalized()
_up = _rt.cross(_fw).normalized()
_tw = 18.0 / LENS
_th = _tw / ASPECT


def cam_ray(u, v):
    return (_fw + _rt * ((2 * u - 1) * _tw) + _up * ((2 * v - 1) * _th)).normalized()


def F2W(u, v, dist):
    """frame coords (0..1, 0..1 from bottom-left) + distance from camera -> world point."""
    return CAM_LOC + cam_ray(u, v) * dist


def W2F(p):
    q = Vector(p) - CAM_LOC
    zc = q.dot(_fw)
    return 0.5 + q.dot(_rt) / (zc * _tw) / 2, 0.5 + q.dot(_up) / (zc * _th) / 2, zc


# =====================================================================================
#  scene reset / collections
# =====================================================================================
if bpy.app.background:
    bpy.ops.wm.read_factory_settings(use_empty=True)
else:
    for _ob in list(bpy.data.objects):
        bpy.data.objects.remove(_ob, do_unlink=True)
    for _c in list(bpy.data.collections):
        bpy.data.collections.remove(_c)
SC = bpy.context.scene
SC.name = "S2_FloatingIsland_v2"
CUR = [SC.collection]


def group(name):
    c = bpy.data.collections.new(name)
    SC.collection.children.link(c)
    CUR[0] = c
    return c


# =====================================================================================
#  node helpers + materials
# =====================================================================================
class NB:
    """tiny node-tree builder."""

    def __init__(self, tree):
        self.t = tree

    def n(self, idname, **kw):
        node = self.t.nodes.new(idname)
        for k, v in kw.items():
            setattr(node, k, v)
        return node

    def sin(self, node, key):
        if isinstance(key, int):
            return node.inputs[key]
        for s in node.inputs:
            if s.name == key and s.enabled:
                return s
        return node.inputs[key]

    def sout(self, node, key=None):
        for s in node.outputs:
            if s.enabled and (key is None or s.name == key):
                return s
        return node.outputs[0]

    def set(self, node, key, val):
        s = self.sin(node, key)
        if isinstance(val, bpy.types.NodeSocket):
            self.t.links.new(val, s)
            return
        try:
            s.default_value = val
        except Exception:
            s.default_value = tuple(val) + (1.0,)

    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.n("ShaderNodeMath", operation=op)
        n.use_clamp = clamp
        self.set(n, 0, a)
        if b is not None:
            self.set(n, 1, b)
        if c is not None:
            self.set(n, 2, c)
        return n.outputs[0]

    def vmath(self, op, a, b=None, out="Vector", scale=None):
        n = self.n("ShaderNodeVectorMath", operation=op)
        self.set(n, 0, a)
        if b is not None:
            self.set(n, 1, b)
        if scale is not None:
            self.set(n, "Scale", scale)
        return n.outputs[out]

    def mix(self, fac, a, b, blend="MIX"):
        n = self.n("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        self.set(n, "Factor", fac)
        self.set(n, "A", a)
        self.set(n, "B", b)
        return self.sout(n, "Result")

    def maprange(self, v, f0, f1, t0=0.0, t1=1.0, interp="LINEAR"):
        n = self.n("ShaderNodeMapRange", interpolation_type=interp)
        n.clamp = True
        for i, x in enumerate((v, f0, f1, t0, t1)):
            self.set(n, i, x)
        return n.outputs[0]

    def ramp(self, fac, stops):
        n = self.n("ShaderNodeValToRGB")
        el = n.color_ramp.elements
        el[0].position, el[0].color = stops[0][0], tuple(stops[0][1]) + (1,)
        el[1].position, el[1].color = stops[-1][0], tuple(stops[-1][1]) + (1,)
        for pos, c in stops[1:-1]:
            e = el.new(pos)
            e.color = tuple(c) + (1,)
        self.set(n, 0, fac)
        return n.outputs["Color"]


def new_mat(name):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    return m, nt, NB(nt)


def mat_vcol(name, rough=0.9, spec=0.25, translucent=0.0, emit=0.0, alpha=False, fog=False, tint=None):
    """Matte material that reads the per-face colour attribute 'Col'.
    translucent: share of light let through (back-lit foliage / cloth / clouds)
    alpha:       use the attribute alpha as opacity
    fog:         blend to the object colour by the object alpha (distance haze for far things)"""
    m, nt, b = new_mat(name)
    out = b.n("ShaderNodeOutputMaterial")
    p = b.n("ShaderNodeBsdfPrincipled")
    at = b.n("ShaderNodeAttribute", attribute_name="Col")
    col = at.outputs["Color"]
    b.set(p, "Base Color", col)
    b.set(p, "Roughness", rough)
    b.set(p, "Specular IOR Level", spec)
    if emit:
        b.set(p, "Emission Color", col)
        b.set(p, "Emission Strength", emit)
    sh = p.outputs[0]
    if translucent:
        tr = b.n("ShaderNodeBsdfTranslucent")
        b.set(tr, "Color", b.mix(1.0, col, tint, "MULTIPLY") if tint else col)
        mx = b.n("ShaderNodeMixShader")
        b.set(mx, 0, translucent)
        nt.links.new(sh, mx.inputs[1])
        nt.links.new(tr.outputs[0], mx.inputs[2])
        sh = mx.outputs[0]
    if fog:
        oi = b.n("ShaderNodeObjectInfo")
        em = b.n("ShaderNodeEmission")
        b.set(em, "Color", oi.outputs["Color"])
        mx = b.n("ShaderNodeMixShader")
        nt.links.new(oi.outputs["Alpha"], mx.inputs[0])
        nt.links.new(sh, mx.inputs[1])
        nt.links.new(em.outputs[0], mx.inputs[2])
        sh = mx.outputs[0]
    if alpha:
        tp = b.n("ShaderNodeBsdfTransparent")
        mx = b.n("ShaderNodeMixShader")
        nt.links.new(at.outputs["Alpha"], mx.inputs[0])
        nt.links.new(tp.outputs[0], mx.inputs[1])
        nt.links.new(sh, mx.inputs[2])
        sh = mx.outputs[0]
    nt.links.new(sh, out.inputs["Surface"])
    return m


def mat_water(name, rough=0.06):
    m, nt, b = new_mat(name)
    out = b.n("ShaderNodeOutputMaterial")
    p = b.n("ShaderNodeBsdfPrincipled")
    at = b.n("ShaderNodeAttribute", attribute_name="Col")
    b.set(p, "Base Color", at.outputs["Color"])
    b.set(p, "Roughness", rough)
    b.set(p, "IOR", 1.33)
    b.set(p, "Specular IOR Level", 1.0)
    nt.links.new(p.outputs[0], out.inputs["Surface"])
    return m


M_VCOL = mat_vcol("s2_matte")
M_ROCK = mat_vcol("s2_rock", rough=0.95, spec=0.15)
M_FOLIAGE = mat_vcol("s2_foliage", rough=0.85, translucent=0.30)
M_FABRIC = mat_vcol("s2_fabric", rough=0.8, translucent=0.40)
M_SAIL = mat_vcol("s2_sail", rough=0.8, translucent=0.80)
M_FAR = mat_vcol("s2_far", rough=1.0, spec=0.0, fog=True)
M_SMOKE = mat_vcol("s2_smoke", rough=1.0, spec=0.0, translucent=0.35, alpha=True)
M_FALL = mat_vcol("s2_fall", rough=0.25, spec=0.6, emit=0.32, alpha=True)
M_POND = mat_water("s2_pond", 0.06)
M_STREAM = mat_water("s2_stream", 0.18)


def mat_glow(name):
    """emission straight from the per-face colour (HDR values allowed): windows, lanterns, flames, crystals."""
    m, nt, b = new_mat(name)
    out = b.n("ShaderNodeOutputMaterial")
    p = b.n("ShaderNodeBsdfPrincipled")
    at = b.n("ShaderNodeAttribute", attribute_name="Col")
    b.set(p, "Base Color", (0.02, 0.02, 0.02, 1.0))
    b.set(p, "Roughness", 0.5)
    b.set(p, "Emission Color", at.outputs["Color"])
    b.set(p, "Emission Strength", 1.0)
    nt.links.new(p.outputs[0], out.inputs["Surface"])
    return m


M_GLOW = mat_glow("s2_glow")


def mat_cloud(name, sun, fog=False, gain=1.0):
    """Clouds: the facet colour comes from how much the facet faces the sun (mauve shade -> pink -> gold),
    so they keep a warm rim and a cool body whatever the fill does. A little real diffuse sits on top."""
    m, nt, b = new_mat(name)
    out = b.n("ShaderNodeOutputMaterial")
    at = b.n("ShaderNodeAttribute", attribute_name="Col")
    geo = b.n("ShaderNodeNewGeometry")
    k = b.vmath("DOT_PRODUCT", geo.outputs["True Normal"], tuple(sun), out="Value")
    rp = b.ramp(b.math("MULTIPLY_ADD", k, 0.5, 0.5),
                [(0.0, srgb("#474488")), (0.30, srgb("#625c9e")), (0.48, srgb("#9a78ac")),
                 (0.60, srgb("#e0909a")), (0.72, srgb("#ffab78")), (0.86, (1.25, 0.72, 0.30)), (1.0, (1.9, 1.15, 0.45))])
    col = b.mix(1.0, rp, at.outputs["Color"], "MULTIPLY")
    em = b.n("ShaderNodeEmission")
    b.set(em, "Color", col)
    b.set(em, "Strength", gain)
    df = b.n("ShaderNodeBsdfDiffuse")
    b.set(df, "Color", b.mix(1.0, at.outputs["Color"], (0.3, 0.3, 0.3), "MULTIPLY"))
    ad = b.n("ShaderNodeAddShader")
    nt.links.new(em.outputs[0], ad.inputs[0])
    nt.links.new(df.outputs[0], ad.inputs[1])
    sh = ad.outputs[0]
    if fog:
        oi = b.n("ShaderNodeObjectInfo")
        fe = b.n("ShaderNodeEmission")
        b.set(fe, "Color", oi.outputs["Color"])
        mx = b.n("ShaderNodeMixShader")
        nt.links.new(oi.outputs["Alpha"], mx.inputs[0])
        nt.links.new(sh, mx.inputs[1])
        nt.links.new(fe.outputs[0], mx.inputs[2])
        sh = mx.outputs[0]
    nt.links.new(sh, out.inputs["Surface"])
    return m


# =====================================================================================
#  mesh builder: flat-shaded, per-face coloured geometry baked into one object
# =====================================================================================
class MB:
    def __init__(self, name):
        self.name = name
        self.v, self.f, self.c, self.mi = [], [], [], []
        self.rng = random.Random(zlib.crc32(name.encode()))

    def add(self, shape, col, M=None, mi=0, vary=0.0, hue=0.0, ctr=None):
        """shape = (verts, faces). col = rgb(a) | list (one per face) | callable(center, outward_normal)."""
        verts, faces = shape
        verts = [M @ Vector(v) for v in verts] if M is not None else [Vector(v) for v in verts]
        base = len(self.v)
        self.v.extend(verts)
        fn = callable(col)
        if fn and ctr is None:
            ctr = Vector((0, 0, 0))
            for v in verts:
                ctr += v
            ctr /= max(1, len(verts))
        per = isinstance(col, list)
        for fi, f in enumerate(faces):
            if fn:
                fc = Vector((0, 0, 0))
                for i in f:
                    fc += verts[i]
                fc /= len(f)
                n = (verts[f[1]] - verts[f[0]]).cross(verts[f[2]] - verts[f[0]])
                if n.length > 1e-12:
                    n.normalize()
                if n.dot(fc - ctr) < 0:
                    n = -n
                c = col(fc, n)
            else:
                c = col[fi] if per else col
            if vary or hue:
                c = jit(c, vary, self.rng, hue)
            self.c.append(c)
            self.f.append(tuple(base + i for i in f))
            self.mi.append(mi)
        return self

    def build(self, mats=None, coll=None, weld=True):
        me = bpy.data.meshes.new(self.name)
        me.from_pydata([tuple(v) for v in self.v], [], self.f)
        ca = me.color_attributes.new("Col", "FLOAT_COLOR", "CORNER")
        flat = []
        for f, c in zip(self.f, self.c):
            flat.extend((c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0) * len(f))
        ca.data.foreach_set("color", flat)
        me.polygons.foreach_set("material_index", self.mi)
        mats = mats or [M_VCOL]
        for m in (mats if isinstance(mats, (list, tuple)) else [mats]):
            me.materials.append(m)
        if weld:
            bm = bmesh.new()
            bm.from_mesh(me)
            bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(me)
            bm.free()
        me.update()
        ob = bpy.data.objects.new(self.name, me)
        (coll or CUR[0]).objects.link(ob)
        return ob


# ---- primitive shapes: each returns (verts, faces) ------------------------------------
def P_box(sx, sy, sz, base=False):
    x, y = sx / 2, sy / 2
    z0, z1 = (0.0, sz) if base else (-sz / 2, sz / 2)
    v = [(-x, -y, z0), (x, -y, z0), (x, y, z0), (-x, y, z0), (-x, -y, z1), (x, -y, z1), (x, y, z1), (-x, y, z1)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return v, f


def P_frustum(r1, r2, h, n, z0=0.0, phase=0.0, cap_bottom=True, cap_top=True):
    v, f = [], []
    for i in range(n):
        a = phase + TAU * i / n
        v.append((r1 * math.cos(a), r1 * math.sin(a), z0))
    if r2 > 1e-6:
        for i in range(n):
            a = phase + TAU * i / n
            v.append((r2 * math.cos(a), r2 * math.sin(a), z0 + h))
        for i in range(n):
            j = (i + 1) % n
            f.append((i, j, n + j, n + i))
        if cap_top:
            f.append(tuple(range(n, 2 * n)))
    else:
        v.append((0, 0, z0 + h))
        for i in range(n):
            f.append((i, (i + 1) % n, n))
    if cap_bottom:
        f.append(tuple(reversed(range(n))))
    return v, f


_ICO = {}


def P_ico(r=1.0, sub=1):
    if sub not in _ICO:
        bm = bmesh.new()
        bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=1.0)
        bm.verts.ensure_lookup_table()
        _ICO[sub] = ([v.co.copy() for v in bm.verts], [tuple(v.index for v in f.verts) for f in bm.faces])
        bm.free()
    v, f = _ICO[sub]
    return [c * r for c in v], list(f)


def P_blob(r, sub=1, seed=0, rough=0.2, scale=(1, 1, 1), freq=1.6, flat=None):
    """noisy icosphere; flat = squash everything below z=0 by this factor (flat cloud / bush base)."""
    v, f = P_ico(1.0, sub)
    o = Vector((seed * 1.37 + 0.5, seed * 2.11 - 3.0, seed * 0.73))
    out = []
    for p in v:
        q = p * (r * (1 + rough * noise.noise(p * freq + o)))
        q = Vector((q.x * scale[0], q.y * scale[1], q.z * scale[2]))
        if flat is not None and q.z < 0:
            q.z *= flat
        out.append(q)
    return out, f


def P_tube(pts, radii, n=5, cap=True):
    """low-poly tube through pts with per-point radii."""
    v, f, m, xprev = [], [], len(pts), None
    for k, p in enumerate(pts):
        p = Vector(p)
        d = (Vector(pts[min(k + 1, m - 1)]) - Vector(pts[max(k - 1, 0)])).normalized()
        ref = Vector((0, 0, 1)) if abs(d.z) < 0.9 else Vector((1, 0, 0))
        x = d.cross(ref).normalized() if xprev is None else (xprev - d * xprev.dot(d))
        x = x.normalized() if x.length > 1e-6 else d.cross(ref).normalized()
        y = d.cross(x).normalized()
        xprev = x
        r = radii[k] if hasattr(radii, "__len__") else radii
        for i in range(n):
            a = TAU * i / n
            v.append(p + (x * math.cos(a) + y * math.sin(a)) * max(r, 0.003))
    for k in range(m - 1):
        for i in range(n):
            j = (i + 1) % n
            f.append((k * n + i, k * n + j, (k + 1) * n + j, (k + 1) * n + i))
    if cap:
        f.append(tuple(reversed(range(n))))
        f.append(tuple(range((m - 1) * n, m * n)))
    return v, f


def P_rod(p1, p2, r, n=5, r2=None):
    return P_tube([p1, p2], [r, r if r2 is None else r2], n)


def zipper(A, angA, B, angB):
    """triangles joining two closed rings with different vertex counts (angles increase with index)."""
    n, m = len(A), len(B)
    a0 = angA[0]
    bA = [(a - a0) % TAU for a in angA]
    bA[0] = 0.0
    bB = [(a - a0) % TAU for a in angB]
    j0 = min(range(m), key=lambda j: bB[j])
    B = B[j0:] + B[:j0]
    bB = bB[j0:] + bB[:j0]
    i = j = 0
    tris = []
    while i < n or j < m:
        na = bA[i + 1] if i + 1 < n else TAU
        nb = bB[j + 1] if j + 1 < m else bB[0] + TAU
        if j >= m or (i < n and na <= nb):
            tris.append((A[i % n], A[(i + 1) % n], B[j % m]))
            i += 1
        else:
            tris.append((A[i % n], B[(j + 1) % m], B[j % m]))
            j += 1
    return tris


# =====================================================================================
#  palette (scene-linear)
# =====================================================================================
C = {
    "g_mid": srgb("#6aa23a"), "g_light": srgb("#93bb47"), "g_dark": srgb("#3f7a35"), "g_dry": srgb("#aeb04f"),
    "turf": srgb("#4f8f2f"), "turf_d": srgb("#2f5a26"),
    "dirt": srgb("#a47c50"), "dirt_l": srgb("#bd9864"), "sand": srgb("#c9b383"), "soil": srgb("#4a3526"),
    "soil_l": srgb("#7a5a3c"), "tilled": srgb("#5a3d2a"),
    "bed": srgb("#3d5a55"), "bed_deep": srgb("#1f3a3e"),
    "stone": srgb("#8f8a86"), "stone_l": srgb("#d2cabd"), "stone_d": srgb("#8c8684"),
    "w_deep": srgb("#1c5f73"), "w_shal": srgb("#3f9aa6"), "w_light": srgb("#8fd3e8"), "w_white": srgb("#f1fbff"),
    "w_mid": srgb("#5db7dd"),
    "cloud_t": (1.0, 0.97, 0.93), "cloud_m": (0.93, 0.90, 0.92), "cloud_b": (0.72, 0.70, 0.82),
}
ROCK = [srgb(h) for h in ("#5a3f2c", "#c08a55", "#a5573a", "#dcb98c", "#8f6447", "#8a7868", "#75677d", "#b86a40",
                           "#8a7c86", "#786a7c", "#6b5c72")]

# =====================================================================================
#  layout of the island top
# =====================================================================================
RX, RY = 6.9, 5.8                                   # island half-extents (x = across frame, y = into frame)
SUN_AZ, SUN_EL = math.radians(62), math.radians(24)  # key light: low, from the right-back
SUN_VEC = Vector((math.sin(SUN_AZ) * math.cos(SUN_EL), math.cos(SUN_AZ) * math.cos(SUN_EL), math.sin(SUN_EL)))
SUN_H = Vector((math.sin(SUN_AZ), math.cos(SUN_AZ)))

M_CLOUD = mat_cloud("s2_cloud", SUN_VEC)
M_CLOUD_FAR = mat_cloud("s2_cloud_far", SUN_VEC, fog=True)

MILL = Vector((-3.4, 2.3))
HILL_H, HILL_R0, HILL_R1 = 1.0, 1.35, 3.6
CAB = Vector((1.0, 2.1))
CAB_ROT = math.radians(-27)                         # cabin front (local -Y) looks at the camera, a bit left
CAB_Z = 0.68
POND = Vector((4.0, -1.9))
POND_RX, POND_RY = 1.5, 1.12
WATER, POND_BED = 0.20, -0.22
STREAM_A = math.radians(-54)
PATH_W = 0.21
_cc, _cs = math.cos(CAB_ROT), math.sin(CAB_ROT)


def cab_w(lx, ly, lz=0.0):
    """cabin-local -> world."""
    return Vector((CAB.x + lx * _cc - ly * _cs, CAB.y + lx * _cs + ly * _cc, CAB_Z + lz))


def cab_l(p):
    d = Vector((p.x, p.y)) - CAB
    return Vector((d.x * _cc + d.y * _cs, -d.x * _cs + d.y * _cc))


def rect_dist(l, cx, cy, hx, hy):
    dx, dy = max(abs(l.x - cx) - hx, 0.0), max(abs(l.y - cy) - hy, 0.0)
    return math.hypot(dx, dy)


def cabin_dist(p):
    return rect_dist(cab_l(p), 0.0, -0.55, 1.7, 2.45)


GARDEN = (3.2, 0.7, 0.85, 1.0)                      # cabin-local centre x,y + half sizes


def garden_dist(p):
    return rect_dist(cab_l(p), *GARDEN)


def ell_r(a, rx=RX, ry=RY):
    return rx * ry / math.hypot(ry * math.cos(a), rx * math.sin(a))


def redge(a):
    r = ell_r(a) * (1 + 0.085 * n3(math.cos(a) * 1.25 + 5.2, math.sin(a) * 1.25 + 1.7, 0.5)
                    + 0.03 * n3(math.cos(a) * 3.6, math.sin(a) * 3.6, 4.0))
    r -= 0.22 * math.exp(-(angdiff(a, STREAM_A) / 0.07) ** 2)          # notch where the stream leaves
    r += 0.40 * math.exp(-(angdiff(a, math.radians(218)) / 0.22) ** 2)   # promontory: bench + autumn tree
    return r


def inside(p, margin=0.0):
    return p.length < redge(math.atan2(p.y, p.x)) - margin


def pond_r(a):
    return ell_r(a, POND_RX, POND_RY) * (1 + 0.10 * n3(math.cos(a) * 1.7 + 2.0, math.sin(a) * 1.7, 8.8))


def pond_q(p):
    d = Vector((p.x, p.y)) - POND
    return d.length / pond_r(math.atan2(d.y, d.x)) if d.length > 1e-6 else 0.0


def pond_pt(a, q=1.0):
    return POND + Vector((math.cos(a), math.sin(a))) * pond_r(a) * q


_E = Vector((math.cos(STREAM_A), math.sin(STREAM_A))) * redge(STREAM_A)
FALL_DIR = Vector((0.49, -0.872)).normalized()
STREAM_PTS = catmull([pond_pt(math.radians(-118), 0.86), Vector((3.0, -3.5)), Vector((3.12, -4.15)), _E], 7)
STEPS_END = cab_w(-0.35, -3.15).to_2d()
FORK = Vector((-1.2, -1.35))
LANDING = Vector((-2.82, 0.70))
PATHS = [catmull([STEPS_END, FORK, Vector((-2.0, -0.95)), Vector((-2.65, -0.45)), Vector((-2.8, 0.1)), LANDING], 6),
         catmull([FORK, Vector((-2.3, -2.2)), Vector((-3.3, -2.9))], 6)]


def stream_z(t):
    return WATER - 0.03 - 0.33 * t


def path_dist(p):
    return min(poly_dist(p, P)[0] for P in PATHS)


def height(x, y):
    p = Vector((x, y))
    h = 0.45 + 0.085 * y + 0.15 * n3(x * 0.33, y * 0.33, 3.1) + 0.045 * n3(x * 0.9, y * 0.9, 7.0)
    h += HILL_H * smooth(HILL_R1, HILL_R0, (p - MILL).length)
    w = smooth(1.3, 0.25, cabin_dist(p))
    h = h * (1 - w) + CAB_Z * w
    w = smooth(0.7, 0.1, garden_dist(p))
    h = h * (1 - w) + (CAB_Z - 0.06) * w
    q = pond_q(p)
    if q < 1.45:
        lip = WATER + 0.07
        if q >= 1.0:
            w = smooth(1.45, 1.0, q)
            h = h * (1 - w) + lip * w
        else:
            h = POND_BED + (lip - POND_BED) * smooth(0.55, 1.0, q)
    d, t = poly_dist(p, STREAM_PTS)
    if d < 0.62:
        zs = stream_z(t)
        w = smooth(0.62, 0.30, d)
        h = min(h, h * (1 - w) + min(h, zs + 0.10) * w)
        w = smooth(0.32, 0.16, d)
        h = min(h, h * (1 - w) + (zs - 0.14) * w)
    return h


MILL_Z = height(MILL.x, MILL.y)


# =====================================================================================
#  sky: an art-directed dusk gradient, warm towards the sun, with a sun disc and glow
# =====================================================================================
SUN_UV = (0.835, 0.715)
SUNDRAW = cam_ray(*SUN_UV)
_sun_el = math.degrees(math.asin(SUNDRAW.z))
BAND_EL = _sun_el - 1.0
_hs = Vector((SUNDRAW.x, SUNDRAW.y)).normalized()
_E0 = BAND_EL
SKY_WARM = [(-90, srgb("#090b24")), (_E0 - 22, srgb("#1b2158")), (_E0 - 12, srgb("#3f3780")),
            (_E0 - 5.5, srgb("#a8506e")), (_E0 - 2.2, srgb("#ee7a45")), (_E0, (1.25, 0.44, 0.07)),
            (_E0 + 2.4, srgb("#e0685c")), (_E0 + 5.0, srgb("#84488e")), (_E0 + 8.5, srgb("#3f3a8a")),
            (_E0 + 16, srgb("#272c6e")), (_E0 + 40, srgb("#131b4a")), (90, srgb("#0a0e30"))]
SKY_COOL = [(-90, srgb("#090b24")), (_E0 - 22, srgb("#191f52")), (_E0 - 12, srgb("#2e3070")),
            (_E0 - 5.5, srgb("#544286")), (_E0 - 2.2, srgb("#84538e")), (_E0, srgb("#a8608c")),
            (_E0 + 2.4, srgb("#7c5090")), (_E0 + 5.0, srgb("#4c4292")), (_E0 + 8.5, srgb("#35388a")),
            (_E0 + 16, srgb("#252b6a")), (_E0 + 40, srgb("#131b48")), (90, srgb("#0a0e30"))]
# extra ambient that only lights the scene (the camera never sees it): blue from above, dusky mauve from below
AMB = [(-90, (0.05, 0.035, 0.07)), (-10, (0.07, 0.06, 0.13)), (10, (0.08, 0.10, 0.24)), (90, (0.07, 0.11, 0.30))]
SKY_TOE = 0.85


def _untoe(c, k=1.0):
    """Khronos PBR Neutral subtracts a small offset from every channel, which turns dark colours
    very saturated. Adding (part of) that offset back keeps the dark sky tones as designed."""
    m = min(c)
    x = 0.4 * math.sqrt(m) if m < 0.04 else m + 0.04
    return tuple(v + k * (x - m) for v in c)


SKY_WARM = [(e, _untoe(c, SKY_TOE)) for e, c in SKY_WARM]
SKY_COOL = [(e, _untoe(c, SKY_TOE)) for e, c in SKY_COOL]
AZ_C0, AZ_P = 0.30, 2.2
ANTI_COL = (0.03, 0.045, 0.16)                 # blue dusk opposite the sun: this is the cool fill
GLOW = [(1500.0, (2.5, 1.2, 0.30)), (120.0, (0.90, 0.36, 0.05)), (14.0, (0.18, 0.05, 0.02))]
DISC_EDGE, DISC_CORE, DISC_R = (7.0, 2.8, 0.6), (16.0, 10.0, 4.0), math.radians(1.35)
FILL_K = 1.3                                 # strength of the sky as a light source (camera sees 1.0)


def _el2pos(e):
    return (e + 90.0) / 180.0


def _ramp_eval(stops, e):
    if e <= stops[0][0]:
        return stops[0][1]
    for (e0, c0), (e1, c1) in zip(stops, stops[1:]):
        if e <= e1:
            return mixc(c0, c1, (e - e0) / (e1 - e0))
    return stops[-1][1]


def sky_rgb(d, camera=False):
    """python mirror of the world shader (used to pick haze colours for far objects)."""
    d = Vector(d).normalized()
    e = math.degrees(math.asin(max(-1, min(1, d.z))))
    h = Vector((d.x, d.y))
    ca = h.normalized().dot(_hs) if h.length > 1e-6 else 0.0
    w = max(0.0, min(1.0, (ca - AZ_C0) / (1 - AZ_C0))) ** AZ_P
    col = mixc(_ramp_eval(SKY_COOL, e), _ramp_eval(SKY_WARM, e), w)
    col = mixc(col, ANTI_COL, smooth(0.35, -0.45, ca))
    cg = max(0.0, d.dot(SUNDRAW))
    for i, (n, c) in enumerate(GLOW):
        if i == 0 and not camera:
            continue
        k = cg ** n
        col = (col[0] + c[0] * k, col[1] + c[1] * k, col[2] + c[2] * k)
    return col


def sky_uv(u, v):
    return sky_rgb(cam_ray(u, v))


def build_world():
    w = bpy.data.worlds.new("S2_Sky")
    SC.world = w
    nt = w.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    b = NB(nt)
    out = b.n("ShaderNodeOutputWorld")
    bg = b.n("ShaderNodeBackground")
    tc = b.n("ShaderNodeTexCoord")
    lp = b.n("ShaderNodeLightPath")
    cam = lp.outputs["Is Camera Ray"]
    D = b.vmath("NORMALIZE", tc.outputs["Generated"])
    sep = b.n("ShaderNodeSeparateXYZ")
    nt.links.new(D, sep.inputs[0])
    z = b.math("MINIMUM", b.math("MAXIMUM", sep.outputs["Z"], -1.0), 1.0)
    pos = b.math("MULTIPLY_ADD", b.math("ARCSINE", z), 57.29578 / 180.0, 0.5)
    warm = b.ramp(pos, [(_el2pos(e), c) for e, c in SKY_WARM])
    cool = b.ramp(pos, [(_el2pos(e), c) for e, c in SKY_COOL])
    cxy = b.n("ShaderNodeCombineXYZ")
    nt.links.new(sep.outputs["X"], cxy.inputs["X"])
    nt.links.new(sep.outputs["Y"], cxy.inputs["Y"])
    Dh = b.vmath("NORMALIZE", cxy.outputs[0])
    ca = b.vmath("DOT_PRODUCT", Dh, (_hs.x, _hs.y, 0.0), out="Value")
    wz = b.math("POWER", b.maprange(ca, AZ_C0, 1.0), AZ_P)
    col = b.mix(wz, cool, warm)
    col = b.mix(b.maprange(ca, 0.35, -0.45, 0.0, 1.0, "SMOOTHSTEP"), col, ANTI_COL)
    sun = tuple(SUNDRAW)
    cg = b.math("MAXIMUM", b.vmath("DOT_PRODUCT", D, sun, out="Value"), 0.0)
    for i, (n_, c_) in enumerate(GLOW):
        k = b.math("POWER", cg, n_)
        if i == 0:
            k = b.math("MULTIPLY", k, cam)
        col = b.vmath("ADD", col, b.vmath("SCALE", c_, scale=k))
    s = b.vmath("LENGTH", b.vmath("CROSS_PRODUCT", D, sun), out="Value")
    disc = b.maprange(s, math.sin(DISC_R * 1.08), math.sin(DISC_R * 0.94), 0.0, 1.0, "SMOOTHSTEP")
    disc = b.math("MULTIPLY", b.math("MULTIPLY", disc, b.math("GREATER_THAN", cg, 0.0)), cam)
    core = b.math("MULTIPLY", disc, b.maprange(s, math.sin(DISC_R * 0.95), 0.0, 0.0, 1.0, "SMOOTHSTEP"))
    col = b.vmath("ADD", col, b.vmath("SCALE", DISC_EDGE, scale=disc))
    col = b.vmath("ADD", col, b.vmath("SCALE", tuple(DISC_CORE[i] - DISC_EDGE[i] for i in range(3)), scale=core))
    seen = b.math("MAXIMUM", cam, lp.outputs["Is Glossy Ray"])          # camera and mirror rays see the painted sky
    lit = b.vmath("ADD", b.vmath("SCALE", col, scale=FILL_K), b.ramp(pos, [(_el2pos(e), c) for e, c in AMB]))
    b.set(bg, "Color", b.mix(seen, lit, col))
    b.set(bg, "Strength", 1.0)
    nt.links.new(bg.outputs[0], out.inputs["Surface"])
    return w


# =====================================================================================
#  island: terrain top (Delaunay over hand-placed feature points) + stratified rock body
# =====================================================================================
def ground_col(c, n, rng):
    x, y = c.x, c.y
    p = Vector((x, y))
    q = pond_q(p)
    ds, ts = poly_dist(p, STREAM_PTS)
    if q < 0.97:
        return jit(mixc(C["bed_deep"], C["bed"], smooth(0.3, 0.97, q)), 0.05, rng)
    if ds < 0.25:
        return jit(C["bed"], 0.06, rng)
    k = 0.5 + 0.5 * n3(x * 0.45 + 3, y * 0.45 - 2, 1.7)
    k2 = 0.5 + 0.5 * n3(x * 1.3, y * 1.3, 9.1)
    g = mixc(C["g_dark"], C["g_mid"], smooth(0.22, 0.50, k))
    g = mixc(g, C["g_light"], smooth(0.52, 0.80, k))
    g = mixc(g, C["g_dry"], 0.45 * smooth(0.55, 0.9, k2))
    slope = 1 - abs(n.z)
    if slope > 0.10:
        g = mixc(g, C["soil_l"], 0.75 * smooth(0.10, 0.30, slope))
    if q < 1.32:
        g = mixc(C["sand"], g, smooth(1.03, 1.32, q))
    if ds < 0.58:
        g = mixc(C["sand"], g, smooth(0.28, 0.58, ds))
    dp = path_dist(p)
    if dp < PATH_W * 0.97:
        g = mixc(C["dirt"], C["dirt_l"], k2)
    elif dp < PATH_W + 0.14:
        g = mixc(C["dirt"], g, 0.62)
    if garden_dist(p) <= 0.0:
        g = C["tilled"]
    return jit(g, 0.075, rng, 0.03)


def build_terrain():
    rng = random.Random(21)
    sa = STREAM_A % TAU
    angs = [a for a in (TAU * i / 84 for i in range(84)) if abs(angdiff(a, sa)) > 0.125]
    angs += [(sa + k) % TAU for k in (-0.105, -0.066, -0.043, -0.015, 0.015, 0.043, 0.066, 0.105)]
    angs.sort()
    rim2 = [Vector((redge(a) * math.cos(a), redge(a) * math.sin(a))) for a in angs]
    pts, exact, grid = list(rim2), [True] * len(rim2), {}

    def key(p):
        return int(math.floor(p.x / 0.25)), int(math.floor(p.y / 0.25))

    def ok(p, dmin):
        kx, ky = key(p)
        for i in range(kx - 2, kx + 3):
            for j in range(ky - 2, ky + 3):
                for q in grid.get((i, j), ()):
                    if (p - q).length < dmin:
                        return False
        return True

    def put(p, dmin, ex=True):
        if not ok(p, dmin):
            return False
        grid.setdefault(key(p), []).append(p)
        pts.append(p)
        exact.append(ex)
        return True

    for p in rim2:
        grid.setdefault(key(p), []).append(p)
    for qv, n, ph in ((1.0, 30, 0.0), (1.2, 30, 0.5), (0.74, 18, 0.2), (0.42, 9, 0.0)):
        for i in range(n):
            put(pond_pt(TAU * (i + ph) / n, qv), 0.1)
    put(POND.copy(), 0.1)
    L = polylen(STREAM_PTS)
    for i in range(1, int(L / 0.27) + 1):
        c, d = poly_at(STREAM_PTS, i * 0.27 / L)
        nrm = Vector((-d.y, d.x))
        for off in (0.0, 0.27, -0.27, 0.52, -0.52):
            p = c + nrm * off
            if inside(p, 0.2) and pond_q(p) > 1.03:
                put(p, 0.13)
    for P in PATHS:
        L = polylen(P)
        for i in range(int(L / 0.3) + 1):
            c, d = poly_at(P, min(1.0, i * 0.3 / L))
            nrm = Vector((-d.y, d.x))
            for off in (0.0, PATH_W, -PATH_W):
                p = c + nrm * (off + rng.uniform(-0.02, 0.02))
                if inside(p, 0.2):
                    put(p, 0.12)
    s, j, y = 0.46, 0, -RY - 0.8
    while y < RY + 0.8:
        x = -RX - 0.9 + (j % 2) * s / 2
        while x < RX + 0.9:
            p = Vector((x + rng.uniform(-0.13, 0.13), y + rng.uniform(-0.13, 0.13)))
            if inside(p, 0.26) and pond_q(p) > 1.3 and poly_dist(p, STREAM_PTS)[0] > 0.66 \
                    and path_dist(p) > PATH_W + 0.14:
                put(p, 0.24, False)
            x += s
        y += s * 0.866
        j += 1
    res = geometry.delaunay_2d_cdt([Vector((p.x, p.y)) for p in pts], [], [list(range(len(rim2)))], 1, 1e-5)
    ov, of, orig = res[0], res[2], res[3]
    rim_out = [None] * len(rim2)
    ex_out = [False] * len(ov)
    for j, src in enumerate(orig):
        for o in src:
            if o < len(rim2):
                rim_out[o] = j
            ex_out[j] = ex_out[j] or exact[o]
    V = [Vector((p.x, p.y, height(p.x, p.y) + (0.0 if ex_out[j] else rng.uniform(-0.03, 0.03))))
         for j, p in enumerate(ov)]
    mb = MB("Island_Top")
    crng = random.Random(8)

    def col(c, n):
        return ground_col(c, n if n.z > 0 else -n, crng)

    mb.add((V, [tuple(f) for f in of]), col)
    ob = mb.build([M_VCOL])
    rim = [(V[rim_out[i]].copy(), angs[i]) for i in range(len(rim2))]
    print("[terrain] verts", len(V), "tris", len(of), "rim", len(rim))
    return ob, rim


def rock_body(mb, cx, cy, rfun, zfun, depth, tip_off=(0.0, 0.0), seed=1, facet=0.6, rim=None, inset=0.24,
              soil_d=0.85, taper=((0, 1.0), (0.14, 0.95), (0.34, 0.82), (0.55, 0.64), (0.75, 0.42), (0.86, 0.28), (1.0, 0.0)),
              bulge=0.17, strata=(0.5, 0.95), drip=(0.26, 0.62),
              ledge=(0.05, 0.16), pal=ROCK, turf=None, band_fn=None):
    """Turf skirt + stratified rock tapering to a tip. Rings sit on the strata boundaries so the
    bands are clean; neighbouring rings have different vertex counts and are zipped into facets."""
    rng = random.Random(seed)
    V, tris, kinds = [], [], []
    ph1, ph2 = rng.uniform(0, TAU), rng.uniform(0, TAU)
    zmean = sum(zfun(TAU * i / 24) for i in range(24)) / 24
    rmean = sum(rfun(TAU * i / 24) for i in range(24)) / 24 - inset
    dipx, dipy = rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)
    D2 = depth - soil_d

    def Tf(s):
        s = max(0.0, min(1.0, s))
        for (s0, t0), (s1, t1) in zip(taper, taper[1:]):
            if s <= s1:
                k = (s - s0) / (s1 - s0)
                return t0 + (t1 - t0) * (k * k * (3 - 2 * k) * 0.35 + k * 0.65)
        return 0.0

    def Bf(a, s):
        env = min(1.0, s / 0.07) * (0.55 + 0.45 * (1 - s))
        return env * (bulge * n3(math.cos(a) * 1.5 + seed * 1.37, math.sin(a) * 1.5, s * 2.4)
                      + 0.5 * bulge * math.sin(3 * a + ph1) * max(0.0, math.sin(PI * min(1.0, s * 1.2))) ** 0.8
                      + 0.3 * bulge * math.sin(5 * a + ph2))

    def surf(a, s, off=0.0):
        r = (rfun(a) - inset) * Tf(s) * (1 + Bf(a, s)) + off
        if s < 0.3:
            r = min(r, rfun(a) - 0.10)
        r = max(r, 0.02)
        k = min(1.0, s * 3.0)
        x = cx + tip_off[0] * s ** 1.3 + r * math.cos(a)
        y = cy + tip_off[1] * s ** 1.3 + r * math.sin(a)
        z = zfun(a) * (1 - k) + zmean * k - soil_d - s * D2 + k * (dipx * (x - cx) + dipy * (y - cy))
        return Vector((x, y, z))

    def ring(s, off, jr=0.05, jz=0.035):
        rr = max(0.05, rmean * Tf(s) + off)
        n = max(5, min(96, int(round(TAU * rr / facet))))
        ph = rng.uniform(0, TAU)
        idx, ang = [], []
        for i in range(n):
            a = ph + TAU * (i + rng.uniform(-0.2, 0.2)) / n
            p = surf(a, s, off + rng.uniform(-jr, jr))
            p.z += rng.uniform(-jz, jz)
            idx.append(len(V))
            ang.append(a)
            V.append(p)
        return idx, ang

    def band(A_, B_, kind):
        for t in zipper(A_[0], A_[1], B_[0], B_[1]):
            tris.append(t)
            kinds.append(kind)

    if rim is not None:
        n = len(rim)
        r0 = list(range(len(V), len(V) + n))
        V.extend(p.copy() for p, a in rim)
        a0 = [a for p, a in rim]
        r1, r2 = [], []
        for i, (p, a) in enumerate(rim):
            d = drip[0] + (drip[1] - drip[0]) * (0.5 + 0.5 * n3(math.cos(a) * 7 + seed, math.sin(a) * 7, 2.2)) \
                * (0.5 + 0.5 * (i % 2))
            u = Vector((math.cos(a), math.sin(a), 0))
            r1.append(len(V))
            V.append(p + u * 0.07 + Vector((0, 0, -d)))
        for i, (p, a) in enumerate(rim):
            u = Vector((math.cos(a), math.sin(a), 0))
            r2.append(len(V))
            V.append(Vector((p.x, p.y, V[r1[i]].z + 0.05)) - u * 0.17)
        for i in range(n):
            j = (i + 1) % n
            for t, kd in (((r0[i], r0[j], r1[j]), "turf"), ((r0[i], r1[j], r1[i]), "turf"),
                          ((r1[i], r1[j], r2[j]), "under"), ((r1[i], r2[j], r2[i]), "under")):
                tris.append(t)
                kinds.append(kd)
        top = ring(0.0, 0.0)
        band((r2, a0), top, ("soil", 0))
    else:
        top = ring(0.0, 0.0)
    cuts, tip_start = [0.0], 0.86
    while cuts[-1] < tip_start - 0.03:
        cuts.append(min(tip_start, cuts[-1] + rng.uniform(*strata) / D2))
    K = len(cuts) - 1
    offs = [0.0] + [rng.choice((-1, 1)) * rng.uniform(*ledge) for _ in range(K)]
    last = top
    for k in range(K):
        s0, s1 = cuts[k], cuts[k + 1]
        off = offs[k] * (1 - s0)
        rings = [top]
        nmid = int((s1 - s0) * D2 / 0.66)
        for m in range(1, nmid + 1):
            rings.append(ring(s0 + (s1 - s0) * m / (nmid + 1), off))
        rings.append(ring(s1, off))
        for A_, B_ in zip(rings, rings[1:]):
            band(A_, B_, ("rise", k))
        last = rings[-1]
        if k < K - 1:
            top = ring(s1, offs[k + 1] * (1 - s1), jz=0.02)
            band(last, top, ("ledge", k + 1))
    apex = len(V)
    V.append(Vector((cx + tip_off[0], cy + tip_off[1], zmean - depth - 0.2 * depth / 6)))
    n = len(last[0])
    for i in range(n):
        tris.append((last[0][i], last[0][(i + 1) % n], apex))
        kinds.append(("rise", K))
    cols = []
    turf = turf or (C["turf"], C["turf_d"])
    for t, kd in zip(tris, kinds):
        a, b_, c_ = V[t[0]], V[t[1]], V[t[2]]
        cen = (a + b_ + c_) / 3
        nrm = (b_ - a).cross(c_ - a)
        if nrm.length > 1e-12:
            nrm.normalize()
        s_ = max(0.0, min(1.0, (zmean - soil_d - cen.z) / D2))
        axis = Vector((cx + tip_off[0] * s_ ** 1.3, cy + tip_off[1] * s_ ** 1.3, cen.z))
        if nrm.dot(cen - axis) < 0:
            nrm = -nrm
        if kd == "turf":
            c = jit(turf[0], 0.10, rng, 0.03)
        elif kd == "under":
            c = jit(mixc(turf[1], pal[0], 0.5), 0.10, rng)
        else:
            typ, k = kd
            if band_fn is not None and typ != "soil":
                k = band_fn(cen.z)
            base = pal[0] if typ == "soil" else pal[min(1 + k, len(pal) - 1)]
            if typ == "ledge":
                base = mixc(base, C["sand"], 0.35) if nrm.z > 0.25 else mulc(base, 0.6)
            elif rng.random() < 0.14:
                base = mixc(base, pal[min(2 + k, len(pal) - 1)], 0.4)
            c = jit(base, 0.09, rng, 0.03)
        cols.append(c)
    mb.add((V, tris), cols)
    return {"surf": surf, "zmean": zmean, "cuts": cuts, "D2": D2, "soil_d": soil_d}


def build_island():
    group("Island")
    top, rim = build_terrain()
    zr = [p.z for p, a in rim]
    ra = [a for p, a in rim]

    def zfun(a):
        a %= TAU
        for i in range(len(ra)):
            a0, a1 = ra[i], ra[(i + 1) % len(ra)]
            if i == len(ra) - 1:
                a1 += TAU
            aa = a if a >= ra[0] else a + TAU
            if a0 <= aa <= a1:
                return lerp(zr[i], zr[(i + 1) % len(ra)], (aa - a0) / (a1 - a0))
        return zr[0]

    mb = MB("Island_Rock")
    body = rock_body(mb, 0.0, 0.0, redge, zfun, depth=7.5, tip_off=(-0.7, 0.3), seed=4, rim=rim, facet=0.62)
    mb.build([M_ROCK])
    cuts, D2, zmean, soil_d = body["cuts"], body["D2"], body["zmean"], body["soil_d"]

    def band(z):
        """which stratum of the main body sits at height z (so the bands run on through the spires)."""
        s_ = (zmean - soil_d - z) / D2
        k = 0
        while k < len(cuts) - 1 and s_ > cuts[k + 1]:
            k += 1
        return k

    # hanging spires: secondary inverted peaks that break up the silhouette
    sp = MB("Island_Spires")
    rng = random.Random(77)
    for i, (a_deg, rho, L, R) in enumerate([(190, 0.62, 2.6, 1.25), (232, 0.50, 2.4, 1.1), (285, 0.42, 1.9, 1.2),
                                            (335, 0.60, 2.3, 1.1), (20, 0.60, 1.8, 0.95), (130, 0.60, 2.0, 1.0),
                                            (250, 0.75, 1.6, 0.9), (315, 0.80, 1.5, 0.85), (210, 0.82, 1.4, 0.8)]):
        a = math.radians(a_deg)
        # find the depth where the body is rho wide, hang the spire from there
        s = 0.0
        while s < 0.95 and (body["surf"](a, s) - Vector((0, 0, body["surf"](a, s).z))).length > rho * redge(a):
            s += 0.02
        p = body["surf"](a, s)
        cxs, cys = p.x - 0.35 * math.cos(a) * R, p.y - 0.35 * math.sin(a) * R
        sd = 100 + i
        rock_body(sp, cxs, cys, lambda t, R=R, sd=sd: R * (1 + 0.18 * n3(math.cos(t) * 1.4 + sd, math.sin(t) * 1.4, 0.3)),
                  lambda t, z=p.z + 0.55: z, depth=L + 0.55, tip_off=(rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25)),
                  seed=sd, facet=0.5, rim=None, inset=0.0, soil_d=0.0, bulge=0.2,
                  taper=((0, 1.0), (0.3, 0.86), (0.6, 0.56), (0.86, 0.22), (1.0, 0.0)),
                  strata=(0.45, 0.8), ledge=(0.03, 0.09), band_fn=band)
    sp.build([M_ROCK])
    return body, rim


# =====================================================================================
#  water: pond, stream, waterfall
# =====================================================================================
def fall_pt(t, X=1.7, H=4.3):
    lip = Vector((_E.x, _E.y, stream_z(1.0)))
    d = Vector((FALL_DIR.x, FALL_DIR.y, 0.0))
    if t <= 0:
        return lip + d * (X * t)
    return lip + d * (X * t) + Vector((0, 0, -H * t * t))


def build_water():
    group("Water")
    rng = random.Random(5)
    # ---- pond surface
    pts = [pond_pt(TAU * i / 34, 1.035) for i in range(34)]
    nrim = len(pts)
    y = POND.y - 1.6
    j = 0
    while y < POND.y + 1.6:
        x = POND.x - 2.0 + (j % 2) * 0.17
        while x < POND.x + 2.0:
            p = Vector((x + rng.uniform(-0.07, 0.07), y + rng.uniform(-0.07, 0.07)))
            if pond_q(p) < 0.9:
                pts.append(p)
            x += 0.34
        y += 0.30
        j += 1
    res = geometry.delaunay_2d_cdt(pts, [], [list(range(nrim))], 1, 1e-5)
    V = [Vector((p.x, p.y, WATER + (rng.uniform(-0.007, 0.007) if pond_q(p) < 0.95 else 0.0))) for p in res[0]]
    mb = MB("Pond")
    mb.add((V, [tuple(f) for f in res[2]]),
           lambda c, n: mixc(C["w_deep"], C["w_shal"], smooth(0.35, 1.0, pond_q(c))), vary=0.06)
    mb.build([M_POND])
    # ---- stream ribbon (starts under the pond surface so there is no visible block at the join)
    mb = MB("Stream")
    N = 22
    V, F, cols = [], [], []
    for i in range(N + 1):
        t = i / N
        c, d = poly_at(STREAM_PTS, t)
        nrm = Vector((-d.y, d.x))
        z = stream_z(t) - (0.006 if i < 3 else 0.0)
        w = 0.34 - 0.05 * t
        V += [Vector((c.x - nrm.x * w, c.y - nrm.y * w, z)), Vector((c.x, c.y, z + 0.014)),
              Vector((c.x + nrm.x * w, c.y + nrm.y * w, z))]
        if i:
            b0, b1 = 3 * (i - 1), 3 * i
            for q in ((b0, b0 + 1, b1 + 1, b1), (b0 + 1, b0 + 2, b1 + 2, b1 + 1)):
                for tri in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
                    F.append(tri)
                    foam = rng.random() < 0.10 + 0.5 * smooth(0.72, 1.0, t)
                    cols.append(jit(C["w_white"] if foam else mixc(C["w_shal"], C["w_mid"], smooth(0.1, 0.9, t)),
                                    0.06, rng))
    mb.add((V, F), cols)
    mb.build([M_STREAM])
    # ---- waterfall: a lens-shaped sheet on a real projectile arc, plus strands, foam, spray
    mb = MB("Waterfall")
    N = 20
    lat = Vector((-FALL_DIR.y, FALL_DIR.x, 0.0))
    fwd = Vector((FALL_DIR.x, FALL_DIR.y, 0.0))
    prof = [(-1.0, 0.0), (-0.6, 0.72), (-0.2, 1.0), (0.2, 1.0), (0.6, 0.72), (1.0, 0.0), (0.5, -0.35), (-0.5, -0.35)]
    tones = [C["w_white"], srgb("#c4e9f6"), srgb("#8ed0ec")]
    base_tone = [1, 0, 2, 0, 1]
    V, F, cols = [], [], []
    for i in range(N + 1):
        t = -0.07 + 1.07 * i / N
        p = fall_pt(t)
        tan = (fall_pt(t + 0.01) - fall_pt(t - 0.01)).normalized()
        nrm = tan.cross(lat).normalized()                    # outward / up side of the sheet
        w = 0.36 * (1 - 0.2 * smooth(0.0, 0.25, t)) + 0.28 * max(0.0, t) ** 2
        th = 0.03 + 0.14 * smooth(-0.07, 0.12, t) + 0.10 * max(0.0, t)
        for a_, b_ in prof:
            wob = 0.04 * (0.3 + max(0.0, t)) * n3(a_ * 2.1, t * 9.0, b_ * 3.0 + 4.0)
            V.append(p + lat * (a_ * w + wob) + nrm * (b_ * th))
        if i:
            for s in range(8):
                a0, a1 = 8 * (i - 1) + s, 8 * (i - 1) + (s + 1) % 8
                b0, b1 = 8 * i + s, 8 * i + (s + 1) % 8
                if s < 5:                                    # the five outer strips carry the streaks
                    k = n3(s * 2.7 + 1.3, t * 1.5, 0.7)              # slow along the fall = long streaks
                    tone = base_tone[s] + (-1 if k > 0.18 else (1 if k < -0.22 else 0))
                    if t < 0.07 or n3(s * 1.7 + 4.0, t * 2.0, 3.3) > 0.25 - 0.7 * smooth(0.5, 1.0, t):
                        tone = 0
                    base = tones[max(0, min(2, tone))]
                else:
                    base = srgb("#6fb6d8")
                al = 1.0 - smooth(0.84 + 0.06 * n3(s * 1.9, 0.3, 2.0), 1.0, t)
                for tri in ((a0, a1, b1), (a0, b1, b0)):
                    F.append(tri)
                    cols.append(jit(base, 0.045, rng) + (al,))
    mb.add((V, F), cols)
    # thinner strands that peel away from the main sheet
    for k in range(7):
        side = -1 if k % 2 else 1
        t0, t1 = rng.uniform(0.0, 0.14), rng.uniform(0.6, 1.0)
        offl, X = side * rng.uniform(0.22, 0.5), 1.7 + rng.uniform(-0.25, 0.4)
        P, R = [], []
        for i in range(14):
            t = lerp(t0, t1, i / 13)
            P.append(fall_pt(t, X) + lat * (offl * (1 + 0.8 * t)) + Vector((0, 0, 0.03)))
            R.append(0.075 * math.sin(PI * (0.06 + 0.94 * i / 13)) ** 0.6 + 0.01)
        mb.add(P_tube(P, R, 4), C["w_white"] if k % 3 else srgb("#c4e9f6"), vary=0.05)
    # foam where the stream tips over the lip
    lip = fall_pt(0.0)
    for k in range(7):
        o = lat * rng.uniform(-0.36, 0.36) + fwd * rng.uniform(-0.22, 0.10)
        mb.add(P_blob(rng.uniform(0.05, 0.085), 1, k, 0.25, (1.2, 1.0, 0.55)), C["w_white"],
               TR(lip + o + Vector((0, 0, 0.07))), vary=0.04)
    # droplets / spray
    for k in range(85):
        t = rng.uniform(0.15, 1.0) ** 0.7
        p = fall_pt(t, 1.7 + rng.uniform(-0.2, 0.5)) + lat * rng.uniform(-1, 1) * (0.35 + 0.75 * t) \
            + Vector((0, 0, rng.uniform(-0.3, 0.3)))
        tan = (fall_pt(t + 0.01) - fall_pt(t - 0.01)).normalized()
        r = rng.uniform(0.02, 0.045)
        mb.add(P_frustum(r, 0.0, r * 3.2, 4, z0=-r * 1.2), C["w_white"], TR(p) @ align_z(-tan), vary=0.05)
    mb.build([M_FALL])
    # the fall dissolves into a small cloud of mist
    cloud("FallMist", fall_pt(0.95) + Vector((0.1, 0.0, -0.42)), 2.6, 0.66, 1.5, 91, math.atan2(_rt.y, _rt.x))
    return lip


# =====================================================================================
#  clouds
# =====================================================================================
def cloud(name, center, L, H, W, seed, yaw=0.0, far=None, kind="cumulus"):
    """flat-based low-poly cloud. kind: 'cumulus' (puffy) or 'streak' (long thin bank).
    far=(fog_rgb, fog_amount) switches to the hazy far material."""
    rng = random.Random(seed)
    mb = MB(name)
    M = TR(center, (0, 0, yaw))

    def col(c, n):
        return mixc((0.80, 0.80, 0.92), (1.0, 1.0, 1.0), smooth(-0.5, 0.5, n.z))

    if kind == "cumulus":
        n = max(3, int(round(L / (H * 1.2))))
        big = rng.uniform(0.35, 0.65)
        for k in range(n):
            t = (k + 0.5) / n
            prof = max(0.0, 1 - abs(t - big) / max(big, 1 - big)) ** 0.7
            r = H * (0.36 + 0.64 * prof) * rng.uniform(0.86, 1.1)
            x, y = (t - 0.5) * L * 0.86, rng.uniform(-0.25, 0.25) * W
            sy = max(0.7, min(1.6, W / (2 * r)))
            mb.add(P_blob(r, 2, seed * 7 + k, 0.13, (1.25, sy, 1.0), 1.5, flat=0.12), col, M @ TR((x, y, r * 0.45)),
                   vary=0.025)
            if prof > 0.55 and rng.random() < 0.75:
                r2 = r * rng.uniform(0.48, 0.62)
                mb.add(P_blob(r2, 2, seed * 11 + k, 0.13, (1.15, sy * 0.9, 1.0), 1.5), col,
                       M @ TR((x + rng.uniform(-0.5, 0.5) * r, y + rng.uniform(-0.2, 0.2) * r, r * 1.12)), vary=0.025)
    else:
        n = 6
        for k in range(n):
            t = (k + 0.5) / n
            lx = L / n * rng.uniform(0.9, 1.5)
            hz = H * (0.45 + 0.55 * math.sin(PI * t)) * rng.uniform(0.7, 1.1)
            mb.add(P_blob(1.0, 2, seed * 5 + k, 0.10, (lx, W * 0.5, hz), 1.2, flat=0.3), col,
                   M @ TR(((t - 0.5) * L * 0.8, rng.uniform(-0.2, 0.2) * W, rng.uniform(-0.35, 0.35) * H)), vary=0.02)
    ob = mb.build([M_CLOUD_FAR if far else M_CLOUD])
    if far:
        ob.color = (far[0][0], far[0][1], far[0][2], far[1])
    return ob


def build_clouds():
    group("Clouds")
    yaw = math.atan2(_rt.y, _rt.x)
    # (u, v, distance, length, height, depth)
    near = [(0.11, 0.12, 22.0, 5.6, 1.1, 2.4), (0.90, 0.17, 24.0, 7.0, 1.3, 2.6),
            (0.17, 0.42, 46.0, 10.0, 1.9, 4.2), (0.90, 0.36, 52.0, 12.0, 2.2, 5.0),
            (0.12, 0.90, 90.0, 24.0, 3.8, 9.0), (0.70, 0.93, 100.0, 16.0, 2.6, 8.0)]
    for i, (u, v, d, L, H, W) in enumerate(near):
        cloud(f"Cloud{i}", F2W(u, v, d), L, H, W, 30 + i, yaw)
    # far, flat cloud banks near the glow band: they give the sky its layers
    far = [(0.60, 0.665, 300.0, 150.0, 4.0, 30.0, 0.50), (0.88, 0.70, 380.0, 170.0, 4.5, 30.0, 0.55),
           (0.16, 0.64, 330.0, 150.0, 5.0, 30.0, 0.55), (0.34, 0.76, 420.0, 190.0, 4.5, 30.0, 0.62),
           (0.80, 0.605, 240.0, 80.0, 3.5, 22.0, 0.42), (0.52, 0.80, 480.0, 150.0, 4.0, 30.0, 0.66),
           (0.10, 0.535, 460.0, 150.0, 4.0, 30.0, 0.72), (0.93, 0.515, 520.0, 150.0, 4.0, 30.0, 0.74)]
    for i, (u, v, d, L, H, W, fog) in enumerate(far):
        cloud(f"CloudBank{i}", F2W(u, v, d), L, H, W, 60 + i, yaw, far=(sky_uv(u, v), fog), kind="streak")


# =====================================================================================
#  shared prop colours + small builders
# =====================================================================================
W_LOG = [srgb(h) for h in ("#8f5d3d", "#7c4e31", "#9a6843")]
W_END = srgb("#d6ab74")
W_PLANK = [srgb(h) for h in ("#a67750", "#96683f", "#b08458")]
W_DARK = srgb("#4a3020")
W_ROOF = [srgb(h) for h in ("#b84a30", "#a53f2a", "#c65a3c", "#93382a", "#8a4a3a")]
CREAM = srgb("#eadbb8")
TEAL = srgb("#3f857c")
STONES = [srgb(h) for h in ("#9a938c", "#847e7a", "#aaa196", "#8d8478")]
WW, WW_D = srgb("#ebe2d0"), srgb("#cdbfa6")
METAL = srgb("#2e2a2c")
BARK = srgb("#6b4a35")
FLOWERS = [srgb(h) for h in ("#f6f1e6", "#f4c93c", "#e8689a", "#b06ad6", "#ef5b3c", "#7fb4f0")]
GLOW_WIN = (1.9, 1.0, 0.22)
GLOW_LAMP = (2.8, 1.4, 0.32)
BOXF = P_box(1, 1, 1)[1]
PINE_PAL = [(srgb("#1f5a3a"), srgb("#4a9a52")), (srgb("#225c4b"), srgb("#4f9a78")), (srgb("#2f612c"), srgb("#73a544"))]
AUT_PAL = [(srgb("#b5441c"), srgb("#f09a2e")), (srgb("#9c2c1c"), srgb("#e2582c")), (srgb("#c0841c"), srgb("#f5cc4a"))]


def bx(mb, M, size, loc, col, rot=(0, 0, 0), base=False, vary=0.06, mi=0):
    """one box = one piece of wood / stone: the whole piece gets its own shade."""
    c = jit(col, vary, mb.rng, 0.02) if vary else col
    mb.add(P_box(size[0], size[1], size[2], base), c, M @ TR(loc, rot), mi=mi, vary=0.012 if vary else 0.0)


def logpiece(mb, M, p1, p2, r, col, end, n=6):
    v, f = P_tube([p1, p2], [r, r], n)
    c = jit(col, 0.07, mb.rng, 0.02)
    mb.add((v, f), [jit(c, 0.02, mb.rng) for _ in range(n)] + [jit(end, 0.05, mb.rng)] * 2, M)


def P_ring(r_in, r_out, h, n, phase=0.0, z0=0.0):
    v, f = [], []
    for r, z in ((r_in, z0), (r_out, z0), (r_out, z0 + h), (r_in, z0 + h)):
        for i in range(n):
            a = phase + TAU * i / n
            v.append((r * math.cos(a), r * math.sin(a), z))
    for i in range(n):
        j = (i + 1) % n
        f += [(i, j, n + j, n + i), (n + i, n + j, 2 * n + j, 2 * n + i), (2 * n + i, 2 * n + j, 3 * n + j, 3 * n + i),
              (3 * n + i, 3 * n + j, j, i)]
    return v, f


def basis(x, y, z):
    return Matrix(((x[0], y[0], z[0], 0), (x[1], y[1], z[1], 0), (x[2], y[2], z[2], 0), (0, 0, 0, 1)))


def lantern(mb, M, s=1.0):
    """small hanging lantern, origin at the hook. Glow goes to material slot 1."""
    bx(mb, M, (0.13 * s, 0.13 * s, 0.025 * s), (0, 0, -0.06 * s), METAL, vary=0)
    mb.add(P_frustum(0.105 * s, 0.0, 0.07 * s, 4, z0=-0.05 * s, phase=PI / 4), METAL, M)
    mb.add(P_box(0.085 * s, 0.085 * s, 0.15 * s), GLOW_LAMP, M @ TR((0, 0, -0.15 * s)), mi=1)
    for sx in (-1, 1):
        for sy in (-1, 1):
            bx(mb, M, (0.016 * s, 0.016 * s, 0.17 * s), (sx * 0.052 * s, sy * 0.052 * s, -0.15 * s), METAL, vary=0)
    bx(mb, M, (0.13 * s, 0.13 * s, 0.02 * s), (0, 0, -0.245 * s), METAL, vary=0)
    mb.add(P_rod((0, 0, 0), (0, 0, -0.05 * s), 0.008 * s, 3), METAL, M)


def window(mb, Mw, w, h, rng, shutters=True, flowers=True):
    """window in a wall-plane frame: x along the wall, z up, the wall faces -Y. Glow = slot 1."""
    for ix in (-1, 1):
        for iz in (-1, 1):
            c = mulc(GLOW_WIN, rng.uniform(0.82, 1.1) * (1.0 if iz < 0 else 0.78))
            mb.add(P_box(w / 2 - 0.02, 0.02, h / 2 - 0.02), c, Mw @ TR((ix * w / 4, -0.03, iz * h / 4)), mi=1)
    t = 0.065
    for sz, lc in (((w + 2 * t, 0.09, t), (0, -0.04, h / 2 + t / 2)), ((w + 2 * t, 0.09, t), (0, -0.04, -h / 2 - t / 2)),
                   ((t, 0.09, h), (w / 2 + t / 2, -0.04, 0)), ((t, 0.09, h), (-w / 2 - t / 2, -0.04, 0)),
                   ((0.035, 0.05, h), (0, -0.045, 0)), ((w, 0.05, 0.035), (0, -0.045, 0))):
        bx(mb, Mw, sz, lc, CREAM, vary=0.03)
    bx(mb, Mw, (w + 0.26, 0.14, 0.05), (0, -0.07, -h / 2 - t - 0.02), CREAM, vary=0.03)
    if shutters:
        sw = w / 2 + 0.02
        for sx in (-1, 1):
            cx = sx * (w / 2 + t + sw / 2 + 0.01)
            bx(mb, Mw, (sw, 0.04, h + 2 * t), (cx, -0.03, 0), TEAL, vary=0.07)
            for k in range(4):
                bx(mb, Mw, (sw - 0.06, 0.02, 0.035), (cx, -0.055, (k - 1.5) * (h + 2 * t) / 4.6), mulc(TEAL, 0.72), vary=0.04)
    if flowers:
        zb = -h / 2 - t - 0.14
        bx(mb, Mw, (w + 0.16, 0.16, 0.13), (0, -0.15, zb), W_PLANK[1], vary=0.05)
        for k in range(5):
            x = (k - 2) * (w + 0.06) / 5 + rng.uniform(-0.02, 0.02)
            mb.add(P_blob(0.075, 1, k, 0.25, (1, 1, 0.8)), srgb("#4f8f3a"), Mw @ TR((x, -0.15, zb + 0.10)), vary=0.1)
            mb.add(P_ico(0.042, 1), rng.choice(FLOWERS[1:5]), Mw @ TR((x + rng.uniform(-0.03, 0.03), -0.2, zb + 0.17)),
                   vary=0.05)


# =====================================================================================
#  cabin: log walls, shingle roof, porch, shutters, chimney, woodpile
# =====================================================================================
CHIM_TOP = [None]


def build_cabin():
    group("Cabin")
    M = TR((CAB.x, CAB.y, CAB_Z), (0, 0, CAB_ROT))
    rng = random.Random(41)
    W2, L2, PY, Z0 = 1.35, 1.6, -2.55, 0.26
    ZW, RISE = 1.68, 1.25
    # ---------------- foundation + log walls + gables
    mb = MB("Cabin_Walls")

    def course(x0, y0, x1, y1, depth=0.26, h=0.32, z0=-0.08):
        p0, p1 = Vector((x0, y0)), Vector((x1, y1))
        L = (p1 - p0).length
        d = (p1 - p0) / L
        ang, t = math.atan2(d.y, d.x), 0.0
        while t < L - 0.02:
            l = min(rng.uniform(0.34, 0.62), L - t)
            if L - t - l < 0.22:
                l = L - t
            c = p0 + d * (t + l / 2)
            bx(mb, M, (l - 0.025, depth, h + rng.uniform(-0.03, 0.02)), (c.x, c.y, z0), rng.choice(STONES),
               (0, 0, ang), True, 0.08)
            t += l

    course(-1.47, PY - 0.05, 1.47, PY - 0.05)
    course(1.47, PY - 0.05, 1.47, 1.72)
    course(1.47, 1.72, -1.47, 1.72)
    course(-1.47, 1.72, -1.47, PY - 0.05)
    bx(mb, M, (2.8, 4.2, 0.12), (0, -0.42, 0.12), W_DARK, base=True, vary=0.03)
    for i in range(7):
        zc = Z0 + 0.10 + 0.2 * i
        for sx in (-1, 1):
            logpiece(mb, M, (sx * W2, -L2 - 0.19, zc), (sx * W2, L2 + 0.19, zc), 0.118, rng.choice(W_LOG), W_END)
        for sy in (-1, 1):
            logpiece(mb, M, (-W2 - 0.19, sy * L2, zc + 0.10), (W2 + 0.19, sy * L2, zc + 0.10), 0.118, rng.choice(W_LOG),
                     W_END)

    def ztop(x):
        return ZW + RISE * (1 - abs(x) / W2) + 0.04

    for sy in (-1, 1):
        nb = 10
        bw = 2 * W2 / nb
        for i in range(nb):
            x0, x1 = -W2 + i * bw + 0.008, -W2 + (i + 1) * bw - 0.008
            y0, y1 = sy * L2 - 0.04, sy * L2 + 0.04
            v = [(x0, y0, 1.58), (x1, y0, 1.58), (x1, y1, 1.58), (x0, y1, 1.58),
                 (x0, y0, ztop(x0)), (x1, y0, ztop(x1)), (x1, y1, ztop(x1)), (x0, y1, ztop(x0))]
            mb.add((v, BOXF), jit(rng.choice(W_PLANK), 0.07, rng), M, vary=0.012)
    mb.build()
    # ---------------- roof: deck + overlapping shingle courses + ridge + barge boards + chimney
    mb = MB("Cabin_Roof")
    th = math.atan2(RISE, W2)
    c_, s_ = math.cos(th), math.sin(th)
    ZR = ZW + RISE + 0.17
    HW = W2 + 0.34
    SL = HW / c_
    Y0, Y1 = PY - 0.28, L2 + 0.32
    for sgn in (-1, 1):
        ds, nr, dy = Vector((sgn * c_, 0, -s_)), Vector((sgn * s_, 0, c_)), Vector((0, 1, 0))
        Md = basis(ds, dy, nr)
        cen = Vector((0, (Y0 + Y1) / 2, ZR - 0.10)) + ds * (SL / 2)
        mb.add(P_box(SL, Y1 - Y0 - 0.06, 0.06), W_DARK, M @ Matrix.Translation(cen) @ Md, vary=0.03)
        NC = 7
        ex = SL / NC
        tau = math.radians(5)
        ds2, nr2 = ds * math.cos(tau) + nr * math.sin(tau), nr * math.cos(tau) - ds * math.sin(tau)
        Ms = basis(ds2, dy, nr2)
        for k in range(NC - 1, -1, -1):
            y = Y0 - rng.uniform(0.0, 0.12)
            while y < Y1 - 0.02:
                w = rng.uniform(0.36, 0.56)
                if Y1 - (y + w) < 0.22:
                    w = Y1 - y
                ln = ex + 0.11 + rng.uniform(-0.02, 0.03)
                cen = Vector((0, y + w / 2, ZR - 0.04)) + ds * (k * ex + ln / 2 - 0.02) + nr * 0.035
                col = rng.choice(W_ROOF)
                if rng.random() < 0.05:
                    col = srgb("#6f7a4a")                       # the odd mossy shingle
                mb.add(P_box(ln, w - 0.02, 0.05), jit(col, 0.08, rng, 0.02), M @ Matrix.Translation(cen) @ Ms, vary=0.012)
                y += w
        for yb in (Y0 + 0.01, Y1 - 0.01):                            # barge boards on the gable ends
            cen = Vector((0, yb, ZR - 0.15)) + ds * (SL / 2 + 0.02)
            mb.add(P_box(SL + 0.06, 0.07, 0.16), CREAM, M @ Matrix.Translation(cen) @ Md, vary=0.03)
    y = Y0 - 0.03
    while y < Y1:
        l = min(0.5, Y1 + 0.03 - y)
        bx(mb, M, (0.26, l - 0.02, 0.075), (0, y + l / 2, ZR + 0.01), srgb("#7b2f22"), vary=0.08)
        y += l
    cx, cy, z = -0.42, 0.95, ZR - 1.0
    for k in range(6):
        bx(mb, M, (0.50 + rng.uniform(-0.04, 0.04), 0.50 + rng.uniform(-0.04, 0.04), 0.24),
           (cx + rng.uniform(-0.015, 0.015), cy + rng.uniform(-0.015, 0.015), z), rng.choice(STONES), base=True, vary=0.09)
        z += 0.235
    bx(mb, M, (0.64, 0.64, 0.07), (cx, cy, z), C["stone_d"], base=True)
    z += 0.07
    mb.add(P_frustum(0.11, 0.09, 0.2, 6, z0=z), srgb("#a8563a"), M @ TR((cx, cy, 0)), vary=0.05)
    CHIM_TOP[0] = M @ Vector((cx, cy, z + 0.22))
    mb.build()
    # ---------------- porch
    mb = MB("Cabin_Porch")
    for i in range(6):
        bx(mb, M, (2.86, 0.155, 0.05), (0, PY - 0.03 + 0.165 * i + 0.08, Z0 - 0.01), rng.choice(W_PLANK), base=True, vary=0.07)
    py = PY + 0.05
    for sx in (-1, 1):
        bx(mb, M, (0.13, 0.13, ZW - Z0 - 0.06), (sx * 1.25, py, Z0 + 0.04), W_PLANK[1], base=True, vary=0.05)
        bx(mb, M, (0.12, -L2 - py, 0.13), (sx * 1.25, (py - L2) / 2, ZW - 0.03), W_PLANK[1], vary=0.05)
    bx(mb, M, (2.76, 0.13, 0.15), (0, py, ZW - 0.02), W_PLANK[0], vary=0.05)
    bx(mb, M, (0.10, 0.10, RISE - 0.12), (0, py, ZW + 0.05), W_PLANK[1], base=True, vary=0.05)
    for sx in (-1, 1):
        mb.add(P_rod((0, py, ZW + 0.10), (sx * 0.74, py, ZW + 0.10 + RISE * (1 - 0.74 / W2) - 0.08), 0.045, 4),
               W_PLANK[1], M, vary=0.05)

    def rail(x0, y0, x1, y1):
        p0, p1 = Vector((x0, y0)), Vector((x1, y1))
        L = (p1 - p0).length
        ang = math.atan2((p1 - p0).y, (p1 - p0).x)
        c = (p0 + p1) / 2
        zt = Z0 + 0.04 + 0.52
        bx(mb, M, (L, 0.06, 0.05), (c.x, c.y, zt), W_PLANK[2], (0, 0, ang), vary=0.05)
        bx(mb, M, (L, 0.04, 0.04), (c.x, c.y, Z0 + 0.14), W_PLANK[2], (0, 0, ang), vary=0.05)
        nb_ = max(1, int(L / 0.17))
        for i in range(nb_):
            q = p0.lerp(p1, (i + 0.5) / nb_)
            bx(mb, M, (0.035, 0.035, 0.40), (q.x, q.y, Z0 + 0.15), CREAM, (0, 0, ang), True, vary=0.04)

    rail(-1.25, py, -1.25, -L2 - 0.12)
    rail(1.25, py, 1.25, -L2 - 0.12)
    rail(-1.25, py, -0.93, py)
    rail(0.23, py, 1.25, py)
    for x in (-0.93, 0.23):
        bx(mb, M, (0.09, 0.09, 0.62), (x, py, Z0 + 0.04), W_PLANK[1], base=True, vary=0.05)
    bx(mb, M, (1.08, 0.28, 0.13), (-0.35, PY - 0.19, 0.07), W_PLANK[0], base=True, vary=0.06)
    bx(mb, M, (1.08, 0.28, 0.13), (-0.35, PY - 0.46, -0.06), W_PLANK[2], base=True, vary=0.06)
    # barrel on the porch
    Mb = M @ TR((0.92, -2.12, Z0 + 0.04))
    mb.add(P_frustum(0.16, 0.195, 0.2, 8), W_PLANK[0], Mb, vary=0.07)
    mb.add(P_frustum(0.195, 0.16, 0.2, 8, z0=0.2), W_PLANK[0], Mb, vary=0.07)
    for zz in (0.05, 0.19, 0.33):
        mb.add(P_frustum(0.20, 0.20, 0.03, 8, z0=zz), METAL, Mb)
    # hanging porch lantern
    lantern(mb, M @ TR((0.55, py + 0.02, ZW - 0.10)), 1.5)
    mb.build([M_VCOL, M_GLOW])
    add_light("PorchLight", "POINT", 90, (1.0, 0.56, 0.22), loc=M @ Vector((0.55, py - 0.05, ZW - 0.42)), size=0.08)
    # ---------------- door + windows
    mb = MB("Cabin_Openings")
    yf = -L2 - 0.125
    bx(mb, M, (0.80, 0.07, 1.27), (-0.35, yf, Z0 + 0.02), CREAM, base=True, vary=0.03)
    for i in range(4):
        bx(mb, M, (0.15, 0.05, 1.13), (-0.35 + (i - 1.5) * 0.158, yf - 0.035, Z0 + 0.04), TEAL, base=True, vary=0.08)
    for zz in (Z0 + 0.22, Z0 + 0.66):
        bx(mb, M, (0.60, 0.02, 0.055), (-0.35, yf - 0.065, zz), mulc(TEAL, 0.7), vary=0.04)
    mb.add(P_box(0.26, 0.02, 0.22), mulc(GLOW_WIN, 0.9), M @ TR((-0.35, yf - 0.066, Z0 + 0.95)), mi=1)
    bx(mb, M, (0.03, 0.03, 0.22), (-0.35, yf - 0.07, Z0 + 0.95), mulc(TEAL, 0.7), vary=0)
    mb.add(P_ico(0.032, 1), srgb("#d9b04a"), M @ TR((-0.13, yf - 0.08, Z0 + 0.58)))
    window(mb, M @ TR((0.63, yf, Z0 + 0.82)), 0.62, 0.55, rng)
    for y in (-0.72, 0.68):
        window(mb, M @ TR((W2 + 0.125, y, Z0 + 0.82), (0, 0, PI / 2)), 0.6, 0.55, rng)
    window(mb, M @ TR((-W2 - 0.125, 0.0, Z0 + 0.82), (0, 0, -PI / 2)), 0.6, 0.55, rng, flowers=False)
    window(mb, M @ TR((0, -L2 - 0.045, ZW + 0.52)), 0.36, 0.36, rng, shutters=False, flowers=False)
    mb.build([M_VCOL, M_GLOW])
    for nm, lp, pw in (("WinSpillFront", (0.63, yf - 0.45, Z0 + 0.75), 70), ("WinSpillDoor", (-0.35, yf - 0.4, Z0 + 0.9), 35),
                       ("WinSpillSideA", (W2 + 0.55, -0.72, Z0 + 0.7), 45), ("WinSpillSideB", (W2 + 0.55, 0.68, Z0 + 0.7), 45)):
        add_light(nm, "POINT", pw, (1.0, 0.55, 0.20), loc=M @ Vector(lp), size=0.18)
    # ---------------- woodpile + chopping block
    mb = MB("Woodpile")
    r = 0.098
    for row, n in enumerate((5, 4, 3)):
        for i in range(n):
            x = 1.66 + (i + 0.5 * row) * 0.20
            z = 0.02 + r + row * 0.172
            y0 = -1.28 + rng.uniform(-0.03, 0.03)
            logpiece(mb, M, (x, y0, z), (x, y0 + 0.62, z), r * rng.uniform(0.88, 1.05), rng.choice(W_LOG), W_END, 7)
    for x in (1.53, 2.59):
        bx(mb, M, (0.06, 0.06, 0.62), (x, -0.97, -0.03), W_DARK, base=True)
    Ms = M @ TR((2.25, -2.15, -0.03))

    def stump_col(c, n):
        return W_END if n.z > 0.5 else BARK

    mb.add(P_frustum(0.25, 0.22, 0.36, 8), stump_col, Ms, vary=0.07)
    bx(mb, Ms, (0.15, 0.035, 0.11), (0.02, 0.0, 0.39), srgb("#8a8f96"), (0, math.radians(-18), 0), vary=0.03)
    mb.add(P_rod((0.05, 0, 0.42), (0.36, 0.03, 0.80), 0.022, 5), W_PLANK[2], Ms, vary=0.05)
    for k, (dx, dy, yaw) in enumerate(((0.42, 0.25, 0.4), (0.50, -0.12, 1.9), (-0.38, 0.22, 2.7))):
        logpiece(mb, Ms @ TR((dx, dy, 0.09), (0, 0, yaw)), (-0.16, 0, 0), (0.16, 0, 0), 0.075, rng.choice(W_LOG), W_END, 5)
    mb.build()


# =====================================================================================
#  windmill: stone base, tapered tower, balcony, lattice sails
# =====================================================================================
MILL_ROT = math.radians(20)      # tower: the door looks towards the path
CAP_ROT = math.radians(-12)      # cap + sails


def build_windmill():
    group("Windmill")
    S_ = 1.1
    M = TR((MILL.x, MILL.y, MILL_Z - 0.06), (0, 0, MILL_ROT), S_)
    rng = random.Random(52)
    PH = math.radians(22.5)
    CA = math.cos(PH)

    def rz(z):
        return 0.80 - (z - 0.35) * 0.30 / 2.65

    mb = MB("Windmill")

    def stone(c, n):
        return rng.choice(STONES)

    mb.add(P_frustum(0.99, 0.95, 0.27, 10, z0=-0.13), stone, M, vary=0.09)
    mb.add(P_frustum(0.93, 0.87, 0.24, 10, z0=0.14, phase=0.31), stone, M, vary=0.09)

    def wash(c, n):
        return mixc(WW_D, WW, smooth(MILL_Z + 0.3, MILL_Z + 2.6, c.z))

    for z0, z1 in ((0.36, 1.30), (1.30, 2.25), (2.25, 3.00)):
        mb.add(P_frustum(rz(z0), rz(z1), z1 - z0, 8, z0=z0, phase=PH, cap_bottom=False, cap_top=False), wash, M, vary=0.035)
    for zb in (2.22, 2.95):
        mb.add(P_frustum(rz(zb) + 0.03, rz(zb + 0.08) + 0.03, 0.08, 8, z0=zb, phase=PH), W_DARK, M, vary=0.05)
    # balcony
    mb.add(P_ring(rz(1.30) - 0.03, 1.25, 0.06, 8, PH, z0=1.30), W_PLANK[0], M, vary=0.09)
    cor = [Vector((1.19 * math.cos(PH + TAU * i / 8), 1.19 * math.sin(PH + TAU * i / 8), 0)) for i in range(8)]
    for i in range(8):
        a = PH + TAU * i / 8
        mb.add(P_rod((rz(0.82) * math.cos(a), rz(0.82) * math.sin(a), 0.82), (1.15 * math.cos(a), 1.15 * math.sin(a), 1.30),
                     0.035, 4), W_DARK, M, vary=0.05)
        bx(mb, M, (0.06, 0.06, 0.46), (cor[i].x, cor[i].y, 1.36), W_PLANK[1], (0, 0, a), True, vary=0.05)
        p, q = cor[i], cor[(i + 1) % 8]
        for zz, rr in ((1.80, 0.027), (1.58, 0.02)):
            mb.add(P_rod((p.x, p.y, zz), (q.x, q.y, zz), rr, 4), W_PLANK[2], M, vary=0.05)
        for t in (1 / 3, 2 / 3):
            m_ = p.lerp(q, t)
            bx(mb, M, (0.03, 0.03, 0.44), (m_.x, m_.y, 1.36), CREAM, (0, 0, a), True, vary=0.04)
    # door (front facet) with a stone surround and a step
    yd = -rz(0.40) * CA + 0.03
    bx(mb, M, (0.62, 0.16, 0.86), (0, yd, 0.34), STONES[2], base=True, vary=0.05)
    bx(mb, M, (0.74, 0.18, 0.10), (0, yd, 1.20), STONES[0], base=True, vary=0.05)
    for i in range(3):
        bx(mb, M, (0.145, 0.05, 0.78), ((i - 1) * 0.15, yd - 0.07, 0.36), W_DARK, base=True, vary=0.12)
    for zz in (0.55, 0.95):
        bx(mb, M, (0.46, 0.02, 0.045), (0, yd - 0.10, zz), METAL, vary=0)
    bx(mb, M, (0.78, 0.36, 0.34), (0, -1.06, -0.10), STONES[1], base=True, vary=0.06)
    bx(mb, M, (0.9, 0.34, 0.22), (0, -1.34, -0.10), STONES[3], base=True, vary=0.06)

    def mwin(z, facet_rot, w=0.20, h=0.32):
        Mw = M @ TR((0, 0, 0), (0, 0, facet_rot)) @ TR((0, -rz(z) * CA - 0.012, z))
        mb.add(P_box(w, 0.03, h), mulc(GLOW_WIN, 0.95), Mw, mi=1)
        for sz, lc in (((w + 0.10, 0.07, 0.05), (0, -0.01, h / 2 + 0.025)), ((w + 0.14, 0.09, 0.05), (0, -0.02, -h / 2 - 0.025)),
                       ((0.05, 0.07, h), (w / 2 + 0.025, -0.01, 0)), ((0.05, 0.07, h), (-w / 2 - 0.025, -0.01, 0)),
                       ((0.025, 0.05, h), (0, -0.015, 0))):
            bx(mb, Mw, sz, lc, W_DARK, vary=0.05)

    mwin(1.92, 0.0, 0.22, 0.36)
    mwin(2.60, math.radians(45))
    mwin(0.95, math.radians(-45))
    mwin(2.60, math.radians(-45))
    # cap, shaft and sails: the cap turns on the tower, so it has its own heading (towards the camera, sun behind it)
    Mc = TR((MILL.x, MILL.y, MILL_Z - 0.06), (0, 0, CAP_ROT), S_)
    mb.add(P_frustum(0.57, 0.57, 0.10, 8, z0=3.0, phase=PH), W_DARK, M, vary=0.05)
    capc = srgb("#8a3526")
    mb.add(P_frustum(0.64, 0.43, 0.34, 8, z0=3.10, phase=PH), capc, Mc, vary=0.13, hue=0.04)
    mb.add(P_frustum(0.43, 0.0, 0.38, 8, z0=3.44, phase=PH), capc, Mc, vary=0.13, hue=0.04)
    mb.add(P_ico(0.05, 1), srgb("#d9b04a"), Mc @ TR((0, 0, 3.86)))
    mb.add(P_rod((0, 0, 3.82), (0, 0, 4.12), 0.012, 4), METAL, Mc)
    mb.add(([(0, 0, 4.10), (0, 0, 3.96), (-0.30, 0.02, 4.02)], [(0, 1, 2)]), srgb("#c8452e"), Mc)
    bx(mb, Mc, (0.30, 0.36, 0.26), (0, -0.50, 3.19), W_DARK, base=True, vary=0.05)
    hub = Vector((0, -1.02, 3.39))
    ax = (hub - Vector((0, -0.45, 3.33))).normalized()
    mb.add(P_rod((0, -0.45, 3.33), hub + ax * 0.06, 0.055, 6), W_DARK, Mc)
    Mh = Mc @ Matrix.Translation(hub) @ align_z(ax)
    mb.add(P_frustum(0.13, 0.13, 0.12, 8, z0=-0.06), W_DARK, Mh, vary=0.05)
    mb.add(P_frustum(0.09, 0.0, 0.14, 6, z0=0.06), METAL, Mh)
    # wall lantern by the door
    lantern(mb, M @ TR((0.47, yd - 0.12, 1.06)), 1.25)
    bx(mb, M, (0.04, 0.16, 0.04), (0.47, yd - 0.05, 1.08), METAL, vary=0)
    # ---- sails: stock + lattice, canvas on the back so the lattice shows against the glowing cloth
    ms = MB("Windmill_Sails")
    e1 = Vector((1, 0, 0))
    e2 = ax.cross(e1).normalized()
    BL = 1.66
    for k in range(4):
        beta = math.radians(27 + 90 * k)
        rad = e2 * math.cos(beta) + e1 * math.sin(beta)
        yb = -ax
        xb = yb.cross(rad).normalized()
        Bk = Mc @ Matrix.Translation(hub + ax * 0.10) @ basis(xb, yb, rad)
        bx(mb, Bk, (0.075, 0.06, BL), (0, 0, 0.03), W_DARK, base=True, vary=0.05)
        for i in range(8):
            bx(mb, Bk, (0.48, 0.03, 0.042), (0.275, -0.005, 0.34 + i * 0.185), srgb("#c7a777"), vary=0.07)
        bx(mb, Bk, (0.048, 0.04, BL - 0.30), (0.51, -0.005, 0.32), srgb("#c7a777"), base=True, vary=0.05)
        bx(mb, Bk, (0.032, 0.03, BL - 0.30), (0.27, -0.005, 0.32), srgb("#b9925f"), base=True, vary=0.05)
        bx(mb, Bk, (0.10, 0.02, BL - 0.36), (-0.088, 0.0, 0.34), srgb("#b9925f"), base=True, vary=0.05)
        V, F = [], []
        z0c, z1c, NS = 0.40, 0.40 + 0.80 * (BL - 0.40), 6
        for i in range(NS + 1):
            t = i / NS
            z = lerp(z0c, z1c, t)
            sag = 0.028 + 0.03 * math.sin(PI * t)
            V += [Vector((0.05, sag * 0.5, z)), Vector((0.24, sag, z)), Vector((0.43, sag * 0.6, z))]
            if i:
                b0, b1 = 3 * (i - 1), 3 * i
                F += [(b0, b0 + 1, b1 + 1), (b0, b1 + 1, b1), (b0 + 1, b0 + 2, b1 + 2), (b0 + 1, b1 + 2, b1 + 1)]
        ms.add((V, F), srgb("#f5dda8"), Bk, vary=0.06)
    mb.build([M_VCOL, M_GLOW])
    ms.build([M_SAIL])
    add_light("MillLantern", "POINT", 28, (1.0, 0.56, 0.22), loc=M @ Vector((0.47, yd - 0.3, 0.9)), size=0.06)


# =====================================================================================
#  plants and trees
# =====================================================================================
PROTECT = [(CAB, 1.9, CAB_Z + 1.3), (MILL, 0.9, MILL_Z + 0.55)]


def fit_tree(x, y, s, per_s, rad=0.6):
    """shrink a tree so its shadow stays below the lit parts of the cabin and the windmill."""
    p = Vector((x, y))
    for q, R, zlim in PROTECT:
        d = p - q
        along = d.dot(SUN_H)
        lat = (d - SUN_H * along).length
        if along > 0 and lat < R + rad * s:
            top_ok = zlim + max(0.0, along - R) * math.tan(SUN_EL)
            s = min(s, (top_ok - height(x, y)) / per_s)
    return s


def pine(name, x, y, s, seed, z=None, check=True, mats=None):
    if check:
        s2 = fit_tree(x, y, s, 3.2)
        if s2 < 0.38:
            print(f"[tree] {name} skipped (would shade)")
            return None
        if s2 < s - 0.01:
            print(f"[tree] {name} shrunk {s:.2f} -> {s2:.2f}")
        s = s2
    rng = random.Random(seed)
    z = (height(x, y) if z is None else z) - 0.06
    dark, light = PINE_PAL[seed % 3]
    lean = Vector((rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05), 0))
    M = TR((x, y, z))
    mb = MB(name)
    mb.add(P_frustum(0.11 * s, 0.07 * s, 0.75 * s, 6), BARK, M, vary=0.08)
    n = 5
    for k in range(n):
        t = k / (n - 1)
        r = (0.80 - 0.52 * t) * s * rng.uniform(0.93, 1.07)
        h = (1.00 - 0.28 * t) * s
        z0 = (0.48 + 0.50 * k) * s
        m = 8
        ph = rng.uniform(0, TAU)
        V = []
        for i in range(m):
            a = ph + TAU * i / m
            rr = r * (1.0 if i % 2 else 0.80) * rng.uniform(0.94, 1.06)
            V.append(Vector((rr * math.cos(a), rr * math.sin(a), z0 + (0.0 if i % 2 else 0.13 * s))) + lean * (z0 / s))
        V.append(Vector((0, 0, z0 + h)) + lean * ((z0 + h) / s))
        F = [(i, (i + 1) % m, m) for i in range(m)] + [tuple(reversed(range(m)))]
        base = mixc(dark, light, 0.25 + 0.75 * t)
        cols = [jit(base, 0.10, rng, 0.03) for _ in range(m)] + [mulc(dark, 0.6)]
        mb.add((V, F), cols, M, mi=1)
    return mb.build(mats or [M_VCOL, M_FOLIAGE])


def autumn_tree(name, x, y, s, seed, pal=0, z=None, mats=None):
    rng = random.Random(seed)
    z = (height(x, y) if z is None else z) - 0.06
    dark, light = AUT_PAL[pal]
    M = TR((x, y, z), (0, 0, rng.uniform(0, TAU)))
    mb = MB(name)
    mb.add(P_tube([(0, 0, 0), (0.05 * s, 0.02 * s, 0.6 * s), (-0.03 * s, 0.05 * s, 1.3 * s)],
                  [0.15 * s, 0.10 * s, 0.07 * s], 6), BARK, M, vary=0.08)
    for a in (0.6, 2.9, 4.6):
        ca, sa = math.cos(a), math.sin(a)
        mb.add(P_tube([(0, 0, 0.75 * s), (0.3 * s * ca, 0.3 * s * sa, 1.15 * s), (0.5 * s * ca, 0.5 * s * sa, 1.45 * s)],
                      [0.06 * s, 0.045 * s, 0.03 * s], 5), BARK, M, vary=0.08)

    def col(c, n):
        return mixc(dark, light, smooth(-0.35, 0.75, n.z))

    blobs = [((0, 0, 1.85), 0.78)] + [((0.58 * math.cos(a), 0.58 * math.sin(a), 1.42 + 0.28 * rng.random()),
                                       rng.uniform(0.46, 0.58)) for a in (0.6, 1.9, 2.9, 4.0, 4.6, 5.6)]
    for k, (o, r) in enumerate(blobs):
        mb.add(P_blob(r * s, 2 if k == 0 else 1, seed * 3 + k, 0.2, (1, 1, 0.85)), col,
               M @ TR((o[0] * s, o[1] * s, o[2] * s)), vary=0.09, hue=0.04, mi=1)
    return mb.build(mats or [M_VCOL, M_FOLIAGE])


def lawn_ok(p, margin=0.25, path=0.1):
    return (inside(p, margin) and pond_q(p) > 1.1 and poly_dist(p, STREAM_PTS)[0] > 0.45
            and path_dist(p) > PATH_W + path and cabin_dist(p) > 0.12 and garden_dist(p) > 0.15
            and (p - MILL).length > 1.25)


def tuft(mb, x, y, s, rng, col):
    z = height(x, y) - 0.012
    for k in range(rng.randint(3, 5)):
        a = rng.uniform(0, TAU)
        lean, h, w = rng.uniform(0.03, 0.11) * s, rng.uniform(0.18, 0.34) * s, 0.042 * s
        bx_, by_ = x + rng.uniform(-0.05, 0.05) * s, y + rng.uniform(-0.05, 0.05) * s
        V = [(bx_ + w * math.cos(a + q), by_ + w * math.sin(a + q), z) for q in (0, 2.1, 4.2)]
        V.append((bx_ + lean * math.cos(a), by_ + lean * math.sin(a), z + h))
        mb.add((V, [(0, 1, 3), (1, 2, 3), (2, 0, 3)]), col, vary=0.12, hue=0.04)


def flower(mb, x, y, col, rng, s=1.0):
    z = height(x, y) - 0.01
    h = rng.uniform(0.14, 0.25) * s
    lx, ly = rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03)
    mb.add(P_tube([(x, y, z), (x + lx, y + ly, z + h)], [0.013, 0.010], 3, cap=False), srgb("#5a8f3a"))
    mb.add(P_ico(rng.uniform(0.052, 0.07) * s, 1), col, TR((x + lx, y + ly, z + h + 0.02), scale=(1, 1, 0.7)), vary=0.08)


def bush(mb, x, y, s, seed, berries=None, z=None):
    rng = random.Random(seed)
    z = height(x, y) if z is None else z
    dark, light = srgb("#35703a"), srgb("#7db24c")

    def col(c, n):
        return mixc(dark, light, smooth(-0.3, 0.8, n.z))

    for k in range(rng.choice((2, 3))):
        o = Vector((rng.uniform(-0.22, 0.22), rng.uniform(-0.22, 0.22), 0)) * s
        r = rng.uniform(0.24, 0.36) * s
        mb.add(P_blob(r, 1, seed * 5 + k, 0.25, (1.15, 1.0, 0.8), flat=0.3), col, TR((x + o.x, y + o.y, z + r * 0.4)),
               vary=0.09, hue=0.03, mi=1)
        if berries:
            for j in range(5):
                a, e = rng.uniform(0, TAU), rng.uniform(0.2, 1.2)
                d = Vector((math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e) * 0.8)) * r * 1.02
                mb.add(P_ico(0.04 * s, 1), berries, TR((x + o.x + d.x, y + o.y + d.y, z + r * 0.4 + d.z)), vary=0.06)


def rock(mb, x, y, r, seed, z=None, scale=(1.0, 0.85, 0.65), light=None, dark=None):
    rng = random.Random(seed)
    z = height(x, y) if z is None else z
    lt, dk = light or C["stone_l"], dark or C["stone_d"]

    def col(c, n):
        return mixc(dk, lt, smooth(-0.2, 0.8, n.z))

    mb.add(P_blob(r, 1, seed, 0.28, scale), col, TR((x, y, z + r * scale[2] * 0.4), (0, 0, rng.uniform(0, TAU))), vary=0.07)


def fence(mb, pts, h=0.5, rails=(0.20, 0.38), post=0.075, rng=None):
    prev = None
    for p in pts:
        z = height(p.x, p.y) - 0.06
        M = TR((p.x, p.y, z), (0, 0, rng.uniform(-0.3, 0.3)))
        hh = h + rng.uniform(-0.03, 0.03)
        bx(mb, M, (post, post, hh + 0.06), (0, 0, 0), rng.choice(W_PLANK), base=True, vary=0.08)
        mb.add(P_frustum(post * 0.72, 0.0, 0.07, 4, z0=hh + 0.06, phase=PI / 4), W_PLANK[2], M, vary=0.06)
        if prev is not None:
            for r_ in rails:
                mb.add(P_rod((prev[0].x, prev[0].y, prev[1] + r_ + rng.uniform(-0.015, 0.015)),
                             (p.x, p.y, z + 0.06 + r_ + rng.uniform(-0.015, 0.015)), 0.027, 4), rng.choice(W_PLANK), vary=0.07)
        prev = (p, z + 0.06)


def build_trees():
    group("Trees")
    pines = [(0.5, 4.75, 1.0), (-0.55, 4.95, 0.85), (-1.55, 4.6, 1.05), (-2.5, 4.9, 0.8), (1.45, 5.05, 0.78),
             (-0.9, 3.55, 0.95), (-0.2, 3.9, 0.7), (-4.55, 4.0, 0.68), (-5.45, 2.9, 0.78), (-5.9, 1.5, 0.8),
             (-5.35, 0.35, 0.6), (-5.0, -0.9, 0.7), (-4.3, -0.2, 0.55), (6.35, -0.75, 0.5), (5.6, -2.9, 0.48),
             (-3.6, 4.6, 0.7), (2.5, 5.0, 0.55)]
    for i, (x, y, s) in enumerate(pines):
        pine(f"Pine_{i:02d}", x, y, s, 200 + i)
    autumn = [(-3.95, -2.75, 1.0, 0), (5.75, 0.3, 0.9, 1), (-4.9, 1.6, 0.8, 2), (5.3, -3.6, 0.62, 0), (-1.9, 4.0, 0.7, 2)]
    lv = MB("FallenLeaves")
    rng = random.Random(9)
    for i, (x, y, s, pal) in enumerate(autumn):
        s2 = fit_tree(x, y, s, 2.6, 0.9)
        if s2 < 0.45:
            print(f"[tree] Autumn_{i} skipped")
            continue
        autumn_tree(f"AutumnTree_{i}", x, y, s2, 300 + i, pal)
        for k in range(int(26 * s2)):
            a, d = rng.uniform(0, TAU), abs(rng.gauss(0, 0.75)) * s2 + 0.1
            px, py = x + d * math.cos(a), y + d * math.sin(a)
            if not inside(Vector((px, py)), 0.1) or pond_q(Vector((px, py))) < 1.05:
                continue
            z = height(px, py) + 0.015
            yaw, ln = rng.uniform(0, TAU), rng.uniform(0.05, 0.08)
            c, s_ = math.cos(yaw), math.sin(yaw)
            V = [(px + ln * c, py + ln * s_, z), (px - 0.5 * ln * s_, py + 0.5 * ln * c, z + 0.006), (px - ln * c, py - ln * s_, z),
                 (px + 0.5 * ln * s_, py - 0.5 * ln * c, z + 0.006)]
            lv.add((V, [(0, 1, 2, 3)]), mixc(AUT_PAL[pal][0], AUT_PAL[pal][1], rng.random()), vary=0.1)
    lv.build([M_VCOL])


def build_props():
    group("Props")
    rng = random.Random(64)
    # ---- stepping stones: porch -> dock
    dock0 = pond_pt(math.radians(172), 1.12)
    dock_d = (POND + Vector((-0.25, 0.0)) - dock0).normalized()
    sp = catmull([Vector((-0.55, -1.45)), Vector((0.6, -2.0)), Vector((1.5, -1.98)), dock0 - dock_d * 0.25], 6)
    mb = MB("SteppingStones")
    L = polylen(sp)
    n = int(L / 0.47)
    for i in range(n + 1):
        c, d = poly_at(sp, i / n)
        c = c + Vector((-d.y, d.x)) * rng.uniform(-0.07, 0.07)
        r = rng.uniform(0.17, 0.24)
        m = rng.choice((6, 7))
        V = [Vector((r * rng.uniform(0.85, 1.1) * math.cos(TAU * k / m), r * rng.uniform(0.85, 1.1) * math.sin(TAU * k / m), 0))
             for k in range(m)]
        V2 = [v + Vector((0, 0, 0.055)) for v in V]
        F = [tuple(range(m, 2 * m))] + [(k, (k + 1) % m, m + (k + 1) % m, m + k) for k in range(m)]
        mb.add((V + V2, F), jit(rng.choice(STONES), 0.07, rng), TR((c.x, c.y, height(c.x, c.y) - 0.012), (0, 0, rng.uniform(0, TAU))),
               vary=0.03)
    # ---- stone steps on the steep part of the windmill path
    P = PATHS[0]
    for t in (0.50, 0.57, 0.64, 0.71, 0.78, 0.85, 0.92):
        c, d = poly_at(P, t)
        bx(mb, TR((c.x, c.y, height(c.x, c.y) - 0.035), (0, 0, math.atan2(d.y, d.x))), (0.26, 0.56, 0.11), (0, 0, 0),
           rng.choice(STONES), base=True, vary=0.08)
    mb.build()
    # ---- dock
    mb = MB("Dock")
    ang = math.atan2(dock_d.y, dock_d.x)
    Md = TR((dock0.x, dock0.y, WATER), (0, 0, ang))
    DL, DWd, zd = 1.25, 0.52, 0.15
    for i in range(8):
        bx(mb, Md, (0.145, DWd + rng.uniform(-0.02, 0.03), 0.04), (-0.12 + 0.16 * i + 0.08, 0, zd), rng.choice(W_PLANK), base=True,
           vary=0.08)
    for sy in (-1, 1):
        bx(mb, Md, (DL + 0.1, 0.05, 0.07), (DL / 2 - 0.06, sy * 0.19, zd - 0.07), W_DARK, base=True, vary=0.05)
    for i, x in enumerate((0.05, 0.62, 1.18)):
        for sy in (-1, 1):
            tall = 0.42 if i == 2 else 0.05
            mb.add(P_frustum(0.045, 0.04, 0.5 + tall, 6, z0=-0.34), W_LOG[1], Md @ TR((x, sy * 0.27, zd)), vary=0.07)
    lantern(mb, Md @ TR((1.18, 0.27, zd + 0.60)), 1.2)
    mb.add(P_rod((1.18, 0.27, zd + 0.42), (1.18, 0.27, zd + 0.60), 0.012, 4), METAL, Md)
    mb.build([M_VCOL, M_GLOW])
    add_light("DockLantern", "POINT", 26, (1.0, 0.56, 0.22), loc=Md @ Vector((1.18, 0.20, zd + 0.38)), size=0.05)
    # ---- lantern post at the fork of the path
    mb = MB("LanternPost")
    lp = Vector((-1.62, -1.72))
    Ml = TR((lp.x, lp.y, height(lp.x, lp.y) - 0.05), (0, 0, math.radians(25)))
    bx(mb, Ml, (0.08, 0.08, 1.42), (0, 0, 0), W_DARK, base=True, vary=0.04)
    bx(mb, Ml, (0.40, 0.055, 0.055), (0.15, 0, 1.36), W_DARK, vary=0.04)
    mb.add(P_rod((0.0, 0, 1.08), (0.24, 0, 1.34), 0.022, 4), W_DARK, Ml)
    lantern(mb, Ml @ TR((0.30, 0, 1.33)), 1.7)
    mb.build([M_VCOL, M_GLOW])
    add_light("PostLantern", "POINT", 70, (1.0, 0.56, 0.22), loc=Ml @ Vector((0.30, -0.12, 1.02)), size=0.07)
    # ---- fences: along the front-left rim, and around the garden
    mb = MB("Fence_Rim")
    pts = []
    a = math.radians(193)
    while a < math.radians(253):
        r = redge(a) - 0.42
        pts.append(Vector((r * math.cos(a), r * math.sin(a))))
        a += 0.74 / r
    fence(mb, pts, rng=rng)
    mb.build()
    mb = MB("Fence_Garden")
    gx, gy, hx, hy = GARDEN
    cs = [(gx - hx, gy - hy), (gx + hx, gy - hy), (gx + hx, gy + hy), (gx - hx, gy + hy)]
    for k in range(4):
        (x0, y0), (x1, y1) = cs[k], cs[(k + 1) % 4]
        Ls = math.hypot(x1 - x0, y1 - y0)
        ns = max(2, int(round(Ls / 0.55)))
        seg = [cab_w(lerp(x0, x1, i / ns), lerp(y0, y1, i / ns)).to_2d() for i in range(ns + 1)]
        if k == 3:                                           # gate gap on the cabin side
            fence(mb, seg[:2], h=0.36, rails=(0.14, 0.27), post=0.06, rng=rng)
            fence(mb, seg[-2:], h=0.36, rails=(0.14, 0.27), post=0.06, rng=rng)
        else:
            fence(mb, seg, h=0.36, rails=(0.14, 0.27), post=0.06, rng=rng)
    mb.build()
    # ---- vegetable garden
    mb = MB("Garden")
    for i, lx in enumerate((gx - 0.6, gx - 0.2, gx + 0.2, gx + 0.6)):
        c = cab_w(lx, gy)
        Mg = TR((c.x, c.y, height(c.x, c.y) - 0.01), (0, 0, CAB_ROT))
        bx(mb, Mg, (0.27, 2 * hy - 0.3, 0.08), (0, 0, 0), srgb("#6b4a33"), base=True, vary=0.06)
        for j in range(6):
            ly = (j - 2.5) * (2 * hy - 0.5) / 5.5 + rng.uniform(-0.03, 0.03)
            if i == 0:                                       # cabbages
                mb.add(P_blob(0.10, 1, j, 0.2, (1, 1, 0.75)), srgb("#6fae5a"), Mg @ TR((0, ly, 0.13)), vary=0.1, mi=1)
            elif i == 1 and j % 2 == 0:                      # pumpkins
                r_ = rng.uniform(0.13, 0.17)
                mb.add(P_blob(r_, 2, j + 3, 0.08, (1, 1, 0.72)), srgb("#e8862a"), Mg @ TR((rng.uniform(-0.03, 0.03), ly, 0.08 + r_ * 0.6)),
                       vary=0.07)
                mb.add(P_frustum(0.02, 0.012, 0.06, 4, z0=0.08 + r_ * 1.25), srgb("#4f7a2f"), Mg @ TR((0, ly, 0)))
            elif i == 1:
                q_ = cab_w(lx, gy + ly)
                tuft(mb, q_.x, q_.y, 0.8, rng, srgb("#5f9a3a"))
            elif i == 2:                                     # leafy tops
                for q in range(3):
                    a_ = rng.uniform(0, TAU)
                    V = [(0.03 * math.cos(a_ + w_), ly + 0.03 * math.sin(a_ + w_), 0.07) for w_ in (0, 2.1, 4.2)]
                    V.append((0.07 * math.cos(a_), ly + 0.07 * math.sin(a_), 0.07 + rng.uniform(0.14, 0.22)))
                    mb.add((V, [(0, 1, 3), (1, 2, 3), (2, 0, 3)]), srgb("#7cb650"), Mg, vary=0.12, mi=1)
            else:                                            # sunflowers
                h_ = rng.uniform(0.55, 0.78)
                mb.add(P_rod((0, ly, 0.06), (0.02, ly, h_), 0.017, 4), srgb("#5a8f3a"), Mg)
                Mh = Mg @ TR((0.02, ly, h_ + 0.04)) @ align_z(Matrix.Rotation(-CAB_ROT, 4, "Z") @ Vector((SUN_VEC.x, SUN_VEC.y, 0.5)))
                mb.add(P_frustum(0.10, 0.085, 0.03, 8), srgb("#f5c22a"), Mh, vary=0.08)
                mb.add(P_frustum(0.045, 0.04, 0.045, 6), srgb("#5a3a22"), Mh)
                for q in (0.25, 0.5):
                    a_ = rng.uniform(0, TAU)
                    V = [(0, ly, h_ * q), (0.13 * math.cos(a_), ly + 0.13 * math.sin(a_), h_ * q + 0.05),
                         (0.07 * math.cos(a_ + 0.6), ly + 0.07 * math.sin(a_ + 0.6), h_ * q - 0.02)]
                    mb.add((V, [(0, 1, 2)]), srgb("#5f9a3a"), Mg, vary=0.1, mi=1)
    mb.build([M_VCOL, M_FOLIAGE])
    # ---- well (right-back of the island)
    mb = MB("Well")
    wp = Vector((3.7, 3.85))
    Mw = TR((wp.x, wp.y, height(wp.x, wp.y) - 0.05), (0, 0, math.radians(-20)))
    for row in range(2):
        for i in range(9):
            a_ = TAU * (i + 0.5 * row) / 9
            bx(mb, Mw, (0.24, 0.15, 0.21), (0.36 * math.cos(a_), 0.36 * math.sin(a_), 0.02 + 0.2 * row), rng.choice(STONES),
               (0, 0, a_ + PI / 2), True, vary=0.09)
    mb.add(P_frustum(0.30, 0.30, 0.02, 9, z0=0.20), C["w_deep"], Mw)
    for sx in (-1, 1):
        bx(mb, Mw, (0.07, 0.07, 1.0), (sx * 0.40, 0, 0.3), W_DARK, base=True, vary=0.05)
    mb.add(P_rod((-0.46, 0, 1.0), (0.50, 0, 1.0), 0.03, 6), W_PLANK[0], Mw)
    mb.add(P_rod((0.50, 0, 1.0), (0.50, 0, 0.86), 0.015, 4), METAL, Mw)
    mb.add(P_rod((0.0, 0, 1.0), (0.0, 0, 0.62), 0.008, 3), METAL, Mw)
    mb.add(P_frustum(0.075, 0.095, 0.14, 7, z0=0.48), W_PLANK[1], Mw, vary=0.06)
    for sgn in (-1, 1):
        tw = math.radians(38)
        ds, nr = Vector((0, sgn * math.cos(tw), -math.sin(tw))), Vector((0, sgn * math.sin(tw), math.cos(tw)))
        for k in range(3):
            for xx in (-0.36, 0.0, 0.36):
                cen = Vector((xx + rng.uniform(-0.01, 0.01), 0, 1.52)) + ds * (0.07 + k * 0.155) + nr * 0.02
                mb.add(P_box(0.34, 0.20, 0.04), jit(rng.choice(W_ROOF), 0.08, rng), Mw @ Matrix.Translation(cen) @ basis((1, 0, 0), ds, nr),
                       vary=0.012)
    bx(mb, Mw, (1.14, 0.07, 0.06), (0, 0, 1.53), srgb("#7b2f22"), vary=0.05)
    mb.build()
    # ---- bench on the promontory, looking out
    mb = MB("Bench")
    bp = Vector((-3.55, -3.55))
    out = math.atan2(bp.y, bp.x)
    Mbn = TR((bp.x, bp.y, height(bp.x, bp.y) - 0.02), (0, 0, out - PI / 2))
    for k in range(3):
        bx(mb, Mbn, (0.9, 0.095, 0.03), (0, -0.12 + 0.105 * k, 0.26), rng.choice(W_PLANK), base=True, vary=0.07)
    for k in range(2):
        bx(mb, Mbn, (0.9, 0.03, 0.10), (0, -0.20, 0.36 + 0.13 * k), rng.choice(W_PLANK), (math.radians(8), 0, 0), True, vary=0.07)
    for sx in (-1, 1):
        bx(mb, Mbn, (0.05, 0.36, 0.26), (sx * 0.36, -0.03, 0), W_DARK, base=True, vary=0.05)
        bx(mb, Mbn, (0.05, 0.05, 0.62), (sx * 0.36, -0.20, 0), W_DARK, (math.radians(8), 0, 0), True, vary=0.05)
    mb.build()


def build_plants():
    group("Plants")
    rng = random.Random(73)
    # ---- rocks: around the pond, along the stream, at the waterfall lip, a few on the lawn
    mb = MB("Rocks")
    for i, (a, q, r) in enumerate([(35, 1.16, 0.26), (62, 1.20, 0.17), (118, 1.17, 0.22), (215, 1.2, 0.2), (300, 1.22, 0.24),
                                   (330, 1.15, 0.15), (5, 1.2, 0.19)]):
        p = pond_pt(math.radians(a), q)
        rock(mb, p.x, p.y, r, 400 + i)
    for i, (t, off, r) in enumerate([(0.3, 0.42, 0.16), (0.52, -0.44, 0.2), (0.74, 0.43, 0.14), (0.97, 0.36, 0.2), (0.96, -0.38, 0.24)]):
        c, d = poly_at(STREAM_PTS, t)
        p = c + Vector((-d.y, d.x)) * off
        rock(mb, p.x, p.y, r, 420 + i)
    c, d = poly_at(STREAM_PTS, 0.62)
    rock(mb, c.x, c.y, 0.11, 431, z=stream_z(0.62) - 0.03)
    for i, (x, y, r) in enumerate([(-2.6, -3.6, 0.22), (0.9, -4.3, 0.26), (1.25, -4.45, 0.14), (-0.2, 0.4, 0.16), (-4.6, 2.9, 0.3),
                                   (4.6, 4.2, 0.42), (4.25, 4.55, 0.2), (2.6, 4.3, 0.18), (-5.6, -1.9, 0.24),
                                   (6.2, -1.0, 0.2), (-1.0, -4.6, 0.17), (2.3, -3.3, 0.15)]):
        rock(mb, x, y, r, 440 + i)
    mb.build([M_ROCK])
    # ---- bushes
    mb = MB("Bushes")
    red = srgb("#d83a2e")
    spots = [(2.75, 3.6, 0.9, None), (3.3, 3.1, 0.75, red), (4.9, 3.3, 0.85, None), (5.4, 2.6, 0.7, None), (0.0, 0.85, 0.7, None),
             (-0.55, 1.5, 0.85, red), (-3.0, -0.9, 0.7, None), (-4.6, -1.7, 0.8, None), (1.9, -3.6, 0.7, None),
             (-1.9, -3.9, 0.75, red), (5.7, -1.6, 0.7, None), (-2.3, 2.9, 0.7, None), (-6.0, 0.9, 0.7, None),
             (0.3, -4.9, 0.65, None), (4.6, -4.0, 0.6, None), (5.5, 2.95, 0.6, red)]
    for i, (x, y, s, b) in enumerate(spots):
        bush(mb, x, y, s, 500 + i, b)
    mb.build([M_VCOL, M_FOLIAGE])
    # ---- flowers in deliberate patches
    mb = MB("Flowers")
    patches = [(-0.1, -2.9, 0.85, 22, (0, 1)), (-2.6, -1.65, 0.7, 18, (2, 3)), (1.5, -3.0, 0.7, 16, (1, 3)),
               (-4.3, -3.5, 0.6, 14, (4, 1)), (-1.3, 0.2, 0.6, 14, (0, 5)), (2.2, -4.4, 0.55, 12, (0, 2)),
               (5.0, -0.2, 0.55, 12, (2, 0)), (-3.9, 0.55, 0.6, 14, (0, 5)), (3.9, 2.9, 0.75, 16, (1, 4)),
               (-5.3, -0.2, 0.5, 10, (3, 0)), (0.1, -4.2, 0.6, 12, (4, 0)), (-2.9, -4.3, 0.5, 10, (1, 0)),
               (5.2, -2.2, 0.45, 9, (3, 1)), (-1.5, -2.6, 0.5, 10, (5, 0))]
    for cx, cy, R, n, ci in patches:
        k = 0
        for _ in range(n * 6):
            if k >= n:
                break
            a, d = rng.uniform(0, TAU), R * math.sqrt(rng.random())
            p = Vector((cx + d * math.cos(a), cy + d * math.sin(a)))
            if lawn_ok(p, 0.2, 0.06):
                flower(mb, p.x, p.y, FLOWERS[rng.choice(ci)], rng)
                k += 1
    mb.build([M_VCOL])
    # ---- grass tufts: clustered near things, sparse on the open lawn
    mb = MB("GrassTufts")
    cols = [srgb("#8cc04a"), srgb("#a6cf5a"), srgb("#79b240"), srgb("#b9c45a")]
    anchors = [(p, 0.5) for p, a in [(Vector((x, y)), 0) for x, y, s, b in spots]]
    anchors += [(pond_pt(TAU * i / 14, 1.28), 0.35) for i in range(14)]
    anchors += [(poly_at(PATHS[0], t)[0], 0.45) for t in (0.1, 0.25, 0.4, 0.55, 0.7, 0.9)]
    anchors += [(poly_at(PATHS[1], t)[0], 0.45) for t in (0.3, 0.6, 0.95)]
    anchors += [(Vector((x, y)), 0.5) for x, y in ((-2.6, -3.6), (0.9, -4.3), (-0.2, 0.4), (4.6, 4.2), (-5.6, -1.9), (2.3, -3.3))]
    n = 0
    for p0, R in anchors:
        for _ in range(4):
            a, d = rng.uniform(0, TAU), R * (0.4 + 0.6 * rng.random())
            p = p0 + Vector((d * math.cos(a), d * math.sin(a)))
            if lawn_ok(p, 0.15, 0.03):
                tuft(mb, p.x, p.y, rng.uniform(0.8, 1.25), rng, rng.choice(cols))
                n += 1
    for _ in range(900):
        if n > 260:
            break
        a, d = rng.uniform(0, TAU), math.sqrt(rng.random())
        p = Vector((RX * d * math.cos(a), RY * d * math.sin(a)))
        if lawn_ok(p, 0.18, 0.05):
            tuft(mb, p.x, p.y, rng.uniform(0.7, 1.15), rng, rng.choice(cols))
            n += 1
    mb.build([M_FOLIAGE])
    # ---- pond life: lily pads, reeds
    mb = MB("LilyPads")
    pads = [(0.55, 0.38, 0.15), (0.78, 0.22, 0.11), (0.42, 0.6, 0.10), (0.9, 0.5, 0.13), (-0.15, -0.55, 0.14),
            (0.12, -0.68, 0.10), (-0.42, -0.42, 0.11), (0.72, -0.35, 0.12), (0.3, 0.1, 0.09), (-0.75, 0.42, 0.12)]
    for i, (dx, dy, r) in enumerate(pads):
        m, rot = 9, rng.uniform(0, TAU)
        V = [Vector((0, 0, 0.004))] + [Vector((r * math.cos(rot + TAU * k / m), r * math.sin(rot + TAU * k / m), 0)) for k in range(m)]
        F = [(0, 1 + k, 1 + (k + 1)) for k in range(m - 1)]                # one wedge left out = the notch
        mb.add((V, F), rng.choice((srgb("#4c8f3f"), srgb("#6fae4e"), srgb("#5a9e46"))), TR((POND.x + dx, POND.y + dy, WATER + 0.012)),
               vary=0.08)
        if i in (0, 4, 7):
            for k in range(6):
                a_ = TAU * k / 6
                V = [(0.02 * math.cos(a_ - 0.5), 0.02 * math.sin(a_ - 0.5), 0.01), (0.02 * math.cos(a_ + 0.5), 0.02 * math.sin(a_ + 0.5), 0.01),
                     (0.075 * math.cos(a_), 0.075 * math.sin(a_), 0.055)]
                mb.add((V, [(0, 1, 2)]), srgb("#f6b6cf"), TR((POND.x + dx, POND.y + dy, WATER + 0.02)), vary=0.06)
            mb.add(P_ico(0.022, 1), srgb("#f4c93c"), TR((POND.x + dx, POND.y + dy, WATER + 0.045)))
    mb.build([M_VCOL])
    mb = MB("Reeds")
    for a0, cnt in ((78, 9), (48, 6), (255, 6)):
        for k in range(cnt):
            p = pond_pt(math.radians(a0 + rng.uniform(-9, 9)), rng.uniform(0.88, 1.04))
            h = rng.uniform(0.42, 0.78)
            lx, ly = rng.uniform(-0.07, 0.07), rng.uniform(-0.07, 0.07)
            mb.add(P_tube([(p.x, p.y, WATER - 0.05), (p.x + lx * 0.4, p.y + ly * 0.4, WATER + h * 0.6), (p.x + lx, p.y + ly, WATER + h)],
                          [0.016, 0.012, 0.004], 3, cap=False), srgb("#6a9a3e"), vary=0.12, mi=1)
            if k % 2 == 0:
                mb.add(P_frustum(0.027, 0.022, 0.13, 5, z0=WATER + h * 0.72), srgb("#5a3a22"), TR((p.x + lx * 0.6, p.y + ly * 0.6, 0)), vary=0.08)
    mb.build([M_VCOL, M_FOLIAGE])


# =====================================================================================
#  underside dressing: vines, roots, crystals, boulders, floating debris
# =====================================================================================
def build_underside(body, rim):
    group("Underside")
    rng = random.Random(88)
    surf = body["surf"]
    # ---- vines hanging from the turf edge, in small bunches on the side the camera sees
    mb = MB("Vines")
    leafc = [srgb("#86c452"), srgb("#6fb146"), srgb("#a5d463")]
    for a_deg, cnt in ((197, 2), (211, 1), (231, 3), (249, 2), (262, 1), (277, 3), (296, 2), (318, 1), (331, 3), (346, 2), (5, 2)):
        for j in range(cnt):
            a = math.radians(a_deg + (j - (cnt - 1) / 2) * 2.4 + rng.uniform(-0.8, 0.8))
            if abs(angdiff(a, STREAM_A)) < 0.15:
                continue
            r = redge(a) + 0.08
            u = Vector((math.cos(a), math.sin(a), 0))
            tv = Vector((-math.sin(a), math.cos(a), 0))
            p0 = Vector((r * math.cos(a), r * math.sin(a), height(redge(a) * math.cos(a), redge(a) * math.sin(a)) - 0.12))
            L = rng.uniform(1.2, 2.3) if j == 0 else rng.uniform(0.5, 1.3)
            n = max(3, int(L / 0.18))
            P, R = [], []
            ph = rng.uniform(0, TAU)
            for i in range(n + 1):
                t = i / n
                P.append(p0 + Vector((0, 0, -L * t)) + tv * (0.07 * math.sin(ph + 5 * t)) + u * (0.03 - 0.08 * t))
                R.append(0.045 * (1 - 0.65 * t))
            mb.add(P_tube(P, R, 4), jit(srgb("#5f9a3c"), 0.1, rng), vary=0.06)
            for i in range(1, n + 1):
                side = 1 if i % 2 else -1
                c = P[i]
                d = (tv * side + u * 0.6 + Vector((0, 0, -0.5))).normalized()
                w = Vector((0, 0, 1)).cross(d).normalized()
                ln = rng.uniform(0.15, 0.21)
                V = [c, c + d * ln * 0.5 + w * 0.06, c + d * ln, c + d * ln * 0.5 - w * 0.06]
                mb.add((V, [(0, 1, 2, 3)]), rng.choice(leafc), vary=0.1, mi=1)
    mb.build([M_VCOL, M_FOLIAGE])
    # ---- roots poking out of the rock
    mb = MB("Roots")
    for a_deg, s in ((200, 0.10), (222, 0.22), (243, 0.07), (256, 0.36), (272, 0.16), (291, 0.30), (306, 0.09), (322, 0.24),
                     (343, 0.14), (186, 0.3), (15, 0.2), (270, 0.52), (230, 0.6), (310, 0.56)):
        a = math.radians(a_deg + rng.uniform(-3, 3))
        p0 = surf(a, s)
        u = Vector((math.cos(a), math.sin(a), 0))
        tv = Vector((-math.sin(a), math.cos(a), 0))
        L = rng.uniform(0.7, 1.8)
        n = 6
        ph = rng.uniform(0, TAU)
        P, R = [p0 - u * 0.15], [0.10]
        for i in range(1, n + 1):
            t = i / n
            P.append(p0 + u * (0.24 * math.sin(PI * min(1.0, t * 1.4)) ** 0.7 + 0.04) + Vector((0, 0, -L * t ** 1.2))
                     + tv * 0.08 * math.sin(ph + 4 * t))
            R.append(0.10 * (1 - t) ** 0.8 + 0.015)
        c = jit(rng.choice((srgb("#8a6644"), srgb("#765338"), srgb("#9a7650"))), 0.06, rng)
        mb.add(P_tube(P, R, 5), c, vary=0.05)
        k = rng.randint(2, 4)
        sd = rng.choice((-1, 1))
        Q = [P[k], P[k] + tv * sd * 0.16 + Vector((0, 0, -0.22)), P[k] + tv * sd * 0.22 + Vector((0, 0, -0.6))]
        mb.add(P_tube(Q, [R[k] * 0.6, 0.028, 0.01], 4), c, vary=0.05)
    mb.build([M_ROCK])
    # ---- boulders bedded in the strata
    mb = MB("Boulders")
    for i, (a_deg, s, r) in enumerate(((214, 0.17, 0.66), (252, 0.34, 0.78), (301, 0.44, 0.7), (333, 0.20, 0.62), (190, 0.42, 0.6))):
        a = math.radians(a_deg)
        p = surf(a, s) - Vector((math.cos(a), math.sin(a), 0)) * r * 0.4
        rock(mb, p.x, p.y, r, 600 + i, z=p.z - r * 0.3, scale=(1.0, 0.9, 0.8), light=srgb("#cdbda6"), dark=srgb("#8a7e74"))
    mb.build([M_ROCK])
    # ---- glowing crystals in the shaded front
    mb = MB("Crystals")
    for ci, (a_deg, s, cnt) in enumerate(((205, 0.45, 5), (263, 0.63, 6), (319, 0.40, 5))):
        a = math.radians(a_deg)
        p = surf(a, s)
        u = Vector((math.cos(a), math.sin(a), 0))
        for k in range(cnt):
            d = (u * rng.uniform(0.5, 1.0) + Vector((rng.uniform(-0.55, 0.55), rng.uniform(-0.55, 0.55), rng.uniform(-1.1, -0.25)))).normalized()
            L, r = rng.uniform(0.5, 1.1) * (1.25 if k == 0 else 1.0), rng.uniform(0.10, 0.18)
            Mc = Matrix.Translation(p - u * 0.15 + Vector((rng.uniform(-0.22, 0.22), rng.uniform(-0.22, 0.22), rng.uniform(-0.15, 0.15)))) @ align_z(d)
            g = rng.uniform(0.8, 1.5)
            cc = (0.03 * g, 0.55 * g, 0.75 * g)
            mb.add(P_frustum(r, r * 0.86, L * 0.72, 6, cap_top=False), cc, Mc, mi=1, vary=0.30)
            mb.add(P_frustum(r * 0.86, 0.0, L * 0.28, 6, z0=L * 0.72, cap_bottom=False), mulc(cc, 1.4), Mc, mi=1, vary=0.30)
        add_light(f"CrystalGlow{ci}", "POINT", 45, (0.20, 0.85, 1.0), loc=p + u * 0.5 + Vector((0, 0, -0.35)), size=0.25)
    mb.build([M_VCOL, M_GLOW])
    # ---- small rocks drifting below
    mb = MB("Debris")
    for i, (x, y, z, r) in enumerate(((-4.6, -2.6, -4.0, 0.3), (-5.3, -1.4, -5.1, 0.17), (-3.4, -3.2, -6.0, 0.22), (4.9, -1.8, -4.6, 0.26),
                                      (5.6, -0.4, -3.4, 0.15), (3.1, -2.6, -6.6, 0.2), (-5.9, 0.6, -3.2, 0.2), (4.2, 1.5, -6.0, 0.18))):
        rock(mb, x, y, r, 700 + i, z=z, scale=(1.0, 0.85, 0.9), light=srgb("#b8a188"), dark=srgb("#6a5a56"))
    mb.build([M_ROCK])


# =====================================================================================
#  islets (near, detailed) and far islands (hazy silhouettes)
# =====================================================================================
def islet(name, cx, cy, cz, R, depth, seed, far=None, tip=(0.0, 0.0)):
    rng = random.Random(seed)
    k = R / 1.5

    def rf(a):
        return R * (1 + 0.16 * n3(math.cos(a) * 1.3 + seed, math.sin(a) * 1.3, 0.7) + 0.05 * n3(math.cos(a) * 3.1, math.sin(a) * 3.1 + seed, 1.9))

    def hf(x, y):
        d = math.hypot(x - cx, y - cy) / R
        return cz + 0.16 * R * (1 - d * d) + 0.05 * R * n3(x * 1.1 / k + seed, y * 1.1 / k, 0.3)

    n = max(10, int(TAU * R / (0.42 * max(1.0, k))))
    angs = [TAU * i / n for i in range(n)]
    rim = [(Vector((cx + rf(a) * math.cos(a), cy + rf(a) * math.sin(a), hf(cx + rf(a) * math.cos(a), cy + rf(a) * math.sin(a)))), a)
           for a in angs]
    V = [p.copy() for p, a in rim]
    rings, ring_a = [list(range(n))], [angs]
    for q in (0.62, 0.30):
        m = max(5, int(n * q))
        ph = rng.uniform(0, TAU)
        idx, aa = [], []
        for i in range(m):
            a = ph + TAU * i / m
            x, y = cx + rf(a) * q * math.cos(a), cy + rf(a) * q * math.sin(a)
            idx.append(len(V))
            aa.append(a)
            V.append(Vector((x, y, hf(x, y))))
        rings.append(idx)
        ring_a.append(aa)
    tris = []
    for i in range(len(rings) - 1):
        tris += zipper(rings[i], ring_a[i], rings[i + 1], ring_a[i + 1])
    c = len(V)
    V.append(Vector((cx, cy, hf(cx, cy))))
    last = rings[-1]
    tris += [(last[i], last[(i + 1) % len(last)], c) for i in range(len(last))]
    mb = MB(name)

    def gcol(c_, n_):
        t = 0.5 + 0.5 * n3(c_.x * 0.8 + seed, c_.y * 0.8, 2.2)
        return mixc(mixc(C["g_dark"], C["g_mid"], smooth(0.2, 0.5, t)), C["g_light"], smooth(0.5, 0.85, t))

    mb.add((V, tris), gcol, vary=0.08)
    body = rock_body(mb, cx, cy, rf, lambda a: hf(cx + rf(a) * math.cos(a), cy + rf(a) * math.sin(a)), depth, tip, seed,
                     facet=0.40 * max(1.0, k), rim=rim, inset=0.12 * k, soil_d=0.36 * k, strata=(0.32 * k, 0.6 * k),
                     drip=(0.12 * k, 0.32 * k), ledge=(0.03 * k, 0.09 * k), bulge=0.2,
                     taper=((0, 1.0), (0.2, 0.88), (0.5, 0.60), (0.8, 0.28), (1.0, 0.0)))
    ob = mb.build([M_FAR] if far else [M_ROCK])
    if far:
        ob.color = (far[0][0], far[0][1], far[0][2], far[1])
    return ob, hf, body, rf


def rope_bridge(name, p0, p1, width=0.44, sag=0.30, n=15):
    """plank-and-rope footbridge between two points on the ground."""
    mb = MB(name)
    rng = random.Random(55)
    d = p1 - p0
    dh = Vector((d.x, d.y, 0)).normalized()
    lat = Vector((-dh.y, dh.x, 0))

    def pt(t, s=sag):
        return p0.lerp(p1, t) + Vector((0, 0, -4 * s * t * (1 - t)))

    for i in range(n):
        t = (i + 0.5) / n
        c = pt(t)
        tan = (pt(t + 0.01) - pt(t - 0.01)).normalized()
        M = Matrix.Translation(c) @ basis(tan, lat, tan.cross(lat)) @ TR(rot=(rng.uniform(-0.05, 0.05), 0, rng.uniform(-0.04, 0.04)))
        bx(mb, M, (d.length / n * 0.78, width, 0.035), (0, 0, 0), rng.choice(W_PLANK), vary=0.09)
    rope = srgb("#b89a6a")
    H = 0.42
    for sgn in (-1, 1):
        o = lat * (sgn * width / 2)
        mb.add(P_tube([pt(i / 12) + o + Vector((0, 0, -0.02)) for i in range(13)], 0.02, 4), rope, vary=0.06)
        rail = [pt(i / 12, sag * 0.8) + o + Vector((0, 0, H)) for i in range(13)]
        mb.add(P_tube(rail, 0.02, 4), rope, vary=0.06)
        for i in range(1, 12, 2):
            mb.add(P_rod(pt(i / 12) + o, rail[i], 0.011, 3), rope)
        for q in (p0, p1):
            bx(mb, TR((q.x + o.x, q.y + o.y, q.z - 0.12)), (0.08, 0.08, H + 0.22), (0, 0, 0), W_DARK, base=True, vary=0.05)
    return mb.build()


def build_islets():
    group("Islets")
    rng = random.Random(15)
    plants = MB("Islet_Plants")
    stones = MB("Islet_Rocks")
    specs = [("Islet_A", 10.6, 1.75, 0.05, 1.75, 3.2, 31), ("Islet_B", -10.3, -0.6, -3.3, 1.35, 2.5, 32),
             ("Islet_C", 8.6, -4.6, -4.4, 0.9, 1.7, 33)]
    for name, cx, cy, cz, R, depth, seed in specs:
        ob, hf, body, rf = islet(name, cx, cy, cz, R, depth, seed, tip=(rng.uniform(-0.2, 0.2) * R, rng.uniform(-0.2, 0.2) * R))
        if name == "Islet_A":
            a_b = math.radians(14)                           # a rope bridge links it to the main island
            r0 = redge(a_b) - 0.32
            p0 = Vector((r0 * math.cos(a_b), r0 * math.sin(a_b), 0.0))
            p0.z = height(p0.x, p0.y) + 0.03
            ang = math.atan2(p0.y - cy, p0.x - cx)
            r1 = rf(ang) * 0.84
            p1 = Vector((cx + r1 * math.cos(ang), cy + r1 * math.sin(ang), 0.0))
            p1.z = hf(p1.x, p1.y) + 0.03
            rope_bridge("RopeBridge", p0, p1)
            autumn_tree("Islet_A_Tree", cx - 0.25, cy + 0.2, 0.82, 311, 1, z=hf(cx - 0.25, cy + 0.2))
            pine("Islet_A_Pine", cx + 0.75, cy - 0.3, 0.42, 231, z=hf(cx + 0.75, cy - 0.3), check=False)
            bush(plants, cx - 0.9, cy - 0.5, 0.7, 520, z=hf(cx - 0.9, cy - 0.5))
            rock(stones, cx + 0.3, cy - 0.9, 0.2, 801, z=hf(cx + 0.3, cy - 0.9))
        elif name == "Islet_B":
            pine("Islet_B_Pine0", cx - 0.2, cy + 0.2, 0.62, 232, z=hf(cx - 0.2, cy + 0.2), check=False)
            pine("Islet_B_Pine1", cx + 0.55, cy - 0.15, 0.4, 233, z=hf(cx + 0.55, cy - 0.15), check=False)
            rock(stones, cx - 0.7, cy - 0.5, 0.2, 802, z=hf(cx - 0.7, cy - 0.5))
        elif name == "Islet_C":
            pine("Islet_C_Pine", cx, cy + 0.1, 0.4, 234, z=hf(cx, cy + 0.1), check=False)
            bush(plants, cx + 0.4, cy - 0.3, 0.5, 521, z=hf(cx + 0.4, cy - 0.3))
        for k in range(int(5 * R)):
            a, d = rng.uniform(0, TAU), R * 0.75 * math.sqrt(rng.random())
            x, y = cx + d * math.cos(a), cy + d * math.sin(a)
            z = hf(x, y) - 0.012
            for j in range(3):
                a_ = rng.uniform(0, TAU)
                V = [(x + 0.04 * math.cos(a_ + w), y + 0.04 * math.sin(a_ + w), z) for w in (0, 2.1, 4.2)]
                V.append((x + 0.07 * math.cos(a_), y + 0.07 * math.sin(a_), z + rng.uniform(0.16, 0.3)))
                plants.add((V, [(0, 1, 3), (1, 2, 3), (2, 0, 3)]), srgb("#8cc04a"), vary=0.12, mi=1)
        # a few roots under each islet
        for j in range(3):
            a = rng.uniform(0, TAU)
            p0 = body["surf"](a, rng.uniform(0.2, 0.6))
            L = rng.uniform(0.3, 0.8) * R / 1.5 + 0.2
            P = [p0, p0 + Vector((0.05 * math.cos(a), 0.05 * math.sin(a), -L * 0.5)), p0 + Vector((0, 0, -L))]
            stones.add(P_tube(P, [0.045, 0.028, 0.008], 4), srgb("#6a4a33"), vary=0.07)
    plants.build([M_VCOL, M_FOLIAGE])
    stones.build([M_ROCK])


def build_far():
    """distant islands, a second tiny balloon and birds: hazy, they sell the depth."""
    group("Far")
    rng = random.Random(123)
    # (u, v, distance, radius, depth, haze, seed)
    for i, (u, v, d, R, depth, fog, seed) in enumerate([(0.27, 0.585, 190.0, 7.5, 9.0, 0.62, 41), (0.955, 0.60, 135.0, 5.0, 6.5, 0.50, 42),
                                                        (0.055, 0.44, 250.0, 10.0, 12.0, 0.72, 43), (0.735, 0.615, 320.0, 9.0, 10.0, 0.78, 44)]):
        c = F2W(u, v, d)
        fc = mixc(sky_uv(u, v - 0.03), sky_uv(u, SUN_UV[1] - 0.02), 0.55)      # haze takes the colour of the horizon glow
        ob, hf, body, rf = islet(f"FarIsland_{i}", c.x, c.y, c.z, R, depth, seed, far=(fc, fog), tip=(rng.uniform(-0.2, 0.2) * R, 0.0))
        mb = MB(f"FarIsland_{i}_Trees")
        for k in range(9):
            a, dd = rng.uniform(0, TAU), R * 0.7 * math.sqrt(rng.random())
            x, y = c.x + dd * math.cos(a), c.y + dd * math.sin(a)
            s = R * rng.uniform(0.10, 0.2)
            mb.add(P_frustum(s * 0.9, 0.0, s * 3.2, 6), srgb("#2d5f3d"), TR((x, y, hf(x, y) - 0.05 * R)), vary=0.1)
        if i == 0:                                           # a little tower on the nearest far island
            mb.add(P_frustum(R * 0.09, R * 0.065, R * 0.55, 6), WW_D, TR((c.x + R * 0.2, c.y, hf(c.x + R * 0.2, c.y))), vary=0.05)
            mb.add(P_frustum(R * 0.10, 0.0, R * 0.16, 6, z0=R * 0.55), W_ROOF[1], TR((c.x + R * 0.2, c.y, hf(c.x + R * 0.2, c.y))))
        tob = mb.build([M_FAR])
        tob.color = (fc[0], fc[1], fc[2], fog * 0.92)
    # birds
    mb = MB("Birds")
    base = F2W(0.665, 0.845, 70.0)
    for k, (dx, dy) in enumerate(((0, 0), (-1.0, -0.5), (1.0, -0.55), (-2.1, -1.05), (2.0, -1.2), (-3.2, -1.5), (3.3, -1.9))):
        c = base + _rt * dx * 1.1 + _up * dy * 0.9 + _fw * rng.uniform(-3, 3)
        w = rng.uniform(0.75, 1.0)
        flap = rng.uniform(0.12, 0.34)
        V = [c, c - _rt * w * 0.5 + _up * w * flap, c - _rt * w * 0.25 + _up * w * (flap * 0.75) - _up * 0.06 * w + _fw * 0.2,
             c + _rt * w * 0.5 + _up * w * flap, c + _rt * w * 0.25 + _up * w * (flap * 0.75) - _up * 0.06 * w + _fw * 0.2,
             c - _up * 0.07 * w + _fw * 0.3]
        mb.add((V, [(0, 1, 2), (0, 2, 5), (0, 4, 3), (0, 5, 4)]), srgb("#2a2238"))
    ob = mb.build([M_FAR])
    fc = sky_uv(0.665, 0.845)
    ob.color = (fc[0], fc[1], fc[2], 0.25)


# =====================================================================================
#  hot-air balloon
# =====================================================================================
def balloon(name, center, Rb, far=None, seed=1, pal=None):
    mb = MB(name)
    A_, B_, D_, T_ = pal or (srgb("#f3e2bd"), srgb("#d9502b"), srgb("#a8301f"), srgb("#2f7f86"))
    prof = [(0.0, Rb)] + [(Rb * math.sin(math.radians(a)), Rb * math.cos(math.radians(a))) for a in (18, 36, 54, 72, 90)]
    prof += [(Rb * (1 - 0.72 * s ** 1.6), -1.45 * Rb * s) for s in (0.2, 0.4, 0.6, 0.8, 1.0)]
    G = 12
    V = [Vector((0, 0, prof[0][1]))]
    for r, z in prof[1:]:
        V += [Vector((r * math.cos(TAU * g / G), r * math.sin(TAU * g / G), z)) for g in range(G)]
    F, cols = [], []
    rows = len(prof) - 1
    for g in range(G):
        F.append((0, 1 + g, 1 + (g + 1) % G))
        cols.append(D_)
    for row in range(1, rows):
        for g in range(G):
            a, b = 1 + (row - 1) * G + g, 1 + (row - 1) * G + (g + 1) % G
            c, d = 1 + row * G + g, 1 + row * G + (g + 1) % G
            F.append((a, c, d, b))
            if row == 1:
                col = D_
            elif row == rows - 1:
                col = T_
            elif row in (4, 5):
                col = A_ if (g + row) % 2 else B_            # chequer band round the middle
            else:
                col = B_ if g % 2 else A_
            cols.append(jit(col, 0.04, mb.rng))
    M = Matrix.Translation(center)
    mb.add((V, F), cols, M, mi=1)
    zm = -1.45 * Rb
    rm = Rb * 0.28
    mb.add(P_ring(rm * 0.9, rm * 1.06, 0.06 * Rb, G, z0=zm - 0.05 * Rb), D_, M)
    bw, bh, zb = 0.36 * Rb, 0.30 * Rb, zm - 0.98 * Rb
    wick = srgb("#8a6238")
    bx(mb, M, (bw, bw, bh), (0, 0, zb), wick, base=True, vary=0.05)
    bx(mb, M, (bw * 1.1, bw * 1.1, bh * 0.14), (0, 0, zb + bh), srgb("#5a3d24"), base=True, vary=0.04)
    for zz in (0.3, 0.62):
        bx(mb, M, (bw * 1.03, bw * 1.03, bh * 0.07), (0, 0, zb + bh * zz), srgb("#6f4c2c"), vary=0.04)
    for k in range(4):
        a = PI / 4 + k * PI / 2
        mb.add(P_rod((bw * 0.52 * math.cos(a) * 1.3, bw * 0.52 * math.sin(a) * 1.3, zb + bh), (rm * math.cos(a), rm * math.sin(a), zm),
                     0.014 * Rb, 4), METAL, M)
        mb.add(P_blob(0.06 * Rb, 1, k, 0.2, (1, 1, 1.25)), srgb("#c9b48a"), M @ TR((bw * 0.62 * math.cos(a + 0.5), bw * 0.62 * math.sin(a + 0.5), zb + bh * 0.45)),
               vary=0.06)
    bx(mb, M, (0.12 * Rb, 0.12 * Rb, 0.08 * Rb), (0, 0, zb + bh + 0.42 * Rb), METAL, vary=0)
    mb.add(P_frustum(0.045 * Rb, 0.0, 0.26 * Rb, 5, z0=zb + bh + 0.47 * Rb), (3.5, 1.5, 0.25), M, mi=2)
    ob = mb.build([M_FAR, M_FAR, M_FAR] if far else [M_VCOL, M_FABRIC, M_GLOW])
    if far:
        ob.color = (far[0][0], far[0][1], far[0][2], far[1])
    return ob, center + Vector((0, 0, zm - 0.25 * Rb))


def build_balloon():
    group("Balloon")
    c = F2W(0.12, 0.75, 46.0)
    ob, mouth = balloon("Balloon", c, 1.75)
    add_light("BalloonBurner", "POINT", 420, (1.0, 0.5, 0.15), loc=mouth, size=0.15)
    c2 = F2W(0.945, 0.875, 230.0)
    balloon("Balloon_Far", c2, 3.4, far=(sky_uv(0.945, 0.875), 0.55),
            pal=(srgb("#f3e2bd"), srgb("#3f8f9a"), srgb("#2f6f7a"), srgb("#d9502b")))


def build_smoke():
    group("Smoke")
    mb = MB("ChimneySmoke")
    top = CHIM_TOP[0]
    rng = random.Random(3)
    sizes = (0.09, 0.13, 0.19, 0.26, 0.34, 0.43, 0.52)
    for i, r in enumerate(sizes):
        t = i / (len(sizes) - 1)
        p = top + Vector((-1.75 * t ** 1.6 + rng.uniform(-0.05, 0.05), -0.3 * t ** 1.5, 0.10 + 2.15 * t ** 0.8 + rng.uniform(-0.04, 0.04)))
        mb.add(P_blob(r, 1 if i < 2 else 2, 40 + i, 0.22, (1.15, 1.0, 0.88)), (0.86, 0.80, 0.78, 0.95 - 0.6 * t),
               TR(p, (0, 0, rng.uniform(0, TAU))), vary=0.04)
    mb.build([M_SMOKE])


# =====================================================================================
#  lights, camera, render
# =====================================================================================
def add_light(name, kind, energy, color, loc=(0, 0, 0), target=None, size=1.0, direction=None):
    ld = bpy.data.lights.new(name, kind)
    ld.energy = energy
    ld.color = color
    if kind == "AREA":
        ld.size = size
    elif kind == "SUN":
        ld.angle = math.radians(size)
    else:
        ld.shadow_soft_size = size
    ob = bpy.data.objects.new(name, ld)
    ob.location = loc
    if direction is not None:
        ob.rotation_euler = Vector(direction).to_track_quat("-Z", "Y").to_euler()
    elif target is not None:
        ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    CUR[0].objects.link(ob)
    return ob


def build_rig():
    group("Rig")
    if A.debuglight:
        bg = next(n for n in SC.world.node_tree.nodes if n.bl_idname == "ShaderNodeBackground")
        for l in list(bg.inputs["Color"].links) + list(bg.inputs["Strength"].links):
            SC.world.node_tree.links.remove(l)
        bg.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1)
        bg.inputs["Strength"].default_value = 1.0
        add_light("DebugSun", "SUN", 2.5, (1, 1, 1), loc=(0, 0, 30), direction=(-0.4, 0.5, -0.75), size=5)
    else:
        _lights()
    cd = bpy.data.cameras.new("Cam")
    cd.lens = LENS
    cd.sensor_width = 36.0
    cd.clip_start, cd.clip_end = 0.5, 4000
    cam = bpy.data.objects.new("Cam", cd)
    cam.location = CAM_LOC
    cam.rotation_euler = (CAM_TGT - CAM_LOC).to_track_quat("-Z", "Y").to_euler()
    CUR[0].objects.link(cam)
    SC.camera = cam
    return cam


def _lights():
    # warm low key from the right-back; every fill is a soft directional lamp so near and far things get the same
    add_light("Sun", "SUN", 10.0, (1.0, 0.56, 0.28), loc=SUN_VEC * 30, direction=-SUN_VEC, size=2.5)
    add_light("SkyFill", "SUN", 0.6, (0.42, 0.58, 1.0), loc=(-15, -22, 14), direction=(0.5, 0.75, -0.45), size=80)
    add_light("UnderBounce", "SUN", 2.0, (1.0, 0.60, 0.48), loc=(8, -10, -18), direction=(-0.45, 0.5, 0.74), size=80)
    add_light("RimCool", "SUN", 0.7, (0.50, 0.62, 1.0), loc=(-24, 15, 9), direction=(0.8, -0.5, -0.3), size=20)


def setup_render():
    SC.render.engine = "CYCLES"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        SC.cycles.device = "GPU"
        print("[gpu] devices:", [(d.name, d.type, d.use) for d in prefs.devices], "scene device:", SC.cycles.device)
    except Exception as e:
        print("[gpu] could not enable GPU:", e)
    cy = SC.cycles
    cy.samples = FINAL_SAMPLES if A.final else A.samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.008 if A.final else 0.05
    cy.use_denoising = True
    try:
        cy.denoiser = "OPENIMAGEDENOISE"
        cy.denoising_use_gpu = True
    except Exception as e:
        print("[denoise]", e)
    cy.max_bounces, cy.diffuse_bounces, cy.glossy_bounces, cy.transmission_bounces = 8, 4, 4, 6
    cy.transparent_max_bounces = 12
    cy.sample_clamp_indirect = 8.0
    cy.caustics_reflective = cy.caustics_refractive = False
    r = SC.render
    r.resolution_x, r.resolution_y = 1920, 1080
    r.resolution_percentage = 100 if A.final else A.res
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.film_transparent = False
    if A.crop:
        x0, y0, x1, y1 = [float(x) for x in A.crop.split(",")]
        r.use_border, r.use_crop_to_border = True, True
        r.border_min_x, r.border_min_y, r.border_max_x, r.border_max_y = x0, y0, x1, y1
    # Khronos PBR Neutral keeps the saturated sunset colours (AgX washed them out to pastel); AgX is the fallback
    try:
        SC.view_settings.view_transform = "Khronos PBR Neutral"
        SC.view_settings.look = "None"
    except Exception:
        SC.view_settings.view_transform = "AgX"
        SC.view_settings.look = "AgX - Punchy"
    SC.view_settings.exposure = EXPOSURE


def setup_comp():
    """gentle bloom so the sun, windows and lanterns glow."""
    try:
        ng = bpy.data.node_groups.new("S2_Comp", "CompositorNodeTree")
        ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        rl = ng.nodes.new("CompositorNodeRLayers")
        gl = ng.nodes.new("CompositorNodeGlare")
        out = ng.nodes.new("NodeGroupOutput")
        for val in ("Bloom", "BLOOM", "Fog Glow"):
            try:
                gl.inputs["Type"].default_value = val
                break
            except Exception:
                continue
        gl.inputs["Threshold"].default_value = 1.4
        gl.inputs["Smoothness"].default_value = 0.5
        gl.inputs["Strength"].default_value = 0.4
        gl.inputs["Size"].default_value = 0.5
        ng.links.new(rl.outputs["Image"], gl.inputs["Image"])
        ng.links.new(gl.outputs["Image"], out.inputs[0])
        SC.compositing_node_group = ng
        SC.render.use_compositing = True
        print("[comp] glare type:", gl.inputs["Type"].default_value)
    except Exception as e:
        print("[comp] disabled:", e)


def frame_report():
    """where every mesh lands in the frame (exact, from its vertices); flags anything cut by the frame edge."""
    print("---- frame report (u, v from bottom-left) ----")
    for ob in SC.objects:
        if ob.type != "MESH":
            continue
        mw = ob.matrix_world
        us, vs = [], []
        step = max(1, len(ob.data.vertices) // 4000)
        for vtx in ob.data.vertices[::step]:
            u, v, zc = W2F(mw @ vtx.co)
            us.append(u)
            vs.append(v)
        flag = "  <-- CUT" if (min(us) < 0 or max(us) > 1 or min(vs) < 0 or max(vs) > 1) else ""
        print(f"  {ob.name:22s} u {min(us):5.2f}..{max(us):5.2f}   v {min(vs):5.2f}..{max(vs):5.2f}{flag}")


# =====================================================================================
#  build
# =====================================================================================
build_world()
BODY, RIM = build_island()
LIP = build_water()
build_clouds()
build_cabin()
build_windmill()
build_trees()
build_props()
build_plants()
build_underside(BODY, RIM)
build_islets()
build_far()
build_balloon()
build_smoke()
build_rig()
setup_render()
setup_comp()
bpy.context.view_layer.update()
frame_report()
print(f"[build] {len(SC.objects)} objects in {time.time() - T0:.1f}s")

if A.final or A.save:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "scene2_floating_island_v2.blend"), compress=True)
if A.exec:
    exec(A.exec)
if not A.norender:
    out = A.out or os.path.join(HERE, "S2_FloatingIsland_v2.png" if A.final else "_preview.png")
    SC.render.filepath = out
    t1 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"[render] {out} in {time.time() - t1:.1f}s")
    for i, lk in enumerate(x for x in A.look.split(",") if x):       # e.g. "Standard,AgX:AgX - Punchy"
        vt, _, look = lk.partition(":")
        try:
            SC.view_settings.view_transform = vt
            SC.view_settings.look = look or "None"
            bpy.data.images["Render Result"].save_render(out.replace(".png", f"_look{i}.png"), scene=SC)
            print("[look]", lk, "->", out.replace(".png", f"_look{i}.png"))
        except Exception as e:
            print("[look] failed", lk, e)
