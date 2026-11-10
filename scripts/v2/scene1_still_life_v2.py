"""Scene 1 v2 -- "simple, but not basic": lamp-lit still life on a round side table.

Builds the whole scene from an empty file (Blender 5.2, Cycles, Metal GPU).

  Final (build + save scene1_still_life_v2.blend + render S1_StillLife_v2.png, 1920x1080, 1024 samples, denoised):
      Blender -b --factory-startup -P scene1_still_life_v2.py
  Quick preview (half res, 48 samples, no .blend written):
      Blender -b --factory-startup -P scene1_still_life_v2.py -- --preview
  Build + save the .blend only:
      Blender -b --factory-startup -P scene1_still_life_v2.py -- --no-render

  Options after "--":
      --preview            half resolution, 48 samples, writes preview.png, does not save the .blend
      --no-render          skip the render
      --scale N            resolution percentage          --samples N   max samples
      --thresh X           adaptive-sampling noise threshold (final 0.008, preview 0.05)
      --cam NAME           QA camera instead of "main": top lamp head books mug plant fruit front edge side low
      --crop x0,y0,x1,y1   render only this part of the frame (fractions, origin bottom-left)
      --out PATH           output image path

  Everything is procedural: no image textures, no external assets.  Spine / cover / pencil lettering is typed with
  two macOS system fonts and converted to meshes (falls back to Blender's built-in font if they are missing).
"""
import bpy, bmesh, math, random, sys, os, time
from math import sin, cos, tan, pi, radians, degrees, sqrt, atan2, exp, acos
from mathutils import Vector, Matrix, Euler, Quaternion
from mathutils import noise as mnoise

T_START = time.time()
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
ARGV = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(name, default=None, cast=str):
    return cast(ARGV[ARGV.index(name) + 1]) if name in ARGV else default


PREVIEW = "--preview" in ARGV
NO_RENDER = "--no-render" in ARGV
SCALE = opt("--scale", 50 if PREVIEW else 100, int)
SAMPLES = opt("--samples", 48 if PREVIEW else 1024, int)
CAM = opt("--cam", "main")
CROP = opt("--crop", None)        # x0,y0,x1,y1 as fractions of the frame (origin bottom-left): QA crops
OUT = opt("--out", os.path.join(HERE, "preview.png" if PREVIEW else "S1_StillLife_v2.png"))
BLEND = os.path.join(HERE, "scene1_still_life_v2.blend")
SAVE = not PREVIEW

# ------------------------------------------------------------------ layout (metres, Z up)
TOP = 0.62                      # table top height
TABLE_R = 0.345
LAMP_XY = (-0.235, 0.030)
LAMP_YAW = radians(-12)         # direction the arm reaches
LAMP_AIM = (0.02, -0.20)        # point on the table the shade looks at
BOOKS_XY = (-0.035, -0.055)
MUG_XY = (0.090, -0.175)
PLANT_XY = (0.235, 0.110)
ORANGE_1 = (0.255, -0.105, 0.038)
ORANGE_2 = (0.215, -0.170, 0.036)
PENCIL_XY = (-0.135, -0.235)
PENCIL_YAW = radians(-8)
TABLE_ROT = radians(66)         # grain direction; the front edge shows end grain
CUP_RING_XY = (0.012, -0.262)   # an old mug ring in the finish

FONT_SERIF = "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"
FONT_SANS = "/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf"

# ------------------------------------------------------------------ start from an empty file
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.name = "S1_StillLife_v2"
COL = {"cur": scene.collection}
V = Vector


def group(name):
    c = bpy.data.collections.new(name)
    scene.collection.children.link(c)
    COL["cur"] = c
    return c


def link(ob, parent=None):
    COL["cur"].objects.link(ob)
    if parent is not None:
        ob.parent = parent
    return ob


def empty(name, loc=(0, 0, 0), rot=(0, 0, 0), parent=None):
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_size = 0.02
    ob.location = loc
    ob.rotation_euler = rot
    return link(ob, parent)


# =================================================================== node helpers
SOCK = bpy.types.NodeSocket


def c4(c):
    if isinstance(c, (int, float)):
        return (c, c, c, 1.0)
    c = tuple(c)
    return c if len(c) == 4 else c + (1.0,)


class NB:
    """Small shader-node builder: every method returns an output socket."""

    def __init__(self, nt):
        self.nt = nt

    def node(self, idn, **props):
        nd = self.nt.nodes.new(idn)
        for k, v in props.items():
            setattr(nd, k, v)
        return nd

    @staticmethod
    def find(coll, key):
        if isinstance(key, int):
            return coll[key]
        for s in coll:
            if s.identifier == key:
                return s
        for s in coll:
            if s.name == key:
                return s
        raise KeyError(key)

    def put(self, sock, val):
        if val is None:
            return
        if isinstance(val, SOCK):
            self.nt.links.new(val, sock)
            return
        if sock.type == "RGBA":
            val = c4(val)
        elif sock.type == "VECTOR" and isinstance(val, (int, float)):
            val = (val, val, val)
        sock.default_value = val

    def math(self, op, a, b=None, c=None, clamp=False):
        nd = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        self.put(nd.inputs[0], a)
        if b is not None:
            self.put(nd.inputs[1], b)
        if c is not None:
            self.put(nd.inputs[2], c)
        return nd.outputs[0]

    def add(self, a, b):
        return self.math("ADD", a, b)

    def sub(self, a, b):
        return self.math("SUBTRACT", a, b)

    def mul(self, a, b):
        return self.math("MULTIPLY", a, b)

    def madd(self, a, b, c):
        return self.math("MULTIPLY_ADD", a, b, c)

    def vmath(self, op, a, b=None, c=None, scale=None):
        nd = self.node("ShaderNodeVectorMath", operation=op)
        self.put(nd.inputs[0], a)
        if b is not None:
            self.put(nd.inputs[1], b)
        if c is not None:
            self.put(nd.inputs[2], c)
        if scale is not None:
            self.put(self.find(nd.inputs, "Scale"), scale)
        return nd.outputs[1] if op in ("LENGTH", "DISTANCE", "DOT_PRODUCT") else nd.outputs[0]

    def mix(self, fac, a, b, blend="MIX"):
        nd = self.node("ShaderNodeMix", data_type="RGBA", blend_type=blend)
        self.put(self.find(nd.inputs, "Factor_Float"), fac)
        self.put(self.find(nd.inputs, "A_Color"), a)
        self.put(self.find(nd.inputs, "B_Color"), b)
        return self.find(nd.outputs, "Result_Color")

    def tint(self, col, fac):
        """colour * scalar (scalar may exceed 1)."""
        return self.mix(1.0, col, fac, "MULTIPLY")

    def lerp(self, fac, a, b):
        nd = self.node("ShaderNodeMix", data_type="FLOAT")
        self.put(self.find(nd.inputs, "Factor_Float"), fac)
        self.put(self.find(nd.inputs, "A_Float"), a)
        self.put(self.find(nd.inputs, "B_Float"), b)
        return self.find(nd.outputs, "Result_Float")

    def ramp(self, fac, stops, interp="LINEAR"):
        nd = self.node("ShaderNodeValToRGB")
        cr = nd.color_ramp
        cr.interpolation = interp
        els = cr.elements
        els[0].position = stops[0][0]
        els[1].position = stops[-1][0]
        for pos, _ in stops[1:-1]:
            els.new(pos)
        for e, (pos, col) in zip(els, stops):
            e.position = pos
            e.color = c4(col)
        self.put(nd.inputs[0], fac)
        return nd.outputs[0]

    def noise(self, vec=None, scale=5.0, detail=2.0, rough=0.5, dist=0.0, lac=2.0, dims="3D", w=None):
        nd = self.node("ShaderNodeTexNoise", noise_dimensions=dims)
        if vec is not None:
            self.put(nd.inputs["Vector"], vec)
        if w is not None:
            self.put(nd.inputs["W"], w)
        self.put(nd.inputs["Scale"], scale)
        self.put(nd.inputs["Detail"], detail)
        self.put(nd.inputs["Roughness"], rough)
        self.put(nd.inputs["Lacunarity"], lac)
        self.put(nd.inputs["Distortion"], dist)
        return nd.outputs[0], nd.outputs[1]

    def voronoi(self, vec=None, scale=5.0, feature="F1", rand=1.0, dims="3D"):
        nd = self.node("ShaderNodeTexVoronoi", voronoi_dimensions=dims, feature=feature)
        if vec is not None:
            self.put(nd.inputs["Vector"], vec)
        self.put(nd.inputs["Scale"], scale)
        self.put(nd.inputs["Randomness"], rand)
        return nd

    def mr(self, v, a=0.0, b=1.0, c=0.0, d=1.0, interp="LINEAR", clamp=True):
        nd = self.node("ShaderNodeMapRange", interpolation_type=interp, clamp=clamp)
        for i, val in enumerate((v, a, b, c, d)):
            self.put(nd.inputs[i], val)
        return nd.outputs[0]

    def smooth(self, v, a, b):
        return self.mr(v, a, b, 0.0, 1.0, interp="SMOOTHSTEP")

    def sep(self, vec):
        nd = self.node("ShaderNodeSeparateXYZ")
        self.put(nd.inputs[0], vec)
        return nd.outputs[0], nd.outputs[1], nd.outputs[2]

    def comb(self, x=0.0, y=0.0, z=0.0):
        nd = self.node("ShaderNodeCombineXYZ")
        for i, val in enumerate((x, y, z)):
            self.put(nd.inputs[i], val)
        return nd.outputs[0]

    def mapping(self, vec, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1)):
        nd = self.node("ShaderNodeMapping")
        self.put(nd.inputs[0], vec)
        self.put(nd.inputs[1], loc)
        self.put(nd.inputs[2], rot)
        self.put(nd.inputs[3], scale)
        return nd.outputs[0]

    def bump(self, height, strength=0.5, dist=0.001, normal=None):
        nd = self.node("ShaderNodeBump")
        self.put(nd.inputs["Strength"], strength)
        self.put(nd.inputs["Distance"], dist)
        self.put(nd.inputs["Height"], height)
        if normal is not None:
            self.put(nd.inputs["Normal"], normal)
        return nd.outputs[0]

    def objco(self):
        return self.node("ShaderNodeTexCoord").outputs["Object"]


PNAMES = {"base": "Base Color", "rough": "Roughness", "metal": "Metallic", "ior": "IOR", "alpha": "Alpha",
          "normal": "Normal", "sss": "Subsurface Weight", "sss_radius": "Subsurface Radius",
          "sss_scale": "Subsurface Scale", "sss_aniso": "Subsurface Anisotropy", "spec": "Specular IOR Level",
          "aniso": "Anisotropic", "aniso_rot": "Anisotropic Rotation", "tangent": "Tangent",
          "trans": "Transmission Weight", "coat": "Coat Weight", "coat_rough": "Coat Roughness",
          "coat_ior": "Coat IOR", "coat_tint": "Coat Tint", "coat_normal": "Coat Normal",
          "sheen": "Sheen Weight", "sheen_rough": "Sheen Roughness", "sheen_tint": "Sheen Tint",
          "emit": "Emission Color", "emit_str": "Emission Strength", "film": "Thin Film Thickness",
          "film_ior": "Thin Film IOR", "diff_rough": "Diffuse Roughness"}


def pbr(name):
    """New material -> (material, NB, principled node, output node)."""
    m = bpy.data.materials.new(name)
    if m.node_tree is None:
        try:
            m.use_nodes = True
        except Exception:
            pass
    nt = m.node_tree
    p = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if out is None:
        out = nt.nodes.new("ShaderNodeOutputMaterial")
    if p is None:
        p = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(p.outputs[0], out.inputs[0])
    return m, NB(nt), p, out


def PS(nb, p, **kw):
    for k, v in kw.items():
        nb.put(p.inputs[PNAMES[k]], v)


def m_simple(name, color, rough=0.5, metal=0.0, **kw):
    m, nb, p, out = pbr(name)
    PS(nb, p, base=color, rough=rough, metal=metal, **kw)
    return m


# =================================================================== mesh helpers
def mesh_obj(name, bm, mats=(), loc=(0, 0, 0), rot=(0, 0, 0), parent=None, smooth=True, sharp=40, wn=False):
    me = bpy.data.meshes.new(name)
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    if smooth:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
        if sharp:
            try:
                me.set_sharp_from_angle(angle=radians(sharp))
            except Exception as e:
                print("sharp warn:", e)
    if not isinstance(mats, (list, tuple)):
        mats = [mats]
    for m in mats:
        if m is not None:
            me.materials.append(m)
    ob = bpy.data.objects.new(name, me)
    ob.location = loc
    ob.rotation_euler = rot
    link(ob, parent)
    if wn:
        md = ob.modifiers.new("WeightedNormal", "WEIGHTED_NORMAL")
        md.keep_sharp = True
        md.weight = 100
    return ob


def aim(ob, direction, axis=(0, 0, 1)):
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = V(axis).rotation_difference(V(direction).normalized())
    return ob


def fillet(pts, n=5, support=True):
    """pts: [(r, z) or (r, z, radius)] -> polyline with the marked corners rounded (+ tight support points)."""
    out = []
    P2 = [V((p[0], p[1])) for p in pts]
    for i, p in enumerate(pts):
        rad = p[2] if len(p) > 2 else 0.0
        if rad <= 0 or i == 0 or i == len(pts) - 1:
            out.append((p[0], p[1]))
            continue
        a, b, c = P2[i - 1], P2[i], P2[i + 1]
        d1, d2 = a - b, c - b
        l1, l2 = d1.length, d2.length
        d1.normalize()
        d2.normalize()
        ang = d1.angle(d2)
        if ang > pi - 1e-3:
            out.append((p[0], p[1]))
            continue
        t = min(rad / tan(ang / 2), l1 * 0.45, l2 * 0.45)
        r_eff = t * tan(ang / 2)
        p1, p2 = b + d1 * t, b + d2 * t
        cen = b + (d1 + d2).normalized() * (r_eff / sin(ang / 2))
        a1 = atan2(p1.y - cen.y, p1.x - cen.x)
        a2 = atan2(p2.y - cen.y, p2.x - cen.x)
        da = a2 - a1
        while da > pi:
            da -= 2 * pi
        while da < -pi:
            da += 2 * pi
        if support:
            e = min(0.5 * r_eff, 0.4 * (l1 - t))
            if e > 2e-5:
                out.append(tuple(p1 + d1 * e))
        for k in range(n + 1):
            aa = a1 + da * k / n
            out.append((cen.x + r_eff * cos(aa), cen.y + r_eff * sin(aa)))
        if support:
            e = min(0.5 * r_eff, 0.4 * (l2 - t))
            if e > 2e-5:
                out.append(tuple(p2 + d2 * e))
    return out


def lathe(name, prof, segs=64, mats=(), mat_ids=None, rmod=None, **kw):
    """Revolve a profile [(r, z)] around Z.  Profile counter-clockwise in the (r, z) half plane
    (bottom centre -> out -> up -> in) gives outward normals.  r == 0 points become poles.
    rmod: optional (fn(angle01) -> multiplier, r_min) applied to rings with r >= r_min (knurling)."""
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new("UVMap")
    plen = [0.0]
    for (r0, z0), (r1, z1) in zip(prof, prof[1:]):
        plen.append(plen[-1] + math.hypot(r1 - r0, z1 - z0))
    rings = []
    for r, z in prof:
        if r <= 1e-7:
            rings.append([bm.verts.new((0.0, 0.0, z))])
        else:
            ring = []
            for i in range(segs):
                rr = r
                if rmod and r >= rmod[1]:
                    rr = r * rmod[0](i / segs)
                ring.append(bm.verts.new((rr * cos(2 * pi * i / segs), rr * sin(2 * pi * i / segs), z)))
            rings.append(ring)
    for k in range(len(prof) - 1):
        a, b = rings[k], rings[k + 1]
        va, vb = plen[k], plen[k + 1]
        if len(a) == 1 and len(b) == 1:
            continue
        for i in range(segs):
            j = (i + 1) % segs
            u0, u1 = i / segs, (i + 1) / segs
            if len(a) == 1:
                vs, uv = (a[0], b[j], b[i]), (((u0 + u1) / 2, va), (u1, vb), (u0, vb))
            elif len(b) == 1:
                vs, uv = (a[i], a[j], b[0]), ((u0, va), (u1, va), ((u0 + u1) / 2, vb))
            else:
                vs, uv = (a[i], a[j], b[j], b[i]), ((u0, va), (u1, va), (u1, vb), (u0, vb))
            f = bm.faces.new(vs)
            if mat_ids:
                f.material_index = mat_ids[k]
            for lp, c in zip(f.loops, uv):
                lp[uvl].uv = c
    return mesh_obj(name, bm, mats, **kw)


def lathe_at(name, prof, origin, direction, **kw):
    ob = lathe(name, prof, **kw)
    ob.location = origin
    return aim(ob, direction)


def tube(name, p1, p2, r, mats=(), segs=24, parent=None):
    p1, p2 = V(p1), V(p2)
    ln = (p2 - p1).length
    return lathe_at(name, [(0, 0), (r, 0), (r, ln), (0, ln)], p1, p2 - p1, segs=segs, mats=mats, parent=parent)


def knurl(name, r, z0, z1, teeth=24, depth=0.07, mats=(), **kw):
    b = min(0.0007, (z1 - z0) * 0.25)
    prof = [(0, z0), (r - b, z0), (r, z0 + b), (r, z1 - b), (r - b, z1), (0, z1)]
    fn = lambda t: 1.0 - depth * (0.5 - 0.5 * cos(2 * pi * teeth * t))
    return lathe(name, prof, segs=teeth * 6, mats=mats, rmod=(fn, r - b * 0.5), sharp=60, **kw)


def circle_sec(n=12, r=1.0):
    return [(r * cos(2 * pi * k / n), r * sin(2 * pi * k / n)) for k in range(n)]


def ellipse_sec(rx, ry, n=16, power=2.0):
    """Super-ellipse section (power 2 = ellipse, >2 = squarer)."""
    out = []
    for k in range(n):
        a = 2 * pi * k / n
        ca, sa = cos(a), sin(a)
        out.append((rx * math.copysign(abs(ca) ** (2 / power), ca), ry * math.copysign(abs(sa) ** (2 / power), sa)))
    return out


def crspline(pts, n=8, closed=False):
    """Catmull-Rom through the control points."""
    P = [V(p) for p in pts]
    m = len(P)
    out = []
    for i in (range(m) if closed else range(m - 1)):
        if closed:
            p0, p1, p2, p3 = P[(i - 1) % m], P[i], P[(i + 1) % m], P[(i + 2) % m]
        else:
            p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[min(i + 1, m - 1)], P[min(i + 2, m - 1)]
        for k in range(n):
            t = k / n
            out.append(0.5 * ((2 * p1) + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    if not closed:
        out.append(P[-1].copy())
    return out


def sweep_bm(bm, path, section, scale=None, closed=False, cap=True, up=(0, 0, 1), mat_index=0, uvl=None, twist=None):
    """Sweep a 2D section along a path (parallel-transport frames).  section x -> N, y -> B (N x B = T)."""
    path = [V(p) for p in path]
    n = len(path)
    T = []
    for i in range(n):
        if closed:
            t = path[(i + 1) % n] - path[(i - 1) % n]
        else:
            t = path[min(i + 1, n - 1)] - path[max(i - 1, 0)]
        T.append(t.normalized())
    upv = V(up)
    N0 = upv - T[0] * upv.dot(T[0])
    if N0.length < 1e-5:
        N0 = V((1, 0, 0)) - T[0] * T[0].x
    N0.normalize()
    Ns = [N0]
    for i in range(1, n):
        q = T[i - 1].rotation_difference(T[i])
        Ni = q @ Ns[-1]
        Ni = (Ni - T[i] * Ni.dot(T[i])).normalized()
        Ns.append(Ni)
    cum = [0.0]
    for i in range(1, n):
        cum.append(cum[-1] + (path[i] - path[i - 1]).length)
    total = cum[-1] or 1.0
    m = len(section)
    rings = []
    for i in range(n):
        t = cum[i] / total
        s = scale(t) if callable(scale) else (1.0 if scale is None else scale)
        sx, sy = s if isinstance(s, (tuple, list)) else (s, s)
        Nv, Bv = Ns[i], T[i].cross(Ns[i])
        if twist:
            q = Quaternion(T[i], twist(t))
            Nv, Bv = q @ Nv, q @ Bv
        rings.append([bm.verts.new(path[i] + Nv * (cx * sx) + Bv * (cy * sy)) for cx, cy in section])
    for i in range(n if closed else n - 1):
        a, b = rings[i], rings[(i + 1) % n]
        for k in range(m):
            k2 = (k + 1) % m
            f = bm.faces.new((a[k], a[k2], b[k2], b[k]))
            f.material_index = mat_index
            if uvl is not None:
                v0 = cum[i]
                v1 = cum[i + 1] if i + 1 < n else total
                for lp, c in zip(f.loops, ((k / m, v0), ((k + 1) / m, v0), ((k + 1) / m, v1), (k / m, v1))):
                    lp[uvl].uv = c
    if cap and not closed:
        f = bm.faces.new(list(reversed(rings[0])))
        f.material_index = mat_index
        f = bm.faces.new(rings[-1])
        f.material_index = mat_index
    return rings


def sweep(name, path, section, mats=(), scale=None, closed=False, cap=True, up=(0, 0, 1), twist=None, **kw):
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new("UVMap")
    sweep_bm(bm, path, section, scale=scale, closed=closed, cap=cap, up=up, uvl=uvl, twist=twist)
    return mesh_obj(name, bm, mats, **kw)


def box(name, size, mats=(), bevel=0.0, bsegs=3, base=False, **kw):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x *= size[0]
        v.co.y *= size[1]
        v.co.z *= size[2]
        if base:
            v.co.z += size[2] / 2
    if bevel:
        bmesh.ops.bevel(bm, geom=list(bm.edges), offset=bevel, segments=bsegs, profile=0.5, affect="EDGES")
    return mesh_obj(name, bm, mats, wn=bool(bevel), **kw)


def uvsphere(name, r, mats=(), u=32, v=16, scale=(1, 1, 1), **kw):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=r)
    for vt in bm.verts:
        vt.co.x *= scale[0]
        vt.co.y *= scale[1]
        vt.co.z *= scale[2]
    return mesh_obj(name, bm, mats, sharp=0, **kw)


FONTS = {}


def get_font(path):
    if path not in FONTS:
        f = None
        if path and os.path.exists(path):
            try:
                f = bpy.data.fonts.load(path)
            except Exception as e:
                print("font warn:", e)
        FONTS[path] = f
    return FONTS[path]


def text_obj(name, body, size, mat, fontpath=None, spacing=1.0, align="CENTER", **kw):
    """Flat text converted to a mesh (so the .blend carries no font dependency).  Faces +Z, reads along +X."""
    cu = bpy.data.curves.new(name + "_cu", "FONT")
    cu.body = body
    cu.size = size
    cu.align_x = align
    cu.align_y = "CENTER"
    cu.space_character = spacing
    cu.resolution_u = 3
    f = get_font(fontpath)
    if f:
        cu.font = f
    tmp = bpy.data.objects.new(name + "_tmp", cu)
    scene.collection.objects.link(tmp)
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(tmp.evaluated_get(dg))
    bpy.data.objects.remove(tmp)
    bpy.data.curves.remove(cu)
    me.name = name
    me.materials.clear()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    ob.location = kw.get("loc", (0, 0, 0))
    ob.rotation_euler = kw.get("rot", (0, 0, 0))
    return link(ob, kw.get("parent"))


def world_min_z(objs):
    bpy.context.view_layer.update()
    mz = 1e9
    for ob in objs:
        if ob.type != "MESH":
            continue
        mw = ob.matrix_world
        for vt in ob.data.vertices:
            z = (mw @ vt.co).z
            if z < mz:
                mz = z
    return mz


def drop(root, objs, z, eps=0.0002):
    """Move `root` vertically so the lowest vertex of `objs` rests on height z."""
    root.location.z += z + eps - world_min_z(objs)
    bpy.context.view_layer.update()


# =================================================================== materials
def m_backdrop():
    m, nb, p, out = pbr("s1_backdrop")
    P = nb.objco()
    n1, _ = nb.noise(P, scale=2.5, detail=3.0, rough=0.6)
    col = nb.mix(nb.smooth(n1, 0.3, 0.7), (0.030, 0.080, 0.088), (0.037, 0.099, 0.107))
    n2, _ = nb.noise(P, scale=400.0, detail=2.0)
    PS(nb, p, base=col, rough=1.0, spec=0.0, normal=nb.bump(n2, 0.08, 0.0005))
    return m


def m_wood(name, planks=True, plank_w=0.096, ring_w=0.0031, tone=1.0, seed=0.0, rough=0.33, coat=0.35, cup_ring=None):
    """Solid 3D wood: growth rings as distorted cylinders around a (tilted) axis along local X.
    Top faces show long/cathedral grain, faces cut across X show end-grain arcs.  Optional glued planks."""
    m, nb, p, out = pbr(name)
    tc = nb.node("ShaderNodeTexCoord")
    P = tc.outputs["Object"]
    x, y, z = nb.sep(P)
    if planks:
        yi = nb.math("DIVIDE", y, plank_w)
        idx = nb.math("ROUND", yi)
        ly = nb.mul(nb.sub(yi, idx), plank_w)
        wn = nb.node("ShaderNodeTexWhiteNoise", noise_dimensions="1D")
        nb.put(wn.inputs["W"], nb.add(idx, 3.7 + seed))
        r1, r2, r3 = nb.sep(wn.outputs["Color"])
    else:
        ly = y
        idx = nb.math("ADD", 0.0, seed)
        r1, r2, r3 = nb.math("ADD", 0.31, 0.0), nb.math("ADD", 0.62, 0.0), nb.math("ADD", 0.8, 0.0)
    cy = nb.mul(nb.sub(r1, 0.5), 0.12)
    sgn = nb.sub(nb.mul(nb.math("GREATER_THAN", r3, 0.42), 2.0), 1.0)
    cz = nb.mul(nb.madd(r2, 0.30, 0.10), nb.mul(sgn, -1.0))
    ty = nb.mul(nb.sub(r3, 0.5), 0.06)
    tz = nb.madd(nb.sub(r1, 0.5), 0.07, 0.012)
    yy = nb.add(nb.sub(ly, cy), nb.mul(x, ty))
    zz = nb.add(nb.sub(z, cz), nb.mul(x, tz))
    d = nb.math("SQRT", nb.add(nb.mul(yy, yy), nb.mul(zz, zz)))
    wseed = nb.madd(idx, 7.13, seed)
    n_low, _ = nb.noise(nb.mapping(P, scale=(1.3, 7.0, 7.0)), scale=1.0, detail=2.0, dims="4D", w=wseed)
    n_hi, _ = nb.noise(nb.mapping(P, scale=(6.0, 55.0, 55.0)), scale=1.0, detail=2.0, dims="4D", w=wseed)
    d2 = nb.add(d, nb.add(nb.mul(nb.sub(n_low, 0.5), 0.013), nb.mul(nb.sub(n_hi, 0.5), 0.0020)))
    jit, _ = nb.noise(dims="1D", w=nb.madd(d2, 46.0, nb.mul(r2, 20.0)), scale=1.0, detail=1.0)
    ph = nb.add(nb.math("DIVIDE", d2, ring_w), nb.mul(nb.sub(jit, 0.5), 1.7))
    saw = nb.math("FRACT", ph)
    li = (0.300 * tone, 0.165 * tone, 0.074 * tone)
    mi = (0.248 * tone, 0.130 * tone, 0.057 * tone)
    da = (0.160 * tone, 0.079 * tone, 0.034 * tone)
    ring = nb.ramp(saw, [(0.0, li), (0.55, mi), (0.86, da), (0.95, da), (1.0, mi)])
    # value modulation: long soft variation, per plank, fibre streaks, glue line, end grain
    nv, _ = nb.noise(nb.mapping(P, scale=(0.9, 5.0, 5.0)), scale=1.0, detail=3.0, dims="4D", w=wseed)
    f = nb.mr(nv, 0.25, 0.75, 0.86, 1.10)
    f = nb.mul(f, nb.madd(r2, 0.20, 0.90))
    stk, _ = nb.noise(nb.mapping(P, scale=(2.2, 190.0, 190.0)), scale=1.0, detail=3.0, rough=0.65)
    f = nb.mul(f, nb.mr(stk, 0.25, 0.75, 0.80, 1.07))
    if planks:
        glue = nb.smooth(nb.math("ABSOLUTE", ly), plank_w / 2 - 0.0011, plank_w / 2 - 0.0003)
        f = nb.mul(f, nb.madd(glue, -0.32, 1.0))
    nx, _, _ = nb.sep(tc.outputs["Normal"])
    eg = nb.smooth(nb.math("ABSOLUTE", nx), 0.30, 0.85)
    f = nb.mul(f, nb.madd(eg, -0.20, 1.0))
    f = nb.mul(f, nb.madd(nb.mul(nb.smooth(saw, 0.55, 0.9), eg), -0.38, 1.0))
    col = nb.tint(ring, f)
    col = nb.mix(nb.mul(r3, 0.45), col, nb.mix(1.0, col, (1.10, 0.92, 0.80, 1.0), "MULTIPLY"))
    # finish: satin lacquer with smudges and a slightly glossier latewood
    smn, _ = nb.noise(P, scale=9.0, detail=4.0, rough=0.65)
    smudge = nb.smooth(smn, 0.42, 0.72)
    late = nb.smooth(saw, 0.6, 0.9)
    r = nb.add(rough, nb.add(nb.mul(smudge, 0.13), nb.add(nb.mul(late, -0.05), nb.mul(eg, 0.22))))
    h = nb.add(nb.mul(late, 0.25), nb.mul(stk, 0.75))
    coat_w = nb.mul(nb.madd(eg, -0.7, 1.0), coat)
    if cup_ring:                                   # dull ring left by a wet mug: (x, y) in object space
        dx, dy = nb.sub(x, cup_ring[0]), nb.sub(y, cup_ring[1])
        dd = nb.math("SQRT", nb.add(nb.mul(dx, dx), nb.mul(dy, dy)))
        band = nb.sub(1.0, nb.smooth(nb.math("ABSOLUTE", nb.sub(dd, 0.0310)), 0.0008, 0.0030))
        brk, _ = nb.noise(P, scale=38.0, detail=2.0)
        mask = nb.mul(band, nb.smooth(brk, 0.40, 0.62))
        r = nb.add(r, nb.mul(mask, 0.15))
        col = nb.tint(col, nb.madd(mask, -0.22, 1.0))
        coat_w = nb.mul(coat_w, nb.madd(mask, -0.6, 1.0))
    PS(nb, p, base=col, rough=r, normal=nb.bump(h, 0.07, 0.0002),
       coat=coat_w, coat_rough=nb.madd(smudge, 0.22, 0.10))
    return m


def m_brass(name="s1_brass", base=(0.87, 0.60, 0.23), rough=0.17, radial=True, smudge=0.17, patina=0.26, aniso=0.5):
    m, nb, p, out = pbr(name)
    P = nb.objco()
    big, _ = nb.noise(P, scale=16.0, detail=4.0, rough=0.65)
    sm = nb.smooth(big, 0.40, 0.72)
    fine, _ = nb.noise(P, scale=260.0, detail=2.0)
    pn, _ = nb.noise(P, scale=7.0, detail=6.0, rough=0.72, dist=0.6)
    pat = nb.smooth(pn, 0.52, 0.80)
    col = nb.mix(nb.mul(pat, patina), base, (0.30, 0.165, 0.055))
    r = nb.add(nb.add(rough, nb.mul(sm, smudge)), nb.add(nb.mul(nb.sub(fine, 0.5), 0.07), nb.mul(pat, 0.12)))
    PS(nb, p, base=col, metal=1.0, rough=r, normal=nb.bump(fine, 0.03, 0.0002))
    if radial:
        tg = nb.node("ShaderNodeTangent", direction_type="RADIAL", axis="Z")
        PS(nb, p, aniso=aniso, tangent=tg.outputs[0])
    return m


ENAMEL_A, ENAMEL_B = (0.26, 0.21, 0.13), (0.32, 0.26, 0.17)      # aged ivory enamel inside the shade


def m_enamel():
    m, nb, p, out = pbr("s1_enamel")
    n, _ = nb.noise(nb.objco(), scale=30.0, detail=3.0)
    PS(nb, p, base=nb.mix(n, ENAMEL_A, ENAMEL_B), rough=nb.mr(n, 0.3, 0.7, 0.30, 0.45), coat=0.3,
       coat_rough=0.15)
    return m


def m_bulb():
    """Frosted bulb: glows to camera / glossy rays only, the real light is a point lamp inside it."""
    m, nb, p, out = pbr("s1_bulb")
    lp = nb.node("ShaderNodeLightPath")
    vis = nb.math("MAXIMUM", lp.outputs["Is Camera Ray"], lp.outputs["Is Glossy Ray"])
    PS(nb, p, base=(1.0, 0.92, 0.80), rough=0.45, emit=(1.0, 0.72, 0.40), emit_str=nb.mul(vis, 420.0))
    return m


def m_cloth(name, color, pitch=0.00085, sheen=0.45, wear=0.45, spec=0.35):
    """Book cloth: woven bumps + slubs along both thread directions, fuzz sheen, rubbed edges."""
    m, nb, p, out = pbr(name)
    u, v, _ = nb.sep(nb.node("ShaderNodeUVMap").outputs[0])
    k = 2 * pi / pitch
    weave = nb.mul(nb.math("SINE", nb.mul(u, k)), nb.math("SINE", nb.mul(v, k)))
    w01 = nb.madd(weave, 0.5, 0.5)
    s1, _ = nb.noise(nb.comb(nb.mul(u, 55.0), nb.mul(v, 1500.0), 0.0), scale=1.0, detail=2.0)
    s2, _ = nb.noise(nb.comb(nb.mul(u, 1500.0), nb.mul(v, 55.0), 3.0), scale=1.0, detail=2.0)
    slub = nb.mul(nb.add(s1, s2), 0.5)
    P = nb.objco()
    blot, _ = nb.noise(P, scale=14.0, detail=4.0, rough=0.65)
    f = nb.add(nb.add(0.74, nb.mul(w01, 0.16)), nb.add(nb.mul(slub, 0.32), nb.mul(nb.sub(blot, 0.5), 0.30)))
    col = nb.tint(color, f)
    geo = nb.node("ShaderNodeNewGeometry")
    edge = nb.smooth(geo.outputs["Pointiness"], 0.53, 0.62)
    en, _ = nb.noise(P, scale=120.0, detail=3.0, rough=0.7)
    edge = nb.mul(nb.mul(edge, nb.smooth(en, 0.35, 0.65)), wear)
    lighter = nb.mix(0.45, color, (0.62, 0.56, 0.48))
    col = nb.mix(edge, col, lighter)
    h = nb.add(nb.mul(w01, 0.6), nb.mul(slub, 0.6))
    PS(nb, p, base=col, rough=nb.madd(blot, 0.14, 0.70), sheen=sheen, sheen_rough=0.45,
       sheen_tint=nb.mix(0.5, color, (1.0, 1.0, 1.0)), normal=nb.bump(h, 0.55, 0.00022), spec=spec)
    return m


def m_paper():
    """Page block: thin page lines + signature bands along the stacking axis (local Z)."""
    m, nb, p, out = pbr("s1_paper")
    P = nb.objco()
    x, y, z = nb.sep(P)
    wob, _ = nb.noise(nb.mapping(P, scale=(70.0, 70.0, 0.0)), scale=1.0, detail=1.0)
    zz = nb.add(z, nb.mul(nb.sub(wob, 0.5), 0.0006))
    l1, _ = nb.noise(dims="1D", w=nb.mul(zz, 4200.0), scale=1.0, detail=1.0, rough=0.7)
    l2, _ = nb.noise(dims="1D", w=nb.mul(zz, 520.0), scale=1.0, detail=1.0)
    val = nb.mul(nb.mr(l1, 0.28, 0.72, 0.66, 1.0), nb.mr(l2, 0.3, 0.7, 0.86, 1.0))
    ton, _ = nb.noise(P, scale=22.0, detail=3.0)
    base = nb.mix(ton, (0.80, 0.70, 0.52), (0.86, 0.78, 0.62))
    PS(nb, p, base=nb.tint(base, val), rough=0.88, spec=0.2,
       normal=nb.bump(nb.add(nb.mul(l1, 0.6), nb.mul(l2, 0.4)), 0.7, 0.00025))
    return m


def m_gold():
    m, nb, p, out = pbr("s1_gold_foil")
    n, _ = nb.noise(nb.objco(), scale=900.0, detail=2.0)
    PS(nb, p, base=(0.96, 0.72, 0.30), metal=1.0, rough=nb.mr(n, 0.3, 0.7, 0.22, 0.42))
    return m


def m_mug():
    """Stoneware: speckled cream glaze dipped over raw clay, toasty rim, throwing rings."""
    m, nb, p, out = pbr("s1_mug")
    P = nb.objco()
    x, y, z = nb.sep(P)
    wav, _ = nb.noise(nb.mapping(P, scale=(14.0, 14.0, 0.3)), scale=1.0, detail=2.0)
    drp, _ = nb.noise(nb.mapping(P, scale=(34.0, 34.0, 0.2)), scale=1.0, detail=1.0)
    zl = nb.add(0.0245, nb.add(nb.mul(nb.sub(wav, 0.5), 0.012), nb.mul(nb.smooth(drp, 0.62, 0.80), -0.010)))
    glz = nb.smooth(z, nb.sub(zl, 0.0008), nb.add(zl, 0.0008))
    band = nb.mul(glz, nb.sub(1.0, nb.smooth(z, zl, nb.add(zl, 0.006))))            # thin, darker edge of the glaze
    vor = nb.voronoi(P, scale=420.0)
    cr, cg, cb = nb.sep(vor.outputs["Color"])
    speck = nb.mul(nb.sub(1.0, nb.smooth(vor.outputs["Distance"], 0.10, 0.22)), nb.math("GREATER_THAN", cr, 0.62))
    cloud, _ = nb.noise(P, scale=22.0, detail=4.0, rough=0.6)
    glaze = nb.mix(cloud, (0.74, 0.70, 0.61), (0.83, 0.80, 0.72))
    rim = nb.smooth(z, 0.0885, 0.0915)
    glaze = nb.mix(nb.mul(rim, 0.75), glaze, (0.36, 0.20, 0.10))
    glaze = nb.mix(nb.mul(band, 0.55), glaze, (0.42, 0.27, 0.15))
    glaze = nb.mix(nb.mul(speck, 0.75), glaze, (0.20, 0.11, 0.06))
    grain, _ = nb.noise(P, scale=500.0, detail=2.0)
    clay = nb.mix(cloud, (0.47, 0.30, 0.19), (0.58, 0.39, 0.26))
    clay = nb.tint(clay, nb.mr(grain, 0.2, 0.8, 0.85, 1.1))
    clay = nb.mix(nb.mul(speck, 0.8), clay, (0.16, 0.09, 0.05))
    col = nb.mix(glz, clay, glaze)
    rings = nb.math("SINE", nb.mul(z, 2 * pi / 0.0046))
    h = nb.add(nb.mul(rings, 0.5), nb.add(nb.mul(grain, nb.madd(glz, -0.5, 0.6)), nb.mul(cloud, 0.6)))
    PS(nb, p, base=col, rough=nb.lerp(glz, 0.82, nb.madd(cloud, 0.16, 0.20)),
       coat=nb.mul(glz, 1.0), coat_rough=nb.madd(cloud, 0.06, 0.03), coat_ior=1.5,
       normal=nb.bump(h, 0.16, 0.0003), sss=0.06, sss_radius=(0.004, 0.003, 0.002), sss_scale=1.0)
    return m


def m_coffee():
    m, nb, p, out = pbr("s1_coffee")
    PS(nb, p, base=(0.045, 0.020, 0.008), rough=0.04, ior=1.34, coat=0.5, coat_rough=0.02)
    return m


def m_cork():
    m, nb, p, out = pbr("s1_cork")
    P = nb.objco()
    vor = nb.voronoi(P, scale=560.0)
    cr, _, _ = nb.sep(vor.outputs["Color"])
    col = nb.ramp(cr, [(0.0, (0.27, 0.16, 0.08)), (0.5, (0.46, 0.30, 0.16)), (1.0, (0.62, 0.44, 0.26))])
    big = nb.voronoi(P, scale=150.0)
    br, _, _ = nb.sep(big.outputs["Color"])
    dark = nb.mul(nb.sub(1.0, nb.smooth(big.outputs["Distance"], 0.10, 0.2)), nb.math("GREATER_THAN", br, 0.6))
    col = nb.mix(nb.mul(dark, 0.8), col, (0.10, 0.055, 0.03))
    PS(nb, p, base=col, rough=0.9, spec=0.25,
       normal=nb.bump(nb.sub(vor.outputs["Distance"], nb.mul(dark, 2.0)), 0.6, 0.0004))
    return m


def m_terracotta():
    m, nb, p, out = pbr("s1_terracotta")
    P = nb.objco()
    x, y, z = nb.sep(P)
    n1, _ = nb.noise(P, scale=9.0, detail=5.0, rough=0.7)
    col = nb.mix(n1, (0.50, 0.205, 0.10), (0.66, 0.31, 0.16))
    n2, _ = nb.noise(nb.mapping(P, scale=(1.0, 1.0, 0.35)), scale=16.0, detail=5.0, rough=0.75, dist=0.4)
    lowmask = nb.sub(1.0, nb.smooth(z, 0.01, 0.07))
    chalk = nb.mul(nb.smooth(n2, 0.44, 0.70), nb.madd(lowmask, 0.55, 0.45))
    col = nb.mix(nb.mul(chalk, 0.75), col, (0.72, 0.60, 0.50))
    damp = nb.sub(1.0, nb.smooth(z, 0.0, 0.025))
    col = nb.tint(col, nb.madd(damp, -0.22, 1.0))
    gr, _ = nb.noise(P, scale=650.0, detail=2.0)
    col = nb.tint(col, nb.mr(gr, 0.2, 0.8, 0.90, 1.08))
    lines = nb.math("SINE", nb.mul(z, 2 * pi / 0.0021))
    PS(nb, p, base=col, rough=nb.madd(chalk, 0.06, 0.84), spec=0.3,
       normal=nb.bump(nb.add(nb.mul(gr, 0.8), nb.mul(lines, 0.2)), 0.35, 0.0003))
    return m


def m_soil():
    m, nb, p, out = pbr("s1_soil")
    P = nb.objco()
    n1, _ = nb.noise(P, scale=120.0, detail=4.0, rough=0.7)
    col = nb.mix(n1, (0.018, 0.011, 0.007), (0.060, 0.036, 0.020))
    vor = nb.voronoi(P, scale=110.0)
    cr, _, _ = nb.sep(vor.outputs["Color"])
    perl = nb.mul(nb.sub(1.0, nb.smooth(vor.outputs["Distance"], 0.12, 0.2)), nb.math("GREATER_THAN", cr, 0.86))
    col = nb.mix(perl, col, (0.62, 0.60, 0.55))
    PS(nb, p, base=col, rough=0.95, spec=0.2, normal=nb.bump(nb.add(n1, nb.mul(vor.outputs["Distance"], 0.6)), 1.0, 0.002))
    return m


def m_leaf(name="s1_leaf", top_a=(0.016, 0.070, 0.018), top_b=(0.040, 0.145, 0.030), vein_col=(0.13, 0.30, 0.07),
           back=(0.085, 0.19, 0.06), n_veins=8.0, transl=0.36, rough=0.38):
    """Leaf blade: midrib + pinnate veins + fine network computed from UVs, glossy top, matte pale underside,
    and real translucency (lights up when back-lit).  Per-object colour shift."""
    m, nb, p, out = pbr(name)
    u, v, _ = nb.sep(nb.node("ShaderNodeUVMap").outputs[0])
    a = nb.mul(nb.math("ABSOLUTE", nb.sub(u, 0.5)), 2.0)
    mid = nb.sub(1.0, nb.smooth(a, 0.0, nb.madd(v, -0.05, 0.075)))
    P = nb.objco()
    wob, _ = nb.noise(P, scale=60.0, detail=1.0)
    q = nb.add(nb.sub(nb.mul(v, n_veins), nb.mul(nb.math("POWER", a, 0.85), 1.7)), nb.mul(nb.sub(wob, 0.5), 0.3))
    tri = nb.mul(nb.math("ABSOLUTE", nb.sub(nb.math("FRACT", q), 0.5)), 2.0)
    lat = nb.sub(1.0, nb.smooth(tri, 0.0, 0.22))
    lat = nb.mul(lat, nb.mul(nb.sub(1.0, nb.smooth(a, 0.68, 0.98)), nb.smooth(v, 0.02, 0.10)))
    vein = nb.math("MAXIMUM", mid, nb.mul(lat, 0.75))
    vor = nb.voronoi(nb.comb(nb.mul(u, 13.0), nb.mul(v, 38.0), 0.0), scale=1.0, feature="DISTANCE_TO_EDGE")
    ter = nb.sub(1.0, nb.smooth(vor.outputs["Distance"], 0.0, 0.07))
    veinall = nb.math("MAXIMUM", vein, nb.mul(ter, 0.3))
    rnd = nb.node("ShaderNodeObjectInfo").outputs["Random"]
    blot, _ = nb.noise(P, scale=26.0, detail=3.0, rough=0.6)
    fac = nb.math("ADD", nb.mul(blot, 0.7), nb.madd(rnd, 0.55, -0.15), clamp=True)
    top = nb.mix(fac, top_a, top_b)
    top = nb.mix(nb.mul(veinall, 0.42), top, vein_col)
    backc = nb.mix(nb.mul(veinall, 0.42), back, (0.20, 0.34, 0.12))
    geo = nb.node("ShaderNodeNewGeometry")
    bf = geo.outputs["Backfacing"]
    col = nb.mix(bf, top, backc)
    rgh = nb.lerp(bf, nb.madd(nb.sub(blot, 0.5), 0.14, rough), 0.62)
    h = nb.sub(nb.mul(nb.smooth(tri, 0.0, 0.6), 0.6), nb.add(vein, nb.mul(ter, 0.2)))
    nrm = nb.bump(h, 0.28, 0.0006)
    PS(nb, p, base=col, rough=rgh, normal=nrm, coat=nb.lerp(bf, 0.06, 0.0), coat_rough=0.25, spec=0.32)
    tr = nb.node("ShaderNodeBsdfTranslucent")
    nb.put(tr.inputs["Color"], nb.mix(nb.mul(veinall, 0.6), (0.13, 0.40, 0.035), (0.30, 0.50, 0.10)))
    nb.put(tr.inputs["Normal"], nrm)
    mx = nb.node("ShaderNodeMixShader")
    nb.put(mx.inputs[0], transl)
    nb.nt.links.new(p.outputs[0], mx.inputs[1])
    nb.nt.links.new(tr.outputs[0], mx.inputs[2])
    nb.nt.links.new(mx.outputs[0], out.inputs["Surface"])
    return m


def m_stem(name="s1_stem", a=(0.060, 0.15, 0.035), b=(0.12, 0.22, 0.06)):
    m, nb, p, out = pbr(name)
    n, _ = nb.noise(nb.mapping(nb.objco(), scale=(60.0, 60.0, 8.0)), scale=1.0, detail=2.0)
    PS(nb, p, base=nb.mix(n, a, b), rough=0.42, sss=0.15, sss_radius=(0.004, 0.008, 0.002))
    return m


def m_orange():
    m, nb, p, out = pbr("s1_orange")
    P = nb.objco()
    x, y, z = nb.sep(P)
    vor = nb.voronoi(P, scale=520.0)
    pit = nb.smooth(vor.outputs["Distance"], 0.0, 0.42)                 # 0 in the pore centre
    lum, _ = nb.noise(P, scale=45.0, detail=3.0, rough=0.6)
    big, _ = nb.noise(P, scale=16.0, detail=3.0)
    col = nb.ramp(big, [(0.25, (0.86, 0.25, 0.012)), (0.5, (0.93, 0.36, 0.022)), (0.78, (0.96, 0.47, 0.04))])
    col = nb.tint(col, nb.madd(pit, 0.22, 0.80))
    green = nb.mul(nb.smooth(z, 0.020, 0.034), nb.smooth(big, 0.3, 0.7))
    col = nb.mix(nb.mul(green, 0.35), col, (0.62, 0.50, 0.05))
    h = nb.add(nb.mul(pit, 0.8), nb.mul(lum, 0.5))
    PS(nb, p, base=col, rough=nb.add(nb.madd(pit, -0.16, 0.48), nb.mul(nb.sub(lum, 0.5), 0.12)),
       normal=nb.bump(h, 0.32, 0.0005), sss=0.45, sss_radius=(1.0, 0.32, 0.06), sss_scale=0.012,
       coat=0.25, coat_rough=0.18, spec=0.5)
    return m


def m_acetate():
    """Tortoiseshell frame."""
    m, nb, p, out = pbr("s1_tortoise")
    P = nb.objco()
    n, _ = nb.noise(P, scale=95.0, detail=3.0, rough=0.6, dist=1.4)
    col = nb.ramp(n, [(0.32, (0.012, 0.005, 0.002)), (0.5, (0.11, 0.035, 0.008)), (0.68, (0.50, 0.21, 0.035))])
    PS(nb, p, base=col, rough=0.14, coat=1.0, coat_rough=0.05, sss=0.25, sss_radius=(0.004, 0.0015, 0.0005))
    return m


def m_lens():
    m, nb, p, out = pbr("s1_lens")
    PS(nb, p, base=(1.0, 1.0, 1.0), rough=0.0, ior=1.52, trans=1.0)
    lp = nb.node("ShaderNodeLightPath")
    tr = nb.node("ShaderNodeBsdfTransparent")
    nb.put(tr.inputs["Color"], (0.93, 0.95, 0.94))
    mx = nb.node("ShaderNodeMixShader")
    nb.put(mx.inputs[0], lp.outputs["Is Shadow Ray"])
    nb.nt.links.new(p.outputs[0], mx.inputs[1])
    nb.nt.links.new(tr.outputs[0], mx.inputs[2])
    nb.nt.links.new(mx.outputs[0], out.inputs["Surface"])
    return m


def m_ribbon():
    m, nb, p, out = pbr("s1_ribbon")
    u, v, _ = nb.sep(nb.node("ShaderNodeUVMap").outputs[0])
    w = nb.math("SINE", nb.mul(v, 2 * pi / 0.0005))
    n, _ = nb.noise(nb.comb(nb.mul(u, 40.0), nb.mul(v, 300.0), 0.0), scale=1.0, detail=2.0)
    col = nb.tint((0.42, 0.022, 0.035), nb.mr(n, 0.2, 0.8, 0.75, 1.2))
    PS(nb, p, base=col, rough=0.38, sheen=0.6, sheen_rough=0.3, sheen_tint=(1.0, 0.5, 0.5),
       normal=nb.bump(w, 0.25, 0.00008), aniso=0.6)
    return m


def m_cord():
    """Twisted cloth-covered flex."""
    m, nb, p, out = pbr("s1_cord")
    u, v, _ = nb.sep(nb.node("ShaderNodeUVMap").outputs[0])
    tw = nb.math("SINE", nb.mul(nb.add(nb.mul(v, 1.0 / 0.0042), nb.mul(u, 2.0)), 2 * pi))
    tw2 = nb.math("SINE", nb.mul(nb.sub(nb.mul(v, 1.0 / 0.0042), nb.mul(u, 2.0)), 2 * pi))
    h = nb.math("MAXIMUM", tw, tw2)
    col = nb.mix(nb.smooth(h, 0.2, 0.9), (0.022, 0.015, 0.010), (0.070, 0.046, 0.028))
    PS(nb, p, base=col, rough=0.8, sheen=0.5, sheen_rough=0.5, normal=nb.bump(h, 0.8, 0.0004))
    return m


def m_steam(height):
    """Procedural wisps of steam (volume) inside a box whose origin sits at the bottom centre."""
    m = bpy.data.materials.new("s1_steam")
    nt = m.node_tree
    for n in list(nt.nodes):
        if n.type != "OUTPUT_MATERIAL":
            nt.nodes.remove(n)
    out = next(n for n in nt.nodes if n.type == "OUTPUT_MATERIAL")
    nb = NB(nt)
    P = nb.objco()
    x, y, z = nb.sep(P)
    h = nb.math("DIVIDE", z, height, clamp=True)
    # slow domain warp so the ribbons are not perfect curves
    wc = nb.noise(nb.mapping(P, scale=(7.0, 7.0, 4.0)), scale=1.0, detail=1.0)[1]
    wx, wy, _ = nb.sep(wc)
    amp = nb.madd(h, 0.050, 0.004)
    px = nb.sub(nb.add(x, nb.mul(nb.sub(wx, 0.5), amp)), nb.mul(nb.mul(h, h), STEAM_DRIFT))
    py = nb.add(y, nb.mul(nb.sub(wy, 0.5), amp))
    total = None
    #        base x, base y, sway amp, sway freq, phase, twist rate, twist0, weight
    for bx, by, A, f, ph, om, th0, wgt in ((-0.009, 0.004, 0.034, 27.0, 0.4, 21.0, 0.3, 1.0),
                                           (0.011, -0.006, 0.040, 21.0, 2.6, -17.0, 1.9, 0.8),
                                           (0.001, 0.012, 0.028, 33.0, 4.4, 26.0, 4.0, 0.6)):
        cx = nb.add(bx, nb.mul(nb.mul(h, A), nb.math("SINE", nb.madd(z, f, ph))))
        cy = nb.add(by, nb.mul(nb.mul(h, A), nb.math("COSINE", nb.madd(z, f * 0.8, ph * 1.7))))
        dx, dy = nb.sub(px, cx), nb.sub(py, cy)
        th = nb.madd(z, om, th0)
        c, sn = nb.math("COSINE", th), nb.math("SINE", th)
        u = nb.add(nb.mul(dx, c), nb.mul(dy, sn))
        v = nb.sub(nb.mul(dy, c), nb.mul(dx, sn))
        wu = nb.madd(h, 0.018, 0.0065)
        wv = nb.madd(h, 0.0055, 0.0018)
        uu, vv = nb.math("DIVIDE", u, wu), nb.math("DIVIDE", v, wv)
        g = nb.math("EXPONENT", nb.mul(nb.add(nb.mul(uu, uu), nb.mul(vv, vv)), -1.0))
        g = nb.mul(g, wgt)
        total = g if total is None else nb.add(total, g)
    wn, _ = nb.noise(nb.comb(nb.mul(px, 18.0), nb.mul(py, 18.0), nb.mul(z, 8.0)), scale=1.0, detail=1.5,
                     rough=0.5, dist=0.6)
    brk = nb.mr(wn, 0.30, 0.72, 0.10, 1.25)
    fo = nb.sub(1.0, nb.smooth(h, 0.18, 0.95))
    fade = nb.mul(nb.smooth(h, 0.0, 0.05), nb.mul(fo, fo))
    dens = nb.mul(nb.mul(total, brk), nb.mul(fade, STEAM_DENSITY))
    vol = nb.node("ShaderNodeVolumePrincipled")
    nb.put(vol.inputs["Color"], (1.0, 1.0, 1.0))
    nb.put(vol.inputs["Density"], dens)
    nb.put(vol.inputs["Anisotropy"], 0.45)
    nt.links.new(vol.outputs[0], out.inputs["Volume"])
    return m


STEAM_DENSITY = 34.0
STEAM_DRIFT = -0.035


def make_materials():
    M = {}
    M["backdrop"] = m_backdrop()
    ca, sa = cos(-TABLE_ROT), sin(-TABLE_ROT)
    M["wood"] = m_wood("s1_wood_top", planks=True,
                       cup_ring=(ca * CUP_RING_XY[0] - sa * CUP_RING_XY[1], sa * CUP_RING_XY[0] + ca * CUP_RING_XY[1]))
    M["wood_leg"] = m_wood("s1_wood_leg", planks=False, tone=0.8, seed=5.0, rough=0.38, coat=0.25)
    M["brass"] = m_brass()
    M["brass_plain"] = m_brass("s1_brass_plain", radial=False)
    M["enamel"] = m_enamel()
    M["bulb"] = m_bulb()
    M["felt"] = m_simple("s1_felt", (0.012, 0.022, 0.016), rough=1.0, sheen=0.6, sheen_rough=0.6, spec=0.1)
    M["bakelite"] = m_simple("s1_bakelite", (0.012, 0.010, 0.009), rough=0.28, coat=0.3)
    M["steel"] = m_simple("s1_steel", (0.62, 0.62, 0.64), rough=0.3, metal=1.0)
    M["cord"] = m_cord()
    M["cloth_red"] = m_cloth("s1_cloth_oxblood", (0.24, 0.026, 0.028), sheen=0.25, pitch=0.0013)
    M["cloth_ochre"] = m_cloth("s1_cloth_ochre", (0.40, 0.235, 0.055), pitch=0.0017, sheen=0.2)
    M["cloth_navy"] = m_cloth("s1_cloth_navy", (0.006, 0.028, 0.120), sheen=0.06, spec=0.22, pitch=0.0013)
    M["paper"] = m_paper()
    M["gold"] = m_gold()
    M["ink"] = m_simple("s1_ink", (0.035, 0.022, 0.014), rough=0.6)
    M["headband"] = m_simple("s1_headband", (0.55, 0.09, 0.06), rough=0.7, sheen=0.4)
    M["ribbon"] = m_ribbon()
    M["mug"] = m_mug()
    M["coffee"] = m_coffee()
    M["cork"] = m_cork()
    M["terracotta"] = m_terracotta()
    M["soil"] = m_soil()
    M["leaf"] = m_leaf()
    M["leaf_citrus"] = m_leaf("s1_leaf_citrus", top_a=(0.012, 0.055, 0.012), top_b=(0.025, 0.10, 0.02),
                              vein_col=(0.12, 0.26, 0.06), n_veins=11.0, transl=0.28, rough=0.22)
    M["stem"] = m_stem()
    M["twig"] = m_stem("s1_twig", (0.09, 0.10, 0.03), (0.16, 0.13, 0.05))
    M["orange"] = m_orange()
    M["calyx"] = m_simple("s1_calyx", (0.10, 0.13, 0.035), rough=0.6)
    M["tortoise"] = m_acetate()
    M["lens"] = m_lens()
    M["pencil_paint"] = m_simple("s1_pencil_paint", (0.012, 0.014, 0.016), rough=0.30, coat=0.5, coat_rough=0.12)
    M["pencil_wood"] = m_simple("s1_pencil_wood", (0.62, 0.40, 0.23), rough=0.75)
    M["graphite"] = m_simple("s1_graphite", (0.045, 0.045, 0.05), rough=0.32, metal=0.7)
    M["eraser"] = m_simple("s1_eraser", (0.66, 0.26, 0.24), rough=0.9, sss=0.1, sss_radius=(0.003, 0.001, 0.001))
    return M


# =================================================================== backdrop + table
def build_backdrop(M):
    group("Backdrop")
    wall_y, cove = 2.7, 1.35
    prof = [(-7.0, 0.0), (wall_y - cove, 0.0)]
    n = 32
    for i in range(1, n + 1):
        a = radians(-90 + i * 90 / n)
        prof.append((wall_y - cove + cove * cos(a), cove + cove * sin(a)))
    prof.append((wall_y, 7.5))
    bm = bmesh.new()
    xs = (-9.0, -3.0, 3.0, 9.0)
    rows = [[bm.verts.new((x, y, z)) for x in xs] for (y, z) in prof]
    for a, b in zip(rows, rows[1:]):
        for i in range(len(xs) - 1):
            bm.faces.new((a[i], a[i + 1], b[i + 1], b[i]))
    return mesh_obj("Backdrop", bm, M["backdrop"], sharp=0)


def build_table(M):
    group("Table")
    R, T = TABLE_R, 0.030
    prof = fillet([(0, -T), (R - 0.012, -T, 0.004), (R, -T + 0.010, 0.006), (R, 0.0, 0.0042), (0, 0.0)], n=6)
    top = lathe("TableTop", prof, segs=160, mats=M["wood"], loc=(0, 0, TOP), rot=(0, 0, TABLE_ROT))
    # mounting plate + three splayed, tapered legs (grain runs along each leg = local X)
    lathe("TablePlate", fillet([(0, -0.008), (0.215, -0.008, 0.002), (0.215, 0.0, 0.002), (0, 0.0)]), segs=64,
          mats=M["bakelite"], loc=(0, 0, TOP - T))
    for i in range(3):
        a = radians(90 + i * 120)
        p_top = V((0.185 * cos(a), 0.185 * sin(a), TOP - T - 0.004))
        p_bot = V((0.300 * cos(a), 0.300 * sin(a), 0.0))
        ln = (p_bot - p_top).length
        leg = lathe(f"TableLeg{i}", fillet([(0, 0), (0.0205, 0, 0.002), (0.0125, ln - 0.03), (0.0105, ln, 0.002), (0, ln)]),
                    segs=32, mats=M["wood_leg"])
        leg.data.transform(Matrix.Rotation(radians(90), 4, "Y"))      # mesh Z -> X so the grain axis is the leg axis
        leg.location = p_top
        aim(leg, p_bot - p_top, axis=(1, 0, 0))
    return top


# =================================================================== lamp
def build_lamp(M):
    group("Lamp")
    br = M["brass"]
    root = empty("Lamp", loc=(LAMP_XY[0], LAMP_XY[1], TOP), rot=(0, 0, LAMP_YAW))
    lathe("LampFelt", [(0, 0), (0.0715, 0), (0.0715, 0.0013), (0, 0.0013)], segs=96, mats=M["felt"], parent=root)
    # ---- weighted base: lower tier, shallow dome, collar
    dome = [(0.0595 - 0.0405 * t, 0.0128 + 0.0122 * sin(t * pi / 2)) for t in [k / 12 for k in range(13)]]
    prof = fillet([(0, 0.0013), (0.0745, 0.0013, 0.0012), (0.0745, 0.0085, 0.0035), (0.0625, 0.0100, 0.0012),
                   (0.0605, 0.0126, 0.0008)] + dome +
                  [(0.0150, 0.0254, 0.0010), (0.0150, 0.0340, 0.0012), (0.0105, 0.0356, 0.0008),
                   (0.0105, 0.0440, 0.0010), (0.0064, 0.0455), (0.0064, 0.0475), (0, 0.0475)])
    lathe("LampBase", prof, segs=128, mats=br, parent=root)
    knurl("LampBaseKnurl", 0.0158, 0.0272, 0.0318, teeth=40, mats=br, parent=root)
    # ---- toggle switch on the dome
    phi = radians(-104)
    rs = 0.043
    t = (0.0595 - rs) / 0.0405
    zs = 0.0128 + 0.0122 * sin(t * pi / 2)
    slope = 0.0122 * cos(t * pi / 2) * (pi / 2) / 0.0405
    nrm = V((cos(phi) * slope, sin(phi) * slope, 1.0)).normalized()
    sw = empty("LampSwitch", loc=(rs * cos(phi), rs * sin(phi), zs - 0.0006), parent=root)
    aim(sw, nrm)
    lathe("LampSwitchNut", [(0, 0), (0.0066, 0), (0.0066, 0.0026), (0, 0.0026)], segs=6, mats=M["brass_plain"],
          sharp=25, parent=sw)
    lathe("LampSwitchCollar", fillet([(0, 0.0026), (0.0046, 0.0026), (0.0046, 0.0052, 0.0008), (0.0026, 0.0060), (0, 0.0060)]),
          segs=32, mats=br, parent=sw)
    lever = lathe("LampSwitchLever", fillet([(0, 0), (0.0017, 0), (0.0026, 0.0105, 0.0022), (0, 0.0128)], n=6),
                  segs=24, mats=M["steel"], parent=sw, loc=(0, 0, 0.0052))
    lever.rotation_euler = (radians(20), 0, radians(40))
    # ---- stem, elbow joint with thumb wheel, arm
    E = V((0, 0, 0.215))
    Hh = V((0.150, 0, 0.318))
    tube("LampStem", (0, 0, 0.046), E - V((0, 0, 0.010)), 0.0058, mats=br, segs=32, parent=root)
    lathe("LampStemFerrule", fillet([(0, 0), (0.0074, 0, 0.0010), (0.0074, 0.014, 0.0010), (0, 0.014)]), segs=32, mats=br,
          parent=root, loc=E - V((0, 0, 0.0235)))
    ry = (radians(-90), 0, 0)                       # local Z -> +Y (the joint axis)
    lathe("LampElbow", fillet([(0, -0.0078), (0.0108, -0.0078, 0.0012), (0.0108, 0.0078, 0.0012), (0, 0.0078)]),
          segs=48, mats=br, parent=root, loc=E, rot=ry)
    knurl("LampElbowWheel", 0.0122, -0.0152, -0.0092, teeth=22, depth=0.09, mats=br, parent=root, loc=E, rot=ry)
    lathe("LampElbowCap", fillet([(0, -0.0172), (0.0050, -0.0170), (0.0086, -0.0152), (0, -0.0152)]), segs=32, mats=br,
          parent=root, loc=E, rot=ry)
    lathe("LampElbowWasher", [(0, -0.0092), (0.0070, -0.0092), (0.0070, -0.0078), (0, -0.0078)], segs=32,
          mats=M["steel"], parent=root, loc=E, rot=ry)
    lathe("LampElbowNut", [(0, 0.0078), (0.0064, 0.0078), (0.0064, 0.0118), (0, 0.0118)], segs=6, mats=M["brass_plain"],
          sharp=25, parent=root, loc=E, rot=ry)
    lathe("LampElbowAcorn", fillet([(0, 0.0118), (0.0046, 0.0118), (0.0038, 0.0140, 0.002), (0, 0.0152)]), segs=24,
          mats=br, parent=root, loc=E, rot=ry)
    ad = (Hh - E).normalized()
    arm = tube("LampArm", E + ad * 0.004, Hh - ad * 0.006, 0.0054, mats=br, segs=32, parent=root)
    for nm, pos in (("LampArmFerruleA", E + ad * 0.0105), ("LampArmFerruleB", Hh - ad * 0.024)):
        f = lathe("%s" % nm, fillet([(0, 0), (0.0070, 0, 0.0010), (0.0070, 0.013, 0.0010), (0, 0.013)]), segs=32, mats=br,
                  parent=root, loc=pos)
        aim(f, ad)
    uvsphere("LampBall", 0.0108, mats=M["brass_plain"], u=32, v=16, parent=root, loc=Hh)
    # ---- head: frame with +Z pointing from the rim plane up to the ball
    c, s = cos(-LAMP_YAW), sin(-LAMP_YAW)
    tx, ty = LAMP_AIM[0] - LAMP_XY[0], LAMP_AIM[1] - LAMP_XY[1]
    tgt = V((c * tx - s * ty, s * tx + c * ty, 0.0))
    a = (tgt - Hh).normalized()
    Ls = 0.118
    head = empty("LampHead", loc=Hh + a * (Ls + 0.040), parent=root)
    aim(head, -a)
    r_rim, r_neck, z_bell = 0.0660, 0.0245, 0.100
    wall = 0.0012

    def r_out(z):
        if z >= z_bell:
            return r_neck
        return r_neck + (r_rim - r_neck) * (1 - (z - 0.004) / (z_bell - 0.004)) ** 1.35

    zs_list = [0.004 + (z_bell - 0.004) * k / 22 for k in range(23)]
    inner = [(r_neck - wall, Ls)] + [(r_out(z) - wall, z) for z in reversed(zs_list)]
    lip = [(r_rim - wall + 0.0003, 0.0016), (r_rim - 0.0003, -0.0002), (r_rim + 0.0008, 0.0003),
           (r_rim + 0.0011, 0.0016), (r_rim + 0.0005, 0.0030)]
    outer = [(r_out(z), z) for z in zs_list] + [(r_neck, Ls)]
    prof = inner + lip + outer
    ids = [1] * (len(inner) - 1) + [1, 0, 0, 0, 0] + [0] * (len(outer))
    lathe("LampShade", prof, segs=128, mats=[br, M["enamel"]], mat_ids=ids[:len(prof) - 1], parent=head, sharp=50)
    cup = fillet([(0.0238, Ls - 0.0015), (0.0262, Ls - 0.0005, 0.0006), (0.0262, Ls + 0.0042, 0.0008),
                  (0.0236, Ls + 0.0060, 0.0008), (0.0236, Ls + 0.0180, 0.0070), (0.0085, Ls + 0.0292, 0.0030),
                  (0.0064, Ls + 0.0310), (0.0064, Ls + 0.0360), (0, Ls + 0.0360)], n=6)
    lathe("LampCup", cup, segs=96, mats=br, parent=head)
    knurl("LampCupKnurl", 0.0268, Ls + 0.0003, Ls + 0.0038, teeth=56, depth=0.05, mats=br, parent=head)
    lathe("LampSocket", fillet([(0, 0.097), (0.0150, 0.097, 0.001), (0.0150, Ls + 0.002), (0, Ls + 0.002)]), segs=48,
          mats=M["bakelite"], parent=head)
    thread = [(0, 0.085)]
    for k in range(7):
        z0 = 0.085 + k * 0.0018
        thread += [(0.0136, z0), (0.0128, z0 + 0.0009)]
    thread += [(0.0136, 0.0976), (0, 0.0976)]
    lathe("LampSocketShell", thread, segs=48, mats=M["brass_plain"], parent=head)
    zb, rb = 0.066, 0.0225
    bulb = [(0, zb - rb)]
    for k in range(1, 20):
        an = radians(180 - k * 145 / 19)
        bulb.append((rb * sin(an), zb + rb * cos(an)))
    bulb += [(0.0125, 0.0875), (0.0125, 0.0915), (0, 0.0915)]
    bo = lathe("LampBulb", bulb, segs=48, mats=M["bulb"], parent=head)
    bo.visible_shadow = False
    ld = bpy.data.lights.new("LampLight", "POINT")
    ld.energy = LAMP_WATTS
    ld.color = (1.0, 0.66, 0.36)
    ld.shadow_soft_size = 0.017
    lo = bpy.data.objects.new("LampLight", ld)
    lo.location = (0, 0, zb)
    link(lo, head)
    # ---- cloth flex leaving the back of the base, across the table and over the edge
    bpy.context.view_layer.update()
    mw = root.matrix_world
    exl = radians(147)                                    # exit direction in lamp space (back-left in the world)
    ex = mw @ V((0.0745 * cos(exl), 0.0745 * sin(exl), 0.0048))
    radial = (ex - V((LAMP_XY[0], LAMP_XY[1], ex.z))).normalized()
    lathe_at("LampGrommet", fillet([(0, 0), (0.0052, 0, 0.0008), (0.0040, 0.007, 0.001), (0.0030, 0.0085), (0, 0.0085)]),
             ex - radial * 0.001, radial, segs=24, mats=M["bakelite"])
    rc = 0.0027

    def pol(r, deg, z):
        return V((r * cos(radians(deg)), r * sin(radians(deg)), z))

    zt = TOP + rc
    th0 = degrees(atan2(ex.y, ex.x))
    r0 = sqrt(ex.x ** 2 + ex.y ** 2)
    R = TABLE_R
    ctrl = [ex, ex + radial * 0.010 + V((0, 0, -0.0012)), pol(r0 + 0.016, th0 - 2.6, zt), pol(r0 + 0.024, th0 - 6.5, zt),
            pol(r0 + 0.027, th0 - 10.5, zt), pol(R - 0.012, th0 - 13.0, zt), pol(R - 0.004, th0 - 14.4, zt - 0.0001),
            pol(R + 0.0010, th0 - 14.8, TOP + 0.0009), pol(R + 0.0032, th0 - 15.0, TOP - 0.0050),
            pol(R + 0.0032, th0 - 15.1, TOP - 0.030), pol(R + 0.0030, th0 - 15.3, TOP - 0.12),
            pol(R + 0.0010, th0 - 16.0, TOP - 0.32), pol(R - 0.004, th0 - 17.5, 0.04)]
    sweep("LampCord", crspline(ctrl, n=10), circle_sec(10, rc), mats=M["cord"], sharp=0)
    return root


LAMP_WATTS = 12.5


# =================================================================== books
def build_book(name, L, W, H, cloth, M, pos, yaw, title=None, author=None, flip=False, cover_title=None,
               foil="gold", title_size=None):
    """Case-bound book lying flat.  Local frame: X = spine direction (head at -X), spine at -Y, Z = thickness."""
    root = empty(name, loc=pos, rot=(0, 0, yaw + (pi if flip else 0)))
    tb, sq, sb = 0.0027, 0.0030, 0.0032
    ys = -W / 2
    # ---- cover: C-shaped centre line (fore-edge bottom -> hinge groove -> rounded spine -> top), thickened
    cl = [(W / 2, tb / 2), (ys + 0.0130, tb / 2), (ys + 0.0108, tb / 2 + 0.0004), (ys + 0.0086, tb / 2 + 0.0011),
          (ys + 0.0064, tb / 2 + 0.0004), (ys + 0.0046, tb / 2)]
    na = 14
    for k in range(na + 1):
        an = radians(-90 + 180 * k / na)
        cl.append((ys + sb - sb * cos(an), H / 2 + (H / 2 - tb / 2) * sin(an)))
    cl += [(ys + 0.0046, H - tb / 2), (ys + 0.0064, H - tb / 2 - 0.0004), (ys + 0.0086, H - tb / 2 - 0.0011),
           (ys + 0.0108, H - tb / 2 - 0.0004), (ys + 0.0130, H - tb / 2), (W / 2, H - tb / 2)]
    K = len(cl)
    outer, inner, arc = [], [], [0.0]
    for k in range(K):
        a2, b2 = V(cl[max(k - 1, 0)]), V(cl[min(k + 1, K - 1)])
        t = (b2 - a2).normalized()
        nrm = V((-t.y, t.x))
        c2 = V(cl[k])
        outer.append(c2 + nrm * tb / 2)
        inner.append(c2 - nrm * tb / 2)
        if k:
            arc.append(arc[-1] + (V(cl[k]) - V(cl[k - 1])).length)
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new("UVMap")
    X = (-L / 2, L / 2)
    VO = [[bm.verts.new((x, q.x, q.y)) for q in outer] for x in X]
    VI = [[bm.verts.new((x, q.x, q.y)) for q in inner] for x in X]

    def quad(vs, uvs):
        f = bm.faces.new(vs)
        for lp, c in zip(f.loops, uvs):
            lp[uvl].uv = c

    for k in range(K - 1):
        quad((VO[0][k], VO[1][k], VO[1][k + 1], VO[0][k + 1]), ((0, arc[k]), (L, arc[k]), (L, arc[k + 1]), (0, arc[k + 1])))
        quad((VI[0][k], VI[0][k + 1], VI[1][k + 1], VI[1][k]), ((0, arc[k]), (0, arc[k + 1]), (L, arc[k + 1]), (L, arc[k])))
        for xi in (0, 1):
            quad((VO[xi][k], VO[xi][k + 1], VI[xi][k + 1], VI[xi][k]),
                 ((0, arc[k]), (0, arc[k + 1]), (tb, arc[k + 1]), (tb, arc[k])))
    quad((VO[0][0], VI[0][0], VI[1][0], VO[1][0]), ((0, 0), (0, tb), (L, tb), (L, 0)))
    quad((VO[0][K - 1], VO[1][K - 1], VI[1][K - 1], VI[0][K - 1]), ((0, 0), (L, 0), (L, tb), (0, tb)))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    cover = mesh_obj(name + "Cover", bm, cloth, parent=root, sharp=45)
    md = cover.modifiers.new("Bevel", "BEVEL")
    md.width, md.segments, md.limit_method, md.angle_limit = 0.0006, 2, "ANGLE", radians(50)
    wn = cover.modifiers.new("WeightedNormal", "WEIGHTED_NORMAL")
    wn.keep_sharp = True
    # ---- page block (chamfered where the hinge grooves bite in)
    y0, y1 = ys + sb + tb / 2 + 0.0006, W / 2 - sq
    z0, z1 = tb + 0.00005, H - tb - 0.00005
    pp = [(y0, z0 + 0.0030), (y0 + 0.0080, z0), (y1, z0), (y1, z1), (y0 + 0.0080, z1), (y0, z1 - 0.0030)]
    bm = bmesh.new()
    xa, xb = -L / 2 + sq, L / 2 - sq
    A = [bm.verts.new((xa, q[0], q[1])) for q in pp]
    B = [bm.verts.new((xb, q[0], q[1])) for q in pp]
    for k in range(len(pp)):
        k2 = (k + 1) % len(pp)
        bm.faces.new((A[k], A[k2], B[k2], B[k]))
    bm.faces.new(A)
    bm.faces.new(list(reversed(B)))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh_obj(name + "Pages", bm, M["paper"], parent=root, smooth=False)
    # ---- head / tail bands
    for sx, xx in ((-1, xa + 0.0006), (1, xb - 0.0006)):
        lathe(name + ("HeadBand" if sx < 0 else "TailBand"),
              [(0, z0 + 0.003), (0.0014, z0 + 0.003), (0.0014, z1 - 0.003), (0, z1 - 0.003)], segs=12,
              mats=M["headband"], parent=root, loc=(xx, y0 + 0.0002, 0))
    fm = M[foil]
    # ---- spine: title, author, rules that follow the spine curve
    if title:
        ts = title_size or H * 0.40
        text_obj(name + "Title", title, ts, fm, FONT_SERIF, spacing=1.08, parent=root,
                 loc=(-L * 0.10, ys - tb / 2 - 0.00015, H / 2), rot=(pi / 2, 0, 0))
    if author:
        text_obj(name + "Author", author, H * 0.21, fm, FONT_SANS, spacing=1.25, parent=root,
                 loc=(L * 0.33, ys - tb / 2 - 0.00015, H / 2), rot=(pi / 2, 0, 0))
    bm = bmesh.new()
    off = 0.00012
    i0, i1 = 5, 5 + na
    for xr, wr in ((-L / 2 + 0.011, 0.0007), (-L / 2 + 0.0135, 0.0003), (L / 2 - 0.0117, 0.0007),
                   (L / 2 - 0.0138, 0.0003), (L * 0.215, 0.0003)):
        strip = []
        for k in range(i0, i1 + 1):
            a2, b2 = V(cl[max(k - 1, 0)]), V(cl[min(k + 1, K - 1)])
            t = (b2 - a2).normalized()
            q = outer[k] + V((-t.y, t.x)) * off
            strip.append((bm.verts.new((xr, q.x, q.y)), bm.verts.new((xr + wr, q.x, q.y))))
        for sa, sb2 in zip(strip, strip[1:]):
            bm.faces.new((sa[0], sa[1], sb2[1], sb2[0]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    mesh_obj(name + "Rules", bm, fm, parent=root, sharp=0)
    # ---- front cover: double rule frame + small title
    if cover_title:
        bm = bmesh.new()
        zc = H + 0.00012

        def frame(xl, xr, yl, yr, w):
            for (ax, ay, bx, by) in ((xl, yl, xr, yl + w), (xl, yr - w, xr, yr), (xl, yl, xl + w, yr), (xr - w, yl, xr, yr)):
                bm.faces.new([bm.verts.new(c) for c in ((ax, ay, zc), (bx, ay, zc), (bx, by, zc), (ax, by, zc))])

        frame(-L / 2 + 0.010, L / 2 - 0.010, ys + 0.024, W / 2 - 0.010, 0.0008)
        frame(-L / 2 + 0.0125, L / 2 - 0.0125, ys + 0.0265, W / 2 - 0.0125, 0.0003)
        mesh_obj(name + "CoverFrame", bm, fm, parent=root, smooth=False)
        text_obj(name + "CoverTitle", cover_title, 0.0095, fm, FONT_SERIF, spacing=1.1, parent=root,
                 loc=(-L * 0.20, 0.008, zc), rot=(0, 0, pi / 2))
    return root, cover


def build_books(M):
    group("Books")
    bx, by = BOOKS_XY
    specs = [  # name, L, W, H, cloth, yaw deg, offset, title, author, flip
        ("BookA", 0.228, 0.158, 0.034, "cloth_red", 40, (0.0, 0.0), "A FIELD GUIDE TO EVENINGS", "M. ARDEN", False, 0.0098),
        ("BookB", 0.208, 0.146, 0.026, "cloth_ochre", 36, (0.004, -0.003), "NOTES ON LIGHT", "H. VALE", True, None),
        ("BookC", 0.196, 0.136, 0.022, "cloth_navy", 45, (-0.004, 0.003), "THE QUIET HOUR", "E. LOWRY", False, None),
    ]
    z = TOP
    out = []
    for nm, L, W, H, cloth, yaw, (ox, oy), title, author, flip, tsz in specs:
        root, cover = build_book(nm, L, W, H, M[cloth], M, (bx + ox, by + oy, z), radians(yaw), title=title,
                                 author=author, flip=flip, cover_title=("THE QUIET HOUR" if nm == "BookC" else None),
                                 foil=("ink" if cloth == "cloth_ochre" else "gold"), title_size=tsz)
        out.append(dict(root=root, L=L, W=W, H=H, z=z))
        z += H
    # ---- ribbon marker: out of the tail of the middle book, over the lower cover, down to the table
    bpy.context.view_layer.update()
    A, B = out[0], out[1]
    mb, ma = B["root"].matrix_world, A["root"].matrix_world
    yr = 0.012
    xs = B["L"] / 2 - 0.003

    def fa(xm):   # book-A local x of a point given in book-B local coords
        return (ma.inverted() @ (mb @ V((xm, yr, 0.0)))).x

    f0, f1 = fa(0.0), fa(1.0)
    # the tail end of flipped B points to the head end (-X) of A
    edge = (-A["L"] / 2 - f0) / (f1 - f0)
    zr = B["H"] * 0.56
    ha = A["H"]
    cxr = B["L"] / 2 + 0.0012
    ctrl = [(xs - 0.030, yr, zr), (xs - 0.002, yr, zr), (cxr, yr, zr - 0.0002), (cxr + 0.0026, yr + 0.0003, zr - 0.0013),
            (cxr + 0.0042, yr + 0.0006, zr - 0.0040), (cxr + 0.0047, yr + 0.001, zr - 0.0080),
            (cxr + 0.0052, yr + 0.002, 0.0042),
            ((B["L"] / 2 + 0.014 + edge) / 2, yr + 0.003, 0.0006), (edge - 0.002, yr + 0.004, 0.0006),
            (edge + 0.0030, yr + 0.005, -0.0030), (edge + 0.0046, yr + 0.007, -ha * 0.5),
            (edge + 0.0070, yr + 0.010, -ha + 0.0050), (edge + 0.0150, yr + 0.016, -ha + 0.0007),
            (edge + 0.0300, yr + 0.028, -ha + 0.0005), (edge + 0.0420, yr + 0.046, -ha + 0.0005),
            (edge + 0.0440, yr + 0.064, -ha + 0.0005)]
    sec = [(0.00016, -0.0032), (0.00016, 0.0032), (-0.00016, 0.0032), (-0.00016, -0.0032)]
    rb = sweep("BookRibbon", crspline(ctrl, n=8), sec, mats=M["ribbon"], parent=B["root"], sharp=60)
    return out, z


# =================================================================== mug, coaster, steam
def build_mug(M):
    group("Mug")
    mx, my = MUG_XY
    lathe("Coaster", fillet([(0, 0), (0.052, 0, 0.0012), (0.052, 0.0050, 0.0012), (0, 0.0050)]), segs=96,
          mats=M["cork"], loc=(mx, my, TOP), rot=(0, 0, 0.7))
    root = empty("Mug", loc=(mx, my, TOP + 0.0050), rot=(0, 0, radians(-32)))
    Hm = 0.092
    out_pts = crspline([(0.0355, 0.0040, 0), (0.0392, 0.016, 0), (0.0412, 0.038, 0), (0.0410, 0.064, 0),
                        (0.0399, 0.0865, 0)], n=7)
    in_pts = crspline([(0.0366, 0.0865, 0), (0.0375, 0.062, 0), (0.0376, 0.034, 0), (0.0352, 0.0145, 0),
                       (0.0270, 0.0082, 0)], n=7)
    prof = [(0, 0.0032), (0.0265, 0.0032), (0.0285, 0.0), (0.0330, 0.0), (0.0348, 0.0012)]
    prof += [(q.x, q.y) for q in out_pts]
    prof += [(0.0397, 0.0895), (0.0391, 0.0913), (0.0382, 0.0920), (0.0373, 0.0913), (0.0367, 0.0895)]
    prof += [(q.x, q.y) for q in in_pts]
    prof += [(0.014, 0.0068), (0, 0.0066)]
    lathe("MugBody", prof, segs=128, mats=M["mug"], parent=root, sharp=0)
    # handle: fat ellipse swept on a D-shaped path, swelling where it meets the wall
    hp = crspline([(0.0385, 0, 0.0745), (0.0480, 0, 0.0790), (0.0610, 0, 0.0760), (0.0690, 0, 0.0620),
                   (0.0690, 0, 0.0440), (0.0600, 0, 0.0290), (0.0480, 0, 0.0220), (0.0385, 0, 0.0215)], n=8)

    def hs(t):
        e = 1.0 + 0.55 * (max(0.0, 1 - t / 0.16) ** 2 + max(0.0, 1 - (1 - t) / 0.16) ** 2)
        return (e, e)

    sweep("MugHandle", hp, ellipse_sec(0.0064, 0.0041, 20, power=2.4), mats=M["mug"], scale=hs, up=(0, 1, 0),
          parent=root, sharp=0)
    lathe("Coffee", [(0, 0.0778), (0.022, 0.0778), (0.033, 0.0780), (0.0362, 0.0786), (0.03745, 0.0796)], segs=96,
          mats=M["coffee"], parent=root, sharp=0)
    # steam volume
    hsb = 0.30
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for vt in bm.verts:
        vt.co.x *= 0.20
        vt.co.y *= 0.16
        vt.co.z = (vt.co.z + 0.5) * hsb
    st = mesh_obj("Steam", bm, m_steam(hsb), loc=(mx, my, TOP + 0.0050 + 0.0815), smooth=False)
    st.visible_shadow = False
    return root


# =================================================================== plant
def leaf_mesh(bm, uvl, petiole, blade, width, lean0, lean1, fold=0.5, wave=0.004, twist=0.0, seed=0.0,
              stem_r=0.0021, pw=1.7, shape_pow=0.62):
    """Petiole + blade arching in the local XZ plane (toward +X).  Blade faces: material 0, petiole: 1."""
    S = petiole + blade
    steps = 260
    pts, ths = [V((0, 0, 0))], []
    for i in range(steps):
        s = (i + 0.5) / steps
        th = lean0 + (lean1 - lean0) * s ** pw
        ths.append(th)
        pts.append(pts[-1] + V((sin(th), 0, cos(th))) * (S / steps))

    def at(s):
        f = max(0.0, min(1.0, s / S)) * steps
        i = min(int(f), steps - 1)
        c = pts[i].lerp(pts[i + 1], f - i)
        th = ths[i]
        return c, V((sin(th), 0, cos(th))), V((-cos(th), 0, sin(th)))

    nB, nA = 30, 6
    rows = []
    rnd = random.Random(int(seed * 1000) + 3)
    ph1, ph2 = rnd.uniform(0, 6.28), rnd.uniform(0, 6.28)
    nw = rnd.uniform(2.2, 3.4)
    for j in range(nB + 1):
        t = j / nB
        C, T, N = at(petiole + t * blade)
        w = width / 2 * max(0.03, sin(pi * t ** shape_pow) ** 0.9 * (1 - 0.10 * t))
        fo = fold * (1 - 0.55 * t)
        row = []
        for i in range(-nA, nA + 1):
            a = i / nA
            y = a * w
            hgt = fo * abs(y) * (1 - 0.35 * abs(a)) - 0.12 * w * a * a * t
            hgt += wave * sin(2 * pi * nw * t + (ph1 if a > 0 else ph2)) * a * a * sin(pi * t) ** 0.6
            tw = twist * t
            yy, hh = y * cos(tw) - hgt * sin(tw), y * sin(tw) + hgt * cos(tw)
            row.append((bm.verts.new(C + V((0, 1, 0)) * yy + N * hh), (0.5 + 0.5 * a, t)))
        rows.append(row)
    for j in range(nB):
        for i in range(2 * nA):
            q = (rows[j][i], rows[j + 1][i], rows[j + 1][i + 1], rows[j][i + 1])
            f = bm.faces.new([v for v, _ in q])
            f.material_index = 0
            for lp, (_, uv) in zip(f.loops, q):
                lp[uvl].uv = uv
    # petiole continuing as the midrib under the blade
    path, rad = [], []
    nP = 14
    for k in range(nP + 1):
        s = petiole * k / nP
        C, T, N = at(s)
        path.append(C)
        rad.append(stem_r * (1 - 0.30 * k / nP))
    for k in range(1, 17):
        t = 0.9 * k / 16
        C, T, N = at(petiole + t * blade)
        r = stem_r * 0.70 * (1 - t / 0.92)
        path.append(C - N * (r * 0.85))
        rad.append(max(r, 0.00015))
    lens = [0.0]
    for a2, b2 in zip(path, path[1:]):
        lens.append(lens[-1] + (b2 - a2).length)

    # scale() gets normalised arc length; remap through the stored lengths
    def rscale(t):
        d = t * lens[-1]
        for i in range(len(lens) - 1):
            if d <= lens[i + 1] + 1e-9:
                f = (d - lens[i]) / max(lens[i + 1] - lens[i], 1e-9)
                return rad[i] + (rad[i + 1] - rad[i]) * f
        return rad[-1]

    sweep_bm(bm, path, circle_sec(8), scale=rscale, up=(0, 1, 0), mat_index=1, uvl=uvl)


def build_leaf(name, base, azim, mats, parent=None, **kw):
    bm = bmesh.new()
    uvl = bm.loops.layers.uv.new("UVMap")
    leaf_mesh(bm, uvl, **kw)
    ob = mesh_obj(name, bm, mats, loc=base, rot=(0, 0, azim), sharp=0, parent=parent)
    return ob


def build_plant(M):
    group("Plant")
    px, py = PLANT_XY
    lathe("PlantSaucer", fillet([(0, 0), (0.050, 0, 0.002), (0.066, 0.013, 0.002), (0.0625, 0.0135, 0.0006),
                                 (0.0480, 0.0045, 0.002), (0, 0.0045)]), segs=96, mats=M["terracotta"],
          loc=(px, py, TOP))
    zp = TOP + 0.0046
    hp = 0.108
    prof = fillet([(0, 0.0), (0.0445, 0.0, 0.003), (0.0590, hp - 0.024, 0.0010), (0.0655, hp - 0.0235, 0.0020),
                   (0.0672, hp, 0.0022), (0.0590, hp, 0.0016), (0.0560, hp - 0.016), (0.0410, 0.006), (0, 0.006)], n=5)
    lathe("PlantPot", prof, segs=128, mats=M["terracotta"], loc=(px, py, zp))
    # soil: bumpy disc
    zsoil = hp - 0.015
    bm = bmesh.new()
    rings = 12
    cen = bm.verts.new((0, 0, 0.004))
    prev = None
    for k in range(1, rings + 1):
        r = 0.0562 * k / rings
        ring = []
        for i in range(48):
            a = 2 * pi * i / 48
            p3 = V((r * cos(a), r * sin(a), 0))
            p3.z = 0.004 * (1 - (k / rings) ** 2) + 0.0035 * mnoise.noise(p3 * 55) + 0.0015 * mnoise.noise(p3 * 160)
            ring.append(bm.verts.new(p3))
        for i in range(48):
            j = (i + 1) % 48
            if prev is None:
                bm.faces.new((cen, ring[i], ring[j]))
            else:
                bm.faces.new((prev[i], ring[i], ring[j], prev[j]))
        prev = ring
    mesh_obj("PlantSoil", bm, M["soil"], loc=(px, py, zp + zsoil), sharp=0)
    zb = zp + zsoil + 0.002
    S = PLANT_SCALE
    leaves = [  # azimuth, petiole, blade, width, lean0, lean1, twist, fold
        (18, 0.105, 0.135, 0.062, 16, 78, 0.25, 0.45),
        (62, 0.125, 0.140, 0.064, 10, 58, -0.2, 0.5),
        (112, 0.115, 0.125, 0.056, 12, 50, 0.3, 0.5),
        (158, 0.100, 0.105, 0.050, 9, 36, -0.15, 0.6),
        (203, 0.085, 0.110, 0.052, 14, 66, 0.2, 0.5),
        (251, 0.078, 0.118, 0.055, 24, 104, -0.3, 0.42),
        (300, 0.095, 0.132, 0.060, 20, 92, 0.25, 0.45),
        (341, 0.085, 0.112, 0.052, 28, 108, -0.2, 0.42),
        (88, 0.140, 0.128, 0.056, 5, 34, 0.15, 0.55),
        (228, 0.128, 0.115, 0.050, 4, 26, -0.25, 0.6),
        (38, 0.132, 0.122, 0.054, 7, 42, 0.2, 0.55),
        (318, 0.118, 0.110, 0.048, 8, 44, -0.1, 0.55),
        (140, 0.105, 0.080, 0.026, 2, 10, 0.5, 1.1),
        (330, 0.085, 0.070, 0.022, 3, 13, -0.6, 1.2),
        (275, 0.060, 0.105, 0.050, 30, 112, 0.2, 0.40),
        (5, 0.070, 0.110, 0.052, 32, 110, -0.2, 0.40),
        (130, 0.070, 0.100, 0.046, 28, 100, 0.2, 0.42),
        (190, 0.062, 0.088, 0.042, 24, 84, -0.2, 0.45),
        (75, 0.100, 0.115, 0.052, 14, 62, 0.1, 0.5),
        (285, 0.105, 0.118, 0.050, 12, 56, -0.15, 0.5),
    ]
    rnd = random.Random(21)
    for i, (az, pe, bl, wd, l0, l1, tw, fo) in enumerate(leaves):
        a = radians(az + rnd.uniform(-5, 5))
        r0 = rnd.uniform(0.004, 0.016)
        base = (px + r0 * cos(a), py + r0 * sin(a), zb)
        build_leaf(f"PlantLeaf{i:02d}", base, a, [M["leaf"], M["stem"]], petiole=pe * S, blade=bl * S, width=wd * S * 1.14,
                   lean0=radians(l0), lean1=radians(l1), twist=tw, fold=fo, wave=0.0035 * S, seed=i + 1.0)


PLANT_SCALE = 0.80


# =================================================================== oranges
def build_orange(name, pos, R, rot, M, seed, leaf=False):
    root = empty(name, loc=(pos[0], pos[1], TOP + R), rot=rot)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=72, v_segments=44, radius=1.0)
    sv = V((seed * 3.1, seed * 1.7, seed * 5.3))
    for vt in bm.verts:
        d = vt.co.normalized()
        th = acos(max(-1.0, min(1.0, d.z)))
        r = 1.0 - 0.085 * exp(-(th / 0.20) ** 2) - 0.028 * exp(-((pi - th) / 0.15) ** 2)
        r += 0.022 * mnoise.noise(d * 1.5 + sv) + 0.008 * mnoise.noise(d * 4.0 + sv)
        vt.co = d * (r * R)
        vt.co.z *= 0.93
    body = mesh_obj(name + "Body", bm, M["orange"], parent=root, sharp=0)
    zt = R * 0.93 * (1 - 0.085)
    # calyx: little five-pointed star with a stub
    bm = bmesh.new()
    cen = bm.verts.new((0, 0, 0.0016))
    ring = []
    for i in range(10):
        a = 2 * pi * i / 10
        rr = 0.0052 if i % 2 == 0 else 0.0029
        ring.append(bm.verts.new((rr * cos(a), rr * sin(a), 0.0002 if i % 2 == 0 else 0.0009)))
    for i in range(10):
        bm.faces.new((cen, ring[i], ring[(i + 1) % 10]))
    mesh_obj(name + "Calyx", bm, M["calyx"], parent=root, loc=(0, 0, zt + 0.0003), sharp=0)
    objs = [body]
    if leaf:
        twig = crspline([(0, 0, zt), (0.0008, 0.0003, zt + 0.006), (0.0030, 0.0012, zt + 0.012),
                         (0.0075, 0.002, zt + 0.0165)], n=6)
        sweep(name + "Twig", twig, circle_sec(8, 0.0016), mats=M["twig"], parent=root, sharp=0,
              scale=lambda t: 1.0 - 0.25 * t)
        lf = build_leaf(name + "Leaf", (0.0030, 0.0012, zt + 0.0115), radians(20), [M["leaf_citrus"], M["twig"]],
                        parent=root, petiole=0.009, blade=0.062, width=0.030, lean0=radians(38), lean1=radians(96),
                        twist=0.35, fold=0.35, wave=0.0015, seed=31.0, stem_r=0.0011, shape_pow=0.75)
        objs.append(lf)
    else:
        lathe(name + "Stub", [(0, 0), (0.0015, 0), (0.0013, 0.0028), (0, 0.0030)], segs=10, mats=M["twig"],
              parent=root, loc=(0, 0, zt + 0.001))
    drop(root, [body], TOP)
    if len(objs) > 1:
        print("  %s leaf min z above table: %.4f" % (name, world_min_z(objs[1:]) - TOP))
    return root


def build_oranges(M):
    group("Fruit")
    build_orange("OrangeA", ORANGE_1, ORANGE_1[2], (radians(34), radians(-22), radians(40)), M, 1.0, leaf=True)
    build_orange("OrangeB", ORANGE_2, ORANGE_2[2], (radians(-62), radians(30), radians(200)), M, 2.3, leaf=False)


# =================================================================== reading glasses
def build_glasses(M, surface_z, pos_xy, yaw):
    group("Glasses")
    root = empty("Glasses", loc=(pos_xy[0], pos_xy[1], surface_z))
    ac = M["tortoise"]
    lr, cx = 0.0222, 0.0335
    Rr = lr + 0.0010
    parts_front, parts_back = [], []
    for s, nm in ((-1, "L"), (1, "R")):
        path = []
        for k in range(48):
            a = 2 * pi * k / 48
            rr = Rr * (1 - 0.045 * max(0.0, sin(a)) ** 2)
            path.append(V((s * cx + rr * cos(a), 0, rr * sin(a))))
        parts_front.append(sweep("GlassesRim" + nm, path, ellipse_sec(0.0021, 0.0019, 12, power=2.6), mats=ac,
                                 closed=True, up=(0, 1, 0), parent=root, sharp=0))
        # lens: thin meniscus, lathed then turned to face +Y
        sag, th = 0.0013, 0.0008
        k2 = sag / lr ** 2
        nr = 8
        prof = [(lr * i / nr, -th - k2 * (lr * i / nr) ** 2) for i in range(nr + 1)]
        prof += [(lr * i / nr, th - k2 * (lr * i / nr) ** 2) for i in range(nr, -1, -1)]
        lathe("GlassesLens" + nm, prof, segs=48, mats=M["lens"], parent=root, loc=(s * cx, 0, 0),
              rot=(radians(-90), 0, 0), sharp=50)
        # end piece, hinge, temple
        ex = Rr * cos(radians(27))
        ez = Rr * sin(radians(27))
        ep = crspline([(s * (cx + ex - 0.001), 0, ez), (s * (cx + ex + 0.0045), -0.0004, ez + 0.0004),
                       (s * (cx + ex + 0.0085), -0.0028, ez + 0.0005), (s * (cx + ex + 0.0098), -0.0068, ez + 0.0005)], n=5)
        parts_front.append(sweep("GlassesLug" + nm, ep, ellipse_sec(0.0026, 0.0019, 12, power=2.6), mats=ac,
                                 up=(0, 0, 1), parent=root, sharp=0))
        hx = s * (cx + ex + 0.0098)
        lathe("GlassesHinge" + nm, [(0, -0.0024), (0.0013, -0.0024), (0.0013, 0.0024), (0, 0.0024)], segs=12,
              mats=M["steel"], parent=root, loc=(hx, -0.0082, ez + 0.0005))
        tp = crspline([(hx, -0.0092, ez + 0.0005), (hx - s * 0.0006, -0.045, ez + 0.0005),
                       (hx - s * 0.0030, -0.085, ez), (hx - s * 0.0065, -0.112, ez - 0.0020),
                       (hx - s * 0.0095, -0.132, ez - 0.0090), (hx - s * 0.0115, -0.146, ez - 0.0200)], n=8)
        parts_back.append(sweep("GlassesTemple" + nm, tp, ellipse_sec(0.0023, 0.0011, 12, power=2.4), mats=ac,
                                up=(0, 0, 1), parent=root, sharp=0,
                                scale=lambda t: (1.0 - 0.10 * t, 1.0 + 0.5 * max(0.0, (t - 0.6) / 0.4))))
        # nose pad
        uvsphere("GlassesPad" + nm, 1.0, mats=ac, u=16, v=8, scale=(0.0019, 0.0030, 0.0052), parent=root,
                 loc=(s * 0.0098, -0.0032, -0.0075), rot=(radians(-12), radians(s * 22), 0))
    br = crspline([(-0.0128, 0, 0.0088), (-0.0065, 0.0006, 0.0125), (0, 0.0010, 0.0136), (0.0065, 0.0006, 0.0125),
                   (0.0128, 0, 0.0088)], n=6)
    parts_front.append(sweep("GlassesBridge", br, ellipse_sec(0.0019, 0.0019, 12, power=2.6), mats=ac, up=(0, 1, 0),
                             parent=root, sharp=0))
    # rest on rim bottoms + temple tips: solve the pitch about local X
    zf = min(v.co.z for o in parts_front for v in o.data.vertices)
    tips = [(v.co.y, v.co.z) for o in parts_back for v in o.data.vertices if v.co.y < -0.14]
    ytip, ztip = min(tips, key=lambda q: q[1])
    pitch = math.atan2(ztip - zf, -ytip)
    root.rotation_euler = Euler((pitch, 0, yaw), "XYZ")
    drop(root, parts_front + parts_back, surface_z)
    return root


# =================================================================== pencil
def build_pencil(M):
    group("Pencil")
    a = 0.0035                                  # hexagon inradius
    L, cone = 0.168, 0.0215
    rc = a / cos(pi / 6)
    slope = (rc - 0.0004) / cone
    root = empty("Pencil", loc=(PENCIL_XY[0], PENCIL_XY[1], TOP + a + 0.0001), rot=(0, -0.0046, PENCIL_YAW))
    bm = bmesh.new()
    segs = 48
    xs = [cone * 1.25 * k / 40 for k in range(41)] + [0.06, 0.11, L]

    def rhex(ph):
        return a / cos(((ph % (pi / 3)) - pi / 6))

    rings = []
    for xx in xs:
        ring = []
        for i in range(segs):
            ph = 2 * pi * i / segs
            r = min(rhex(ph), 0.0004 + slope * xx)
            ring.append(bm.verts.new((xx - L / 2, r * cos(ph), r * sin(ph))))
        rings.append(ring)
    for k in range(len(xs) - 1):
        xc = (xs[k] + xs[k + 1]) / 2
        for i in range(segs):
            j = (i + 1) % segs
            f = bm.faces.new((rings[k][i], rings[k + 1][i], rings[k + 1][j], rings[k][j]))
            phc = 2 * pi * (i + 0.5) / segs
            if xc < 0.0050:
                f.material_index = 2
            elif 0.0004 + slope * xc < rhex(phc):
                f.material_index = 1
            else:
                f.material_index = 0
    f = bm.faces.new(rings[0])
    f.material_index = 2
    f = bm.faces.new(list(reversed(rings[-1])))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    body = mesh_obj("PencilBody", bm, [M["pencil_paint"], M["pencil_wood"], M["graphite"]], parent=root, sharp=28)
    # ferrule + eraser (lathed along the pencil axis)
    fer = [(0, 0), (0.0040, 0), (0.0042, 0.0006)]
    for k in range(3):
        z0 = 0.0018 + k * 0.0030
        fer += [(0.0042, z0), (0.0038, z0 + 0.0005), (0.0038, z0 + 0.0011), (0.0042, z0 + 0.0016)]
    fer += [(0.0042, 0.0118), (0.0040, 0.0124), (0, 0.0124)]
    ferr = o = lathe("PencilFerrule", fer, segs=32, mats=M["brass"], parent=root, loc=(L / 2 - 0.0008, 0, 0), sharp=35)
    o.rotation_euler = (0, radians(90), 0)
    o = lathe("PencilEraser", fillet([(0, 0), (0.0037, 0), (0.0037, 0.0075, 0.0012), (0, 0.0075)]), segs=32,
              mats=M["eraser"], parent=root, loc=(L / 2 + 0.0110, 0, 0))
    o.rotation_euler = (0, radians(90), 0)
    # gold imprint on the upper flat that faces the camera
    n = V((0, -cos(radians(30)), sin(radians(30))))
    yv = n.cross(V((1, 0, 0)))
    mat3 = Matrix((V((1, 0, 0)), yv, n)).transposed()
    t = text_obj("PencilImprint", "CEDAR & SLATE   HB 2", 0.0026, M["gold"], FONT_SANS, spacing=1.15, parent=root,
                 loc=V((0.030, 0, 0)) + n * (a + 0.00006))
    t.rotation_euler = mat3.to_euler()
    drop(root, [body, ferr], TOP, eps=0.0001)
    return root


# =================================================================== lights, camera, world
def add_light(name, kind, loc, target, energy, color, size=1.0, size_y=None, spot=None, blend=0.5, spread=None):
    ld = bpy.data.lights.new(name, kind)
    ld.energy = energy
    ld.color = color
    if kind == "AREA":
        ld.size = size
        if size_y:
            ld.shape = "RECTANGLE"
            ld.size_y = size_y
        if spread is not None:
            ld.spread = spread
    else:
        ld.shadow_soft_size = size
    if kind == "SPOT":
        ld.spot_size = spot
        ld.spot_blend = blend
    ob = bpy.data.objects.new(name, ld)
    ob.location = loc
    ob.rotation_euler = (V(target) - V(loc)).to_track_quat("-Z", "Y").to_euler()
    link(ob)
    ob.visible_camera = False
    return ob


def build_lights():
    group("Rig")
    c = V((0, 0, TOP + 0.10))
    # cool, soft key from front-left
    add_light("Key_Cool", "AREA", (-1.9, -1.9, 2.35), c, KEY_W, (0.62, 0.78, 1.0), size=1.5, spread=radians(70))
    # warm rim from back-right
    add_light("Rim_Warm", "AREA", (1.55, 1.95, TOP + 0.62), (0.10, -0.05, TOP + 0.10), RIM_W, (1.0, 0.60, 0.32), size=0.5,
              size_y=0.8, spread=radians(55))
    # faint cool kicker from back-left to separate the lamp from the backdrop
    add_light("Kicker_Cool", "AREA", (-1.7, 1.2, 1.5), (-0.15, 0.0, TOP + 0.15), KICK_W, (0.55, 0.80, 1.0), size=0.5,
              size_y=0.8)
    # backdrop glow: a soft pool on the sweep behind the table, falling off to the corners
    add_light("Backdrop_Glow", "SPOT", (-0.25, 1.15, 2.7), (-0.22, 2.02, 0.10), GLOW_W, (0.78, 0.95, 1.0), size=0.2,
              spot=radians(30), blend=1.0)
    # reflection cards: glossy-only strips that give the brass, glaze and fruit something to reflect
    for nm, loc, tgt, w, col, sx, sy in (
            ("Card_Left", (-1.35, -1.75, 1.45), (-0.10, 0.0, TOP + 0.20), CARD_L_W, (0.80, 0.90, 1.0), 0.45, 1.7),
            ("Card_Right", (1.45, -1.55, 1.25), (0.10, -0.05, TOP + 0.10), CARD_R_W, (1.0, 0.86, 0.70), 0.35, 1.5)):
        ob = add_light(nm, "AREA", loc, tgt, w, col, size=sx, size_y=sy)
        ob.visible_diffuse = False
        ob.visible_volume_scatter = False


KEY_W, RIM_W, KICK_W, GLOW_W = 17.0, 25.0, 8.0, 220.0
CARD_L_W, CARD_R_W = 20.0, 16.0


def build_world():
    w = bpy.data.worlds.new("S1_World")
    if w.node_tree is None:
        try:
            w.use_nodes = True
        except Exception:
            pass
    bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.010, 0.022, 0.028, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    scene.world = w


CAMS = {
    # name: (target, distance, azimuth deg (0 = straight from the front), elevation deg, lens, fstop, focus point)
    "main": ((0.004, -0.010, TOP + 0.100), 2.30, 2.5, 18.5, 85.0, 2.8, (0.03, -0.12, TOP + 0.07)),
    "top": ((0.0, 0.0, TOP), 2.0, 0.0, 89.0, 60.0, None, None),
    "lamp": ((-0.15, 0.0, TOP + 0.17), 0.95, -8.0, 14.0, 60.0, None, None),
    "head": ((-0.07, -0.05, TOP + 0.24), 0.42, 20.0, -8.0, 50.0, None, None),
    "books": ((-0.04, -0.07, TOP + 0.06), 0.62, 8.0, 22.0, 60.0, None, None),
    "mug": ((0.11, -0.16, TOP + 0.09), 0.62, 6.0, 16.0, 60.0, None, None),
    "plant": ((0.235, 0.10, TOP + 0.16), 0.95, 4.0, 14.0, 60.0, None, None),
    "fruit": ((0.225, -0.14, TOP + 0.04), 0.42, 0.0, 22.0, 60.0, None, None),
    "front": ((-0.10, -0.20, TOP + 0.02), 0.50, -6.0, 26.0, 60.0, None, None),
    "edge": ((0.0, -0.33, TOP - 0.01), 0.45, 0.0, 12.0, 60.0, None, None),
    "side": ((0.0, -0.03, TOP + 0.12), 1.6, 88.0, 6.0, 60.0, None, None),
    "low": ((0.0, -0.05, TOP + 0.10), 1.6, 0.0, 1.5, 60.0, None, None),
}


def build_camera(name):
    tgt, dist, az, el, lens, fstop, focus = CAMS[name]
    tgt = V(tgt)
    az, el = radians(az), radians(el)
    loc = tgt + V((sin(az) * cos(el), -cos(az) * cos(el), sin(el))) * dist
    cd = bpy.data.cameras.new("Cam_" + name)
    cd.lens = lens
    cd.sensor_width = 36.0
    cd.clip_start, cd.clip_end = 0.05, 60.0
    ob = bpy.data.objects.new("Cam_" + name, cd)
    ob.location = loc
    ob.rotation_euler = (tgt - loc).to_track_quat("-Z", "Y").to_euler()
    link(ob)
    if fstop:
        cd.dof.use_dof = True
        cd.dof.aperture_fstop = fstop
        cd.dof.aperture_blades = 9
        cd.dof.focus_distance = (V(focus) - loc).length
    return ob


def setup_render():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
        print("CYCLES DEVICE:", d.name, d.type, "use =", d.use)
    r = scene.render
    r.engine = "CYCLES"
    cy = scene.cycles
    cy.device = "GPU"
    cy.samples = SAMPLES
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = opt("--thresh", 0.05 if PREVIEW else 0.008, float)
    cy.use_denoising = True
    cy.denoiser = "OPENIMAGEDENOISE"
    cy.max_bounces = 12
    cy.diffuse_bounces = 4
    cy.glossy_bounces = 6
    cy.transmission_bounces = 10
    cy.transparent_max_bounces = 12
    cy.volume_bounces = 1
    cy.volume_biased = True
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.blur_glossy = 0.8
    cy.sample_clamp_indirect = 8.0
    r.resolution_x, r.resolution_y = 1920, 1080
    r.resolution_percentage = SCALE
    r.image_settings.file_format = "PNG"
    r.image_settings.color_mode = "RGB"
    r.image_settings.compression = 30
    r.film_transparent = False
    r.use_persistent_data = False
    r.filepath = OUT
    if CROP:
        x0, y0, x1, y1 = [float(v) for v in CROP.split(",")]
        r.use_border, r.use_crop_to_border = True, True
        r.border_min_x, r.border_min_y, r.border_max_x, r.border_max_y = x0, y0, x1, y1
    vs = scene.view_settings
    vs.view_transform = "AgX"
    try:
        vs.look = LOOK
    except Exception as e:
        print("look warn:", e)
    vs.exposure = EXPOSURE


def setup_compositor():
    """Soft bloom around the hot lamp interior and the brightest glints."""
    try:
        ng = bpy.data.node_groups.new("S1_Comp", "CompositorNodeTree")
        scene.compositing_node_group = ng
        ng.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        rl = ng.nodes.new("CompositorNodeRLayers")
        gl = ng.nodes.new("CompositorNodeGlare")
        gl.inputs["Type"].default_value = "Bloom"
        gl.inputs["Quality"].default_value = "High"
        gl.inputs["Threshold"].default_value = BLOOM_THRESHOLD
        gl.inputs["Smoothness"].default_value = 0.6
        gl.inputs["Clamp"].default_value = True
        gl.inputs["Maximum"].default_value = 40.0
        gl.inputs["Strength"].default_value = BLOOM_STRENGTH
        gl.inputs["Saturation"].default_value = 1.0
        gl.inputs["Tint"].default_value = (1.0, 0.86, 0.68, 1.0)
        gl.inputs["Size"].default_value = BLOOM_SIZE
        out = ng.nodes.new("NodeGroupOutput")
        ng.links.new(rl.outputs["Image"], gl.inputs["Image"])
        ng.links.new(gl.outputs["Image"], out.inputs[0])
        scene.render.use_compositing = True
    except Exception as e:
        print("compositor warn:", e)


BLOOM_THRESHOLD, BLOOM_STRENGTH, BLOOM_SIZE = 5.0, 0.28, 0.42
LOOK = "AgX - Medium High Contrast"
EXPOSURE = 0.0

# =================================================================== build
M = make_materials()
build_backdrop(M)
build_table(M)
build_lamp(M)
books, book_top = build_books(M)
build_mug(M)
build_plant(M)
build_oranges(M)
# glasses on the top book: lenses looking left-front, temples trailing along the cover
tb_ = books[2]
uvec = V((cos(tb_["root"].rotation_euler.z), sin(tb_["root"].rotation_euler.z)))
gpos = V((tb_["root"].location.x, tb_["root"].location.y)) - uvec * 0.066 + V((-uvec.y, uvec.x)) * 0.004
build_glasses(M, book_top + 0.00015, gpos, tb_["root"].rotation_euler.z + radians(90 + 3))
build_pencil(M)
build_lights()
build_world()
group("Cameras")
cam_main = build_camera("main")
scene.camera = cam_main
if CAM != "main":
    scene.camera = build_camera(CAM)
setup_render()
setup_compositor()
for f in list(bpy.data.fonts):
    bpy.data.fonts.remove(f)
bpy.context.view_layer.update()

# ------------------------------------------------------------------ report
print("OBJECTS:", len(scene.objects), "| meshes:", sum(1 for o in scene.objects if o.type == "MESH"),
      "| materials:", len(bpy.data.materials), "| build %.1fs" % (time.time() - T_START))
for col in scene.collection.children:
    obs = [o for o in col.all_objects if o.type == "MESH"]
    if not obs:
        continue
    lo, hi = V((1e9, 1e9, 1e9)), V((-1e9, -1e9, -1e9))
    for o in obs:
        for corner in o.bound_box:
            w = o.matrix_world @ V(corner)
            lo = V((min(lo.x, w.x), min(lo.y, w.y), min(lo.z, w.z)))
            hi = V((max(hi.x, w.x), max(hi.y, w.y), max(hi.z, w.z)))
    print("  %-9s x[%.3f %.3f] y[%.3f %.3f] z[%.3f %.3f]  (above table: %.4f .. %.3f)" % (
        col.name, lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, lo.z - TOP, hi.z - TOP))

if SAVE:
    if "--out" not in ARGV:
        scene.render.filepath = "//" + os.path.basename(OUT)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=BLEND, compress=True)
    print("SAVED:", BLEND)
if not NO_RENDER:
    t0 = time.time()
    bpy.ops.render.render(write_still=True)
    print("RENDERED: %s  %dx%d @ %d%%  samples %d  in %.1fs" % (
        OUT, scene.render.resolution_x, scene.render.resolution_y, SCALE, SAMPLES, time.time() - t0))
