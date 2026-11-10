"""Scene 3 v2: rainy neon city street at night.

One script, builds everything from an empty file:
  /Applications/Blender.app/Contents/MacOS/Blender -b --factory-startup -P scene3_neon_city_v2.py -- [options]
Options (after --):
  --final             full 1920x1080 render to S3_NeonCity_v2.png and save scene3_neon_city_v2.blend
  --pct N             resolution percentage (preview)          --samples N    Cycles samples
  --out FILE          output png (default preview.png)         --cam NAME     main | left | right | car | people | rail | tower | top
  --region x0,y0,x1,y1  render only this part of the frame (0..1, origin bottom-left)
  --nofog --norain    skip the volumes / the rain              --norender     build (and --save) only
  --save              write the .blend                          --exposure E   override view exposure
Self-contained: the helper library is inlined below.
"""
import bpy, sys, os, math, random, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
# ======================= helpers: mesh builder, shader-node DSL, stroke font, render setup =======================
"""Helpers for scene 3 v2 (rainy neon city), inlined into scene3_neon_city_v2.py when it is assembled. Blender 5.2.

Changed copy of ../../common.py: adds a multi-material mesh builder (MB), a shader-node DSL (NT),
a stroke font for neon tubes, light / camera / Cycles / compositor setup for headless rendering.
"""
import bpy, bmesh, math, random, os, sys, time
from mathutils import Vector, Matrix, Euler, Quaternion, noise

S = {"scene": None, "col": None}
ID4 = Matrix.Identity(4)


# ---------------------------------------------------------------- scene / collections
def reset_scene(name):
    """Start from the factory file: empty it and reuse its scene."""
    sc = bpy.context.scene
    sc.name = name
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for c in list(bpy.data.collections):
        bpy.data.collections.remove(c)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras, bpy.data.curves):
        for d in list(coll):
            if d.users == 0:
                coll.remove(d)
    S["scene"] = sc
    S["col"] = sc.collection
    return sc


def group(name, parent=None):
    c = bpy.data.collections.new(name)
    (parent or S["scene"].collection).children.link(c)
    S["col"] = c
    return c


def use(col):
    S["col"] = col
    return col


# ---------------------------------------------------------------- shader node DSL
class NT:
    """Tiny DSL over a node tree: every helper returns an output socket."""

    def __init__(self, nt):
        self.nt = nt
        self.N = nt.nodes.new
        self.L = nt.links.new
        self._geo = self._tc = self._oi = self._lp = None

    def set(self, sock, v):
        if v is None:
            return
        if isinstance(v, bpy.types.NodeSocket):
            self.L(v, sock)
            return
        if sock.type == "RGBA":
            if isinstance(v, (int, float)):
                v = (v, v, v, 1.0)
            elif len(v) == 3:
                v = tuple(v) + (1.0,)
        elif sock.type == "VECTOR" and isinstance(v, (int, float)):
            v = (v, v, v)
        sock.default_value = v

    def node(self, typ, **props):
        n = self.N(typ)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    # --- inputs
    def geo(self):
        self._geo = self._geo or self.N("ShaderNodeNewGeometry")
        return self._geo

    def texco(self):
        self._tc = self._tc or self.N("ShaderNodeTexCoord")
        return self._tc

    def objinfo(self):
        self._oi = self._oi or self.N("ShaderNodeObjectInfo")
        return self._oi

    def lightpath(self):
        self._lp = self._lp or self.N("ShaderNodeLightPath")
        return self._lp

    def attr(self, name):
        n = self.N("ShaderNodeAttribute")
        n.attribute_name = name
        return n

    # --- math
    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.N("ShaderNodeMath")
        n.operation = op
        n.use_clamp = clamp
        self.set(n.inputs[0], a)
        if b is not None:
            self.set(n.inputs[1], b)
        if c is not None:
            self.set(n.inputs[2], c)
        return n.outputs[0]

    def add(self, a, b): return self.math("ADD", a, b)
    def sub(self, a, b): return self.math("SUBTRACT", a, b)
    def mul(self, a, b): return self.math("MULTIPLY", a, b)
    def div(self, a, b): return self.math("DIVIDE", a, b)
    def mn(self, a, b): return self.math("MINIMUM", a, b)
    def mx(self, a, b): return self.math("MAXIMUM", a, b)
    def gt(self, a, b): return self.math("GREATER_THAN", a, b)
    def lt(self, a, b): return self.math("LESS_THAN", a, b)
    def abs(self, a): return self.math("ABSOLUTE", a)
    def fract(self, a): return self.math("FRACT", a)
    def floor(self, a): return self.math("FLOOR", a)
    def pow(self, a, b): return self.math("POWER", a, b)
    def sin(self, a): return self.math("SINE", a)
    def sat(self, a): return self.math("ADD", a, 0.0, clamp=True)

    def band(self, v, lo, hi):
        """1 where lo < v < hi."""
        return self.mul(self.gt(v, lo), self.lt(v, hi))

    def vmath(self, op, a, b=None, scale=None, out=0):
        n = self.N("ShaderNodeVectorMath")
        n.operation = op
        self.set(n.inputs[0], a)
        if b is not None:
            self.set(n.inputs[1], b)
        if scale is not None:
            self.set(n.inputs[3], scale)
        return n.outputs[out]

    def sep(self, v):
        n = self.N("ShaderNodeSeparateXYZ")
        self.set(n.inputs[0], v)
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def sepc(self, c):
        n = self.N("ShaderNodeSeparateColor")
        self.set(n.inputs[0], c)
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def comb(self, x=0.0, y=0.0, z=0.0):
        n = self.N("ShaderNodeCombineXYZ")
        self.set(n.inputs[0], x)
        self.set(n.inputs[1], y)
        self.set(n.inputs[2], z)
        return n.outputs[0]

    def mix(self, fac, a, b, blend="MIX"):
        n = self.N("ShaderNodeMix")
        n.data_type = "RGBA"
        n.blend_type = blend
        n.clamp_factor = True
        self.set(n.inputs[0], fac)
        self.set(n.inputs[6], a)
        self.set(n.inputs[7], b)
        return n.outputs[2]

    def mixf(self, fac, a, b):
        n = self.N("ShaderNodeMix")
        n.data_type = "FLOAT"
        n.clamp_factor = True
        self.set(n.inputs[0], fac)
        self.set(n.inputs[2], a)
        self.set(n.inputs[3], b)
        return n.outputs[0]

    def ramp(self, fac, stops, interp="LINEAR"):
        n = self.N("ShaderNodeValToRGB")
        cr = n.color_ramp
        cr.interpolation = interp
        while len(cr.elements) < len(stops):
            cr.elements.new(1.0)
        for e, (pos, col) in zip(cr.elements, stops):
            e.position = pos
            e.color = tuple(col)[:3] + (1.0,) if not isinstance(col, (int, float)) else (col, col, col, 1.0)
        self.set(n.inputs[0], fac)
        return n.outputs[0]

    def maprange(self, v, fmin, fmax, tmin=0.0, tmax=1.0, interp="LINEAR", clamp=True):
        n = self.N("ShaderNodeMapRange")
        n.interpolation_type = interp
        n.clamp = clamp
        self.set(n.inputs[0], v)
        self.set(n.inputs[1], fmin)
        self.set(n.inputs[2], fmax)
        self.set(n.inputs[3], tmin)
        self.set(n.inputs[4], tmax)
        return n.outputs[0]

    def smooth(self, v, lo, hi):
        return self.maprange(v, lo, hi, 0.0, 1.0, "SMOOTHSTEP")

    # --- textures
    def noise(self, vec, scale=1.0, detail=2.0, rough=0.5, dist=0.0, color=False):
        n = self.N("ShaderNodeTexNoise")
        self.set(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = scale
        n.inputs["Detail"].default_value = detail
        n.inputs["Roughness"].default_value = rough
        n.inputs["Distortion"].default_value = dist
        return n.outputs["Color"] if color else n.outputs["Factor"]

    def voronoi(self, vec, scale=1.0, feature="F1", rand=1.0):
        n = self.N("ShaderNodeTexVoronoi")
        n.feature = feature
        self.set(n.inputs["Vector"], vec)
        n.inputs["Scale"].default_value = scale
        n.inputs["Randomness"].default_value = rand
        return n

    def white(self, vec):
        n = self.N("ShaderNodeTexWhiteNoise")
        n.noise_dimensions = "3D"
        self.set(n.inputs["Vector"], vec)
        return n.outputs["Value"], n.outputs["Color"]

    def brick(self, vec, w, h, mortar, c1, c2, cm, offset=0.5, smooth=0.1):
        n = self.N("ShaderNodeTexBrick")
        n.offset = offset
        self.set(n.inputs["Vector"], vec)
        self.set(n.inputs["Color1"], c1)
        self.set(n.inputs["Color2"], c2)
        self.set(n.inputs["Mortar"], cm)
        n.inputs["Scale"].default_value = 1.0
        n.inputs["Mortar Size"].default_value = mortar
        n.inputs["Mortar Smooth"].default_value = smooth
        n.inputs["Bias"].default_value = 0.0
        n.inputs["Brick Width"].default_value = w
        n.inputs["Row Height"].default_value = h
        return n.outputs["Color"], n.outputs["Factor"]

    def bump(self, height, strength=1.0, dist=1.0, normal=None):
        n = self.N("ShaderNodeBump")
        n.inputs["Strength"].default_value = strength
        n.inputs["Distance"].default_value = dist
        self.set(n.inputs["Height"], height)
        if normal is not None:
            self.set(n.inputs["Normal"], normal)
        return n.outputs[0]

    def shader(self, typ, **inputs):
        n = self.N(typ)
        for k, v in inputs.items():
            self.set(n.inputs[k.replace("_", " ")], v)
        return n.outputs[0]

    def mixshader(self, fac, a, b):
        n = self.N("ShaderNodeMixShader")
        self.set(n.inputs[0], fac)
        self.L(a, n.inputs[1])
        self.L(b, n.inputs[2])
        return n.outputs[0]

    def addshader(self, a, b):
        n = self.N("ShaderNodeAddShader")
        self.L(a, n.inputs[0])
        self.L(b, n.inputs[1])
        return n.outputs[0]


# ---------------------------------------------------------------- materials
def _new_mat(name):
    old = bpy.data.materials.get(name)
    if old:
        bpy.data.materials.remove(old)
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    p = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
    out = next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)
    if out is None:
        out = nt.nodes.new("ShaderNodeOutputMaterial")
    if p is None:
        p = nt.nodes.new("ShaderNodeBsdfPrincipled")
        nt.links.new(p.outputs[0], out.inputs[0])
    return m, nt, p, out


def mat_nodes(name):
    """Fresh principled material -> (material, NT helper, principled node, output node)."""
    m, nt, p, out = _new_mat(name)
    return m, NT(nt), p, out


def mat_custom(name):
    """Fresh material without the principled node -> (material, NT helper, output node)."""
    m, nt, p, out = _new_mat(name)
    nt.nodes.remove(p)
    return m, NT(nt), out


_MATS = {}


def mat(name, color, rough=0.5, metal=0.0, emit=None, emit_str=0.0, trans=0.0, ior=1.45, coat=0.0,
        coat_rough=0.05, alpha=1.0, sss=0.0, sheen=0.0, spec=None):
    if name in _MATS:
        return _MATS[name]
    m, nt, p, out = _new_mat(name)
    c = tuple(color)[:3] + (1.0,)
    i = p.inputs
    i["Base Color"].default_value = c
    i["Roughness"].default_value = rough
    i["Metallic"].default_value = metal
    i["IOR"].default_value = ior
    if trans:
        i["Transmission Weight"].default_value = trans
    if coat:
        i["Coat Weight"].default_value = coat
        i["Coat Roughness"].default_value = coat_rough
    if sss:
        i["Subsurface Weight"].default_value = sss
    if sheen:
        i["Sheen Weight"].default_value = sheen
    if spec is not None:
        i["Specular IOR Level"].default_value = spec
    if alpha < 1:
        i["Alpha"].default_value = alpha
    if emit is not None:
        i["Emission Color"].default_value = tuple(emit)[:3] + (1.0,)
        i["Emission Strength"].default_value = emit_str
    m.diffuse_color = c
    _MATS[name] = m
    return m


def emit_mat(name, color, strength):
    if name in _MATS:
        return _MATS[name]
    m, n, out = mat_custom(name)
    n.L(n.shader("ShaderNodeEmission", Color=tuple(color), Strength=strength), out.inputs["Surface"])
    m.diffuse_color = tuple(color)[:3] + (1.0,)
    _MATS[name] = m
    return m


# ---------------------------------------------------------------- mesh builder
class MB:
    """Accumulates many primitives with different materials into one mesh object."""

    def __init__(self, name, uv=False, col=False):
        self.name = name
        self.bm = bmesh.new()
        self.mats = []
        self._mi = {}
        self.uvl = self.bm.loops.layers.uv.new("UVMap") if uv else None
        self.coll = self.bm.loops.layers.float_color.new("wcol") if col else None
        self.M = ID4.copy()

    def mi(self, m):
        i = self._mi.get(m.name)
        if i is None:
            i = self._mi[m.name] = len(self.mats)
            self.mats.append(m)
        return i

    def _faces(self, vs, quads, m, smooth, col=None):
        """Faces from index tuples into vs. Hand-built (bmesh operators cost O(mesh size) per call)."""
        idx = self.mi(m)
        new = self.bm.faces.new
        out = []
        for q in quads:
            f = new([vs[i] for i in q])
            f.material_index = idx
            f.smooth = smooth
            if col is not None and self.coll is not None:
                for l in f.loops:
                    l[self.coll] = col
            out.append(f)
        return out

    def _mat(self, loc, rot=None, scale=None):
        M = self.M @ Matrix.Translation(loc)
        if rot is not None:
            M = M @ (rot.to_matrix().to_4x4() if isinstance(rot, (Quaternion, Euler)) else Euler(rot).to_matrix().to_4x4())
        if scale is not None:
            M = M @ Matrix.Diagonal((scale[0], scale[1], scale[2], 1.0))
        return M

    # --- primitives
    _BOXV = [(-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5),
             (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]
    _BOXF = [(0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4), (2, 6, 7, 3), (0, 4, 6, 2), (1, 3, 7, 5)]

    def box(self, size, loc, m, rot=None, base=False, smooth=False, col=None):
        M = self._mat(loc, rot)
        if base:
            M = M @ Matrix.Translation((0, 0, size[2] / 2))
        M = M @ Matrix.Diagonal((size[0], size[1], size[2], 1.0))
        new = self.bm.verts.new
        vs = [new(M @ Vector(p)) for p in self._BOXV]
        return self._faces(vs, self._BOXF, m, smooth, col)

    def bx(self, lo, hi, m, **kw):
        """Axis-aligned box from min / max corners."""
        return self.box((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]),
                        ((hi[0] + lo[0]) / 2, (hi[1] + lo[1]) / 2, (hi[2] + lo[2]) / 2), m, **kw)

    def _cone(self, M, r1, r2, h, segs, m, smooth, caps):
        new = self.bm.verts.new
        cs = [(math.cos(2 * math.pi * k / segs), math.sin(2 * math.pi * k / segs)) for k in range(segs)]
        bot = [new(M @ Vector((r1 * c, r1 * s, -h / 2))) for c, s in cs] if r1 > 1e-6 else None
        top = [new(M @ Vector((r2 * c, r2 * s, h / 2))) for c, s in cs] if r2 > 1e-6 else None
        idx = self.mi(m)
        fn = self.bm.faces.new
        fs = []
        if bot and top:
            for k in range(segs):
                k2 = (k + 1) % segs
                fs.append(fn((bot[k], bot[k2], top[k2], top[k])))
        elif bot:
            apex = new(M @ Vector((0, 0, h / 2)))
            for k in range(segs):
                fs.append(fn((bot[k], bot[(k + 1) % segs], apex)))
        elif top:
            apex = new(M @ Vector((0, 0, -h / 2)))
            for k in range(segs):
                fs.append(fn((apex, top[(k + 1) % segs], top[k])))
        for f in fs:
            f.material_index = idx
            f.smooth = smooth
        if caps:
            for ring in ((bot[::-1] if bot else None), top):
                if ring:
                    f = fn(ring)
                    f.material_index = idx
                    fs.append(f)
        return fs

    def cyl(self, r, h, loc, m, rot=None, segs=12, r2=None, base=True, smooth=True, caps=True):
        M = self._mat(loc, rot)
        if base:
            M = M @ Matrix.Translation((0, 0, h / 2))
        return self._cone(M, r, r if r2 is None else r2, h, segs, m, smooth, caps)

    def rod(self, p1, p2, r, m, segs=8, r2=None, smooth=True, caps=True):
        p1, p2 = Vector(p1), Vector(p2)
        d = p2 - p1
        ln = d.length
        if ln < 1e-7:
            return []
        q = Vector((0, 0, 1)).rotation_difference(d / ln)
        M = self.M @ Matrix.Translation((p1 + p2) / 2) @ q.to_matrix().to_4x4()
        return self._cone(M, r, r if r2 is None else r2, ln, segs, m, smooth, caps)

    def sphere(self, r, loc, m, scale=(1, 1, 1), u=12, v=8, rot=None, smooth=True):
        M = self._mat(loc, rot, (r * scale[0], r * scale[1], r * scale[2]))
        new = self.bm.verts.new
        top, bot = new(M @ Vector((0, 0, 1))), new(M @ Vector((0, 0, -1)))
        rings = []
        for j in range(1, v):
            th = math.pi * j / v
            st, ct = math.sin(th), math.cos(th)
            rings.append([new(M @ Vector((st * math.cos(2 * math.pi * k / u), st * math.sin(2 * math.pi * k / u), ct))) for k in range(u)])
        idx = self.mi(m)
        fn = self.bm.faces.new
        fs = []
        for k in range(u):
            k2 = (k + 1) % u
            fs.append(fn((top, rings[0][k], rings[0][k2])))
            fs.append(fn((bot, rings[-1][k2], rings[-1][k])))
            for a, b in zip(rings, rings[1:]):
                fs.append(fn((a[k], b[k], b[k2], a[k2])))
        for f in fs:
            f.material_index = idx
            f.smooth = smooth
        return fs

    def disc(self, r, loc, m, rot=None, segs=16):
        M = self._mat(loc, rot)
        f = self.bm.faces.new([self.bm.verts.new(M @ Vector((r * math.cos(2 * math.pi * k / segs), r * math.sin(2 * math.pi * k / segs), 0.0)))
                               for k in range(segs)])
        f.material_index = self.mi(m)
        return [f]

    def poly(self, pts, m, smooth=False, col=None, uvs=None, flip=False):
        vs = [self.bm.verts.new(self.M @ Vector(p)) for p in pts]
        if flip:
            vs.reverse()
            if isinstance(col, list):
                col = col[::-1]
            if uvs is not None:
                uvs = uvs[::-1]
        f = self.bm.faces.new(vs)
        f.material_index = self.mi(m)
        f.smooth = smooth
        if col is not None and self.coll is not None:
            for k, l in enumerate(f.loops):
                l[self.coll] = col[k] if isinstance(col, list) else col
        if uvs is not None and self.uvl is not None:
            for l, uv in zip(f.loops, uvs):
                l[self.uvl].uv = uv
        return f

    def tube(self, pts, r, m, segs=6, caps=True, closed=False, smooth=True):
        """Sweep a circle along a polyline (parallel-transport frame)."""
        pts = [Vector(p) for p in pts]
        n = len(pts)
        if n < 2:
            return
        idx = self.mi(m)
        tans = []
        for i in range(n):
            a = pts[i - 1] if (i > 0 or closed) else pts[i]
            b = pts[(i + 1) % n] if (i < n - 1 or closed) else pts[i]
            t = b - a
            tans.append(t.normalized() if t.length > 1e-9 else Vector((0, 0, 1)))
        t0 = tans[0]
        ref = Vector((0, 1, 0)) if abs(t0.y) < 0.9 else Vector((1, 0, 0))
        nrm = (ref - t0 * ref.dot(t0)).normalized()
        rings = []
        for i in range(n):
            t = tans[i]
            nrm = nrm - t * nrm.dot(t)
            nrm = nrm.normalized() if nrm.length > 1e-6 else t.orthogonal().normalized()
            bn = t.cross(nrm)
            rings.append([self.bm.verts.new(self.M @ (pts[i] + r * (math.cos(2 * math.pi * k / segs) * nrm +
                                                               math.sin(2 * math.pi * k / segs) * bn)))
                          for k in range(segs)])
        rng = range(n) if closed else range(n - 1)
        for i in rng:
            a, b = rings[i], rings[(i + 1) % n]
            for k in range(segs):
                k2 = (k + 1) % segs
                f = self.bm.faces.new((a[k], a[k2], b[k2], b[k]))
                f.material_index = idx
                f.smooth = smooth
        if caps and not closed:
            for ring in (rings[0][::-1], rings[-1]):
                f = self.bm.faces.new(ring)
                f.material_index = idx

    def loft(self, sections, m, closed=True, caps=True, smooth=True, mfun=None):
        """sections: list of rings (same point count). mfun(i, k) -> material for the quad, optional."""
        idx = self.mi(m)
        rings = [[self.bm.verts.new(self.M @ Vector(p)) for p in sec] for sec in sections]
        n = len(rings[0])
        faces = {}
        for i in range(len(rings) - 1):
            a, b = rings[i], rings[i + 1]
            for k in (range(n) if closed else range(n - 1)):
                k2 = (k + 1) % n
                f = self.bm.faces.new((a[k], a[k2], b[k2], b[k]))
                f.material_index = self.mi(mfun(i, k)) if mfun else idx
                f.smooth = smooth
                faces[(i, k)] = f
        if caps:
            for ring in (rings[0][::-1], rings[-1]):
                f = self.bm.faces.new(ring)
                f.material_index = idx
                f.smooth = smooth
        return rings, faces

    def lathe(self, profile, loc, m, segs=16, smooth=True, rot=None):
        """profile: [(radius, z)], revolved around local Z at loc."""
        M = self._mat(loc, rot)
        idx = self.mi(m)
        rings = []
        for r, z in profile:
            if r <= 1e-6:
                rings.append([self.bm.verts.new(M @ Vector((0, 0, z)))])
            else:
                rings.append([self.bm.verts.new(M @ Vector((r * math.cos(2 * math.pi * i / segs),
                                                           r * math.sin(2 * math.pi * i / segs), z)))
                              for i in range(segs)])
        for a, b in zip(rings, rings[1:]):
            for i in range(segs):
                j = (i + 1) % segs
                if len(a) == 1 and len(b) == 1:
                    continue
                if len(a) == 1:
                    f = self.bm.faces.new((a[0], b[i], b[j]))
                elif len(b) == 1:
                    f = self.bm.faces.new((a[i], a[j], b[0]))
                else:
                    f = self.bm.faces.new((a[i], a[j], b[j], b[i]))
                f.material_index = idx
                f.smooth = smooth

    def add_mesh(self, me, M=None):
        """Append an existing mesh datablock (materials are mapped by name)."""
        tmp = bmesh.new()
        tmp.from_mesh(me)
        if M is not None or self.M != ID4:
            bmesh.ops.transform(tmp, matrix=self.M @ (M or ID4), verts=tmp.verts[:])
        remap = [self.mi(mm) for mm in me.materials] or [0]
        tmp_me = bpy.data.meshes.new("_tmp")
        for f in tmp.faces:
            f.material_index = remap[min(f.material_index, len(remap) - 1)]
        tmp.to_mesh(tmp_me)
        tmp.free()
        self.bm.from_mesh(tmp_me)
        bpy.data.meshes.remove(tmp_me)

    # --- output
    def mesh(self, sharp=35.0, recalc=False):
        bm = self.bm
        if recalc:
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        if sharp is not None:
            thr = math.radians(sharp)
            for e in bm.edges:
                if len(e.link_faces) == 2 and e.calc_face_angle(0.0) > thr:
                    e.smooth = False
        me = bpy.data.meshes.new(self.name)
        bm.to_mesh(me)
        bm.free()
        for m in self.mats:
            me.materials.append(m)
        return me

    def finish(self, col=None, loc=(0, 0, 0), rot=(0, 0, 0), sharp=35.0, recalc=False, bevel=0.0,
               bevel_segs=2, subsurf=0, matrix=None):
        me = self.mesh(sharp, recalc)
        ob = bpy.data.objects.new(self.name, me)
        if matrix is not None:
            ob.matrix_world = matrix
        else:
            ob.location = loc
            ob.rotation_euler = rot
        if subsurf:
            md = ob.modifiers.new("Subsurf", "SUBSURF")
            md.levels = md.render_levels = subsurf
        if bevel:
            md = ob.modifiers.new("Bevel", "BEVEL")
            md.width = bevel
            md.segments = bevel_segs
            md.limit_method = "ANGLE"
            md.angle_limit = math.radians(40)
            md.harden_normals = True
        (col or S["col"]).objects.link(ob)
        return ob


def inst(src, name, loc=(0, 0, 0), rot=(0, 0, 0), scale=(1, 1, 1), col=None, matrix=None):
    """Linked duplicate: new object sharing src's mesh (and modifiers are NOT copied)."""
    ob = bpy.data.objects.new(name, src.data if isinstance(src, bpy.types.Object) else src)
    if matrix is not None:
        ob.matrix_world = matrix
    else:
        ob.location = loc
        ob.rotation_euler = rot
        ob.scale = scale
    (col or S["col"]).objects.link(ob)
    return ob


def text(name, body, loc, material, size=1.0, extrude=0.0, rot=(math.pi / 2, 0, 0), align="CENTER",
         align_y="CENTER", line=1.0, matrix=None, col=None):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.extrude = extrude
    cu.align_x = align
    cu.align_y = align_y
    cu.space_line = line
    cu.materials.append(material)
    ob = bpy.data.objects.new(name, cu)
    if matrix is not None:
        ob.matrix_world = matrix
    else:
        ob.location = loc
        ob.rotation_euler = rot
    (col or S["col"]).objects.link(ob)
    return ob


def basis(right, front_normal, origin):
    """Matrix whose local X = right, local -Y = front_normal (viewer side), Z = up."""
    r = Vector(right).normalized()
    n = Vector(front_normal).normalized()
    up = Vector((0, 0, 1))
    M = Matrix((r, -n, up)).transposed().to_4x4()
    M.translation = Vector(origin)
    return M


# ---------------------------------------------------------------- lights / camera
def light(name, kind, loc, energy, color=(1, 1, 1), size=1.0, target=None, rot=None, spot=60, blend=0.3,
          size_y=None, col=None, glossy=True, volume=True, diffuse=True, shadow=True):
    ld = bpy.data.lights.new(name, kind)
    ld.energy = energy
    ld.color = color
    if kind == "AREA":
        ld.size = size
        if size_y:
            ld.shape = "RECTANGLE"
            ld.size_y = size_y
    elif kind == "SUN":
        ld.angle = math.radians(size)
    else:
        ld.shadow_soft_size = size
    if kind == "SPOT":
        ld.spot_size = math.radians(spot)
        ld.spot_blend = blend
    ld.use_shadow = shadow
    ob = bpy.data.objects.new(name, ld)
    ob.location = loc
    if target is not None:
        ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    elif rot is not None:
        ob.rotation_euler = rot
    ob.visible_glossy = glossy
    ob.visible_volume_scatter = volume
    ob.visible_diffuse = diffuse
    (col or S["col"]).objects.link(ob)
    return ob


class Cam:
    """Camera + an analytic projection so the build can test what lands in frame."""

    def __init__(self, name, loc, target, lens=28.0, shift_y=0.0, shift_x=0.0, fstop=None, focus=None,
                 res=(1920, 1080), roll=0.0):
        cd = bpy.data.cameras.new(name)
        cd.lens = lens
        cd.sensor_width = 36.0
        cd.shift_x, cd.shift_y = shift_x, shift_y
        cd.clip_start, cd.clip_end = 0.05, 3000.0
        ob = bpy.data.objects.new(name, cd)
        q = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y")
        ob.rotation_euler = q.to_euler()
        ob.location = loc
        if fstop:
            cd.dof.use_dof = True
            cd.dof.aperture_fstop = fstop
            cd.dof.focus_distance = focus if isinstance(focus, (int, float)) else (Vector(focus) - Vector(loc)).length
        S["col"].objects.link(ob)
        self.ob, self.loc, self.R = ob, Vector(loc), q.to_matrix()
        self.Rinv = self.R.transposed()
        self.lens, self.sx, self.sy, self.aspect = lens, shift_x, shift_y, res[0] / res[1]

    def ndc(self, p):
        """World point -> (x, y) in 0..1 frame coords (y up) and depth; None when behind the camera."""
        v = self.Rinv @ (Vector(p) - self.loc)
        if v.z > -1e-4:
            return None
        k = self.lens / 36.0
        return (0.5 + v.x / -v.z * k - self.sx, 0.5 + (v.y / -v.z * k - self.sy) * self.aspect, -v.z)

    def unproject(self, x, y, depth):
        """Frame coords + depth along the view axis -> world point."""
        k = self.lens / 36.0
        v = Vector(((x - 0.5 + self.sx) / k * depth, ((y - 0.5) / self.aspect + self.sy) / k * depth, -depth))
        return self.R @ v + self.loc

    def frame_box(self, pts):
        """Bounding box of world points in frame coords -> (x0, y0, x1, y1) or None if any is behind."""
        qs = [self.ndc(p) for p in pts]
        if any(q is None for q in qs):
            return None
        return (min(q[0] for q in qs), min(q[1] for q in qs), max(q[0] for q in qs), max(q[1] for q in qs))


# ---------------------------------------------------------------- render setup
def setup_cycles(samples=256, res=(1920, 1080), pct=100, exposure=0.0, look="AgX - Punchy"):
    sc = S["scene"]
    sc.render.engine = "CYCLES"
    info = "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        names = []
        for d in prefs.devices:
            d.use = True
            names.append(f"{d.name} [{d.type}]")
        sc.cycles.device = "GPU"
        info = "GPU compute_device_type=%s devices=%s" % (prefs.compute_device_type, names)
    except Exception as e:
        info = "CPU (GPU setup failed: %s)" % e
    c = sc.cycles
    c.samples = samples
    c.use_adaptive_sampling = True
    c.adaptive_threshold = 0.012
    c.use_denoising = True
    c.denoiser = "OPENIMAGEDENOISE"
    try:
        c.denoising_use_gpu = True
    except Exception:
        pass
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = pct
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    sc.render.image_settings.compression = 30
    sc.render.use_persistent_data = False
    sc.view_settings.view_transform = "AgX"
    try:
        sc.view_settings.look = look
    except Exception:
        sc.view_settings.look = "None"
    sc.view_settings.exposure = exposure
    return info


def setup_compositor(bloom=0.5, bloom_size=0.55, threshold=1.0, vignette=0.35, pct=100, dispersion=0.0, power=1.0, slope=1.0):
    """Bloom on highlights + soft vignette (Blender 5.x compositor node group)."""
    sc = S["scene"]
    ng = bpy.data.node_groups.new("S3_Comp", "CompositorNodeTree")
    ng.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    N, L = ng.nodes.new, ng.links.new
    rl = N("CompositorNodeRLayers")
    rl.scene = sc
    go = N("NodeGroupOutput")
    cur = rl.outputs["Image"]
    if bloom > 0:
        gl = N("CompositorNodeGlare")
        gl.inputs["Type"].default_value = "Bloom"
        gl.inputs["Quality"].default_value = "High"
        gl.inputs["Threshold"].default_value = threshold
        gl.inputs["Smoothness"].default_value = 0.35
        gl.inputs["Strength"].default_value = bloom
        gl.inputs["Saturation"].default_value = 1.0
        gl.inputs["Size"].default_value = bloom_size
        L(cur, gl.inputs["Image"])
        cur = gl.outputs["Image"]
    if power != 1.0 or slope != 1.0:
        cb = N("CompositorNodeColorBalance")
        cb.inputs["Type"].default_value = "Offset/Power/Slope (ASC-CDL)"
        cb.inputs[11].default_value = power          # > 1 pulls the shadows down around scene value 1.0
        cb.inputs[13].default_value = slope
        L(cur, cb.inputs["Image"])
        cur = cb.outputs["Image"]
    if dispersion > 0:
        ld = N("CompositorNodeLensdist")
        ld.inputs["Dispersion"].default_value = dispersion
        L(cur, ld.inputs["Image"])
        cur = ld.outputs["Image"]
    if vignette > 0:
        em = N("CompositorNodeEllipseMask")
        em.inputs["Size"].default_value = (0.86, 0.80)
        bl = N("CompositorNodeBlur")
        s = 340.0 * pct / 100.0
        bl.inputs["Size"].default_value = (s, s)
        L(em.outputs["Mask"], bl.inputs["Image"])
        mx = N("ShaderNodeMix")
        mx.data_type = "RGBA"
        mx.blend_type = "MULTIPLY"
        mx.inputs[0].default_value = vignette
        L(cur, mx.inputs[6])
        L(bl.outputs["Image"], mx.inputs[7])
        cur = mx.outputs[2]
    L(cur, go.inputs[0])
    sc.compositing_node_group = ng
    sc.render.use_compositing = True
    return ng


# ---------------------------------------------------------------- stroke font (neon tubes)
# Glyphs on a grid 4 wide x 6 tall. Each stroke is a polyline; corners get filleted like bent glass.
GLYPHS = {
    "A": [[(0, 0), (2, 6), (4, 0)], [(0.85, 2.1), (3.15, 2.1)]],
    "B": [[(0, 0), (0, 6), (2.7, 6), (3.7, 5.1), (3.7, 4), (2.7, 3.1), (0, 3.1)], [(2.7, 3.1), (4, 2.1), (4, 1), (2.9, 0), (0, 0)]],
    "C": [[(4, 4.9), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1.1)]],
    "D": [[(0, 0), (0, 6), (2.4, 6), (4, 4.4), (4, 1.6), (2.4, 0), (0, 0)]],
    "E": [[(4, 6), (0, 6), (0, 0), (4, 0)], [(0, 3.1), (3, 3.1)]],
    "F": [[(4, 6), (0, 6), (0, 0)], [(0, 3.1), (3, 3.1)]],
    "G": [[(4, 4.9), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1), (4, 2.9), (2.2, 2.9)]],
    "H": [[(0, 0), (0, 6)], [(4, 0), (4, 6)], [(0, 3.1), (4, 3.1)]],
    "I": [[(0.5, 0), (0.5, 6)]],
    "J": [[(3.6, 6), (3.6, 1.2), (2.6, 0), (1, 0), (0, 1.2)]],
    "K": [[(0, 0), (0, 6)], [(3.9, 6), (0, 2.4)], [(1.5, 3.8), (4, 0)]],
    "L": [[(0, 6), (0, 0), (3.8, 0)]],
    "M": [[(0, 0), (0, 6), (2.5, 2.4), (5, 6), (5, 0)]],
    "N": [[(0, 0), (0, 6), (4, 0), (4, 6)]],
    "O": [[(1, 0), (0, 1), (0, 5), (1, 6), (3, 6), (4, 5), (4, 1), (3, 0), (1, 0)]],
    "P": [[(0, 0), (0, 6), (2.9, 6), (4, 5), (4, 3.9), (2.9, 2.9), (0, 2.9)]],
    "Q": [[(1, 0), (0, 1), (0, 5), (1, 6), (3, 6), (4, 5), (4, 1), (3, 0), (1, 0)], [(2.5, 1.6), (4.2, -0.3)]],
    "R": [[(0, 0), (0, 6), (2.9, 6), (4, 5), (4, 3.9), (2.9, 2.9), (0, 2.9)], [(2.1, 2.9), (4, 0)]],
    "S": [[(4, 5), (3, 6), (1, 6), (0, 5), (0, 4.1), (1, 3.1), (3, 3.1), (4, 2.1), (4, 1), (3, 0), (1, 0), (0, 1)]],
    "T": [[(0, 6), (4, 6)], [(2, 6), (2, 0)]],
    "U": [[(0, 6), (0, 1), (1, 0), (3, 0), (4, 1), (4, 6)]],
    "V": [[(0, 6), (2, 0), (4, 6)]],
    "W": [[(0, 6), (1.2, 0), (2.5, 4.2), (3.8, 0), (5, 6)]],
    "X": [[(0, 0), (4, 6)], [(0, 6), (4, 0)]],
    "Y": [[(0, 6), (2, 3), (4, 6)], [(2, 3), (2, 0)]],
    "Z": [[(0, 6), (4, 6), (0, 0), (4, 0)]],
    "0": [[(1, 0), (0, 1), (0, 5), (1, 6), (2.6, 6), (3.6, 5), (3.6, 1), (2.6, 0), (1, 0)]],
    "1": [[(0, 4.7), (1.3, 6), (1.3, 0)]],
    "2": [[(0, 4.9), (1, 6), (3, 6), (4, 5), (4, 3.9), (0, 0), (4, 0)]],
    "3": [[(0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (3, 3.1), (1.6, 3.1)], [(3, 3.1), (4, 2.1), (4, 1), (3, 0), (1, 0), (0, 1)]],
    "4": [[(3, 0), (3, 6), (0, 2), (4, 2)]],
    "5": [[(3.8, 6), (0.3, 6), (0, 3.3), (2.9, 3.5), (4, 2.5), (4, 1), (3, 0), (1, 0), (0, 1)]],
    "6": [[(3.8, 5.2), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1), (4, 2.4), (3, 3.4), (0, 3.2)]],
    "7": [[(0, 6), (4, 6), (1.5, 0)]],
    "8": [[(1, 3.1), (0, 4.1), (0, 5), (1, 6), (3, 6), (4, 5), (4, 4.1), (3, 3.1), (1, 3.1), (0, 2.1), (0, 1), (1, 0), (3, 0), (4, 1), (4, 2.1), (3, 3.1)]],
    "9": [[(0.2, 0.8), (1, 0), (3, 0), (4, 1), (4, 5), (3, 6), (1, 6), (0, 5), (0, 3.6), (1, 2.6), (4, 2.8)]],
    "-": [[(0.6, 3), (3.4, 3)]],
    "+": [[(0.5, 3), (3.5, 3)], [(2, 1.5), (2, 4.5)]],
    "!": [[(0.5, 6), (0.5, 1.9)], [(0.5, 0.35), (0.5, 0.0)]],
    ">": [[(0.5, 5), (3.3, 3), (0.5, 1)]],
    " ": [],
}
GLYPH_W = {"I": 1.0, "M": 5.0, "W": 5.0, "1": 1.3, "0": 3.6, " ": 2.0, "!": 1.0, "L": 3.8, "J": 3.6}


def fillet(pts, d, n=4, closed=False):
    """Round every interior corner with a quadratic bezier that cuts back d along both edges."""
    pts = [Vector(p) for p in pts]
    N = len(pts)
    out = []
    for i in range(N):
        P = pts[i]
        if not closed and (i == 0 or i == N - 1):
            out.append(P)
            continue
        A, B = pts[i - 1], pts[(i + 1) % N]
        la, lb = (A - P).length, (B - P).length
        dd = min(d, la * 0.5, lb * 0.5)
        if dd < 1e-5 or abs((A - P).normalized().dot((B - P).normalized()) + 1.0) < 1e-4:
            out.append(P)
            continue
        s = P + (A - P) / la * dd
        e = P + (B - P) / lb * dd
        for k in range(n + 1):
            t = k / n
            out.append((1 - t) ** 2 * s + 2 * (1 - t) * t * P + t * t * e)
    return out


def glyph_strokes(ch, h, origin=(0.0, 0.0), rad=0.16, n=3):
    """-> (list of (points[(x, z)], closed), advance width) for one character of cap height h."""
    s = h / 6.0
    w = GLYPH_W.get(ch, 4.0) * s
    res = []
    for st in GLYPHS.get(ch, []):
        closed = len(st) > 2 and st[0] == st[-1]
        pts = [Vector((origin[0] + x * s, origin[1] + z * s, 0.0)) for x, z in (st[:-1] if closed else st)]
        res.append((fillet(pts, rad * h, n, closed), closed))
    return res, w


def word_width(word, h, gap=0.22):
    return sum(GLYPH_W.get(c, 4.0) * h / 6.0 for c in word) + gap * h * (len(word) - 1)

# ======================= end of helpers =======================

_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
_ap = argparse.ArgumentParser()
_ap.add_argument("--final", action="store_true")
_ap.add_argument("--pct", type=int, default=50)
_ap.add_argument("--samples", type=int, default=48)
_ap.add_argument("--out", default="preview.png")
_ap.add_argument("--cam", default="main")
_ap.add_argument("--region", default="")
_ap.add_argument("--nofog", action="store_true")
_ap.add_argument("--norain", action="store_true")
_ap.add_argument("--norender", action="store_true")
_ap.add_argument("--save", action="store_true")
_ap.add_argument("--exposure", type=float, default=None)
ARGS = _ap.parse_args(_argv)
if ARGS.final:
    ARGS.pct, ARGS.samples, ARGS.out, ARGS.save = 100, 480, "S3_NeonCity_v2.png", True

T0 = time.time()


def tick(label):
    print(f"[s3 +{time.time() - T0:6.1f}s] {label}", flush=True)


random.seed(11)
RNG = random.Random(2077)
sc = reset_scene("S3_NeonCity_v2")
GPU_INFO = setup_cycles(samples=ARGS.samples, pct=ARGS.pct, exposure=0.0)
print("[s3] render device:", GPU_INFO)
cy = sc.cycles
cy.max_bounces = 8
cy.diffuse_bounces = 2
cy.glossy_bounces = 4
cy.transmission_bounces = 4
cy.volume_bounces = 0
cy.transparent_max_bounces = 24
cy.sample_clamp_indirect = 6.0
cy.sample_clamp_direct = 0.0
cy.caustics_reflective = False
cy.caustics_refractive = False
cy.blur_glossy = 0.6
cy.use_light_tree = True

# ---------------------------------------------------------------- layout (metres; street runs along +Y)
RW = 5.0                 # half road width
SWK = 3.6                # sidewalk width
FRONT = RW + SWK         # building line |x|
CURB = 0.15              # sidewalk height
B0_END = 29.0            # first block ends here
XS0, XS1 = 32.0, 42.0    # cross-street roadway (under the elevated rail); its sidewalks are 29..32 and 42..45
RAIL_Y, RAIL_Z = 37.0, 7.7   # rail centre line and deck top
B1_START = 45.0
TOWER_Y = 330.0
GH = 4.3                 # ground floor height of the near buildings

# camera first: the build asks it what lands in frame
CAM_POS = Vector((-1.25, -10.5, 1.5))
CAM_TGT = Vector((0.35, 60.0, 2.75))
LENS, SHIFT_Y = 22.0, 0.0235
use(group("Rig"))
CAM = Cam("Cam", CAM_POS, CAM_TGT, lens=LENS, shift_y=SHIFT_Y, fstop=2.2, focus=24.0)
sc.camera = CAM.ob


def lw(side, u, d, z):
    """Facade-local -> world. u runs along the street, d is depth into the building (negative = out over the sidewalk)."""
    return Vector((side * (FRONT + d), u, z))


def fbox(mb, side, u0, u1, d0, d1, z0, z1, m, **kw):
    """Axis-aligned box given in facade-local ranges."""
    return mb.box((abs(d1 - d0), abs(u1 - u0), abs(z1 - z0)), lw(side, (u0 + u1) / 2, (d0 + d1) / 2, (z0 + z1) / 2), m, **kw)


def fquad(mb, side, pts, m, face_out=True, **kw):
    """Quad from facade-local (u, d, z) points; winding fixed so the normal faces the street when face_out."""
    w = [lw(side, *p) for p in pts]
    nrm = (w[1] - w[0]).cross(w[2] - w[1])
    want = Vector((-side, 0, 0))
    flip = (nrm.dot(want) < 0) == face_out and abs(nrm.x) > 1e-6
    return mb.poly(w, m, flip=flip, **kw)

tick("p10_materials")
# ================================================================ MATERIALS
NEON = {"magenta": (1.0, 0.05, 0.40), "pink": (1.0, 0.22, 0.50), "cyan": (0.04, 0.78, 1.0), "teal": (0.04, 1.0, 0.72),
        "amber": (1.0, 0.52, 0.07), "red": (1.0, 0.06, 0.04), "white": (1.0, 0.88, 0.74), "green": (0.12, 1.0, 0.32),
        "blue": (0.12, 0.28, 1.0), "purple": (0.52, 0.18, 1.0), "yellow": (1.0, 0.80, 0.14), "ice": (0.72, 0.9, 1.0)}
TUBE = 3.2      # neon tube emission: kept moderate so AgX keeps the colour


def neon(key, strength=TUBE):
    return emit_mat(f"neon_{key}_{strength:g}", NEON[key], strength)


def make_glass(name, tint=(0.82, 0.9, 0.95), refl=1.0, rough=0.0, fogged=0.0, floor=0.04):
    """Architectural glass: straight-through transparency + fresnel mirror (no refraction, so light passes cheaply)."""
    m, n, out = mat_custom(name)
    fr = n.node("ShaderNodeFresnel")
    fr.inputs["IOR"].default_value = 1.5
    fac = n.math("MAXIMUM", n.math("MULTIPLY", fr.outputs[0], refl, clamp=True), floor)
    tr = n.shader("ShaderNodeBsdfTransparent", Color=tint)
    gl = n.shader("ShaderNodeBsdfGlossy", Color=(1, 1, 1), Roughness=rough)
    sh = n.mixshader(fac, tr, gl)
    if fogged:
        P = n.geo().outputs["Position"]
        x, y, z = n.sep(P)
        cond = n.math("SUBTRACT", 1.0, n.smooth(z, 1.0, 2.05))
        drip = n.noise(n.comb(n.mul(n.add(x, y), 9.0), n.mul(z, 0.6), 0.0), scale=1.0, detail=3.0)
        cond = n.math("MULTIPLY", n.mul(cond, n.maprange(drip, 0.3, 0.75, 1.0, 0.35)), fogged, clamp=True)
        fogsh = n.mixshader(0.35, n.shader("ShaderNodeBsdfTranslucent", Color=(0.85, 0.85, 0.85)),
                            n.shader("ShaderNodeBsdfDiffuse", Color=(0.5, 0.5, 0.5)))
        sh = n.mixshader(cond, sh, fogsh)
    n.L(sh, out.inputs["Surface"])
    return m


m_glass = make_glass("glass")
m_glass_fog = make_glass("glass_fogged", fogged=0.7)
m_glass_dark = make_glass("glass_dark", tint=(0.35, 0.42, 0.46), refl=1.3, floor=0.07)
m_vinyl = make_glass("umbrella_vinyl", tint=(0.9, 0.95, 0.97), refl=1.6, rough=0.05, floor=0.1)

m_alu = mat("alu_dark", (0.035, 0.037, 0.04), rough=0.35, metal=0.9)
m_alu_l = mat("alu_light", (0.45, 0.46, 0.47), rough=0.3, metal=0.9)
m_steel = mat("steel_galv", (0.30, 0.31, 0.32), rough=0.32, metal=1.0)
m_chrome = mat("chrome", (0.75, 0.76, 0.78), rough=0.08, metal=1.0)
m_black = mat("black_matte", (0.010, 0.010, 0.012), rough=0.75)
m_blackg = mat("black_gloss", (0.012, 0.012, 0.014), rough=0.18, coat=0.5)
m_rubber = mat("rubber", (0.018, 0.018, 0.02), rough=0.55)
m_rust = mat("rust", (0.10, 0.045, 0.025), rough=0.7, metal=0.4)
m_white = mat("white_paint", (0.62, 0.62, 0.60), rough=0.35)
m_whiteg = mat("white_gloss", (0.70, 0.70, 0.68), rough=0.15, coat=0.6)
m_railp = mat("rail_paint", (0.36, 0.37, 0.37), rough=0.25, coat=0.5)
m_cream = mat("cream", (0.55, 0.48, 0.36), rough=0.5)
m_wood = mat("wood", (0.20, 0.10, 0.045), rough=0.4, coat=0.3)
m_woodl = mat("wood_light", (0.42, 0.27, 0.13), rough=0.45)
m_board = mat("sign_board", (0.014, 0.014, 0.017), rough=0.7)
m_tube_off = mat("tube_off", (0.10, 0.10, 0.11), rough=0.2)
m_paper = mat("paper", (0.75, 0.70, 0.58), rough=0.7)
m_red = mat("red_paint", (0.42, 0.03, 0.025), rough=0.3, coat=0.6)
m_redv = mat("red_vinyl", (0.38, 0.04, 0.03), rough=0.35)
m_blue = mat("blue_paint", (0.03, 0.10, 0.34), rough=0.3, coat=0.6)
m_teal = mat("teal_paint", (0.02, 0.22, 0.22), rough=0.35, coat=0.4)
m_yellow = mat("yellow_paint", (0.62, 0.42, 0.03), rough=0.4)
m_green = mat("green_paint", (0.04, 0.22, 0.08), rough=0.4)
m_indigo = mat("indigo_cloth", (0.015, 0.03, 0.10), rough=0.8, sheen=0.4)
m_plant = mat("plant", (0.03, 0.12, 0.035), rough=0.5)
m_skin = [mat("skin_a", (0.55, 0.38, 0.29), rough=0.55, sss=0.15), mat("skin_b", (0.42, 0.27, 0.19), rough=0.55, sss=0.15),
          mat("skin_c", (0.62, 0.46, 0.36), rough=0.55, sss=0.15)]
m_hair = [mat("hair_black", (0.012, 0.010, 0.010), rough=0.35), mat("hair_brown", (0.07, 0.04, 0.025), rough=0.4),
          mat("hair_grey", (0.25, 0.25, 0.26), rough=0.45)]
PRODUCT = [mat(f"product_{i}", c, rough=0.35) for i, c in enumerate(
    [(0.7, 0.08, 0.06), (0.9, 0.62, 0.06), (0.08, 0.35, 0.75), (0.1, 0.55, 0.2), (0.85, 0.85, 0.8), (0.75, 0.25, 0.5),
     (0.95, 0.4, 0.08), (0.2, 0.65, 0.7), (0.4, 0.2, 0.6), (0.9, 0.9, 0.3)])]


def make_wall(name, base, rough=0.6, tile=None, streak=0.4, mortar=(0.02, 0.02, 0.02), bump=0.4):
    """Wet facade: big tonal variation, vertical run-off streaks that are darker and glossier, optional tile grid."""
    m, n, p, out = mat_nodes(name)
    g = n.geo()
    x, y, z = n.sep(g.outputs["Position"])
    nx, ny, nz = n.sep(g.outputs["True Normal"])
    u = n.add(n.mul(x, n.abs(ny)), n.mul(y, n.abs(nx)))
    st = n.smooth(n.noise(n.comb(n.mul(u, 2.4), n.mul(z, 0.16), 0.0), scale=1.0, detail=3.0, rough=0.6), 0.5, 0.78)
    big = n.noise(n.comb(n.mul(u, 0.22), n.mul(z, 0.22), 1.7), scale=1.0, detail=2.0)
    col = n.mix(n.maprange(big, 0.3, 0.7, 0.0, 1.0), [c * 0.7 for c in base], [min(1.0, c * 1.2) for c in base])
    ro = n.maprange(big, 0.3, 0.7, rough - 0.08, rough + 0.08)
    if tile:
        bc, bf = n.brick(n.comb(u, z, 0.0), tile[0], tile[1], tile[2], (0.8, 0.8, 0.8), (1, 1, 1), mortar, offset=tile[3])
        col = n.mix(1.0, col, bc, blend="MULTIPLY")
        n.L(n.bump(n.math("SUBTRACT", 1.0, bf), strength=bump, dist=0.004), p.inputs["Normal"])
    col = n.mix(n.mul(st, streak), col, [c * 0.35 for c in base])
    ro = n.math("SUBTRACT", ro, n.mul(st, 0.28), clamp=True)
    n.L(col, p.inputs["Base Color"])
    n.L(ro, p.inputs["Roughness"])
    return m


m_conc = make_wall("concrete", (0.105, 0.10, 0.098), rough=0.62)
m_conc_d = make_wall("concrete_dark", (0.05, 0.05, 0.052), rough=0.6)
m_conc_w = make_wall("concrete_warm", (0.13, 0.105, 0.085), rough=0.6)
m_tile_w = make_wall("tile_white", (0.34, 0.33, 0.31), rough=0.28, tile=(0.22, 0.11, 0.008, 0.5))
m_tile_b = make_wall("tile_brown", (0.13, 0.065, 0.04), rough=0.3, tile=(0.24, 0.07, 0.008, 0.5))
m_tile_g = make_wall("tile_green", (0.03, 0.09, 0.075), rough=0.22, tile=(0.15, 0.15, 0.007, 0.0))
m_brick = make_wall("brick", (0.16, 0.06, 0.04), rough=0.6, tile=(0.23, 0.075, 0.012, 0.5), mortar=(0.25, 0.24, 0.22))
m_stucco = make_wall("stucco_grey", (0.16, 0.16, 0.165), rough=0.7)
m_panel = make_wall("panel_dark", (0.03, 0.033, 0.04), rough=0.35, tile=(1.2, 0.9, 0.012, 0.0), streak=0.25)
m_curb = make_wall("curb", (0.19, 0.185, 0.18), rough=0.5, streak=0.0)


def make_room():
    """Interior seen through a window. wcol = (colour pick, lit level, surface factor, random)."""
    m, n, out = mat_custom("room")
    a = n.attr("wcol")
    r, g, b = n.sepc(a.outputs["Color"])
    col = n.ramp(r, [(0.0, (1.0, 0.55, 0.24)), (0.38, (1.0, 0.72, 0.44)), (0.62, (0.80, 0.92, 1.0)), (0.80, (0.28, 0.5, 1.0)),
                     (0.88, (1.0, 0.2, 0.5)), (0.95, (0.3, 1.0, 0.72))], interp="CONSTANT")
    var = n.noise(n.geo().outputs["Position"], scale=2.1, detail=2.0)
    k = n.mul(n.mul(g, b), n.maprange(var, 0.3, 0.7, 0.7, 1.3))
    em = n.shader("ShaderNodeEmission", Color=col, Strength=n.mul(k, 2.6))
    n.L(n.addshader(em, n.shader("ShaderNodeBsdfDiffuse", Color=(0.035, 0.035, 0.04))), out.inputs["Surface"])
    return m


def make_blind():
    """Blinds (alpha < .5) or curtains (alpha > .5) right behind the glass, backlit by the room."""
    m, n, out = mat_custom("blind")
    a = n.attr("wcol")
    r, g, b = n.sepc(a.outputs["Color"])
    col = n.ramp(r, [(0.0, (1.0, 0.6, 0.3)), (0.38, (1.0, 0.76, 0.5)), (0.62, (0.82, 0.92, 1.0)), (0.80, (0.4, 0.6, 1.0)),
                     (0.88, (1.0, 0.3, 0.55)), (0.95, (0.4, 1.0, 0.75))], interp="CONSTANT")
    x, y, z = n.sep(n.geo().outputs["Position"])
    slat = n.mixf(n.lt(n.fract(n.mul(z, 15.0)), 0.78), 1.5, 0.32)
    fold = n.maprange(n.sin(n.mul(n.add(x, y), 34.0)), -1.0, 1.0, 0.22, 0.5)
    k = n.mul(g, n.mixf(n.gt(a.outputs["Alpha"], 0.5), slat, fold))
    em = n.shader("ShaderNodeEmission", Color=col, Strength=n.mul(k, 1.5))
    n.L(n.addshader(em, n.shader("ShaderNodeBsdfDiffuse", Color=(0.12, 0.11, 0.10))), out.inputs["Surface"])
    return m


m_room = make_room()
m_blind = make_blind()


def make_windows(name, wall, cw=1.5, ch=3.2, fu=(0.14, 0.86), fv=(0.26, 0.80), density=0.3, cool=0.3, strength=1.3,
                 floorlit=0.35, rough=0.6, warm=(1.0, 0.62, 0.30), coolc=(0.72, 0.9, 1.0), accent=(1.0, 0.2, 0.55)):
    """Far-building facade: a grid of windows drawn in the shader; each window gets its own random state."""
    m, n, p, out = mat_nodes(name)
    g = n.geo()
    nx, ny, nz = n.sep(g.outputs["True Normal"])
    px, py, pz = n.sep(n.texco().outputs["Object"])
    anx, any_ = n.abs(nx), n.abs(ny)
    u = n.add(n.add(n.mul(px, any_), n.mul(py, anx)), 500.0)
    rnd = n.objinfo().outputs["Random"]
    cu, cv = n.div(u, cw), n.div(pz, ch)
    iu, iv = n.floor(cu), n.floor(cv)
    fu_, fv_ = n.sub(cu, iu), n.sub(cv, iv)
    win = n.mul(n.mul(n.band(fu_, fu[0], fu[1]), n.band(fv_, fv[0], fv[1])), n.lt(n.abs(nz), 0.5))
    wv, wc = n.white(n.comb(n.add(iu, n.mul(anx, 31.7)), iv, n.mul(rnd, 113.0)))
    fl, _ = n.white(n.comb(iv, n.mul(rnd, 57.0), 3.3))
    prob = n.add(density, n.mul(n.gt(fl, 0.80), floorlit))
    lit = n.lt(wv, prob)
    wr, wg, wb = n.sepc(wc)
    inten = n.add(0.22, n.mul(n.mul(wg, wg), 0.78))
    vpos = n.div(n.sub(fv_, fv[0]), fv[1] - fv[0])
    blind = n.mixf(n.mul(n.gt(wb, 0.45), n.gt(vpos, n.sub(1.25, wb))), 1.0, 0.3)
    col = n.mix(n.lt(wr, cool), warm, coolc)
    col = n.mix(n.gt(wr, 0.95), col, accent)
    n.L(col, p.inputs["Emission Color"])
    n.L(n.mul(n.mul(n.mul(win, lit), n.mul(inten, blind)), strength), p.inputs["Emission Strength"])
    big = n.noise(n.comb(n.mul(u, 0.1), n.mul(pz, 0.1), rnd), scale=1.0, detail=2.0)
    wallc = n.mix(big, [c * 0.7 for c in wall], [c * 1.25 for c in wall])
    n.L(n.mix(win, wallc, (0.012, 0.014, 0.018)), p.inputs["Base Color"])
    n.L(n.mixf(win, rough, 0.08), p.inputs["Roughness"])
    return m


WIN_MATS = [
    make_windows("win_resi_warm", (0.07, 0.065, 0.06), cw=1.7, ch=3.1, density=0.27, cool=0.2, strength=1.5),
    make_windows("win_resi_tile", (0.10, 0.09, 0.08), cw=2.1, ch=3.0, fu=(0.2, 0.8), fv=(0.3, 0.78), density=0.22, cool=0.25, strength=1.5),
    make_windows("win_office_cool", (0.03, 0.035, 0.045), cw=1.35, ch=3.6, fu=(0.06, 0.94), fv=(0.22, 0.86), density=0.3, cool=0.85,
                 strength=1.1, floorlit=0.55, rough=0.3),
    make_windows("win_office_dark", (0.02, 0.022, 0.03), cw=1.1, ch=3.4, fu=(0.05, 0.95), fv=(0.2, 0.9), density=0.14, cool=0.7,
                 strength=1.0, floorlit=0.6, rough=0.22),
    make_windows("win_hotel", (0.085, 0.07, 0.06), cw=2.4, ch=3.0, fu=(0.25, 0.75), fv=(0.28, 0.82), density=0.42, cool=0.08, strength=1.4),
    make_windows("win_mixed", (0.06, 0.055, 0.065), cw=1.5, ch=2.9, density=0.24, cool=0.4, strength=1.6),
]
WIN_TOWER = make_windows("win_tower", (0.03, 0.04, 0.06), cw=1.8, ch=3.6, fu=(0.08, 0.92), fv=(0.18, 0.9), density=0.2, cool=0.9,
                         strength=1.6, floorlit=0.6, rough=0.18, coolc=(0.55, 0.85, 1.0), accent=(0.3, 1.0, 0.9))
WIN_SKY = [
    make_windows("win_sky_a", (0.02, 0.022, 0.03), cw=2.0, ch=3.8, fu=(0.1, 0.9), fv=(0.2, 0.85), density=0.22, cool=0.7, strength=1.3, floorlit=0.5),
    make_windows("win_sky_b", (0.03, 0.028, 0.03), cw=2.6, ch=3.4, fu=(0.2, 0.8), fv=(0.25, 0.8), density=0.3, cool=0.25, strength=1.3),
]


def make_road():
    """Wet asphalt in world coordinates: worn markings, gutters and wheel ruts that hold puddles, rain ripples."""
    m, n, p, out = mat_nodes("road")
    P = n.geo().outputs["Position"]
    x, y, z = n.sep(P)
    ax = n.abs(x)
    flat = n.comb(x, y, 0.0)

    def tri(v, c, w):
        return n.math("SUBTRACT", 1.0, n.div(n.abs(n.sub(v, c)), w), clamp=True)

    def blob(cx, cy, rx, ry, amp):
        dx, dy = n.div(n.sub(x, cx), rx), n.div(n.sub(y, cy), ry)
        r2 = n.add(n.mul(dx, dx), n.mul(dy, dy))
        return n.mul(n.math("SUBTRACT", 1.0, n.smooth(r2, 0.25, 1.0)), amp)

    # --- where the water stands
    n1 = n.noise(n.vmath("MULTIPLY", flat, (0.34, 0.12, 1.0)), scale=1.0, detail=3.0, rough=0.55)
    n2 = n.noise(flat, scale=0.05, detail=1.0)
    gut = n.smooth(ax, RW - 1.6, RW - 0.15)
    rut = n.mx(tri(ax, 0.95, 0.55), tri(ax, 2.7, 0.55))
    pv = n.add(n.add(n1, n.mul(gut, 0.2)), n.add(n.mul(rut, 0.06), n.mul(n.sub(n2, 0.5), 0.4)))
    for b in ((-1.25, -5.3, 1.9, 2.6, 0.32), (-2.6, 0.4, 1.3, 2.0, 0.2), (1.4, 4.5, 1.5, 2.6, 0.2), (-2.9, 9.0, 1.3, 3.2, 0.2), (0.3, 15.5, 2.2, 3.5, 0.18),
              (2.6, 27.0, 1.6, 2.5, 0.2), (-1.0, 33.0, 3.5, 2.2, 0.2)):
        pv = n.add(pv, blob(*b))
    TH = 0.625
    pud = n.smooth(pv, TH, TH + 0.028)
    film = n.smooth(pv, TH - 0.22, TH)

    # --- paint
    not_inter = n.math("SUBTRACT", 1.0, n.band(y, 26.9, 47.1))      # no lane lines through the junction
    dash = n.mul(n.mul(n.lt(ax, 0.075), n.lt(n.fract(n.div(y, 8.0)), 0.5)), not_inter)
    edge = n.mul(n.lt(n.abs(n.sub(ax, RW - 0.45)), 0.065), n.math("SUBTRACT", 1.0, n.band(y, 28.6, 45.4)))
    stripes = n.mul(n.lt(n.fract(n.add(n.div(x, 0.9), 0.25)), 0.5), n.lt(ax, RW - 0.25))
    zeb = n.mul(n.add(n.band(y, 28.6, 31.6), n.band(y, 42.4, 45.4)), stripes)
    zeb2 = n.mul(n.mul(n.band(ax, RW + 0.4, RW + 3.2), n.band(y, XS0 + 0.3, XS1 - 0.3)), n.lt(n.fract(n.div(y, 0.9)), 0.5))
    stop = n.add(n.mul(n.band(y, 26.9, 27.35), n.band(x, 0.15, RW - 0.5)), n.mul(n.band(y, 46.65, 47.1), n.band(x, -(RW - 0.5), -0.15)))
    mark = n.mx(n.mx(dash, edge), n.mx(n.mx(zeb, zeb2), stop))
    wear = n.mul(n.smooth(n.noise(flat, scale=5.5, detail=3.0), 0.28, 0.5), n.maprange(n.noise(flat, scale=45.0, detail=1.0), 0.3, 0.7, 0.65, 1.0))
    mark = n.mul(mark, wear)

    # --- colour
    fine = n.noise(flat, scale=7.0, detail=3.0)
    asph = n.mix(n.maprange(fine, 0.3, 0.7, 0.0, 1.0), (0.016, 0.016, 0.019), (0.042, 0.041, 0.043))
    crack = n.lt(n.abs(n.sub(n.noise(flat, scale=0.55, detail=2.0, dist=1.2), 0.5)), 0.005)
    asph = n.mix(crack, asph, (0.004, 0.004, 0.005))
    base = n.mix(n.mul(mark, 0.95), asph, (0.62, 0.61, 0.56))
    base = n.mix(n.mul(pud, 0.55), base, (0.0, 0.0, 0.0))
    n.L(base, p.inputs["Base Color"])

    # --- roughness: damp asphalt, glossier film near puddles, mirror in the puddle
    r_dry = n.maprange(n.noise(flat, scale=9.0, detail=2.0), 0.3, 0.7, 0.10, 0.25)
    r_dry = n.math("SUBTRACT", r_dry, n.mul(crack, 0.2), clamp=True)
    rough = n.mixf(pud, n.mixf(film, r_dry, 0.06), 0.006)
    n.L(rough, p.inputs["Roughness"])

    # --- bump: aggregate + gentle undulation outside puddles, raindrop rings inside
    agg = n.noise(flat, scale=80.0, detail=1.0)
    und = n.noise(flat, scale=2.0, detail=2.0)
    h_as = n.add(n.mul(agg, 0.0011), n.mul(und, 0.012))
    vor = n.voronoi(flat, scale=2.7)
    d = vor.outputs["Distance"]
    cr_, cg_, cb_ = n.sepc(vor.outputs["Color"])
    r0 = n.add(n.mul(cr_, 0.36), 0.04)
    q = n.div(n.sub(d, r0), 0.06)
    pk = n.math("EXPONENT", n.mul(n.mul(q, q), -1.0))
    ring = n.mul(n.mul(n.sin(n.mul(n.sub(d, r0), 52.0)), pk), n.math("SUBTRACT", 1.15, cr_))
    h = n.mixf(pud, h_as, n.mul(ring, 0.0013))
    n.L(n.bump(h, strength=1.0, dist=1.0), p.inputs["Normal"])
    p.inputs["IOR"].default_value = 1.4
    return m


def make_walk():
    m, n, p, out = mat_nodes("sidewalk")
    P = n.geo().outputs["Position"]
    x, y, z = n.sep(P)
    ax = n.abs(x)
    flat = n.comb(x, y, 0.0)
    bc, bf = n.brick(n.comb(y, x, 0.0), 0.40, 0.20, 0.010, (0.075, 0.073, 0.07), (0.115, 0.11, 0.105), (0.015, 0.015, 0.015))
    tact = n.band(ax, RW + 0.80, RW + 1.10)
    dots = n.mul(n.mul(n.sin(n.mul(x, 104.7)), n.sin(n.mul(y, 104.7))), tact)
    col = n.mix(tact, bc, (0.42, 0.29, 0.03))
    wn = n.noise(n.vmath("MULTIPLY", flat, (0.9, 0.35, 1.0)), scale=1.0, detail=3.0)
    pud = n.smooth(wn, 0.63, 0.66)
    col = n.mix(n.mul(pud, 0.5), col, (0, 0, 0))
    n.L(col, p.inputs["Base Color"])
    n.L(n.mixf(pud, n.maprange(wn, 0.3, 0.63, 0.42, 0.14), 0.01), p.inputs["Roughness"])
    h = n.add(n.mul(n.math("SUBTRACT", 1.0, bf), 0.003), n.mul(n.mx(dots, 0.0), 0.004))
    n.L(n.bump(n.mixf(pud, h, 0.0031), strength=1.0, dist=1.0), p.inputs["Normal"])
    return m


m_road = make_road()
m_walk = make_walk()


def make_shopstrip(name, c1, c2, strength=1.4):
    """Distant shopfront: bright ceiling, darker goods and people below, mullions."""
    m, n, p, out = mat_nodes(name)
    x, y, z = n.sep(n.geo().outputs["Position"])
    u = n.add(y, n.mul(x, 0.37))
    mull = n.lt(n.fract(n.div(u, 1.9)), 0.045)
    vg = n.smooth(z, 0.3, 3.0)
    blobs = n.smooth(n.noise(n.comb(n.mul(u, 1.3), n.mul(z, 1.6), 0.0), scale=1.0, detail=2.0), 0.45, 0.62)
    cm = n.noise(n.comb(n.mul(u, 0.11), 0.0, 4.0), scale=1.0, detail=0.0)
    col = n.mix(n.smooth(cm, 0.4, 0.6), c1, c2)
    k = n.mul(n.mul(n.add(0.2, n.mul(vg, 0.8)), n.math("SUBTRACT", 1.0, n.mul(blobs, 0.75))), n.math("SUBTRACT", 1.0, mull))
    n.L(col, p.inputs["Emission Color"])
    n.L(n.mul(k, strength), p.inputs["Emission Strength"])
    p.inputs["Base Color"].default_value = (0.01, 0.01, 0.012, 1)
    p.inputs["Roughness"].default_value = 0.06
    return m


SHOPSTRIPS = [make_shopstrip("shopstrip_warm", (1.0, 0.55, 0.22), (1.0, 0.8, 0.5)),
              make_shopstrip("shopstrip_cool", (0.5, 0.85, 1.0), (0.9, 0.97, 1.0)),
              make_shopstrip("shopstrip_pink", (1.0, 0.25, 0.5), (1.0, 0.7, 0.45)),
              make_shopstrip("shopstrip_teal", (0.2, 1.0, 0.75), (0.8, 1.0, 0.9))]


def make_lightbox(name, color, strength=1.6):
    """Backlit acrylic: hot bands where the tubes sit behind it, dirt, darker rim."""
    m, n, p, out = mat_nodes(name)
    px, py, pz = n.sep(n.texco().outputs["Object"])
    band_ = n.maprange(n.math("COSINE", n.mul(pz, 21.0)), -1.0, 1.0, 0.72, 1.0)
    dirt = n.maprange(n.noise(n.comb(px, pz, 0.0), scale=3.0, detail=3.0), 0.3, 0.7, 0.8, 1.0)
    n.L(n.mul(n.mul(band_, dirt), strength), p.inputs["Emission Strength"])
    p.inputs["Emission Color"].default_value = tuple(color) + (1,)
    p.inputs["Base Color"].default_value = tuple(c * 0.5 for c in color) + (1,)
    p.inputs["Roughness"].default_value = 0.2
    return m


_LB = {}


def lightbox_mat(color, strength=1.6):
    k = (tuple(round(c, 3) for c in color), strength)
    if k not in _LB:
        _LB[k] = make_lightbox(f"lightbox_{len(_LB)}", color, strength)
    return _LB[k]


def make_screen(name, kind, c1, c2, strength=1.6, led=0.0, scale=1.0):
    """LED screen content generated in object space (x across, z up, face centred on the object origin).
    kind: 'sun' (retro sun over a grid), 'bars' (equaliser), 'wave' (scan bands)."""
    m, n, p, out = mat_nodes(name)
    px, py, pz = n.sep(n.vmath("SCALE", n.texco().outputs["Object"], scale=1.0 / scale))
    oi = n.objinfo().outputs["Random"]
    if kind == "sun":
        r = n.math("SQRT", n.add(n.mul(px, px), n.mul(n.sub(pz, 0.9), n.sub(pz, 0.9))))
        disc = n.lt(r, 2.4)
        slits = n.gt(n.fract(n.mul(pz, 1.9)), n.maprange(pz, -1.6, 2.6, 0.75, 0.1))
        sun = n.mul(disc, n.mx(slits, n.gt(pz, 1.3)))
        sky = n.mix(n.smooth(pz, -4.0, 5.0), c2, [c * 0.12 for c in c1])
        grid = n.mul(n.lt(pz, -1.6), n.mx(n.lt(n.fract(n.div(px, n.mx(n.mul(n.sub(-1.2, pz), 0.5), 0.05))), 0.07),
                                          n.lt(n.fract(n.math("POWER", n.mx(n.sub(-1.6, pz), 0.0), 0.6)), 0.12)))
        col = n.mix(sun, sky, n.mix(n.smooth(pz, -1.5, 3.2), c1, (1.0, 0.85, 0.3)))
        col = n.mix(grid, col, c2)
        k = n.add(0.35, n.mul(n.mx(sun, grid), 0.65))
    elif kind == "bars":
        iu = n.floor(n.div(px, 0.55))
        hv, _ = n.white(n.comb(iu, oi, 2.0))
        top = n.sub(n.mul(hv, 7.0), 3.5)
        bar = n.mul(n.mul(n.lt(pz, top), n.lt(n.fract(n.div(px, 0.55)), 0.72)), n.lt(n.fract(n.mul(pz, 2.6)), 0.78))
        col = n.mix(n.smooth(pz, -3.5, 3.5), c1, c2)
        k = n.add(0.05, n.mul(bar, 0.95))
    else:
        w = n.math("SINE", n.add(n.mul(pz, 1.7), n.mul(n.sin(n.mul(px, 0.6)), 1.4)))
        col = n.mix(n.smooth(w, -0.4, 0.4), c1, c2)
        k = n.maprange(w, -1.0, 1.0, 0.2, 1.0)
    if led > 0:
        dot = n.mul(n.abs(n.sin(n.mul(px, math.pi / led))), n.abs(n.sin(n.mul(pz, math.pi / led))))
        k = n.mul(k, n.maprange(dot, 0.0, 0.6, 0.25, 1.0))
    n.L(col, p.inputs["Emission Color"])
    n.L(n.mul(k, strength), p.inputs["Emission Strength"])
    p.inputs["Base Color"].default_value = (0.005, 0.005, 0.006, 1)
    p.inputs["Roughness"].default_value = 0.25
    return m


m_scr_sun = make_screen("screen_sun", "sun", NEON["magenta"], NEON["blue"], strength=1.5, scale=0.41)
m_scr_bars = make_screen("screen_bars", "bars", NEON["cyan"], NEON["magenta"], strength=2.0)
m_scr_wave = make_screen("screen_wave", "wave", NEON["teal"], NEON["purple"], strength=1.2)
m_scr_wave2 = make_screen("screen_wave2", "wave", NEON["amber"], NEON["magenta"], strength=1.1)


_LANT = {}


def lantern_mat(key, strength=1.6):
    """Paper lantern: lit from inside (bright middle, darker rim), bamboo ribs as fine dark rings."""
    k = (key, strength)
    if k in _LANT:
        return _LANT[k]
    m, n, out = mat_custom(f"lantern_{key}_{strength:g}")
    x, y, z = n.sep(n.geo().outputs["Position"])
    rib = n.mixf(n.lt(n.fract(n.mul(z, 26.0)), 0.16), 1.0, 0.45)
    lw_ = n.node("ShaderNodeLayerWeight")
    lw_.inputs["Blend"].default_value = 0.35
    core = n.maprange(lw_.outputs["Facing"], 0.0, 1.0, 1.25, 0.3)
    em = n.shader("ShaderNodeEmission", Color=NEON[key], Strength=n.mul(n.mul(rib, core), strength))
    n.L(n.addshader(em, n.shader("ShaderNodeBsdfDiffuse", Color=[c * 0.25 for c in NEON[key]])), out.inputs["Surface"])
    _LANT[k] = m
    return m


def make_paint(name, color, flake=0.6):
    """Car paint under rain: metallic base, clear coat, beads of water in the coat normal."""
    m, n, p, out = mat_nodes(name)
    P = n.texco().outputs["Object"]
    v = n.voronoi(P, scale=95.0)
    bead = n.math("SUBTRACT", 1.0, n.smooth(v.outputs["Distance"], 0.0, 0.42))
    cr_, cg_, cb_ = n.sepc(v.outputs["Color"])
    bead = n.mul(bead, n.gt(cr_, 0.45))
    i = p.inputs
    i["Base Color"].default_value = tuple(color) + (1,)
    i["Metallic"].default_value = flake
    i["Roughness"].default_value = 0.3
    i["Coat Weight"].default_value = 1.0
    i["Coat Roughness"].default_value = 0.03
    n.L(n.bump(bead, strength=0.5, dist=0.004), i["Coat Normal"])
    return m


def make_rain(name="rain", opacity=0.26):
    """Streak of falling water: mostly see-through, lights up when lit from behind or the side."""
    m, n, out = mat_custom(name)
    uv = n.node("ShaderNodeUVMap").outputs["UV"]
    u, v, _ = n.sep(uv)
    across = n.math("SUBTRACT", 1.0, n.pow(n.abs(n.sub(n.mul(u, 2.0), 1.0)), 2.0))
    along = n.sin(n.mul(v, math.pi))
    a = n.mul(n.mul(across, along), opacity)
    lit = n.addshader(n.shader("ShaderNodeBsdfTranslucent", Color=(0.9, 0.95, 1.0)),
                      n.shader("ShaderNodeBsdfGlossy", Color=(0.5, 0.5, 0.5), Roughness=0.3))
    n.L(n.mixshader(a, n.shader("ShaderNodeBsdfTransparent", Color=(1, 1, 1)), lit), out.inputs["Surface"])
    return m


def make_water():
    m, n, out = mat_custom("water_splash")
    lit = n.addshader(n.shader("ShaderNodeBsdfTranslucent", Color=(0.9, 0.95, 1.0)),
                      n.shader("ShaderNodeBsdfGlossy", Color=(1, 1, 1), Roughness=0.12))
    n.L(n.mixshader(0.2, n.shader("ShaderNodeBsdfTransparent", Color=(1, 1, 1)), lit), out.inputs["Surface"])
    return m


m_rain = make_rain()
m_water = make_water()
m_drip = make_rain("drip", 0.6)


def make_fog(name, density, color=(0.75, 0.86, 1.0), aniso=0.55):
    m, n, out = mat_custom(name)
    n.L(n.shader("ShaderNodeVolumeScatter", Color=color, Density=density, Anisotropy=aniso), out.inputs["Volume"])
    return m


def make_steam():
    m, n, out = mat_custom("steam")
    P = n.texco().outputs["Object"]
    x, y, z = n.sep(P)
    nz_ = n.noise(n.vmath("ADD", n.vmath("MULTIPLY", P, (1.6, 1.6, 0.9)), (0.0, 0.0, 3.1)), scale=1.0, detail=4.0, rough=0.6, dist=0.6)
    hgt = n.smooth(z, -0.2, 3.6)                                   # 0 at the grate, 1 at the top
    rad = n.math("SQRT", n.add(n.mul(n.sub(x, n.mul(hgt, 0.9)), n.sub(x, n.mul(hgt, 0.9))), n.mul(y, y)))
    core = n.math("SUBTRACT", 1.0, n.smooth(rad, 0.05, n.add(0.28, n.mul(hgt, 1.1))))
    dens = n.mul(n.mul(n.smooth(nz_, 0.36, 0.72), core), n.math("SUBTRACT", 1.0, n.pow(hgt, 1.6)))
    n.L(n.shader("ShaderNodeVolumeScatter", Color=(0.95, 0.96, 1.0), Density=n.mul(dens, 3.2), Anisotropy=0.3), out.inputs["Volume"])
    return m


def make_holo(name, color, strength=3.0, kind="ticker"):
    """Hologram ring: light only (emission + transparent), pattern runs round the ring."""
    m, n, out = mat_custom(name)
    px, py, pz = n.sep(n.texco().outputs["Object"])
    ang = n.math("ARCTAN2", py, px)
    if kind == "ticker":
        cell = n.mul(ang, 40.0)
        hv, hc = n.white(n.comb(n.floor(cell), n.floor(n.mul(pz, 0.9)), 1.0))
        on = n.mul(n.gt(hv, 0.42), n.lt(n.fract(cell), 0.8))
        edge = n.gt(n.abs(n.sub(n.fract(n.mul(pz, 0.9)), 0.5)), 0.4)
        k = n.mx(n.mul(on, 0.9), n.mul(edge, 0.0))
    else:
        k = n.maprange(n.sin(n.mul(ang, 90.0)), -1.0, 1.0, 0.25, 1.0)
    em = n.shader("ShaderNodeEmission", Color=color, Strength=n.mul(k, strength))
    n.L(n.addshader(em, n.shader("ShaderNodeBsdfTransparent", Color=(1, 1, 1))), out.inputs["Surface"])
    return m


def make_trail(name, color, strength=4.0):
    """Light trail ribbon: fades along its length (uv.x 0 = head, 1 = tail)."""
    m, n, out = mat_custom(name)
    u, v, _ = n.sep(n.node("ShaderNodeUVMap").outputs["UV"])
    fade = n.pow(n.math("SUBTRACT", 1.0, u, clamp=True), 1.8)
    across = n.math("SUBTRACT", 1.0, n.pow(n.abs(n.sub(n.mul(v, 2.0), 1.0)), 2.0))
    em = n.shader("ShaderNodeEmission", Color=color, Strength=n.mul(n.mul(fade, across), strength))
    n.L(n.addshader(em, n.shader("ShaderNodeBsdfTransparent", Color=(1, 1, 1))), out.inputs["Surface"])
    return m


def make_world():
    w = bpy.data.worlds.new("S3_Night")
    n = NT(w.node_tree)
    bg = next(nd for nd in w.node_tree.nodes if nd.type == "BACKGROUND")
    d = n.texco().outputs["Generated"]
    x, y, z = n.sep(d)
    col = n.ramp(z, [(0.0, (0.030, 0.016, 0.040)), (0.10, (0.020, 0.013, 0.034)), (0.35, (0.008, 0.008, 0.02)), (1.0, (0.002, 0.003, 0.008))])
    cl = n.noise(n.vmath("MULTIPLY", d, (2.5, 2.5, 7.0)), scale=1.0, detail=5.0, rough=0.62, dist=0.5)
    col = n.mix(n.smooth(cl, 0.42, 0.75), col, n.mix(n.smooth(z, 0.0, 0.6), (0.075, 0.04, 0.085), (0.02, 0.018, 0.04)))
    n.L(col, bg.inputs["Color"])
    n.L(n.mixf(n.lightpath().outputs["Is Camera Ray"], 0.6, 1.0), bg.inputs["Strength"])
    sc.world = w
    return w


make_world()

tick("p15_kit")
# ================================================================ KIT: people, umbrellas, cars
COATS = [mat(f"coat_{i}", c, rough=0.55, sheen=0.3) for i, c in enumerate(
    [(0.012, 0.012, 0.015), (0.035, 0.02, 0.015), (0.02, 0.03, 0.06), (0.10, 0.085, 0.06), (0.03, 0.045, 0.035),
     (0.16, 0.03, 0.03), (0.07, 0.07, 0.08), (0.22, 0.19, 0.15), (0.20, 0.12, 0.02), (0.02, 0.07, 0.08)])]
PANTS = [mat(f"pants_{i}", c, rough=0.6) for i, c in enumerate([(0.01, 0.01, 0.012), (0.02, 0.025, 0.045), (0.045, 0.04, 0.035)])]
m_shoe = mat("shoe", (0.012, 0.01, 0.01), rough=0.22, coat=0.4)
m_bag = [mat("bag_tan", (0.16, 0.09, 0.04), rough=0.5), mat("bag_black", (0.015, 0.015, 0.017), rough=0.4),
         mat("bag_red", (0.3, 0.03, 0.03), rough=0.45), mat("bag_white", (0.6, 0.6, 0.58), rough=0.5, alpha=1.0)]
m_scarf = [mat("scarf_red", (0.45, 0.03, 0.03), rough=0.8, sheen=0.5), mat("scarf_yellow", (0.55, 0.36, 0.03), rough=0.8, sheen=0.5),
           mat("scarf_grey", (0.3, 0.3, 0.32), rough=0.8, sheen=0.5)]
m_phone = emit_mat("phone_screen", (0.7, 0.85, 1.0), 2.5)
UMB = {"black": mat("umb_black", (0.012, 0.012, 0.014), rough=0.28, coat=0.6),
       "red": mat("umb_red", (0.40, 0.025, 0.03), rough=0.3, coat=0.6),
       "navy": mat("umb_navy", (0.02, 0.05, 0.2), rough=0.3, coat=0.6),
       "yellow": mat("umb_yellow", (0.62, 0.45, 0.03), rough=0.3, coat=0.6),
       "teal": mat("umb_teal", (0.02, 0.26, 0.26), rough=0.3, coat=0.6),
       "white": mat("umb_white", (0.6, 0.6, 0.6), rough=0.3, coat=0.6),
       "clear": m_vinyl}


def umbrella_geo(mb, top, axis, kind="black", R=0.55, depth=0.25, shaft=0.86):
    """Open umbrella: 8 ribs with the cloth pulled in between them, stretchers, shaft and J handle."""
    M0 = mb.M.copy()
    q = Vector((0, 0, 1)).rotation_difference(Vector(axis).normalized())
    mb.M = M0 @ Matrix.Translation(Vector(top)) @ q.to_matrix().to_4x4()
    bm, m = mb.bm, UMB[kind]
    idx = mb.mi(m)
    segs, rings = 16, 5
    c = bm.verts.new(mb.M @ Vector((0, 0, 0)))
    V, L = [], []
    for j in range(1, rings + 1):
        t = j / rings
        ring, loc = [], []
        for k in range(segs):
            a = 2 * math.pi * k / segs
            rr, zz = R * t, -depth * t ** 1.9
            if k % 2:
                rr *= 1 - 0.06 * t * t
                zz += 0.02 * t * t
            p = Vector((rr * math.cos(a), rr * math.sin(a), zz))
            loc.append(p)
            ring.append(bm.verts.new(mb.M @ p))
        V.append(ring)
        L.append(loc)
    for k in range(segs):
        f = bm.faces.new((c, V[0][k], V[0][(k + 1) % segs]))
        f.material_index, f.smooth = idx, True
    for j in range(rings - 1):
        for k in range(segs):
            k2 = (k + 1) % segs
            f = bm.faces.new((V[j][k], V[j + 1][k], V[j + 1][k2], V[j][k2]))
            f.material_index, f.smooth = idx, True
    bm.edges.index_update()
    for k in range(0, segs, 2):
        chain = [c] + [V[j][k] for j in range(rings)]
        for a_, b_ in zip(chain, chain[1:]):
            e = bm.edges.get((a_, b_))
            if e:
                e.smooth = False
        mb.tube([Vector((0, 0, -0.006))] + [L[j][k] + Vector((0, 0, -0.006)) for j in range(rings)], 0.0032,
                m_white if kind == "clear" else m_steel, segs=4)
        mb.rod((0, 0, -0.33), L[2][k] + Vector((0, 0, -0.008)), 0.0025, m_steel, segs=4)
        mb.sphere(0.008, L[rings - 1][k], m_steel, u=6, v=4)
    mb.rod((0, 0, 0.05), (0, 0, -shaft), 0.006, m_steel, segs=6)
    mb.cyl(0.012, 0.03, (0, 0, -0.34), m_black, segs=6)
    hz = -shaft
    mb.tube([(0, 0, hz + 0.02), (0, 0, hz - 0.09), (0.02, 0, hz - 0.125), (0.055, 0, hz - 0.125), (0.075, 0, hz - 0.09)], 0.012,
            m_wood if kind != "clear" else m_white, segs=6)
    mb.M = M0


def person_geo(mb, rng, h=1.70, pose="walk", phase=0.9, coat=None, hem=0.52, pants=None, skin=None, hair=None,
               hairstyle="short", umb=None, bag=None, scarf=None, phone=False, lean=0.03, umb_tilt=(-0.05, -0.10)):
    """A proportioned figure at the builder's current matrix, facing local +Y, feet on z = 0 (or seat top for 'sit')."""
    k = h / 1.70
    coat = coat or rng.choice(COATS)
    pants = pants or rng.choice(PANTS)
    skin = skin or rng.choice(m_skin)
    hair = hair or rng.choice(m_hair)
    A = 0.42 if pose == "walk" else 0.0
    legs = []
    for sgn in (-1, 1):
        hip = Vector((sgn * 0.085 * k, 0, 0))
        if pose == "sit":
            knee = hip + Vector((sgn * 0.02, 0.42 * k, 0.0))
            ankle = knee + Vector((0, 0.05 * k, -0.42 * k))
        else:
            s = math.sin(phase) * sgn
            a = A * s
            knee = hip + Vector((0, math.sin(a), -math.cos(a))) * 0.45 * k
            kb = (0.10 + 0.55 * max(0.0, -s)) if pose == "walk" else 0.02
            ankle = knee + Vector((0, math.sin(a - kb), -math.cos(a - kb))) * 0.42 * k
        legs.append((hip, knee, ankle))
    zoff = 0.10 * k if pose == "sit" else 0.065 * k - min(l[2].z for l in legs)
    O = Vector((0, 0, zoff))
    for hip, knee, ankle in legs:
        mb.rod(O + hip, O + knee, 0.084 * k, pants, segs=8, r2=0.063 * k)
        mb.sphere(0.063 * k, O + knee, pants, u=8, v=6)
        mb.rod(O + knee, O + ankle, 0.060 * k, pants, segs=8, r2=0.044 * k)
        mb.sphere(1.0, O + ankle + Vector((0, 0.06 * k, -0.022 * k)), m_shoe, scale=(0.05 * k, 0.125 * k, 0.045 * k), u=8, v=6)

    def ring(z, rx, ry, n=14):
        return [O + Vector((rx * math.cos(2 * math.pi * i / n), lean * z + ry * math.sin(2 * math.pi * i / n), z)) for i in range(n)]

    hemz = -0.02 * k if pose == "sit" else (hem - 0.90) * k
    fl = max(0.0, (0.9 - hem) / 0.4)
    secs = [ring(hemz, (0.195 + 0.045 * fl) * k, (0.132 + 0.05 * fl) * k), ring(0.02 * k, 0.185 * k, 0.128 * k),
            ring(0.20 * k, 0.158 * k, 0.112 * k), ring(0.38 * k, 0.188 * k, 0.126 * k), ring(0.50 * k, 0.205 * k, 0.115 * k),
            ring(0.555 * k, 0.15 * k, 0.10 * k), ring(0.585 * k, 0.07 * k, 0.065 * k)]
    mb.loft(secs, coat)
    hands = []
    shz = 0.50 * k
    for sgn in (-1, 1):
        sh = O + Vector((sgn * 0.205 * k, lean * shz, shz))
        mb.sphere(0.068 * k, sh, coat, u=8, v=6)
        if umb and sgn == 1:
            el = sh + Vector((0.03, 0.10, -0.26)).normalized() * 0.28 * k
            hand = el + Vector((-0.11, 0.20, 0.17)).normalized() * 0.25 * k
        elif phone and sgn == -1:
            el = sh + Vector((-0.02, 0.06, -0.27)).normalized() * 0.28 * k
            hand = el + Vector((0.09, 0.19, 0.15)).normalized() * 0.25 * k
        elif pose == "sit":
            el = sh + Vector((sgn * 0.03, 0.14, -0.24)).normalized() * 0.28 * k
            hand = el + Vector((-sgn * 0.04, 0.24, 0.03)).normalized() * 0.25 * k
        else:
            b = -A * 0.8 * math.sin(phase) * sgn
            el = sh + Vector((sgn * 0.035, math.sin(b), -math.cos(b))).normalized() * 0.29 * k
            hand = el + Vector((0, math.sin(b + 0.3), -math.cos(b + 0.3))) * 0.25 * k
        mb.rod(sh, el, 0.064 * k, coat, segs=8, r2=0.054 * k)
        mb.sphere(0.054 * k, el, coat, u=8, v=6)
        mb.rod(el, hand, 0.053 * k, coat, segs=8, r2=0.044 * k)
        mb.sphere(0.043 * k, hand + (hand - el).normalized() * 0.04 * k, skin, u=8, v=6, scale=(0.85, 1, 1.1))
        hands.append(hand)
    nb = O + Vector((0, lean * 0.58 * k, 0.57 * k))
    mb.cyl(0.048 * k, 0.10 * k, nb, skin, segs=8)
    mb.cyl(0.085 * k, 0.045 * k, nb + Vector((0, -0.005, -0.012 * k)), coat, segs=10, r2=0.07 * k)            # collar
    hc = nb + Vector((0, 0.012 * k, 0.148 * k))
    mb.sphere(1.0, hc, skin, scale=(0.092 * k, 0.104 * k, 0.118 * k), u=12, v=8)
    if hairstyle == "hood":
        mb.sphere(1.0, hc + Vector((0, -0.03 * k, 0.004 * k)), coat, scale=(0.118 * k, 0.122 * k, 0.136 * k), u=12, v=8)
    elif hairstyle != "bald":
        mb.sphere(1.0, hc + Vector((0, -0.016 * k, 0.016 * k)), hair, scale=(0.099 * k, 0.106 * k, 0.117 * k), u=12, v=8)
        if hairstyle in ("bob", "long"):
            mb.sphere(1.0, hc + Vector((0, -0.032 * k, -0.055 * k)), hair, scale=(0.106 * k, 0.098 * k, 0.125 * k), u=12, v=8)
        if hairstyle == "long":
            mb.sphere(1.0, hc + Vector((0, -0.06 * k, -0.16 * k)), hair, scale=(0.095 * k, 0.055 * k, 0.17 * k), u=10, v=8)
        if hairstyle == "cap":
            mb.sphere(1.0, hc + Vector((0, 0.0, 0.05 * k)), coat, scale=(0.101 * k, 0.11 * k, 0.075 * k), u=12, v=8)
            mb.box((0.15 * k, 0.11 * k, 0.012 * k), hc + Vector((0, 0.13 * k, 0.045 * k)), coat)
    if scarf:
        mb.cyl(0.088 * k, 0.075 * k, nb + Vector((0, 0, -0.025 * k)), scarf, segs=10)
    if bag == "tote":
        bm_ = rng.choice(m_bag[:3])
        bc = O + Vector((-0.275 * k, 0.02, -0.02 * k))
        mb.box((0.09 * k, 0.30 * k, 0.25 * k), bc, bm_)
        for dy in (-0.09, 0.09):
            mb.rod(bc + Vector((0.02, dy * k, 0.12 * k)), O + Vector((-0.2 * k, lean * shz, shz + 0.06 * k)), 0.008, bm_, segs=4)
    elif bag == "back":
        mb.sphere(1.0, O + Vector((0, -0.18 * k + lean * 0.3, 0.30 * k)), m_bag[1], scale=(0.14 * k, 0.08 * k, 0.2 * k), u=10, v=8)
        for sg in (-1, 1):
            mb.rod(O + Vector((sg * 0.11 * k, -0.1 * k, 0.46 * k)), O + Vector((sg * 0.13 * k, 0.09 * k, 0.47 * k)), 0.014, m_bag[1], segs=4)
    elif bag == "brief":
        mb.box((0.07 * k, 0.36 * k, 0.26 * k), hands[0] + Vector((0, 0, -0.19 * k)), m_bag[1])
    elif bag == "shop":
        mb.sphere(1.0, hands[0] + Vector((0, 0, -0.2 * k)), m_bag[3], scale=(0.09 * k, 0.07 * k, 0.13 * k), u=8, v=6)
    if phone:
        mb.box((0.075, 0.012, 0.15), hands[0] + Vector((0.0, 0.035, 0.075)), m_phone, rot=(-0.45, 0, 0))
    if umb:
        axis = Vector((umb_tilt[0], umb_tilt[1], 1)).normalized()
        umbrella_geo(mb, hands[1] + axis * 0.80 * k, axis, kind=umb, shaft=0.86 * k)
    return {"head": hc, "hands": hands, "O": O}


def make_person(name, x, y, heading, rng, col=None, z=None, **kw):
    """heading in degrees: 0 = walking +Y (away from camera), 180 = toward camera, 90 = toward -X."""
    mb = MB(name)
    info = person_geo(mb, rng, **kw)
    if z is None:
        z = CURB if abs(x) > RW else 0.0
    ob = mb.finish(col=col, loc=(x, y, z), rot=(0, 0, math.radians(heading)), sharp=50)
    return ob, info


# ---------------------------------------------------------------- cars
from mathutils.bvhtree import BVHTree
m_carglass = mat("car_glass", (0.008, 0.01, 0.012), rough=0.03, coat=0.3)
m_cartrim = mat("car_trim", (0.012, 0.012, 0.013), rough=0.45)
m_tire = mat("tire", (0.014, 0.014, 0.015), rough=0.5)
m_rim = mat("rim", (0.35, 0.36, 0.38), rough=0.25, metal=1.0)
m_tail = emit_mat("lamp_tail", (1.0, 0.03, 0.02), 7.0)
m_brake = emit_mat("lamp_brake", (1.0, 0.04, 0.02), 16.0)
m_head = emit_mat("lamp_head", (1.0, 0.95, 0.85), 40.0)
m_head_off = mat("lamp_head_off", (0.5, 0.5, 0.52), rough=0.1, metal=0.6)
m_plate = mat("plate", (0.42, 0.42, 0.38), rough=0.4, emit=(1, 0.95, 0.85), emit_str=0.08)
m_amber = emit_mat("lamp_amber", (1.0, 0.45, 0.03), 3.0)
PAINTS = {k: make_paint(f"paint_{k}", c, f) for k, (c, f) in {
    "black": ((0.012, 0.012, 0.014), 0.5), "white": ((0.62, 0.62, 0.6), 0.1), "silver": ((0.32, 0.33, 0.35), 0.9),
    "red": ((0.30, 0.012, 0.012), 0.5), "blue": ((0.012, 0.04, 0.16), 0.6), "taxi": ((0.62, 0.36, 0.02), 0.1),
    "teal": ((0.02, 0.12, 0.13), 0.6), "grey": ((0.08, 0.085, 0.09), 0.7)}.items()}

# station tables: (y, z_top, w_top, z_belt, w_belt, z_bot); rear -> front
CAR_SPEC = {
    "sedan": dict(st=[(-2.275, 0.78, 0.55, 0.72, 0.70, 0.44), (-2.20, 0.93, 0.68, 0.80, 0.82, 0.31), (-1.95, 0.985, 0.75, 0.86, 0.885, 0.24),
                      (-1.45, 1.00, 0.76, 0.90, 0.90, 0.21), (-1.18, 1.03, 0.75, 0.92, 0.90, 0.21), (-0.95, 1.165, 0.695, 0.93, 0.90, 0.21),
                      (-0.55, 1.40, 0.60, 0.93, 0.90, 0.21), (0.00, 1.44, 0.61, 0.93, 0.90, 0.21), (0.07, 1.44, 0.61, 0.93, 0.90, 0.21),
                      (0.50, 1.42, 0.60, 0.93, 0.90, 0.21), (0.95, 1.17, 0.69, 0.925, 0.90, 0.21), (1.22, 1.02, 0.74, 0.92, 0.90, 0.21),
                      (1.60, 0.965, 0.76, 0.88, 0.895, 0.22), (2.05, 0.87, 0.72, 0.80, 0.86, 0.25), (2.21, 0.75, 0.62, 0.70, 0.78, 0.31),
                      (2.275, 0.65, 0.48, 0.62, 0.64, 0.42)],
                  top_glass=(4, 5, 9, 10), side_glass=(5, 6, 8, 9), pillar=(7,), axles=(-1.32, 1.38), wr=0.33, track=0.80,
                  tail=(0.60, 0.835, 0.40, 0.11), head=(0.62, 0.70, 0.34, 0.085), mirror=(0.80, 0.99), bar=True),
    "kei": dict(st=[(-1.70, 1.18, 0.58, 0.95, 0.64, 0.46), (-1.64, 1.48, 0.65, 1.0, 0.72, 0.31), (-1.35, 1.60, 0.62, 1.0, 0.735, 0.24),
                    (-0.40, 1.62, 0.62, 1.0, 0.74, 0.22), (-0.33, 1.62, 0.62, 1.0, 0.74, 0.22), (0.45, 1.60, 0.62, 1.0, 0.74, 0.22),
                    (0.80, 1.30, 0.66, 1.0, 0.74, 0.22), (1.10, 1.02, 0.68, 0.96, 0.74, 0.22), (1.45, 0.92, 0.66, 0.86, 0.72, 0.24),
                    (1.66, 0.72, 0.58, 0.68, 0.66, 0.32), (1.70, 0.62, 0.46, 0.60, 0.56, 0.42)],
                top_glass=(5, 6), side_glass=(2, 4, 5), pillar=(3,), axles=(-1.12, 1.16), wr=0.29, track=0.64,
                tail=(0.60, 1.15, 0.10, 0.34), head=(0.50, 0.74, 0.26, 0.11), mirror=(0.72, 1.05), bar=False, rear_glass=(1.02, 1.32, 0.36)),
    "van": dict(st=[(-2.30, 1.70, 0.70, 1.05, 0.78, 0.46), (-2.24, 1.86, 0.78, 1.08, 0.86, 0.30), (-1.2, 1.90, 0.80, 1.10, 0.88, 0.25),
                    (0.2, 1.90, 0.80, 1.10, 0.88, 0.25), (0.28, 1.90, 0.80, 1.10, 0.88, 0.25), (1.05, 1.88, 0.78, 1.10, 0.88, 0.25),
                    (1.60, 1.24, 0.80, 1.08, 0.88, 0.25), (2.02, 1.02, 0.78, 0.95, 0.87, 0.27), (2.24, 0.85, 0.70, 0.80, 0.80, 0.32),
                    (2.30, 0.72, 0.55, 0.68, 0.66, 0.42)],
                top_glass=(5,), side_glass=(4, 5), pillar=(3,), axles=(-1.35, 1.45), wr=0.34, track=0.78,
                tail=(0.74, 1.15, 0.11, 0.42), head=(0.60, 0.78, 0.30, 0.10), mirror=(1.15, 1.22), bar=False, rear_glass=(1.2, 1.5, 0.45)),
}
_CAR_MESH = {}


def car_mesh(kind, taxi=False, lights="tail"):
    """Lofted + subdivided body with cut wheel arches, wheels and lamps -> one mesh. Slot 0 is the paint."""
    key = (kind, taxi, lights)
    if key in _CAR_MESH:
        return _CAR_MESH[key]
    sp = CAR_SPEC[kind]
    st = sp["st"]
    paint0 = PAINTS["black"]
    mb = MB(f"CarBody_{kind}")
    mb.mi(paint0), mb.mi(m_carglass), mb.mi(m_cartrim)
    secs = []
    for (y, zt, wt, zb, wb, z0) in st:
        half = [(0.0, zt), (0.55 * wt, zt - 0.004), (wt - 0.045, zt - 0.02), (wt, zt - 0.065), (wb, zb), (wb + 0.018, zb - 0.22),
                (wb, z0 + 0.10), (wb - 0.10, z0), (0.0, z0)]
        ring = [Vector((x, y, z)) for x, z in half] + [Vector((-x, y, z)) for x, z in half[-2:0:-1]]
        secs.append(ring)

    def mfun(i, k):
        if k in (0, 1, 14, 15) and i in sp["top_glass"]:
            return m_carglass
        if k in (3, 12) and i in sp["side_glass"]:
            return m_carglass
        if k in (3, 12) and i in sp["pillar"]:
            return m_cartrim
        if k in (6, 7, 8, 9):
            return m_cartrim
        return paint0

    mb.loft(secs, paint0, mfun=mfun)
    body = mb.finish(col=sc.collection, subsurf=2, sharp=None)
    # wheel arches: boolean pockets
    cutters = []
    for ay in sp["axles"]:
        for sx in (-1, 1):
            cb = MB("cut")
            cb.mi(paint0), cb.mi(m_carglass), cb.mi(m_cartrim)
            cb.cyl(sp["wr"] + 0.055, 0.6, (sx * (sp["track"] + 0.12), ay, sp["wr"] + 0.01), m_cartrim, rot=(0, math.pi / 2, 0), segs=28, base=False)
            co = cb.finish(col=sc.collection)
            co.hide_render = True
            md = body.modifiers.new("arch", "BOOLEAN")
            md.operation = "DIFFERENCE"
            md.solver = "FLOAT"
            md.object = co
            cutters.append(co)
    dg = bpy.context.evaluated_depsgraph_get()
    body_me = bpy.data.meshes.new_from_object(body.evaluated_get(dg), depsgraph=dg)
    # details
    d = MB("det")
    wr, tr = sp["wr"], sp["track"]
    tire = [(wr * 0.60, -0.105), (wr * 0.92, -0.112), (wr * 0.985, -0.09), (wr, -0.05), (wr, 0.05), (wr * 0.985, 0.09), (wr * 0.92, 0.112), (wr * 0.60, 0.105)]
    for ay in sp["axles"]:
        for sx in (-1, 1):
            c = Vector((sx * tr, ay, wr))
            d.lathe(tire, c, m_tire, segs=20, rot=(0, math.pi / 2, 0))
            d.cyl(wr * 0.62, 0.20, c, m_rim, rot=(0, math.pi / 2, 0), segs=16, base=False)
            for sp_i in range(5):
                a = 2 * math.pi * sp_i / 5
                d.box((0.012, wr * 0.52, 0.05), c + Vector((sx * 0.106, 0, 0)) + Vector((0, math.cos(a), math.sin(a))) * wr * 0.3, m_cartrim, rot=(a, 0, 0))
            d.cyl(0.05, 0.02, c + Vector((sx * 0.10, 0, 0)), m_rim, rot=(0, math.pi / 2, 0), segs=8, base=False)
    y0, y1 = st[0][0], st[-1][0]
    tx, tz, tw, th = sp["tail"]
    hx, hz, hw, hh = sp["head"]
    m_t = m_brake if lights == "brake" else m_tail
    m_h = m_head if lights == "head" else m_head_off
    bmb = bmesh.new()
    bmb.from_mesh(body_me)
    bvh = BVHTree.FromBMesh(bmb)

    def patch(xa, xb, za, zb, front, mm, nx=6, nz=2, off=0.007):
        """Lamp / glass / plate that hugs the body: rays cast in from the front or the rear."""
        org_y, dy_ = (y1 + 1.0, -1.0) if front else (y0 - 1.0, 1.0)
        grid = []
        for j in range(nz + 1):
            row = []
            for i in range(nx + 1):
                loc, nrm, _i, _d = bvh.ray_cast(Vector((xa + (xb - xa) * i / nx, org_y, za + (zb - za) * j / nz)), Vector((0, dy_, 0)))
                row.append(None if loc is None else loc + Vector((0, -dy_ * off, 0)))
            grid.append(row)
        for j in range(nz):
            for i in range(nx):
                q = [grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]]
                if all(v is not None for v in q) and max(abs(q[0].y - v.y) for v in q) < 0.25:
                    d.poly(q, mm, smooth=True)

    for sx in (-1, 1):
        xa, xb = sorted((sx * (tx - tw / 2), sx * (tx + tw / 2)))
        patch(xa, xb, tz - th / 2, tz + th / 2, False, m_t, nx=6, nz=3)
        xa, xb = sorted((sx * (hx - hw / 2), sx * (hx + hw / 2)))
        patch(xa, xb, hz - hh / 2, hz + hh / 2, True, m_h, nx=6, nz=2)
        patch(min(sx * (hx - hw / 2), sx * (hx + hw / 2)), max(sx * (hx - hw / 2), sx * (hx + hw / 2)), hz - hh / 2 - 0.05, hz - hh / 2 - 0.02, True, m_amber, nx=6, nz=1)
        mx_, mz_ = sp["mirror"]
        d.sphere(1.0, (sx * (st[len(st) // 2][4] + 0.085), mx_, mz_), paint0, scale=(0.10, 0.055, 0.065), u=8, v=6)
        for hy in (-0.55, 0.35):
            d.box((0.02, 0.13, 0.022), (sx * (st[len(st) // 2][4] + 0.012), hy, st[len(st) // 2][3] - 0.07), m_chrome)
    if sp["bar"]:
        patch(-tx, tx, tz + 0.005, tz + 0.045, False, m_t, nx=24, nz=1)
    if "rear_glass" in sp:
        rw, rz, rh = sp["rear_glass"]
        patch(-rw / 2, rw / 2, rz - rh / 2, rz + rh / 2, False, m_carglass, nx=8, nz=3, off=0.004)
    pz = 0.56 if kind == "sedan" else 0.62
    patch(-0.17, 0.17, pz - 0.085, pz + 0.085, False, m_plate, nx=3, nz=2)
    patch(-0.17, 0.17, 0.40, 0.52, True, m_plate, nx=3, nz=2)
    bmb.free()
    d.box((st[2][4] * 2 - 0.25, 0.10, 0.10), (0, y0 + 0.09, 0.34), m_cartrim)
    d.box((st[-3][4] * 2 - 0.2, 0.10, 0.12), (0, y1 - 0.12, 0.33), m_cartrim)
    d.cyl(0.035, 0.12, (0.45, y0 + 0.06, 0.27), m_chrome, rot=(math.pi / 2, 0, 0), segs=8, base=False)
    if taxi:
        zr = max(s[1] for s in st)
        d.box((0.55, 0.20, 0.17), (0, 0.1, zr + 0.075), lightbox_mat((1.0, 0.75, 0.2), 2.4))
        d.box((0.6, 0.24, 0.03), (0, 0.1, zr - 0.005), m_cartrim)
    det_me = d.mesh(sharp=35)
    bm = bmesh.new()
    bm.from_mesh(body_me)
    nf = len(bm.faces)
    nb = len(body_me.materials)
    bm.from_mesh(det_me)
    bm.faces.ensure_lookup_table()
    for f in bm.faces[nf:]:
        f.material_index += nb
    me = bpy.data.meshes.new(f"Car_{kind}{'_taxi' if taxi else ''}_{lights}")
    bm.to_mesh(me)
    bm.free()
    for mm in list(body_me.materials) + list(det_me.materials):
        me.materials.append(mm)
    stats = (len(body_me.polygons), len(me.polygons))
    for o in cutters + [body]:
        dm = o.data
        bpy.data.objects.remove(o, do_unlink=True)
        bpy.data.meshes.remove(dm)
    bpy.data.meshes.remove(body_me)
    bpy.data.meshes.remove(det_me)
    print(f"[s3] car mesh {key}: body faces {stats[0]}, total {stats[1]}")
    _CAR_MESH[key] = me
    return me


def make_car(name, kind, x, y, heading, paint, taxi=False, lights="tail", col=None):
    """heading degrees: 0 = nose toward +Y (we see the tail), 180 = nose toward the camera, 90 = nose toward -X."""
    ob = inst(car_mesh(kind, taxi, lights), name, (x, y, 0.0), (0, 0, math.radians(heading)), col=col)
    ob.material_slots[0].link = "OBJECT"
    ob.material_slots[0].material = PAINTS[paint]
    return ob

tick("p20_ground")
# ================================================================ GROUND
use(group("Ground"))
g = MB("Road")
g.poly([(-260, -90, 0), (260, -90, 0), (260, 640, 0), (-260, 640, 0)], m_road)
g.finish()

g = MB("Sidewalks")
BLOCKS_Y = [(-90.0, XS0 - 3.0), (XS1 + 3.0, 133.0), (141.0, 252.0)]
for side in (-1, 1):
    for (ya, yb) in BLOCKS_Y:
        g.bx((min(side * RW, side * 200), ya, 0.0), (max(side * RW, side * 200), yb, CURB), m_walk)
        cx0, cx1 = sorted((side * RW, side * (RW + 0.16)))
        g.bx((cx0, ya, 0.0), (cx1, yb, CURB + 0.004), m_curb)                      # kerb stones along our street
        for ye in (ya, yb):                                                        # ... and along the cross streets
            if -80 < ye < 250:
                e0, e1 = sorted((ye, ye + (0.16 if ye == ya else -0.16)))
                g.bx((min(side * RW, side * 200), e0, 0.0), (max(side * RW, side * 200), e1, CURB + 0.004), m_curb)
g.bx((-200, 262, 0.0), (200, 600, 0.22), m_walk)                                    # plaza under the tower
g.finish(bevel=0.012)

m_iron = mat("cast_iron", (0.035, 0.033, 0.03), rough=0.38, metal=0.9)
g = MB("RoadIron")
MANHOLES = [(-1.9, 0.5), (1.9, 5.0), (-2.2, 13.0), (-2.7, 26.6), (0.6, 21.0), (-1.8, 36.5), (2.2, 55.0)]
STEAM_AT = (-2.7, 26.6)
for (mx_, my_) in MANHOLES:
    g.cyl(0.34, 0.006, (mx_, my_, 0.0), m_iron, segs=24)
    g.cyl(0.39, 0.003, (mx_, my_, 0.0), m_conc_d, segs=24)
    for k in range(6):
        a = math.pi * k / 6
        g.box((0.56, 0.022, 0.004), (mx_, my_, 0.007), m_iron, rot=(0, 0, a))
    for rr in (0.12, 0.22):
        g.tube([(mx_ + rr * math.cos(2 * math.pi * i / 16), my_ + rr * math.sin(2 * math.pi * i / 16), 0.007) for i in range(16)], 0.008,
               m_iron, segs=4, closed=True)
for side in (-1, 1):                                                                # drain grates in the gutter
    for gy in (-3.0, 9.5, 21.5, 49.0, 66.0):
        gx = side * (RW - 0.3)
        g.box((0.46, 0.62, 0.004), (gx, gy, 0.002), m_black)
        g.box((0.50, 0.66, 0.006), (gx, gy, 0.0), m_iron)
        for k in range(7):
            g.box((0.40, 0.03, 0.012), (gx, gy - 0.27 + k * 0.09, 0.004), m_iron)
g.finish()

tick("p25_signs")
# ================================================================ SIGN BUILDERS
K_SIGN, K_FOGL, K_SHOP, K_LAMP = 1.0, 1.0, 1.0, 1.0      # global trims for helper lights
SIGNS = group("Signs")
LIGHTS = group("Lights")
SIGN_N = [0]

# little tube drawings, addressed by lower-case letters inside a sign word
GLYPHS.update({
    "b": [[(0, 3.2), (0.4, 1.4), (1.4, 0.3), (2.6, 0.3), (3.6, 1.4), (4, 3.2), (0, 3.2)], [(1.2, 3.9), (0.9, 4.6), (1.4, 5.2), (1.1, 5.9)],
          [(2.7, 3.9), (2.4, 4.6), (2.9, 5.2), (2.6, 5.9)]],                                     # noodle bowl + steam
    "c": [[(0.3, 4.2), (0.6, 0.6), (2.9, 0.6), (3.2, 4.2), (0.3, 4.2)], [(3.15, 3.5), (4.0, 3.3), (4.0, 2.0), (3.0, 1.8)],
          [(1.75, 4.8), (1.45, 5.4), (1.95, 6.0)]],                                               # cup
    "g": [[(0, 6), (4, 6), (2, 3.2), (0, 6)], [(2, 3.2), (2, 0.4)], [(0.9, 0.2), (3.1, 0.2)]],    # cocktail glass
    "a": [[(2, 6), (2, 0.6)], [(0.6, 2.3), (2, 0.5), (3.4, 2.3)]],                                # arrow down
    "s": [[(2, 6), (2.6, 3.9), (4, 3.7), (2.95, 2.4), (3.3, 0.3), (2, 1.4), (0.7, 0.3), (1.05, 2.4), (0, 3.7), (1.4, 3.9), (2, 6)]],
    "n": [[(1.0, 1.2), (1.0, 5.4), (3.6, 6.0), (3.6, 1.8)], [(1.0, 1.2), (0.4, 0.5), (0.0, 1.1), (0.5, 1.7), (1.0, 1.2)],
          [(3.6, 1.8), (3.0, 1.1), (2.6, 1.7), (3.1, 2.3), (3.6, 1.8)]],                          # music note
    "h": [[(2, 0.4), (0.2, 3.0), (0.2, 4.6), (1.1, 5.6), (2, 4.6), (2.9, 5.6), (3.8, 4.6), (3.8, 3.0), (2, 0.4)]],
    "p": [[(2, 0.2), (2, 2.6)], [(2, 2.6), (0.4, 3.4), (0.3, 5.0), (2, 6.0), (3.7, 5.0), (3.6, 3.4), (2, 2.6)]],  # generic "pin / tree"
})


def sign_light(center, normal, key, ps=120.0, pv=700.0, size=1.0, color=None):
    """Helper lights of a sign: one lights surfaces only, one lights the fog only. Neither shows in reflections."""
    c, n = Vector(center), Vector(normal).normalized()
    col = color or NEON[key]
    k = SIGN_N[0]
    if ps > 0:
        light(f"SL{k}", "AREA", c + n * 0.4, ps * K_SIGN, col, size=size, target=c + n * 5 + Vector((0, 0, -2.5)), col=LIGHTS,
              glossy=False, volume=False)
    if pv > 0:
        light(f"SV{k}", "POINT", c + n * 0.6, pv * 1.5 * K_FOGL, col, size=0.4, col=LIGHTS, glossy=False, diffuse=False, volume=True)


def neon_sign(word, center, normal, h=0.8, key="magenta", key2=None, vertical=False, board=True, border=True, wall=0,
              arm=0.6, dead=(), dim=(), light_s=120.0, light_v=700.0, name=None, tube_r=None, col=None, board_mat=None,
              pad=0.32, strength=TUBE, keys=None):
    """Bent-tube neon sign. Lower-case letters in `word` are icons and use key2.
    wall: -1 / +1 = blade sign, its wall is on the sign's local -x / +x side (adds arms, stay wire, transformer)."""
    i = SIGN_N[0]
    SIGN_N[0] += 1
    mb = MB(name or f"Neon_{word.strip()}_{i}")
    r = tube_r or max(0.012, h * 0.032)
    key2 = key2 or key
    gap, bt = 0.24, 0.08
    if vertical:
        W = 4.0 * h / 6 + 2 * pad * h + 0.1 * h
        pitch = 1.28 * h
        H = len(word) * pitch - 0.28 * h + 2 * pad * h
    else:
        tw = word_width(word, h, gap)
        W = tw + 2 * pad * h + 0.2 * h
        H = h + 2 * pad * h + 0.1 * h
    if board:
        mb.box((W, bt, H), (0, 0, 0), board_mat or m_board)
        mb.box((W + 0.05, bt * 0.6, 0.035), (0, 0, H / 2), m_alu)
        mb.box((W + 0.05, bt * 0.6, 0.035), (0, 0, -H / 2), m_alu)
    yt = -(bt / 2 + 0.03 + r)
    cx = 0.0 if vertical else -tw / 2
    for li, ch in enumerate(word):
        gw = GLYPH_W.get(ch, 4.0) * h / 6
        if vertical:
            ox, oz = -gw / 2, H / 2 - pad * h - h - li * pitch
        else:
            ox, oz = cx, -h / 2
            cx += gw + gap * h
        strokes, _ = glyph_strokes(ch, h, (ox, oz))
        kk = keys[li] if keys else (key2 if ch.islower() else key)
        m = m_tube_off if li in dead else (neon(kk, round(strength * 0.2, 2)) if li in dim else neon(kk, strength))
        for pts, closed in strokes:
            mb.tube([(p.x, yt, p.y) for p in pts], r, m, segs=6, closed=closed)
            if board and not closed:
                for pe in (pts[0], pts[-1]):          # glass tube turns back into the board at its ends
                    mb.rod((pe.x, yt, pe.y), (pe.x, -bt / 2, pe.y), r * 0.8, m_black, segs=5)
    if border:
        e = 0.06 * h + r
        bw, bh = W / 2 - e, H / 2 - e
        rect = fillet([Vector((-bw, -bh, 0)), Vector((bw, -bh, 0)), Vector((bw, bh, 0)), Vector((-bw, bh, 0))], 0.14 * h, 3, closed=True)
        mb.tube([(p.x, yt, p.y) for p in rect], r * 0.85, neon(key2, strength), segs=6, closed=True)
    if wall:
        zs = (H / 2 - 0.12, -H / 2 + 0.12) if H > 0.9 else (0.0,)
        for z in zs:
            mb.box((arm + 0.06, 0.05, 0.05), (wall * (W / 2 + arm / 2), 0.0, z), m_alu)
        mb.box((0.03, 0.22, H * 0.9), (wall * (W / 2 + arm), 0, 0), m_alu)
        mb.rod((wall * (W / 2 - 0.1), 0.0, H / 2), (wall * (W / 2 + arm), 0.0, H / 2 + 0.55), 0.01, m_steel, segs=5)
        mb.box((0.26, 0.15, 0.2), (wall * (W / 2 + 0.22), 0.03, -H / 2 - 0.02), m_alu)
        mb.tube([(wall * (W / 2 + 0.3), 0.03, -H / 2 - 0.1), (wall * (W / 2 + 0.45), 0.03, -H / 2 - 0.3), (wall * (W / 2 + arm), 0.03, -H / 2 - 0.25)],
                0.012, m_black, segs=5)
    M = basis(Vector((0, 0, 1)).cross(Vector(normal)), normal, center)
    ob = mb.finish(col=col or SIGNS, matrix=M, sharp=40)
    sign_light(center, normal, key, light_s, light_v, size=max(W, H) * 0.6)
    return ob, (W, H)


def lightbox(word, center, normal, h=0.45, face=(0.9, 0.9, 0.85), ink=None, vertical=False, W=None, H=None, strength=1.5,
             wall=0, arm=0.45, name=None, light_s=70.0, light_v=250.0, key="white", stripes=(), depth=0.18, col=None, lcolor=None):
    """Backlit box sign with (built-in font) lettering. stripes: [(z offset, height, colour)] emissive bands."""
    i = SIGN_N[0]
    SIGN_N[0] += 1
    name = name or f"Lightbox_{word.strip().replace(' ', '_')}_{i}"
    mb = MB(name)
    if vertical:
        W = W or h * 1.6
        H = H or len(word) * h * 0.98 + 0.5 * h
    else:
        W = W or 0.72 * h * len(word) + 0.9 * h
        H = H or h * 1.75
    mb.box((W, depth, H), (0, 0, 0), m_alu)
    mb.box((W - 0.07, 0.012, H - 0.07), (0, -depth / 2 - 0.003, 0), lightbox_mat(face, strength))
    for (zo, zh, c) in stripes:
        mb.box((W - 0.07, 0.008, zh), (0, -depth / 2 - 0.011, zo), lightbox_mat(c, strength))
    if wall:
        for z in ((H / 2 - 0.1, -H / 2 + 0.1) if H > 0.8 else (0.0,)):
            mb.box((arm + 0.06, 0.05, 0.05), (wall * (W / 2 + arm / 2), 0.0, z), m_alu)
        mb.box((0.03, 0.2, H * 0.85), (wall * (W / 2 + arm), 0, 0), m_alu)
    M = basis(Vector((0, 0, 1)).cross(Vector(normal)), normal, center)
    ob = mb.finish(col=col or SIGNS, matrix=M, sharp=40)
    if word.strip():
        tm = M @ Matrix.Translation((0, -depth / 2 - 0.018, 0)) @ Matrix.Rotation(math.pi / 2, 4, "X")
        text(name + "_txt", "\n".join(word) if vertical else word, None, ink or m_black, size=h, extrude=0.003, matrix=tm,
             line=0.9 if vertical else 1.0, col=col or SIGNS)
    sign_light(center, normal, key, light_s, light_v, size=max(W, H) * 0.6, color=lcolor or face)
    return ob, (W, H)

tick("p30_near")
# ================================================================ NEAR BUILDINGS (first block): built facades
NEAR = group("Buildings_Near")
SHOPS = group("Shops")
m_sill = mat("sill", (0.17, 0.17, 0.165), rough=0.42)
m_acbody = mat("ac_body", (0.42, 0.42, 0.40), rough=0.4)
m_pipe = mat("pipe_grey", (0.22, 0.22, 0.21), rough=0.4)
m_balpanel = mat("balcony_panel", (0.30, 0.31, 0.32), rough=0.25, coat=0.3)
m_grate = mat("grating", (0.02, 0.02, 0.022), rough=0.5, metal=0.8)


def wall_holes(mb, side, u0, u1, z0, z1, holes, m, d=0.0, reveal=0.22, m_rev=None):
    """Wall sheet in the facade plane with rectangular holes (hu0, hz0, hu1, hz1); reveals run back into the wall."""
    rd = lambda v: round(v, 4)
    us = sorted({rd(u0), rd(u1)} | {rd(v) for h in holes for v in (h[0], h[2]) if u0 < v < u1})
    zs = sorted({rd(z0), rd(z1)} | {rd(v) for h in holes for v in (h[1], h[3]) if z0 < v < z1})
    for j in range(len(zs) - 1):
        zc = (zs[j] + zs[j + 1]) / 2
        run = None
        for i in range(len(us) - 1):
            uc = (us[i] + us[i + 1]) / 2
            solid = not any(h[0] < uc < h[2] and h[1] < zc < h[3] for h in holes)
            if solid and run is None:
                run = us[i]
            if run is not None and (not solid or i == len(us) - 2):
                end = us[i + 1] if solid else us[i]
                fquad(mb, side, [(run, d, zs[j]), (end, d, zs[j]), (end, d, zs[j + 1]), (run, d, zs[j + 1])], m)
                run = None
    if reveal:
        mr = m_rev or m
        for (a, b, c, e) in holes:
            a_, c_, b_, e_ = max(a, u0), min(c, u1), max(b, z0), min(e, z1)
            d1 = d + reveal
            mb.poly([lw(side, a_, d, b_), lw(side, c_, d, b_), lw(side, c_, d1, b_), lw(side, a_, d1, b_)], mr)
            mb.poly([lw(side, a_, d, e_), lw(side, c_, d, e_), lw(side, c_, d1, e_), lw(side, a_, d1, e_)], mr)
            mb.poly([lw(side, a_, d, b_), lw(side, a_, d1, b_), lw(side, a_, d1, e_), lw(side, a_, d, e_)], mr)
            mb.poly([lw(side, c_, d, b_), lw(side, c_, d1, b_), lw(side, c_, d1, e_), lw(side, c_, d, e_)], mr)


def window(mb, side, u0, z0, ww, wh, bu0, bu1, zf, zc, rng, frame=None, glass=None, lit_p=0.5, sill=True, mullion=True,
           transom=False, depth=3.0, cpick=None):
    """Frame + glass + a lit or dark room box behind it, sometimes blinds or curtains."""
    frame, glass = frame or m_alu, glass or m_glass
    r_col = cpick(rng) if cpick else rng.random()
    r2 = rng.random()
    lit = rng.choice((0.3, 0.5, 0.75, 1.0, 1.0)) if rng.random() < lit_p else 0.0
    fd0, fd1, ft = 0.11, 0.18, 0.05
    u1, z1 = u0 + ww, z0 + wh
    for (a, b, c, e) in ((u0, u1, z0, z0 + ft), (u0, u1, z1 - ft, z1), (u0, u0 + ft, z0, z1), (u1 - ft, u1, z0, z1)):
        fbox(mb, side, a, b, fd0, fd1, c, e, frame)
    if mullion:
        um = u0 + ww * (0.5 if r2 < 0.7 else 0.34)
        fbox(mb, side, um - ft / 2, um + ft / 2, fd0 + 0.01, fd1 - 0.01, z0, z1, frame)
    if transom:
        zt = z0 + wh * 0.72
        fbox(mb, side, u0, u1, fd0 + 0.01, fd1 - 0.01, zt - ft / 2, zt + ft / 2, frame)
    fquad(mb, side, [(u0, 0.145, z0), (u1, 0.145, z0), (u1, 0.145, z1), (u0, 0.145, z1)], glass)
    if sill:
        fbox(mb, side, u0 - 0.07, u1 + 0.07, -0.08, 0.10, z0 - 0.07, z0, m_sill)
    e = 0.02
    a, b, f0, f1, d0, d1 = bu0 + e, bu1 - e, zf + e, zc - e, 0.20, 0.20 + depth
    c = lambda fac: (r_col, lit, fac, r2)
    mb.poly([lw(side, a, d0, f0), lw(side, b, d0, f0), lw(side, b, d1, f0), lw(side, a, d1, f0)], m_room, col=c(0.2))
    mb.poly([lw(side, a, d0, f1), lw(side, b, d0, f1), lw(side, b, d1, f1), lw(side, a, d1, f1)], m_room, col=c(1.0))
    mb.poly([lw(side, a, d1, f0), lw(side, b, d1, f0), lw(side, b, d1, f1), lw(side, a, d1, f1)], m_room, col=[c(0.3), c(0.3), c(0.7), c(0.7)])
    for u in (a, b):
        mb.poly([lw(side, u, d0, f0), lw(side, u, d1, f0), lw(side, u, d1, f1), lw(side, u, d0, f1)], m_room, col=[c(0.28), c(0.28), c(0.62), c(0.62)])
    t = rng.random()
    if t < 0.32:
        zb = z1 - wh * rng.uniform(0.3, 1.0)
        mb.poly([lw(side, u0, 0.2, zb), lw(side, u1, 0.2, zb), lw(side, u1, 0.2, z1), lw(side, u0, 0.2, z1)], m_blind, col=(r_col, lit, 1.0, 0.0))
    elif t < 0.44:
        mb.poly([lw(side, u0, 0.2, z0), lw(side, u1, 0.2, z0), lw(side, u1, 0.2, z1), lw(side, u0, 0.2, z1)], m_blind, col=(r_col, lit, 1.0, 1.0))
    return lit, r_col


def ac_unit(mb, side, u, d, z):
    """Outdoor AC unit, its fan grille facing the street; d = wall-side depth of the unit."""
    fbox(mb, side, u - 0.4, u + 0.4, d - 0.3, d, z, z + 0.55, m_acbody)
    c = lw(side, u + 0.1, d - 0.305, z + 0.275)
    mb.cyl(0.21, 0.012, c, m_black, rot=(0, math.pi / 2, 0), segs=14, base=False)
    mb.cyl(0.05, 0.02, c, m_acbody, rot=(0, math.pi / 2, 0), segs=8, base=False)
    for k in (-1, 1):
        fbox(mb, side, u + k * 0.3 - 0.02, u + k * 0.3 + 0.02, d - 0.3, d + 0.02, z - 0.05, z, m_alu)
    mb.rod(lw(side, u - 0.35, d - 0.1, z + 0.4), lw(side, u - 0.35, d + 0.02, z + 0.9), 0.02, m_white, segs=5)


def balcony(mb, side, u0, u1, zf, style="panel", depth=1.1, ac=False, rng=None):
    fbox(mb, side, u0, u1, -depth, 0.0, zf - 0.14, zf, m_conc)
    if style == "panel":
        fbox(mb, side, u0 + 0.03, u1 - 0.03, -depth, -depth + 0.05, zf + 0.10, zf + 1.02, m_balpanel)
        fbox(mb, side, u0, u1, -depth - 0.02, -depth + 0.07, zf + 1.02, zf + 1.08, m_alu_l)
        for ue in (u0, u1 - 0.05):
            fbox(mb, side, ue, ue + 0.05, -depth, 0.0, zf, zf + 1.08, m_balpanel)
    else:
        mb.rod(lw(side, u0, -depth, zf + 1.05), lw(side, u1, -depth, zf + 1.05), 0.022, m_alu, segs=6)
        mb.rod(lw(side, u0, -depth, zf + 0.12), lw(side, u1, -depth, zf + 0.12), 0.015, m_alu, segs=6)
        n = max(2, int((u1 - u0) / 0.13))
        for i in range(n + 1):
            u = u0 + (u1 - u0) * i / n
            mb.rod(lw(side, u, -depth, zf + 0.12), lw(side, u, -depth, zf + 1.05), 0.008, m_alu, segs=4)
        for ue in (u0, u1):
            mb.rod(lw(side, ue, -depth, zf + 1.05), lw(side, ue, 0.0, zf + 1.05), 0.022, m_alu, segs=6)
            mb.rod(lw(side, ue, -depth, zf), lw(side, ue, -depth, zf + 1.05), 0.022, m_alu, segs=6)
    if ac:
        ac_unit(mb, side, u0 + 0.6, -0.12, zf)
    if rng and rng.random() < 0.4:
        up = rng.uniform(u0 + 0.4, u1 - 0.4)
        mb.cyl(0.13, 0.22, lw(side, up, -depth + 0.3, zf), m_rust, segs=8, r2=0.16)
        mb.sphere(0.2, lw(side, up, -depth + 0.3, zf + 0.38), m_plant, u=8, v=6, scale=(1, 1, 1.2))


def drainpipe(mb, side, u, z0, z1, d=-0.07, r=0.05):
    mb.cyl(r, z1 - z0, lw(side, u, d, z0), m_pipe, segs=8)
    z = z0 + 1.2
    while z < z1:
        fbox(mb, side, u - 0.08, u + 0.08, d - 0.06, 0.0, z, z + 0.04, m_alu)
        z += 2.6


def fire_escape(mb, side, ua, ub, zfs, depth=1.05):
    """Steel landings, railings and zig-zag stairs on the facade."""
    for k, zf in enumerate(zfs):
        fbox(mb, side, ua, ub, -depth, 0.0, zf - 0.05, zf, m_grate)
        for ue in (ua, ub):
            fbox(mb, side, ue - 0.03, ue + 0.03, -depth, 0.0, zf - 0.14, zf - 0.05, m_grate)
        for h in (0.5, 1.0):
            mb.rod(lw(side, ua, -depth, zf + h), lw(side, ub, -depth, zf + h), 0.018, m_grate, segs=5)
            for ue in (ua, ub):
                mb.rod(lw(side, ue, -depth, zf + h), lw(side, ue, 0.0, zf + h), 0.018, m_grate, segs=5)
        n = int((ub - ua) / 0.55)
        for i in range(n + 1):
            u = ua + (ub - ua) * i / n
            mb.rod(lw(side, u, -depth, zf), lw(side, u, -depth, zf + 1.0), 0.012, m_grate, segs=4)
        if k + 1 < len(zfs):
            zn = zfs[k + 1]
            s0, s1 = (ua + 0.5, ub - 0.6) if k % 2 == 0 else (ub - 0.5, ua + 0.6)
            for dd in (-depth + 0.12, -depth + 0.6):
                mb.rod(lw(side, s0, dd, zf), lw(side, s1, dd, zn), 0.03, m_grate, segs=5)
            steps = 11
            for i in range(1, steps):
                t = i / steps
                fbox(mb, side, s0 + (s1 - s0) * t - 0.11, s0 + (s1 - s0) * t + 0.11, -depth + 0.12, -depth + 0.6, zf + (zn - zf) * t - 0.012,
                     zf + (zn - zf) * t + 0.012, m_grate)
    z0 = zfs[0]
    for dd in (-depth + 0.15, -depth + 0.55):                         # drop ladder
        mb.rod(lw(side, ua + 0.3, dd, z0 - 2.3), lw(side, ua + 0.3, dd, z0 + 0.9), 0.015, m_grate, segs=4)
    for i in range(8):
        mb.rod(lw(side, ua + 0.3, -depth + 0.15, z0 - 2.2 + i * 0.3), lw(side, ua + 0.3, -depth + 0.55, z0 - 2.2 + i * 0.3), 0.011, m_grate, segs=4)


def near_building(name, side, y0, y1, floors, fh=3.2, wall=None, wall_g=None, bays=None, win=(1.5, 1.6, 0.9), gh=GH,
                  depth=15.0, balc=None, balc_style="panel", lit_p=0.5, openings=(), frame=None, ledge=True, seed=0,
                  glass=None, mull=True, transom=False, cpick=None, pipes=True, acs=0.25):
    """openings: ground-floor holes (u_a, u_b, z_top). balc(f, i) -> bool picks balcony bays."""
    rng = random.Random(seed)
    wall = wall or m_conc
    W, H = y1 - y0, gh + floors * fh
    mb = MB(name, col=True)
    fbox(mb, side, y0, y1, 3.25, depth, gh, H, wall)                               # core above the shops
    fbox(mb, side, y0, y1, depth - 0.3, depth, 0.0, gh, wall)                      # rear wall at street level
    for ue in (y0, y1 - 0.25):
        fbox(mb, side, ue, ue + 0.25, 0.0, 3.25, gh, H, wall)                      # party walls
        fbox(mb, side, ue, ue + 0.25, 0.0, depth, 0.0, gh, wall)
    fbox(mb, side, y0, y1, 0.0, 3.25, H - 0.2, H, wall)
    fbox(mb, side, y0, y1, 0.0, 3.25, gh - 0.25, gh, wall)
    wall_holes(mb, side, y0, y1, 0.0, gh, [(a, CURB - 0.2, b, zt) for (a, b, zt) in openings], wall_g or wall, reveal=0.30)
    bays = bays or max(2, round(W / 2.6))
    bw = W / bays
    ww, wh, sh = win
    holes, cells = [], []
    for f in range(floors):
        zf = gh + f * fh
        for i in range(bays):
            uc = y0 + (i + 0.5) * bw
            isb = bool(balc and balc(f, i))
            holes.append((uc - ww / 2, zf + 0.08, uc + ww / 2, zf + 2.2) if isb else (uc - ww / 2, zf + sh, uc + ww / 2, zf + sh + wh))
            cells.append((f, i, isb))
    wall_holes(mb, side, y0, y1, gh, H, holes, wall, reveal=0.22)
    for h, (f, i, isb) in zip(holes, cells):
        zf = gh + f * fh
        window(mb, side, h[0], h[1], h[2] - h[0], h[3] - h[1], max(y0 + 0.27, y0 + i * bw), min(y1 - 0.27, y0 + (i + 1) * bw), zf, zf + fh - 0.3,
               rng, frame=frame, glass=glass, lit_p=lit_p, sill=not isb, mullion=mull, transom=transom, cpick=cpick)
        if isb:
            balcony(mb, side, y0 + i * bw + 0.08, y0 + (i + 1) * bw - 0.08, zf, style=balc_style, ac=rng.random() < 0.6, rng=rng)
        elif rng.random() < acs and f > 0:
            ac_unit(mb, side, h[0] + 0.45, -0.02, h[1] - 0.75)
    if ledge:
        for f in range(floors + 1):
            zf = gh + f * fh
            fbox(mb, side, y0, y1, -0.09, 0.0, zf - 0.13, zf + 0.05, m_sill)
    fbox(mb, side, y0, y1, -0.2, 0.35, H, H + 0.95, wall)                          # parapet
    fbox(mb, side, y0, y1, -0.26, 0.4, H + 0.95, H + 1.05, m_sill)
    if pipes:
        drainpipe(mb, side, y0 + 0.2, CURB, H)
        if rng.random() < 0.7:
            drainpipe(mb, side, y1 - 0.2, CURB, H, r=0.04)
    ob = mb.finish(col=NEAR, sharp=40)
    return ob, {"H": H, "bw": bw, "bays": bays, "y0": y0, "y1": y1, "side": side, "gh": gh, "fh": fh}


# ---------------------------------------------------------------- shop pieces (facade-local coordinates)
def shop_shell(mb, side, ua, ub, zt, depth, zc, m_floor, m_wallm, m_ceil):
    zf, a, b, d0 = CURB + 0.02, ua - 0.26, ub + 0.26, 0.30
    mb.poly([lw(side, a, d0, zf), lw(side, b, d0, zf), lw(side, b, depth, zf), lw(side, a, depth, zf)], m_floor)
    mb.poly([lw(side, a, d0, zc), lw(side, b, d0, zc), lw(side, b, depth, zc), lw(side, a, depth, zc)], m_ceil)
    mb.poly([lw(side, a, depth, zf), lw(side, b, depth, zf), lw(side, b, depth, zc), lw(side, a, depth, zc)], m_wallm)
    for u in (a, b):
        mb.poly([lw(side, u, d0, zf), lw(side, u, depth, zf), lw(side, u, depth, zc), lw(side, u, d0, zc)], m_wallm)
    wall_holes(mb, side, a, b, zf, zc, [(ua, zf - 0.3, ub, zt)], m_wallm, d=d0, reveal=0)
    return zf


def shop_glazing(mb, side, ua, ub, zt, doors=(), sill=0.4, mull=1.6, glass=None, frame=None, riser=None, d=0.12,
                 door_h=2.2, open_door=False):
    glass, frame, riser = glass or m_glass, frame or m_alu, riser or m_conc_d
    zb = CURB
    fbox(mb, side, ua, ub, d - 0.035, d + 0.035, zt - 0.07, zt, frame)
    segs, cur = [], ua
    for (da, db) in sorted(doors):
        if da > cur + 0.05:
            segs.append((cur, da))
        cur = db
    if ub > cur + 0.05:
        segs.append((cur, ub))
    for (sa, sb) in segs:
        fbox(mb, side, sa, sb, d - 0.07, d + 0.07, zb, zb + sill, riser)
        fbox(mb, side, sa, sb, d - 0.035, d + 0.035, zb + sill, zb + sill + 0.05, frame)
        n = max(1, round((sb - sa) / mull))
        for i in range(n + 1):
            um = sa + (sb - sa) * i / n
            fbox(mb, side, um - 0.025, um + 0.025, d - 0.035, d + 0.035, zb + sill, zt, frame)
        fquad(mb, side, [(sa, d, zb + sill), (sb, d, zb + sill), (sb, d, zt), (sa, d, zt)], glass)
    for (da, db) in doors:
        for ue in (da, db):
            fbox(mb, side, ue - 0.035, ue + 0.035, d - 0.045, d + 0.045, zb, zt, frame)
        fbox(mb, side, da, db, d - 0.035, d + 0.035, zb + door_h, zb + door_h + 0.06, frame)
        fquad(mb, side, [(da, d, zb + door_h + 0.06), (db, d, zb + door_h + 0.06), (db, d, zt - 0.07), (da, d, zt - 0.07)], glass)
        if open_door:
            continue
        leaves = 2 if db - da > 1.25 else 1
        for li in range(leaves):
            la, lb = da + (db - da) * li / leaves, da + (db - da) * (li + 1) / leaves
            dd = d + 0.03
            fquad(mb, side, [(la + 0.04, dd, zb + 0.03), (lb - 0.04, dd, zb + 0.03), (lb - 0.04, dd, zb + door_h), (la + 0.04, dd, zb + door_h)], glass)
            fbox(mb, side, la + 0.04, lb - 0.04, dd - 0.02, dd + 0.02, zb + 0.03, zb + 0.13, frame)
            fbox(mb, side, la + 0.04, lb - 0.04, dd - 0.02, dd + 0.02, zb + door_h - 0.06, zb + door_h, frame)
            for ue in (la + 0.04, lb - 0.08):
                fbox(mb, side, ue, ue + 0.04, dd - 0.02, dd + 0.02, zb + 0.03, zb + door_h, frame)
            uh = (lb - 0.14) if (li == 0 and leaves == 2) or leaves == 1 else (la + 0.14)
            mb.rod(lw(side, uh, dd - 0.06, zb + 0.85), lw(side, uh, dd - 0.06, zb + 1.45), 0.013, m_chrome, segs=6)


def p_stool(mb, side, u, d, zf, h=0.66, seat=None):
    mb.cyl(0.17, 0.02, lw(side, u, d, zf), m_chrome, segs=12)
    mb.cyl(0.024, h - 0.07, lw(side, u, d, zf + 0.02), m_chrome, segs=6)
    mb.cyl(0.165, 0.06, lw(side, u, d, zf + h - 0.06), seat or m_redv, segs=12)


def p_pendant(mb, side, u, d, zc, drop, key="amber", strength=9.0, r=0.14):
    p = lw(side, u, d, zc - drop)
    mb.rod(lw(side, u, d, zc), p + Vector((0, 0, 0.1)), 0.006, m_black, segs=4)
    mb.cyl(r, 0.13, p, m_black, segs=12, r2=0.04, caps=False)
    mb.sphere(0.05, p + Vector((0, 0, 0.01)), neon(key, strength), u=8, v=6)


def p_products(mb, side, u0, u1, d0, d1, z, rng, hmax=0.26, axis="u"):
    """A row of boxes / cans on a shelf."""
    if axis == "u":
        u = u0
        while u < u1 - 0.07:
            w = rng.uniform(0.07, 0.17)
            hh = rng.uniform(0.5, 1.0) * hmax
            fbox(mb, side, u, u + w, d0, d1, z, z + hh, rng.choice(PRODUCT))
            u += w + 0.012
    else:
        d = d0
        while d < d1 - 0.07:
            w = rng.uniform(0.07, 0.17)
            hh = rng.uniform(0.5, 1.0) * hmax
            fbox(mb, side, u0, u1, d, d + w, z, z + hh, rng.choice(PRODUCT))
            d += w + 0.012


def p_gondola(mb, side, uc, d0, d1, zf, rng, w=0.9, h=1.45, rows=4):
    """Shop shelving unit running back into the shop, goods on both faces."""
    fbox(mb, side, uc - w / 2, uc + w / 2, d0, d1, zf, zf + 0.12, m_white)
    fbox(mb, side, uc - 0.02, uc + 0.02, d0, d1, zf, zf + h, m_white)
    for r in range(rows):
        z = zf + 0.12 + r * (h - 0.15) / rows
        fbox(mb, side, uc - w / 2, uc + w / 2, d0, d1, z, z + 0.02, m_white)
        for sg in (-1, 1):
            a, b = sorted((uc + sg * 0.06, uc + sg * (w / 2 - 0.03)))
            p_products(mb, side, a, b, d0 + 0.03, d1 - 0.03, z + 0.02, rng, hmax=(h - 0.15) / rows - 0.06, axis="d")
    fbox(mb, side, uc - w / 2, uc + w / 2, d0 - 0.03, d0, zf, zf + h, m_white)                    # end panel
    p_products(mb, side, uc - w / 2 + 0.05, uc + w / 2 - 0.05, d0 - 0.16, d0 - 0.04, zf + 0.5, rng, hmax=0.3)


def p_wallshelf(mb, side, u0, u1, d_back, zf, h, rows, rng, depth=0.3, m=None, hmax=None):
    m = m or m_woodl
    fbox(mb, side, u0, u1, d_back - 0.02, d_back, zf, zf + h, m)
    for r in range(rows + 1):
        z = zf + 0.05 + r * (h - 0.1) / rows
        fbox(mb, side, u0, u1, d_back - depth, d_back, z, z + 0.025, m)
        if r < rows:
            p_products(mb, side, u0 + 0.03, u1 - 0.03, d_back - depth + 0.04, d_back - 0.05, z + 0.025, rng, hmax=hmax or (h - 0.1) / rows - 0.07)


def p_plant(mb, pos, h=0.9, pot=None):
    p = Vector(pos)
    mb.cyl(0.17, 0.32, p, pot or m_conc_d, segs=10, r2=0.21)
    for k in range(5):
        a = k * 1.3
        mb.sphere(0.2, p + Vector((0.1 * math.cos(a), 0.1 * math.sin(a), 0.42 + (h - 0.6) * k / 4)), m_plant, u=8, v=6, scale=(1, 1, 1.25))


def p_lantern(mb, pos, r=0.17, key="red", strength=1.6, cord=0.25):
    p = Vector(pos)
    mb.sphere(1.0, p, lantern_mat(key, strength), scale=(r, r, r * 1.25), u=14, v=10)
    for dz in (-1, 1):
        mb.cyl(r * 0.5, 0.035, p + Vector((0, 0, dz * r * 1.2 - 0.0175)), m_black, segs=10)
    mb.rod(p + Vector((0, 0, r * 1.2)), p + Vector((0, 0, r * 1.2 + cord)), 0.005, m_black, segs=4)


def p_aframe(mb, pos, heading=0.0, face=None):
    """Sidewalk sandwich board."""
    M0 = mb.M.copy()
    mb.M = M0 @ Matrix.Translation(Vector(pos)) @ Matrix.Rotation(heading, 4, "Z")
    for sg in (-1, 1):
        mb.box((0.55, 0.03, 0.95), (0, sg * 0.16, 0.46), m_wood, rot=(sg * 0.33, 0, 0))
        mb.box((0.45, 0.012, 0.6), (0, sg * 0.185, 0.52), face or m_black, rot=(sg * 0.33, 0, 0))
    mb.M = M0


def p_seated(mb, side, u, d, zseat, rng, **kw):
    """Customer on a stool, facing into the shop."""
    M0 = mb.M.copy()
    mb.M = M0 @ Matrix.Translation(lw(side, u, d, zseat)) @ Matrix.Rotation(math.radians(-90 * side), 4, "Z")
    person_geo(mb, rng, pose="sit", **kw)
    mb.M = M0


def p_standing(mb, side, u, d, z, rng, facing_street=True, **kw):
    M0 = mb.M.copy()
    ang = 90 * side if facing_street else -90 * side
    mb.M = M0 @ Matrix.Translation(lw(side, u, d, z)) @ Matrix.Rotation(math.radians(ang), 4, "Z")
    person_geo(mb, rng, pose="stand", **kw)
    mb.M = M0


def shop_light(name, side, u, d, z, power, color, size=1.5, size_y=None):
    """Soft ceiling light inside a shop (does not show up as a rectangle in reflections)."""
    return light(name, "AREA", lw(side, u, d, z), power * K_SHOP, color, size=size, size_y=size_y, rot=(0, 0, 0), col=LIGHTS, glossy=False)


def awning(mb, side, ua, ub, z, out=1.5, drop=0.45, m=None, m2=None, valance=0.22, stripes=0):
    """Sloped fabric awning with a hanging valance and steel arms."""
    m = m or m_indigo
    n = stripes or 1
    for i in range(n):
        a, b = ua + (ub - ua) * i / n, ua + (ub - ua) * (i + 1) / n
        mm = m if (i % 2 == 0 or not m2) else m2
        mb.poly([lw(side, a, 0.0, z), lw(side, b, 0.0, z), lw(side, b, -out, z - drop), lw(side, a, -out, z - drop)], mm)
        mb.poly([lw(side, a, -out, z - drop), lw(side, b, -out, z - drop), lw(side, b, -out, z - drop - valance), lw(side, a, -out, z - drop - valance)], mm)
    for ue in (ua + 0.05, ub - 0.05):
        mb.rod(lw(side, ue, 0.0, z - 0.02), lw(side, ue, -out, z - drop - 0.02), 0.015, m_alu, segs=5)
        mb.rod(lw(side, ue, 0.0, z - drop - 0.5), lw(side, ue, -out * 0.8, z - drop * 0.8 - 0.05), 0.012, m_alu, segs=5)
    mb.rod(lw(side, ua, -out, z - drop - 0.01), lw(side, ub, -out, z - drop - 0.01), 0.015, m_alu, segs=5)

tick("p35_block0")
# ================================================================ FIRST BLOCK: six built facades and their shops
def pick_warm(r):
    return r.uniform(0.0, 0.6)


def pick_party(r):
    return r.choice((0.84, 0.9, 0.92, 0.97, 0.7, 0.2))


def pick_office(r):
    return r.uniform(0.62, 0.79) if r.random() < 0.8 else 0.5


def pick_hotel(r):
    return r.uniform(0.0, 0.58) if r.random() < 0.92 else 0.82


B = {}
B["L0"] = near_building("Bld_L0_shutters", -1, -9.5, 2.0, 9, fh=3.1, wall=m_stucco, wall_g=m_conc_d, bays=4, win=(1.4, 1.5, 0.95),
                        lit_p=0.24, openings=[(-8.9, -5.2, 3.0), (-4.3, 1.3, 3.0)], seed=29, cpick=lambda r: r.uniform(0.0, 0.75), acs=0.3)[1]
B["L1"] = near_building("Bld_L1_ramen", -1, 2.0, 12.5, 9, fh=3.1, wall=m_tile_b, wall_g=m_conc_d, bays=4, win=(1.5, 1.45, 0.95),
                        balc=lambda f, i: i in (1, 2) and f >= 1, lit_p=0.5, openings=[(2.5, 9.3, 3.25), (10.0, 11.7, 2.75)], seed=3,
                        cpick=lambda r: 0.9 if r.random() < 0.12 else r.uniform(0.0, 0.7))[1]
B["L2"] = near_building("Bld_L2_arcade", -1, 12.5, 20.5, 8, fh=3.3, wall=m_panel, bays=3, win=(2.1, 1.9, 0.75), lit_p=0.72,
                        openings=[(12.9, 20.1, 3.4)], seed=5, transom=True, cpick=pick_party, acs=0.1)[1]
B["L3"] = near_building("Bld_L3_cafe", -1, 20.5, 29.0, 3, fh=3.3, wall=m_brick, wall_g=m_tile_g, bays=3, win=(1.6, 1.7, 0.85),
                        lit_p=0.6, openings=[(20.9, 28.6, 3.3)], seed=7, transom=True, cpick=pick_warm)[1]
B["R1"] = near_building("Bld_R1_mart", 1, 0.5, 11.0, 10, fh=3.2, wall=m_conc, wall_g=m_tile_w, bays=4, win=(1.45, 1.55, 0.9),
                        lit_p=0.45, openings=[(0.9, 7.6, 3.3)], seed=11, acs=0.35)[1]
B["R2"] = near_building("Bld_R2_laundry", 1, 11.0, 19.5, 6, fh=3.1, wall=m_tile_w, wall_g=m_stucco, bays=3, win=(1.7, 1.45, 0.95),
                        balc=lambda f, i: f >= 2, balc_style="bars", lit_p=0.55, openings=[(11.4, 15.6, 3.2), (16.0, 19.1, 3.0)], seed=13,
                        cpick=lambda r: r.uniform(0.0, 0.78))[1]
B["R3"] = near_building("Bld_R3_hotel", 1, 19.5, 29.0, 12, fh=3.1, wall=m_conc_w, wall_g=m_panel, bays=4, win=(1.35, 1.7, 0.8),
                        lit_p=0.66, openings=[(20.2, 25.2, 3.6), (25.9, 28.6, 3.2)], seed=17, cpick=pick_hotel, acs=0.0)[1]

# fire escape on the near right building (floors 1..9), exhaust duct on the ramen building
fx = MB("FireEscape_R1")
fire_escape(fx, 1, 3.9, 7.3, [GH + 3.2 * f + 0.05 for f in range(1, 10)])
fx.finish(col=NEAR, sharp=45)
dx = MB("Duct_L1")
fbox(dx, -1, 12.0, 12.4, -0.4, 0.0, 2.9, 32.5, m_steel)
fbox(dx, -1, 11.95, 12.45, -0.45, 0.0, 12.0, 12.08, m_alu)
fbox(dx, -1, 11.95, 12.45, -0.45, 0.0, 22.0, 22.08, m_alu)
fbox(dx, -1, 11.95, 12.45, -0.45, 0.0, 6.4, 6.48, m_alu)
dx.finish(col=NEAR)


# ---------------------------------------------------------------- RAMEN (left, nearest)
def shop_ramen(side=-1, ua=2.5, ub=9.3, zt=3.25):
    rng = random.Random(21)
    mb = MB("Shop_Ramen")
    zc = 3.15
    zf = shop_shell(mb, side, ua, ub, zt, 5.2, zc, m_conc_d, m_cream, m_wood)
    shop_glazing(mb, side, ua, ub, zt, doors=[(7.75, 9.0)], sill=0.85, mull=1.35, glass=m_glass_fog, frame=m_wood, riser=m_wood, door_h=2.1)
    # counter with stools and customers, kitchen behind
    fbox(mb, side, 2.9, 8.5, 1.9, 2.45, zf, zf + 1.0, m_woodl)
    fbox(mb, side, 2.8, 8.6, 1.8, 2.5, zf + 1.0, zf + 1.05, m_wood)
    for k, u in enumerate((3.3, 4.05, 4.8, 5.55, 6.3, 7.05)):
        p_stool(mb, side, u, 1.42, zf)
        if k in (0, 2, 3, 5):
            p_seated(mb, side, u, 1.42, zf + 0.66, rng, hairstyle=rng.choice(("short", "bob", "short")))
    fbox(mb, side, 2.6, 8.8, 3.3, 4.0, zf, zf + 0.9, m_steel)                       # stove line
    for u in (3.4, 4.3, 6.2):
        mb.cyl(0.19, 0.3, lw(side, u, 3.65, zf + 0.9), m_steel, segs=12)
    p_wallshelf(mb, side, 2.6, 8.8, 5.15, zf + 1.3, 1.4, 3, rng, depth=0.28, m=m_wood, hmax=0.2)
    p_standing(mb, side, 5.2, 2.95, zf, rng, coat=m_white, hairstyle="cap", h=1.72)
    for u in (3.2, 4.6, 6.0, 7.4):
        p_pendant(mb, side, u, 2.1, zc, 0.75, "amber", 10.0)
    for k in range(9):                                                              # paper menu strips over the kitchen
        fbox(mb, side, 2.9 + k * 0.62, 3.3 + k * 0.62, 4.82, 4.84, zf + 2.72, zf + 2.95, m_paper if k % 3 else m_yellow)
    fbox(mb, side, 2.75, 3.25, 0.55, 0.95, zf, zf + 1.45, m_red)                     # ticket machine
    fbox(mb, side, 2.8, 3.2, 0.53, 0.55, zf + 0.9, zf + 1.35, lightbox_mat((1.0, 0.8, 0.5), 0.5))
    # noren over the door, eave with lanterns
    for k in range(3):
        a = 7.8 + k * 0.4
        fbox(mb, side, a, a + 0.37, -0.02, 0.0, 1.75, 2.38, m_indigo)
        mb.cyl(0.09, 0.006, lw(side, a + 0.185, -0.03, 2.05), m_white, rot=(0, math.pi / 2, 0), segs=12, base=False)
    fbox(mb, side, 7.7, 9.05, -0.05, 0.0, 2.38, 2.43, m_wood)
    fbox(mb, side, ua - 0.3, ub + 0.3, -1.25, 0.0, 3.30, 3.42, m_wood)                 # wooden eave
    fbox(mb, side, ua - 0.3, ub + 0.3, -1.30, -1.20, 3.22, 3.45, m_wood)
    for k in range(8):
        fbox(mb, side, ua - 0.2 + k * 0.99, ua - 0.12 + k * 0.99, -1.2, 0.0, 3.2, 3.3, m_wood)
    for k, u in enumerate((2.8, 3.9, 5.0, 6.1, 7.2)):
        p_lantern(mb, lw(side, u, -1.05, 2.72), 0.19, "red", 1.5, cord=0.28)
    p_lantern(mb, lw(side, 9.55, -0.55, 2.2), 0.26, "amber", 1.3, cord=0.7)
    mb.finish(col=SHOPS, sharp=40)
    shop_light("ShopL_Ramen", side, 5.6, 2.3, zc - 0.25, 160, (1.0, 0.62, 0.3), size=4.5, size_y=1.6)
    shop_light("ShopL_RamenK", side, 5.6, 4.0, zc - 0.25, 70, (1.0, 0.75, 0.5), size=4.5, size_y=1.0)
    light("Lantern_Ramen", "POINT", lw(side, 5.0, -1.05, 2.5), 90 * K_SIGN, NEON["red"], size=0.3, col=LIGHTS, glossy=False)
    neon_sign("RAMENb", lw(side, 5.9, -0.14, 4.45), (-side, 0, 0), h=0.6, key="red", key2="amber", light_s=90, light_v=450)


shop_ramen()


def stair_door(side=-1, ua=10.0, ub=11.7, zt=2.75):
    mb = MB("Stair_Bar")
    zf = shop_shell(mb, side, ua, ub, zt, 3.0, 2.9, m_conc_d, m_black, m_black)
    for k in range(9):
        fbox(mb, side, ua, ub, 0.5 + k * 0.28, 3.0, zf + k * 0.19, zf + (k + 1) * 0.19, m_conc_d)
    fbox(mb, side, ua, ub, 2.95, 3.0, zf + 1.75, zf + 2.3, neon("purple", 1.2))
    mb.finish(col=SHOPS)
    light("ShopL_Stair", "POINT", lw(side, 10.85, 1.2, 2.3), 30 * K_SHOP, NEON["purple"], size=0.2, col=LIGHTS, glossy=False)
    neon_sign("BARg", lw(side, 10.85, -0.12, 3.45), (-side, 0, 0), h=0.42, key="pink", key2="cyan", border=True, light_s=40, light_v=160)


stair_door()


# ---------------------------------------------------------------- 24H MART (right, nearest)
def shop_mart(side=1, ua=0.9, ub=7.6, zt=3.3):
    rng = random.Random(31)
    mb = MB("Shop_Mart")
    zc = 3.22
    m_tilef = mat("mart_floor", (0.55, 0.56, 0.55), rough=0.12, coat=0.4)
    m_mwall = mat("mart_wall", (0.42, 0.43, 0.42), rough=0.5)
    zf = shop_shell(mb, side, ua, ub, zt, 8.0, zc, m_tilef, m_mwall, m_mwall)
    shop_glazing(mb, side, ua, ub, zt, doors=[(5.75, 7.35)], sill=0.22, mull=1.6, frame=m_alu_l, riser=m_tile_w, door_h=2.25)
    for k in range(3):                                                              # ceiling panels
        for j in range(3):
            fbox(mb, side, 1.6 + k * 2.2, 2.8 + k * 2.2, 1.3 + j * 2.3, 1.9 + j * 2.3, zc - 0.03, zc, neon("ice", 4.5))
    for uc in (2.0, 3.6):                                                           # aisles
        p_gondola(mb, side, uc, 1.9, 5.6, zf, rng)
    # magazine rack along the window, posters on the glass
    fbox(mb, side, 1.1, 5.3, 0.38, 0.72, zf, zf + 1.25, m_white)
    for r in range(3):
        p_products(mb, side, 1.15, 5.25, 0.36, 0.40, zf + 0.15 + r * 0.36, rng, hmax=0.3)
    for (a, b, c, e, mm) in ((1.25, 1.95, 1.7, 2.6, PRODUCT[1]), (3.05, 3.6, 1.9, 2.7, PRODUCT[2]), (4.35, 5.1, 1.55, 2.5, PRODUCT[0])):
        fbox(mb, side, a, b, 0.128, 0.134, c, e, mm)
    # drinks wall at the back
    fbox(mb, side, 0.75, 7.75, 7.3, 7.95, zf, zf + 2.2, m_alu_l)
    fbox(mb, side, 0.9, 7.6, 7.26, 7.28, zf + 0.12, zf + 2.1, neon("ice", 1.6))
    for r in range(5):
        fbox(mb, side, 0.9, 7.6, 6.95, 7.26, zf + 0.12 + r * 0.4, zf + 0.14 + r * 0.4, m_alu_l)
        u = 0.95
        while u < 7.5:
            mb.cyl(0.035, 0.23, lw(side, u, 7.1, zf + 0.14 + r * 0.4), rng.choice(PRODUCT), segs=6)
            u += 0.095
    for k in range(6):
        fbox(mb, side, 0.9 + k * 1.34 - 0.02, 0.9 + k * 1.34 + 0.02, 6.9, 6.95, zf, zf + 2.2, m_alu)
    fquad(mb, side, [(0.9, 6.92, zf + 0.1), (7.6, 6.92, zf + 0.1), (7.6, 6.92, zf + 2.15), (0.9, 6.92, zf + 2.15)], m_glass)
    # counter + clerk + customer
    fbox(mb, side, 5.2, 7.5, 2.6, 3.2, zf, zf + 0.95, m_white)
    fbox(mb, side, 5.15, 7.55, 2.55, 3.25, zf + 0.95, zf + 1.0, m_teal)
    fbox(mb, side, 6.3, 6.7, 2.75, 3.05, zf + 1.0, zf + 1.3, m_black)
    p_wallshelf(mb, side, 5.3, 7.7, 4.5, zf + 0.9, 1.3, 3, rng, depth=0.22, m=m_white, hmax=0.12)
    p_standing(mb, side, 6.4, 3.75, zf, rng, coat=m_teal, hairstyle="short", h=1.68)
    p_standing(mb, side, 2.85, 3.3, zf, rng, facing_street=False, hairstyle="bob", bag="tote", h=1.62)
    # north wall: stripe band, posters, cash machine, coffee corner
    fbox(mb, side, 7.84, 7.86, 0.3, 8.0, zc - 0.55, zc - 0.4, m_teal)
    fbox(mb, side, 7.84, 7.86, 0.3, 8.0, zc - 0.36, zc - 0.3, mat("mart_pink", (0.6, 0.08, 0.2), rough=0.4))
    fbox(mb, side, 0.64, 0.66, 0.3, 8.0, zc - 0.55, zc - 0.4, m_teal)
    for k, (d0, d1, z0, z1) in enumerate(((0.7, 1.5, 1.35, 2.45), (1.7, 2.3, 1.6, 2.4), (3.5, 4.4, 1.3, 2.5), (5.0, 5.7, 1.5, 2.35))):
        fbox(mb, side, 7.82, 7.84, d0, d1, zf + z0, zf + z1, PRODUCT[(k * 3 + 1) % len(PRODUCT)])
        fbox(mb, side, 7.815, 7.82, d0 + 0.08, d1 - 0.08, zf + z0 + 0.1, zf + z0 + 0.35, m_white)
    fbox(mb, side, 7.3, 7.8, 0.55, 1.15, zf, zf + 1.55, m_teal)
    fbox(mb, side, 7.28, 7.3, 0.65, 1.05, zf + 0.95, zf + 1.35, lightbox_mat((0.7, 0.9, 1.0), 1.2))
    fbox(mb, side, 7.2, 7.8, 4.9, 6.6, zf, zf + 0.9, m_white)
    for k in range(3):
        fbox(mb, side, 7.35, 7.75, 5.05 + k * 0.5, 5.4 + k * 0.5, zf + 0.9, zf + 1.45, m_blackg if k != 1 else m_red)
    # outside: bins, mat, umbrella stand
    for k, mm in enumerate((m_blue, m_green, m_alu_l)):
        fbox(mb, side, 1.15 + k * 0.56, 1.65 + k * 0.56, -0.48, -0.06, CURB, CURB + 0.85, mm)
        mb.cyl(0.11, 0.012, lw(side, 1.4 + k * 0.56, -0.485, CURB + 0.62), m_black, rot=(0, math.pi / 2, 0), segs=10, base=False)
    fbox(mb, side, 5.85, 7.25, -1.0, -0.1, CURB, CURB + 0.012, m_rubber)
    fbox(mb, side, 5.2, 5.6, -0.38, -0.08, CURB, CURB + 0.45, m_alu)
    for k in range(4):
        mb.rod(lw(side, 5.28 + k * 0.08, -0.2, CURB + 0.1), lw(side, 5.26 + k * 0.09, -0.26, CURB + 0.95), 0.012, rng.choice(list(UMB.values())[:5]), segs=5)
    mb.finish(col=SHOPS, sharp=40)
    for j in range(3):
        shop_light(f"ShopL_Mart{j}", side, 4.2, 1.6 + j * 2.3, zc - 0.2, 130, (0.82, 0.92, 1.0), size=5.5, size_y=1.0)
    # fascia: lit box with stripes
    lightbox("24H MART", lw(side, 4.25, -0.16, 3.83), (-side, 0, 0), h=0.42, face=(0.85, 0.88, 0.85), W=6.9, H=0.78, strength=1.5,
             stripes=[(0.31, 0.07, (0.05, 0.8, 0.6)), (-0.31, 0.07, (1.0, 0.25, 0.45))], ink=mat("ink_teal", (0.01, 0.16, 0.14), rough=0.4),
             light_s=110, light_v=350, lcolor=(0.8, 0.95, 1.0))
    neon_sign("OPEN", lw(side, 1.85, 0.2, 2.75), (-side, 0, 0), h=0.2, key="green", key2="red", board=False, border=True, light_s=0, light_v=0, tube_r=0.009)


shop_mart()


# ---------------------------------------------------------------- vending machines (right, beside the mart)
def vending(mb, side, uc, body, rng, w=0.95, hh=1.83, dp=0.66, glow="ice"):
    """Drinks machine against the wall: lit display with rows of cans behind glass, buttons, coin panel, flap."""
    u0, u1, d0, d1, z = uc - w / 2, uc + w / 2, -0.06 - dp, -0.06, CURB
    fbox(mb, side, u0, u1, d0, d1, z + 0.05, z + hh, body)
    fbox(mb, side, u0 + 0.04, u1 - 0.04, d0 + 0.04, d1 - 0.04, z, z + 0.05, m_black)
    wz0, wz1 = z + 0.78, z + hh - 0.08
    a, b = u0 + 0.06, u1 - 0.22
    fbox(mb, side, a, b, d0 - 0.006, d0, wz0, wz1, lightbox_mat(NEON[glow], 1.2))                  # lit back panel
    for (x0, x1, y0, y1) in ((a - 0.03, a, wz0 - 0.03, wz1 + 0.03), (b, b + 0.03, wz0 - 0.03, wz1 + 0.03),
                             (a, b, wz0 - 0.03, wz0), (a, b, wz1, wz1 + 0.03)):
        fbox(mb, side, x0, x1, d0 - 0.09, d0, y0, y1, body)                                        # bezel
    rows = 3
    for r in range(rows):
        zr = wz0 + 0.035 + r * (wz1 - wz0 - 0.03) / rows
        fbox(mb, side, a, b, d0 - 0.075, d0, zr - 0.03, zr - 0.004, m_whiteg)
        u = a + 0.045
        while u < b - 0.04:
            mb.cyl(0.028, 0.14 if r else 0.2, lw(side, u, d0 - 0.04, zr), rng.choice(PRODUCT), segs=6)
            mb.box((0.004, 0.02, 0.012), lw(side, u, d0 - 0.077, zr - 0.017), neon("red" if rng.random() < 0.3 else "blue", 2.0))
            u += 0.08
    fquad(mb, side, [(a, d0 - 0.085, wz0), (b, d0 - 0.085, wz0), (b, d0 - 0.085, wz1), (a, d0 - 0.085, wz1)], m_glass)
    fbox(mb, side, u1 - 0.17, u1 - 0.05, d0 - 0.012, d0, z + 0.95, z + 1.45, m_black)              # coin / card panel
    fbox(mb, side, u1 - 0.15, u1 - 0.07, d0 - 0.016, d0 - 0.01, z + 1.3, z + 1.4, neon("green", 1.5))
    fbox(mb, side, u0 + 0.12, u1 - 0.25, d0 - 0.012, d0, z + 0.22, z + 0.42, m_black)              # flap
    fbox(mb, side, u0 + 0.06, u1 - 0.06, d0 - 0.008, d0, z + 0.52, z + 0.7, lightbox_mat((1.0, 0.95, 0.9), 0.9))
    fbox(mb, side, u0 - 0.01, u1 + 0.01, d0 - 0.1, d1, z + hh, z + hh + 0.03, m_black)


vm = MB("VendingMachines")
_r = random.Random(41)
vending(vm, 1, 8.45, m_whiteg, _r, glow="ice")
vending(vm, 1, 9.5, m_red, _r, glow="white")
vending(vm, 1, 10.5, m_blue, _r, w=0.85, glow="ice")
vm.finish(col=SHOPS, bevel=0.012, sharp=40)
light("VendGlow", "AREA", lw(1, 9.45, -1.3, 1.35), 150 * K_SHOP, (0.8, 0.92, 1.0), size=2.6, size_y=0.9, target=lw(1, 9.45, -4.0, 0.9), col=LIGHTS, glossy=False)


# ---------------------------------------------------------------- ARCADE (left, middle)
def make_pixels(name, c1, c2, cell=0.06, strength=2.4):
    """Arcade screen facing the street: chunky pixels in world Y / Z."""
    m, n, p_, out = mat_nodes(name)
    x, y, z = n.sep(n.geo().outputs["Position"])
    iu, iv = n.floor(n.div(y, cell)), n.floor(n.div(z, cell))
    wv, wc = n.white(n.comb(iu, iv, 0.0))
    row, _ = n.white(n.comb(iv, 7.0, 0.0))
    on = n.gt(wv, n.maprange(row, 0.0, 1.0, 0.25, 0.8))
    n.L(n.mix(n.smooth(row, 0.3, 0.7), c1, c2), p_.inputs["Emission Color"])
    n.L(n.mul(n.add(0.12, n.mul(on, 0.88)), strength), p_.inputs["Emission Strength"])
    p_.inputs["Base Color"].default_value = (0.004, 0.004, 0.005, 1)
    p_.inputs["Roughness"].default_value = 0.1
    return m


SCREENS = [make_pixels(f"cab_screen_{i}", NEON[a_], NEON[b_]) for i, (a_, b_) in enumerate(
    [("cyan", "magenta"), ("green", "yellow"), ("magenta", "blue"), ("amber", "red"), ("cyan", "purple")])]


def cabinet(mb, side, u, d, zf, rng, facing=-1):
    """Arcade cabinet; facing -1 = screen toward the street."""
    f = facing
    fbox(mb, side, u - 0.32, u + 0.32, d - 0.4, d + 0.4, zf, zf + 1.75, m_black)
    scr = rng.choice(SCREENS)
    dd = d + f * 0.405
    fbox(mb, side, u - 0.26, u + 0.26, min(dd, dd + f * 0.012), max(dd, dd + f * 0.012), zf + 1.0, zf + 1.48, scr)
    fbox(mb, side, u - 0.3, u + 0.3, min(dd, dd + f * 0.03), max(dd, dd + f * 0.03), zf + 1.55, zf + 1.73, neon(rng.choice(("magenta", "cyan", "yellow")), 2.0))
    fbox(mb, side, u - 0.32, u + 0.32, min(dd, dd + f * 0.25), max(dd, dd + f * 0.25), zf + 0.82, zf + 0.9, m_alu)
    mb.sphere(0.03, lw(side, u - 0.1, dd + f * 0.14, zf + 0.96), neon("red", 2.0), u=6, v=4)


def shop_arcade(side=-1, ua=12.9, ub=20.1, zt=3.4):
    rng = random.Random(51)
    mb = MB("Shop_Arcade")
    zc = 3.3
    m_carpet = mat("arcade_carpet", (0.03, 0.012, 0.05), rough=0.8)
    m_awall = mat("arcade_wall", (0.02, 0.015, 0.035), rough=0.5)
    zf = shop_shell(mb, side, ua, ub, zt, 8.5, zc, m_carpet, m_awall, m_black)
    shop_glazing(mb, side, ua, ub, zt, doors=[(15.4, 17.6)], sill=0.3, mull=1.25, frame=m_alu, riser=m_black, door_h=2.5, open_door=True)
    for u in (13.4, 14.25, 15.1, 17.9, 18.75, 19.6):
        cabinet(mb, side, u, 5.4, zf, rng, facing=-1)
    for u in (13.6, 14.5, 18.4, 19.3):
        cabinet(mb, side, u, 8.0, zf, rng, facing=-1)
    for (u, key) in ((13.75, "pink"), (14.85, "cyan"), (18.25, "cyan"), (19.35, "pink")):           # claw machines at the window
        fbox(mb, side, u - 0.45, u + 0.45, 0.75, 1.65, zf, zf + 0.85, m_whiteg)
        fbox(mb, side, u - 0.45, u + 0.45, 0.75, 1.65, zf + 1.85, zf + 2.05, m_whiteg)
        for (a, b) in ((u - 0.45, 0.75), (u + 0.41, 0.75), (u - 0.45, 1.61), (u + 0.41, 1.61)):
            fbox(mb, side, a, a + 0.04, b, b + 0.04, zf + 0.85, zf + 1.85, m_alu_l)
        fbox(mb, side, u - 0.41, u + 0.41, 0.79, 1.61, zf + 1.82, zf + 1.85, neon(key, 4.0))
        for k in range(14):
            mb.sphere(rng.uniform(0.07, 0.11), lw(side, u + rng.uniform(-0.32, 0.32), 1.2 + rng.uniform(-0.3, 0.3), zf + 0.93 + rng.uniform(0, 0.12)),
                      rng.choice(PRODUCT), u=8, v=6)
        fquad(mb, side, [(u - 0.43, 0.76, zf + 0.85), (u + 0.43, 0.76, zf + 0.85), (u + 0.43, 0.76, zf + 1.85), (u - 0.43, 0.76, zf + 1.85)], m_glass)
    for k, key in enumerate(("purple", "cyan", "purple", "magenta")):                              # ceiling tubes
        mb.rod(lw(side, 13.2, 1.5 + k * 1.9, zc - 0.06), lw(side, 19.8, 1.5 + k * 1.9, zc - 0.06), 0.02, neon(key, 3.5), segs=6)
    p_standing(mb, side, 18.0, 4.55, zf, rng, facing_street=False, hairstyle="short", bag="back")
    p_standing(mb, side, 14.3, 4.5, zf, rng, facing_street=False, hairstyle="cap")
    mb.finish(col=SHOPS, sharp=40)
    shop_light("ShopL_Arcade", side, 16.5, 2.4, zc - 0.25, 150, (0.6, 0.25, 1.0), size=6.0, size_y=2.0)
    shop_light("ShopL_Arcade2", side, 16.5, 5.8, zc - 0.25, 90, (0.1, 0.7, 1.0), size=6.0, size_y=2.0)
    sg, (W, H) = neon_sign("ARCADEs", lw(side, 16.5, -0.14, 4.05), (-side, 0, 0), h=0.6, key="cyan", key2="magenta", light_s=110, light_v=500)
    cb = MB("ArcadeChaser")
    for k in range(30):
        cb.sphere(0.035, lw(side, 13.0 + k * 0.241, -0.2, 3.47), neon("yellow" if k % 3 else "white", 3.0), u=6, v=4)
    cb.finish(col=SHOPS)


shop_arcade()


# ---------------------------------------------------------------- CAFE (left, corner)
def shop_cafe(side=-1, ua=20.9, ub=28.6, zt=3.3):
    rng = random.Random(61)
    mb = MB("Shop_Cafe")
    zc = 3.2
    zf = shop_shell(mb, side, ua, ub, zt, 6.5, zc, m_wood, m_cream, m_cream)
    shop_glazing(mb, side, ua, ub, zt, doors=[(24.3, 25.5)], sill=0.55, mull=1.15, frame=m_green, riser=m_tile_g, door_h=2.2)
    for u in (21.6, 22.9, 26.4, 27.7):
        mb.cyl(0.32, 0.03, lw(side, u, 1.25, zf + 0.72), m_woodl, segs=14)
        mb.cyl(0.03, 0.72, lw(side, u, 1.25, zf), m_black, segs=6)
        mb.cyl(0.2, 0.02, lw(side, u, 1.25, zf), m_black, segs=10)
        for sg in (-1, 1):
            cu = u + sg * 0.55
            fbox(mb, side, cu - 0.19, cu + 0.19, 1.06, 1.44, zf + 0.42, zf + 0.46, m_woodl)
            fbox(mb, side, cu + sg * 0.17 - 0.02, cu + sg * 0.17 + 0.02, 1.06, 1.44, zf + 0.46, zf + 0.9, m_woodl)
            for (a, b) in ((cu - 0.17, 1.08), (cu + 0.15, 1.08), (cu - 0.17, 1.4), (cu + 0.15, 1.4)):
                fbox(mb, side, a, a + 0.025, b, b + 0.025, zf, zf + 0.42, m_black)
        p_pendant(mb, side, u, 1.25, zc, 0.9, "amber", 8.0, r=0.17)
    p_seated(mb, side, 22.35, 1.25, zf + 0.46, rng, hairstyle="long")
    fbox(mb, side, 21.5, 27.9, 4.4, 5.0, zf, zf + 1.05, m_wood)
    fbox(mb, side, 22.0, 24.0, 4.3, 4.45, zf + 0.3, zf + 1.0, neon("amber", 1.0))
    p_wallshelf(mb, side, 21.4, 28.0, 6.45, zf + 1.3, 1.5, 3, rng, depth=0.25, m=m_wood, hmax=0.22)
    p_standing(mb, side, 25.8, 5.5, zf, rng, coat=m_black, hairstyle="bob", h=1.64)
    p_plant(mb, lw(side, 23.9, 0.75, zf), 1.2)
    mb.finish(col=SHOPS, sharp=40)
    ex = MB("Cafe_Outside")
    awning(ex, side, ua - 0.2, ub + 0.2, 3.42, out=1.45, drop=0.5, m=m_teal, m2=m_cream, stripes=14)
    p_plant(ex, lw(side, 21.2, -0.4, CURB), 1.0)
    p_plant(ex, lw(side, 28.3, -0.4, CURB), 1.1)
    p_aframe(ex, lw(side, 25.9, -1.5, CURB), heading=0.3)
    ex.finish(col=SHOPS, sharp=40)
    shop_light("ShopL_Cafe", side, 24.7, 2.4, zc - 0.25, 150, (1.0, 0.66, 0.36), size=6.5, size_y=2.2)
    neon_sign("CAFEc", lw(side, 24.75, -0.14, 3.92), (-side, 0, 0), h=0.5, key="amber", key2="teal", light_s=60, light_v=300)


shop_cafe()


# ---------------------------------------------------------------- LAUNDRY + shuttered shop (right, middle)
def shop_laundry(side=1, ua=11.4, ub=15.6, zt=3.2):
    rng = random.Random(71)
    mb = MB("Shop_Laundry")
    zc = 3.1
    m_lfloor = mat("laundry_floor", (0.34, 0.4, 0.42), rough=0.2)
    zf = shop_shell(mb, side, ua, ub, zt, 5.5, zc, m_lfloor, m_white, m_white)
    shop_glazing(mb, side, ua, ub, zt, doors=[(14.3, 15.4)], sill=0.3, mull=1.45, frame=m_alu_l, riser=m_tile_w, door_h=2.2)
    for k in range(5):                                                              # washers along the back wall
        u = 11.75 + k * 0.82
        fbox(mb, side, u - 0.36, u + 0.36, 4.7, 5.45, zf, zf + 1.0, m_whiteg)
        fbox(mb, side, u - 0.36, u + 0.36, 4.7, 5.45, zf + 1.02, zf + 2.0, m_whiteg)
        for zz in (0.52, 1.5):
            c = lw(side, u, 4.69, zf + zz)
            mb.cyl(0.25, 0.03, c, m_chrome, rot=(0, math.pi / 2, 0), segs=16, base=False)
            mb.cyl(0.19, 0.04, c, m_blackg, rot=(0, math.pi / 2, 0), segs=16, base=False)
        fbox(mb, side, u - 0.3, u + 0.3, 4.68, 4.7, zf + 0.86, zf + 0.96, neon("ice", 1.0))
    fbox(mb, side, 11.6, 13.4, 2.1, 2.6, zf + 0.4, zf + 0.46, m_woodl)               # bench
    for u in (11.7, 13.3):
        fbox(mb, side, u - 0.03, u + 0.03, 2.15, 2.55, zf, zf + 0.4, m_alu)
    fbox(mb, side, 12.3, 12.9, 2.2, 2.5, zf + 0.46, zf + 0.72, PRODUCT[2])
    for j in range(2):
        fbox(mb, side, 11.9, 15.0, 1.6 + j * 2.0, 1.72 + j * 2.0, zc - 0.05, zc, neon("ice", 8.0))
    mb.finish(col=SHOPS, sharp=40)
    shop_light("ShopL_Laundry", side, 13.5, 2.6, zc - 0.25, 220, (0.75, 0.93, 1.0), size=3.6, size_y=3.0)
    lightbox("LAUNDRY", lw(side, 13.5, -0.16, 3.72), (-side, 0, 0), h=0.38, face=(0.35, 0.75, 1.0), W=4.3, H=0.72, strength=1.4,
             ink=mat("ink_navy", (0.01, 0.03, 0.12), rough=0.4), light_s=70, light_v=250)


shop_laundry()


def shop_shutter(side=1, ua=16.0, ub=19.1, zt=3.0, word="TEA + BOOKS", name="Shop_Shutter", lamp=True):
    """Closed shop: corrugated roller shutter, letter slot, a faded sign board and a wall lamp over it."""
    mb = MB(name)
    uc = (ua + ub) / 2
    n = 36
    prof = []
    for k in range(n + 1):
        z = CURB + (zt - CURB) * k / n
        prof.append((0.14 + (0.022 if k % 2 else 0.0), z))
    for (d0, z0), (d1, z1) in zip(prof, prof[1:]):
        fquad(mb, side, [(ua, d0, z0), (ub, d0, z0), (ub, d1, z1), (ua, d1, z1)], m_steel)
    for ue in (ua, ub - 0.06):
        fbox(mb, side, ue, ue + 0.06, 0.08, 0.2, CURB, zt, m_alu)
    fbox(mb, side, ua, ub, 0.05, 0.3, zt - 0.28, zt, m_alu)
    fbox(mb, side, uc - 0.25, uc + 0.25, 0.118, 0.14, CURB + 0.95, CURB + 1.02, m_black)
    fbox(mb, side, ua + 0.2, ub - 0.2, -0.1, 0.0, 3.25, 3.95, mat("old_sign", (0.25, 0.22, 0.16), rough=0.6))
    mb.finish(col=SHOPS, sharp=20)
    text(name + "_txt", word, None, mat("old_ink", (0.08, 0.02, 0.02), rough=0.6), size=0.42, extrude=0.004,
         matrix=basis((0, -side, 0), (-side, 0, 0), lw(side, uc, -0.108, 3.6)) @ Matrix.Rotation(math.pi / 2, 4, "X"), col=SHOPS)
    if lamp:
        fl = MB(name + "_lamp")
        fl.rod(lw(side, uc, 0.0, 4.15), lw(side, uc, -0.5, 4.3), 0.015, m_black, segs=5)
        fl.cyl(0.14, 0.1, lw(side, uc, -0.5, 4.18), m_black, segs=10, r2=0.05, caps=False)
        fl.sphere(0.045, lw(side, uc, -0.5, 4.19), neon("white", 12.0), u=6, v=4)
        fl.finish(col=SHOPS)
        light(name + "_spot", "SPOT", lw(side, uc, -0.5, 4.15), 260 * K_LAMP, (1.0, 0.8, 0.55), size=0.05, rot=(0, 0, 0), spot=110, blend=0.6,
              col=LIGHTS, glossy=False)


shop_shutter()
shop_shutter(-1, -4.3, 1.3, 3.0, word="PAWN  +  LOANS", name="Shop_Shutter_L0")
shop_shutter(-1, -8.9, -5.2, 3.0, word="TOYS", name="Shop_Shutter_L0b", lamp=False)


# ---------------------------------------------------------------- HOTEL lobby + bookshop (right, corner)
def shop_lobby(side=1, ua=20.2, ub=25.2, zt=3.6):
    rng = random.Random(81)
    mb = MB("Shop_HotelLobby")
    zc = 3.6
    m_marble = mat("lobby_floor", (0.42, 0.36, 0.28), rough=0.08, coat=0.5)
    zf = shop_shell(mb, side, ua, ub, zt, 7.5, zc, m_marble, m_woodl, m_cream)
    shop_glazing(mb, side, ua, ub, zt, doors=[(21.9, 23.5)], sill=0.12, mull=1.7, frame=mat("brass", (0.5, 0.33, 0.1), rough=0.25, metal=1.0), riser=m_black, door_h=2.5)
    fbox(mb, side, 21.2, 24.2, 5.6, 6.3, zf, zf + 1.1, m_wood)
    fbox(mb, side, 21.1, 24.3, 5.5, 6.4, zf + 1.1, zf + 1.15, m_blackg)
    fbox(mb, side, 21.4, 24.0, 7.42, 7.46, zf + 1.5, zf + 2.7, neon("amber", 1.1))
    p_standing(mb, side, 22.7, 6.8, zf, rng, coat=m_black, hairstyle="short")
    fbox(mb, side, 20.4, 21.0, 1.5, 3.3, zf, zf + 0.42, m_red)
    fbox(mb, side, 20.4, 20.6, 1.5, 3.3, zf + 0.42, zf + 0.85, m_red)
    p_plant(mb, lw(side, 24.7, 1.0, zf), 1.5)
    for k in range(9):
        a = k * 2.4
        mb.sphere(0.06, lw(side, 22.7 + 0.28 * math.cos(a), 2.6 + 0.28 * math.sin(a), zc - 0.5 - 0.06 * k), neon("amber", 12.0), u=6, v=4)
        mb.rod(lw(side, 22.7 + 0.28 * math.cos(a), 2.6 + 0.28 * math.sin(a), zc), lw(side, 22.7 + 0.28 * math.cos(a), 2.6 + 0.28 * math.sin(a), zc - 0.5 - 0.06 * k), 0.004, m_black, segs=4)
    mb.finish(col=SHOPS, sharp=40)
    ex = MB("Hotel_Canopy")
    fbox(ex, side, 20.6, 24.8, -2.6, 0.0, 3.75, 3.95, m_blackg)
    fbox(ex, side, 20.6, 24.8, -2.66, -2.6, 3.68, 4.02, mat("brass", (0.5, 0.33, 0.1)))
    for u in (21.2, 22.7, 24.2):
        for d in (-0.8, -1.9):
            ex.cyl(0.07, 0.012, lw(side, u, d, 3.738), neon("white", 14.0), segs=10)
    for u in (20.8, 24.6):
        ex.cyl(0.035, 3.6, lw(side, u, -2.45, CURB), mat("brass", (0.5, 0.33, 0.1)), segs=8)
        ex.cyl(0.24, 0.45, lw(side, u, -0.5, CURB), m_blackg, segs=10, r2=0.28)
        ex.sphere(0.3, lw(side, u, -0.5, CURB + 0.75), m_plant, u=10, v=8)
    fbox(ex, side, 21.9, 23.5, -2.5, -0.05, CURB, CURB + 0.012, m_red)
    ex.finish(col=SHOPS, sharp=40)
    shop_light("ShopL_Lobby", side, 22.7, 3.0, zc - 0.3, 190, (1.0, 0.72, 0.42), size=4.0, size_y=3.5)
    light("CanopyDown", "AREA", lw(side, 22.7, -1.3, 3.7), 130 * K_SHOP, (1.0, 0.8, 0.55), size=3.6, size_y=2.0, rot=(0, 0, 0), col=LIGHTS, glossy=False)


shop_lobby()


def shop_books(side=1, ua=25.9, ub=28.6, zt=3.2):
    rng = random.Random(91)
    mb = MB("Shop_Books")
    zc = 3.1
    zf = shop_shell(mb, side, ua, ub, zt, 5.0, zc, m_woodl, m_cream, m_cream)
    shop_glazing(mb, side, ua, ub, zt, doors=[(27.5, 28.4)], sill=0.6, mull=0.8, frame=m_wood, riser=m_wood, door_h=2.15)
    p_wallshelf(mb, side, 26.0, 28.5, 4.95, zf, 2.6, 6, rng, depth=0.26, m=m_wood, hmax=0.3)
    p_wallshelf(mb, side, 26.0, 27.3, 0.75, zf + 0.5, 0.9, 2, rng, depth=0.3, m=m_wood, hmax=0.28)
    fbox(mb, side, 26.3, 27.9, 2.2, 3.1, zf, zf + 0.8, m_wood)
    p_products(mb, side, 26.35, 27.85, 2.25, 3.05, zf + 0.8, rng, hmax=0.12)
    p_pendant(mb, side, 27.1, 2.6, zc, 0.7, "white", 10.0, r=0.2)
    mb.finish(col=SHOPS, sharp=40)
    shop_light("ShopL_Books", side, 27.2, 2.4, zc - 0.25, 120, (1.0, 0.82, 0.6), size=2.2, size_y=3.0)
    neon_sign("BOOKS", lw(side, 27.25, -0.14, 3.75), (-side, 0, 0), h=0.4, key="white", key2="amber", light_s=40, light_v=160)


shop_books()

tick("p40_blades")
# ================================================================ BLADE SIGNS (hang off the facades, face down the street)
def blade_dims(word, h, vertical, kind):
    n = len(word)
    if kind == "neon":
        if vertical:
            return 1.407 * h, h * (1.28 * n + 0.36)
        return word_width(word, h, 0.24) + 0.84 * h, 1.74 * h
    if vertical:
        return 1.6 * h, (0.98 * n + 0.5) * h
    return (0.72 * n + 0.9) * h, 1.75 * h


# frame boxes (x0, y0, x1, y1) that signs must keep clear of: the hero tower with its rings, the train
def _keep(pts, pad=0.004):
    fb = CAM.frame_box(pts)
    return (fb[0] - pad, fb[1] - pad, fb[2] + pad, fb[3] + pad)


PLACED = [_keep([(sx * 23.0, TOWER_Y - 23.0, z) for sx in (-1, 1) for z in (62.0, 174.0)]),                    # tower + rings
          _keep([(sx * 7.8, RAIL_Y - 3.2, z) for sx in (-1, 1) for z in (RAIL_Z + 0.9, RAIL_Z + 4.0)]),          # train
          _keep([(x, 45.5, z) for x in (-14.6, -8.6) for z in (23.2, 27.2)])]                                   # roof billboard


def _hits(a, b, m=0.005):
    return not (a[2] + m < b[0] or b[2] + m < a[0] or a[3] + m < b[1] or b[3] + m < a[1])


def blade(word, side, y, z, h=0.8, key="magenta", key2=None, vertical=True, kind="neon", arm=0.45, zmin=4.75, zmax=17.0, front=FRONT, tries=None, **kw):
    """Blade sign near height z. It is nudged up / down until it sits fully inside the frame and overlaps
    no other sign (as seen by the camera); if no height works it is left out."""
    W, H = blade_dims(word, h, vertical, kind)
    x = side * (front - arm - W / 2)
    spot = None
    for dz in (tries or (0.0, -0.4, 0.4, -0.8, 0.8, -1.2, 1.3, -1.7, 1.8, -2.2, 2.4, -2.8, 3.0, -3.4, 3.8, 4.6, 5.4)):
        zc = z + dz
        if zc - H / 2 < zmin or zc + H / 2 > zmax:
            continue
        fb = CAM.frame_box([(x - W / 2, y, zc - H / 2), (x + W / 2, y, zc + H / 2), (x - W / 2, y, zc + H / 2), (x + W / 2, y, zc - H / 2)])
        if fb is None:
            return None
        if fb[3] > 0.945 or fb[1] < 0.0:
            if os.environ.get("S3_DEBUG"):
                print("   ", word, "z=%.1f" % zc, [round(v, 3) for v in fb], "cut by top/bottom")
            continue
        if fb[0] < 0.015 or fb[2] > 0.985:
            break
        if any(_hits(fb, o) for o in PLACED):
            if os.environ.get("S3_DEBUG"):
                print("   ", word, "z=%.1f" % zc, [round(v, 3) for v in fb], "hits", [(i, [round(v, 3) for v in o]) for i, o in enumerate(PLACED) if _hits(fb, o)])
            continue
        spot = (zc, fb)
        break
    if spot is None:
        print(f"[s3] blade sign {word} at y={y:.1f} left out (no clear spot in frame)")
        return None
    PLACED.append(spot[1])
    if os.environ.get("S3_DEBUG"):
        print("PLACED %-9s y=%6.1f z=%5.1f  box x %.3f-%.3f  y %.3f-%.3f" % (word, y, spot[0], spot[1][0], spot[1][2], spot[1][1], spot[1][3]))
    c = Vector((x, y, spot[0]))
    if kind == "neon":
        return neon_sign(word, c, (0, -1, 0), h=h, key=key, key2=key2, vertical=vertical, wall=side, arm=arm, **kw)
    return lightbox(word, c, (0, -1, 0), h=h, vertical=vertical, wall=side, arm=arm, **kw)


m_ink_w = mat("ink_white", (0.8, 0.8, 0.78), rough=0.4, emit=(1, 0.97, 0.9), emit_str=2.6)
m_ink_r = mat("ink_red", (0.45, 0.02, 0.02), rough=0.4)
m_ink_n = mat("ink_navy", (0.01, 0.03, 0.12), rough=0.4)
m_ink_k = m_black
RED_BAND = (0.9, 0.06, 0.05)
# --- first block, placed by hand for this framing (frame coordinates were checked with S3_DEBUG=1)
NEAR_DZ = (0.0, -0.2, 0.2, -0.4, 0.4)
# left side: warm family (red, pink, amber, magenta)
blade("RAMEN", -1, 3.0, 5.95, h=0.42, key="red", key2="amber", light_s=120, light_v=700, zmin=4.45, tries=NEAR_DZ)
blade("KARAOKE", -1, 12.9, 9.2, h=0.62, key="magenta", key2="purple", dim=(4,), light_s=150, light_v=900, tries=NEAR_DZ)
blade("BARg", -1, 11.0, 5.2, h=0.5, key="pink", key2="cyan", vertical=False, light_s=60, light_v=300, tries=NEAR_DZ)
blade("GAMES", -1, 18.6, 8.9, h=0.5, kind="box", vertical=False, face=(1.0, 0.72, 0.1), ink=m_ink_k, strength=1.6, light_s=70, light_v=260, tries=NEAR_DZ)
blade("LIVEn", -1, 27.2, 5.6, h=0.42, key="amber", key2="pink", vertical=False, light_s=40, light_v=200, tries=NEAR_DZ)
# right side: cool family (cyan, teal, blue, green, white)
blade("24H", 1, 5.6, 5.6, h=0.46, kind="box", vertical=False, face=(0.03, 0.62, 0.42), ink=m_ink_w, strength=1.3, light_s=60, light_v=220,
      stripes=[(0.33, 0.05, (0.9, 0.9, 0.85)), (-0.33, 0.05, (0.9, 0.9, 0.85))], tries=NEAR_DZ)
blade("CLUB", 1, 8.8, 8.4, h=0.7, key="purple", key2="cyan", light_s=120, light_v=700, tries=NEAR_DZ)
_sw, _sh = blade_dims("SUSHI", 0.66, True, "box")
blade("SUSHI", 1, 13.4, 7.4, h=0.66, kind="box", face=(0.85, 0.83, 0.77), ink=m_ink_r, strength=1.25, light_s=80, light_v=300, lcolor=NEON["white"],
      stripes=[(_sh / 2 - 0.16, 0.12, RED_BAND), (-_sh / 2 + 0.16, 0.12, RED_BAND)], tries=NEAR_DZ)
blade("NOODLEb", 1, 17.0, 5.5, h=0.46, key="cyan", key2="amber", vertical=False, light_s=60, light_v=300, tries=NEAR_DZ)
blade("HOTEL", 1, 21.6, 11.2, h=1.1, key="cyan", key2="ice", dim=(3,), light_s=220, light_v=1400, arm=0.6, tries=NEAR_DZ)
blade("+", 1, 26.6, 5.4, h=0.7, key="green", key2="green", vertical=False, light_s=60, light_v=320, tries=NEAR_DZ)

tick("p45_far")
# ================================================================ REST OF THE STREET, SKYLINE
FAR = group("Buildings_Far")
m_beacon = neon("red", 9.0)
m_tank = mat("tank_wood", (0.09, 0.055, 0.035), rough=0.7)
WORDS = ["HOTEL", "BAR", "SOBA", "CLUB", "TAXI", "CAFE", "NOODLE", "OPEN", "KARAOKE", "TEA", "BOOKS", "LASER", "PIXEL", "SAUNA",
         "DINER", "VIDEO", "MOTEL", "PIZZA", "UDON", "SPA", "JAZZ", "GRILL", "TOYS", "INN", "CINEMA", "MUSIC", "RECORDS", "CURRY"]
WARM, COOL = ["magenta", "pink", "red", "amber", "purple"], ["cyan", "teal", "blue", "ice", "green"]


def roof_kit(mb, x0, x1, y0, y1, h, rng, tank=0.6, antenna=0.6, big=1.0):
    """Parapet, AC boxes, stair bulkhead, water tank on legs, lattice antenna with a beacon."""
    mb.bx((x0, y0, h), (x1, y0 + 0.25, h + 0.9), m_conc_d)
    mb.bx((x0, y1 - 0.25, h), (x1, y1, h + 0.9), m_conc_d)
    xa, xb = (x0, x0 + 0.25) if abs(x0) < abs(x1) else (x1 - 0.25, x1)
    mb.bx((xa, y0, h), (xb, y1, h + 0.9), m_conc_d)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, d = abs(x1 - x0), abs(y1 - y0)
    for _ in range(rng.randint(1, 4)):
        sx_, sy_ = rng.uniform(1.0, 2.2), rng.uniform(1.0, 2.2)
        px_, py_ = cx + rng.uniform(-w / 3, w / 3), cy + rng.uniform(-d / 3, d / 3)
        mb.bx((px_ - sx_ / 2, py_ - sy_ / 2, h), (px_ + sx_ / 2, py_ + sy_ / 2, h + rng.uniform(0.8, 1.4)), m_acbody)
    if rng.random() < 0.6:
        bx_, by_ = cx + rng.uniform(-w / 4, w / 4), cy + rng.uniform(-d / 4, d / 4)
        mb.bx((bx_ - 1.6, by_ - 1.3, h), (bx_ + 1.6, by_ + 1.3, h + 2.8), m_conc)
    if rng.random() < tank:
        tx, ty = cx + rng.uniform(-w / 4, w / 4), cy + rng.uniform(-d / 4, d / 4)
        tr = 1.25 * big
        for dx_ in (-0.8, 0.8):
            for dy_ in (-0.8, 0.8):
                mb.rod((tx + dx_ * big, ty + dy_ * big, h), (tx + dx_ * big, ty + dy_ * big, h + 2.0 * big), 0.07, m_rust, segs=5)
        mb.cyl(tr, 2.4 * big, (tx, ty, h + 2.0 * big), m_tank, segs=14)
        mb.cyl(tr * 1.08, 0.8 * big, (tx, ty, h + 4.4 * big), m_rust, segs=14, r2=0.05)
        for k in (0.5, 1.6):
            mb.cyl(tr * 1.02, 0.07, (tx, ty, h + (2.0 + k) * big), m_rust, segs=14)
    if rng.random() < antenna:
        ax, ay = cx + rng.uniform(-w / 3, w / 3), cy + rng.uniform(-d / 3, d / 3)
        ah = rng.uniform(5, 12)
        for (dx_, dy_) in ((-0.35, -0.35), (0.35, -0.35), (0.0, 0.4)):
            mb.rod((ax + dx_, ay + dy_, h), (ax, ay, h + ah), 0.035, m_steel, segs=4)
        for k in range(1, 5):
            t = k / 5
            pts = [(ax + dx_ * (1 - t), ay + dy_ * (1 - t), h + ah * t) for (dx_, dy_) in ((-0.35, -0.35), (0.35, -0.35), (0.0, 0.4))]
            for a_, b_ in zip(pts, pts[1:] + pts[:1]):
                mb.rod(a_, b_, 0.02, m_steel, segs=4)
        mb.sphere(0.16, (ax, ay, h + ah + 0.1), m_beacon, u=8, v=6)
        if rng.random() < 0.5:
            mb.cyl(0.55, 0.12, (ax + 0.5, ay, h + ah * 0.6), m_white, rot=(math.pi / 2, 0, 0.5), segs=12, r2=0.2)


def billboard(mb, cx, y, z0, w, hgt, mscreen, legs=2.2):
    """Roof billboard facing the camera (-Y): returns the screen centre so the caller can make the screen object."""
    for sx_ in (-w / 3, w / 3):
        mb.rod((cx + sx_, y + 0.25, z0), (cx + sx_, y + 0.25, z0 + legs + hgt), 0.09, m_steel, segs=6)
        mb.rod((cx + sx_, y + 2.6, z0), (cx + sx_, y + 0.3, z0 + legs + hgt * 0.75), 0.06, m_steel, segs=5)
    mb.bx((cx - w / 2 - 0.15, y, z0 + legs - 0.15), (cx + w / 2 + 0.15, y + 0.22, z0 + legs + hgt + 0.15), m_alu)
    mb.bx((cx - w / 2 - 0.3, y - 0.5, z0 + legs - 0.25), (cx + w / 2 + 0.3, y + 0.1, z0 + legs - 0.15), m_grate)   # catwalk
    s = MB(mb.name + "_screen")
    s.box((w, 0.04, hgt), (0, 0, 0), mscreen)
    return s.finish(col=FAR, loc=(cx, y - 0.03, z0 + legs + hgt / 2))


def far_building(name, side, y0, y1, h, depth, wm, rng, strip=True, front=FRONT, roof=True, n_signs=0, palette=None, tier=0.4, balc=0.3):
    W = y1 - y0
    xs = sorted((side * front, side * (front + depth)))
    b = MB(name)
    b.bx((xs[0], y0 + 0.04, 0.0), (xs[1], y1 - 0.04, h), wm)
    ob = b.finish(col=FAR)
    d = MB(name + "_detail")
    top, x0r, x1r, y0r, y1r = h, xs[0], xs[1], y0, y1
    if rng.random() < tier and h > 30:                                             # set-back upper tier
        h2 = h * rng.uniform(0.15, 0.35)
        t = MB(name + "_tier")
        x0r, x1r = sorted((side * (front + depth * 0.22), side * (front + depth * 0.9)))
        y0r, y1r = y0 + W * 0.12, y1 - W * 0.12
        t.bx((x0r, y0r, h), (x1r, y1r, h + h2), wm)
        t.finish(col=FAR)
        d.bx((xs[0] - 0.2, y0, h - 0.3), (xs[1] + 0.2, y1, h + 0.5), m_conc_d)
        top = h + h2
    if strip:
        sm = rng.choice(SHOPSTRIPS)
        fbox(d, side, y0 + 0.5, y1 - 0.5, front - FRONT - 0.04, front - FRONT + 0.02, 0.35, 3.25, sm)
        fbox(d, side, y0 + 0.3, y1 - 0.3, front - FRONT - 1.3, front - FRONT, 3.3, 3.42, rng.choice((m_indigo, m_red, m_teal, m_black, m_green)))
        fbox(d, side, y0, y1, front - FRONT - 0.12, front - FRONT, 3.42, 4.2, m_conc_d)
        light(name + "_spill", "AREA", lw(side, (y0 + y1) / 2, front - FRONT - 1.0, 2.6), 140 * K_SHOP * W / 12.0,
              (1.0, 0.75, 0.5) if "warm" in sm.name or "pink" in sm.name else (0.7, 0.95, 1.0), size=W * 0.7, size_y=1.0,
              target=lw(side, (y0 + y1) / 2, front - FRONT - 3.5, 0.0), col=LIGHTS, glossy=False)
    floors = int((h - 4.2) // 3.1)
    if rng.random() < balc:                                                         # balcony slabs
        for f in range(1, floors):
            z = 4.2 + f * 3.1
            fbox(d, side, y0 + 0.4, y1 - 0.4, front - FRONT - 1.05, front - FRONT, z - 0.12, z, m_conc)
            fbox(d, side, y0 + 0.4, y1 - 0.4, front - FRONT - 1.05, front - FRONT - 1.0, z, z + 0.95, m_balpanel)
    else:
        for f in range(1, floors, 2):
            z = 4.2 + f * 3.1
            fbox(d, side, y0, y1, front - FRONT - 0.12, front - FRONT, z - 0.15, z + 0.05, m_conc_d)
        for _ in range(rng.randint(3, 9)):
            ac_unit(d, side, rng.uniform(y0 + 1, y1 - 1), front - FRONT - 0.02, 4.9 + rng.randint(0, max(1, min(floors, 9) - 1)) * 3.1)
    drainpipe(d, side, y0 + 0.25, CURB, min(h, 40), d=front - FRONT - 0.07)
    if roof:
        roof_kit(d, x0r, x1r, y0r, y1r, top, rng)
    d.finish(col=FAR, sharp=40)
    for k in range(n_signs):
        word = rng.choice(WORDS)
        pal = palette or (WARM if side < 0 else COOL)
        key = rng.choice(pal)
        yy = y0 + W * (k + 0.5) / n_signs + rng.uniform(-1.0, 1.0)
        zc = rng.uniform(6.5, 11.5)
        hh = rng.uniform(0.85, 1.25)
        if rng.random() < 0.6:
            blade(word, side, yy, zc, h=hh, key=key, key2=rng.choice(pal + ["white"]), vertical=rng.random() < 0.7, front=front,
                  light_s=110, light_v=600, tube_r=hh * 0.045)
        else:
            blade(word, side, yy, zc, h=hh, kind="box", vertical=rng.random() < 0.7, face=NEON[key], ink=m_ink_w if key not in ("ice",) else m_ink_k,
                  strength=1.5, front=front, light_s=90, light_v=450)
    return ob, top


rngf = random.Random(404)
ROOFS = {}
# block 1 (y 45..133) and block 2 (141..252); heights fall away toward the tower so the canyon opens up
PLAN = {
    -1: [(45.0, 59.0, 22.0, 3), (59.0, 72.0, 26.0, 0), (72.0, 86.0, 30.0, 1), (86.0, 101.0, 34.0, 5), (101.0, 117.0, 38.0, 1), (117.0, 133.0, 43.0, 4),
         (141.0, 158.0, 40.0, 2), (158.0, 176.0, 34.0, 0), (176.0, 196.0, 40.0, 3), (196.0, 216.0, 30.0, 1), (216.0, 234.0, 26.0, 5), (234.0, 252.0, 22.0, 0)],
    1: [(45.0, 60.0, 58.0, 4), (60.0, 74.0, 40.0, 1), (74.0, 90.0, 50.0, 2), (90.0, 104.0, 36.0, 0), (104.0, 120.0, 46.0, 5), (120.0, 133.0, 33.0, 3),
        (141.0, 160.0, 38.0, 4), (160.0, 178.0, 44.0, 2), (178.0, 198.0, 30.0, 0), (198.0, 218.0, 35.0, 1), (218.0, 236.0, 24.0, 3), (236.0, 252.0, 20.0, 5)],
}
for side, plan in PLAN.items():
    for k, (ya, yb, hh, wi) in enumerate(plan):
        first_left = (side == -1 and k == 0)
        ob, top = far_building(f"Far_{'L' if side < 0 else 'R'}{k}", side, ya, yb, hh, rngf.uniform(14, 20), WIN_MATS[wi], rngf,
                               n_signs=(2 if ya < 133 else 1) if not first_left else 2, tier=0.0 if (side < 0 and ya < 140) else 0.4,
                               roof=not first_left)
        ROOFS[(side, k)] = top
# plain buildings beside / behind the camera: they close the canyon and spill shop light on the foreground
far_building("Near_L0", -1, -40.0, -9.5, 30.0, 16.0, WIN_MATS[0], rngf, n_signs=0, tier=0.0, roof=False, strip=False)
far_building("Near_R0", 1, -40.0, 0.5, 34.0, 16.0, WIN_MATS[5], rngf, n_signs=0, tier=0.0, roof=False, strip=False)
light("Spill_L0", "AREA", lw(-1, -9.0, -1.0, 2.8), 110 * K_SHOP, (1.0, 0.7, 0.45), size=8.0, size_y=1.0, target=lw(-1, -4.0, -4.0, 0.0), col=LIGHTS, glossy=False)
light("Spill_R0", "AREA", lw(1, -5.0, -1.0, 2.8), 150 * K_SHOP, (0.75, 0.95, 1.0), size=8.0, size_y=1.0, target=lw(1, -4.0, -4.0, 0.0), col=LIGHTS, glossy=False)
# rooftop of the low corner building across the junction: billboard + tank, the far city shows above it
rb = MB("Roof_L_corner")
roof_kit(rb, -FRONT - 16.0, -FRONT, 45.0, 59.0, 22.0, random.Random(5), tank=0.0, antenna=1.0)
billboard(rb, -11.6, 45.5, 22.0, 5.4, 3.3, m_scr_sun, legs=1.5)
rb.cyl(1.3, 2.5, (-12.5, 54.0, 24.0), m_tank, segs=14)
rb.cyl(1.4, 0.8, (-12.5, 54.0, 26.5), m_rust, segs=14, r2=0.05)
for dx_ in (-0.85, 0.85):
    for dy_ in (-0.85, 0.85):
        rb.rod((-12.5 + dx_, 54.0 + dy_, 22.0), (-12.5 + dx_, 54.0 + dy_, 24.0), 0.07, m_rust, segs=5)
rb.finish(col=FAR, sharp=40)
light("Billboard_L_glow", "POINT", (-11.6, 44.2, 25.0), 1500 * K_FOGL, NEON["magenta"], size=1.0, col=LIGHTS, glossy=False, diffuse=False)

# water tanks and masts right at the street edge of a few roofs, where the camera sees them against the sky
def edge_tank(mb, x, y, z, s=1.0, mast=0.0):
    for dx_ in (-0.8, 0.8):
        for dy_ in (-0.8, 0.8):
            mb.rod((x + dx_ * s, y + dy_ * s, z), (x + dx_ * s, y + dy_ * s, z + 2.0 * s), 0.07, m_rust, segs=5)
            mb.rod((x + dx_ * s, y + dy_ * s, z + 0.2), (x - dx_ * s, y + dy_ * s, z + 1.9 * s), 0.03, m_rust, segs=4)
    mb.cyl(1.25 * s, 2.5 * s, (x, y, z + 2.0 * s), m_tank, segs=16)
    mb.cyl(1.36 * s, 0.85 * s, (x, y, z + 4.5 * s), m_rust, segs=16, r2=0.05)
    for k in (0.4, 1.3, 2.2):
        mb.cyl(1.28 * s, 0.07, (x, y, z + (2.0 + k) * s), m_rust, segs=16)
    if mast:
        mb.rod((x + 2.6, y, z), (x + 2.6, y, z + mast), 0.06, m_steel, segs=5, r2=0.02)
        mb.sphere(0.2, (x + 2.6, y, z + mast), m_beacon, u=8, v=6)


et = MB("Roof_EdgeTanks")
edge_tank(et, -10.8, 65.0, ROOFS[(-1, 1)], 1.1, mast=8.0)
edge_tank(et, -10.6, 92.0, ROOFS[(-1, 3)], 1.2)
edge_tank(et, 10.8, 97.0, ROOFS[(1, 3)], 1.2, mast=10.0)
edge_tank(et, 10.7, 126.0, ROOFS[(1, 5)], 1.3)
edge_tank(et, -10.9, 24.6, B["L3"]["H"], 0.9)
et.finish(col=FAR, sharp=40)

# roof clutter of the low cafe building (near left, 14 m): parapet rail, small units
rl = MB("Roof_L3")
roof_kit(rl, -FRONT - 15.0, -FRONT + 0.2, 20.5, 29.0, B["L3"]["H"] + 1.0, random.Random(8), tank=1.0, antenna=1.0, big=0.8)
rl.finish(col=NEAR, sharp=40)

# second row: taller slabs behind the street buildings
for i in range(26):
    side = -1 if i % 2 else 1
    y = 50 + i * 8.5 + rngf.uniform(-4, 4)
    x = side * rngf.uniform(30, 62)
    w, d_, hh = rngf.uniform(14, 24), rngf.uniform(14, 24), rngf.uniform(55, 120)
    if side < 0:
        hh = min(hh, 1.5 + 0.32 * (y + 10.5))                 # keep the left gap open toward the far city
    s = MB(f"Row2_{i}")
    s.bx((x - w / 2, y - d_ / 2, 0), (x + w / 2, y + d_ / 2, hh), rngf.choice(WIN_MATS + WIN_SKY))
    if rngf.random() < 0.7:
        s.rod((x, y, hh), (x, y, hh + rngf.uniform(6, 16)), 0.25, m_steel, segs=5, r2=0.05)
        s.sphere(0.45, (x, y, hh + 1.0), m_beacon, u=8, v=6)
    s.finish(col=FAR)

# far skyline; a few placed by hand in the gap over the low left corner
SKY = group("Skyline")
HAND = [(-33.0, 150.0, 118.0, 20.0), (-55.0, 232.0, 176.0, 26.0), (-74.0, 318.0, 214.0, 30.0), (-44.0, 196.0, 96.0, 18.0),
        (-95.0, 400.0, 250.0, 34.0), (-26.0, 118.0, 74.0, 15.0)]
rs = random.Random(909)
towers = list(HAND)
for i in range(46):
    x = rs.uniform(-260, 260)
    y = rs.uniform(120, 560)
    if abs(x) < 70 and y > 240:
        continue
    if abs(x) < 26:
        continue
    towers.append((x, y, rs.uniform(80, 230), rs.uniform(16, 32)))
for i, (x, y, hh, w) in enumerate(towers):
    s = MB(f"Sky_{i}")
    wm = rs.choice(WIN_SKY + [WIN_MATS[2], WIN_MATS[3]])
    s.bx((x - w / 2, y - w / 2, 0), (x + w / 2, y + w / 2, hh * 0.78), wm)
    s.bx((x - w * 0.36, y - w * 0.36, hh * 0.78), (x + w * 0.36, y + w * 0.36, hh), wm)
    s.rod((x, y, hh), (x, y, hh + 18), 0.5, m_steel, segs=5, r2=0.08)
    for zz in (hh + 0.5, hh + 9, hh + 18):
        s.sphere(0.7, (x, y, zz), m_beacon, u=8, v=6)
    if rs.random() < 0.5:
        e = w / 2 + 0.2
        for (ex, ey) in ((-e, -e), (e, -e)):
            s.bx((x + ex - 0.25, y + ey - 0.25, hh * 0.3), (x + ex + 0.25, y + ey + 0.25, hh * 0.78), neon(rs.choice(("cyan", "magenta", "ice", "amber")), 2.5))
    s.finish(col=SKY)

tick("p50_street")
# ================================================================ STREET FURNITURE
STREET = group("Street")
m_pole = mat("pole_green", (0.03, 0.05, 0.045), rough=0.4, metal=0.5)
m_polec = make_wall("pole_concrete", (0.2, 0.2, 0.195), rough=0.65, streak=0.3)
m_lamp_lens = neon("amber", 22.0)
SODIUM = (1.0, 0.56, 0.24)


def lamp_mesh():
    mb = MB("StreetLamp")
    mb.cyl(0.11, 8.0, (0, 0, 0), m_pole, segs=10, r2=0.07)
    mb.cyl(0.17, 0.55, (0, 0, 0), m_pole, segs=10, r2=0.12)
    mb.tube([(0, 0, 7.9), (0.03, 0, 8.25), (0.45, 0, 8.6), (1.4, 0, 8.7), (2.2, 0, 8.62)], 0.045, m_pole, segs=6)
    mb.sphere(1.0, (2.5, 0, 8.58), m_pole, scale=(0.5, 0.19, 0.085), u=12, v=8)
    mb.box((0.62, 0.22, 0.03), (2.52, 0, 8.515), m_lamp_lens)
    mb.box((0.34, 0.03, 0.34), (0.0, -0.14, 2.9), m_blue, rot=(0, math.radians(45), 0))        # small street sign
    mb.box((0.26, 0.2, 0.34), (0.0, 0.2, 1.3), m_alu)                                           # control box
    return mb.mesh(sharp=40)


LAMP_ME = lamp_mesh()
LAMPS = [(1, -9.5), (-1, -7.5), (-1, 29.3), (1, 52.0), (-1, 70.0), (1, 92.0), (-1, 112.0), (1, 150.0), (-1, 176.0), (1, 204.0), (-1, 232.0)]
for k, (side, y) in enumerate(LAMPS):
    x = side * (RW + 0.55)
    inst(LAMP_ME, f"Lamp_{k}", (x, y, CURB), (0, 0, 0 if side < 0 else math.pi), col=STREET)
    hx = x - side * 2.5
    near = y < 40
    light(f"LampSpot_{k}", "SPOT", (hx, y, 8.62), ((1000 if y < 0 else 1500) if near else 1100) * K_LAMP, SODIUM, size=0.12, rot=(0, 0, 0), spot=128, blend=0.55,
          col=LIGHTS, glossy=False, volume=False)
    if 10 < y < 120:                               # the cone you see in the rain: a fog-only twin of the lamp
        light(f"LampCone_{k}", "SPOT", (hx, y, 8.6), 70000 * K_FOGL, SODIUM, size=0.12, rot=(0, 0, 0), spot=96, blend=0.8,
              col=LIGHTS, glossy=False, diffuse=False)

# --- traffic signals at the junction
m_sig_red = neon("red", 14.0)
m_sig_off_g = mat("sig_off_g", (0.02, 0.07, 0.04), rough=0.3)
m_sig_off_y = mat("sig_off_y", (0.08, 0.06, 0.01), rough=0.3)
m_sig_green = neon("green", 10.0)


def signal_head(mb, c, lit="red", facing=-1):
    """Horizontal three-lamp head centred at c, lenses facing -Y (facing=-1) or +Y."""
    c = Vector(c)
    mb.box((1.25, 0.24, 0.42), c, m_black)
    mb.box((1.33, 0.03, 0.5), c + Vector((0, -facing * 0.13, 0)), m_blackg)
    for k, nm in enumerate(("green", "yellow", "red")):
        p = c + Vector((-0.4 + k * 0.4, facing * 0.125, 0))
        mm = {"red": m_sig_red, "green": m_sig_green}.get(nm) if nm == lit else {"green": m_sig_off_g, "yellow": m_sig_off_y, "red": mat("sig_off_r", (0.08, 0.01, 0.01), rough=0.3)}[nm]
        mb.cyl(0.13, 0.02, p, mm, rot=(math.pi / 2, 0, 0), segs=14, base=False)
        mb.cyl(0.155, 0.2, p + Vector((0, facing * 0.1, 0.02)), m_black, rot=(math.pi / 2, 0, 0), segs=14, base=False, caps=False)


def traffic_signal(name, x, y, arm_dx, lit="red", facing=-1, ped=True):
    mb = MB(name)
    mb.cyl(0.11, 6.6, (x, y, CURB), m_steel, segs=10, r2=0.09)
    mb.cyl(0.18, 0.4, (x, y, CURB), m_steel, segs=10)
    mb.tube([(x, y, 6.2), (x + arm_dx * 0.08, y, 6.42), (x + arm_dx, y, 6.5)], 0.06, m_steel, segs=6)
    hc = (x + arm_dx * 0.72, y + facing * 0.05, 6.1)
    signal_head(mb, hc, lit, facing)
    mb.rod((hc[0], y, 6.3), (hc[0], y, 6.5), 0.03, m_steel, segs=5)
    if ped:
        pc = Vector((x - math.copysign(0.25, x), y + facing * 0.02, 2.75))
        mb.box((0.32, 0.2, 0.62), pc, m_black)
        mb.box((0.22, 0.02, 0.22), pc + Vector((0, facing * 0.1, -0.14)), neon("green", 5.0))
        mb.box((0.22, 0.02, 0.22), pc + Vector((0, facing * 0.1, 0.14)), mat("sig_off_r", (0.08, 0.01, 0.01), rough=0.3))
        mb.box((0.2, 0.16, 0.3), (x, y - 0.18 * facing, 1.25), m_yellow)                        # push-button box
    mb.finish(col=STREET, sharp=40)
    if lit and facing < 0:
        col = NEON[lit]
        light(name + "_glow", "POINT", (hc[0] + 0.4, y - 0.45, 6.1), 60 * K_SIGN, col, size=0.1, col=LIGHTS, glossy=False)
        light(name + "_fog", "POINT", (hc[0] + 0.4, y - 0.5, 6.1), 500 * K_FOGL, col, size=0.2, col=LIGHTS, glossy=False, diffuse=False)


traffic_signal("Signal_NearRight", RW + 0.75, 28.0, -4.9, "red", -1)
traffic_signal("Signal_FarRight", RW + 0.75, 45.6, -4.9, "red", -1, ped=False)
traffic_signal("Signal_FarLeft", -(RW + 0.75), 46.0, 4.9, "red", 1)

# --- utility poles, wires, string lights
def cable(mb, p1, p2, sag, r=0.012, m=None, n=14, bulbs=None, bulb_r=0.05):
    p1, p2 = Vector(p1), Vector(p2)
    pts = []
    for i in range(n + 1):
        t = i / n
        p = p1.lerp(p2, t)
        p.z -= sag * 4 * t * (1 - t)
        pts.append(p)
    mb.tube(pts, r, m or m_black, segs=4, caps=False)
    if bulbs:
        nb = max(4, int((p2 - p1).length / 0.55))
        for i in range(1, nb):
            t = i / nb
            p = p1.lerp(p2, t)
            p.z -= sag * 4 * t * (1 - t) + 0.07
            mb.rod(p + Vector((0, 0, 0.07)), p + Vector((0, 0, 0.02)), 0.012, m_black, segs=4)
            mb.sphere(bulb_r, p, bulbs[i % len(bulbs)], u=8, v=6)
    return pts


def utility_pole(mb, x, y, h=10.5, transformer=False, arms=((9.6, 1.5), (8.7, 1.1))):
    mb.cyl(0.17, h, (x, y, CURB), m_polec, segs=10, r2=0.11)
    tops = []
    for (z, w) in arms:
        mb.box((0.09, w * 2, 0.09), (x, y, z), m_steel)
        for dy_ in (-w + 0.1, 0.0, w - 0.1):
            mb.cyl(0.035, 0.14, (x, y + dy_, z + 0.045), m_whiteg, segs=6)
            tops.append(Vector((x, y + dy_, z + 0.19)))
    for k in range(6):
        mb.rod((x - 0.17, y, 1.8 + k * 0.42), (x + 0.17, y, 1.8 + k * 0.42), 0.012, m_steel, segs=4)   # climbing pegs
    if transformer:
        for dy_ in (-0.42, 0.42):
            mb.cyl(0.27, 0.8, (x, y + dy_, 7.2), m_pipe, segs=12)
            mb.cyl(0.05, 0.16, (x, y + dy_, 8.0), m_whiteg, segs=6)
        mb.box((0.5, 1.2, 0.08), (x, y, 7.16), m_steel)
    mb.box((0.02, 0.26, 0.5), (x - math.copysign(0.165, x), y, 2.2), m_white)                  # notice plate
    return tops


up = MB("UtilityPoles")
PX = RW + 0.38
T = {}
for nm, (x, y, tr) in {"L0": (-PX, 1.5, True), "R0": (PX, -7.0, False), "R1": (PX, 11.0, True), "L2": (-PX, 60.0, False),
                       "R2": (PX, 53.5, True)}.items():
    T[nm] = utility_pole(up, x, y, transformer=tr)
up.finish(col=STREET, sharp=40)
wr = MB("Wires")
for a_, b_ in (("L0", "L2"), ("R0", "R1"), ("R1", "R2")):
    for k in range(6):
        cable(wr, T[a_][k], T[b_][k], (0.5 if a_ != "L0" else 1.1) + 0.1 * (k % 3), r=0.011, n=22)
for a_, b_, ks in (("R0", "L0", (2, 4)),):
    for k in ks:
        cable(wr, T[a_][k], T[b_][k], 0.75, r=0.01)
for (p, q, sg) in ((T["L0"][3], (-FRONT + 0.05, 2.4, 8.4), 0.3), (T["L0"][5], (-FRONT + 0.05, 5.4, 9.3), 0.35), (T["R1"][3], (FRONT - 0.05, 8.2, 7.2), 0.4),
                   (T["R1"][4], (FRONT - 0.05, 14.5, 6.8), 0.4)):
    cable(wr, p, q, sg, r=0.009)
wr.finish(col=STREET)
sl = MB("StringLights")
bulb_w = [neon("amber", 7.0), neon("white", 7.0)]
bulb_c = [neon("pink", 6.0), neon("amber", 6.0), neon("cyan", 6.0)]
cable(sl, (-FRONT + 0.05, 18.4, 7.1), (FRONT - 0.05, 19.4, 7.3), 1.3, r=0.008, bulbs=bulb_w, n=20)
cable(sl, (-FRONT + 0.05, 23.6, 7.6), (FRONT - 0.05, 22.6, 7.2), 1.2, r=0.008, bulbs=bulb_c, n=20)
sl.finish(col=STREET)
for (yy, colr) in ((18.9, (1.0, 0.6, 0.3)), (23.1, (1.0, 0.4, 0.6))):
    light(f"StringGlow_{yy}", "AREA", (0, yy, 5.9), 90 * K_SIGN, colr, size=12.0, size_y=0.4, rot=(0, 0, 0), col=LIGHTS, glossy=False)

# --- guard rails, bollards, small things
gr = MB("GuardRails")
for side in (-1, 1):
    x = side * (RW + 0.28)
    for (ya, yb) in ((-14.0, -2.5), (-0.5, 9.0), (11.0, 16.2), (18.2, 24.8))[(1 if side < 0 else 0):]:
        n = max(1, round((yb - ya) / 2.0))
        for i in range(n + 1):
            y = ya + (yb - ya) * i / n
            gr.cyl(0.03, 0.82, (x, y, CURB), m_railp, segs=6)
        for z in (0.45, 0.8):
            gr.rod((x, ya, CURB + z), (x, yb, CURB + z), 0.024, m_railp, segs=6)
    for i in range(4):
        gr.cyl(0.07, 0.75, (side * (RW + 0.45 + i * 0.75), 31.4, CURB), m_steel, segs=8)
        gr.cyl(0.075, 0.07, (side * (RW + 0.45 + i * 0.75), 31.4, CURB + 0.55), neon("amber", 1.0), segs=8)
gr.finish(col=STREET)

sm_ = MB("StreetBits")
sm_.cyl(0.1, 0.5, (RW + 0.9, 12.6, CURB), m_red, segs=10)                                    # hydrant
sm_.sphere(0.115, (RW + 0.9, 12.6, CURB + 0.52), m_red, u=8, v=6)
sm_.cyl(0.035, 0.3, (RW + 0.9, 12.6, CURB + 0.3), m_red, rot=(math.pi / 2, 0, 0), segs=6, base=False)
sm_.box((0.42, 0.34, 1.05), (-(RW + 2.9), 12.0, CURB + 0.525), m_red)                        # post box
sm_.box((0.46, 0.38, 0.05), (-(RW + 2.9), 12.0, CURB + 1.07), m_red)
sm_.box((0.28, 0.02, 0.04), (-(RW + 2.9) + 0.0, 12.0 - 0.175, CURB + 0.82), m_black)
p_aframe(sm_, lw(-1, 6.9, -1.75, CURB), heading=1.75, face=lightbox_mat((1.0, 0.8, 0.5), 0.8))
for k, (cx_, cy_, cz_, mm) in enumerate(((-8.15, 9.7, 0.0, m_yellow), (-8.15, 9.7, 0.3, m_yellow), (-8.15, 9.7, 0.6, m_blue), (-7.75, 9.75, 0.0, m_red))):
    sm_.box((0.36, 0.5, 0.29), (cx_, cy_, CURB + 0.145 + cz_), mm)                           # crates by the ramen shop
sm_.cyl(0.05, 2.6, (RW + 0.5, 4.5, CURB), m_steel, segs=8)                                   # bus-stop style sign
sm_.cyl(0.26, 0.03, (RW + 0.5, 4.47, CURB + 2.5), m_blue, rot=(math.pi / 2, 0, 0), segs=16, base=False)
sm_.box((0.36, 0.03, 0.5), (RW + 0.5, 4.46, CURB + 1.75), lightbox_mat((0.9, 0.9, 0.8), 0.5))
sm_.box((0.36, 0.034, 0.1), (RW + 0.5, 4.46, CURB + 1.94), m_blue)
for k in range(5):
    sm_.box((0.28, 0.034, 0.012), (RW + 0.5, 4.458, CURB + 1.58 + k * 0.06), m_black)
sm_.finish(col=STREET, sharp=40, bevel=0.008)


def bicycle(mb, pos, heading=0.0, frame=None):
    """Parked city bike: two wheels, diamond frame, bars, saddle, basket."""
    frame = frame or m_teal
    M0 = mb.M.copy()
    mb.M = M0 @ Matrix.Translation(Vector(pos)) @ Matrix.Rotation(heading, 4, "Z") @ Matrix.Rotation(math.radians(7), 4, "Y")
    R = 0.33
    for y in (-0.52, 0.52):
        mb.tube([(0, y + R * math.cos(2 * math.pi * i / 20), R + R * math.sin(2 * math.pi * i / 20)) for i in range(20)], 0.02, m_tire, segs=5, closed=True)
        for i in range(8):
            a = math.pi * i / 8
            mb.rod((0, y - R * math.cos(a), R - R * math.sin(a)), (0, y + R * math.cos(a), R + R * math.sin(a)), 0.003, m_steel, segs=3)
    bb, seat, head = Vector((0, -0.05, 0.3)), Vector((0, -0.2, 0.86)), Vector((0, 0.38, 0.9))
    for a_, b_ in ((bb, seat), (bb, head), (seat, head), (seat, Vector((0, -0.52, R))), (bb, Vector((0, -0.52, R))), (head, Vector((0, 0.52, R)))):
        mb.rod(a_, b_, 0.016, frame, segs=6)
    mb.rod(head, head + Vector((0, -0.04, 0.2)), 0.014, m_steel, segs=5)
    mb.rod(head + Vector((-0.26, -0.06, 0.2)), head + Vector((0.26, -0.06, 0.2)), 0.012, m_steel, segs=5)
    mb.sphere(1.0, seat + Vector((0, -0.03, 0.06)), m_black, scale=(0.08, 0.13, 0.035), u=8, v=6)
    mb.box((0.3, 0.24, 0.2), head + Vector((0, 0.22, 0.02)), m_grate)
    mb.rod(Vector((0.06, -0.1, 0.28)), Vector((0.2, -0.16, 0.0)), 0.008, m_steel, segs=4)       # stand
    mb.M = M0


bk = MB("Bicycles")
bicycle(bk, lw(-1, 11.9, -0.55, CURB), heading=0.1, frame=m_teal)
bicycle(bk, lw(-1, 12.3, -0.95, CURB), heading=-0.05, frame=m_red)
bicycle(bk, lw(1, 15.9, -0.6, CURB), heading=3.2, frame=m_whiteg)
bicycle(bk, lw(-1, 20.4, -0.5, CURB), heading=0.2, frame=m_yellow)
bk.finish(col=STREET, sharp=45)

tick("p55_rail")
# ================================================================ ELEVATED RAIL + TRAIN (crosses the street at the junction)
RAIL = group("Rail")
m_viaduct = make_wall("viaduct_concrete", (0.12, 0.12, 0.118), rough=0.6, streak=0.6)
m_train = mat("train_steel", (0.42, 0.44, 0.46), rough=0.28, metal=0.85)
m_train_band = mat("train_band", (0.02, 0.32, 0.30), rough=0.3, coat=0.5)
m_train_in = mat("train_interior", (0.36, 0.35, 0.31), rough=0.5)
m_train_seat = mat("train_seat", (0.03, 0.2, 0.26), rough=0.7, sheen=0.4)


def sheet_holes(mb, O, U, V, w, h, holes, m):
    """Flat sheet O + u*U + v*V (0..w, 0..h) with rectangular holes (u0, v0, u1, v1)."""
    O, U, V = Vector(O), Vector(U), Vector(V)
    rd = lambda v: round(v, 4)
    us = sorted({0.0, rd(w)} | {rd(v) for hh in holes for v in (hh[0], hh[2]) if 0 < v < w})
    vs = sorted({0.0, rd(h)} | {rd(v) for hh in holes for v in (hh[1], hh[3]) if 0 < v < h})
    for j in range(len(vs) - 1):
        vc = (vs[j] + vs[j + 1]) / 2
        run = None
        for i in range(len(us) - 1):
            uc = (us[i] + us[i + 1]) / 2
            solid = not any(hh[0] < uc < hh[2] and hh[1] < vc < hh[3] for hh in holes)
            if solid and run is None:
                run = us[i]
            if run is not None and (not solid or i == len(us) - 2):
                end = us[i + 1] if solid else us[i]
                mb.poly([O + U * run + V * vs[j], O + U * end + V * vs[j], O + U * end + V * vs[j + 1], O + U * run + V * vs[j + 1]], m)
                run = None


rv = MB("Viaduct")
Y0, Y1 = RAIL_Y - 4.2, RAIL_Y + 4.2
rv.bx((-110, Y0, RAIL_Z - 1.25), (110, Y1, RAIL_Z), m_viaduct)                      # deck slab / girder
rv.bx((-110, RAIL_Y - 2.7, RAIL_Z - 1.95), (110, RAIL_Y + 2.7, RAIL_Z - 1.25), m_viaduct)
for yy in (Y0, Y1 - 0.22):
    rv.bx((-110, yy, RAIL_Z), (110, yy + 0.22, RAIL_Z + 1.05), m_viaduct)           # parapets
    rv.bx((-110, yy - 0.04, RAIL_Z + 1.05), (110, yy + 0.26, RAIL_Z + 1.12), m_conc_d)
rv.rod((-110, Y0 - 0.1, RAIL_Z + 0.55), (110, Y0 - 0.1, RAIL_Z + 0.55), 0.06, m_pipe, segs=6)      # cable duct on the near parapet
for k in range(-12, 13):
    rv.box((0.06, 0.16, 0.2), (k * 8.8 + 1.0, Y0 - 0.1, RAIL_Z + 0.5), m_steel)
    rv.bx((k * 8.8 - 0.02, Y0 - 0.02, RAIL_Z - 1.25), (k * 8.8 + 0.02, Y0, RAIL_Z + 1.05), m_conc_d)   # joints
PIERS = (-83.0, -64.5, -46.0, -27.5, -9.6, 9.6, 27.5, 46.0, 64.5, 83.0)
for px_ in PIERS:
    rv.bx((px_ - 0.85, RAIL_Y - 1.05, 0.0), (px_ + 0.85, RAIL_Y + 1.05, RAIL_Z - 2.75), m_viaduct)
    rv.bx((px_ - 1.1, RAIL_Y - 3.9, RAIL_Z - 2.75), (px_ + 1.1, RAIL_Y + 3.9, RAIL_Z - 1.95), m_viaduct)
    rv.bx((px_ - 1.5, RAIL_Y - 1.7, 0.0), (px_ + 1.5, RAIL_Y + 1.7, 0.2), m_curb)
    rv.cyl(0.06, RAIL_Z - 2.0, (px_ + 0.9, RAIL_Y - 1.1, 0.2), m_pipe, segs=6)
    for k in range(4):                                                              # hazard stripes on the pier base
        rv.bx((px_ - 0.86, RAIL_Y - 1.06, 0.3 + k * 0.4), (px_ + 0.86, RAIL_Y + 1.06, 0.5 + k * 0.4), m_yellow if k % 2 == 0 else m_black)
n = 16
for k in range(n):                                                                  # clearance chevrons over the road
    x0 = -RW - 1.0 + (2 * RW + 2.0) * k / n
    rv.bx((x0, Y0 - 0.015, RAIL_Z - 1.25), (x0 + (2 * RW + 2.0) / n, Y0, RAIL_Z - 0.95), m_yellow if k % 2 == 0 else m_black)
rv.cyl(0.42, 0.03, (0.0, Y0 - 0.03, RAIL_Z - 0.4), m_white, rot=(math.pi / 2, 0, 0), segs=20, base=False)
rv.cyl(0.36, 0.035, (0.0, Y0 - 0.03, RAIL_Z - 0.4), m_red, rot=(math.pi / 2, 0, 0), segs=20, base=False)
rv.cyl(0.29, 0.04, (0.0, Y0 - 0.03, RAIL_Z - 0.4), m_white, rot=(math.pi / 2, 0, 0), segs=20, base=False)
for xx in (-6.5, 6.5):                                                              # catenary portals
    for yy in (Y0 + 0.11, Y1 - 0.11):
        rv.cyl(0.11, 6.6, (xx, yy, RAIL_Z + 1.05), m_steel, segs=8)
    rv.bx((xx - 0.1, Y0, RAIL_Z + 7.2), (xx + 0.1, Y1, RAIL_Z + 7.45), m_steel)
    for yy in (RAIL_Y - 1.75, RAIL_Y + 1.75):
        rv.rod((xx, yy, RAIL_Z + 7.2), (xx, yy, RAIL_Z + 6.2), 0.025, m_steel, segs=5)
        rv.cyl(0.05, 0.25, (xx, yy, RAIL_Z + 6.0), m_whiteg, segs=6)
for yy in (RAIL_Y - 1.75, RAIL_Y + 1.75):
    rv.rod((-110, yy, RAIL_Z + 5.45), (110, yy, RAIL_Z + 5.45), 0.012, m_black, segs=4)
    for (xa, xb) in ((-110, -6.5), (-6.5, 6.5), (6.5, 110)):
        pts = [(xa + (xb - xa) * i / 10, yy, RAIL_Z + 6.0 - (0.45 if xb - xa < 20 else 2.0) * 4 * (i / 10) * (1 - i / 10) * 0.5) for i in range(11)]
        rv.tube(pts, 0.012, m_black, segs=4, caps=False)
# fluorescent tubes under the deck
for xx in (-4.4, 0.0, 4.4):
    for yy in (RAIL_Y - 1.6, RAIL_Y + 1.6):
        rv.rod((xx - 0.6, yy, RAIL_Z - 2.02), (xx + 0.6, yy, RAIL_Z - 2.02), 0.02, neon("ice", 10.0), segs=6)
        rv.box((1.4, 0.12, 0.05), (xx, yy, RAIL_Z - 1.97), m_alu)
rv.finish(col=RAIL, sharp=40)
text("Clearance_txt", "3.8m", None, m_black, size=0.26, extrude=0.002, matrix=basis((1, 0, 0), (0, -1, 0), (0.0, Y0 - 0.055, RAIL_Z - 0.4)) @ Matrix.Rotation(math.pi / 2, 4, "X"), col=RAIL)
light("UnderDeck", "AREA", (0, RAIL_Y, RAIL_Z - 2.1), 330 * K_LAMP, (0.72, 1.0, 0.86), size=9.0, size_y=3.5, rot=(0, 0, 0), col=LIGHTS, glossy=False)
light("UnderDeck_L", "AREA", (-14, RAIL_Y, RAIL_Z - 2.1), 200 * K_LAMP, (0.72, 1.0, 0.86), size=6.0, size_y=3.5, rot=(0, 0, 0), col=LIGHTS, glossy=False)
light("UnderDeck_R", "AREA", (14, RAIL_Y, RAIL_Z - 2.1), 200 * K_LAMP, (0.72, 1.0, 0.86), size=6.0, size_y=3.5, rot=(0, 0, 0), col=LIGHTS, glossy=False)


def train_car(name, x0, yc, rng, L=19.4, Wd=2.9):
    mb = MB(name)
    yb = yc - Wd / 2                 # near side (faces the camera)
    zr = RAIL_Z + 0.18               # rail top
    zb, zfl, zw0, zw1, zt = zr + 0.75, zr + 0.95, zr + 1.78, zr + 2.72, zr + 3.3
    # near side with openings
    holes = []
    doors = (2.55, 7.35, 12.05, 16.85)
    for dc in doors:
        for sg in (-1, 1):
            holes.append((dc + sg * 0.33 - 0.2, zw0 - zb + 0.05, dc + sg * 0.33 + 0.2, zw1 - zb))
    for (a, b) in ((0.45, 1.6), (3.5, 6.4), (8.3, 11.1), (13.0, 15.9), (17.8, 18.95)):
        holes.append((a, zw0 - zb, b, zw1 - zb))
    sheet_holes(mb, (x0, yb, zb), (1, 0, 0), (0, 0, 1), L, zt - zb, holes, m_train)
    for (a, v0, b, v1) in holes:
        mb.poly([(x0 + a, yb + 0.035, zb + v0), (x0 + b, yb + 0.035, zb + v0), (x0 + b, yb + 0.035, zb + v1), (x0 + a, yb + 0.035, zb + v1)], m_glass)
        for (p, q) in (((a, v0), (b, v0)), ((a, v1), (b, v1)), ((a, v0), (a, v1)), ((b, v0), (b, v1))):
            mb.rod((x0 + p[0], yb + 0.01, zb + p[1]), (x0 + q[0], yb + 0.01, zb + q[1]), 0.02, m_black, segs=4)
        if b - a > 2:
            mb.box((0.05, 0.05, v1 - v0), (x0 + (a + b) / 2, yb + 0.02, zb + (v0 + v1) / 2), m_black)
    for dc in doors:                                                                # door outlines
        for dxx in (-0.66, 0.0, 0.66):
            mb.box((0.025, 0.02, zt - zb - 0.45), (x0 + dc + dxx, yb - 0.005, zb + (zt - zb - 0.45) / 2 + 0.08), m_black)
        mb.box((1.36, 0.02, 0.03), (x0 + dc, yb - 0.005, zt - 0.37), m_black)
    mb.bx((x0, yb - 0.012, zw0 - 0.5), (x0 + L, yb, zw0 - 0.22), m_train_band)       # colour band
    mb.bx((x0, yb - 0.012, zw0 - 0.17), (x0 + L, yb, zw0 - 0.13), neon("amber", 0.5))
    mb.bx((x0 + 6.75, yb - 0.02, zw1 + 0.12), (x0 + 7.95, yb, zw1 + 0.36), neon("amber", 3.0))   # destination display
    # shell: roof, ends, floor, far side, underframe
    mb.bx((x0, yb, zt), (x0 + L, yb + Wd, zt + 0.12), m_train)
    mb.bx((x0 + 0.15, yb + 0.25, zt + 0.12), (x0 + L - 0.15, yb + Wd - 0.25, zt + 0.32), m_train)
    for (a, b) in ((3.0, 6.0), (13.0, 16.0)):
        mb.bx((x0 + a, yb + 0.5, zt + 0.32), (x0 + b, yb + Wd - 0.5, zt + 0.62), m_alu_l)
    for xe in (x0, x0 + L - 0.06):
        mb.bx((xe, yb, zb), (xe + 0.06, yb + Wd, zt), m_train)
    mb.bx((x0, yb, zb - 0.1), (x0 + L, yb + Wd, zfl), m_alu)
    mb.bx((x0, yb + Wd - 0.06, zfl), (x0 + L, yb + Wd, zt), m_train_in)
    mb.bx((x0 + 1.0, yb + 0.3, zr + 0.05), (x0 + L - 1.0, yb + Wd - 0.3, zb - 0.1), m_black)
    # interior: far wall windows (night outside), ads, bench, ceiling light, poles, passengers
    for (a, b) in ((0.45, 1.6), (3.5, 6.4), (8.3, 11.1), (13.0, 15.9), (17.8, 18.95)):
        mb.bx((x0 + a, yb + Wd - 0.075, zw0), (x0 + b, yb + Wd - 0.06, zw1), m_blackg)
        mb.bx((x0 + a + 0.1, yb + Wd - 0.5, zfl), (x0 + b - 0.1, yb + Wd - 0.08, zfl + 0.44), m_train_seat)
        mb.bx((x0 + a + 0.1, yb + Wd - 0.16, zfl + 0.44), (x0 + b - 0.1, yb + Wd - 0.08, zfl + 0.85), m_train_seat)
        for k in range(int((b - a) / 0.62)):
            mb.bx((x0 + a + 0.08 + k * 0.62, yb + Wd - 0.075, zw1 + 0.1), (x0 + a + 0.6 + k * 0.62, yb + Wd - 0.06, zw1 + 0.38), rng.choice(PRODUCT))
    for dc in doors:
        mb.bx((x0 + dc - 0.62, yb + Wd - 0.075, zfl + 0.05), (x0 + dc + 0.62, yb + Wd - 0.06, zt - 0.45), m_alu_l)
        for sg in (-1, 1):
            mb.bx((x0 + dc + sg * 0.33 - 0.2, yb + Wd - 0.085, zw0 + 0.05), (x0 + dc + sg * 0.33 + 0.2, yb + Wd - 0.07, zw1), m_blackg)
            mb.cyl(0.018, zt - zfl - 0.1, (x0 + dc + sg * 0.72, yb + Wd / 2 + sg * 0.0, zfl), m_chrome, segs=5)
    for yy in (yb + 0.8, yb + Wd - 0.8):
        mb.bx((x0 + 0.4, yy - 0.07, zt - 0.045), (x0 + L - 0.4, yy + 0.07, zt - 0.02), neon("ice", 11.0))
        mb.rod((x0 + 0.5, yy, zt - 0.5), (x0 + L - 0.5, yy, zt - 0.5), 0.012, m_chrome, segs=4)
        for k in range(int(L / 0.45) - 2):
            mb.rod((x0 + 0.7 + k * 0.45, yy, zt - 0.5), (x0 + 0.7 + k * 0.45, yy, zt - 0.68), 0.006, m_white, segs=3)
    M0 = mb.M.copy()
    for k in range(13):                                                             # passengers
        px_ = x0 + rng.uniform(0.8, L - 0.8)
        if rng.random() < 0.45:                                                     # seated on the far bench, facing us
            seat_ok = any(a + 0.3 < px_ - x0 < b - 0.3 for (a, b) in ((0.45, 1.6), (3.5, 6.4), (8.3, 11.1), (13.0, 15.9), (17.8, 18.95)))
            if not seat_ok:
                continue
            mb.M = M0 @ Matrix.Translation((px_, yb + Wd - 0.42, zfl + 0.44)) @ Matrix.Rotation(math.pi, 4, "Z")
            person_geo(mb, rng, pose="sit", hairstyle=rng.choice(("short", "bob", "long", "cap")))
        else:
            mb.M = M0 @ Matrix.Translation((px_, yb + rng.uniform(0.75, 1.9), zfl)) @ Matrix.Rotation(rng.choice((0.0, math.pi, math.pi / 2, -math.pi / 2)), 4, "Z")
            person_geo(mb, rng, pose="stand", h=rng.uniform(1.58, 1.8), hairstyle=rng.choice(("short", "bob", "long", "short")),
                       bag=rng.choice((None, "tote", "back", None)))
    mb.M = M0
    mb.finish(col=RAIL, sharp=40)
    light(name + "_in", "AREA", (x0 + L / 2, yc, zt - 0.1), 380 * K_SHOP, (0.8, 0.93, 1.0), size=L - 1.0, size_y=1.6, rot=(0, 0, 0), col=LIGHTS, glossy=False)
    light(name + "_glow", "AREA", (x0 + L / 2, yb - 0.6, zw0 + 0.4), 1100 * K_FOGL, (0.75, 0.92, 1.0), size=L - 2.0, size_y=0.8,
          target=(x0 + L / 2, yb - 10.0, zw0 - 3.0), col=LIGHTS, glossy=False, diffuse=False)


_rt = random.Random(1212)
train_car("Train_A", -16.6, RAIL_Y - 1.75, _rt)
train_car("Train_B", 3.4, RAIL_Y - 1.75, _rt)
tg = MB("Train_Gangway")
tg.bx((2.8, RAIL_Y - 1.75 - 1.2, RAIL_Z + 1.1), (3.4, RAIL_Y - 1.75 + 1.2, RAIL_Z + 3.3), m_rubber)
tg.finish(col=RAIL)

tick("p60_tower")
# ================================================================ HERO TOWER with holographic rings (end of the street)
TOWER = group("HeroTower")
TZ = [(0.0, 21.0, 5.0), (72.0, 19.5, 5.0), (72.01, 17.6, 4.6), (112.0, 15.8, 4.2), (112.01, 13.6, 3.8), (136.0, 12.2, 3.4),
      (136.01, 9.6, 3.0), (150.0, 8.4, 2.6), (150.01, 4.6, 1.5), (158.0, 3.0, 1.0)]


def tsec(z, hw, ch):
    return [Vector((x, TOWER_Y + y, z)) for (x, y) in ((hw - ch, -hw), (hw, -hw + ch), (hw, hw - ch), (hw - ch, hw), (-hw + ch, hw),
                                                      (-hw, hw - ch), (-hw, -hw + ch), (-hw + ch, -hw))]


def thw(z):
    for (z0, h0, c0), (z1, h1, c1) in zip(TZ, TZ[1:]):
        if z0 <= z <= z1 and z1 > z0:
            t = (z - z0) / (z1 - z0)
            return h0 + (h1 - h0) * t, c0 + (c1 - c0) * t
    return TZ[-1][1], TZ[-1][2]


tw = MB("HeroTower")
tw.loft([tsec(*s) for s in TZ], WIN_TOWER, smooth=False)
tw.finish(col=TOWER, sharp=None)
td = MB("HeroTower_Lights")
m_fin = neon("cyan", 4.5)
m_slit = neon("ice", 5.5)
m_crown = neon("ice", 7.0)
segz = [60.0, 72.0, 72.02, 112.0, 112.02, 136.0, 136.02, 150.0]
for za, zb in zip(segz, segz[1:]):
    if zb - za < 1:
        continue
    (ha, ca), (hb, cb) = thw(za + 0.02), thw(zb - 0.02)
    td.poly([(-1.0, TOWER_Y - ha - 0.3, za), (1.0, TOWER_Y - ha - 0.3, za), (0.9, TOWER_Y - hb - 0.3, zb), (-0.9, TOWER_Y - hb - 0.3, zb)], m_slit)   # light blade
    for sx_ in (-1, 1):                                                              # corner fins
        td.tube([(sx_ * (ha - ca * 0.5) , TOWER_Y - ha + ca * 0.5 - 0.5, za), (sx_ * (hb - cb * 0.5), TOWER_Y - hb + cb * 0.5 - 0.5, zb)], 0.32, m_fin, segs=5)
        td.tube([(sx_ * (ha - ca), TOWER_Y - ha - 0.25, za), (sx_ * (hb - cb), TOWER_Y - hb - 0.25, zb)], 0.16, m_fin, segs=4)
for zz in (72.0, 112.0, 136.0, 150.0):                                              # set-back bands
    hw, ch = thw(zz - 0.5)
    ring = tsec(zz + 0.4, hw + 0.25, ch)
    td.tube(ring, 0.42, m_crown if zz > 140 else neon("ice", 3.5), segs=5, closed=True)
for k in range(6):                                                                  # crown louvres
    zz = 137.5 + k * 2.1
    hw, ch = thw(zz)
    td.tube(tsec(zz, hw + 0.15, ch), 0.18, m_crown, segs=4, closed=True)
td.rod((0, TOWER_Y, 158.0), (0, TOWER_Y, 170.0), 0.7, m_steel, segs=6, r2=0.12)
for zz in (160.0, 165.0, 170.0):
    td.sphere(0.9, (0, TOWER_Y, zz), neon("red", 12.0), u=8, v=6)
td.finish(col=TOWER, sharp=None)

RINGS = [  # z, radius, band height, tilt x, tilt y (deg), colour, strength
    (97.0, 31.0, 5.0, 7.0, -4.0, "cyan", 3.4), (122.0, 25.0, 4.2, -9.0, 5.0, "magenta", 3.6), (144.5, 17.5, 3.2, 5.0, 9.0, "amber", 3.4)]
for k, (zz, R, bh, tx_, ty_, key, st) in enumerate(RINGS):
    rg = MB(f"HoloRing_{k}")
    rg.lathe([(R, -bh / 2), (R, bh / 2)], (0, 0, 0), make_holo(f"holo_{k}", NEON[key], st), segs=128)
    for dz in (-bh / 2 - 0.5, bh / 2 + 0.5):
        rg.tube([(R * math.cos(2 * math.pi * i / 96), R * math.sin(2 * math.pi * i / 96), dz) for i in range(96)], 0.2, neon(key, st + 1.5), segs=4, closed=True)
    rg.tube([((R + 2.4) * math.cos(2 * math.pi * i / 96), (R + 2.4) * math.sin(2 * math.pi * i / 96), 0.0) for i in range(96)], 0.09, neon("ice", 3.0), segs=4, closed=True)
    ob = rg.finish(col=TOWER, loc=(0, TOWER_Y, zz), rot=(math.radians(tx_), math.radians(ty_), 0.0), sharp=None)
    ob.visible_shadow = False
    light(f"RingGlow_{k}", "POINT", (0, TOWER_Y - R - 2, zz), 45000 * K_FOGL, NEON[key], size=3.0, col=LIGHTS, glossy=False, diffuse=False)
light("CrownGlow", "POINT", (0, TOWER_Y - 14, 150.0), 90000 * K_FOGL, NEON["ice"], size=4.0, col=LIGHTS, glossy=False, diffuse=False)
light("TowerWash", "AREA", (0, TOWER_Y - 70, 40.0), 2.2e5, (0.25, 0.55, 0.9), size=40.0, target=(0, TOWER_Y, 115.0), col=LIGHTS, glossy=False, volume=False)
for k, (tx_, tz_) in enumerate(((-260.0, 420.0), (300.0, 380.0))):                  # search beams from the crown
    light(f"Beam_{k}", "SPOT", (0, TOWER_Y - 6, 152.0), 1.2e8, (0.65, 0.85, 1.0), size=0.6, target=(tx_, TOWER_Y - 120, tz_), spot=3.2, blend=0.3,
          col=LIGHTS, glossy=False, diffuse=False)
# podium (seen under the rail deck, far down the street)
pd = MB("Tower_Podium")
pd.bx((-34, TOWER_Y - 34, 0.2), (34, TOWER_Y + 34, 13.0), WIN_MATS[2])
pd.bx((-12, TOWER_Y - 34.3, 0.2), (12, TOWER_Y - 34.0, 11.0), m_scr_wave)
pd.finish(col=TOWER)
light("PodiumGlow", "POINT", (0, TOWER_Y - 40, 7.0), 30000 * K_FOGL, NEON["teal"], size=3.0, col=LIGHTS, glossy=False, diffuse=False)

# ================================================================ FLYING VEHICLES with light trails
FLY = group("Flyers")
m_flybody = mat("flyer_body", (0.05, 0.055, 0.06), rough=0.25, metal=0.8, coat=0.5)
m_trail_r = make_trail("trail_red", (1.0, 0.08, 0.04), 5.0)
m_trail_c = make_trail("trail_cyan", (0.2, 0.85, 1.0), 4.0)
m_trail_w = make_trail("trail_white", (1.0, 0.92, 0.8), 5.0)


def flyer(name, pos, heading, scale=1.0, trail=18.0, tcol=None):
    """Pod with canopy, side nacelles glowing underneath, head / tail lights and a fading trail behind it."""
    mb = MB(name, uv=True)
    secs = []
    for (y, w, zc, hz) in ((-2.3, 0.25, 0.22, 0.12), (-1.8, 0.75, 0.18, 0.37), (-0.5, 1.0, 0.18, 0.54), (0.9, 0.95, 0.18, 0.52), (1.9, 0.6, 0.11, 0.31),
                           (2.4, 0.2, 0.1, 0.1)):
        secs.append([Vector((w * math.cos(2 * math.pi * i / 10), y, zc + hz * math.sin(2 * math.pi * i / 10))) for i in range(10)])
    mb.loft(secs, m_flybody, mfun=lambda i, k: m_carglass if (i in (2, 3) and k in (1, 2, 3)) else m_flybody)
    for sx_ in (-1, 1):
        mb.cyl(0.32, 1.7, (sx_ * 1.25, -0.2, -0.1), m_flybody, rot=(math.pi / 2, 0, 0), segs=10, base=False)
        mb.box((0.5, 0.25, 0.08), (sx_ * 0.9, -0.2, 0.0), m_flybody)
        mb.cyl(0.26, 0.03, (sx_ * 1.25, -0.2, -0.43), neon("cyan", 8.0), segs=10, base=False)
        mb.box((0.34, 0.06, 0.1), (sx_ * 0.55, 2.2, 0.12), neon("white", 30.0))
        mb.box((0.5, 0.06, 0.08), (sx_ * 0.5, -2.15, 0.25), neon("red", 14.0))
    tm = tcol or m_trail_r
    n = 24
    for sx_ in (-0.5, 0.5):
        for i in range(n):
            t0, t1 = i / n, (i + 1) / n
            mb.poly([(sx_ - 0.11, -2.2 - trail * t0, 0.25), (sx_ + 0.11, -2.2 - trail * t0, 0.25), (sx_ + 0.11, -2.2 - trail * t1, 0.25), (sx_ - 0.11, -2.2 - trail * t1, 0.25)],
                    tm, uvs=[(t0, 0), (t0, 1), (t1, 1), (t1, 0)])
            mb.poly([(sx_, -2.2 - trail * t0, 0.14), (sx_, -2.2 - trail * t0, 0.36), (sx_, -2.2 - trail * t1, 0.36), (sx_, -2.2 - trail * t1, 0.14)],
                    tm, uvs=[(t0, 0), (t0, 1), (t1, 1), (t1, 0)])
    ob = mb.finish(col=FLY, loc=pos, rot=(0, 0, math.radians(heading)), sharp=40, recalc=True)
    ob.scale = (scale, scale, scale)
    ob.visible_shadow = False
    return ob


FLYERS = [  # pos, heading (0 = flying +Y away from us, 180 = toward us, +-90 = crossing), scale, trail, trail material
    ((6.4, 58.0, 27.0), 0, 1.0, 9.0, m_trail_r), ((-6.6, 84.0, 33.0), 180, 1.0, 8.0, m_trail_w), ((6.9, 118.0, 46.0), 0, 1.1, 11.0, m_trail_r),
    ((-7.0, 150.0, 55.0), 180, 1.1, 11.0, m_trail_w), ((7.4, 205.0, 70.0), 0, 1.3, 14.0, m_trail_r), ((-7.4, 240.0, 82.0), 180, 1.4, 14.0, m_trail_c),
    ((-52.0, 262.0, 126.0), -90, 1.6, 22.0, m_trail_c), ((44.0, 276.0, 104.0), 90, 1.6, 22.0, m_trail_r), ((-58.0, 190.0, 92.0), -80, 1.5, 20.0, m_trail_w),
    ((-36.0, 140.0, 76.0), 105, 1.3, 16.0, m_trail_r)]
for k, (pos, hd, scl, tr, tm) in enumerate(FLYERS):
    flyer(f"Flyer_{k}", pos, hd, scl, tr, tm)
    light(f"FlyerGlow_{k}", "POINT", (pos[0], pos[1], pos[2] - 1.5), 2500 * K_FOGL * scl, NEON["cyan"], size=0.8, col=LIGHTS, glossy=False, diffuse=False)

tick("p65_actors")
# ================================================================ CARS
CARS = group("Cars")


def car_lights(name, x, y, heading, head=False, tail=True, brake=False, L=4.55, beam=900.0, shaft=5.0):
    """Real lights for a car: beams that show in the rain, a red pool behind."""
    a = math.radians(heading)
    fwd = Vector((-math.sin(a), math.cos(a), 0.0))
    right = Vector((math.cos(a), math.sin(a), 0.0))
    c = Vector((x, y, 0.0))
    if head:
        for sgn in (-1, 1):
            p = c + fwd * (L / 2 + 0.05) + right * sgn * 0.62 + Vector((0, 0, 0.68))
            light(f"{name}_beam{sgn}", "SPOT", p, beam * K_LAMP, (1.0, 0.94, 0.84), size=0.06, target=p + fwd * 14 + Vector((0, 0, -0.75)), spot=58, blend=0.6,
                  col=LIGHTS, glossy=False, volume=False)
            light(f"{name}_shaft{sgn}", "SPOT", p, beam * shaft * K_FOGL, (1.0, 0.94, 0.84), size=0.06, target=p + fwd * 14 + Vector((0, 0, -0.5)), spot=30, blend=0.9,
                  col=LIGHTS, glossy=False, diffuse=False)
    if tail:
        p = c - fwd * (L / 2 + 0.25) + Vector((0, 0, 0.8))
        light(f"{name}_tailglow", "AREA", p, (95 if brake else 40) * K_LAMP, (1.0, 0.05, 0.03), size=1.5, size_y=0.3, target=p - fwd * 4 + Vector((0, 0, -1.2)),
              col=LIGHTS, glossy=False)
        light(f"{name}_tailfog", "POINT", p - fwd * 0.3, (420 if brake else 160) * K_FOGL, (1.0, 0.05, 0.03), size=0.3, col=LIGHTS, glossy=False, diffuse=False)


# queue at the red light in our lane (we see their brake lights)
make_car("Car_Silver", "sedan", 1.72, 10.4, 1.0, "silver", lights="brake", col=CARS)
car_lights("Car_Silver", 1.72, 10.4, 1.0, brake=True)
make_car("Car_Kei", "kei", 1.85, 17.3, -1.5, "white", lights="brake", col=CARS)
car_lights("Car_Kei", 1.85, 17.3, -1.5, brake=True, L=3.4)
make_car("Car_Taxi", "sedan", 1.7, 23.7, 0.5, "taxi", taxi=True, lights="brake", col=CARS)
car_lights("Car_Taxi", 1.7, 23.7, 0.5, brake=True)
# cross traffic under the rail: headlights sweep across the junction
make_car("Car_Cross", "sedan", -4.6, 35.0, -90.0, "blue", lights="head", col=CARS)
car_lights("Car_Cross", -4.6, 35.0, -90.0, head=True, tail=True, beam=2200.0, shaft=55.0)
make_car("Car_Cross2", "van", 12.5, 39.2, 90.0, "white", lights="head", col=CARS)
car_lights("Car_Cross2", 12.5, 39.2, 90.0, head=True, tail=False, beam=1200.0, L=4.6, shaft=50.0)
# oncoming, waiting on the far side
make_car("Car_Oncoming", "sedan", -1.75, 49.6, 180.0, "grey", lights="head", col=CARS)
car_lights("Car_Oncoming", -1.75, 49.6, 180.0, head=True, tail=False, beam=1100.0)
make_car("Car_Oncoming2", "kei", -1.8, 55.8, 181.0, "red", lights="head", col=CARS)
car_lights("Car_Oncoming2", -1.8, 55.8, 181.0, head=True, tail=False, beam=700.0, L=3.4)
# parked van on the left kerb, farther up
make_car("Car_ParkedVan", "van", -3.95, 21.8, 180.0, "teal", lights="tail", col=CARS)
# traffic down the far street
for k, (x, y, hd, kind, paint) in enumerate(((1.7, 62.0, 0, "sedan", "black"), (1.8, 74.0, 0, "kei", "teal"), (-1.7, 70.0, 180, "sedan", "white"),
                                             (1.75, 92.0, 0, "sedan", "red"), (-1.8, 96.0, 180, "van", "silver"), (1.7, 118.0, 0, "kei", "white"),
                                             (-1.75, 128.0, 180, "sedan", "black"), (1.75, 150.0, 0, "sedan", "grey"), (-1.7, 166.0, 180, "kei", "blue"),
                                             (1.7, 190.0, 0, "sedan", "taxi"), (-1.75, 206.0, 180, "sedan", "silver"), (1.75, 228.0, 0, "van", "white"))):
    make_car(f"Car_Far{k}", kind, x, y, hd, paint, taxi=(paint == "taxi"), lights="head" if hd == 180 else "tail", col=CARS)
    if hd == 180:
        car_lights(f"Car_Far{k}", x, y, hd, head=True, tail=False, beam=800.0)

# ================================================================ PEOPLE
PEOPLE = group("People")
rp = random.Random(77)


def P(name, x, y, heading, **kw):
    return make_person("P_" + name, x, y, heading, rp, col=PEOPLE, **kw)


# right sidewalk (camera side of the mart)
P("clear_umbrella", 6.55, 3.4, 4, umb="clear", coat=COATS[7], hem=0.5, hairstyle="bob", bag="tote", phase=1.0, h=1.63)
P("toward_black", 6.5, 12.6, 178, umb="black", coat=COATS[0], hem=0.56, hairstyle="short", bag="brief", phase=2.2, h=1.78)
_, info = P("vending_phone", 7.1, 11.25, -60, pose="stand", coat=COATS[4], hem=0.84, hairstyle="hood", phone=True, bag="back", h=1.72)
light("PhoneGlow", "POINT", Vector((7.1, 11.25, CURB)) + Matrix.Rotation(math.radians(-60), 3, "Z") @ (info["hands"][0] + Vector((0, 0.0, 0.1))), 2.5,
      (0.7, 0.85, 1.0), size=0.05, col=LIGHTS, glossy=False)
P("couple_a", 6.3, 15.4, 2, umb="navy", coat=COATS[2], hem=0.58, hairstyle="short", phase=0.7, h=1.8)
P("couple_b", 6.78, 15.5, 2, coat=COATS[5], hem=0.5, hairstyle="long", phase=3.9, h=1.62, scarf=m_scarf[2])
P("laundry_wait", 7.7, 18.0, 95, pose="stand", coat=COATS[6], hem=0.86, hairstyle="cap", bag="shop", h=1.7)
P("hotel_guest", 6.2, 22.6, -90, pose="stand", umb="black", coat=COATS[3], hem=0.52, hairstyle="short", bag="brief", h=1.76)
P("corner_r1", 6.4, 30.4, 85, umb="red", coat=COATS[0], hem=0.52, hairstyle="bob", phase=0.4, h=1.64)
P("corner_r2", 7.4, 30.9, 92, umb="clear", coat=COATS[8], hem=0.86, hairstyle="short", phase=2.6, h=1.75, bag="back")
# left sidewalk
P("ramen_menu", -7.0, 6.6, 82, pose="stand", umb="yellow", coat=COATS[1], hem=0.54, hairstyle="short", h=1.74, scarf=m_scarf[0])
P("toward_red", -6.2, 10.2, 184, umb="red", coat=COATS[0], hem=0.5, hairstyle="long", bag="tote", phase=5.2, h=1.65)
P("arcade_teen1", -7.75, 16.2, -100, pose="stand", coat=COATS[9], hem=0.86, hairstyle="cap", h=1.66, bag="back")
P("arcade_teen2", -7.9, 17.3, -70, pose="stand", coat=COATS[5], hem=0.86, hairstyle="short", h=1.72, phone=True)
P("away_teal", -6.0, 19.4, -3, umb="teal", coat=COATS[6], hem=0.56, hairstyle="short", phase=1.2, h=1.77, bag="back")
P("cafe_pair1", -7.2, 23.6, 170, umb="white", coat=COATS[3], hem=0.5, hairstyle="bob", phase=0.5, h=1.62)
P("cafe_pair2", -6.75, 23.9, 172, coat=COATS[2], hem=0.84, hairstyle="short", phase=3.5, h=1.79)
P("corner_l1", -6.3, 30.6, -88, pose="stand", umb="black", coat=COATS[0], hem=0.55, hairstyle="short", h=1.75)
# on the crossing, silhouetted against the junction lights
P("cross_1", -2.9, 29.5, -90, umb="black", coat=COATS[0], hem=0.52, hairstyle="short", phase=0.9, h=1.78, z=0.0)
P("cross_2", -0.4, 30.5, 90, umb="red", coat=COATS[1], hem=0.5, hairstyle="bob", phase=2.5, h=1.63, bag="tote", z=0.0)
P("cross_3", 0.5, 30.2, 88, coat=COATS[4], hem=0.84, hairstyle="hood", phase=5.5, h=1.72, z=0.0)
P("cross_4", 3.4, 29.8, -92, umb="clear", coat=COATS[7], hem=0.52, hairstyle="long", phase=4.1, h=1.66, z=0.0)
# beyond the junction (small in frame)
for k in range(12):
    side = -1 if k % 2 else 1
    P(f"far_{k}", side * rp.uniform(5.8, 7.8), 46.5 + k * 3.7 + rp.uniform(-1, 1), rp.choice((0, 180)) + rp.uniform(-6, 6),
      umb=rp.choice(("black", "black", "clear", "navy", "red", None)), hem=rp.choice((0.52, 0.56, 0.86)), hairstyle=rp.choice(("short", "bob", "long")),
      phase=rp.uniform(0, 6.28), h=rp.uniform(1.58, 1.82))

tick("p80_atmo")
# ================================================================ ATMOSPHERE: rain, splashes, drips, steam, fog
ATMO = group("Atmosphere")
FALL = Vector((0.075, -0.02, -1.0)).normalized()


def dry(p):
    """True where rain cannot fall: inside buildings, under awnings, under the rail deck."""
    ax = abs(p.x)
    if ax > FRONT - 0.05 and (p.y < 29.0 or 45.0 < p.y < 133.0 or 141.0 < p.y < 252.0):
        return True
    if ax > FRONT - 1.3 and p.z < 3.4 and p.y < 29.0:
        return True
    if RAIL_Y - 4.3 < p.y < RAIL_Y + 4.3 and p.z < RAIL_Z + 4.2:
        return True
    return False


def streak(mb, p, ln, w, m=None):
    view = (p - CAM.loc).normalized()
    sv = FALL.cross(view)
    if sv.length < 1e-6:
        return
    sv = sv.normalized() * w
    a, b = p - FALL * ln / 2, p + FALL * ln / 2
    mb.poly([a - sv, a + sv, b + sv, b - sv], m or m_rain, uvs=[(0, 0), (1, 0), (1, 1), (0, 1)])


def rain_layer(name, n, d0, d1, lens, w0, wk, seed):
    """n streaks spread evenly through the part of the view volume where rain can fall."""
    rng = random.Random(seed)
    mb = MB(name, uv=True)
    made = tries = 0
    while made < n and tries < n * 12:
        tries += 1
        d = (d0 ** 3 + rng.random() * (d1 ** 3 - d0 ** 3)) ** (1 / 3)
        p = CAM.unproject(rng.uniform(-0.08, 1.08), rng.uniform(-0.1, 1.15), d)
        if p.z < 0.2 or dry(p):
            continue
        streak(mb, p, rng.uniform(*lens), w0 + wk * d)
        made += 1
    ob = mb.finish(col=ATMO, sharp=None)
    ob.visible_shadow = False
    return made


if not ARGS.norain:
    nr = rain_layer("Rain_Near", 260, 1.5, 6.0, (0.16, 0.3), 0.0009, 0.00022, 1)
    nr += rain_layer("Rain_Mid", 9000, 6.0, 30.0, (0.22, 0.4), 0.0014, 0.0002, 2)
    nr += rain_layer("Rain_Far", 7000, 30.0, 80.0, (0.4, 0.7), 0.0025, 0.0002, 3)
    # splashes where drops land: ring, jet and a few droplets
    rng = random.Random(4)
    sp = MB("Splashes")
    ns = 0
    for _ in range(1500):
        t = rng.random() ** 1.8
        y = CAM.loc.y + 4.0 + t * 34.0
        x = rng.uniform(-FRONT + 0.1, FRONT - 0.1)
        z = CURB if abs(x) > RW + 0.16 else 0.0
        p = Vector((x, y, z))
        if dry(p + Vector((0, 0, 0.5))):
            continue
        s = rng.uniform(0.45, 1.0)
        rr = 0.03 * s
        sp.tube([p + Vector((rr * math.cos(2 * math.pi * i / 8), rr * math.sin(2 * math.pi * i / 8), 0.004)) for i in range(8)], 0.0035 * s, m_water,
                segs=3, closed=True)
        sp.cyl(0.006 * s, rng.uniform(0.03, 0.07) * s, p, m_water, segs=4, r2=0.001)
        for _k in range(rng.randint(3, 5)):
            a = rng.uniform(0, 6.283)
            q = p + Vector((math.cos(a) * rr * rng.uniform(0.8, 1.8), math.sin(a) * rr * rng.uniform(0.8, 1.8), rng.uniform(0.015, 0.06) * s))
            sp.sphere(rng.uniform(0.003, 0.006) * s, q, m_water, u=5, v=3)
        ns += 1
    ob = sp.finish(col=ATMO, sharp=None)
    ob.visible_shadow = False
    # drips off eaves, awnings, canopies and sign boxes
    dr = MB("Drips", uv=True)
    EDGES = [(lw(-1, 2.2, -1.3, 3.22), lw(-1, 9.6, -1.3, 3.22)), (lw(-1, 20.7, -1.45, 2.7), lw(-1, 28.8, -1.45, 2.7)),
             (lw(1, 20.6, -2.66, 3.68), lw(1, 24.8, -2.66, 3.68)), (lw(1, 0.8, -0.26, 3.44), lw(1, 7.7, -0.26, 3.44)),
             (lw(-1, 12.9, -0.2, 3.72), lw(-1, 20.1, -0.2, 3.72)), (lw(1, 11.4, -0.26, 3.36), lw(1, 15.6, -0.26, 3.36))]
    nd = 0
    for (a_, b_) in EDGES:
        ln = (b_ - a_).length
        t = 0.0
        while t < ln:
            base = a_.lerp(b_, t / ln)
            for _k in range(rng.randint(1, 3)):
                drop = rng.uniform(0.0, base.z - CURB - 0.3)
                streak(dr, base - Vector((0, 0, drop + 0.1)), rng.uniform(0.2, 0.55), 0.005, m_drip)
                nd += 1
            dr.sphere(0.008, base - Vector((0, 0, 0.012)), m_water, u=5, v=4, scale=(1, 1, 1.8))
            t += rng.uniform(0.1, 0.32)
    ob = dr.finish(col=ATMO, sharp=None)
    ob.visible_shadow = False
    print(f"[s3] rain streaks {nr}, splashes {ns}, drips {nd}")

if not ARGS.nofog:
    K_FOG = 1.0
    for nm, z1, dens in (("Fog_Low", 7.0, 0.0022), ("Fog_Mid", 20.0, 0.0009), ("Fog_High", 320.0, 0.00028)):
        fb_ = MB(nm)
        fb_.bx((-320, -80, -0.6), (320, 620, z1), make_fog(nm + "_mat", dens * K_FOG))
        ob = fb_.finish(col=ATMO)
        ob.display_type = "WIRE"
    st = MB("ManholeSteam")
    st.bx((-0.9, -0.9, -0.15), (3.4, 0.9, 4.2), make_steam())
    ob = st.finish(col=ATMO, loc=(STEAM_AT[0], STEAM_AT[1], 0.05))
    ob.display_type = "WIRE"

# off-screen "signs" behind the camera: coloured fill on the foreground
light("Fill_Magenta", "AREA", (-7.6, -14.0, 5.5), 420 * K_SIGN, NEON["pink"], size=5.0, target=(-1.0, 3.0, 0.0), col=LIGHTS, glossy=False, volume=False)
light("Fill_Cyan", "AREA", (7.6, -13.0, 6.0), 380 * K_SIGN, NEON["cyan"], size=5.0, target=(2.0, 5.0, 0.0), col=LIGHTS, glossy=False, volume=False)

tick("p90_render")
# ================================================================ CAMERAS, OUTPUT
use(bpy.data.collections["Rig"])
DEBUG_CAMS = {  # name: (position, target, lens)
    "left": ((-2.2, 0.5, 1.6), (-8.6, 7.0, 2.2), 22), "right": ((1.5, -1.0, 1.6), (8.6, 5.5, 2.0), 22),
    "car": ((-1.2, 5.6, 1.25), (1.72, 10.6, 0.75), 32), "people": ((3.2, -0.5, 1.45), (6.7, 5.0, 1.15), 40),
    "rail": ((-1.0, 12.0, 1.8), (0.0, 34.0, 8.6), 35), "tower": ((0.0, 40.0, 30.0), (0.0, TOWER_Y, 110.0), 70),
    "top": ((0.0, 8.0, 85.0), (0.0, 20.0, 0.0), 24), "arcade": ((-2.0, 12.0, 1.6), (-8.6, 16.6, 2.0), 22),
    "back": ((0.0, 26.0, 2.0), (0.0, 0.0, 1.5), 24), "cars3": ((6.5, 1.0, 2.2), (1.8, 15.0, 0.6), 35),
    "junction": ((0.0, 14.0, 1.6), (0.0, 36.0, 3.0), 30),
}
if ARGS.cam != "main":
    pos, tgt, lens = DEBUG_CAMS[ARGS.cam]
    dc = Cam("Cam_" + ARGS.cam, pos, tgt, lens=lens)
    sc.camera = dc.ob
setup_compositor(bloom=0.45, bloom_size=0.5, threshold=1.0, vignette=0.32, pct=ARGS.pct, power=1.16, slope=1.0)
if ARGS.exposure is not None:
    sc.view_settings.exposure = ARGS.exposure
if ARGS.region:
    x0, y0, x1, y1 = [float(v) for v in ARGS.region.split(",")]
    sc.render.use_border = True
    sc.render.use_crop_to_border = True
    sc.render.border_min_x, sc.render.border_min_y, sc.render.border_max_x, sc.render.border_max_y = x0, y0, x1, y1

n_obj = len(sc.objects)
n_light = sum(1 for o in sc.objects if o.type == "LIGHT")
n_poly = sum(len(o.data.polygons) for o in sc.objects if o.type == "MESH")
print(f"[s3] built in {time.time() - T0:.1f}s: {n_obj} objects ({n_light} lights), {n_poly} mesh faces, {len(bpy.data.materials)} materials")
result = {"objects": n_obj, "lights": n_light, "faces": n_poly}
if ARGS.save:
    blend = os.path.join(HERE, "scene3_neon_city_v2.blend")
    sc.render.filepath = "//S3_NeonCity_v2.png"
    bpy.ops.wm.save_as_mainfile(filepath=blend, compress=True)
    print("[s3] saved", blend)
if not ARGS.norender:
    sc.render.filepath = os.path.join(HERE, ARGS.out)
    t1 = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"[s3] rendered {ARGS.out} ({sc.render.resolution_x * ARGS.pct // 100}x{sc.render.resolution_y * ARGS.pct // 100}, {sc.cycles.samples} spp) in {time.time() - t1:.1f}s")
