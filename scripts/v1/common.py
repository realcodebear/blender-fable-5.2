"""Shared helpers for the three scenes. Run inside Blender (exec'd by scene files)."""
import bpy, bmesh, math, random
from mathutils import Vector, Matrix, Euler, noise

S = {"scene": None, "col": None}


# ---------- scene / collections ----------
def new_scene(name):
    old = bpy.data.scenes.get(name)
    if old:
        for o in list(old.collection.all_objects):
            bpy.data.objects.remove(o, do_unlink=True)
        for c in list(old.collection.children_recursive):
            bpy.data.collections.remove(c)
        if len(bpy.data.scenes) > 1:
            bpy.data.scenes.remove(old)
            old = None
    sc = old or bpy.data.scenes.new(name)
    sc.name = name
    S["scene"] = sc
    S["col"] = sc.collection
    return sc


def group(name):
    c = bpy.data.collections.new(name)
    S["scene"].collection.children.link(c)
    S["col"] = c
    return c


# ---------- materials ----------
def _principled(m):
    try:
        m.use_nodes = True
    except Exception:
        pass
    nt = m.node_tree
    for n in nt.nodes:
        if n.type == "BSDF_PRINCIPLED":
            return nt, n
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    n = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(n.outputs[0], out.inputs[0])
    return nt, n


def mat(name, color, rough=0.5, metal=0.0, emit=None, emit_str=0.0, trans=0.0,
        ior=1.45, coat=0.0, alpha=1.0, sss=0.0):
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    nt, p = _principled(m)
    c = tuple(color)[:3] + (1.0,)
    p.inputs["Base Color"].default_value = c
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    p.inputs["IOR"].default_value = ior
    if trans:
        p.inputs["Transmission Weight"].default_value = trans
    if coat:
        p.inputs["Coat Weight"].default_value = coat
    if sss:
        p.inputs["Subsurface Weight"].default_value = sss
    if alpha < 1:
        p.inputs["Alpha"].default_value = alpha
    if emit is not None:
        p.inputs["Emission Color"].default_value = tuple(emit)[:3] + (1.0,)
        p.inputs["Emission Strength"].default_value = emit_str
    m.diffuse_color = c
    return m


def mat_nodes(name):
    """Fresh material -> (material, node_tree, principled)."""
    m = bpy.data.materials.get(name)
    if m:
        bpy.data.materials.remove(m)
    m = bpy.data.materials.new(name)
    nt, p = _principled(m)
    return m, nt, p


def wood_mat(name, dark, light, scale=2.5, rough=0.45):
    """Wood grain: noise stretched along X, with a finer streak layer and bump."""
    m, nt, p = mat_nodes(name)
    N, L = nt.nodes.new, nt.links.new
    tc = N("ShaderNodeTexCoord")
    mp = N("ShaderNodeMapping"); mp.inputs["Scale"].default_value = (1.0, 16.0, 5.0)
    mp.inputs["Rotation"].default_value = (0, 0, math.radians(18))
    nz = N("ShaderNodeTexNoise")
    nz.inputs["Scale"].default_value = scale; nz.inputs["Detail"].default_value = 9.0
    nz.inputs["Roughness"].default_value = 0.62; nz.inputs["Distortion"].default_value = 1.4
    fine = N("ShaderNodeTexNoise")
    fine.inputs["Scale"].default_value = scale * 9; fine.inputs["Detail"].default_value = 4.0
    cr = N("ShaderNodeValToRGB")
    cr.color_ramp.elements[0].position = 0.32; cr.color_ramp.elements[0].color = tuple(dark) + (1,)
    cr.color_ramp.elements[1].position = 0.72; cr.color_ramp.elements[1].color = tuple(light) + (1,)
    mx = N("ShaderNodeMix"); mx.data_type = "RGBA"; mx.blend_type = "MULTIPLY"; mx.inputs["Factor"].default_value = 0.35
    bp = N("ShaderNodeBump"); bp.inputs["Strength"].default_value = 0.12
    rr = N("ShaderNodeMapRange"); rr.inputs["To Min"].default_value = rough - 0.12; rr.inputs["To Max"].default_value = rough + 0.15
    L(tc.outputs["Object"], mp.inputs["Vector"])
    L(mp.outputs[0], nz.inputs["Vector"]); L(mp.outputs[0], fine.inputs["Vector"])
    L(nz.outputs["Fac"], cr.inputs["Fac"])
    L(cr.outputs["Color"], mx.inputs["A"]); L(fine.outputs["Color"], mx.inputs["B"])
    L(mx.outputs["Result"], p.inputs["Base Color"])
    L(fine.outputs["Fac"], bp.inputs["Height"]); L(bp.outputs[0], p.inputs["Normal"])
    L(nz.outputs["Fac"], rr.inputs["Value"]); L(rr.outputs[0], p.inputs["Roughness"])
    return m


# ---------- mesh builders ----------
def _finish(name, bm, material=None, loc=(0, 0, 0), rot=(0, 0, 0), smooth=False,
            bevel=0.0, sharp=40):
    me = bpy.data.meshes.new(name)
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    if smooth:
        for poly in me.polygons:
            poly.use_smooth = True
        if sharp:
            try:
                me.set_sharp_from_angle(angle=math.radians(sharp))
            except Exception:
                pass
    ob = bpy.data.objects.new(name, me)
    ob.location = loc
    ob.rotation_euler = rot
    if material:
        me.materials.append(material)
    if bevel:
        md = ob.modifiers.new("Bevel", "BEVEL")
        md.width = bevel
        md.segments = 3
        md.limit_method = "ANGLE"
    S["col"].objects.link(ob)
    return ob


def box(name, size, loc=(0, 0, 0), material=None, rot=(0, 0, 0), bevel=0.0, base=False):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co.x *= size[0]; v.co.y *= size[1]; v.co.z *= size[2]
        if base:
            v.co.z += size[2] / 2
    return _finish(name, bm, material, loc, rot, smooth=bool(bevel), bevel=bevel)


def cyl(name, r, h, loc=(0, 0, 0), material=None, rot=(0, 0, 0), segs=32, r2=None,
        base=True, bevel=0.0, smooth=True):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segs, radius1=r,
                          radius2=r if r2 is None else r2, depth=h)
    if base:
        for v in bm.verts:
            v.co.z += h / 2
    return _finish(name, bm, material, loc, rot, smooth=smooth, bevel=bevel)


def sphere(name, r, loc=(0, 0, 0), material=None, ico=None, scale=(1, 1, 1), smooth=True,
           rough=0.0, seed=0):
    bm = bmesh.new()
    if ico is not None:
        bmesh.ops.create_icosphere(bm, subdivisions=ico, radius=r)
    else:
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=r)
    for v in bm.verts:
        if rough:
            n = noise.noise(v.co * (1.5 / r) + Vector((seed, seed * 2.3, seed * 0.7)))
            v.co *= 1 + rough * n
        v.co.x *= scale[0]; v.co.y *= scale[1]; v.co.z *= scale[2]
    return _finish(name, bm, material, loc, smooth=smooth, sharp=0)


def lathe(name, profile, loc=(0, 0, 0), material=None, segs=48, smooth=True, sharp=50):
    """profile: list of (radius, z). Revolved around Z. r==0 points become poles."""
    bm = bmesh.new()
    rings = []
    for r, z in profile:
        if r <= 1e-6:
            rings.append([bm.verts.new((0, 0, z))])
        else:
            rings.append([bm.verts.new((r * math.cos(2 * math.pi * i / segs),
                                        r * math.sin(2 * math.pi * i / segs), z))
                          for i in range(segs)])
    for a, b in zip(rings, rings[1:]):
        for i in range(segs):
            j = (i + 1) % segs
            if len(a) == 1 and len(b) == 1:
                continue
            if len(a) == 1:
                bm.faces.new((a[0], b[i], b[j]))
            elif len(b) == 1:
                bm.faces.new((a[i], a[j], b[0]))
            else:
                bm.faces.new((a[i], a[j], b[j], b[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return _finish(name, bm, material, loc, smooth=smooth, sharp=sharp)


def torus(name, R, r, loc=(0, 0, 0), material=None, rot=(0, 0, 0), seg=32, ring=12):
    bm = bmesh.new()
    vs = [[bm.verts.new(((R + r * math.cos(2 * math.pi * j / ring)) * math.cos(2 * math.pi * i / seg),
                         (R + r * math.cos(2 * math.pi * j / ring)) * math.sin(2 * math.pi * i / seg),
                         r * math.sin(2 * math.pi * j / ring))) for j in range(ring)] for i in range(seg)]
    for i in range(seg):
        for j in range(ring):
            bm.faces.new((vs[i][j], vs[(i + 1) % seg][j], vs[(i + 1) % seg][(j + 1) % ring], vs[i][(j + 1) % ring]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return _finish(name, bm, material, loc, rot, smooth=True, sharp=0)


def rod(name, p1, p2, r, material=None, segs=16, r2=None):
    p1, p2 = Vector(p1), Vector(p2)
    d = p2 - p1
    ob = cyl(name, r, d.length, loc=p1, material=material, segs=segs, r2=r2)
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d.normalized())
    return ob


def plane(name, sx, sy, loc=(0, 0, 0), material=None, rot=(0, 0, 0)):
    bm = bmesh.new()
    vs = [bm.verts.new((x * sx / 2, y * sy / 2, 0)) for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    bm.faces.new(vs)
    return _finish(name, bm, material, loc, rot)


def mesh_from(name, verts, faces, material=None, loc=(0, 0, 0), rot=(0, 0, 0), smooth=False, sharp=40):
    bm = bmesh.new()
    vs = [bm.verts.new(v) for v in verts]
    for f in faces:
        bm.faces.new([vs[i] for i in f])
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return _finish(name, bm, material, loc, rot, smooth=smooth, sharp=sharp)


def text(name, body, loc, material, size=1.0, extrude=0.03, rot=(math.pi / 2, 0, 0), align="CENTER"):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.size = size
    cu.extrude = extrude
    cu.align_x = align
    cu.materials.append(material)
    ob = bpy.data.objects.new(name, cu)
    ob.location = loc
    ob.rotation_euler = rot
    S["col"].objects.link(ob)
    return ob


# ---------- lights / camera / render ----------
def light(name, kind, loc, energy, color=(1, 1, 1), size=1.0, target=None, rot=None, spot=60, blend=0.3,
          size_y=None):
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
    ob = bpy.data.objects.new(name, ld)
    ob.location = loc
    if target is not None:
        ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    elif rot is not None:
        ob.rotation_euler = rot
    S["col"].objects.link(ob)
    return ob


def camera(name, loc, target, lens=50, fstop=None, focus=None):
    cd = bpy.data.cameras.new(name)
    cd.lens = lens
    ob = bpy.data.objects.new(name, cd)
    ob.location = loc
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    if fstop:
        cd.dof.use_dof = True
        cd.dof.aperture_fstop = fstop
        cd.dof.focus_distance = (Vector(focus if focus is not None else target) - Vector(loc)).length
    S["col"].objects.link(ob)
    S["scene"].camera = ob
    return ob


def setup_cycles(samples=160, res=(1920, 1080), look=None, exposure=0.0):
    sc = S["scene"]
    sc.render.engine = "CYCLES"
    try:
        sc.cycles.device = "GPU"
    except Exception:
        pass
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.cycles.use_adaptive_sampling = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = "PNG"
    sc.view_settings.exposure = exposure
    try:
        sc.view_settings.view_transform = "AgX"
        if look:
            sc.view_settings.look = look
    except Exception:
        pass


def world_color(color, strength=1.0):
    sc = S["scene"]
    w = bpy.data.worlds.new(sc.name + "_World")
    try:
        w.use_nodes = True
    except Exception:
        pass
    bg = next(n for n in w.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = tuple(color) + (1,)
    bg.inputs["Strength"].default_value = strength
    sc.world = w
    return w
