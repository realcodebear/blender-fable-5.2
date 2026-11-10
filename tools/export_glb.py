"""Export one Blender scene to a web-ready GLB.

  blender -b scene.blend --python export_glb.py -- <out_dir> <key> [atlas_px] [pos_bits]

What it does
  1. Writes <key>.json: render camera, every light, colour settings, stats.
  2. Renders the world (sky) to <key>_sky.jpg as an equirect panorama.
  3. Turns text / curves into meshes and applies all modifiers.
  4. Plain materials (fixed colours) are exported as they are.
     Procedural materials are baked into one shared texture atlas
     (colour, emission, roughness), because node trees do not exist on the web.
  5. Drops volume-only objects (fog, steam): the viewer adds its own fog.
  6. Joins everything into one mesh (few draw calls) and writes a Draco GLB with WebP textures.
"""
import bpy, sys, json, math, os, time
import numpy as np
from mathutils import Vector

argv = sys.argv[sys.argv.index('--') + 1:]
OUT, KEY = argv[0], argv[1]
ATLAS = int(argv[2]) if len(argv) > 2 else 4096
POSBITS = int(argv[3]) if len(argv) > 3 else 16
os.makedirs(OUT, exist_ok=True)
sc = bpy.context.scene
vl = bpy.context.view_layer
T0 = time.time()


def log(*a):
    print('[exp %6.1fs]' % (time.time() - T0), *a, flush=True)


def yup(v):
    return [round(v[0], 5), round(v[2], 5), round(-v[1], 5)]


# ---------------------------------------------------------------- meta
def hidden_objects():
    hid = set()

    def walk(lc, h):
        h = h or lc.exclude or lc.collection.hide_render
        if h:
            hid.update(o.name for o in lc.collection.objects)
        for c in lc.children:
            walk(c, h)
    walk(vl.layer_collection, False)
    return hid


HIDDEN = hidden_objects()
objs = [o for o in sc.objects if not o.hide_render and o.name not in HIDDEN]
meta = {'key': KEY, 'file': os.path.basename(bpy.data.filepath)}
cam = sc.camera
mw = cam.matrix_world
cpos = mw.translation.copy()
fwd = (mw.to_3x3() @ Vector((0, 0, -1))).normalized()
aspect = sc.render.resolution_x / sc.render.resolution_y
hf = cam.data.angle
vf = 2 * math.atan(math.tan(hf / 2) / aspect) if aspect >= 1 else hf
fd = 0.0
if cam.data.dof.focus_object:
    fd = (cam.data.dof.focus_object.matrix_world.translation - cpos).dot(fwd)
elif cam.data.dof.use_dof:
    fd = cam.data.dof.focus_distance
meta['camera'] = {'pos': yup(cpos), 'dir': yup(fwd), 'vfov': round(math.degrees(vf), 3), 'focus': round(fd, 3),
                  'near': cam.data.clip_start, 'far': cam.data.clip_end}
meta['view'] = {'transform': sc.view_settings.view_transform, 'look': sc.view_settings.look,
                'exposure': sc.view_settings.exposure, 'gamma': sc.view_settings.gamma}
lights = []
for o in objs:
    if o.type != 'LIGHT':
        continue
    d = o.data
    m = o.matrix_world
    L = {'name': o.name, 'type': d.type, 'pos': yup(m.translation), 'dir': yup((m.to_3x3() @ Vector((0, 0, -1))).normalized()),
         'color': [round(c, 4) for c in d.color], 'energy': round(d.energy, 4)}
    if d.type == 'SPOT':
        L['angle'] = round(d.spot_size, 4); L['blend'] = round(d.spot_blend, 3)
    if d.type == 'AREA':
        L['size'] = [round(d.size, 3), round(d.size_y if d.shape in ('RECTANGLE', 'ELLIPSE') else d.size, 3)]
    if d.type in ('POINT', 'SPOT'):
        L['radius'] = round(d.shadow_soft_size, 3)
    lights.append(L)
meta['lights'] = lights
w = sc.world
meta['world'] = None
if w and w.node_tree:
    bg = next((n for n in w.node_tree.nodes if n.type == 'BACKGROUND'), None)
    if bg:
        meta['world'] = {'color': [round(c, 4) for c in bg.inputs[0].default_value[:3]], 'strength': round(bg.inputs[1].default_value, 4),
                         'procedural': bg.inputs[0].is_linked}
meta['fog_volumes'] = []


# ---------------------------------------------------------------- sky panorama
def render_sky(path):
    s2 = bpy.data.scenes.new('sky_tmp')
    s2.world = sc.world
    cd = bpy.data.cameras.new('sky_cam')
    cd.type = 'PANO'
    if hasattr(cd, 'panorama_type'):
        cd.panorama_type = 'EQUIRECTANGULAR'
    else:
        cd.cycles.panorama_type = 'EQUIRECTANGULAR'
    co = bpy.data.objects.new('sky_cam', cd)
    s2.collection.objects.link(co)
    co.rotation_euler = (math.pi / 2, 0, -math.pi / 2)   # image centre looks along +X
    s2.camera = co
    s2.render.engine = 'CYCLES'
    s2.cycles.samples = 8
    s2.cycles.use_denoising = False
    s2.cycles.device = 'CPU'
    s2.render.resolution_x, s2.render.resolution_y, s2.render.resolution_percentage = 2048, 1024, 100
    s2.view_settings.view_transform = 'Standard'
    s2.view_settings.look = 'None'
    s2.render.image_settings.file_format = 'JPEG'
    s2.render.image_settings.quality = 88
    s2.render.filepath = path
    bpy.ops.render.render(write_still=True, scene=s2.name)
    bpy.data.scenes.remove(s2)


try:
    render_sky(os.path.join(OUT, KEY + '_sky.jpg'))
    log('sky rendered')
except Exception as e:
    log('sky render failed', e)
bpy.context.window.scene = sc if bpy.context.window else sc

# ---------------------------------------------------------------- everything becomes a plain mesh
dg = bpy.context.evaluated_depsgraph_get()
conv = []
for o in list(sc.objects):
    if o.type in ('LIGHT', 'CAMERA', 'EMPTY'):
        continue
    if o not in objs or o.type not in ('MESH', 'CURVE', 'FONT', 'SURFACE', 'META'):
        conv.append((o, None)); continue
    try:
        me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    except Exception:
        me = None
    if me is not None and len(me.polygons) == 0:
        me = None
    conv.append((o, me))
meshes = []
for o, me in conv:
    if me is None:
        bpy.data.objects.remove(o, do_unlink=True); continue
    if o.type == 'MESH':
        o.modifiers.clear()
        o.data = me
        meshes.append(o)
    else:
        name = o.name
        mat = o.matrix_world.copy()
        coll = o.users_collection[0] if o.users_collection else sc.collection
        o.name = name + '__old'
        n = bpy.data.objects.new(name, me)
        coll.objects.link(n)
        n.matrix_world = mat
        bpy.data.objects.remove(o, do_unlink=True)
        meshes.append(n)
vl.update()
log('meshes', len(meshes))


# ---------------------------------------------------------------- classify materials
def tree_info(m):
    nt = m.node_tree
    out = nt.get_output_node('CYCLES') or nt.get_output_node('ALL')
    if out is None:
        return None, False, None
    s, v = out.inputs['Surface'], out.inputs['Volume']
    surf = s.links[0].from_node if s.is_linked else None
    pr = surf if (surf and surf.type == 'BSDF_PRINCIPLED') else next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    return surf, v.is_linked, pr


METAL_WORDS = ('brass', 'metal', 'steel', 'chrome', 'gold', 'iron', 'alu', 'copper', 'silver')
CLS, INFO = {}, {}


def classify(m):
    if m is None or m.node_tree is None:
        return 'const'
    surf, vol, pr = tree_info(m)
    if surf is None:
        return 'drop'
    types = {n.type for n in m.node_tree.nodes}
    info = {'metal': 0.0, 'rough': 0.6, 'rough_tex': False, 'alpha': 1.0, 'emit': False}
    if pr:
        mi, ri = pr.inputs['Metallic'], pr.inputs['Roughness']
        info['metal'] = (1.0 if any(k in m.name.lower() for k in METAL_WORDS) else 0.0) if mi.is_linked else mi.default_value
        info['rough'] = ri.default_value
        info['rough_tex'] = ri.is_linked
        a = pr.inputs['Alpha']
        if not a.is_linked and a.default_value < 0.99:
            info['alpha'] = a.default_value
        tw = pr.inputs.get('Transmission Weight')
        if tw is not None and not tw.is_linked and tw.default_value > 0.3:
            info['alpha'] = min(info['alpha'], 0.3)
        es, ec = pr.inputs['Emission Strength'], pr.inputs['Emission Color']
        if es.is_linked or ec.is_linked or (es.default_value > 0 and max(ec.default_value[:3]) > 0):
            info['emit'] = True
    elif 'BSDF_ANISOTROPIC' in types or 'BSDF_GLOSSY' in types:
        info['rough'] = 0.12
    if 'EMISSION' in types:
        info['emit'] = True
    if 'BSDF_TRANSPARENT' in types:
        info['alpha'] = min(info['alpha'], 0.35)
    INFO[m.name] = info
    # a bare Emission or Diffuse shader with fixed values is just a plain material: rebuild it as Principled
    if surf.type in ('EMISSION', 'BSDF_DIFFUSE') and not any(i.is_linked for i in surf.inputs if i.name in ('Color', 'Strength', 'Roughness')):
        nt = m.node_tree
        out = nt.get_output_node('CYCLES') or nt.get_output_node('ALL')
        p2 = nt.nodes.new('ShaderNodeBsdfPrincipled')
        col = tuple(surf.inputs['Color'].default_value)
        if surf.type == 'EMISSION':
            p2.inputs['Base Color'].default_value = (0, 0, 0, 1)
            p2.inputs['Emission Color'].default_value = col
            p2.inputs['Emission Strength'].default_value = surf.inputs['Strength'].default_value
            p2.inputs['Roughness'].default_value = 0.5
        else:
            p2.inputs['Base Color'].default_value = col
            p2.inputs['Roughness'].default_value = 0.8
        nt.links.new(p2.outputs[0], out.inputs['Surface'])
        return 'const'
    if surf.type == 'BSDF_PRINCIPLED' and not any(surf.inputs[k].is_linked for k in ('Base Color', 'Emission Color', 'Emission Strength', 'Alpha')):
        for inp in surf.inputs:          # bump / procedural roughness cannot be exported: keep the plain value
            for l in list(inp.links):
                m.node_tree.links.remove(l)
        return 'const'
    return 'bake'


used = {}
for o in meshes:
    for s in o.material_slots:
        if s.material and s.material.name not in CLS:
            CLS[s.material.name] = classify(s.material)
            used[s.material.name] = s.material


def cls(m):
    return CLS.get(m.name, 'const') if m else 'const'


# remove volume-only objects (fog boxes, steam) and note them
for o in list(meshes):
    cs = [cls(s.material) for s in o.material_slots]
    if cs and all(c == 'drop' for c in cs):
        bb = [o.matrix_world @ Vector(c) for c in o.bound_box]
        meta['fog_volumes'].append({'name': o.name, 'material': o.material_slots[0].material.name,
                                    'min': yup(Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))),
                                    'max': yup(Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb))))})
        meshes.remove(o)
        bpy.data.objects.remove(o, do_unlink=True)
log('materials', {k: list(CLS.values()).count(k) for k in ('const', 'bake', 'drop')}, 'fog', len(meta['fog_volumes']))

bake_objs = [o for o in meshes if any(cls(s.material) == 'bake' for s in o.material_slots)]
meta['atlas'] = None

if bake_objs:
    # ------------------------------------------------------------ atlas UVs
    for o in bake_objs:
        me = o.data
        uv = me.uv_layers.new(name='atlas', do_init=False)
        me.uv_layers.active = uv
        idx = [i for i, s in enumerate(o.material_slots) if cls(s.material) == 'bake']
        mi = np.empty(len(me.polygons), 'i'); me.polygons.foreach_get('material_index', mi)
        sel = np.isin(mi, idx)
        me.vertices.foreach_set('select', np.zeros(len(me.vertices), '?'))
        me.edges.foreach_set('select', np.zeros(len(me.edges), '?'))
        me.polygons.foreach_set('hide', np.zeros(len(me.polygons), '?'))
        me.polygons.foreach_set('select', sel)
        me.update()
    bpy.ops.object.select_all(action='DESELECT')
    for o in bake_objs:
        o.select_set(True)
    vl.objects.active = bake_objs[0]
    sc.tool_settings.use_uv_select_sync = True
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.smart_project(angle_limit=math.radians(66), island_margin=0.0, area_weight=0.0, correct_aspect=True, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    log('smart project done', len(bake_objs), 'objects')

    rec = []
    for o in bake_objs:
        me = o.data
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        tl = np.empty(nt * 3, 'i'); me.loop_triangles.foreach_get('loops', tl)
        tv = np.empty(nt * 3, 'i'); me.loop_triangles.foreach_get('vertices', tv)
        tp = np.empty(nt, 'i'); me.loop_triangles.foreach_get('polygon_index', tp)
        mi = np.empty(len(me.polygons), 'i'); me.polygons.foreach_get('material_index', mi)
        idx = [i for i, s in enumerate(o.material_slots) if cls(s.material) == 'bake']
        keep = np.isin(mi, idx)[tp]
        co = np.empty(len(me.vertices) * 3, 'f'); me.vertices.foreach_get('co', co)
        co = co.reshape(-1, 3)
        M = np.array(o.matrix_world)
        cw = co @ M[:3, :3].T + M[:3, 3]
        uvs = np.empty(len(me.loops) * 2, 'f'); me.uv_layers['atlas'].uv.foreach_get('vector', uvs)
        uvs = uvs.reshape(-1, 2)
        t3 = cw[tv.reshape(-1, 3)][keep]
        t2 = uvs[tl.reshape(-1, 3)][keep]
        a3 = 0.5 * np.linalg.norm(np.cross(t3[:, 1] - t3[:, 0], t3[:, 2] - t3[:, 0]), axis=1).sum()
        e1, e2 = t2[:, 1] - t2[:, 0], t2[:, 2] - t2[:, 0]
        a2 = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum()
        c = cw.mean(axis=0); r = np.linalg.norm(cw - c, axis=1).max()
        d = max(np.linalg.norm(c - np.array(cpos)) - r, 0.0)
        rec.append([o, uvs, float(a3), float(a2), d])
    ds = sorted(r[4] for r in rec)
    d0 = max(ds[len(ds) // 10], 0.5)
    for r in rec:                      # nearer to the render camera = more texels
        wgt = min(1.0, d0 / max(r[4], d0)) ** 0.75
        r.append(r[2] * max(wgt, 0.1) ** 2)
    for _ in range(3):                 # no single object may take more than ~18% of the atlas
        tot = sum(r[5] for r in rec)
        for r in rec:
            r[5] = min(r[5], 0.18 * tot)
    for o, uvs, a3, a2, d, a in rec:
        if a2 > 1e-12:
            uvs *= math.sqrt(a / a2)
        o.data.uv_layers['atlas'].uv.foreach_set('vector', uvs.ravel())
    bpy.ops.object.mode_set(mode='EDIT')
    marg = 3.0 / ATLAS
    try:
        bpy.ops.uv.pack_islands(rotate=True, scale=True, margin_method='FRACTION', margin=marg, shape_method='AABB')
    except TypeError:
        bpy.ops.uv.pack_islands(rotate=True, margin=marg)
    bpy.ops.object.mode_set(mode='OBJECT')
    log('packed')

    # ------------------------------------------------------------ bake
    sc.render.engine = 'CYCLES'
    sc.cycles.device = 'CPU'
    sc.cycles.samples = 2
    sc.cycles.use_auto_tile = False      # tiles write big temp files per object and slow the bake a lot
    sc.cycles.use_denoising = False
    img_col = bpy.data.images.new('atlas_col', ATLAS, ATLAS, alpha=False)
    img_em = bpy.data.images.new('atlas_emit_f', ATLAS, ATLAS, alpha=False, float_buffer=True)
    img_r = bpy.data.images.new('atlas_rough', ATLAS, ATLAS, alpha=False, is_data=True)
    dummy = bpy.data.images.new('dummy', 8, 8)
    bnodes = {}
    for o in bake_objs:
        for s in o.material_slots:
            m = s.material
            if m is None or m.name in bnodes or m.node_tree is None:
                continue
            n = m.node_tree.nodes.new('ShaderNodeTexImage')
            n.image = dummy
            for x in m.node_tree.nodes:
                x.select = False
            n.select = True
            m.node_tree.nodes.active = n
            bnodes[m.name] = n

    def bake(kind, img, **kw):
        for name, n in bnodes.items():
            n.image = img if CLS.get(name) == 'bake' else dummy
        t = time.time()
        bpy.ops.object.bake(type=kind, margin=6, use_clear=True, target='IMAGE_TEXTURES', **kw)
        log('baked', kind, '%.0fs' % (time.time() - t))

    bake('EMIT', img_em)
    need_rough = any(INFO.get(n, {}).get('rough_tex') for n, c in CLS.items() if c == 'bake')
    if need_rough:
        bake('ROUGHNESS', img_r)
    for name, c in CLS.items():        # metal / glass have no diffuse colour: switch them off so the colour bakes
        if c != 'bake':
            continue
        nt = used[name].node_tree
        for n in nt.nodes:
            if n.type == 'BSDF_PRINCIPLED':
                for k in ('Metallic', 'Transmission Weight'):
                    inp = n.inputs.get(k)
                    if inp is None:
                        continue
                    for l in list(inp.links):
                        nt.links.remove(l)
                    inp.default_value = 0.0
    bake('DIFFUSE', img_col, pass_filter={'COLOR'})

    em = np.empty(ATLAS * ATLAS * 4, 'f'); img_em.pixels.foreach_get(em)
    em = em.reshape(-1, 4)
    mx = em[:, :3].max(axis=1)
    lit = mx[mx > 0.01]
    K = 1.0
    has_emit = lit.size > 50
    if has_emit:
        K = float(min(max(np.percentile(lit, 99.5), 1.0), 24.0))
        e = np.clip(em[:, :3] / K, 0, 1)
        e = np.where(e <= 0.0031308, e * 12.92, 1.055 * np.power(e, 1 / 2.4) - 0.055)
        em[:, :3] = e; em[:, 3] = 1
        img_e8 = bpy.data.images.new('atlas_emit', ATLAS, ATLAS, alpha=False)
        img_e8.pixels.foreach_set(em.ravel())
    del em

    def save(img, name):
        img.filepath_raw = os.path.join(OUT, 'tmp_%s_%s.png' % (KEY, name))
        img.file_format = 'PNG'
        img.save()
    save(img_col, 'col')
    if has_emit:
        save(img_e8, 'emit')
    if need_rough:
        save(img_r, 'rough')
    meta['atlas'] = {'px': ATLAS, 'emit_scale': round(K, 3), 'emit': bool(has_emit), 'rough': bool(need_rough), 'objects': len(bake_objs)}
    log('atlas', meta['atlas'])

    # ------------------------------------------------------------ simple web materials that read the atlas
    newm = {}

    def web_mat(info):
        sig = (round(info['metal'], 2), 'T' if info['rough_tex'] else round(info['rough'], 2), round(info['alpha'], 2), info['emit'] and has_emit)
        if sig in newm:
            return newm[sig]
        m = bpy.data.materials.new('atlas_%d' % len(newm))
        m.use_nodes = True
        nt = m.node_tree
        pr = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED')
        t = nt.nodes.new('ShaderNodeTexImage'); t.image = img_col
        nt.links.new(t.outputs['Color'], pr.inputs['Base Color'])
        pr.inputs['Metallic'].default_value = info['metal']
        pr.inputs['Roughness'].default_value = info['rough']
        if info['rough_tex'] and need_rough:
            tr = nt.nodes.new('ShaderNodeTexImage'); tr.image = img_r
            sp = nt.nodes.new('ShaderNodeSeparateColor')
            nt.links.new(tr.outputs['Color'], sp.inputs[0])
            nt.links.new(sp.outputs[1], pr.inputs['Roughness'])
        if sig[3]:
            te = nt.nodes.new('ShaderNodeTexImage'); te.image = img_e8
            nt.links.new(te.outputs['Color'], pr.inputs['Emission Color'])
            pr.inputs['Emission Strength'].default_value = K
        if info['alpha'] < 0.99:
            pr.inputs['Alpha'].default_value = info['alpha']
            try:
                m.surface_render_method = 'BLENDED'
            except Exception:
                pass
        newm[sig] = m
        return m

    for o in bake_objs:
        for s in o.material_slots:
            if s.material and cls(s.material) == 'bake':
                s.material = web_mat(INFO[s.material.name])
    for name, n in bnodes.items():
        if CLS.get(name) != 'bake':
            used[name].node_tree.nodes.remove(n)
    log('web materials', len(newm))

# ---------------------------------------------------------------- tidy, join, export
for o in meshes:
    me = o.data
    for l in [l for l in me.uv_layers if l.name != 'atlas']:
        me.uv_layers.remove(l)
    if not me.uv_layers:
        me.uv_layers.new(name='atlas')
    for a in list(me.color_attributes):
        me.color_attributes.remove(a)
    if not o.material_slots or all(s.material is None for s in o.material_slots):
        dm = bpy.data.materials.get('default_grey') or bpy.data.materials.new('default_grey')
        if o.material_slots:
            for s in o.material_slots:
                s.material = dm
        else:
            me.materials.append(dm)
bpy.ops.object.select_all(action='DESELECT')
for o in meshes:
    o.select_set(True)
vl.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
J = vl.objects.active
J.name = KEY
J.data.calc_loop_triangles()
meta['stats'] = {'objects_in_blender': len(objs), 'meshes_joined': len(meshes), 'triangles': len(J.data.loop_triangles),
                 'materials': len([s for s in J.material_slots if s.material])}
bb = [J.matrix_world @ Vector(c) for c in J.bound_box]
meta['bounds'] = {'min': yup(Vector((min(v.x for v in bb), min(v.y for v in bb), min(v.z for v in bb)))),
                  'max': yup(Vector((max(v.x for v in bb), max(v.y for v in bb), max(v.z for v in bb))))}
for o in list(sc.objects):
    if o is not J:
        bpy.data.objects.remove(o, do_unlink=True)
glb = os.path.join(OUT, KEY + '.glb')
bpy.ops.export_scene.gltf(filepath=glb, export_format='GLB', export_apply=False, export_yup=True,
                          export_cameras=False, export_lights=False, export_animations=False, export_tangents=False,
                          export_vertex_color='NONE', export_materials='EXPORT',
                          export_image_format='WEBP', export_image_quality=86,
                          export_draco_mesh_compression_enable=True, export_draco_mesh_compression_level=6,
                          export_draco_position_quantization=POSBITS, export_draco_normal_quantization=10,
                          export_draco_texcoord_quantization=12)
meta['stats']['glb_bytes'] = os.path.getsize(glb)
for f in os.listdir(OUT):
    if f.startswith('tmp_%s_' % KEY):
        os.remove(os.path.join(OUT, f))
json.dump(meta, open(os.path.join(OUT, KEY + '.json'), 'w'), indent=1)
log('DONE', glb, '%.2f MB' % (meta['stats']['glb_bytes'] / 1e6), meta['stats'])
