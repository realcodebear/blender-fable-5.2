import bpy, sys, collections
sc=bpy.context.scene
print("=== FILE", bpy.data.filepath)
print("engine",sc.render.engine,"cam",sc.camera and sc.camera.name, "scenes",[s.name for s in bpy.data.scenes])
types=collections.Counter(o.type for o in sc.objects); print("types",dict(types))
dg=bpy.context.evaluated_depsgraph_get()
tris=0; inst=0
for oi in dg.object_instances:
    o=oi.object
    if o.type=='MESH' and o.data:
        o.data.calc_loop_triangles(); tris+=len(o.data.loop_triangles); inst+=1
print("eval mesh instances",inst,"tris",tris)
mods=collections.Counter(m.type for o in sc.objects for m in o.modifiers); print("mods",dict(mods))
print("particles",[(o.name,ps.settings.type,ps.settings.count,ps.settings.render_type) for o in sc.objects for ps in o.particle_systems])
print("images",[(i.name,i.size[:],i.source) for i in bpy.data.images])
print("lights",[(o.name,o.data.type,round(o.data.energy,1),tuple(round(c,2) for c in o.data.color)) for o in sc.objects if o.type=='LIGHT'])
w=sc.world
print("world",w and [n.bl_idname for n in w.node_tree.nodes] if w and w.use_nodes else w)
for m in bpy.data.materials:
    if not m.use_nodes or not m.users: continue
    ids=collections.Counter(n.bl_idname.replace('ShaderNode','') for n in m.node_tree.nodes)
    print(" mat",m.name,dict(ids))
print("hidden_render",[o.name for o in sc.objects if o.hide_render][:30])
print("nonmesh",[(o.name,o.type) for o in sc.objects if o.type not in('MESH','LIGHT','CAMERA','EMPTY')][:40])
print("uvless",sum(1 for o in sc.objects if o.type=='MESH' and not o.data.uv_layers),"of",types.get('MESH'))
print("collections",[(c.name,len(c.objects)) for c in bpy.data.collections])
