"""Re-save a .blend without machine-specific absolute paths (render output, old file-browser folders).

  blender -b --factory-startup --python clean_blend.py -- in.blend out.blend
"""
import bpy, sys
src, out = sys.argv[sys.argv.index('--') + 1:][:2]
bpy.ops.wm.open_mainfile(filepath=src, load_ui=False)
for s in bpy.data.scenes:
    s.render.filepath = '//render_'
    for n in (s.node_tree.nodes if getattr(s, 'node_tree', None) else []):
        if hasattr(n, 'base_path'):
            n.base_path = '//'
for coll in (bpy.data.images, bpy.data.libraries, bpy.data.fonts, bpy.data.sounds, bpy.data.movieclips, bpy.data.cache_files):
    for d in coll:
        fp = getattr(d, 'filepath', '')
        if fp.startswith('/'):
            print('ABS PATH', d.name, fp)
            if fp.startswith('/Users/'):
                d.filepath = '//' + fp.split('/')[-1]
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=out, compress=True, copy=True, relative_remap=False)
print('SAVED', out)
