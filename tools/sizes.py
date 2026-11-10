"""Print the file-size table used in README.md (run from the repo root)."""
import json, os
rows = [('s1', 'S1_StillLife', 'Still life'), ('s2', 'S2_FloatingIsland', 'Floating island'), ('s3', 'S3_NeonCity', 'Neon city')]
mb = lambda p: '%.2f MB' % (os.path.getsize(p) / 1e6)
print('| Scene | Version | .blend | GLB (web) | Triangles | Materials | Atlas |')
print('|---|---|---|---|---|---|---|')
for k, n, label in rows:
    for v in ('v1', 'v2'):
        m = json.load(open('models/%s_%s.json' % (k, v)))
        a = m['atlas']
        print('| %s | %s | %s | %s | %s | %d | %s |' % (label, v, mb('blend/%s_%s.blend' % (n, v)), mb('models/%s_%s.glb' % (k, v)),
              '{:,}'.format(m['stats']['triangles']), m['stats']['materials'], ('%d px' % a['px']) if a else 'none'))
