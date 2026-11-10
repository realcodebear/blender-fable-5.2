"""Turn the original comparison page (tools/page_source.html) into the published site (index.html).

Adds: a proper HTML head, local fonts, a 3D viewer block per scene, .blend download links.
Run from the repo root:  python3 tools/build_site.py
"""
import json, os, re, html

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = 'https://github.com/realcodebear/blender-fable-5.2'
s = open(os.path.join(ROOT, 'tools', 'page_source.html')).read()

SCENES = [
    ('still-life', 's1', 'S1_StillLife', 'still life'),
    ('island', 's2', 'S2_FloatingIsland', 'floating island'),
    ('city', 's3', 'S3_NeonCity', 'neon city'),
]
NOTES = json.load(open(os.path.join(ROOT, 'tools', 'viewer_notes.json')))


def mb(path):
    return '%.1f MB' % (os.path.getsize(os.path.join(ROOT, path)) / 1e6)


def meta(key):
    return json.load(open(os.path.join(ROOT, 'models', key + '.json')))


# ---- head
head = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Blender Effort Test · medium vs max effort</title>
<meta name="description" content="Three Blender scenes built by an AI model from Python, once in a quick pass and once at max effort. Compare the renders, orbit both versions in 3D, and download the .blend files.">
<meta property="og:title" content="Blender Effort Test · medium vs max effort">
<meta property="og:description" content="Three Blender scenes, two effort levels. Compare renders, explore in 3D, download the .blend files.">
<meta property="og:image" content="img/s3_v2.jpg">
<meta name="theme-color" content="#131417">
<link rel="stylesheet" href="vendor/fonts/fonts.css">
<script type="importmap">{"imports":{"three":"./vendor/three/three.module.min.js","three/addons/":"./vendor/three/addons/"}}</script>
'''
s = re.sub(r'^<title>.*?</title>\n<meta name="claude:viewport-fit-sides"[^>]*>\n<link rel="stylesheet" href="https://fonts.googleapis.com[^>]*>\n', '', s, count=1, flags=re.S)
assert 'fonts.googleapis' not in s
i = s.index('</style>') + len('</style>')
s = head + s[:i] + '\n</head>\n<body>\n' + s[i:] + '\n</body>\n</html>\n'

# ---- css
css = '''
/* 3D viewer */
.v3d { display: grid; gap: 12px; min-width: 0; }
.v3d-head { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
.v3d-head small { font: 400 .68rem var(--mono); letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin-left: auto; }
.v3d-box { position: relative; aspect-ratio: 16 / 9; border-radius: var(--r); overflow: hidden; background: #000; box-shadow: 0 30px 80px -30px rgba(0,0,0,.9), 0 0 0 1px var(--line); }
.v3d-box:fullscreen { border-radius: 0; aspect-ratio: auto; width: 100vw; height: 100vh; }
.v3d-poster { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; filter: brightness(.42) saturate(.9); transition: opacity .5s; }
.v3d-canvas { position: absolute; inset: 0; width: 100%; height: 100%; display: block; touch-action: none; cursor: grab; outline: none; }
.v3d-canvas:active { cursor: grabbing; }
.v3d-start { position: absolute; inset: 0; display: grid; place-content: center; justify-items: center; gap: 12px; text-align: center; padding: 16px; }
.v3d-start button { font: 600 1rem var(--display); color: #131417; background: var(--v2); border: 0; border-radius: 999px; padding: 14px 26px; cursor: pointer; box-shadow: 0 10px 40px rgba(0,0,0,.6); }
.v3d-start button:hover { background: var(--v2-soft); }
.v3d-start span { font: 400 .74rem var(--mono); letter-spacing: .06em; color: var(--soft); background: rgba(10,11,13,.6); padding: 4px 10px; border-radius: 6px; }
.v3d-load { position: absolute; left: 16px; right: 16px; bottom: 16px; display: grid; gap: 8px; pointer-events: none; }
.v3d-load i { display: block; height: 4px; border-radius: 4px; background: rgba(255,255,255,.14); overflow: hidden; }
.v3d-load i b { display: block; height: 100%; width: 0; background: var(--v2); transition: width .2s; }
.v3d-load span { font: 400 .74rem var(--mono); color: var(--soft); justify-self: start; background: rgba(10,11,13,.7); padding: 4px 9px; border-radius: 6px; }
.v3d-tag { position: absolute; top: 14px; left: 14px; font: 500 .7rem var(--mono); letter-spacing: .1em; text-transform: uppercase; padding: 6px 10px; border-radius: 6px; background: rgba(10,11,13,.78); pointer-events: none; color: var(--v2); }
.v3d-tag.one { color: var(--soft); }
.v3d-err { position: absolute; inset: 0; display: grid; place-content: center; padding: 20px; text-align: center; color: var(--soft); background: rgba(10,11,13,.8); }
.v3d [hidden] { display: none !important; }
.v3d-tag, .v3d-load, .v3d-err { z-index: 2; }
.v3d-note { color: var(--muted); font-size: .9rem; max-width: 86ch; }
.v3d-note b { color: var(--soft); font-weight: 500; }
a.ocopy { text-decoration: none; display: inline-block; text-align: center; }
.osize { font: 400 .78rem var(--mono); color: var(--muted); }
.ofile { font: 400 .8rem var(--mono); color: var(--soft); background: var(--bg); border-radius: 8px; padding: 9px 12px; overflow-x: auto; white-space: nowrap; min-width: 0; display: flex; gap: 14px; justify-content: space-between; }
.nav nav a.gh { color: var(--fg); background: var(--panel-2); }
@media (max-width: 560px) { .v3d-box { aspect-ratio: 4 / 3; } .v3d-tag { font-size: .58rem; top: 10px; left: 10px; } .v3d-start button { padding: 12px 20px; font-size: .92rem; } .v3d-head small { margin-left: 0; } }
'''
s = s.replace('\n/* cost */', css + '\n/* cost */', 1)

# ---- viewer block + downloads per scene
for sid, k, name, label in SCENES:
    a = s.index('id="%s"' % sid)
    b = s.index('<div class="stats">', a)
    m1, m2 = meta(k + '_v1'), meta(k + '_v2')
    sz = lambda m: '%.1f MB' % (m['stats']['glb_bytes'] / 1e6)
    tri = lambda m: '{:,}'.format(m['stats']['triangles'])
    block = (
        '<div class="v3d" data-v3d="%s">'
        '<div class="v3d-head"><h3>Explore it in 3D</h3><small>drag = orbit · scroll / pinch = zoom · right-drag / two fingers = pan</small></div>'
        '<div class="v3d-box">'
        '<img class="v3d-poster" src="img/%s_v2.jpg" alt="" loading="lazy">'
        '<div class="v3d-start"><button type="button">Load the 3D scene</button><span>v2 %s · v1 %s · loads only when you click</span></div>'
        '<span class="v3d-tag" hidden></span>'
        '<div class="v3d-load" hidden><span></span><i><b></b></i></div>'
        '<div class="v3d-err" hidden></div>'
        '</div>'
        '<div class="ctrl"><span>Web preview of the same geometry. Lighting is approximated.</span>'
        '<button type="button" data-ver="v1" data-size="%s" data-tris="%s">v1 · quick pass</button>'
        '<button type="button" data-ver="v2" data-size="%s" data-tris="%s">v2 · max effort</button>'
        '<button type="button" data-act="reset">Render view</button>'
        '<button type="button" data-act="full">Full screen</button></div>'
        '<p class="v3d-note"><b>What differs from Blender:</b> %s</p>'
        '</div>\n  '
    ) % (k, k, sz(m2), sz(m1), sz(m1), tri(m1), sz(m2), tri(m2), NOTES[k])
    s = s[:b] + block + s[b:]

    a = s.index('<div class="open">', s.index('id="%s"' % sid))
    b = s.index('<div class="sp">', a)
    rows = ''
    for lab, ver in (('v2 · max effort', 'v2'), ('v1 · quick pass', 'v1')):
        f = 'blend/%s_%s.blend' % (name, ver)
        rows += ('<div class="orow"><span class="olab">%s</span><span class="ofile"><span>%s_%s.blend</span><span class="osize">%s</span></span>'
                 '<a class="ocopy" href="%s" download>Download .blend</a></div>') % (lab, name, ver, mb(f), f)
    rows += ('<div class="orow"><span class="olab">Source</span><span class="ofile"><span>Python build scripts, renders, GLB files</span></span>'
             '<a class="ocopy" href="%s">GitHub repo</a></div>') % REPO
    s = s[:a] + '<div class="open"><h3>Download the Blender files <small>Blender 5.2 or newer</small></h3>' + rows + '</div>\n  ' + s[b:]

assert 'open -a Blender' not in s.split('<section class="scene gut" id="prompts"')[0]
s = s.replace('<a href="#prompts">Prompts</a></nav>', '<a href="#prompts">Prompts</a><a class="gh" href="%s">GitHub</a></nav>' % REPO)
s = s.replace('<footer class="gut"><div class="in">', '<footer class="gut"><div class="in">\n  <p>Files and source: <a href="%s">%s</a>. The 3D viewers use three.js and show Draco-compressed GLB exports of the same .blend files; they are a real-time preview, not the Cycles render.</p>' % (REPO, REPO.replace('https://', '')))

# old "copy command" handler is no longer needed
a = s.index("document.querySelectorAll('button.ocopy')")
b = s.index('});\n});\n', a) + len('});\n});\n')
s = s[:a] + s[b:]
s = s.replace('</body>', '<script src="viewer/ui.js"></script>\n</body>')
open(os.path.join(ROOT, 'index.html'), 'w').write(s)
print('index.html', len(s))
