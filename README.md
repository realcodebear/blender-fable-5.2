# Blender Effort Test

**Live page: https://realcodebear.github.io/blender-effort-test/**

Three Blender scenes, each built twice by the same AI model (Claude Fable 5.2) from Python, with no downloaded models and no image textures.

- **v1 = quick pass.** One prompt, one session, medium effort, all three scenes.
- **v2 = max effort.** Three separate agents, one scene each, told what v1 got wrong.

The page lets you compare the renders with a slider, orbit both versions in 3D, read the exact prompts and the cost, and download the `.blend` files.

| | Scene | v1 | v2 |
|---|---|---|---|
| 01 | Lamp-lit still life | [S1_StillLife_v1.blend](blend/S1_StillLife_v1.blend) | [S1_StillLife_v2.blend](blend/S1_StillLife_v2.blend) |
| 02 | Low-poly floating island | [S2_FloatingIsland_v1.blend](blend/S2_FloatingIsland_v1.blend) | [S2_FloatingIsland_v2.blend](blend/S2_FloatingIsland_v2.blend) |
| 03 | Rainy neon city street | [S3_NeonCity_v1.blend](blend/S3_NeonCity_v1.blend) | [S3_NeonCity_v2.blend](blend/S3_NeonCity_v2.blend) |

Open the `.blend` files in **Blender 5.2 or newer**. They render with Cycles.

## What is in here

| Folder | What |
|---|---|
| `blend/` | The six Blender files. |
| `renders/` | The six final 1920×1080 Cycles renders (PNG). |
| `scripts/v1/`, `scripts/v2/` | The Python scripts the model wrote. Each one builds its scene from an empty file. |
| `models/` | Web versions of the scenes: Draco-compressed GLB, a sky panorama and a small JSON with the camera and lights. |
| `viewer/` | The three.js viewer used on the page. |
| `tools/` | `export_glb.py` (Blender → GLB), `clean_blend.py`, `build_site.py`. |
| `vendor/` | three.js and the fonts, stored locally. |

## File sizes

| Scene | Version | .blend | GLB (web) | Triangles | Materials | Atlas |
|---|---|---|---|---|---|---|
| Still life | v1 | 0.23 MB | 0.44 MB | 19,076 | 17 | 2048 px |
| Still life | v2 | 2.38 MB | 2.06 MB | 182,492 | 24 | 4096 px |
| Floating island | v1 | 0.36 MB | 0.52 MB | 13,020 | 23 | 2048 px |
| Floating island | v2 | 1.34 MB | 3.16 MB | 50,312 | 11 | 4096 px |
| Neon city | v1 | 0.95 MB | 7.04 MB | 94,350 | 85 | 4096 px |
| Neon city | v2 | 7.50 MB | 11.07 MB | 795,310 | 191 | 4096 px |

Each GLB loads only when a visitor clicks "Load the 3D scene". Whole repository: about 58 MB. No file is over 100 MB, so nothing needs Git LFS.

## The 3D viewers are a preview, not the render

The `.blend` files use procedural node materials, volumes and path-traced light. None of that exists in a browser, so `tools/export_glb.py` does this:

- Materials with fixed colours are exported as they are.
- Procedural materials are **baked into one texture atlas per scene** (colour, glow, roughness).
- Text and curves become meshes. All modifiers are applied. Everything is joined into one mesh.
- Fog and steam volumes are dropped. The viewer adds simple distance fog instead.
- The render camera and every light are written to a JSON file. The viewer starts at the render camera.

What looks different from Blender:

- **Still life:** procedural wood, cloth, glaze and brass patterns are textures now and go soft when you zoom far in. Bump detail is flat. The steam over the mug (a volume) is missing. No bounce light and no depth of field.
- **Floating island:** the sky is a still panorama. Shadows are real-time and sharper. Waterfall and smoke lose their soft light scatter.
- **Neon city:** fog is simple distance fog, so no light shafts or searchlight beams. Only the strongest few lamps are real lights; the rest of the glow is the neon itself plus bloom. The wet road is one flat mirror layer, not the puddle map. Rain does not move.
- **All scenes:** real-time lighting was tuned by eye against the renders. It is close, not equal.

## Rebuild

```bash
# export one scene (run with Blender 5.2+)
blender -b blend/S1_StillLife_v2.blend --python tools/export_glb.py -- models s1_v2 4096 16

# serve the page locally
python3 -m http.server 8000
```

## Licence

Code: MIT. Scenes, renders and models: CC BY 4.0. Third-party files keep their own licences. See [LICENSE](LICENSE) and [CREDITS.md](CREDITS.md).
