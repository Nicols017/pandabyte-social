# PandaByte video studio

Turns a storyboard JSON into a vertical MP4 (1080x1920, 30 fps) with Nibble's voice,
animated scenes and burned-in captions.

```bash
pip install --break-system-packages kokoro-onnx soundfile pillow numpy   # ffmpeg must be installed
python3 studio/render.py storyboards/<slug>.json --stills   # key frames to check layout
python3 studio/render.py storyboards/<slug>.json            # full render -> build/<slug>/<slug>.mp4
```

Voice: Kokoro-82M (Apache-2.0), voice `am_puck`, en-us. Model files download on first run.

## Storyboard (new style, #30)

```json
{"slug": "2026-11-22-checks-the-listing", "series": "Nibble Checks the Listing - Ep. 1", "voice": "am_puck", "speed": 1.0,
 "source": "Source: ... official specs",
 "cover": {"text": "3 RED *FLAGS?*", "pose": "suspicious"},
 "scenes": [{"type": "game|versus|stat|hook|cta", "title": "Text with *orange* words",
             "poses": [["suspicious", 0], ["surprised", "@1"]],
             "say": ["spoken line", {"text": "RX 7600", "speak": "R X seventy-six hundred"}, {"pause": 2.0}]}]}
```

Motion in every scene: push-in (`push`, default 0.04), word-by-word kinetic titles, orange circle/underline
on any word with a number, boxes sliding in. Captions keep numbers orange.

Times: seconds from scene start, `"@n"` (start of say line n), `"@n+0.5"`, `"@ne-0.7"` (end of line n).

- `poses`: Nibble changes pose at each time with a small jump (or `"pose"` for one). The render stops if any
  pose is used more than 2x in a video (cover pose counts when it differs from the first scene's pose).
- `cover` (storyboard level): at most 4 words; it is drawn over the first scene and is the first 8 frames of the
  video; writes `cover.jpg` and `cover-270x480.jpg`. Options: `size`, `y`, `pose_height`, `pose_x`, `pose_bottom`,
  `backdrop_dy`, `backdrop_opacity`, `cards` (versus).
- `game`: fictional listing card drawn in code (labelled SAMPLE / FICTIONAL, no store, logo or price number).
  `listing{title, condition, badge, was_label}`, `countdown{at, seconds, text}`, `flags[{target: condition|title|badge|image|word:<w>, at, kind: bad|good, label}]`,
  `stamp{text, at}`, `continue: true` (keep the card in place from the previous game scene).
- `versus`: `left`/`right` `{name, box, value, sub, reveal_at}`, `reveal_at`, `vs_at`, `highlight: left|right`, `highlight_at`,
  `boxes` (generic retail boxes, default true), `blocks` + `blocks_at` (1 block = 1 GB), `note`, `note_at`, `no_vs`.
- `stat` (title, value, value_at, label) · `hook` (title, subtitle) · `cta` (title, button, button_at, comment, comment_at, disclosure, footnote).
- `stills`: list of times for `--stills` (default 80% of the scene). `--stills` writes the stills, `contact.png` and the cover.
- `bullets` was removed as a scene type (no topic-list scenes).
- `sfx` (default true) / `sfx_volume`: short effects from `studio/sfx.py` (synthesized in code, no samples) mixed low:
  whoosh on cuts, pop on pose changes, ticks on the countdown, buzz/ding on flags, thump on VS and stamp.

Poses: gpu, cpu, motherboard, ram, ssd, laptop, mouse, keyboard, pointing, thumbsup, arms-crossed, suspicious, surprised, thinking.

## Quality check (Video Quality Agent)

A clean render is not an approval. Every video goes through `studio/qa.py` before it can be scheduled:

```bash
python3 studio/qa.py build/<slug>/<slug>.mp4 \
  --storyboard storyboards/<slug>.json \
  --package build/<slug>/package.json \
  --catalog ../pandabyte/data/catalog          # or set PANDABYTE_CATALOG
```

It writes `<video>.qa.json` (schema `schemas/video-qa-report.schema.json`) and `<video>.qa.png`
(contact sheet with the 1080x1920 safe-area box). Exit code: 0 pass, 1 fail, 2 needs review.
Checks: H.264/yuv420p 1080x1920 30 fps, AAC, duration, A/V length, decode errors, loudness
(-16 to -12 LUFS, true peak <= -1 dBTP), silence, black/frozen frames, faststart; spec claims in
`bars`/`compare` scenes against the catalog; on-screen source; on-screen + spoken disclosure;
Nibble poses and voice; asset licenses, affiliate links and caption disclosure from the package
(`schemas/video-package.schema.json`). Items marked `review` are settled by agent 14.
