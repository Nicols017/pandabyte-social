# PandaByte video studio

Turns a storyboard JSON into a vertical MP4 (1080x1920, 30 fps) with Nibble's voice,
animated scenes and burned-in captions.

```bash
pip install --break-system-packages kokoro-onnx soundfile pillow numpy   # ffmpeg must be installed
python3 studio/render.py storyboards/<slug>.json --stills   # key frames to check layout
python3 studio/render.py storyboards/<slug>.json            # full render -> build/<slug>/<slug>.mp4
```

Voice: Kokoro-82M (Apache-2.0), voice `am_puck`, en-us. Model files download on first run.

## Storyboard

```json
{"slug": "2026-11-22-what-is-vram", "voice": "am_puck", "speed": 1.0,
 "source": "Source: ... official specs",
 "scenes": [{"type": "hook|stat|bars|compare|bullets|cta", "pose": "<assets/nibble name>", "title": "Text with *orange* words", "say": ["one spoken line", "another"]}]}
```

Scene fields: `hook` (title, subtitle) · `stat` (title, value, label) · `bars` (title, bars[{label, value, unit, highlight}])
· `compare` (title, cols[2], rows[[label, a, b]]) · `bullets` (title, items[]) · `cta` (title, button, disclosure).
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
