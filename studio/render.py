"""PandaByte video studio: storyboard JSON -> vertical MP4 (1080x1920, 30 fps).

Pipeline per scene: Kokoro narration (Nibble's voice) -> animated frames (PIL) piped to
ffmpeg -> scenes joined with the concat demuxer -> narration muxed -> burned-in captions
are drawn per frame from the narration timing.

Usage:
    python3 studio/render.py storyboards/<slug>.json            # full render
    python3 studio/render.py storyboards/<slug>.json --stills    # key frames only (PNG)

Requirements: pip install kokoro-onnx soundfile pillow numpy ; ffmpeg on PATH.
Kokoro model files are downloaded on first use from the kokoro-onnx GitHub releases
(model weights: Apache-2.0, commercial use allowed).
"""
import json, math, os, re, subprocess, sys, urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / 'assets' / 'nibble'
CACHE = Path(os.environ.get('PANDABYTE_STUDIO_CACHE', Path.home() / '.cache' / 'pandabyte-studio'))
MODEL_URL = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/'

W, H, FPS = 1080, 1920, 30
SAFE_L, SAFE_R, SAFE_T, SAFE_B = 60, 960, 220, 1500  # platform safe area (see studio/qa.py)
SAFE_W = SAFE_R - SAFE_L
BG = (18, 18, 20)
ORANGE = (255, 138, 61)
WHITE = (245, 245, 247)
GRAY = (165, 165, 175)
PANEL = (32, 32, 36)

# ---------- fonts ----------
def _font(candidates, size):
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()

POPPINS = ['/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf']
INTER = ['/usr/share/fonts/opentype/inter/Inter-Medium.otf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
INTERB = ['/usr/share/fonts/opentype/inter/Inter-Bold.otf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']
_fc = {}
def H1(s): return _fc.setdefault(('h', s), _font(POPPINS + INTERB, s))
def BODY(s): return _fc.setdefault(('b', s), _font(INTER, s))
def BOLD(s): return _fc.setdefault(('B', s), _font(INTERB, s))

# ---------- easing ----------
def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease_out(x): x = clamp(x); return 1 - (1 - x) ** 3
def back_out(x):
    x = clamp(x); c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2
def prog(t, start, dur): return clamp((t - start) / dur) if dur > 0 else 1.0

# ---------- narration ----------
_kokoro = None
def kokoro():
    global _kokoro
    if _kokoro is None:
        from kokoro_onnx import Kokoro
        CACHE.mkdir(parents=True, exist_ok=True)
        for f in ('kokoro-v1.0.onnx', 'voices-v1.0.bin'):
            p = CACHE / f
            if not p.exists():
                print('downloading', f); urllib.request.urlretrieve(MODEL_URL + f, p)
        _kokoro = Kokoro(str(CACHE / 'kokoro-v1.0.onnx'), str(CACHE / 'voices-v1.0.bin'))
    return _kokoro

def narrate(lines, voice, speed, gap=0.28, tail=0.35, lead=0.15):
    """Returns (audio float32 @24k, caption cues [(start, end, text)])."""
    sr = 24000
    parts = [np.zeros(int(sr * lead), dtype=np.float32)]
    cues, t = [], lead
    for i, line in enumerate(lines):
        s, sr = kokoro().create(line, voice=voice, speed=speed, lang='en-us')
        s = s.astype(np.float32)
        d = len(s) / sr
        cues += chunk_cues(line, t, t + d)
        parts.append(s); t += d
        pause = gap if i < len(lines) - 1 else tail
        parts.append(np.zeros(int(sr * pause), dtype=np.float32)); t += pause
    return np.concatenate(parts), cues

def chunk_cues(line, start, end, max_words=4):
    words = line.split()
    n = max(1, math.ceil(len(words) / max_words))
    base, extra = divmod(len(words), n)
    chunks, i = [], 0
    for k in range(n):
        size = base + (1 if k < extra else 0)
        chunks.append(' '.join(words[i:i + size])); i += size
    total = sum(len(c) for c in chunks) or 1
    cues, t = [], start
    for c in chunks:
        d = (end - start) * len(c) / total
        cues.append((t, t + d, c)); t += d
    return cues

# ---------- drawing helpers ----------
_img = {}
def pose(name):
    if name not in _img:
        p = Image.open(ASSETS / f'{name}.png').convert('RGBA')
        _img[name] = p.crop(p.getbbox())
    return _img[name]

def make_background():
    big = Image.new('RGB', (W + 400, H + 400), BG)
    g = ImageDraw.Draw(big)
    g.ellipse((W - 500, H - 900, W + 600, H + 300), fill=(78, 40, 16))
    g.ellipse((-300, 300, 500, 1100), fill=(40, 24, 14))
    return big.filter(ImageFilter.GaussianBlur(190))
_bg = None
def background(t):
    global _bg
    if _bg is None: _bg = make_background()
    ox = int(200 + 60 * math.sin(t * 0.5)); oy = int(200 + 80 * math.cos(t * 0.4))
    return _bg.crop((ox, oy, ox + W, oy + H)).copy()

def rich_words(text):
    out = []
    for tok in re.findall(r'\*[^*]+\*|\S+', text):
        if tok.startswith('*'):
            out += [(w, ORANGE) for w in tok.strip('*').split()]
        else:
            out.append((tok, WHITE))
    return out

def wrap(d, words, fnt, maxw):
    lines, cur, cw = [], [], 0
    sp = d.textlength(' ', font=fnt)
    for w, c in words:
        ww = d.textlength(w, font=fnt)
        if cur and cw + sp + ww > maxw:
            lines.append(cur); cur, cw = [], 0
        cur.append((w, c)); cw += (sp if len(cur) > 1 else 0) + ww
    if cur: lines.append(cur)
    return lines

def draw_rich(layer, x, y, text, fnt, maxw, lh, alpha=255, center=False):
    d = ImageDraw.Draw(layer)
    sp = d.textlength(' ', font=fnt)
    for line in wrap(d, rich_words(text), fnt, maxw):
        lw = sum(d.textlength(w, font=fnt) for w, _ in line) + sp * (len(line) - 1)
        xx = x + (maxw - lw) / 2 if center else x
        for w, c in line:
            d.text((xx, y), w, font=fnt, fill=c + (alpha,))
            xx += d.textlength(w, font=fnt) + sp
        y += lh
    return y

def paste_pose(frame, name, cx, bottom, height, t, start=0.0):
    p = prog(t, start, 0.45)
    if p <= 0: return
    s = 0.55 + 0.45 * back_out(p)
    bob = 7 * math.sin((t - start) * 2.6)
    img = pose(name)
    h = max(2, int(height * s)); w = max(2, int(img.width * h / img.height))
    im = img.resize((w, h), Image.LANCZOS)
    if p < 1:
        a = im.getchannel('A').point(lambda v: int(v * clamp(p * 1.6)))
        im.putalpha(a)
    frame.alpha_composite(im, (int(cx - w / 2), int(bottom - h + bob)))

def header(layer, t):
    d = ImageDraw.Draw(layer)
    d.text((70, 110), 'PANDABYTE', font=BOLD(34), fill=ORANGE + (255,))

def title(layer, text, t, y=230, size=96, start=0.05):
    p = ease_out(prog(t, start, 0.45))
    if p <= 0: return y
    off = int((1 - p) * 40)
    return draw_rich(layer, 70, y + off, text, H1(size), SAFE_R - 10 - 70, int(size * 1.18), alpha=int(255 * p))

def caption(layer, cues, t):
    cur = next((c for c in cues if c[0] <= t < c[1]), None)
    if not cur: return
    d = ImageDraw.Draw(layer)
    words = cur[2].split()
    size = 62  # shrink to fit the safe area (x 60..960, bottom <= 1500)
    while True:
        fnt = H1(size)
        sp = d.textlength(' ', font=fnt)
        tw = sum(d.textlength(w, font=fnt) for w in words) + sp * (len(words) - 1)
        if tw + 2 * 8 <= SAFE_W - 40 or size <= 40: break  # 8 px stroke on each side
        size -= 2
    x = SAFE_L + (SAFE_W - tw) / 2; y = 1400  # centered in the safe area, not the frame
    pop = back_out(prog(t, cur[0], 0.12))
    y += int((1 - pop) * 14)
    for w in words:
        col = ORANGE if re.search(r'\d', w) else WHITE
        d.text((x, y), w, font=fnt, fill=col + (255,), stroke_width=8, stroke_fill=(10, 10, 12, 255))
        x += d.textlength(w, font=fnt) + sp

def footer_source(layer, text):
    if text:
        ImageDraw.Draw(layer).text((70, 1330), text, font=BODY(30), fill=GRAY + (255,))

# ---------- scene types ----------
def scene_hook(L, sc, t):
    y = title(L, sc['title'], t, y=300, size=110)
    if sc.get('subtitle'):
        p = ease_out(prog(t, 0.35, 0.4))
        draw_rich(L, 70, y + 20, sc['subtitle'], BODY(52), W - 140, 68, alpha=int(255 * p))
    paste_pose(L, sc.get('pose', 'surprised'), W // 2, 1360, 640, t, 0.2)

def scene_stat(L, sc, t):
    y = title(L, sc['title'], t)
    p = ease_out(prog(t, 0.4, 0.9))
    val = sc['value']
    m = re.match(r'([\d,.]+)(.*)', val)
    if m:
        num = float(m.group(1).replace(',', ''))
        shown = num * p
        txt = (f'{shown:,.0f}' if num >= 100 or num == int(num) else f'{shown:.1f}') + m.group(2)
    else:
        txt = val
    d = ImageDraw.Draw(L)
    d.text((70, y + 40), txt, font=H1(190), fill=ORANGE + (int(255 * clamp(p * 2)),))
    if sc.get('label'):
        draw_rich(L, 70, y + 290, sc['label'], BODY(48), W - 140, 64, alpha=int(255 * ease_out(prog(t, 0.6, 0.4))))
    paste_pose(L, sc.get('pose', 'pointing'), W - 290, 1300, 430, t, 0.3)

def scene_bars(L, sc, t):
    y = title(L, sc['title'], t) + 40
    d = ImageDraw.Draw(L)
    bars = sc['bars']; vmax = max(b['value'] for b in bars)
    maxw = W - 140 - 500
    for i, b in enumerate(bars):
        p = ease_out(prog(t, 0.45 + i * 0.35, 0.7))
        a = int(255 * clamp(prog(t, 0.35 + i * 0.35, 0.2)))
        col = ORANGE if b.get('highlight') else GRAY
        d.text((70, y), b['label'], font=BOLD(42), fill=col + (a,))
        w = max(1, int(maxw * b['value'] / vmax * p))
        d.rounded_rectangle((70, y + 62, 70 + w, y + 142), radius=22, fill=col + (a,))
        d.text((70 + w + 24, y + 102), f"{b['value'] * p:.0f} {b.get('unit', '')}".strip(), font=H1(58), fill=WHITE + (a,), anchor='lm')
        y += 190
    paste_pose(L, sc.get('pose', 'pointing'), W - 260, 1310, 330, t, 0.3)

def scene_compare(L, sc, t):
    y = title(L, sc['title'], t) + 30
    d = ImageDraw.Draw(L)
    ca, cb = 640, 900
    a0 = int(255 * clamp(prog(t, 0.3, 0.3)))
    d.text((ca, y), sc['cols'][0], font=BOLD(36), fill=GRAY + (a0,), anchor='mt')
    d.text((cb, y), sc['cols'][1], font=BOLD(36), fill=ORANGE + (a0,), anchor='mt')
    y += 70
    for i, row in enumerate(sc['rows']):
        p = ease_out(prog(t, 0.45 + i * 0.3, 0.35)); a = int(255 * p); off = int((1 - p) * 30)
        same = row[1] == row[2]
        d.rounded_rectangle((50, y - 16 + off, W - 50, y + 92 + off), radius=24, fill=PANEL + (a,))
        d.text((84, y + 38 + off), row[0], font=BODY(34), fill=GRAY + (a,), anchor='lm')
        d.text((ca, y + 38 + off), row[1], font=H1(44), fill=WHITE + (a,), anchor='mm')
        d.text((cb, y + 38 + off), row[2], font=H1(44), fill=(WHITE if same else ORANGE) + (a,), anchor='mm')
        y += 128
    paste_pose(L, sc.get('pose', 'thinking'), W - 270, 1300, 380, t, 0.3)

def scene_bullets(L, sc, t):
    y = title(L, sc['title'], t) + 30
    d = ImageDraw.Draw(L)
    for i, it in enumerate(sc['items']):
        p = ease_out(prog(t, 0.4 + i * 0.45, 0.35)); a = int(255 * p); off = int((1 - p) * 40)
        d.ellipse((70 + off, y + 6, 130 + off, y + 66), fill=ORANGE + (a,))
        d.text((100 + off, y + 36), str(i + 1), font=H1(36), fill=BG + (a,), anchor='mm')
        draw_rich(L, 160 + off, y + 4, it, BOLD(46), W - 260, 60, alpha=a)
        y += 130
    paste_pose(L, sc.get('pose', 'pointing'), W - 270, 1300, 380, t, 0.3)

def scene_cta(L, sc, t):
    y = title(L, sc['title'], t, y=300, size=100)
    p = back_out(prog(t, 0.5, 0.4))
    if p > 0:
        d = ImageDraw.Draw(L)
        bw, bh = int(620 * p), int(130 * p)
        cx, cy = W // 2, y + 140
        d.rounded_rectangle((cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2), radius=bh // 2, fill=ORANGE + (255,))
        if p > 0.6:
            d.text((cx, cy), sc.get('button', 'pandabyte.tech'), font=H1(58), fill=BG + (255,), anchor='mm')
    if sc.get('disclosure'):
        ImageDraw.Draw(L).text((SAFE_L + SAFE_W // 2, 1330), sc['disclosure'], font=BODY(28), fill=GRAY + (255,), anchor='mm')
    paste_pose(L, sc.get('pose', 'thumbsup'), W // 2, 1290, 560, t, 0.25)

SCENES = {'hook': scene_hook, 'stat': scene_stat, 'bars': scene_bars, 'compare': scene_compare,
          'bullets': scene_bullets, 'cta': scene_cta}

def render_frame(sc, t, cues, source):
    frame = background(t + sc.get('_offset', 0)).convert('RGBA')
    layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    header(layer, t)
    SCENES[sc['type']](layer, sc, t)
    if sc['type'] in ('bars', 'compare', 'stat') or sc.get('source'):
        footer_source(layer, sc.get('source', source))
    caption(layer, cues, t)
    frame.alpha_composite(layer)
    return frame.convert('RGB')

# ---------- render ----------
def render(sb_path, stills=False):
    sb = json.loads(Path(sb_path).read_text())
    slug = sb['slug']
    out = ROOT / 'build' / slug; out.mkdir(parents=True, exist_ok=True)
    voice, speed = sb.get('voice', 'am_puck'), sb.get('speed', 1.0)
    audio_parts, scene_files, offset = [], [], 0.0
    for i, sc in enumerate(sb['scenes']):
        audio, cues = narrate(sc['say'], voice, speed)
        nframes = max(1, round(len(audio) / 24000 * FPS))
        dur = nframes / FPS
        audio = np.pad(audio, (0, max(0, int(dur * 24000) - len(audio))))[:int(dur * 24000)]
        audio_parts.append(audio)
        sc['_offset'] = offset; offset += dur
        if stills:
            for frac in (0.15, 0.6, 0.95):
                render_frame(sc, dur * frac, cues, sb.get('source')).save(out / f'still-{i + 1:02d}-{int(frac * 100)}.jpg', quality=90)
            continue
        tmp = out / f'scene-{i + 1:02d}.tmp.mp4'; final = out / f'scene-{i + 1:02d}.mp4'
        proc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                 '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast',
                                 '-crf', '20', '-pix_fmt', 'yuv420p', '-r', str(FPS), str(tmp)], stdin=subprocess.PIPE)
        for f in range(nframes):
            proc.stdin.write(render_frame(sc, f / FPS, cues, sb.get('source')).tobytes())
        proc.stdin.close()
        if proc.wait() != 0: raise SystemExit(f'ffmpeg failed on scene {i + 1}')
        tmp.rename(final); scene_files.append(final)
        print(f'scene {i + 1}/{len(sb["scenes"])} done ({dur:.1f}s)')
    if stills:
        print('stills in', out); return
    sf.write(out / 'narration.wav', np.concatenate(audio_parts), 24000)
    (out / 'concat.txt').write_text(''.join(f"file '{p.name}'\n" for p in scene_files))
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(out / 'concat.txt'),
                    '-c', 'copy', str(out / 'video-only.mp4')], check=True)
    final = out / f'{slug}.mp4'; tmp = out / f'{slug}.tmp.mp4'
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(out / 'video-only.mp4'), '-i', str(out / 'narration.wav'),
                    '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11', '-ar', '48000',
                    '-c:a', 'aac', '-b:a', '160k', '-shortest',
                    '-movflags', '+faststart', str(tmp)], check=True)
    tmp.rename(final)
    print('done:', final, f'{offset:.1f}s')

if __name__ == '__main__':
    render(sys.argv[1], stills='--stills' in sys.argv)
