"""PandaByte video studio: storyboard JSON -> vertical MP4 (1080x1920, 30 fps).

"New style" engine (#30): continuous motion in every scene (push-in, word-by-word kinetic
titles, orange circle/underline on numbers, sliding boxes), Nibble acting in time with the
narration (pose changes with a small jump), game/versus scenes, a cover that is also the
first frame, and short synthesized sound effects mixed low under the voice.

Usage:
    python3 studio/render.py storyboards/<slug>.json            # full render
    python3 studio/render.py storyboards/<slug>.json --stills   # key frames + contact sheet + cover
    add --no-voice to time scenes from a words-per-second estimate (no Kokoro needed)
    add --vo <dir> to use external voice files <dir>/<storyboard vo.prefix><scene#>.mp3 (or "audio" per scene);
    line timing and captions then come from ffmpeg silencedetect. Kokoro stays the fallback.

Requirements: pip install kokoro-onnx soundfile pillow numpy ; ffmpeg on PATH.
Kokoro model files are downloaded on first use from the kokoro-onnx GitHub releases
(model weights: Apache-2.0, commercial use allowed).
"""
import json, math, os, re, subprocess, sys, time, urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sfx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / 'assets' / 'nibble'
CACHE = Path(os.environ.get('PANDABYTE_STUDIO_CACHE', Path.home() / '.cache' / 'pandabyte-studio'))
MODEL_URL = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/'

W, H, FPS = 1080, 1920, 30
SR = 48000                  # audio buffers (Kokoro 24 kHz is resampled; external MP3s are decoded here)
sfx.SR = SR
SAFE_L, SAFE_R, SAFE_T, SAFE_B = 60, 960, 220, 1500  # platform safe area (see studio/qa.py)
SAFE_W = SAFE_R - SAFE_L
X0 = 84                     # content left margin: stays inside the safe area after the push-in
CX, CY = W // 2, 860        # push-in anchor (centre of the safe area)
BG = (18, 18, 20)
ORANGE = (255, 138, 61)
WHITE = (245, 245, 247)
GRAY = (165, 165, 175)
PANEL = (32, 32, 36)
RED = (235, 64, 52)
GREEN = (60, 200, 120)
INK = (24, 24, 28)
MAX_POSE_USES = 2           # acceptance: the same pose at most twice per video
COVER_FRAMES = 15           # owner cover rule: frames 0.0-0.5 s are the cover itself

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
def ease_io(x): x = clamp(x); return x * x * (3 - 2 * x)
def back_out(x):
    x = clamp(x); c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (x - 1) ** 3 + c1 * (x - 1) ** 2
def prog(t, start, dur): return clamp((t - start) / dur) if dur > 0 else float(t >= start)

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

def say_text(item):
    """Caption text of a say item: "line", {"text": shown, "speak": phonetic} or {"pause": s}."""
    return item if isinstance(item, str) else item.get('text', '')

def narrate(lines, voice, speed, use_voice=True, gap=0.22, tail=0.3, lead=0.12):
    """Returns (audio float32 @24k, caption cues [(start, end, text)], line starts, line ends).

    A `say` item is a spoken line (str), {"text": caption, "speak": what Kokoro reads}
    (e.g. "RX 7600" read as "R X seventy-six hundred") or {"pause": seconds} (silence, no caption)."""
    parts = [np.zeros(int(SR * lead), dtype=np.float32)]
    cues, starts, ends, t = [], [], [], lead
    for i, line in enumerate(lines):
        starts.append(t)
        if isinstance(line, dict) and 'pause' in line:
            d = float(line['pause'])
            parts.append(np.zeros(int(SR * d), dtype=np.float32)); t += d; ends.append(t)
            continue
        spoken = line.get('speak', line.get('text')) if isinstance(line, dict) else line
        line = say_text(line)
        if use_voice:
            s, sr = kokoro().create(spoken, voice=voice, speed=speed, lang='en-us')
            s = resample(s.astype(np.float32), sr)
        else:  # layout preview: ~2.7 words/s
            s = np.zeros(int(SR * max(0.6, len(line.split()) / 2.7 / speed)), dtype=np.float32)
        d = len(s) / SR
        cues += chunk_cues(line, t, t + d)
        parts.append(s); t += d; ends.append(t)
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        pause = 0 if (isinstance(nxt, dict) and 'pause' in nxt) else (gap if nxt is not None else tail)
        parts.append(np.zeros(int(SR * pause), dtype=np.float32)); t += pause
    return np.concatenate(parts), cues, starts, ends

def resample(x, sr):
    if sr == SR: return x
    n = int(len(x) * SR / sr)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)

def load_audio(path):
    raw = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(path), '-f', 'f32le', '-ac', '1', '-ar', str(SR), '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()

def speech_segments(path, noise='-38dB', min_sil=0.2):
    """[(start, end)] of speech inside an audio file, from ffmpeg silencedetect."""
    log = subprocess.run(['ffmpeg', '-hide_banner', '-i', str(path), '-af', f'silencedetect=noise={noise}:d={min_sil}', '-f', 'null', '-'],
                         capture_output=True, text=True).stderr
    dur = float(re.search(r'Duration: (\d+):(\d+):([\d.]+)', log).group(3)) + 60 * int(re.search(r'Duration: (\d+):(\d+)', log).group(2))
    sil_s = [float(x) for x in re.findall(r'silence_start: (-?[\d.]+)', log)]
    sil_e = [float(x) for x in re.findall(r'silence_end: ([\d.]+)', log)]
    sils = [(max(0.0, a), sil_e[i] if i < len(sil_e) else dur) for i, a in enumerate(sil_s)]
    segs, t = [], 0.0
    for a, b in sils:
        if a - t > 0.05: segs.append((t, a))
        t = b
    if dur - t > 0.05: segs.append((t, dur))
    return segs, dur

def fit_segments(segs, lines):
    """Groups speech segments so there is one per spoken line (merges across the shortest gaps)."""
    segs = list(segs)
    while len(segs) > len(lines):
        k = min(range(len(segs) - 1), key=lambda i: segs[i + 1][0] - segs[i][1])
        segs[k:k + 2] = [(segs[k][0], segs[k + 1][1])]
    while len(segs) < len(lines):  # fewer pauses than lines: split the longest segment by text length
        k = max(range(len(segs)), key=lambda i: segs[i][1] - segs[i][0])
        a, b = segs[k]; m = a + (b - a) * 0.5
        segs[k:k + 1] = [(a, m), (m, b)]
    return segs

def narrate_external(lines, path, tail=0.3):
    """Scene narration from an external voice file (e.g. ElevenLabs). Line timing and captions come from
    silencedetect; {"pause": s} items insert silence at that point (the pause is not in the file)."""
    audio = load_audio(path)
    spoken = [l for l in lines if not (isinstance(l, dict) and 'pause' in l)]
    segs, dur = speech_segments(path)
    if not segs: raise SystemExit(f'no speech found in {path}')
    segs = fit_segments(segs, spoken)
    # cut points between lines: middle of the silence after each line
    cuts = [(segs[k][1] + segs[k + 1][0]) / 2 for k in range(len(segs) - 1)] + [dur]
    parts, cues, starts, ends = [], [], [], []
    t_out, src, k = 0.0, 0.0, 0
    for line in lines:
        if isinstance(line, dict) and 'pause' in line:
            d = float(line['pause']); starts.append(t_out)
            parts.append(np.zeros(int(SR * d), dtype=np.float32)); t_out += d; ends.append(t_out)
            continue
        a, b = segs[k]; cut = cuts[k]
        shift = t_out - src
        starts.append(a + shift); ends.append(b + shift)
        cues += chunk_cues(say_text(line), a + shift, b + shift)
        parts.append(audio[int(src * SR):int(cut * SR)]); t_out += cut - src; src = cut; k += 1
    parts.append(np.zeros(int(SR * tail), dtype=np.float32))
    return np.concatenate(parts), cues, starts, ends

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

def at(v, ctx, default=0.0):
    """Scene-relative time: a number (seconds) or "@n" / "@n+0.4" (start of say line n) or "@ne-0.5" (end of line n)."""
    if v is None: return default
    if isinstance(v, (int, float)): return float(v)
    m = re.fullmatch(r'@(\d+)(e?)(?:\s*([+-])\s*([\d.]+))?', str(v).strip())
    if not m: raise SystemExit(f'bad time value: {v!r} (use seconds or "@<line>[+s]")')
    i = int(m.group(1))
    base = (ctx['ends'] if m.group(2) else ctx['starts'])[min(i, len(ctx['starts']) - 1)]
    off = float(m.group(4) or 0) * (-1 if m.group(3) == '-' else 1)
    return base + off

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

def has_num(w): return bool(re.search(r'\d', w))

def mark(d, box, p, style='circle', color=ORANGE, width=9, pad=(26, 20)):
    """Hand-drawn orange circle or underline that draws itself as p goes 0 -> 1."""
    if p <= 0: return
    x0, y0, x1, y1 = box
    if style == 'underline':
        n = 24; pts = []
        for k in range(int(n * p) + 1):
            u = k / n; pts.append((x0 - 6 + (x1 - x0 + 12) * u, y1 + 8 + 5 * math.sin(u * math.pi)))
        if len(pts) > 1: d.line(pts, fill=color + (255,), width=width, joint='curve')
        return
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    rx, ry = (x1 - x0) / 2 + pad[0], (y1 - y0) / 2 + pad[1]
    a0, sweep, n = math.radians(-110), math.radians(385), 48
    pts = []
    for k in range(int(n * p) + 1):
        u = k / n; a = a0 + sweep * u; grow = 1 + 0.06 * u
        pts.append((cx + rx * grow * math.cos(a), cy + ry * grow * math.sin(a) - 4 * u))
    if len(pts) > 1: d.line(pts, fill=color + (255,), width=width, joint='curve')

def kinetic(layer, x, y, text, fnt, maxw, lh, t, start=0.05, stagger=0.09, center=False, mark_style='underline', mark_nums=True):
    """Word-by-word title: each word pops up in turn; words with digits get an orange mark."""
    d = ImageDraw.Draw(layer)
    sp = d.textlength(' ', font=fnt)
    k = 0; asc = fnt.getbbox('Hg')
    for line in wrap(d, rich_words(text), fnt, maxw):
        lw = sum(d.textlength(w, font=fnt) for w, _ in line) + sp * (len(line) - 1)
        xx = x + (maxw - lw) / 2 if center else x
        for w, c in line:
            ws = start + k * stagger
            p = back_out(prog(t, ws, 0.3)); a = int(255 * clamp(prog(t, ws, 0.18)))
            ww = d.textlength(w, font=fnt)
            if a > 0:
                d.text((xx, y + int((1 - p) * 34)), w, font=fnt, fill=c + (a,))
                if mark_nums and has_num(w):
                    mark(d, (xx, y + asc[1], xx + ww, y + asc[3]), ease_out(prog(t, ws + 0.3, 0.45)), mark_style)
            xx += ww + sp; k += 1
        y += lh
    return y

def draw_rich(layer, x, y, text, fnt, maxw, lh, alpha=255, center=False, stroke=0):
    d = ImageDraw.Draw(layer)
    sp = d.textlength(' ', font=fnt)
    for line in wrap(d, rich_words(text), fnt, maxw):
        lw = sum(d.textlength(w, font=fnt) for w, _ in line) + sp * (len(line) - 1)
        xx = x + (maxw - lw) / 2 if center else x
        for w, c in line:
            d.text((xx, y), w, font=fnt, fill=c + (alpha,), stroke_width=stroke, stroke_fill=(10, 10, 12, alpha))
            xx += d.textlength(w, font=fnt) + sp
        y += lh
    return y

def title(layer, text, t, y=236, size=88, start=0.05):
    return kinetic(layer, X0, y, text, H1(size), SAFE_R - 20 - X0, int(size * 1.16), t, start)

# ---------- Nibble ----------
def pose_track(sc, ctx):
    """[(pose, start_s)] from "poses": [[name, time], ...] or the single "pose"."""
    if 'poses' in sc:
        return [(p, at(v, ctx)) for p, v in sc['poses']]
    return [(sc['pose'], 0.0)] if sc.get('pose') else []

def nibble(L, sc, t, ctx, cx, bottom, height):
    track = ctx['track']
    if not track: return
    idx = max([i for i, (_, s) in enumerate(track) if s <= t] or [0])
    name, s0 = track[idx]
    img = pose(name)
    if idx == 0 and not ctx.get('continues'):
        p = prog(t, max(0.0, s0) + 0.12, 0.42)  # first entrance: scale up + fade
        if p <= 0: return
        scale, lift, alpha = 0.55 + 0.45 * back_out(p), 0, clamp(p * 1.6)
    else:  # pose change: small jump with a squash on landing
        dt = t - s0
        hop = math.sin(math.pi * clamp(dt / 0.34)) if dt < 0.34 else 0.0
        land = prog(dt, 0.30, 0.14)
        squash = 1 - 0.06 * math.sin(math.pi * land) if 0 < land < 1 else 1.0
        scale, lift, alpha = (1 + 0.05 * hop) * squash, int(52 * hop), 1.0
    bob = 6 * math.sin((t + ctx['offset']) * 2.6)
    h = max(2, int(height * scale)); w = max(2, int(img.width * h / img.height))
    im = img.resize((w, h), Image.BILINEAR)
    if alpha < 1:
        im.putalpha(im.getchannel('A').point(lambda v: int(v * alpha)))
    L.alpha_composite(im, (int(cx - w / 2), int(bottom - h - lift + bob)))

# ---------- props (drawn, generic, no logos) ----------
def draw_gpu(d, box, a=255):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    d.rounded_rectangle(box, radius=int(h * 0.12), fill=(46, 48, 54, a), outline=(90, 92, 100, a), width=4)
    d.rectangle((x0 + w * 0.12, y1 - 4, x0 + w * 0.55, y1 + h * 0.08), fill=(212, 170, 60, a))  # PCIe contacts
    r = h * 0.36
    for fx in (x0 + w * 0.29, x0 + w * 0.71):
        fy = y0 + h * 0.5
        d.ellipse((fx - r, fy - r, fx + r, fy + r), fill=(28, 29, 33, a), outline=(110, 112, 120, a), width=4)
        for k in range(7):
            ang = k * 2 * math.pi / 7
            d.line((fx, fy, fx + r * 0.86 * math.cos(ang), fy + r * 0.86 * math.sin(ang)), fill=(80, 82, 90, a), width=7)
        d.ellipse((fx - r * 0.24, fy - r * 0.24, fx + r * 0.24, fy + r * 0.24), fill=(60, 62, 68, a))

def draw_box(L, box, name, a=255, accent=ORANGE):
    """Generic retail box: front face + side face, product name, no brand art."""
    d = ImageDraw.Draw(L)
    x0, y0, x1, y1 = box; dx = 26
    d.polygon([(x1, y0), (x1 + dx, y0 - dx), (x1 + dx, y1 - dx), (x1, y1)], fill=(30, 30, 34, a))
    d.polygon([(x0, y0), (x0 + dx, y0 - dx), (x1 + dx, y0 - dx), (x1, y0)], fill=(58, 58, 64, a))
    d.rectangle(box, fill=(40, 40, 46, a), outline=(80, 80, 88, a), width=3)
    d.polygon([(x0, y1 - (y1 - y0) * 0.38), (x1, y1 - (y1 - y0) * 0.62), (x1, y1 - (y1 - y0) * 0.5), (x0, y1 - (y1 - y0) * 0.26)], fill=accent + (a,))
    draw_gpu(d, (x0 + 40, y0 + 120, x1 - 40, y0 + 120 + (x1 - x0 - 80) * 0.42), a)
    fnt = H1(40 if len(name) < 14 else 34)
    d.text(((x0 + x1) / 2, y0 + 58), name, font=fnt, fill=WHITE + (a,), anchor='mm')
    d.text(((x0 + x1) / 2, y1 - 40), 'GRAPHICS CARD', font=BOLD(22), fill=GRAY + (a,), anchor='mm')

# ---------- scene types ----------
def scene_hook(L, sc, t, ctx):
    y = kinetic(L, X0, 300, sc['title'], H1(108), SAFE_R - 20 - X0, 124, t)
    if sc.get('subtitle'):
        p = ease_out(prog(t, 0.5, 0.4))
        draw_rich(L, X0, y + 20, sc['subtitle'], BODY(50), SAFE_R - 20 - X0, 66, alpha=int(255 * p))
    nibble(L, sc, t, ctx, W // 2, 1330, 640)

def scene_stat(L, sc, t, ctx):
    y = title(L, sc['title'], t)
    t0 = at(sc.get('value_at'), ctx, 0.4)
    p = ease_out(prog(t, t0, 0.8))
    val = sc['value']
    m = re.match(r'([\d,.]+)(.*)', val)
    if m:
        num = float(m.group(1).replace(',', ''))
        shown = num * p
        txt = (f'{shown:,.0f}' if num >= 100 or num == int(num) else f'{shown:.1f}') + m.group(2)
    else:
        txt = val
    d = ImageDraw.Draw(L)
    fnt = H1(190)
    d.text((X0, y + 40), txt, font=fnt, fill=ORANGE + (int(255 * clamp(p * 2)),))
    if p > 0:
        bb = d.textbbox((X0, y + 40), val, font=fnt)
        mark(d, bb, ease_out(prog(t, t0 + 0.8, 0.5)))
    if sc.get('label'):
        draw_rich(L, X0, y + 300, sc['label'], BODY(48), SAFE_R - 20 - X0, 64, alpha=int(255 * ease_out(prog(t, t0 + 0.3, 0.4))))
    nibble(L, sc, t, ctx, W - 290, 1310, 430)

def scene_cta(L, sc, t, ctx):
    y = kinetic(L, X0, 270, sc['title'], H1(92), SAFE_R - 20 - X0, 106, t)
    d = ImageDraw.Draw(L)
    if sc.get('comment'):  # comment box being typed into (an invitation, never a fake user comment)
        p = ease_out(prog(t, at(sc.get('comment_at'), ctx, 0.4), 0.4))
        if p > 0:
            off = int((1 - p) * 80); a = int(255 * p)
            bx = (X0, y + 30 + off, SAFE_R - 20, y + 150 + off)
            d.rounded_rectangle(bx, radius=60, fill=(44, 44, 50, a), outline=ORANGE + (a,), width=4)
            typed = sc['comment'][:int(len(sc['comment']) * clamp(prog(t, at(sc.get('comment_at'), ctx, 0.4) + 0.4, 0.8)))]
            d.text((bx[0] + 50, (bx[1] + bx[3]) / 2), typed or 'Add a comment...', font=BOLD(46),
                   fill=(WHITE if typed else GRAY) + (a,), anchor='lm')
            if int(t * 2) % 2 == 0 and typed != sc['comment']:
                tx = bx[0] + 50 + d.textlength(typed, font=BOLD(46)) + 4
                d.line((tx, bx[1] + 34, tx, bx[3] - 34), fill=ORANGE + (a,), width=4)
            y += 170
    p = back_out(prog(t, at(sc.get('button_at'), ctx, 0.6), 0.4))
    if p > 0:
        bw, bh = int(600 * p), int(120 * p)
        cx, cy = W // 2, y + 110
        d.rounded_rectangle((cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2), radius=max(1, bh // 2), fill=ORANGE + (255,))
        if p > 0.6:
            d.text((cx, cy), sc.get('button', 'pandabyte.tech'), font=H1(56), fill=BG + (255,), anchor='mm')
    nh = sc.get('nibble_height', 420)
    nibble(L, sc, t, ctx, W // 2, 1300, nh)

# ----- game: fictional listing, "find the red flags", countdown, reveal -----
CARD = (100, 470, 944, 1120)

def listing_layout(d, li):
    x0, y0, x1, y1 = CARD
    lay = {'image': (x0 + 150, y0 + 96, x1 - 150, y0 + 330)}
    fnt = BOLD(46); y = y0 + 372
    words = [(w, INK) for w in li['title'].split()]
    lines = wrap(d, words, fnt, x1 - x0 - 90)
    wboxes, sp, maxx = {}, d.textlength(' ', font=fnt), x0 + 45
    for line in lines:
        xx = x0 + 45
        for w, _ in line:
            ww = d.textlength(w, font=fnt); wboxes.setdefault(w, (xx, y + 4, xx + ww, y + 52)); xx += ww + sp
        maxx = max(maxx, xx - sp); y += 60
    lay['title'] = (x0 + 45, y0 + 376, maxx, y - 6); lay['title_lines'] = lines; lay['words'] = wboxes
    y += 22
    cond = f"Condition: {li.get('condition', '')}"
    lay['condition'] = (x0 + 45, y, x0 + 45 + d.textlength(cond, font=BODY(40)), y + 48); lay['cond_text'] = cond
    y += 84
    if li.get('badge'):
        bw = d.textlength(li['badge'], font=H1(44)) + 50
        lay['badge'] = (x0 + 45, y, x0 + 45 + bw, y + 70)
    return lay

def scene_game(L, sc, t, ctx):
    li = sc['listing']; d = ImageDraw.Draw(L)
    if sc.get('title'): title(L, sc['title'], t, size=sc.get('title_size', 80))
    lay = listing_layout(d, li)
    enter = 0.0 if ctx.get('continues') else 0.0
    x0, y0, x1, y1 = CARD
    def slide(i):  # boxes slide in one after another (only on the first game scene)
        if ctx.get('continues'): return 1.0, 0
        p = ease_out(prog(t, enter + 0.08 * i, 0.35)); return p, int((1 - p) * 120)
    p, off = slide(0); a = int(255 * p)
    d.rounded_rectangle((x0, y0 + off, x1, y1 + off), radius=34, fill=(242, 242, 245, a))
    d.rounded_rectangle((x0, y0 + off, x1, y0 + 70 + off), radius=34, fill=(222, 222, 228, a))
    d.rectangle((x0, y0 + 40 + off, x1, y0 + 70 + off), fill=(222, 222, 228, a))
    d.text((x0 + 36, y0 + 36 + off), 'SAMPLE LISTING', font=BOLD(28), fill=(90, 90, 100, a), anchor='lm')
    d.text((x1 - 36, y0 + 36 + off), 'FICTIONAL - NOT A REAL STORE', font=BOLD(22), fill=(130, 130, 140, a), anchor='rm')
    p, off = slide(1); a = int(255 * p)
    ix0, iy0, ix1, iy1 = lay['image']
    d.rounded_rectangle((ix0 - 40, iy0 - 16 + off, ix1 + 40, iy1 + 16 + off), radius=24, fill=(228, 228, 234, a))
    draw_gpu(d, (ix0 + 30, iy0 + 40 + off, ix1 - 30, iy1 - 50 + off), a)
    p, off = slide(2); a = int(255 * p)
    yy = lay['title'][1]
    for line in lay['title_lines']:
        xx = x0 + 45
        for w, _ in line:
            d.text((xx + off, yy), w, font=BOLD(46), fill=INK + (a,)); xx += d.textlength(w, font=BOLD(46)) + d.textlength(' ', font=BOLD(46))
        yy += 60
    p, off = slide(3); a = int(255 * p)
    cx0, cy0, _, _ = lay['condition']
    d.text((cx0 + off, cy0), lay['cond_text'], font=BODY(40), fill=(70, 70, 80, a))
    if 'badge' in lay:
        p, off = slide(4); a = int(255 * p)
        b = lay['badge']
        d.rounded_rectangle((b[0] + off, b[1], b[2] + off, b[3]), radius=16, fill=RED + (a,))
        d.text(((b[0] + b[2]) / 2 + off, (b[1] + b[3]) / 2), li['badge'], font=H1(44), fill=(255, 255, 255, a), anchor='mm')
        wx = b[2] + 30 + off
        d.text((wx, (b[1] + b[3]) / 2), li.get('was_label', 'Was'), font=BODY(38), fill=(110, 110, 120, a), anchor='lm')
        rx = wx + d.textlength(li.get('was_label', 'Was'), font=BODY(38)) + 18
        d.rounded_rectangle((rx, b[1] + 18, rx + 150, b[3] - 18), radius=8, fill=(200, 200, 208, a))  # no number on purpose
        d.line((rx - 8, (b[1] + b[3]) / 2, rx + 158, (b[1] + b[3]) / 2), fill=(110, 110, 120, a), width=4)
    # countdown ("find the red flags")
    if sc.get('countdown'):
        c0 = at(sc['countdown'].get('at'), ctx, 1.2); cd = float(sc['countdown'].get('seconds', 2.0))
        if c0 <= t < c0 + cd + 0.25:
            u = clamp((t - c0) / cd)
            cx, cy, r = (ix0 + ix1) / 2, (iy0 + iy1) / 2, 118
            pop_ = back_out(prog(t, c0, 0.25))
            rr = r * pop_
            d.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), fill=(18, 18, 20, 235))
            d.arc((cx - rr + 10, cy - rr + 10, cx + rr - 10, cy + rr - 10), -90, -90 + 360 * (1 - u), fill=ORANGE + (255,), width=16)
            n = max(1, math.ceil(cd * (1 - u))) if u < 1 else 0
            d.text((cx, cy), str(n) if n else 'GO', font=H1(110 if n else 80), fill=WHITE + (255,), anchor='mm')
            band = sc['countdown'].get('text', 'FIND THE 3 RED FLAGS')
            bw = d.textlength(band, font=H1(48)) + 60
            d.rounded_rectangle((W / 2 - bw / 2, iy1 + 4, W / 2 + bw / 2, iy1 + 80), radius=38, fill=ORANGE + (255,))
            d.text((W / 2, iy1 + 42), band, font=H1(48), fill=BG + (255,), anchor='mm')
    # flags: circle around the target + red X (bad) or orange check (good)
    for f in sc.get('flags', []):
        f0 = at(f.get('at'), ctx, 0.3)
        tgt = f['target']
        box = lay['words'].get(tgt[5:]) if tgt.startswith('word:') else lay.get(tgt)
        if not box: raise SystemExit(f'game flag target not found: {tgt}')
        good = f.get('kind') == 'good'
        col = ORANGE if good else RED
        mark(d, box, 1.0 if t >= f0 + 0.45 else ease_out(prog(t, f0, 0.45)), 'circle', col, width=8, pad=(16, 9))
        q = back_out(prog(t, f0 + 0.25, 0.3))
        if q > 0:
            bx, by, r = box[2] + 40, box[1] - 10, 34 * q
            bx = min(bx, x1 - 30)
            d.ellipse((bx - r, by - r, bx + r, by + r), fill=col + (255,))
            if good:
                d.line([(bx - r * 0.45, by), (bx - r * 0.1, by + r * 0.38), (bx + r * 0.5, by - r * 0.4)], fill=(255, 255, 255, 255), width=8)
            else:
                k = r * 0.42
                d.line((bx - k, by - k, bx + k, by + k), fill=(255, 255, 255, 255), width=8)
                d.line((bx - k, by + k, bx + k, by - k), fill=(255, 255, 255, 255), width=8)
            if f.get('label'):
                lf = BOLD(30); lw = d.textlength(f['label'], font=lf) + 32
                lx = min(max(box[0], x0 + 20), x1 - 20 - lw); ly = box[3] + 14
                if f.get('label_above'): ly = box[1] - 58
                d.rounded_rectangle((lx, ly, lx + lw, ly + 44), radius=22, fill=col + (int(255 * clamp(q)),))
                d.text((lx + lw / 2, ly + 22), f['label'], font=lf, fill=(255, 255, 255, int(255 * clamp(q))), anchor='mm')
    if sc.get('stamp'):
        s0 = at(sc['stamp'].get('at'), ctx, 0.5)
        q = prog(t, s0, 0.22)
        if q > 0:
            txt = sc['stamp']['text']; fnt = H1(110)
            tw = int(d.textlength(txt, font=fnt)) + 80
            st = Image.new('RGBA', (tw, 190), (0, 0, 0, 0)); sd = ImageDraw.Draw(st)
            sd.rounded_rectangle((6, 6, tw - 6, 184), radius=24, fill=(255, 255, 255, 215), outline=RED + (255,), width=14)
            sd.text((tw / 2, 97), txt, font=fnt, fill=RED + (255,), anchor='mm')
            sc_ = 1.8 - 0.8 * ease_out(q)
            st = st.resize((int(tw * sc_), int(190 * sc_)), Image.BILINEAR).rotate(-12, expand=True, resample=Image.BILINEAR)
            if q < 1: st.putalpha(st.getchannel('A').point(lambda v: int(v * q)))
            L.alpha_composite(st, (int(W / 2 - st.width / 2), int((y0 + y1) / 2 - st.height / 2)))
    nb = sc.get('nibble_at', [860, 1330, 420])
    nibble(L, sc, t, ctx, nb[0], nb[1], nb[2])

# ----- versus: two cards slide in, VS slams, values reveal, difference circled -----
def scene_versus(L, sc, t, ctx):
    d = ImageDraw.Draw(L)
    top = 236
    if sc.get('title'):
        top = title(L, sc['title'], t, size=sc.get('title_size', 80)) + 30
    top = max(top, sc.get('cards_y', 0))
    cw, gap = (410, 40) if sc.get('boxes', True) else (390, 110)
    lx0 = (W - 2 * cw - gap) / 2
    rv = at(sc.get('reveal_at'), ctx, 10 ** 6) if 'reveal_at' in sc else -1
    hl_t = at(sc.get('highlight_at'), ctx, 10 ** 6) if sc.get('highlight') else 10 ** 6
    bl_t = at(sc.get('blocks_at'), ctx, 10 ** 6) if sc.get('blocks') else 10 ** 6
    for side, info, x in (('left', sc['left'], lx0), ('right', sc['right'], lx0 + cw + gap)):
        if ctx.get('continues'): p = 1.0
        else: p = ease_out(prog(t, 0.05 if side == 'left' else 0.18, 0.4))
        off = int((1 - p) * (-600 if side == 'left' else 600)); a = int(255 * clamp(p * 1.5))
        hot = sc.get('highlight') == side and t >= hl_t
        y = top
        if sc.get('boxes', True):
            draw_box(L, (x + off, y + 30, x + cw - 30 + off, y + 380), info.get('box', info['name']), a)
            y += 420
        if sc.get('cards') is False: continue
        card = (x + off, y, x + cw + off, y + 300 + (170 if sc.get('blocks') else 0))
        d.rounded_rectangle(card, radius=30, fill=PANEL + (a,), outline=(ORANGE if hot else (60, 60, 66)) + (a,), width=6 if hot else 3)
        d.text(((card[0] + card[2]) / 2, y + 50), info['name'], font=BOLD(36 if len(info['name']) < 16 else 31), fill=WHITE + (a,), anchor='mm')
        srv = at(info['reveal_at'], ctx) if 'reveal_at' in info else rv
        q = ease_out(prog(t, srv, 0.35)) if srv >= 0 else 1.0
        vf = H1(118 if sc.get('boxes', True) else 108)
        if q <= 0:
            d.text(((card[0] + card[2]) / 2, y + 160), '?', font=vf, fill=GRAY + (a,), anchor='mm')
        else:
            col = ORANGE if (hot or info.get('orange')) else WHITE
            d.text(((card[0] + card[2]) / 2, y + 160 + int((1 - q) * 30)), info['value'], font=vf, fill=col + (int(a * q),), anchor='mm')
            if hot:
                vb = d.textbbox(((card[0] + card[2]) / 2, y + 160), info['value'], font=vf, anchor='mm')
                mark(d, vb, ease_out(prog(t, hl_t, 0.5)))
            if info.get('sub'):
                d.text(((card[0] + card[2]) / 2, y + 252), info['sub'], font=BODY(34), fill=GRAY + (int(a * q),), anchor='mm')
        if sc.get('blocks'):  # 1 block = 1 GB, filled in order (illustrative, to scale)
            n = int(re.match(r'\d+', info['value']).group())
            per, bs, bg_ = 8, 34, 10
            bx0 = (card[0] + card[2]) / 2 - (per * bs + (per - 1) * bg_) / 2
            for k in range(16):
                r_, c_ = divmod(k, per)
                bx, by = bx0 + c_ * (bs + bg_), y + 300 + r_ * (bs + bg_)
                if k < n:
                    on = prog(t, bl_t + k * 0.06, 0.12)
                    d.rounded_rectangle((bx, by, bx + bs, by + bs), radius=7, fill=(ORANGE if on >= 1 else (70, 70, 78)) + (a,))
                else:
                    d.rounded_rectangle((bx, by, bx + bs, by + bs), radius=7, outline=(70, 70, 78, a), width=3)
    vs_t = at(sc.get('vs_at'), ctx, 0.35)
    q = back_out(prog(t, vs_t, 0.3))
    if q > 0 and not sc.get('no_vs'):
        cy = top + (200 if sc.get('boxes', True) else 150); r = (62 if sc.get('boxes', True) else 50) * q
        d.ellipse((W / 2 - r, cy - r, W / 2 + r, cy + r), fill=ORANGE + (255,), outline=BG + (255,), width=8)
        if q > 0.5: d.text((W / 2, cy), 'VS', font=H1(int(56 * min(q, 1.1))), fill=BG + (255,), anchor='mm')
    if sc.get('note'):
        n0 = at(sc.get('note_at'), ctx, 0.8)
        p = ease_out(prog(t, n0, 0.4))
        if p > 0:
            draw_rich(L, X0 + int((1 - p) * -60), sc.get('note_y', 1150), sc['note'], BOLD(44), sc.get('note_w', 560), 56, alpha=int(255 * p))
    nb = sc.get('nibble_at', [850, 1320, 330])
    nibble(L, sc, t, ctx, nb[0], nb[1], nb[2])

def scene_bullets(L, sc, t, ctx):
    raise SystemExit('"bullets" is no longer a scene type (new style #30): use versus, game, stat or hook')

SCENES = {'hook': scene_hook, 'stat': scene_stat, 'cta': scene_cta, 'game': scene_game, 'versus': scene_versus,
          'bullets': scene_bullets}
DATA_SCENES = ('stat', 'versus')

# ---------- frame composition ----------
def header(layer, sb):
    d = ImageDraw.Draw(layer)
    d.text((70, 110), 'PANDABYTE', font=BOLD(34), fill=ORANGE + (255,))
    if sb.get('series'):
        d.text((W - 70, 118), sb['series'].upper(), font=BOLD(24), fill=GRAY + (255,), anchor='rm')

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
        if tw + 2 * 8 <= SAFE_W - 40 or size <= 40: break
        size -= 2
    x = SAFE_L + (SAFE_W - tw) / 2; y = 1400
    pop_ = back_out(prog(t, cur[0], 0.12))
    y += int((1 - pop_) * 14)
    for w in words:
        col = ORANGE if has_num(w) else WHITE
        d.text((x, y), w, font=fnt, fill=col + (255,), stroke_width=8, stroke_fill=(10, 10, 12, 255))
        x += d.textlength(w, font=fnt) + sp

def footer(layer, text, small=None):
    d = ImageDraw.Draw(layer)
    y = 1322
    if text:
        lines = wrap(d, [(w, GRAY) for w in text.split()], BODY(27), SAFE_W - 40)
        y -= 17 * (len(lines) - 1)
        for ln in lines:
            d.text((SAFE_L + SAFE_W // 2, y), ' '.join(w for w, _ in ln), font=BODY(27), fill=GRAY + (255,), anchor='mm'); y += 34
    if small:
        d.text((SAFE_L + SAFE_W // 2, y + 2), small, font=BODY(23), fill=GRAY + (255,), anchor='mm')

def render_frame(sc, t, ctx, sb):
    frame = background(t + ctx['offset']).convert('RGBA')
    content = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    SCENES[sc['type']](content, sc, t, ctx)
    z = 1 + sc.get('push', 0.04) * ease_io(t / max(ctx['dur'], 0.01))  # push-in on every scene
    if z > 1.0005:
        content = content.transform((W, H), Image.AFFINE, (1 / z, 0, CX - CX / z, 0, 1 / z, CY - CY / z), resample=Image.BILINEAR)
    frame.alpha_composite(content)
    top = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    header(top, sb)
    src = sc.get('source') or (sb.get('source') if sc['type'] in DATA_SCENES else None)
    footer(top, src if sc['type'] != 'cta' else sc.get('disclosure'), sc.get('footnote'))
    caption(top, ctx['cues'], t)
    frame.alpha_composite(top)
    return frame.convert('RGB')

# ---------- cover (also the first frame of the video) ----------
def render_cover(sb, first_sc, first_ctx):
    cv = sb['cover']
    words = re.sub(r'\*', '', cv['text']).split()
    if not 3 <= len(words) <= 5: raise SystemExit(f'cover text has {len(words)} words; the owner cover rule asks for 3 to 5')
    frame = background(0).convert('RGBA')
    back = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ctx = dict(first_ctx, track=[], continues=True)
    sc = dict(first_sc, title=None, flags=cv.get('flags', []), countdown=None, stamp=None)
    if sc['type'] == 'versus':
        sc.update(cards=cv.get('cards', True), no_vs=cv.get('no_vs', False))
    SCENES[sc['type']](back, sc, 30.0, ctx)
    dy = cv.get('backdrop_dy', 0)
    back = back.transform((W, H), Image.AFFINE, (1 / 1.06, 0, CX - CX / 1.06, 0, 1 / 1.06, 700 - 700 / 1.06 - dy / 1.06), resample=Image.BILINEAR)
    back = back.filter(ImageFilter.GaussianBlur(1.5))
    op = cv.get('backdrop_opacity', 0.75)
    back.putalpha(back.getchannel('A').point(lambda v: int(v * op)))
    frame.alpha_composite(back)
    d = ImageDraw.Draw(frame)
    fnt = H1(cv.get('size', 168))
    y = cv.get('y', 250)
    y_end = draw_rich(frame, SAFE_L, y, cv['text'], fnt, SAFE_W, int(cv.get('size', 168) * 1.04), center=True, stroke=14)
    grid_top, grid_bottom = (H - 1350) // 2, (H + 1350) // 2   # Instagram grid crops 9:16 to the central 4:5
    if y < grid_top or y_end > grid_bottom or y_end > H - 250:
        raise SystemExit(f'cover text {y}-{y_end}px is outside the central 1080x1350 area ({grid_top}-{grid_bottom})')
    p = pose(cv['pose']); h = cv.get('pose_height', 900)
    p = p.resize((int(p.width * h / p.height), h), Image.LANCZOS)
    frame.alpha_composite(p, (int(cv.get('pose_x', W // 2) - p.width / 2), int(cv.get('pose_bottom', 1500) - h)))
    d.text((70, 110), 'PANDABYTE', font=BOLD(34), fill=ORANGE + (255,))
    return frame.convert('RGB')

# ---------- storyboard ----------
def check_storyboard(sb, tracks):
    errs = []
    uses = {}
    prev = None
    for i, tr in enumerate(tracks):
        for j, (p, _) in enumerate(tr):
            if not (ASSETS / f'{p}.png').exists(): errs.append(f'scene {i + 1}: pose "{p}" not in assets/nibble')
            if j == 0 and p == prev: continue  # same pose carried across a cut is one use
            uses[p] = uses.get(p, 0) + 1
            prev = p
        if tr: prev = tr[-1][0]
    if sb.get('cover'):  # owner rule: the cover pose is exclusive to the cover
        cp = sb['cover']['pose']
        if cp in uses: errs.append(f'cover pose "{cp}" is also used in the scenes; the cover needs its own pose')
        uses[cp] = uses.get(cp, 0) + 1
    over = {p: n for p, n in uses.items() if n > MAX_POSE_USES}
    if over: errs.append(f'pose used more than {MAX_POSE_USES}x: {over}')
    for i, sc in enumerate(sb['scenes']):
        if sc['type'] not in SCENES or sc['type'] == 'bullets':
            errs.append(f'scene {i + 1}: type "{sc["type"]}" not supported')
    return errs, uses

def scene_audio(sb, sc, i, vo_dir):
    if sc.get('audio'): return Path(sc['audio'])
    if vo_dir and sb.get('vo', {}).get('prefix'):
        p = Path(vo_dir) / f"{sb['vo']['prefix']}{i + 1}.mp3"
        if p.exists(): return p
        raise SystemExit(f'missing voice file {p}')
    return None

def build(sb, use_voice, vo_dir=None):
    voice, speed = sb.get('voice', 'am_puck'), sb.get('speed', 1.0)
    plan, offset, prev_type = [], 0.0, None
    for i, sc in enumerate(sb['scenes']):
        ext = scene_audio(sb, sc, i, vo_dir)
        if ext: audio, cues, starts, ends = narrate_external(sc['say'], ext)
        else: audio, cues, starts, ends = narrate(sc['say'], voice, speed, use_voice)
        nframes = max(1, round(len(audio) / SR * FPS)); dur = nframes / FPS
        audio = np.pad(audio, (0, max(0, int(dur * SR) - len(audio))))[:int(dur * SR)]
        ctx = {'dur': dur, 'cues': cues, 'starts': starts, 'ends': ends, 'offset': offset,
               'continues': bool(sc.get('continue')) and prev_type == sc['type']}
        ctx['track'] = pose_track(sc, ctx)
        plan.append((sc, ctx, audio, nframes)); offset += dur; prev_type = sc['type']
    return plan

def sfx_events(plan):
    ev = []
    for i, (sc, ctx, _, _) in enumerate(plan):
        o = ctx['offset']
        if i: ev.append((o, 'whoosh'))
        for j, (_, s) in enumerate(ctx['track']):
            if j or (i and ctx['track']): ev.append((o + max(0, s) + (0.02 if j else 0.12), 'pop'))
        if sc.get('countdown'):
            c0 = at(sc['countdown'].get('at'), ctx, 1.2)
            for k in range(int(round(float(sc['countdown'].get('seconds', 2.0)) * 2))): ev.append((o + c0 + k * 0.5, 'tick'))
        for f in sc.get('flags', []):
            f0 = at(f.get('at'), ctx, 0.3)
            if f0 > 0.01 or not ctx['continues']: ev.append((o + f0 + 0.25, 'ding' if f.get('kind') == 'good' else 'buzz'))
        if sc.get('stamp'): ev.append((o + at(sc['stamp'].get('at'), ctx, 0.5), 'slam'))
        if sc['type'] == 'versus' and not ctx['continues'] and not sc.get('no_vs'): ev.append((o + at(sc.get('vs_at'), ctx, 0.35), 'slam'))
        if sc['type'] == 'versus':
            for side in ('left', 'right'):
                v = sc[side].get('reveal_at', sc.get('reveal_at'))
                if v is not None and at(v, ctx) < ctx['dur']: ev.append((o + at(v, ctx), 'pop'))
    return sorted(ev)

def contact_sheet(tiles, title_text, path):
    cols = 3; tw, th = 360, 640; pad = 24; head = 90
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new('RGB', (cols * tw + (cols + 1) * pad, head + rows * (th + 56) + pad), (12, 12, 14))
    d = ImageDraw.Draw(sheet)
    d.text((pad, 30), title_text, font=BOLD(30), fill=ORANGE)
    for k, (img, label) in enumerate(tiles):
        r, c = divmod(k, cols)
        x = pad + c * (tw + pad); y = head + r * (th + 56)
        sheet.paste(img.resize((tw, th), Image.LANCZOS), (x, y))
        sx = lambda v: x + v * tw / W; sy = lambda v: y + v * th / H
        d.rectangle((sx(SAFE_L), sy(SAFE_T), sx(SAFE_R), sy(SAFE_B)), outline=(200, 40, 40), width=1)
        d.text((x, y + th + 8), label, font=BODY(22), fill=(220, 220, 225))
    sheet.save(path)

# ---------- render ----------
def render(sb_path, stills=False, use_voice=True, vo_dir=None):
    sb = json.loads(Path(sb_path).read_text())
    slug = sb['slug']
    out = ROOT / 'build' / slug; out.mkdir(parents=True, exist_ok=True)
    if use_voice:
        try: kokoro()
        except Exception as e:  # stills can still be checked without the voice
            if not stills: raise
            print('Kokoro unavailable, timing from estimate:', e); use_voice = False
    t_start = time.time()
    plan = build(sb, use_voice, vo_dir)
    errs, uses = check_storyboard(sb, [ctx['track'] for _, ctx, _, _ in plan])
    total = sum(n for *_, n in plan) / FPS
    print(f'{slug}: {total:.1f}s, {len(plan)} scenes, poses {uses}')
    if errs:
        for e in errs: print('ERROR:', e)
        raise SystemExit(1)
    cover = render_cover(sb, plan[0][0], plan[0][1]) if sb.get('cover') else None
    timing = {'slug': slug, 'duration_s': round(total, 3), 'voice_files': [str(scene_audio(sb, sc, i, vo_dir) or '') for i, sc in enumerate(sb['scenes'])],
              'cues': [[round(ctx['offset'] + a, 3), round(ctx['offset'] + b, 3), txt] for sc, ctx, _, _ in plan for a, b, txt in ctx['cues']],
              'scenes': [{'type': sc['type'], 'start': round(ctx['offset'], 3), 'dur': round(ctx['dur'], 3), 'poses': ctx['track']} for sc, ctx, _, _ in plan]}
    (out / 'timing.json').write_text(json.dumps(timing, indent=1))
    if cover:
        cover.save(out / 'cover.jpg', quality=92); cover.resize((270, 480), Image.LANCZOS).save(out / 'cover-270x480.jpg', quality=92)
    if stills:
        tiles = [(cover, 'cover = frame 0')] if cover else []
        for i, (sc, ctx, _, _) in enumerate(plan):
            for k, v in enumerate(sc.get('stills', [ctx['dur'] * 0.8])):
                tt = min(at(v, ctx), ctx['dur'] - 1 / FPS)
                img = render_frame(sc, tt, ctx, sb)
                img.save(out / f'still-{i + 1:02d}-{k}.jpg', quality=90)
                tiles.append((img, f'{ctx["offset"] + tt:4.1f}s  s{i + 1} {sc["type"]}'))
        t0 = time.time(); render_frame(plan[0][0], 1.0, plan[0][1], sb); ms = (time.time() - t0)
        vo = sb.get('vo') or {}
        voice_label = (f"{vo.get('provider')} voice timing" if vo_dir and vo.get('prefix') else ("Kokoro am_puck timing" if use_voice else "estimated timing (no voice)"))
        contact_sheet(tiles, f'{slug}  |  {total:.1f}s  |  {voice_label}', out / 'contact.png')
        print(f'stills + contact sheet in {out} (~{ms * 1000:.0f} ms/frame -> full render ~{ms * total * FPS / 60:.1f} min of frames)')
        return out
    scene_files = []
    for i, (sc, ctx, audio, nframes) in enumerate(plan):
        tmp = out / f'scene-{i + 1:02d}.tmp.mp4'; final = out / f'scene-{i + 1:02d}.mp4'
        proc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                                 '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast',
                                 '-crf', '20', '-pix_fmt', 'yuv420p', '-r', str(FPS), str(tmp)], stdin=subprocess.PIPE)
        for f in range(nframes):
            img = cover if (i == 0 and cover is not None and f < COVER_FRAMES) else render_frame(sc, f / FPS, ctx, sb)
            proc.stdin.write(img.tobytes())
        proc.stdin.close()
        if proc.wait() != 0: raise SystemExit(f'ffmpeg failed on scene {i + 1}')
        tmp.rename(final); scene_files.append(final)
        print(f'scene {i + 1}/{len(plan)} done ({ctx["dur"]:.1f}s)')
    import soundfile as sf
    narration = np.concatenate([a for _, _, a, _ in plan])
    if sb.get('sfx', True): narration = sfx.mix(narration, sfx_events(plan), sb.get('sfx_volume', 1.0))
    sf.write(out / 'narration.wav', narration, SR)
    (out / 'concat.txt').write_text(''.join(f"file '{p.name}'\n" for p in scene_files))
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', str(out / 'concat.txt'),
                    '-c', 'copy', str(out / 'video-only.mp4')], check=True)
    final = out / f'{slug}.mp4'; tmp = out / f'{slug}.tmp.mp4'
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(out / 'video-only.mp4'), '-i', str(out / 'narration.wav'),
                    '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11', '-ar', '48000',
                    '-c:a', 'aac', '-b:a', '160k', '-shortest',
                    '-movflags', '+faststart', str(tmp)], check=True)
    tmp.rename(final)
    print('done:', final, f'{total:.1f}s in {time.time() - t_start:.0f}s')

if __name__ == '__main__':
    args = sys.argv[1:]
    vo = args[args.index('--vo') + 1] if '--vo' in args else None
    render(args[0], stills='--stills' in args, use_voice='--no-voice' not in args, vo_dir=vo)
