#!/usr/bin/env python3
"""Render a PandaByte/Nibble Instagram carousel (1080x1350) from a JSON spec.

Usage: python3 studio/carousel.py carousels/<slug>.json  ->  posts/<slug>/<slug>-N.png

Spec: {"slug": "...", "source": "Source: ...", "slides": [
  {"type": "cover",   "title": "Text with *orange* words", "lines": ["..."], "pose": "suspicious"},
  {"type": "bullets", "title": "...", "items": ["..."], "pose": "thinking"},
  {"type": "tiers",   "title": "...", "tiers": [{"label": "STRONGEST", "name": "...", "note": "..."}], "pose": "pointing"},
  {"type": "cta",     "title": "...", "lines": ["..."], "button": "@pandabyte.tech", "pose": "thumbsup"}]}
"""
import json, sys, re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
W, H = 1080, 1350
BG = (18, 18, 20); ORANGE = (255, 138, 61); WHITE = (245, 245, 247); GRAY = (160, 160, 168); CARD = (30, 30, 34)
POPPINS = '/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf'
INTER = '/usr/share/fonts/opentype/inter/Inter-Medium.otf'
INTERB = '/usr/share/fonts/opentype/inter/Inter-Bold.otf'
def F(path, size):
    try: return ImageFont.truetype(path, size)
    except OSError: return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size)
MARGIN = 72; MAXW = W - 2 * MARGIN

def background():
    im = Image.new('RGB', (W, H), BG)
    glow = Image.new('L', (W, H), 0)
    ImageDraw.Draw(glow).ellipse((W - 700, H - 760, W + 500, H + 440), fill=255)
    glow = glow.filter(ImageFilter.GaussianBlur(220))
    im.paste(Image.new('RGB', (W, H), (70, 38, 16)), (0, 0), glow)
    d = ImageDraw.Draw(im)
    d.text((MARGIN, 66), 'PANDABYTE', font=F(INTERB, 30), fill=ORANGE)
    return im

def rich(d, x, y, text, font, maxw, lh, base=WHITE):
    """Draw text where *words* are orange; wraps; returns the y after the last line."""
    tokens = []
    for part in re.split(r'(\*[^*]+\*)', text):
        if not part: continue
        hl = part.startswith('*')
        tokens += [(w, hl) for w in part.strip('*').split()]
    sp = d.textlength(' ', font=font); line = []; lw = 0
    def flush(yy):
        xx = x
        for w, hl in line:
            d.text((xx, yy), w, font=font, fill=ORANGE if hl else base); xx += d.textlength(w, font=font) + sp
    for w, hl in tokens:
        ww = d.textlength(w, font=font)
        if line and lw + sp + ww > maxw:
            flush(y); y += lh; line = []; lw = 0
        lw = ww if not line else lw + sp + ww; line.append((w, hl))
    if line: flush(y); y += lh
    return y

def nibble(im, pose, size, right=40, bottom=40):
    p = Image.open(ROOT / 'assets' / 'nibble' / f'{pose}.png').convert('RGBA')
    p.thumbnail((size, size))
    im.paste(p, (W - p.width - right, H - p.height - bottom), p)

def source(d, text):
    if text: d.text((MARGIN, H - 72), text, font=F(INTER, 28), fill=GRAY)

def slide(spec, sc):
    im = background(); d = ImageDraw.Draw(im); t = sc['type']
    if t == 'cover':
        y = rich(d, MARGIN, 170, sc['title'], F(POPPINS, 104), MAXW, 118)
        for line in sc.get('lines', []):
            y = rich(d, MARGIN, y + 36, line, F(INTER, 46), MAXW, 58)
        nibble(im, sc.get('pose', 'suspicious'), 700)
    elif t == 'bullets':
        y = rich(d, MARGIN, 170, sc['title'], F(POPPINS, 80), MAXW, 94) + 40
        for i, it in enumerate(sc['items']):
            d.ellipse((MARGIN, y + 4, MARGIN + 52, y + 56), fill=ORANGE)
            d.text((MARGIN + 26, y + 30), str(i + 1), font=F(INTERB, 30), fill=BG, anchor='mm')
            y = rich(d, MARGIN + 80, y + 4, it, F(INTER, 42), MAXW - 300, 54) + 34
        nibble(im, sc.get('pose', 'thinking'), 400, bottom=110)
    elif t == 'tiers':
        y = rich(d, MARGIN, 170, sc['title'], F(POPPINS, 80), MAXW, 94) + 36
        for i, tr in enumerate(sc['tiers']):
            top = y; col = ORANGE if i == 0 else (WHITE if i == 1 else GRAY)
            d.rounded_rectangle((MARGIN - 12, top, W - MARGIN + 12, top + 170), radius=22, fill=CARD)
            d.text((MARGIN + 16, top + 22), tr['label'], font=F(INTERB, 26), fill=col)
            d.text((MARGIN + 16, top + 60), tr['name'], font=F(POPPINS, 46), fill=WHITE)
            d.text((MARGIN + 16, top + 122), tr['note'], font=F(INTER, 30), fill=GRAY)
            y = top + 196
        nibble(im, sc.get('pose', 'pointing'), 250, right=60, bottom=100)
    elif t == 'cta':
        y = rich(d, MARGIN, 170, sc['title'], F(POPPINS, 92), MAXW, 106)
        for line in sc.get('lines', []):
            y = rich(d, MARGIN, y + 26, line, F(INTER, 44), MAXW, 56)
        if sc.get('button'):
            bw = d.textlength(sc['button'], font=F(INTERB, 40)) + 80
            d.rounded_rectangle((MARGIN, y + 50, MARGIN + bw, y + 140), radius=45, fill=ORANGE)
            d.text((MARGIN + bw / 2, y + 95), sc['button'], font=F(INTERB, 40), fill=BG, anchor='mm')
        nibble(im, sc.get('pose', 'thumbsup'), 560, bottom=110)
    if sc.get('source', t != 'cover'): source(d, spec.get('source'))
    return im

def main(path):
    spec = json.loads(Path(path).read_text())
    out = ROOT / 'posts' / spec['slug']; out.mkdir(parents=True, exist_ok=True)
    for i, sc in enumerate(spec['slides'], 1):
        slide(spec, sc).save(out / f"{spec['slug']}-{i}.png", optimize=True)
    print('done:', out)

if __name__ == '__main__':
    main(sys.argv[1])
