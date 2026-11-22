"""Builds the VideoPackage for a rendered video: package.json, SRT, transcript.

    python3 studio/package.py storyboards/<slug>.json      # after studio/render.py

Reads build/<slug>/timing.json (written by render.py) and the storyboard's "package" block
(platforms, captions per platform, Shorts title, links, assets). Checks the caption rules:
TikTok without "link in bio", Reels with at most 5 hashtags, Shorts title up to 70 characters.
Schema: schemas/video-package.schema.json.
"""
import datetime, hashlib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def srt_time(t):
    ms = int(round(t * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f'{h:02d}:{m:02d}:{s:02d},{ms:03d}'


def say_text(x): return x if isinstance(x, str) else x.get('text', '')


def check_captions(pk):
    errs = []
    cap = pk.get('caption', {})
    if 'tiktok' in cap and re.search(r'link\s+in\s+(the\s+)?bio', cap['tiktok'], re.I): errs.append('TikTok caption says "link in bio"')
    if 'instagram_reels' in cap and len(re.findall(r'#\w+', cap['instagram_reels'])) > 5: errs.append('Reels caption has more than 5 hashtags')
    if 'youtube_shorts' in pk.get('platforms', []):
        t = pk.get('youtube_shorts_title', '')
        if not t or len(t) > 70: errs.append(f'Shorts title missing or over 70 characters ({len(t)})')
    for plat, text in cap.items():
        if not re.search(r'commission|affiliate', text, re.I): errs.append(f'{plat} caption has no affiliate disclosure')
    return errs


def main(sb_path):
    sb = json.loads(Path(sb_path).read_text()); slug = sb['slug']; pk = sb['package']
    out = ROOT / 'build' / slug
    timing = json.loads((out / 'timing.json').read_text())
    video = out / f'{slug}.mp4'
    errs = check_captions(pk)
    if errs:
        for e in errs: print('ERROR:', e)
        raise SystemExit(1)
    srt = ''.join(f'{i}\n{srt_time(a)} --> {srt_time(b)}\n{txt}\n\n' for i, (a, b, txt) in enumerate(timing['cues'], 1))
    (out / f'{slug}.srt').write_text(srt)
    lines = [f'# {slug} - transcript', '']
    for sc, st in zip(sb['scenes'], timing['scenes']):
        spoken = ' '.join(say_text(x) for x in sc['say'] if not (isinstance(x, dict) and 'pause' in x))
        lines.append(f'[{st["start"]:05.2f}] {spoken}')
    (out / f'{slug}.transcript.txt').write_text('\n'.join(lines) + '\n')
    cta = next((s for s in sb['scenes'] if s['type'] == 'cta'), {})
    pkg = {
        'schema': 'video-package/v1', 'slug': slug, 'brief_id': pk.get('brief_id'), 'revision': 1,
        'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'video': {'path': video.name, 'sha256': hashlib.sha256(video.read_bytes()).hexdigest(), 'duration_s': timing['duration_s']},
        'storyboard': f'storyboards/{slug}.json',
        'platforms': pk['platforms'], 'caption': pk['caption'], 'links': pk.get('links', []),
        'disclosure': {'on_screen': cta.get('disclosure'), 'spoken': True, 'caption': pk.get('caption_disclosure')},
        'assets': pk['assets'], 'tools': pk.get('tools', []),
        'ai_generated': {'voice': True, 'visuals': True},
        'files': {'cover': 'cover.jpg', 'cover_small': 'cover-270x480.jpg', 'srt': f'{slug}.srt', 'transcript': f'{slug}.transcript.txt'},
        'notes': pk.get('notes', ''),
    }
    if pk.get('youtube_shorts_title'): pkg['youtube_shorts_title'] = pk['youtube_shorts_title']
    (out / 'package.json').write_text(json.dumps(pkg, indent=2, ensure_ascii=False) + '\n')
    print('package:', out / 'package.json')


if __name__ == '__main__':
    main(sys.argv[1])
