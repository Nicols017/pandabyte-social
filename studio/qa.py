"""PandaByte video QA: objective checks for a rendered video (and its storyboard/package).

A clean render is not an approval. This script only measures; the Video Quality Agent (14)
reads the JSON report, looks at the contact sheet, settles the `review` items and issues
the final VideoQAReport verdict (pass | fail | needs_owner).

Usage:
    python3 studio/qa.py <video.mp4> [--storyboard sb.json] [--package package.json]
                         [--catalog ../pandabyte/data/catalog] [--out report.json] [--no-frames]

Exit codes: 0 auto pass · 1 at least one fail · 2 no fail but items need review · 3 usage error.
Requirements: Python 3.9+, ffmpeg + ffprobe on PATH. Standard library only.
"""
import argparse, datetime, hashlib, json, os, re, struct, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / 'assets' / 'nibble'

# ---------- thresholds (change here, keep the skill in sync) ----------
WIDTH, HEIGHT, FPS = 1080, 1920, 30
DURATION_FAIL = (5.0, 180.0)        # outside -> fail (Shorts and Reels top out at 3 min)
DURATION_TARGET = (15.0, 60.0)      # outside -> warn
AV_DRIFT_MAX = 0.5                  # s between audio and video stream length
LUFS_PASS, LUFS_WARN = (-16.0, -12.0), (-18.0, -10.0)
TRUE_PEAK_PASS, TRUE_PEAK_WARN = -1.0, 0.0
SILENCE_DB, SILENCE_WARN, SILENCE_FAIL = -50, 1.0, 2.0   # mid-video gaps, seconds
LEAD_SILENCE_WARN = 0.5
BLACK_MIN = 0.5                     # s of black frames -> fail
FREEZE_WARN = 4.0                   # s of identical frames -> warn
MAX_SIZE_MB = 250
# safe zone shared by TikTok / Reels / Shorts on 1080x1920 (UI overlays)
SAFE_TOP, SAFE_BOTTOM, SAFE_LEFT, SAFE_RIGHT = 220, 420, 60, 120

LICENSE_OK = {'apache-2.0', 'mit', 'cc0-1.0', 'cc0', 'ofl-1.1', 'pandabyte-owned', 'public-domain'}
LICENSE_ATTRIB = {'cc-by-4.0', 'cc-by-3.0', 'cc-by-sa-4.0'}
LICENSE_BAD = re.compile(r'(^|\b)(unknown|none|all rights reserved|cc-by-nc|nc-|editorial)', re.I)
AFFILIATE_URL = re.compile(r'amzn\.to|amazon\.[a-z.]+/.*[?&]tag=|ebay\.us/|[?&]campid=|[?&]mkcid=|bestbuy\.7tiv|newegg\..*[?&]aff', re.I)
AMAZON_URL = re.compile(r'amzn\.to|amazon\.[a-z.]+', re.I)
DISCLOSURE_WORDS = re.compile(r'commission|affiliate|#ad\b|paid link|qualifying purchases', re.I)
AMAZON_STATEMENT = re.compile(r'as an amazon associate', re.I)

checks = []

def add(cid, category, status, detail, value=None, expected=None):
    c = {'id': cid, 'category': category, 'status': status, 'detail': detail}
    if value is not None: c['value'] = value
    if expected is not None: c['expected'] = expected
    checks.append(c)

def band(v, ok, warn):
    if ok[0] <= v <= ok[1]: return 'pass'
    return 'warn' if warn[0] <= v <= warn[1] else 'fail'

def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def frac(s):
    try:
        a, b = s.split('/'); return float(a) / float(b) if float(b) else 0.0
    except Exception: return 0.0

# ---------- technical ----------
def top_atoms(p):
    order = []
    with open(p, 'rb') as f:
        while len(order) < 64:
            hdr = f.read(8)
            if len(hdr) < 8: break
            size, kind = struct.unpack('>I4s', hdr)
            if size == 1: size = struct.unpack('>Q', f.read(8))[0]; f.seek(size - 16, 1)
            elif size == 0: order.append(kind.decode('latin1')); break
            else: f.seek(size - 8, 1)
            order.append(kind.decode('latin1'))
    return order

def technical(video):
    pr = run(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', video])
    if pr.returncode: add('probe', 'technical', 'fail', 'ffprobe could not read the file', pr.stderr.strip()[:300]); return None
    info = json.loads(pr.stdout)
    fmt, streams = info['format'], info['streams']
    vs = [s for s in streams if s['codec_type'] == 'video']
    aus = [s for s in streams if s['codec_type'] == 'audio']
    dur = float(fmt.get('duration', 0))

    add('container', 'technical', 'pass' if 'mp4' in fmt.get('format_name', '') else 'fail', 'MP4 container', fmt.get('format_name'), 'mp4')
    atoms = top_atoms(video)
    fast = 'moov' in atoms and 'mdat' in atoms and atoms.index('moov') < atoms.index('mdat')
    add('faststart', 'technical', 'pass' if fast else 'warn', 'moov atom before mdat (+faststart)', atoms[:6])
    size_mb = round(int(fmt.get('size', 0)) / 1e6, 1)
    add('file_size', 'technical', 'pass' if size_mb <= MAX_SIZE_MB else 'warn', 'File size (MB)', size_mb, f'<= {MAX_SIZE_MB}')

    if len(vs) != 1: add('video_stream', 'technical', 'fail', 'Exactly one video stream', len(vs), 1)
    if vs:
        v = vs[0]
        add('video_codec', 'technical', 'pass' if v['codec_name'] == 'h264' else 'fail', 'H.264 video', v['codec_name'], 'h264')
        add('pixel_format', 'technical', 'pass' if v.get('pix_fmt') == 'yuv420p' else 'fail', '8-bit 4:2:0', v.get('pix_fmt'), 'yuv420p')
        res = f"{v['width']}x{v['height']}"
        add('resolution', 'technical', 'pass' if (v['width'], v['height']) == (WIDTH, HEIGHT) else 'fail', 'Vertical 9:16', res, f'{WIDTH}x{HEIGHT}')
        sar = v.get('sample_aspect_ratio', '1:1')
        add('sar', 'technical', 'pass' if sar in ('1:1', '0:1', 'N/A') else 'fail', 'Square pixels', sar, '1:1')
        r, avg = frac(v.get('r_frame_rate', '0/1')), frac(v.get('avg_frame_rate', '0/1'))
        ok = abs(r - FPS) < 0.01 and abs(avg - FPS) < 0.05
        add('frame_rate', 'technical', 'pass' if ok else 'fail', 'Constant frame rate', {'r': round(r, 3), 'avg': round(avg, 3)}, FPS)
    if len(aus) != 1: add('audio_stream', 'technical', 'fail', 'Exactly one audio stream', len(aus), 1)
    if aus:
        a = aus[0]
        add('audio_codec', 'technical', 'pass' if a['codec_name'] == 'aac' else 'fail', 'AAC audio', a['codec_name'], 'aac')
        sr = int(a.get('sample_rate', 0))
        add('sample_rate', 'technical', 'pass' if sr >= 44100 else 'warn', 'Sample rate (Hz)', sr, '>= 44100')
        if vs:
            drift = abs(float(vs[0].get('duration', dur)) - float(a.get('duration', dur)))
            add('av_sync_length', 'technical', 'pass' if drift <= AV_DRIFT_MAX else 'fail', 'Audio/video stream length difference (s)', round(drift, 3), f'<= {AV_DRIFT_MAX}')

    st = 'fail' if not DURATION_FAIL[0] <= dur <= DURATION_FAIL[1] else ('pass' if DURATION_TARGET[0] <= dur <= DURATION_TARGET[1] else 'warn')
    add('duration', 'technical', st, 'Duration (s)', round(dur, 2), f'{DURATION_TARGET[0]}-{DURATION_TARGET[1]} target, {DURATION_FAIL[0]}-{DURATION_FAIL[1]} hard')

    dec = run(['ffmpeg', '-v', 'error', '-i', video, '-f', 'null', '-'])
    errs = [l for l in dec.stderr.splitlines() if l.strip()]
    add('decode_errors', 'technical', 'pass' if not errs else 'fail', 'Full decode without errors', len(errs), 0)

    if aus:
        log = run(['ffmpeg', '-hide_banner', '-nostats', '-i', video, '-map', '0:a:0',
                   '-af', f'ebur128=peak=true,silencedetect=noise={SILENCE_DB}dB:d={SILENCE_WARN}', '-f', 'null', '-']).stderr
        i = re.findall(r'I:\s+(-?[\d.]+|-inf) LUFS', log)
        tp = re.findall(r'Peak:\s+(-?[\d.]+|-inf) dBFS', log)
        if i and i[-1] != '-inf':
            v = float(i[-1]); add('loudness', 'technical', band(v, LUFS_PASS, LUFS_WARN), 'Integrated loudness (LUFS)', v, f'{LUFS_PASS[0]} to {LUFS_PASS[1]}')
        else: add('loudness', 'technical', 'fail', 'Integrated loudness could not be measured (silent audio?)')
        if tp:
            v = float(tp[-1]) if tp[-1] != '-inf' else -99.0
            add('true_peak', 'technical', 'pass' if v <= TRUE_PEAK_PASS else ('warn' if v <= TRUE_PEAK_WARN else 'fail'), 'True peak (dBTP)', v, f'<= {TRUE_PEAK_PASS}')
        starts = [float(x) for x in re.findall(r'silence_start: (-?[\d.]+)', log)]
        ends = [(float(e), float(d)) for e, d in re.findall(r'silence_end: ([\d.]+) \| silence_duration: ([\d.]+)', log)]
        gaps = []
        for k, s0 in enumerate(starts):
            e, d = ends[k] if k < len(ends) else (dur, dur - s0)
            gaps.append({'start': round(max(s0, 0), 2), 'end': round(e, 2), 'duration': round(d, 2)})
        lead = next((g['duration'] for g in gaps if g['start'] <= 0.05), 0.0)
        add('lead_silence', 'technical', 'pass' if lead <= LEAD_SILENCE_WARN else 'warn', 'Silence before the hook (s)', lead, f'<= {LEAD_SILENCE_WARN}')
        mid = [g for g in gaps if g['start'] > 0.05 and g['end'] < dur - 0.05]
        worst = max((g['duration'] for g in mid), default=0.0)
        st = 'pass' if worst < SILENCE_WARN else ('warn' if worst < SILENCE_FAIL else 'fail')
        add('mid_silence', 'technical', st, f'Longest gap below {SILENCE_DB} dB inside the video (s)', worst, f'< {SILENCE_WARN}')

    if vs:
        log = run(['ffmpeg', '-hide_banner', '-nostats', '-i', video, '-map', '0:v:0',
                   '-vf', f'blackdetect=d={BLACK_MIN}:pix_th=0.05,freezedetect=n=0.001:d={FREEZE_WARN}', '-f', 'null', '-']).stderr
        blacks = re.findall(r'black_start:([\d.]+) black_end:([\d.]+) black_duration:([\d.]+)', log)
        add('black_frames', 'technical', 'pass' if not blacks else 'fail', f'Black segments >= {BLACK_MIN}s',
            [{'start': float(a), 'duration': float(c)} for a, b, c in blacks] or 0)
        frz = [float(x) for x in re.findall(r'freeze_duration: ([\d.]+)', log)]
        add('frozen_frames', 'technical', 'pass' if not frz else 'warn', f'Frozen segments >= {FREEZE_WARN}s', frz or 0)
    return dur

def contact_sheet(video, dur, out_png):
    n = 6
    rate = max(n / max(dur, 1), 0.01)
    vf = (f'fps={rate:.5f},'
          f'drawbox=x={SAFE_LEFT}:y={SAFE_TOP}:w=iw-{SAFE_LEFT + SAFE_RIGHT}:h=ih-{SAFE_TOP + SAFE_BOTTOM}:color=red:t=8,'
          f'scale=360:-2,tile=3x2')
    r = run(['ffmpeg', '-v', 'error', '-y', '-i', video, '-vf', vf, '-frames:v', '1', '-update', '1', str(out_png)])
    ok = r.returncode == 0 and Path(out_png).exists()
    add('safe_zone_visual', 'visual', 'review' if ok else 'fail',
        'Look at the contact sheet: no caption, title, price or disclosure outside the red safe-area box; Nibble and text legible',
        str(out_png) if ok else r.stderr.strip()[:300])
    return str(out_png) if ok else None

# ---------- storyboard: specs, source, disclosure, brand ----------
def norm(s):
    s = re.sub(r'\*', '', s.lower())
    return re.findall(r'[a-z]+|\d+(?:\.\d+)?', s.replace(',', ''))

VARIANT_WORDS = {'ti', 'xt', 'xtx', 'super', 'gre', 'gb', 'oc', 'max', 'pro', 'plus'}

def load_catalog(path):
    recs = []
    for f in sorted(Path(path).glob('*.json')):
        if f.name.endswith('.schema.json'): continue
        try: d = json.loads(f.read_text())
        except Exception: continue
        recs += d.get('records', []) if isinstance(d, dict) else []
    return recs

def find_record(label, recs):
    want = set(norm(label))
    best = []
    for r in recs:
        toks = norm(r.get('name', '')); manu = set(norm(r.get('manufacturer', '')))
        if want <= set(toks):
            extra = [t for t in toks if t not in want and t not in manu]
            best.append((len(extra), extra, r))
    if not best: return None, 'no catalog record matches this label'
    best.sort(key=lambda x: x[0])
    n, extra, r = best[0]
    if len(best) > 1 and best[1][0] == n: return None, f'ambiguous: {[b[2]["name"] for b in best if b[0] == n]}'
    if VARIANT_WORDS & set(extra): return r, f'label is less specific than the record ({" ".join(extra)})'
    return r, None

def num_units(text):
    return [(float(n.replace(',', '')), u.lower()) for n, u in re.findall(r'(\d[\d,]*(?:\.\d+)?)\s*([A-Za-z%]+)?', text or '') if u is not None]

def check_value(label, value, unit, recs, cid):
    r, note = find_record(label, recs)
    claim = f'{label}: {value} {unit or ""}'.strip()
    if not r: add(cid, 'spec', 'review', f'Verify by hand: {claim} ({note})'); return
    specs = r.get('specs', [])
    unit = (unit or '').lower()
    same_unit = [(s, nu) for s in specs for nu in num_units(s.get('value')) if nu[1] == unit] if unit else []
    try: v = float(str(value).replace(',', ''))
    except ValueError: v = None
    if v is not None and any(abs(nu[0] - v) < 1e-6 for _, nu in same_unit):
        hit = next(s for s, nu in same_unit if abs(nu[0] - v) < 1e-6)
        st = 'review' if note else 'pass'
        add(cid, 'spec', st, f'{claim} matches catalog "{r["slug"]}" -> {hit["label"]}: {hit["value"]}' + (f' — {note}' if note else ''))
    elif same_unit:
        add(cid, 'spec', 'fail', f'{claim} contradicts catalog "{r["slug"]}"', value, [f'{s["label"]}: {s["value"]}' for s, _ in same_unit])
    else:
        add(cid, 'spec', 'review', f'Verify by hand: {claim} — catalog "{r["slug"]}" has no spec in that unit')

def storyboard(sb_path, catalog):
    sb = json.loads(Path(sb_path).read_text())
    scenes = sb.get('scenes', [])
    recs = load_catalog(catalog) if catalog and Path(catalog).is_dir() else []
    if not recs: add('catalog', 'spec', 'review', f'Catalog not found at {catalog}; spec claims must be checked by hand')
    data_scene = any(s.get('type') in ('stat', 'bars', 'compare', 'versus') for s in scenes)
    if data_scene:
        add('source_line', 'spec', 'pass' if (sb.get('source') or '').strip() else 'fail', 'On-screen source for data scenes', sb.get('source'))
    for i, s in enumerate(scenes, 1):
        t = s.get('type')
        if t == 'bars' and recs:
            for j, b in enumerate(s.get('bars', []), 1):
                check_value(b['label'], b['value'], b.get('unit'), recs, f'spec_s{i}_bar{j}')
        elif t == 'compare' and recs:
            cols = s.get('cols', [])
            for j, row in enumerate(s.get('rows', []), 1):
                for k, col in enumerate(cols[:2]):
                    cell = str(row[k + 1]) if len(row) > k + 1 else ''
                    nu = num_units(cell)
                    if nu: check_value(col, nu[0][0], nu[0][1], recs, f'spec_s{i}_r{j}_{k}')
                    else: add(f'spec_s{i}_r{j}_{k}', 'spec', 'review', f'Verify by hand: {col} / {row[0]} = {cell}')
        elif t == 'versus' and recs:
            for side in ('left', 'right'):
                info = s.get(side) or {}
                nu = num_units(str(info.get('value', '')))
                if nu: check_value(info.get('name', ''), nu[0][0], nu[0][1], recs, f'spec_s{i}_{side}')
        elif t == 'game':
            add(f'listing_s{i}', 'spec', 'review', 'Game scene: listing must be visibly fictional (no real store, retailer brand, logo or price)')
        elif t == 'stat':
            add(f'spec_s{i}_stat', 'spec', 'review', f'Verify by hand: "{s.get("value")}" {s.get("label", "")}')
        for p in ([s.get('pose')] if s.get('pose') else []) + [q[0] for q in s.get('poses', [])]:
            if not (ASSETS / f'{p}.png').exists(): add(f'pose_s{i}', 'brand', 'fail', f'Pose "{p}" not in assets/nibble')
    add('spoken_claims', 'spec', 'review', 'Read every "say" line: spoken numbers and model names must match the screen and the catalog')
    cta = [s for s in scenes if s.get('type') == 'cta']
    disc = ' '.join((s.get('disclosure') or '') for s in cta)
    if cta:
        add('disclosure_on_screen', 'disclosure', 'pass' if DISCLOSURE_WORDS.search(disc) else 'fail', 'Affiliate disclosure shown on screen in the CTA', disc or None)
        spoken = ' '.join(' '.join(x if isinstance(x, str) else x.get('text', '') for x in s.get('say', [])) for s in cta)
        add('disclosure_spoken', 'disclosure', 'pass' if DISCLOSURE_WORDS.search(spoken) else 'warn', 'Disclosure also spoken', spoken or None)
    else:
        add('disclosure_on_screen', 'disclosure', 'review', 'No CTA scene: confirm the video has no commercial call to action or link')
    voice = sb.get('voice'); vo = sb.get('vo') or {}
    if vo.get('provider'):
        add('voice', 'brand', 'review', f'External voice {vo.get("provider")} "{vo.get("voice_name")}" ({vo.get("voice_id")}, {vo.get("model")}): confirm owner approval and the license row in assets/LICENSES.md', vo.get('voice_id'))
    else:
        add('voice', 'brand', 'pass' if voice == 'am_puck' else 'warn', 'Nibble voice (Kokoro am_puck)', voice, 'am_puck')

# ---------- package: licenses, links, caption disclosure ----------
def package(pkg_path, video):
    pkg = json.loads(Path(pkg_path).read_text())
    want = (pkg.get('video') or {}).get('sha256')
    if want: add('package_hash', 'license', 'pass' if want == sha256(video) else 'fail', 'Video file is the one listed in the package')
    assets = pkg.get('assets', [])
    if not assets: add('assets_listed', 'license', 'fail', 'VideoPackage lists no assets')
    for a in assets:
        lic = (a.get('license') or '').strip(); key = lic.lower(); aid = a.get('id') or a.get('source') or '?'
        if not lic or LICENSE_BAD.search(lic): st, why = 'fail', 'missing, unknown or non-commercial license'
        elif key in LICENSE_OK: st, why = 'pass', 'license allows commercial use'
        elif key in LICENSE_ATTRIB: st, why = ('pass', 'attribution present') if a.get('attribution') else ('fail', 'attribution required but missing')
        else: st, why = 'review', 'license not on the allowlist; read its terms'
        if st == 'pass' and not (a.get('source') or a.get('source_url')) and key != 'pandabyte-owned': st, why = 'review', 'no source recorded'
        add(f'license_{aid}', 'license', st, f'{aid} ({a.get("type", "?")}): {why}', lic or None)
    for t in pkg.get('tools', []):
        if t.get('paid'): add(f'tool_{t.get("name")}', 'license', 'review', f'Paid tool used: {t.get("name")} — owner must have approved it')
    links = [l.get('url', '') for l in pkg.get('links', [])]
    captions = pkg.get('caption') or {}
    caps = captions if isinstance(captions, dict) else {'all': captions}
    aff = [u for u in links if AFFILIATE_URL.search(u)] + [l.get('url', '') for l in pkg.get('links', []) if l.get('affiliate') and not AFFILIATE_URL.search(l.get('url', ''))]
    if aff:
        for plat, text in caps.items():
            add(f'caption_disclosure_{plat}', 'disclosure', 'pass' if DISCLOSURE_WORDS.search(text or '') else 'fail', f'Caption ({plat}) discloses affiliate links')
            if any(AMAZON_URL.search(u) for u in aff):
                add(f'amazon_statement_{plat}', 'disclosure', 'pass' if AMAZON_STATEMENT.search(text or '') else 'fail',
                    f'Caption ({plat}) carries "As an Amazon Associate I earn from qualifying purchases."')
    if (pkg.get('ai_generated') or {}).get('voice') or (pkg.get('ai_generated') or {}).get('visuals'):
        add('ai_label', 'disclosure', 'review', 'Synthetic voice/visuals: decide per platform whether the AI-generated label is required')

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('video'); ap.add_argument('--storyboard'); ap.add_argument('--package')
    ap.add_argument('--catalog', default=os.environ.get('PANDABYTE_CATALOG', str(ROOT.parent / 'pandabyte' / 'data' / 'catalog')))
    ap.add_argument('--out'); ap.add_argument('--no-frames', action='store_true')
    a = ap.parse_args()
    video = Path(a.video)
    if not video.is_file(): print(f'not a file: {video}', file=sys.stderr); return 3
    out = Path(a.out) if a.out else video.with_suffix('.qa.json')
    dur = technical(str(video))
    sheet = contact_sheet(str(video), dur, out.with_suffix('.png')) if dur and not a.no_frames else None
    if a.storyboard: storyboard(a.storyboard, a.catalog)
    else: add('storyboard', 'spec', 'review', 'No storyboard given: spec claims and disclosure must be checked by hand')
    if a.package: package(a.package, str(video))
    else: add('package', 'license', 'review', 'No VideoPackage given: asset licenses must be checked by hand')
    statuses = {c['status'] for c in checks}
    auto = 'fail' if 'fail' in statuses else ('review' if 'review' in statuses else 'pass')
    ver = run(['ffmpeg', '-version']).stdout.split('\n')[0]
    report = {
        'schema': 'video-qa-report/v1', 'verdict': 'pending', 'auto_verdict': auto,
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'tool': {'name': 'studio/qa.py', 'ffmpeg': ver},
        'video': {'path': str(video), 'sha256': sha256(video), 'duration_s': round(dur or 0, 2)},
        'inputs': {'storyboard': a.storyboard, 'package': a.package, 'catalog': a.catalog},
        'summary': {s: sum(c['status'] == s for c in checks) for s in ('pass', 'warn', 'review', 'fail', 'skip')},
        'checks': checks, 'artifacts': {'contact_sheet': sheet},
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f'{auto.upper()}  {report["summary"]}  -> {out}')
    for c in checks:
        if c['status'] in ('fail', 'warn', 'review'): print(f'  [{c["status"]}] {c["id"]}: {c["detail"]}' + (f' = {c["value"]}' if 'value' in c else ''))
    return {'pass': 0, 'fail': 1, 'review': 2}[auto]

if __name__ == '__main__':
    sys.exit(main())
