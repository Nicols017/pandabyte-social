"""Short sound effects synthesized from code (no samples, no third-party files).

Every sound is generated here with NumPy, so PandaByte owns the output outright and no
license row beyond `assets/LICENSES.md` -> "Sound effects (studio/sfx.py)" is needed.
All effects are mono float32 at 24 kHz and are mixed under the narration at low volume.
"""
import numpy as np

SR = 24000
_rng = np.random.default_rng(30)


def _t(d): return np.arange(int(SR * d)) / SR


def _env(n, attack=0.005, release=None):
    e = np.ones(n, dtype=np.float32)
    a = max(1, int(SR * attack)); e[:a] = np.linspace(0, 1, a)
    r = n - a if release is None else max(1, int(SR * release))
    e[-r:] *= np.linspace(1, 0, r) ** 2
    return e


def _lowpass(x, k):
    k = max(1, int(k))
    return np.convolve(x, np.ones(k) / k, mode='same')


def whoosh(d=0.32):
    """Filtered noise sweep: scene transition."""
    n = int(SR * d); x = _rng.standard_normal(n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32); step = n // 16
    for i in range(16):  # sweep the filter from dull to bright and back
        k = 40 - 30 * np.sin(np.pi * i / 15)
        seg = slice(i * step, n if i == 15 else (i + 1) * step)
        out[seg] = _lowpass(x, k)[seg]
    bell = np.sin(np.pi * np.linspace(0, 1, n)) ** 2
    return (out * bell / (np.abs(out).max() + 1e-9)).astype(np.float32)


def pop(d=0.11, f0=900, f1=420):
    """Pitch-drop blip: Nibble changes pose / an element lands."""
    t = _t(d); f = np.linspace(f0, f1, len(t))
    return (np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(len(t), 0.002)).astype(np.float32)


def tick(d=0.045):
    """Clock tick: countdown."""
    t = _t(d)
    return (np.sin(2 * np.pi * 2200 * t) * np.exp(-t * 90)).astype(np.float32)


def buzz(d=0.28):
    """Low 'wrong answer' buzz: red flag revealed."""
    t = _t(d); x = np.sign(np.sin(2 * np.pi * 140 * t)) * 0.6 + np.sign(np.sin(2 * np.pi * 147 * t)) * 0.4
    return (_lowpass(x, 6) * _env(len(t), 0.004, 0.08)).astype(np.float32)


def ding(d=0.45):
    """Bright two-partial chime: correct / good sign."""
    t = _t(d)
    x = np.sin(2 * np.pi * 1320 * t) + 0.5 * np.sin(2 * np.pi * 1980 * t)
    return (x / 1.5 * np.exp(-t * 7)).astype(np.float32)


def slam(d=0.3):
    """Low thump: VS badge / stamp."""
    t = _t(d); f = np.linspace(120, 45, len(t))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 14)
    click = _lowpass(_rng.standard_normal(len(t)), 3) * np.exp(-t * 120) * 0.4
    return (body + click).astype(np.float32)


BANK = {'whoosh': whoosh, 'pop': pop, 'tick': tick, 'buzz': buzz, 'ding': ding, 'slam': slam}
LEVEL = {'whoosh': 0.05, 'pop': 0.06, 'tick': 0.07, 'buzz': 0.07, 'ding': 0.06, 'slam': 0.10}  # ~ -20 to -26 dBFS


def mix(audio, events, volume=1.0):
    """Adds [(time_s, name)] effects into a 24 kHz narration buffer (in place copy)."""
    out = audio.copy()
    cache = {}
    for t0, name in events:
        if name not in BANK: continue
        s = cache.setdefault(name, BANK[name]() * LEVEL[name] * volume)
        i = int(t0 * SR)
        if i >= len(out) or i < 0: continue
        j = min(len(out), i + len(s))
        out[i:j] += s[:j - i]
    return out
