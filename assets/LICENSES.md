# Asset license register

Every asset that can appear in a published PandaByte video or post is listed here with its source and license.
The Video Production agent (13) adds a row before using a new asset; the Video Quality agent (14) checks each VideoPackage against this file.
No row means the asset cannot be used.

Status values: `ok` (license verified), `pending` (waiting for owner confirmation, do not publish), `blocked` (do not use).

## Character art

| Asset | Source | License / rights | Added | Status |
|---|---|---|---|---|
| `assets/nibble/*.png` (avatar + 14 poses) | PandaByte original character | Owner to confirm: who created the art and whether PandaByte holds full commercial rights | 2026-11-21 | pending |

## Voice

| Asset | Source | License / rights | Added | Status |
|---|---|---|---|---|
| Kokoro-82M model + voices (`kokoro-v1.0.onnx`, `voices-v1.0.bin`), voice `am_puck` | hexgrad/Kokoro-82M, ONNX files from github.com/thewh1teagle/kokoro-onnx releases `model-files-v1.0` | Apache-2.0 (model weights); kokoro-onnx runtime MIT. Synthetic voice, not a clone of a real person | 2026-11-21 | ok |

## Fonts (used by `studio/render.py`)

| Asset | Source | License / rights | Added | Status |
|---|---|---|---|---|
| Poppins Bold | Google Fonts (Indian Type Foundry) | SIL Open Font License 1.1 | 2026-11-21 | ok |
| Inter Medium / Bold | Google Fonts (Rasmus Andersson) | SIL Open Font License 1.1 | 2026-11-21 | ok |
| DejaVu Sans / Sans Bold (fallback) | DejaVu Fonts project | Bitstream Vera / DejaVu license (free, permissive) | 2026-11-21 | ok |

## Music and sound effects

| Asset | Source | License / rights | Added | Status |
|---|---|---|---|---|
| Sound effects whoosh, pop, tick, buzz, ding, slam (`studio/sfx.py`) | Synthesized at render time from NumPy code written for PandaByte (no samples, no downloaded files) | PandaByte original output; no third-party license involved | 2026-11-21 | ok |
| Music | — | none: videos ship without a music track until a licensed track is added here | — | — |

## Tools (no license impact on output)

FFmpeg, Pillow, NumPy, soundfile: render tools only. Their licenses do not restrict the videos they produce.

## Rules

- No stock image, footage, music or voice from any site without a row here that shows a license allowing commercial social use.
- No AI-generated real people next to Nibble.
- Paid tools (e.g. ElevenLabs) need owner approval before use; record the plan tier that grants commercial rights.
- Product images from retailers or manufacturers are not covered by default; add each one with its specific terms or don't use it.
