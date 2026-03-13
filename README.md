# BPM Timecoded Buffer (VJ.Tools)

Daydream Scope plugin that stamps beat-grid timecodes on video frames using Ableton Link and preserves them through AI generation via VACE masking.

The barcode survives diffusion — the client decodes it on AI output to know the exact beat position that produced each frame, regardless of variable inference latency.

## How it works

```
Input Video  -->  [Stamp Barcode]  -->  [VACE Mask]  -->  Scope AI  -->  Output
                       |                    |                              |
                  Link beat clock      mask=0: preserve            Barcode survives!
                  encodes beat#        mask=1: AI generates        Client decodes beat#
                  into bottom 16px     (barcode protected)         for beat-synced buffer
```

1. **Ableton Link** provides a shared beat clock (joins any Link session on the network)
2. **BCH barcode** encodes beat position, BPM, frame sequence into a 16px strip at the bottom of each frame
3. **VACE mask** tells Scope's diffusion pipeline to preserve the barcode (mask=0) while generating content above it (mask=1)
4. **ControlNet** (optional) extracts canny/depth/scribble from the content region to guide AI generation
5. The barcode **survives through AI** and the client decodes it to build a beat-synchronized output buffer

## Install

```bash
# From this repo
pip install .

# With Ableton Link support
pip install ".[link]"
```

## Barcode Spec

- **BCH(71,50,3)**: corrects up to 3 bit errors per frame
- **85 bars**: 14 sync + 71 data/ECC, 6px per bar, 16px tall
- **ITU-R limited range**: Y=16 (black), Y=235 (white)
- **50-bit payload**:
  - `beatWhole` (12b): beat number 0-4095
  - `beatFrac` (8b): fractional beat 0-255
  - `frameSeq` (14b): frame sequence 0-16383
  - `bpm` (9b): BPM 60-571 (offset encoding)
  - `flags` (7b): reserved

## Parameters

### Load-time
| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_bpm` | 120.0 | Starting BPM for the Link session |

### Runtime
| Parameter | Default | Description |
|-----------|---------|-------------|
| `barcode_height` | 16 | Height of barcode strip (px) |
| `control_mode` | none | ControlNet mode: none, canny, depth, scribble |
| `canny_low` | 50 | Canny edge low threshold |
| `canny_high` | 150 | Canny edge high threshold |
| `mask_feather` | 2 | Soft transition at barcode boundary (px) |
| `strip_barcode` | false | Black out barcode in display output |

## Test

```bash
python tests/test_pipeline.py
```

## License

MIT
