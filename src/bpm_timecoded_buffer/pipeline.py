"""
BPM Timecoded Buffer — Beat-Grid Timecoding Pipeline for Daydream Scope

Preprocessor that:
1. Joins an Ableton Link session for a shared beat clock
2. Stamps a VJSync barcode on each input frame (encoding current beat position)
3. Creates a VACE inpainting mask that preserves the barcode through AI generation
4. Optionally applies ControlNet preprocessing (canny, depth, scribble)

The barcode survives AI processing via VACE masking (mask=0 = preserve).
The client decodes the surviving barcode on the output to know exactly which
beat produced each frame — regardless of variable Scope inference latency.

VACE integration follows Scope's dual-stream encoding:
  - vace_input_frames: [1, 3, F, H, W] in [-1, 1] — full stamped frames
  - vace_input_masks:  [1, 1, F, H, W] binary — 1=inpaint, 0=preserve
  Scope internally splits: inactive = frame*(1-mask), reactive = frame*mask

Barcode spec:
  - 16px tall, full frame width, bottom of frame
  - BCH(71,50,3): corrects up to 3 bit errors
  - Payload: beatWhole(12b) + beatFrac(8b) + frameSeq(14b) + bpm(9b) + flags(7b)
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import torch

from .vjsync_codec import (
    VJSyncPayload,
    STRIP_HEIGHT,
    encode_bpm,
    encode_beat_frac,
    decode_bpm,
    decode_beat_frac,
    stamp_barcode,
    read_barcode,
)
from .test_source import TestPatternSource

# --- Scope imports: match actual Scope package structure ---
try:
    from scope.core.pipelines.interface import Pipeline, Requirements
    from scope.core.pipelines.base_schema import (
        BasePipelineConfig, UsageType, ModeDefaults, ui_field_config,
    )
    _HAS_SCOPE = True
except ImportError:
    # Fallback for running outside Scope (standalone tests)
    class Pipeline:
        pass
    class Requirements:
        def __init__(self, input_size: int = 1):
            self.input_size = input_size
    class BasePipelineConfig:
        pass
    class UsageType:
        PREPROCESSOR = "preprocessor"
        POSTPROCESSOR = "postprocessor"
    class ModeDefaults:
        def __init__(self, default=False):
            self.default = default
    def ui_field_config(**kwargs):
        return kwargs
    _HAS_SCOPE = False

# Also try importing Field from pydantic (only needed when Scope is present)
try:
    from pydantic import Field
except ImportError:
    def Field(**kwargs):
        return kwargs.get("default")

logger = logging.getLogger(__name__)


# --- Ableton Link wrapper (runs in background thread) ---

class LinkClock:
    """
    Thin wrapper around aalink that runs the async event loop in a background
    thread. Provides synchronous getters for beat/tempo/phase.
    """

    def __init__(self, initial_bpm: float = 120.0):
        self._beat = 0.0
        self._tempo = initial_bpm
        self._phase = 0.0
        self._num_peers = 0
        self._enabled = False
        self._link = None
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event = threading.Event()

    def start(self, bpm: float = 120.0):
        """Start the Link session in a background thread."""
        if self._thread is not None:
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, args=(bpm,), daemon=True, name="link-clock"
        )
        self._thread.start()
        logger.info(f"[BPM Buffer/Link] Clock started at {bpm} BPM")

    def stop(self):
        """Stop the Link session."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._enabled = False
        logger.info("[BPM Buffer/Link] Clock stopped")

    def _run_loop(self, bpm: float):
        """Background thread: run async Link polling loop."""
        try:
            from aalink import Link
        except ImportError:
            logger.warning(
                "[BPM Buffer/Link] aalink not installed -- using free-running clock. "
                "Install with: pip install aalink"
            )
            self._run_freerunning(bpm)
            return

        loop = asyncio.new_event_loop()
        self._loop = loop

        async def poll():
            link = Link(bpm)
            link.enabled = True
            self._link = link
            self._enabled = True
            self._tempo = bpm

            logger.info(f"[BPM Buffer/Link] Ableton Link session joined at {bpm} BPM")

            while not self._stop_event.is_set():
                try:
                    beat_val = await asyncio.wait_for(
                        link.sync(1 / 16), timeout=0.1
                    )
                    self._beat = beat_val
                    self._tempo = bpm
                    self._phase = beat_val % 4.0
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    if not self._stop_event.is_set():
                        logger.debug(f"[BPM Buffer/Link] poll error: {e}")
                    await asyncio.sleep(0.016)

            link.enabled = False
            self._enabled = False

        try:
            loop.run_until_complete(poll())
        except Exception as e:
            logger.error(f"[BPM Buffer/Link] Loop error: {e}")
        finally:
            loop.close()

    def _run_freerunning(self, bpm: float):
        """Fallback: free-running beat clock when aalink is not available."""
        self._enabled = True
        self._tempo = bpm
        start_time = time.monotonic()

        while not self._stop_event.is_set():
            elapsed = time.monotonic() - start_time
            beats = elapsed * (bpm / 60.0)
            self._beat = beats
            self._phase = beats % 4.0
            time.sleep(0.008)  # ~125 Hz

        self._enabled = False

    @property
    def beat(self) -> float:
        return self._beat

    @property
    def tempo(self) -> float:
        return self._tempo

    @property
    def phase(self) -> float:
        return self._phase

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def num_peers(self) -> int:
        return self._num_peers


class ControlMode(str, Enum):
    NONE = "none"
    CANNY = "canny"
    DEPTH = "depth"
    SCRIBBLE = "scribble"


class BufferMode(str, Enum):
    STRIP = "strip"          # Just strip barcode (simplest)
    BEAT = "beat"            # Beat-quantized: release frames on beat boundaries
    LOOP = "loop"            # Loop collection: record N beats, loop playback
    LATENCY = "latency"      # Adjustable latency: FIFO buffer to smooth jitter


# --- Config ---

if _HAS_SCOPE:
    class BpmBufferConfig(BasePipelineConfig):
        """Configuration schema for BPM Timecoded Buffer."""

        # --- Class attributes (no type annotation = plain class var, not Pydantic field) ---
        # Scope registry accesses these on the CLASS, not instances
        pipeline_id = "bpm_timecoded_buffer"
        pipeline_name = "BPM Timecoded Buffer (VJ.Tools)"
        pipeline_description = (
            "Beat-grid timecoding via Ableton Link. Stamps barcode on input, "
            "preserves through AI via VACE masking. Client decodes surviving "
            "barcode on output for beat-accurate timing. "
            "Canny/Depth/Scribble control modes."
        )
        supports_prompts = False
        modified = True
        usage = [UsageType.PREPROCESSOR]
        modes = {
            "video": ModeDefaults(default=True),
            "text": ModeDefaults(default=True),
        }

        # --- Load-time parameters (Pydantic fields with annotations) ---

        initial_bpm: float = Field(
            default=120.0,
            ge=60.0,
            le=300.0,
            json_schema_extra=ui_field_config(
                order=0,
                label="Initial BPM",
                is_load_param=True,
            ),
        )

        # --- Runtime parameters ---

        barcode_height: int = Field(
            default=16,
            ge=4,
            le=128,
            json_schema_extra=ui_field_config(
                order=1,
                label="Barcode Height (px)",
                is_load_param=False,
            ),
        )

        control_mode: ControlMode = Field(
            default=ControlMode.NONE,
            json_schema_extra=ui_field_config(
                order=2,
                label="Control Mode",
                is_load_param=False,
            ),
        )

        canny_low: int = Field(
            default=50,
            ge=0,
            le=255,
            json_schema_extra=ui_field_config(
                order=3,
                label="Canny Low Threshold",
                is_load_param=False,
            ),
        )

        canny_high: int = Field(
            default=150,
            ge=0,
            le=255,
            json_schema_extra=ui_field_config(
                order=4,
                label="Canny High Threshold",
                is_load_param=False,
            ),
        )

        mask_feather: int = Field(
            default=2,
            ge=0,
            le=16,
            json_schema_extra=ui_field_config(
                order=5,
                label="Mask Feather (px)",
                is_load_param=False,
            ),
        )

        strip_barcode: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=6,
                label="Strip Barcode on Output",
                is_load_param=False,
            ),
        )

        test_input: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=7,
                label="Test Pattern Input",
                is_load_param=False,
            ),
        )

        tap_tempo: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=8,
                label="Tap Tempo",
                is_load_param=False,
            ),
        )
else:
    class BpmBufferConfig:
        """Standalone config (no Pydantic) for testing outside Scope."""
        def __init__(self, **kwargs):
            self.pipeline_id = kwargs.get("pipeline_id", "bpm_timecoded_buffer")
            self.pipeline_name = kwargs.get("pipeline_name", "BPM Timecoded Buffer (VJ.Tools)")
            self.initial_bpm = kwargs.get("initial_bpm", 120.0)
            self.barcode_height = kwargs.get("barcode_height", 16)
            self.control_mode = kwargs.get("control_mode", "none")
            self.canny_low = kwargs.get("canny_low", 50)
            self.canny_high = kwargs.get("canny_high", 150)
            self.mask_feather = kwargs.get("mask_feather", 2)
            self.strip_barcode = kwargs.get("strip_barcode", False)
            self.test_input = kwargs.get("test_input", False)
            self.tap_tempo = kwargs.get("tap_tempo", False)


# --- Pipeline ---

class BpmTimecodedBufferPipeline(Pipeline):
    """
    Scope preprocessor that stamps beat-grid timecodes using Ableton Link
    and creates VACE masks to preserve them through AI generation.

    The barcode is stamped on input, survives through diffusion via VACE
    masking (mask=0 at the barcode strip), and the client decodes it on
    the output to know the exact beat position that produced each frame.

    VACE mask format (matching Scope's VaceEncodingBlock):
      - vace_input_frames [1,3,F,H,W] [-1,1]: full frames with barcode
        Scope splits internally: inactive = frame*(1-mask), reactive = frame*mask
      - vace_input_masks [1,1,F,H,W] binary: 1=generate, 0=preserve
    """

    @classmethod
    def get_config_class(cls):
        return BpmBufferConfig

    def __init__(
        self,
        config=None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,  # Scope passes height, width, quantization, loras, etc.
    ):
        if config is None:
            config = BpmBufferConfig() if _HAS_SCOPE else type('Config', (), kwargs)()
        self.config = config
        self.device = (
            device if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self._frame_seq = 0

        initial_bpm = getattr(config, "initial_bpm", 120.0)
        self._clock = LinkClock(initial_bpm)
        self._clock.start(initial_bpm)

        # Test pattern source (lazy-initialized on first use)
        self._test_source: Optional[TestPatternSource] = None

        logger.info(
            f"[BPM Buffer] Pipeline initialized "
            f"(device={self.device}, bpm={initial_bpm})"
        )

    def __del__(self):
        if hasattr(self, "_clock"):
            self._clock.stop()

    def tap_bpm(self) -> Optional[float]:
        """
        Tap tempo — call repeatedly to tap in a BPM.
        Useful on RunPod or anywhere without Ableton Link.
        Returns detected BPM after 2+ taps, or None after first tap.
        """
        if self._test_source is None:
            self._test_source = TestPatternSource()
        detected = self._test_source.tap()
        if detected is not None:
            self._clock._tempo = detected
            logger.info(f"[BPM Buffer] Tap tempo: {detected:.1f} BPM")
        return detected

    def set_bpm(self, bpm: float):
        """Manually set BPM (for RunPod / no-Link scenarios)."""
        bpm = max(20.0, min(999.0, bpm))
        self._clock._tempo = bpm
        if self._test_source:
            self._test_source.set_bpm(bpm)
        logger.info(f"[BPM Buffer] Manual BPM: {bpm:.1f}")

    def prepare(self, **kwargs) -> "Requirements":
        """Tell Scope how many input frames we need per call."""
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """
        Process input frames (Scope Pipeline interface).

        Scope passes video as kwargs["video"] — a list of (1, H, W, C) uint8 tensors.

        Returns:
            dict with video, vace_input_frames, vace_input_masks
        """
        video = kwargs.get("video", [])
        barcode_h = kwargs.get("barcode_height", getattr(self.config, "barcode_height", 16))
        control_mode = str(kwargs.get("control_mode", getattr(self.config, "control_mode", "none")))
        canny_low = kwargs.get("canny_low", getattr(self.config, "canny_low", 50))
        canny_high = kwargs.get("canny_high", getattr(self.config, "canny_high", 150))
        mask_feather = kwargs.get("mask_feather", getattr(self.config, "mask_feather", 2))
        strip_barcode = kwargs.get("strip_barcode", getattr(self.config, "strip_barcode", False))
        test_input = kwargs.get("test_input", getattr(self.config, "test_input", False))
        tap_tempo = kwargs.get("tap_tempo", getattr(self.config, "tap_tempo", False))

        # --- Tap tempo (boolean toggle triggers a tap) ---
        if tap_tempo:
            self.tap_bpm()

        # Handle both list and tensor input, or empty
        if isinstance(video, list) and len(video) == 0:
            return {"video": torch.zeros(1, 1, 1, 3)}

        # --- Test pattern override ---
        if test_input:
            ref = video[0]
            _, H, W, _ = ref.shape
            if self._test_source is None or self._test_source.width != W or self._test_source.height != H:
                self._test_source = TestPatternSource(width=W, height=H)
            video = self._test_source.generate_batch(
                self._clock, num_frames=len(video), barcode_height=barcode_h
            )

        # Stack frames
        frames = torch.cat(video, dim=0).float()  # (F, H, W, C), [0, 255]
        F, H, W, C = frames.shape
        barcode_h = min(barcode_h, H // 4)

        # --- 1. Stamp barcode on each frame using Link beat clock ---
        frames_np = frames.cpu().numpy().astype(np.uint8)

        for f_idx in range(F):
            beat = self._clock.beat
            bpm = self._clock.tempo

            beat_whole = int(beat) & 0xFFF
            beat_frac = encode_beat_frac(beat - int(beat))
            bpm_enc = encode_bpm(bpm)

            payload = VJSyncPayload(
                beat_whole=beat_whole,
                beat_frac=beat_frac,
                frame_seq=self._frame_seq & 0x3FFF,
                bpm_encoded=bpm_enc,
                flags=0,
            )

            stamp_barcode(frames_np[f_idx], payload)
            self._frame_seq += 1

        # Convert back to tensor with stamped barcodes
        frames_stamped = torch.from_numpy(frames_np).float()

        # --- 2. Generate VACE mask ---
        # mask=1 -> inpaint (AI generates), mask=0 -> preserve (barcode survives)
        mask = torch.ones(F, H, W, dtype=torch.float32)
        mask[:, -barcode_h:, :] = 0.0  # Preserve barcode strip

        # Feathering at boundary for smooth transition
        if mask_feather > 0:
            boundary_y = H - barcode_h
            for dy in range(mask_feather):
                if boundary_y - dy - 1 >= 0:
                    alpha = (dy + 1) / (mask_feather + 1)
                    mask[:, boundary_y - dy - 1, :] = alpha

        # --- 3. Build VACE tensors ---
        frames_01 = frames_stamped / 255.0  # (F, H, W, C), [0, 1]
        vace_frames = (frames_01 * 2.0 - 1.0).permute(3, 0, 1, 2).unsqueeze(0)
        # -> (1, C=3, F, H, W), [-1, 1]
        vace_frames = vace_frames.to(device=self.device, dtype=self.dtype)

        vace_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, F, H, W)
        vace_mask = vace_mask.to(device=self.device, dtype=self.dtype)

        # --- 4. Optional ControlNet preprocessing ---
        control_video = None
        if control_mode != "none":
            control_video = self._apply_control(
                frames_stamped, barcode_h, control_mode, canny_low, canny_high
            )

        # --- 5. Display output ---
        # Scope pipeline_processor does: if dtype != uint8: value = (value * 255).clamp(0,255).uint8
        # So we return [0, 1] float and let Scope handle the conversion.
        frames_01 = frames_stamped / 255.0  # (F, H, W, C), [0, 1]
        display = frames_01.clone()
        if strip_barcode:
            display[:, -barcode_h:, :, :] = 0.0
        else:
            # Show barcode with subtle green tint so you can verify it survives
            mask_cpu = mask.unsqueeze(-1)  # (F, H, W, 1)
            preserve = 1.0 - mask_cpu
            display = (display + preserve * torch.tensor([0.0, 0.05, 0.0])).clamp(0.0, 1.0)

        logger.info(
            f"[BPM Buffer] Output: shape={display.shape}, dtype={display.dtype}, "
            f"min={display.min():.3f}, max={display.max():.3f}, device={display.device}"
        )

        result = {"video": display}

        # Only forward VACE tensors when the downstream pipeline supports VACE.
        # Scope sets vace_enabled=True in kwargs when the frontend has VACE mode on.
        # If we always forward these, they poison the downstream call_params and
        # prevent video from being assigned (pipeline_processor skips video assignment
        # when vace_input_frames is already in call_params).
        if kwargs.get("vace_enabled", False):
            result["vace_input_frames"] = vace_frames
            result["vace_input_masks"] = vace_mask

        if control_video is not None:
            result["control_video"] = control_video

        # Link state for diagnostics
        result["_bpm_buffer_meta"] = {
            "beat": self._clock.beat,
            "bpm": self._clock.tempo,
            "link_enabled": self._clock.enabled,
            "frame_seq": self._frame_seq,
        }

        return result

    def _apply_control(
        self,
        frames: torch.Tensor,
        barcode_h: int,
        mode: str,
        canny_low: int,
        canny_high: int,
    ) -> torch.Tensor:
        """Apply ControlNet preprocessing to content region (above barcode)."""
        F, H, W, C = frames.shape
        content_h = H - barcode_h

        frames_np = frames.cpu().numpy().astype(np.uint8)
        control_np = np.zeros_like(frames_np, dtype=np.uint8)

        for f_idx in range(F):
            content = frames_np[f_idx, :content_h, :, :]

            if mode == "canny":
                gray = cv2.cvtColor(content, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, canny_low, canny_high)
                control_np[f_idx, :content_h, :, :] = np.stack([edges] * 3, axis=-1)

            elif mode == "depth":
                gray = cv2.cvtColor(content, cv2.COLOR_RGB2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                lap = np.abs(cv2.Laplacian(blurred, cv2.CV_64F))
                depth = (lap / (lap.max() + 1e-8) * 255).astype(np.uint8)
                depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
                control_np[f_idx, :content_h, :, :] = cv2.cvtColor(
                    depth_color, cv2.COLOR_BGR2RGB
                )

            elif mode == "scribble":
                gray = cv2.cvtColor(content, cv2.COLOR_RGB2GRAY)
                blur = cv2.GaussianBlur(gray, (3, 3), 0)
                edges = cv2.adaptiveThreshold(
                    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                kernel = np.ones((2, 2), np.uint8)
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
                control_np[f_idx, :content_h, :, :] = np.stack([edges] * 3, axis=-1)

        control = torch.from_numpy(control_np).float() / 255.0
        return control.to(device=self.device)


# --- Postprocessor Config ---

if _HAS_SCOPE:
    class BpmStripConfig(BasePipelineConfig):
        """Configuration schema for BPM Timecoded Buffer postprocessor."""

        # Class attributes (no annotation = not a Pydantic field)
        pipeline_id = "bpm_timecode_strip"
        pipeline_name = "BPM Timecode Buffer Output (VJ.Tools)"
        pipeline_description = (
            "Postprocessor that decodes timecode barcodes from AI output and "
            "provides beat-quantized, loop, or latency-compensated buffering. "
            "Strips the barcode from output so viewers don't see it."
        )
        supports_prompts = False
        modified = True
        usage = [UsageType.POSTPROCESSOR]
        modes = {
            "video": ModeDefaults(default=True),
            "text": ModeDefaults(default=True),
        }

        # Pydantic fields
        barcode_height: int = Field(
            default=16,
            ge=4,
            le=128,
            json_schema_extra=ui_field_config(
                order=0,
                label="Barcode Height (px)",
                is_load_param=False,
            ),
        )

        buffer_mode: BufferMode = Field(
            default=BufferMode.STRIP,
            json_schema_extra=ui_field_config(
                order=1,
                label="Buffer Mode",
                is_load_param=False,
            ),
        )

        loop_length_beats: int = Field(
            default=8,
            ge=1,
            le=64,
            json_schema_extra=ui_field_config(
                order=2,
                label="Loop Length (beats)",
                is_load_param=False,
            ),
        )

        latency_delay_ms: int = Field(
            default=100,
            ge=0,
            le=2000,
            json_schema_extra=ui_field_config(
                order=3,
                label="Latency Buffer (ms)",
                is_load_param=False,
            ),
        )

        beat_hold_beats: int = Field(
            default=2,
            ge=1,
            le=64,
            json_schema_extra=ui_field_config(
                order=4,
                label="Beat Hold Window",
                is_load_param=False,
            ),
        )

        link_sync: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=5,
                label="Ableton Link Sync",
                is_load_param=False,
            ),
        )

        link_bpm: float = Field(
            default=120.0,
            ge=20.0,
            le=999.0,
            json_schema_extra=ui_field_config(
                order=6,
                label="Link BPM (read-only when synced)",
                is_load_param=False,
            ),
        )

        hold: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=7,
                label="HOLD (freeze playback)",
                is_load_param=False,
            ),
        )

        min_buffer_beats: int = Field(
            default=8,
            ge=1,
            le=32,
            json_schema_extra=ui_field_config(
                order=8,
                label="Min Buffer (beats)",
                is_load_param=False,
            ),
        )

        auto_loop_beats: int = Field(
            default=4,
            ge=2,
            le=8,
            json_schema_extra=ui_field_config(
                order=9,
                label="Auto-Loop Length (beats)",
                is_load_param=False,
            ),
        )

        jump_ahead: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=10,
                label="Jump Ahead (trim to min buffer)",
                is_load_param=False,
            ),
        )

        loop_reset: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=11,
                label="Reset Loop",
                is_load_param=False,
            ),
        )

        latency_nudge_ms: int = Field(
            default=10,
            ge=1,
            le=100,
            json_schema_extra=ui_field_config(
                order=12,
                label="Latency Nudge Step (ms)",
                is_load_param=False,
            ),
        )
else:
    class BpmStripConfig:
        """Standalone config for testing outside Scope."""
        def __init__(self, **kwargs):
            self.pipeline_id = kwargs.get("pipeline_id", "bpm_timecode_strip")
            self.pipeline_name = kwargs.get("pipeline_name", "BPM Timecode Buffer Output (VJ.Tools)")
            self.barcode_height = kwargs.get("barcode_height", 16)
            self.buffer_mode = kwargs.get("buffer_mode", "strip")
            self.loop_length_beats = kwargs.get("loop_length_beats", 8)
            self.latency_delay_ms = kwargs.get("latency_delay_ms", 100)
            self.beat_hold_beats = kwargs.get("beat_hold_beats", 2)
            self.link_sync = kwargs.get("link_sync", False)
            self.link_bpm = kwargs.get("link_bpm", 120.0)
            self.hold = kwargs.get("hold", False)
            self.min_buffer_beats = kwargs.get("min_buffer_beats", 8)
            self.auto_loop_beats = kwargs.get("auto_loop_beats", 4)
            self.jump_ahead = kwargs.get("jump_ahead", False)
            self.loop_reset = kwargs.get("loop_reset", False)
            self.latency_nudge_ms = kwargs.get("latency_nudge_ms", 10)


# --- Buffered Frame ---

@dataclass
class _BufferedFrame:
    """A decoded frame with its timecode metadata."""
    frame: np.ndarray        # (H, W, C) uint8, barcode stripped
    beat: float              # decoded beat position
    bpm: float               # decoded BPM
    frame_seq: int           # decoded frame sequence number
    timestamp: float         # time.monotonic() when received


# --- Postprocessor Pipeline ---

class BpmTimecodeStripPipeline(Pipeline):
    """
    Scope postprocessor that decodes timecode barcodes from AI output,
    strips the barcode, and optionally buffers frames for beat-accurate
    playback.

    Buffer modes:
      - strip:   Just strip the barcode (pass-through, simplest)
      - beat:    Beat-quantized — hold frames until the next beat boundary,
                 then release the frame closest to that beat
      - loop:    Loop collection — record N beats of frames into a loop,
                 then cycle playback beat-synced
      - latency: Adjustable latency — FIFO buffer with configurable delay
                 to smooth out variable inference jitter
    """

    @classmethod
    def get_config_class(cls):
        return BpmStripConfig

    def __init__(
        self,
        config=None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,  # Scope passes height, width, quantization, loras, etc.
    ):
        if config is None:
            config = BpmStripConfig() if _HAS_SCOPE else type('Config', (), kwargs)()
        self.config = config
        self.device = (
            device if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = dtype if self.device.type == "cuda" else torch.float32

        # Ableton Link clock for local sync
        self._clock: Optional[LinkClock] = None
        self._link_active = False

        # Auto-start Link if config says so
        link_sync = getattr(config, "link_sync", False)
        if link_sync:
            link_bpm = getattr(config, "link_bpm", 120.0)
            self._start_link(link_bpm)

        # Beat-quantized buffer
        self._beat_buffer: list[_BufferedFrame] = []
        self._last_released_beat: int = -1
        self._last_released_frame: Optional[np.ndarray] = None
        self._latest_decoded_beat: float = -1.0
        self._prev_decoded_beat: float = -1.0

        # Hold state (CDJ-style freeze)
        self._hold_active: bool = False
        self._hold_playhead_beat: int = -1
        self._playback_cursor: int = -1  # -1 = normal, >= 0 = offset playback

        # Auto-loop on catchup
        self._auto_loop_active: bool = False
        self._auto_loop_start: int = -1
        self._auto_loop_end: int = -1
        self._auto_loop_cursor: int = -1

        # Loop buffer
        self._loop_frames: list[_BufferedFrame] = []
        self._loop_recording: bool = True
        self._loop_start_beat: Optional[int] = None
        self._loop_playhead: int = 0

        # Latency FIFO
        self._latency_fifo: list[_BufferedFrame] = []

        # Stats
        self._decode_success = 0
        self._decode_fail = 0

        logger.info("[BPM Buffer Output] Postprocessor initialized")

    def _start_link(self, bpm: float = 120.0):
        """Start Ableton Link clock for local beat sync."""
        if self._clock is not None:
            return
        self._clock = LinkClock(bpm)
        self._clock.start(bpm)
        self._link_active = True
        logger.info(f"[BPM Buffer Output] Ableton Link started at {bpm} BPM")

    def _stop_link(self):
        """Stop Ableton Link clock."""
        if self._clock is not None:
            self._clock.stop()
            self._clock = None
        self._link_active = False
        logger.info("[BPM Buffer Output] Ableton Link stopped")

    def __del__(self):
        if hasattr(self, "_clock") and self._clock is not None:
            self._clock.stop()

    def prepare(self, **kwargs) -> "Requirements":
        """Tell Scope how many input frames we need per call."""
        return Requirements(input_size=1)

    def __call__(self, **kwargs) -> dict:
        """
        Decode barcode, strip it, and apply buffer mode.

        Scope passes video as kwargs["video"] — a list of (1, H, W, C) uint8 tensors.
        """
        video = kwargs.get("video", [])

        # Handle both list and tensor input, or empty
        if isinstance(video, list) and len(video) == 0:
            return {"video": torch.zeros(1, 1, 1, 3)}

        # Read params from kwargs first (Scope runtime updates), fall back to config
        barcode_h = kwargs.get("barcode_height", getattr(self.config, "barcode_height", 16))
        mode = str(kwargs.get("buffer_mode", getattr(self.config, "buffer_mode", "strip")))
        loop_len = kwargs.get("loop_length_beats", getattr(self.config, "loop_length_beats", 8))
        latency_ms = kwargs.get("latency_delay_ms", getattr(self.config, "latency_delay_ms", 100))
        beat_hold = kwargs.get("beat_hold_beats", getattr(self.config, "beat_hold_beats", 2))
        loop_reset = kwargs.get("loop_reset", getattr(self.config, "loop_reset", False))
        link_sync = kwargs.get("link_sync", getattr(self.config, "link_sync", False))
        hold = kwargs.get("hold", getattr(self.config, "hold", False))
        min_buffer = kwargs.get("min_buffer_beats", getattr(self.config, "min_buffer_beats", 8))
        auto_loop_beats = kwargs.get("auto_loop_beats", getattr(self.config, "auto_loop_beats", 4))
        jump_ahead = kwargs.get("jump_ahead", getattr(self.config, "jump_ahead", False))

        # --- Ableton Link toggle ---
        if link_sync and not self._link_active:
            link_bpm = kwargs.get("link_bpm", getattr(self.config, "link_bpm", 120.0))
            self._start_link(link_bpm)
        elif not link_sync and self._link_active:
            self._stop_link()

        # --- Loop reset trigger ---
        if loop_reset:
            self._loop_frames.clear()
            self._loop_recording = True
            self._loop_start_beat = None
            self._loop_playhead = 0
            logger.info("[BPM Buffer Output] Loop reset")

        # --- Hold toggle (CDJ-style) ---
        if hold and not self._hold_active:
            # Engage hold — freeze at current playback position
            self._hold_active = True
            if self._playback_cursor >= 0:
                self._hold_playhead_beat = self._playback_cursor
            elif self._latest_decoded_beat >= 0:
                self._hold_playhead_beat = int(self._latest_decoded_beat) - beat_hold
            else:
                self._hold_playhead_beat = self._last_released_beat
            logger.info(f"[BPM Buffer Output] HOLD engaged at beat {self._hold_playhead_beat}")
        elif not hold and self._hold_active:
            # Release hold — resume from held position
            self._hold_active = False
            self._playback_cursor = self._hold_playhead_beat
            self._hold_playhead_beat = -1
            self._auto_loop_active = False
            logger.info(f"[BPM Buffer Output] HOLD released, cursor at {self._playback_cursor}")

        # --- Jump Ahead trigger ---
        if jump_ahead and self._latest_decoded_beat >= 0:
            effective_min = max(min_buffer, beat_hold)
            latest_whole = int(self._latest_decoded_beat)
            normal_target = latest_whole - beat_hold
            current_playback = self._playback_cursor if self._playback_cursor >= 0 else normal_target
            gap = normal_target - current_playback
            if gap > 0:
                new_target = latest_whole - effective_min
                self._playback_cursor = new_target
                self._auto_loop_active = False
                # Prune frames we jumped past
                self._beat_buffer = [
                    bf for bf in self._beat_buffer
                    if bf.beat >= new_target - 2
                ]
                logger.info(
                    f"[BPM Buffer Output] JUMP AHEAD: skipped {gap - effective_min} beats, "
                    f"cursor now at {new_target}"
                )

        # Stack frames — handle both list of tensors and single tensor
        if isinstance(video, list):
            frames = torch.cat(video, dim=0).float()
        else:
            frames = video.float() if video.dim() == 4 else video.unsqueeze(0).float()

        # Auto-detect [0,1] vs [0,255] range and normalize to [0,255]
        if frames.max() <= 1.0:
            frames = frames * 255.0

        F, H, W, C = frames.shape
        barcode_h = min(barcode_h, H // 4)
        now = time.monotonic()

        # --- Decode barcodes and build buffered frames ---
        incoming: list[_BufferedFrame] = []
        for f_idx in range(F):
            frame_np = frames[f_idx].cpu().numpy().astype(np.uint8)
            payload = read_barcode(frame_np, barcode_h)

            if payload is not None:
                self._decode_success += 1
                beat = payload.beat_whole + decode_beat_frac(payload.beat_frac)
                bpm = decode_bpm(payload.bpm_encoded)

                # Strip barcode
                frame_np[-barcode_h:, :, :] = 0

                incoming.append(_BufferedFrame(
                    frame=frame_np,
                    beat=beat,
                    bpm=bpm,
                    frame_seq=payload.frame_seq,
                    timestamp=now,
                ))
            else:
                self._decode_fail += 1
                # Still strip barcode even if decode fails
                frame_np[-barcode_h:, :, :] = 0

                # Use Link beat if available, otherwise fallback
                if self._link_active and self._clock is not None:
                    fallback_beat = self._clock.beat
                    fallback_bpm = self._clock.tempo
                else:
                    fallback_beat = 0.0
                    fallback_bpm = 120.0

                incoming.append(_BufferedFrame(
                    frame=frame_np,
                    beat=fallback_beat,
                    bpm=fallback_bpm,
                    frame_seq=0,
                    timestamp=now,
                ))

        # --- Apply buffer mode ---
        if mode == "beat":
            output_frames = self._process_beat_quantized(
                incoming, beat_hold, auto_loop_beats
            )
        elif mode == "loop":
            output_frames = self._process_loop(incoming, loop_len)
        elif mode == "latency":
            output_frames = self._process_latency(incoming, latency_ms)
        else:
            # strip mode: pass through with barcode stripped
            output_frames = [bf.frame for bf in incoming]

        # Log every 100 frames for diagnostics
        total = self._decode_success + self._decode_fail
        if total % 100 == 1:
            # Calculate buffer fill
            if self._latest_decoded_beat >= 0:
                current_pb = self._playback_cursor if self._playback_cursor >= 0 else int(self._latest_decoded_beat) - beat_hold
                fill = int(self._latest_decoded_beat) - current_pb
            else:
                fill = 0
            logger.info(
                f"[BPM Buffer Output] mode={mode}, decode={self._decode_success}/{total}, "
                f"link={self._link_active}, hold={self._hold_active}, "
                f"fill={fill}b, autoloop={self._auto_loop_active}, "
                f"incoming={len(incoming)}, output={len(output_frames)}"
            )

        # Convert back to tensor
        if not output_frames:
            output_frames = [incoming[0].frame] if incoming else [np.zeros((H, W, C), dtype=np.uint8)]

        out_np = np.stack(output_frames, axis=0)  # (F, H, W, C)
        out_tensor = torch.from_numpy(out_np).float() / 255.0  # [0, 1] float

        result = {"video": out_tensor}

        # Diagnostics
        total = self._decode_success + self._decode_fail
        rate = self._decode_success / total if total > 0 else 0.0
        if self._latest_decoded_beat >= 0:
            current_pb = self._playback_cursor if self._playback_cursor >= 0 else int(self._latest_decoded_beat) - beat_hold
            fill = int(self._latest_decoded_beat) - current_pb
        else:
            fill = 0
        result["_bpm_buffer_output_meta"] = {
            "decode_rate": rate,
            "decode_success": self._decode_success,
            "decode_fail": self._decode_fail,
            "buffer_mode": mode,
            "buffer_size": len(self._beat_buffer) + len(self._loop_frames) + len(self._latency_fifo),
            "buffer_fill_beats": fill,
            "hold_active": self._hold_active,
            "auto_loop_active": self._auto_loop_active,
            "playback_cursor": self._playback_cursor,
            "link_active": self._link_active,
            "link_beat": self._clock.beat if self._clock else None,
            "link_bpm": self._clock.tempo if self._clock else None,
            "link_peers": self._clock.num_peers if self._clock else 0,
        }

        return result

    def _process_beat_quantized(
        self,
        incoming: list[_BufferedFrame],
        hold_beats: int = 2,
        auto_loop_beats: int = 4,
    ) -> list[np.ndarray]:
        """
        Beat-quantized mode with CDJ-style hold, auto-loop, and playback cursor.

        Three states:
        1. HOLD active → freeze at held beat, keep ingesting
        2. AUTO-LOOP → cycle last N beats when buffer exhausted
        3. Normal → advance playback cursor or use (latest - depth)

        Between beat boundaries, the last released frame is held frozen
        (stepped/stutter effect synced to the beat grid).
        """
        self._beat_buffer.extend(incoming)

        if not self._beat_buffer:
            return [incoming[-1].frame] if incoming else []

        # Track latest decoded beat
        if self._link_active and self._clock is not None:
            latest_beat = self._clock.beat
        else:
            latest_beat = max(bf.beat for bf in self._beat_buffer)

        latest_whole = int(latest_beat)

        # Advance playback cursor when new beats arrive
        if latest_whole > int(self._latest_decoded_beat) and self._latest_decoded_beat >= 0:
            advance = latest_whole - int(self._latest_decoded_beat)
            if self._playback_cursor >= 0 and advance > 0 and advance < 16:
                self._playback_cursor += advance
                # Check if cursor caught up to normal target
                normal_target = latest_whole - hold_beats
                if self._playback_cursor >= normal_target:
                    self._playback_cursor = -1  # resume normal

        self._prev_decoded_beat = self._latest_decoded_beat
        self._latest_decoded_beat = latest_beat

        # --- HOLD: freeze playback ---
        if self._hold_active:
            if self._hold_playhead_beat >= 0:
                # Find frame at held beat
                best = None
                best_dist = float('inf')
                for bf in self._beat_buffer:
                    d = abs(bf.beat - self._hold_playhead_beat)
                    if d < best_dist:
                        best_dist = d
                        best = bf
                if best is not None:
                    self._last_released_frame = best.frame

            # Don't evict during hold — keep all frames
            # But cap at 256 to prevent memory blowout
            if len(self._beat_buffer) > 256:
                self._beat_buffer = self._beat_buffer[-256:]

            if self._last_released_frame is not None:
                return [self._last_released_frame]
            return [self._beat_buffer[-1].frame]

        # --- Compute target beat ---
        if self._playback_cursor >= 0:
            target_whole = self._playback_cursor
        else:
            target_whole = latest_whole - hold_beats

        # --- AUTO-LOOP: check catchup ---
        if self._auto_loop_active:
            # Check if enough new frames arrived to exit
            available = latest_whole - target_whole
            if available >= auto_loop_beats:
                self._auto_loop_active = False
                self._auto_loop_cursor = -1
            else:
                # Advance loop cursor
                self._auto_loop_cursor += 1
                loop_len = self._auto_loop_end - self._auto_loop_start
                if loop_len > 0 and (self._auto_loop_cursor - self._auto_loop_start) >= loop_len:
                    self._auto_loop_cursor = self._auto_loop_start

                # Find frame at loop cursor position
                best = None
                best_dist = float('inf')
                for bf in self._beat_buffer:
                    d = abs(bf.beat - self._auto_loop_cursor)
                    if d < best_dist:
                        best_dist = d
                        best = bf
                if best is not None:
                    self._last_released_frame = best.frame

                if self._last_released_frame is not None:
                    return [self._last_released_frame]
                return [self._beat_buffer[-1].frame]

        # --- Normal playback ---
        if target_whole > self._last_released_beat:
            # New beat boundary
            target_f = float(target_whole)
            best = min(self._beat_buffer, key=lambda bf: abs(bf.beat - target_f))
            self._last_released_frame = best.frame
            self._last_released_beat = target_whole
        elif target_whole < self._last_released_beat and not self._beat_buffer:
            # No frames at target — check if we should auto-loop
            pass

        # Check if buffer exhausted (playback caught up to live)
        available = latest_whole - target_whole
        if available <= 0 and len(self._beat_buffer) >= 2 and not self._auto_loop_active:
            self._auto_loop_active = True
            self._auto_loop_end = target_whole
            self._auto_loop_start = target_whole - auto_loop_beats
            self._auto_loop_cursor = self._auto_loop_start
            logger.info(
                f"[BPM Buffer Output] Auto-loop engaged: "
                f"beats {self._auto_loop_start}-{self._auto_loop_end}"
            )

        # Evict old frames — keep frames from earliest needed position
        earliest = min(target_whole, self._auto_loop_start if self._auto_loop_active else target_whole)
        self._beat_buffer = [
            bf for bf in self._beat_buffer
            if bf.beat >= earliest - 2
        ]
        # Hard cap
        if len(self._beat_buffer) > 256:
            self._beat_buffer = self._beat_buffer[-256:]

        if self._last_released_frame is not None:
            return [self._last_released_frame]
        return [self._beat_buffer[-1].frame]

    def _process_loop(
        self, incoming: list[_BufferedFrame], loop_length_beats: int
    ) -> list[np.ndarray]:
        """
        Loop mode: record frames for N beats, then loop playback.
        Uses Ableton Link beat when available for accurate loop boundaries.
        """
        if self._loop_recording:
            # Recording phase
            for bf in incoming:
                # Use Link beat for loop timing if available
                current_beat = bf.beat
                if self._link_active and self._clock is not None:
                    current_beat = self._clock.beat

                if self._loop_start_beat is None:
                    self._loop_start_beat = int(current_beat)

                beats_elapsed = current_beat - self._loop_start_beat
                if beats_elapsed < loop_length_beats:
                    self._loop_frames.append(bf)
                else:
                    # Done recording
                    self._loop_recording = False
                    logger.info(
                        f"[BPM Buffer Output] Loop recorded: {len(self._loop_frames)} "
                        f"frames over {loop_length_beats} beats"
                    )
                    break

            if self._loop_recording:
                # Still recording — pass through
                return [bf.frame for bf in incoming]

        # Playback phase — cycle through recorded frames
        if not self._loop_frames:
            return [bf.frame for bf in incoming]

        output = []
        for _ in incoming:
            idx = self._loop_playhead % len(self._loop_frames)
            output.append(self._loop_frames[idx].frame)
            self._loop_playhead += 1

        return output

    def _process_latency(
        self, incoming: list[_BufferedFrame], delay_ms: int
    ) -> list[np.ndarray]:
        """
        Latency compensation mode: FIFO buffer with configurable delay.
        Smooths out variable inference latency for consistent output timing.
        """
        self._latency_fifo.extend(incoming)

        # Calculate how many frames the delay corresponds to
        # Use latest BPM to estimate frame rate (Scope typically runs ~30fps)
        fps_estimate = 30.0
        delay_frames = max(1, int((delay_ms / 1000.0) * fps_estimate))

        output = []
        while len(self._latency_fifo) > delay_frames:
            bf = self._latency_fifo.pop(0)
            output.append(bf.frame)

        if not output:
            # Buffer not full yet — output oldest available
            if self._latency_fifo:
                output.append(self._latency_fifo[0].frame)
            else:
                output.append(incoming[-1].frame)

        # Cap FIFO size to prevent memory growth
        max_fifo = delay_frames * 3
        if len(self._latency_fifo) > max_fifo:
            self._latency_fifo = self._latency_fifo[-delay_frames:]

        return output
