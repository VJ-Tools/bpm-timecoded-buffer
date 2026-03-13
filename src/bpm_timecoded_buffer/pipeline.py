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
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import torch
from pydantic import Field

from .vjsync_codec import (
    VJSyncPayload,
    STRIP_HEIGHT,
    encode_bpm,
    encode_beat_frac,
    stamp_barcode,
)

try:
    from scope.core.pipeline import Pipeline
    from scope.core.types import BasePipelineConfig, UsageType
    from scope.core.ui import ui_field_config
    _HAS_SCOPE = True
except ImportError:
    class Pipeline:
        pass
    class BasePipelineConfig:
        pass
    class UsageType:
        PREPROCESSOR = "preprocessor"
    def ui_field_config(**kwargs):
        return kwargs
    _HAS_SCOPE = False

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


# --- Config ---

if _HAS_SCOPE:
    class BpmBufferConfig(BasePipelineConfig):
        """Configuration schema for BPM Timecoded Buffer."""

        pipeline_id: str = "bpm_timecoded_buffer"
        pipeline_name: str = "BPM Timecoded Buffer (VJ.Tools)"
        pipeline_description: str = (
            "Beat-grid timecoding via Ableton Link. Stamps barcode on input, "
            "preserves through AI via VACE masking. Client decodes surviving "
            "barcode on output for beat-accurate timing. "
            "Canny/Depth/Scribble control modes."
        )
        supports_prompts: bool = False

        # --- Load-time parameters ---

        initial_bpm: float = Field(
            default=120.0,
            ge=60.0,
            le=300.0,
            json_schema_extra=ui_field_config(
                order=0,
                label="Initial BPM",
                description="Starting BPM for the Ableton Link session",
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
                description="Height of the timecode barcode strip at the bottom of the frame",
                is_load_param=False,
            ),
        )

        control_mode: str = Field(
            default="none",
            json_schema_extra=ui_field_config(
                order=2,
                label="Control Mode",
                description="ControlNet preprocessing: none, canny, depth, or scribble",
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
                description="Soft transition at barcode boundary (0 = hard edge)",
                is_load_param=False,
            ),
        )

        strip_barcode: bool = Field(
            default=False,
            json_schema_extra=ui_field_config(
                order=6,
                label="Strip Barcode on Output",
                description="Replace barcode strip with black on the display output. "
                            "Leave OFF to verify barcode survives AI generation.",
                is_load_param=False,
            ),
        )

        usage: list = [UsageType.PREPROCESSOR]
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

    def __init__(self, config: BpmBufferConfig, **kwargs):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self._frame_seq = 0

        self._clock = LinkClock(config.initial_bpm)
        self._clock.start(config.initial_bpm)

        logger.info(
            f"[BPM Buffer] Pipeline initialized "
            f"(device={self.device}, bpm={config.initial_bpm})"
        )

    def __del__(self):
        if hasattr(self, "_clock"):
            self._clock.stop()

    @staticmethod
    def get_config_class():
        return BpmBufferConfig

    def __call__(self, video: list, **kwargs) -> dict:
        """
        Process input frames:
        1. Stamp timecode barcode with current Link beat position
        2. Create VACE mask to preserve barcode through AI
        3. Optionally apply ControlNet preprocessing

        Args:
            video: List of frame tensors, each (1, H, W, C), range [0, 255]
            **kwargs: Runtime parameters

        Returns:
            dict with video, vace_input_frames, vace_input_masks
        """
        barcode_h = kwargs.get("barcode_height", self.config.barcode_height)
        control_mode = kwargs.get("control_mode", self.config.control_mode)
        canny_low = kwargs.get("canny_low", self.config.canny_low)
        canny_high = kwargs.get("canny_high", self.config.canny_high)
        mask_feather = kwargs.get("mask_feather", self.config.mask_feather)
        strip_barcode = kwargs.get("strip_barcode", self.config.strip_barcode)

        if not video:
            return {"video": torch.zeros(1, 1, 1, 3)}

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
        # Scope's VaceEncodingBlock handles the split internally:
        #   inactive stream = frame * (1 - mask)  -> preserves barcode
        #   reactive stream = frame * mask         -> AI generates content
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
        # Pass FULL stamped frames -- Scope handles masking internally
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
        display = frames_01.clone()
        if strip_barcode:
            # Clean output: replace barcode strip with black
            display[:, -barcode_h:, :, :] = 0.0
        else:
            # Show barcode with subtle green tint so you can verify it survives
            mask_cpu = mask.unsqueeze(-1)  # (F, H, W, 1)
            preserve = 1.0 - mask_cpu
            display = (display + preserve * torch.tensor([0.0, 0.05, 0.0])).clamp(0.0, 1.0)

        result = {
            "video": display.cpu(),
            "vace_input_frames": vace_frames,
            "vace_input_masks": vace_mask,
        }

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
