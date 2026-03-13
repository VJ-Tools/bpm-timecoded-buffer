"""
Test the BPM Timecoded Buffer pipeline outside of Scope.

Run: python -m pytest tests/test_pipeline.py -v
  or: python tests/test_pipeline.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np


def make_test_frame(width=576, height=336, barcode_height=16):
    """Create a test frame with a synthetic barcode at the bottom."""
    frame = torch.randint(64, 200, (1, height, width, 3), dtype=torch.uint8)
    bar_width = 6
    for x in range(width):
        bar_idx = x // bar_width
        level = 235 if (bar_idx % 2 == 0) else 16
        frame[0, -barcode_height:, x, :] = level
    return frame


def test_basic_mask():
    """Test that the pipeline generates correct VACE mask dimensions."""
    from bpm_timecoded_buffer.pipeline import BpmTimecodedBufferPipeline, BpmBufferConfig

    config = BpmBufferConfig()
    pipeline = BpmTimecodedBufferPipeline(config)

    frame = make_test_frame()
    result = pipeline(video=[frame], barcode_height=16, control_mode="none")

    assert "video" in result
    assert "vace_input_frames" in result
    assert "vace_input_masks" in result

    video = result["video"]
    vace_frames = result["vace_input_frames"]
    vace_masks = result["vace_input_masks"]

    assert video.shape == (1, 336, 576, 3), f"Video shape wrong: {video.shape}"
    assert vace_frames.shape == (1, 3, 1, 336, 576), f"VACE frames shape wrong: {vace_frames.shape}"
    assert vace_masks.shape == (1, 1, 1, 336, 576), f"VACE mask shape wrong: {vace_masks.shape}"

    assert video.min() >= 0.0 and video.max() <= 1.0
    assert vace_frames.min() >= -1.0 and vace_frames.max() <= 1.0

    print("  [OK] Basic mask test passed")
    return True


def test_barcode_preserved():
    """Test that the barcode region has mask=0 (preserve)."""
    from bpm_timecoded_buffer.pipeline import BpmTimecodedBufferPipeline, BpmBufferConfig

    config = BpmBufferConfig()
    pipeline = BpmTimecodedBufferPipeline(config)

    barcode_h = 16
    frame = make_test_frame(barcode_height=barcode_h)
    result = pipeline(video=[frame], barcode_height=barcode_h, control_mode="none")

    vace_masks = result["vace_input_masks"]
    barcode_mask = vace_masks[0, 0, 0, -barcode_h:, :]
    assert barcode_mask.max() == 0.0, f"Barcode region should be all 0, got max={barcode_mask.max()}"

    content_mask = vace_masks[0, 0, 0, :-barcode_h, :]
    assert content_mask[0, 0] == 1.0, f"Content region should start at 1.0, got {content_mask[0, 0]}"

    print("  [OK] Barcode preservation test passed")
    return True


def test_barcode_in_vace_frames():
    """Test that vace_input_frames contain actual barcode data (not pre-masked)."""
    from bpm_timecoded_buffer.pipeline import BpmTimecodedBufferPipeline, BpmBufferConfig

    config = BpmBufferConfig()
    pipeline = BpmTimecodedBufferPipeline(config)

    barcode_h = 16
    frame = make_test_frame(barcode_height=barcode_h)
    result = pipeline(video=[frame], barcode_height=barcode_h, control_mode="none")

    vace_frames = result["vace_input_frames"]
    barcode_region = vace_frames[0, :, 0, -barcode_h:, :]
    barcode_uint8 = ((barcode_region + 1.0) / 2.0 * 255.0).cpu().numpy().astype(np.uint8)
    unique_vals = np.unique(barcode_uint8)
    assert len(unique_vals) >= 2, f"VACE frames should have barcode data, got: {unique_vals}"

    print("  [OK] Barcode in VACE frames test passed")
    return True


def test_control_modes():
    """Test all control modes produce valid output."""
    from bpm_timecoded_buffer.pipeline import BpmTimecodedBufferPipeline, BpmBufferConfig

    config = BpmBufferConfig()
    pipeline = BpmTimecodedBufferPipeline(config)

    for mode in ["none", "canny", "depth", "scribble"]:
        frame = make_test_frame()
        result = pipeline(video=[frame], barcode_height=16, control_mode=mode)

        assert "video" in result
        assert "vace_input_masks" in result

        if mode != "none":
            assert "control_video" in result, f"Mode '{mode}' should produce control_video"
            cv = result["control_video"]
            assert cv.shape[0] == 1 and cv.shape[3] == 3

        print(f"  [OK] Control mode '{mode}' passed")

    return True


def test_multi_frame():
    """Test with multiple frames (batch processing)."""
    from bpm_timecoded_buffer.pipeline import BpmTimecodedBufferPipeline, BpmBufferConfig

    config = BpmBufferConfig()
    pipeline = BpmTimecodedBufferPipeline(config)

    frames = [make_test_frame() for _ in range(4)]
    result = pipeline(video=frames, barcode_height=16, control_mode="none")

    assert result["video"].shape[0] == 4
    assert result["vace_input_frames"].shape[2] == 4
    assert result["vace_input_masks"].shape[2] == 4

    print("  [OK] Multi-frame test passed")
    return True


def test_strip_barcode():
    """Test strip_barcode option removes barcode from display output."""
    from bpm_timecoded_buffer.pipeline import BpmTimecodedBufferPipeline, BpmBufferConfig

    config = BpmBufferConfig()
    pipeline = BpmTimecodedBufferPipeline(config)

    barcode_h = 16
    frame = make_test_frame(barcode_height=barcode_h)

    # Default: barcode visible
    result_visible = pipeline(
        video=[frame], barcode_height=barcode_h, control_mode="none",
        strip_barcode=False,
    )
    bottom_visible = (result_visible["video"][0, -barcode_h:, :, :] * 255).numpy().astype(np.uint8)
    assert len(np.unique(bottom_visible)) >= 2, "Barcode should be visible by default"

    # strip_barcode=True: barcode blacked out in display
    result_stripped = pipeline(
        video=[frame], barcode_height=barcode_h, control_mode="none",
        strip_barcode=True,
    )
    bottom_stripped = (result_stripped["video"][0, -barcode_h:, :, :] * 255).numpy().astype(np.uint8)
    assert np.all(bottom_stripped == 0), "Bottom strip should be black when strip_barcode=True"

    # VACE frames must still have barcode regardless of strip_barcode
    vace_barcode = result_stripped["vace_input_frames"][0, :, 0, -barcode_h:, :]
    vace_uint8 = ((vace_barcode + 1.0) / 2.0 * 255.0).cpu().numpy().astype(np.uint8)
    assert len(np.unique(vace_uint8)) >= 2, "VACE frames must have barcode even when display stripped"

    print("  [OK] Strip barcode test passed")
    return True


if __name__ == "__main__":
    print("\n=== BPM Timecoded Buffer Pipeline Tests ===\n")

    tests = [
        test_basic_mask,
        test_barcode_preserved,
        test_barcode_in_vace_frames,
        test_control_modes,
        test_multi_frame,
        test_strip_barcode,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test.__name__} FAILED: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed!")
    sys.exit(failed)
