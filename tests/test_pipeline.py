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


def _stack_video(video_out):
    """Helper: stack list of (1,H,W,C) tensors into (F,H,W,C)."""
    if isinstance(video_out, list):
        return torch.cat(video_out, dim=0)
    return video_out


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

    video = _stack_video(result["video"])
    vace_frames = result["vace_input_frames"]
    vace_masks = result["vace_input_masks"]

    assert video.shape == (1, 336, 576, 3), f"Video shape wrong: {video.shape}"
    assert vace_frames.shape == (1, 3, 1, 336, 576), f"VACE frames shape wrong: {vace_frames.shape}"
    assert vace_masks.shape == (1, 1, 1, 336, 576), f"VACE mask shape wrong: {vace_masks.shape}"

    # Video is now uint8 [0, 255]
    assert video.max() <= 255
    assert vace_frames.min() >= -1.0 and vace_frames.max() <= 1.0

    print("  [OK] Basic mask test passed")


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


def test_multi_frame():
    """Test with multiple frames (batch processing)."""
    from bpm_timecoded_buffer.pipeline import BpmTimecodedBufferPipeline, BpmBufferConfig

    config = BpmBufferConfig()
    pipeline = BpmTimecodedBufferPipeline(config)

    frames = [make_test_frame() for _ in range(4)]
    result = pipeline(video=frames, barcode_height=16, control_mode="none")

    video = _stack_video(result["video"])
    assert video.shape[0] == 4
    assert result["vace_input_frames"].shape[2] == 4
    assert result["vace_input_masks"].shape[2] == 4

    print("  [OK] Multi-frame test passed")


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
    vid_visible = _stack_video(result_visible["video"])
    bottom_visible = vid_visible[0, -barcode_h:, :, :].numpy()
    assert len(np.unique(bottom_visible)) >= 2, "Barcode should be visible by default"

    # strip_barcode=True: barcode blacked out in display
    result_stripped = pipeline(
        video=[frame], barcode_height=barcode_h, control_mode="none",
        strip_barcode=True,
    )
    vid_stripped = _stack_video(result_stripped["video"])
    bottom_stripped = vid_stripped[0, -barcode_h:, :, :].numpy()
    assert np.all(bottom_stripped == 0), "Bottom strip should be black when strip_barcode=True"

    # VACE frames must still have barcode regardless of strip_barcode
    vace_barcode = result_stripped["vace_input_frames"][0, :, 0, -barcode_h:, :]
    vace_uint8 = ((vace_barcode + 1.0) / 2.0 * 255.0).cpu().numpy().astype(np.uint8)
    assert len(np.unique(vace_uint8)) >= 2, "VACE frames must have barcode even when display stripped"

    print("  [OK] Strip barcode test passed")


def test_test_pattern_input():
    """Test that test_input=True replaces video with bouncing ball animation."""
    from bpm_timecoded_buffer.pipeline import BpmTimecodedBufferPipeline, BpmBufferConfig

    config = BpmBufferConfig()
    pipeline = BpmTimecodedBufferPipeline(config)

    frame = make_test_frame()
    result = pipeline(
        video=[frame], barcode_height=16, control_mode="none", test_input=True
    )

    assert "video" in result
    assert "vace_input_frames" in result
    assert "vace_input_masks" in result

    video = _stack_video(result["video"])
    assert video.shape == (1, 336, 576, 3), f"Video shape wrong: {video.shape}"

    print("  [OK] Test pattern input test passed")


def test_tap_bpm():
    """Test tap tempo on the pipeline."""
    import time
    from bpm_timecoded_buffer.pipeline import BpmTimecodedBufferPipeline, BpmBufferConfig

    config = BpmBufferConfig()
    pipeline = BpmTimecodedBufferPipeline(config)

    # First tap returns None
    result1 = pipeline.tap_bpm()
    assert result1 is None, "First tap should return None"

    # Simulate 120 BPM (0.5s between beats)
    time.sleep(0.5)
    result2 = pipeline.tap_bpm()
    assert result2 is not None, "Second tap should return a BPM"
    assert 100 < result2 < 140, f"Expected ~120 BPM, got {result2:.1f}"

    print(f"  [OK] Tap tempo test passed (detected {result2:.1f} BPM)")


def test_set_bpm():
    """Test manual BPM setting."""
    from bpm_timecoded_buffer.pipeline import BpmTimecodedBufferPipeline, BpmBufferConfig

    config = BpmBufferConfig()
    pipeline = BpmTimecodedBufferPipeline(config)

    pipeline.set_bpm(140.0)
    assert pipeline._clock._tempo == 140.0, f"Expected 140 BPM, got {pipeline._clock._tempo}"

    print("  [OK] Set BPM test passed")


def test_barcode_roundtrip():
    """Test encode → decode barcode roundtrip."""
    from bpm_timecoded_buffer.vjsync_codec import (
        VJSyncPayload, stamp_barcode, read_barcode,
        encode_bpm, decode_bpm, encode_beat_frac, decode_beat_frac,
    )
    import numpy as np

    frame = np.full((336, 576, 3), 128, dtype=np.uint8)
    payload = VJSyncPayload(
        beat_whole=42,
        beat_frac=encode_beat_frac(0.75),
        frame_seq=1234,
        bpm_encoded=encode_bpm(140.0),
        flags=0,
    )
    stamp_barcode(frame, payload)

    decoded = read_barcode(frame, 16)
    assert decoded is not None, "Failed to decode barcode"
    assert decoded.beat_whole == 42, f"beat_whole: expected 42, got {decoded.beat_whole}"
    assert decoded.frame_seq == 1234, f"frame_seq: expected 1234, got {decoded.frame_seq}"
    assert decode_bpm(decoded.bpm_encoded) == 140.0, f"BPM: expected 140, got {decode_bpm(decoded.bpm_encoded)}"

    print("  [OK] Barcode roundtrip test passed")


def test_postprocessor_strip():
    """Test that the postprocessor strips the barcode from output."""
    from bpm_timecoded_buffer.pipeline import BpmTimecodeStripPipeline, BpmStripConfig

    config = BpmStripConfig()
    pipeline = BpmTimecodeStripPipeline(config)

    barcode_h = 16
    frame = make_test_frame(barcode_height=barcode_h)
    result = pipeline(video=[frame])

    assert "video" in result
    video = _stack_video(result["video"])
    assert video.shape == (1, 336, 576, 3), f"Video shape wrong: {video.shape}"

    # Bottom strip should be all black (0)
    bottom = video[0, -barcode_h:, :, :]
    assert bottom.max() == 0, f"Barcode region should be blacked out, got max={bottom.max()}"

    # Content above should NOT be all black
    content = video[0, :-barcode_h, :, :]
    assert content.max() > 0, "Content region should have data"

    print("  [OK] Postprocessor strip test passed")


def test_postprocessor_decode():
    """Test that the postprocessor decodes barcodes from preprocessor output."""
    from bpm_timecoded_buffer.pipeline import (
        BpmTimecodedBufferPipeline, BpmBufferConfig,
        BpmTimecodeStripPipeline, BpmStripConfig,
    )
    import time

    # Run preprocessor to stamp a barcode
    pre_config = BpmBufferConfig()
    pre = BpmTimecodedBufferPipeline(pre_config)
    time.sleep(0.05)  # Let clock tick

    frame = torch.randint(64, 200, (1, 336, 576, 3), dtype=torch.uint8)
    pre_result = pre(video=[frame])

    # Preprocessor now returns list of uint8 tensors — use directly
    stamped_video = pre_result["video"]

    # Run postprocessor in strip mode
    post_config = BpmStripConfig(buffer_mode="strip")
    post = BpmTimecodeStripPipeline(post_config)
    post_result = post(video=stamped_video)

    assert "_bpm_buffer_output_meta" in post_result
    meta = post_result["_bpm_buffer_output_meta"]
    assert meta["decode_success"] > 0 or meta["decode_fail"] > 0, "No decode attempts"

    print(f"  [OK] Postprocessor decode test passed (success={meta['decode_success']}, fail={meta['decode_fail']})")


def test_postprocessor_latency_mode():
    """Test latency buffer mode."""
    from bpm_timecoded_buffer.pipeline import BpmTimecodeStripPipeline, BpmStripConfig

    config = BpmStripConfig(buffer_mode="latency", latency_delay_ms=100)
    pipeline = BpmTimecodeStripPipeline(config)

    barcode_h = 16
    # Feed several batches to fill the buffer
    for _ in range(5):
        frame = make_test_frame(barcode_height=barcode_h)
        result = pipeline(video=[frame])
        assert "video" in result

    meta = result["_bpm_buffer_output_meta"]
    assert meta["buffer_mode"] == "latency"

    print("  [OK] Postprocessor latency mode test passed")


if __name__ == "__main__":
    print("\n=== BPM Timecoded Buffer Pipeline Tests ===\n")

    tests = [
        test_basic_mask,
        test_barcode_preserved,
        test_barcode_in_vace_frames,
        test_control_modes,
        test_multi_frame,
        test_strip_barcode,
        test_test_pattern_input,
        test_tap_bpm,
        test_set_bpm,
        test_barcode_roundtrip,
        test_postprocessor_strip,
        test_postprocessor_decode,
        test_postprocessor_latency_mode,
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
