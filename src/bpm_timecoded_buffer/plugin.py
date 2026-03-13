"""
BPM Timecoded Buffer — Scope Plugin Registration

Registers the BPM Timecoded Buffer preprocessor with Daydream Scope.
Stamps beat-grid timecodes using Ableton Link, preserves them through
AI generation via VACE masking.
"""

import logging

try:
    from scope.core.plugins.hookspecs import hookimpl
except ImportError:
    def hookimpl(f):
        return f

from .pipeline import BpmTimecodedBufferPipeline

logger = logging.getLogger(__name__)


@hookimpl
def register_pipelines(register):
    """Register the BPM Timecoded Buffer pipeline with Scope."""
    register(BpmTimecodedBufferPipeline)
    logger.info("[BPM Buffer] Registered BPM Timecoded Buffer preprocessor")
