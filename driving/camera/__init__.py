"""Camera source adapters."""

from .video import OpenCvCamera, VideoFileCamera

__all__ = ["OpenCvCamera", "VideoFileCamera"]
