#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Real-time Human Detection & Counting
File: video_io.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-11
Updated: 2025-11-11
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Abstractions for reading frames from webcam, video files, and images, and writing output video.

Usage:
    from human_detection_counter.video_io import open_capture, create_video_writer

Notes:
- Uses cv2.VideoCapture for both webcam and video files.
- Single-image mode is handled at CLI level.
=========================================================================================================
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import numpy as np


def _parse_source(source: str) -> Union[int, str]:
    """Parse source string into cv2.VideoCapture-compatible value."""
    if source.lower() == "webcam":
        return 0
    # Allow numeric strings to refer to webcam indices
    if source.isdigit():
        return int(source)
    # Otherwise treat as file path
    return source


def open_capture(source: str) -> cv2.VideoCapture:
    cap_source = _parse_source(source)
    cap = cv2.VideoCapture(cap_source)
    return cap


def create_video_writer(
    output_path: str,
    fps: float,
    frame_size: Tuple[int, int],
    is_color: bool = True,
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size, is_color)
    return writer


def resize_frame(frame: np.ndarray, width: Optional[int]) -> np.ndarray:
    """Resize frame to a given width while preserving aspect ratio."""
    if width is None:
        return frame
    h, w = frame.shape[:2]
    if w == 0:
        return frame
    scale = width / float(w)
    new_size = (int(width), int(h * scale))
    return cv2.resize(frame, new_size)
