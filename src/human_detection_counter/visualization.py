#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Real-time Human Detection & Counting
File: visualization.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-11
Updated: 2025-11-11
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Visualization utilities for drawing detections and overlaying metadata on frames.

Usage:
    from human_detection_counter.visualization import draw_detections, overlay_info

Notes:
- These functions modify the input frame in-place and also return it for convenience.
=========================================================================================================
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2

from .detector import BoundingBox


def draw_detections(
    frame,
    boxes: List[BoundingBox],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw bounding boxes for each detection."""
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)


def overlay_info(
    frame,
    count: int,
    fps: Optional[float] = None,
    font_scale: float = 0.7,
    thickness: int = 2,
) -> None:
    """Overlay detection count and FPS on the frame."""
    text = f"Humans: {count}"
    if fps is not None:
        text += f" | FPS: {fps:.1f}"

    cv2.putText(
        frame,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 255),
        thickness,
        cv2.LINE_AA,
    )
