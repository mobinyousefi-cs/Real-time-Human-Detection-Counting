#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Real-time Human Detection & Counting
File: detector.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-11
Updated: 2025-11-11
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
HOG + SVM-based human detector built on top of OpenCV.

Usage:
    import cv2
    from human_detection_counter.config import HogConfig
    from human_detection_counter.detector import HumanDetector

    hog_cfg = HogConfig()
    detector = HumanDetector(hog_cfg)

    frame = cv2.imread("some_image.jpg")
    boxes, weights = detector.detect(frame)

Notes:
- Uses OpenCV's default people detector: HOGDescriptor_getDefaultPeopleDetector().
=========================================================================================================
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from .config import HogConfig


BoundingBox = Tuple[int, int, int, int]


class HumanDetector:
    """Wrapper around OpenCV's HOG + SVM people detector."""

    def __init__(self, config: HogConfig) -> None:
        self.config = config
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, frame: np.ndarray) -> Tuple[List[BoundingBox], List[float]]:
        """
        Detect humans in a given frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image frame.

        Returns
        -------
        boxes : list of (x, y, w, h)
            Detected bounding boxes.
        weights : list of float
            Confidence weights for each detection.
        """
        if frame is None or frame.size == 0:
            return [], []

        # Convert to grayscale can help, but HOG people detector works with color as well.
        # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        (rects, weights) = self._hog.detectMultiScale(
            frame,
            winStride=self.config.win_stride,
            padding=self.config.padding,
            scale=self.config.scale,
            hitThreshold=self.config.hit_threshold,
            useMeanshiftGrouping=self.config.use_meanshift_grouping,
        )

        boxes: List[BoundingBox] = []
        for (x, y, w, h) in rects:
            boxes.append((int(x), int(y), int(w), int(h)))

        return boxes, [float(w) for w in weights]
