#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Real-time Human Detection & Counting
File: test_imports.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-11
Updated: 2025-11-11
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Basic smoke tests to ensure that the package imports correctly and the main components can be instantiated.

Usage:
    pytest tests/test_imports.py

Notes:
- These tests intentionally avoid using the webcam or actual files.
=========================================================================================================
"""

from __future__ import annotations

import numpy as np

from human_detection_counter.config import HogConfig
from human_detection_counter.detector import HumanDetector


def test_detector_init() -> None:
    cfg = HogConfig()
    detector = HumanDetector(cfg)
    assert detector is not None


def test_detector_empty_frame() -> None:
    cfg = HogConfig()
    detector = HumanDetector(cfg)
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    boxes, weights = detector.detect(empty)
    assert boxes == []
    assert weights == []
