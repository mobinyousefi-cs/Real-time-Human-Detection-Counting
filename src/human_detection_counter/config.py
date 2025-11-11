#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Real-time Human Detection & Counting
File: config.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-11
Updated: 2025-11-11
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Configuration dataclasses and defaults for the human detection & counting system.

Usage:
    from human_detection_counter.config import AppConfig, HogConfig

Notes:
- Centralized configuration makes the code easier to maintain and extend.
=========================================================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class HogConfig:
    """Configuration for the HOG + SVM detector."""

    win_stride: Tuple[int, int] = (8, 8)
    padding: Tuple[int, int] = (8, 8)
    scale: float = 1.05
    hit_threshold: float = 0.0
    use_meanshift_grouping: bool = False


@dataclass
class AppConfig:
    """High-level application configuration."""

    source: str = "webcam"  # "webcam", integer index as string ("0"), or path
    is_image: bool = False
    resize_width: Optional[int] = 640
    output_path: Optional[str] = None
    display: bool = True
    max_frames: Optional[int] = None
    hog: HogConfig = HogConfig()
