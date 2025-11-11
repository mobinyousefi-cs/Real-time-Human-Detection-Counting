#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Real-time Human Detection & Counting
File: __init__.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-11
Updated: 2025-11-11
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Package initialization for the human_detection_counter project.

Usage:
Import the core components of the package:

    from human_detection_counter.detector import HumanDetector

Notes:
- This file exposes a minimal public API for external users.
=========================================================================================================
"""

from .detector import HumanDetector

__all__ = ["HumanDetector"]
