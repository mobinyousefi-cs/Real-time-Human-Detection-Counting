#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Real-time Human Detection & Counting
File: run_webcam.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-11
Updated: 2025-11-11
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Convenience script for launching the human detection & counting system using the default webcam.

Usage:
    python scripts/run_webcam.py

Notes:
- This script simply delegates to the CLI main() function with the default arguments.
=========================================================================================================
"""

from __future__ import annotations

from human_detection_counter.cli import main


if __name__ == "__main__":
    main(["--source", "webcam"])
