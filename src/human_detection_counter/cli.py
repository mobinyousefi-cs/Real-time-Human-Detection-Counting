#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=========================================================================================================
Project: Real-time Human Detection & Counting
File: cli.py
Author: Mobin Yousefi (GitHub: https://github.com/mobinyousefi-cs)
Created: 2025-11-11
Updated: 2025-11-11
License: MIT License (see LICENSE file for details)
=========================================================================================================

Description:
Command-line interface for the real-time human detection & counting system.

Usage:
    python -m human_detection_counter.cli --source webcam
    python -m human_detection_counter.cli --source path/to/video.mp4 --output annotated.mp4

Or via entry point:
    human-detection-counter --source webcam

Notes:
- Uses argparse for CLI argument parsing.
- Supports webcam, video file, and single image detection.
=========================================================================================================
"""

from __future__ import annotations

import argparse
import time
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import AppConfig, HogConfig
from .detector import HumanDetector
from .video_io import create_video_writer, open_capture, resize_frame
from .visualization import draw_detections, overlay_info


def parse_args(argv: Optional[Sequence[str]] = None) -> AppConfig:
    parser = argparse.ArgumentParser(
        description="Real-time Human Detection & Counting with HOG + SVM (OpenCV)"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="webcam",
        help='Input source: "webcam", webcam index (e.g., "0"), video file path, or image path.',
    )
    parser.add_argument(
        "--image",
        action="store_true",
        help="Treat the source as a single image instead of a video stream.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=640,
        help="Resize width for frames/images (preserves aspect ratio). Use 0 to disable.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save annotated output video. Ignored in image mode.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable window display. Useful in headless environments.",
    )
    parser.add_argument(
        "--hit-threshold",
        type=float,
        default=0.0,
        help="HOG SVM hit threshold. Higher -> fewer detections, lower -> more (with more false positives).",
    )
    parser.add_argument(
        "--win-stride",
        nargs=2,
        type=int,
        default=[8, 8],
        metavar=("W", "H"),
        help="HOG detection window stride, e.g., --win-stride 8 8",
    )
    parser.add_argument(
        "--padding",
        nargs=2,
        type=int,
        default=[8, 8],
        metavar=("W", "H"),
        help="HOG padding, e.g., --padding 8 8",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.05,
        help="Image pyramid scale factor. Typical values are between 1.01 and 1.1.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (for testing).",
    )

    args = parser.parse_args(argv)

    hog_cfg = HogConfig(
        win_stride=(args.win_stride[0], args.win_stride[1]),
        padding=(args.padding[0], args.padding[1]),
        scale=args.scale,
        hit_threshold=args.hit_threshold,
        use_meanshift_grouping=False,
    )

    resize_width = None if args.resize_width == 0 else args.resize_width

    app_cfg = AppConfig(
        source=args.source,
        is_image=args.image,
        resize_width=resize_width,
        output_path=args.output,
        display=not args.no_display,
        max_frames=args.max_frames,
        hog=hog_cfg,
    )

    return app_cfg


def _process_image(config: AppConfig, detector: HumanDetector) -> int:
    """Process a single image and return the number of detected humans."""
    image = cv2.imread(config.source)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {config.source}")

    if config.resize_width is not None:
        image = resize_frame(image, config.resize_width)

    boxes, _ = detector.detect(image)
    draw_detections(image, boxes)
    overlay_info(image, count=len(boxes), fps=None)

    if config.display:
        cv2.imshow("Human Detection (Image)", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return len(boxes)


def _process_video(config: AppConfig, detector: HumanDetector) -> None:
    """Process webcam or video stream."""
    cap = open_capture(config.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {config.source}")

    writer = None
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0

    # Fallback FPS if not available
    if fps <= 1e-3:
        fps = 25.0

    frame_count = 0
    last_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1

            if config.resize_width is not None:
                frame = resize_frame(frame, config.resize_width)

            boxes, _ = detector.detect(frame)
            count = len(boxes)
            draw_detections(frame, boxes)

            # Compute simple FPS estimate
            now = time.time()
            elapsed = now - last_time
            current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            last_time = now

            overlay_info(frame, count=count, fps=current_fps)

            # Lazy-init writer when we know frame size
            if config.output_path and writer is None:
                h, w = frame.shape[:2]
                writer = create_video_writer(
                    output_path=config.output_path,
                    fps=fps,
                    frame_size=(w, h),
                    is_color=True,
                )

            if writer is not None:
                writer.write(frame)

            if config.display:
                cv2.imshow("Human Detection (Video)", frame)
                # Press 'q' to exit early
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if config.max_frames is not None and frame_count >= config.max_frames:
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if config.display:
            cv2.destroyAllWindows()


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for the CLI."""
    config = parse_args(argv)
    detector = HumanDetector(config.hog)

    if config.is_image:
        num = _process_image(config, detector)
        print(f"Detected {num} humans in image: {config.source}")
    else:
        _process_video(config, detector)
