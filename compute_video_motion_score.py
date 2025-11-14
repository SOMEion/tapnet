#!/usr/bin/env python3
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Compute motion score from video without manual point selection.

Usage:
    python compute_video_motion_score.py <video_path> <checkpoint_path>

Example:
    python compute_video_motion_score.py video.mp4 checkpoints/tapir_checkpoint.npy
"""

import sys
import jax
import numpy as np
import mediapy as media
from tapnet.models import tapir_model
from tapnet.utils import model_utils
from tapnet.utils import motion_scoring


def compute_video_motion_score(video_path, checkpoint_path, stride=8):
    """Compute motion score from video.

    Args:
        video_path: Path to input video file
        checkpoint_path: Path to TAPIR checkpoint (.npy file)
        stride: Spacing between grid points in pixels (default 8)
                stride=8 gives ~32x32 grid for 256x256 video

    Returns:
        Single scalar motion score (higher = more motion)
    """
    # Load video
    video = media.read_video(video_path)
    num_frames, height, width = video.shape[:3]

    # Load TAPIR model
    ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
    params, state = ckpt_state['params'], ckpt_state['state']
    tapir = tapir_model.ParameterizedTAPIR(params, state)

    # Sample grid of query points at frame 0
    query_points = motion_scoring.sample_grid_points(
        frame_idx=0, height=height, width=width, stride=stride
    )

    # Preprocess video for model
    frames = model_utils.preprocess_frames(video)
    frames = frames[None]  # Add batch dimension
    query_points = query_points[None].astype(np.float32)  # Add batch dimension

    # Run TAPIR inference
    outputs = tapir(
        video=frames,
        query_points=query_points,
        is_training=False,
    )

    # Extract predictions
    tracks = outputs['tracks'][0]  # Remove batch dimension
    occlusions = outputs['occlusion'][0]
    expected_dist = outputs['expected_dist'][0]

    # Combine occlusion and uncertainty into binary occlusion mask
    occluded = model_utils.postprocess_occlusions(occlusions, expected_dist)

    # Compute motion score
    score = motion_scoring.compute_motion_score(tracks, occluded)

    return score


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(1)

    video_path = sys.argv[1]
    checkpoint_path = sys.argv[2]

    score = compute_video_motion_score(video_path, checkpoint_path)
