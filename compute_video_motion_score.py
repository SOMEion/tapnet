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


def compute_video_motion_score(video_path, checkpoint_path, stride=8, resize_height=256, resize_width=256, query_chunk_size=64):
    """Compute motion score from video.

    Args:
        video_path: Path to input video file
        checkpoint_path: Path to TAPIR checkpoint (.npy file)
        stride: Spacing between grid points in pixels (default 8)
        resize_height: Resize video height (default 256, prevents memory overflow)
        resize_width: Resize video width (default 256, prevents memory overflow)
        query_chunk_size: Process points in chunks to save memory (default 64)

    Returns:
        Single scalar motion score (higher = more motion)
    """
    # Load video
    video = media.read_video(video_path)

    # Resize video to prevent memory overflow
    video = media.resize_video(video, (resize_height, resize_width))
    num_frames, height, width = video.shape[:3]

    # Load TAPIR model
    ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
    params, state = ckpt_state['params'], ckpt_state['state']
    tapir_kwargs = dict(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
    tapir = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=tapir_kwargs)

    # Preprocess video for model
    frames = model_utils.preprocess_frames(video)
    frames = frames[None]  # Add batch dimension

    # Pre-compute feature grids once for efficiency
    feature_grids = tapir.get_feature_grids(frames, is_training=False)

    # Sample grid of query points at frame 0 (after resize)
    query_points = motion_scoring.sample_grid_points(
        frame_idx=0, height=height, width=width, stride=stride
    )

    # Process in chunks to save memory
    num_points = query_points.shape[0]
    all_tracks = []
    all_occlusions = []
    all_expected_dist = []

    for i in range(0, num_points, query_chunk_size):
        query_chunk = query_points[i:i + query_chunk_size]
        query_chunk = query_chunk[None].astype(np.float32)  # Add batch dimension

        # Run TAPIR inference on chunk
        outputs = tapir(
            video=frames,
            query_points=query_chunk,
            is_training=False,
            query_chunk_size=query_chunk_size,
            feature_grids=feature_grids,
        )

        all_tracks.append(outputs['tracks'][0])
        all_occlusions.append(outputs['occlusion'][0])
        all_expected_dist.append(outputs['expected_dist'][0])

    # Concatenate all chunks
    tracks = np.concatenate(all_tracks, axis=0)
    occlusions = np.concatenate(all_occlusions, axis=0)
    expected_dist = np.concatenate(all_expected_dist, axis=0)

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
