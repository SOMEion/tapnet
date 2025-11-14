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


def compute_video_motion_score(video_path, checkpoint_path, stride=8, resize_height=256, resize_width=256, chunk_size=64):
    """Compute motion score from video using chunked inference pattern from notebook.

    Args:
        video_path: Path to input video file
        checkpoint_path: Path to TAPIR checkpoint (.npy file)
        stride: Spacing between grid points in pixels (default 8)
        resize_height: Resize video height (default 256, prevents memory overflow)
        resize_width: Resize video width (default 256, prevents memory overflow)
        chunk_size: Number of query points to process per chunk (default 64)

    Returns:
        Single scalar motion score (higher = more motion)
    """
    # ========================================================================
    # STEP 1: Load and resize video
    # ========================================================================
    # Function: media.read_video()
    # Location: mediapy library (external)
    # Returns: np.ndarray of shape [num_frames, height, width, 3], dtype uint8, range [0, 255]
    video = media.read_video(video_path)

    # Function: media.resize_video()
    # Location: mediapy library (external)
    # Computes: Resizes each frame to target resolution using bilinear interpolation
    # Returns: np.ndarray of shape [num_frames, resize_height, resize_width, 3]
    video = media.resize_video(video, (resize_height, resize_width))
    num_frames, height, width = video.shape[:3]

    # ========================================================================
    # STEP 2: Load TAPIR model
    # ========================================================================
    # Load checkpoint containing model parameters and state
    ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
    params, state = ckpt_state['params'], ckpt_state['state']

    # Model configuration kwargs
    tapir_kwargs = dict(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)

    # Class: tapir_model.ParameterizedTAPIR
    # Location: /home/user/tapnet/tapnet/models/tapir_model.py:1200-1262
    # Purpose: Wrapper that injects Haiku parameters/state into TAPIR model
    # Computes: Creates a stateful TAPIR model that can be called like a regular function
    tapir = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=tapir_kwargs)

    # ========================================================================
    # STEP 3: Preprocess video and extract feature grids
    # ========================================================================
    # Function: model_utils.preprocess_frames()
    # Location: /home/user/tapnet/tapnet/utils/model_utils.py:362-373
    # Computes: Converts frames from [0, 255] uint8 to [-1, 1] float32
    # Returns: np.ndarray of shape [num_frames, height, width, 3], dtype float32
    frames = model_utils.preprocess_frames(video)
    frames = frames[None]  # Add batch dimension → [1, num_frames, height, width, 3]

    # Method: tapir.get_feature_grids()
    # Location: /home/user/tapnet/tapnet/models/tapir_model.py:619-724
    # Computes: Extracts ResNet features from video at multiple resolutions
    #   - Runs ResNet on video frames to get convolutional features
    #   - Creates feature pyramids at different scales for multi-resolution tracking
    #   - Caches features to avoid recomputing for each query point
    # Returns: FeatureGrids object containing:
    #   - lowres: List of feature grids at different resolutions
    #   - hires: List of high-resolution features for refinement
    #   - resolutions: List of (height, width) tuples for each resolution
    feature_grids = tapir.get_feature_grids(frames, is_training=False)

    # ========================================================================
    # STEP 4: Sample grid of query points
    # ========================================================================
    # Function: motion_scoring.sample_grid_points()
    # Location: /home/user/tapnet/tapnet/utils/motion_scoring.py:22-46
    # Computes: Creates evenly-spaced grid of points on frame 0
    #   - Uses np.mgrid to create grid starting at stride//2 with spacing of stride
    #   - For 256x256 video with stride=8: creates 32x32 = 1,024 points
    # Returns: np.ndarray of shape [num_points, 3] with format [t, y, x]
    query_points = motion_scoring.sample_grid_points(
        frame_idx=0, height=height, width=width, stride=stride
    )

    # ========================================================================
    # STEP 5: Process query points in chunks (mimics notebook pattern)
    # ========================================================================
    num_points = query_points.shape[0]
    all_tracks = []
    all_visibles = []

    # Loop through query points in chunks of size chunk_size
    # Pattern from: /home/user/tapnet/colabs/tapir_rainbow_demo.ipynb cell 10
    for i in range(0, num_points, chunk_size):
        # Extract chunk of query points
        query_points_chunk = query_points[i:i + chunk_size]

        # Pad last chunk to chunk_size if needed (notebook pattern)
        num_extra = chunk_size - query_points_chunk.shape[0]
        if num_extra > 0:
            query_points_chunk = np.concatenate(
                [query_points_chunk, np.zeros([num_extra, 3])], axis=0
            )

        # Convert to float32 and add batch dimension
        query_points_chunk = query_points_chunk.astype(np.float32)[None]

        # Method: tapir.__call__()
        # Location: /home/user/tapnet/tapnet/models/tapir_model.py:1061-1145
        # Computes: Full TAPIR forward pass
        #   1. get_query_features(): Extract features for query points from video
        #   2. estimate_trajectories(): Predict tracks using cost volumes and refinement
        #      - Builds cost volumes by correlating query features with frame features
        #      - Uses PIPs (Persistent Independent Particles) iterative refinement
        #      - Predicts occlusion and uncertainty at each frame
        #   3. Returns averaged predictions across multiple refinement iterations
        # Parameters:
        #   - video: Preprocessed video frames [1, T, H, W, 3]
        #   - query_points: Points to track [1, chunk_size, 3]
        #   - is_training: False for inference
        #   - query_chunk_size: Further chunk queries inside the model (memory saving)
        #   - feature_grids: Pre-computed features (avoids recomputation)
        # Returns: Dict with keys:
        #   - tracks: [1, chunk_size, num_frames, 2] - predicted (x, y) positions
        #   - occlusion: [1, chunk_size, num_frames] - occlusion logits
        #   - expected_dist: [1, chunk_size, num_frames] - uncertainty logits
        outputs = tapir(
            video=frames,
            query_points=query_points_chunk,
            is_training=False,
            query_chunk_size=chunk_size,
            feature_grids=feature_grids,
        )

        # Extract outputs
        tracks = outputs['tracks'][0]  # Remove batch dimension
        occlusions = outputs['occlusion'][0]
        expected_dist = outputs['expected_dist'][0]

        # Function: model_utils.postprocess_occlusions()
        # Location: /home/user/tapnet/tapnet/utils/model_utils.py:376-389
        # Computes: Converts occlusion and uncertainty logits to binary visibility
        #   - Applies sigmoid to both occlusion and expected_dist logits
        #   - Computes: visible = (1 - sigmoid(occlusion)) * (1 - sigmoid(expected_dist)) > 0.5
        #   - Point is visible if both low occlusion AND low uncertainty
        # Returns: Boolean array of shape [num_points, num_frames]
        visibles = model_utils.postprocess_occlusions(occlusions, expected_dist)

        # Remove padding from last chunk (notebook pattern)
        if num_extra > 0:
            tracks = tracks[:-num_extra]
            visibles = visibles[:-num_extra]

        # Accumulate results
        all_tracks.append(tracks)
        all_visibles.append(visibles)

    # Concatenate all chunks
    tracks = np.concatenate(all_tracks, axis=0)
    visibles = np.concatenate(all_visibles, axis=0)

    # ========================================================================
    # STEP 6: Compute motion score
    # ========================================================================
    # Convert visibility to occlusion for motion scoring
    occluded = 1.0 - visibles

    # Function: motion_scoring.compute_motion_score()
    # Location: /home/user/tapnet/tapnet/utils/motion_scoring.py:49-79
    # Computes: Single scalar motion metric
    #   1. Calculates displacement between consecutive frames: diff(tracks)
    #   2. Computes L2 norm of displacement vectors
    #   3. Masks out occluded points (only counts visible displacements)
    #   4. Returns: mean displacement per visible point per frame
    # Returns: Float scalar (higher = more motion in video)
    score = motion_scoring.compute_motion_score(tracks, occluded)

    return score


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(1)

    video_path = sys.argv[1]
    checkpoint_path = sys.argv[2]

    score = compute_video_motion_score(video_path, checkpoint_path)
