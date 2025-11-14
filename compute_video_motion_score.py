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
    python compute_video_motion_score.py <video_path> <checkpoint_path> [model_type] [--visualize]

Arguments:
    video_path: Path to input video file
    checkpoint_path: Path to model checkpoint (.npy file)
    model_type: 'tapir' or 'bootstapir' (optional, default: 'tapir')
    --visualize: Save visualization video instead of computing motion score

Examples:
    # Compute motion score with TAPIR
    python compute_video_motion_score.py video.mp4 checkpoints/tapir_checkpoint_panning.npy

    # Compute motion score with BootsTAPIR
    python compute_video_motion_score.py video.mp4 checkpoints/bootstapir_checkpoint_v2.npy bootstapir

    # Generate visualization video
    python compute_video_motion_score.py video.mp4 checkpoints/bootstapir_checkpoint_v2.npy bootstapir --visualize
"""

import sys
import jax
import numpy as np
import mediapy as media
from tapnet.models import tapir_model
from tapnet.utils import model_utils
from tapnet.utils import motion_scoring
from tapnet.utils import transforms
from tapnet.utils import viz_utils


def compute_video_motion_score(video_path, checkpoint_path, model_type='tapir',
                               stride=8, resize_height=256, resize_width=256,
                               chunk_size=32, visualize=False, output_path=None):
    """Compute motion score or generate visualization following notebook pattern EXACTLY.

    This implementation follows /home/user/tapnet/colabs/tapir_demo.ipynb cell 12
    with NO modifications to the chunking or inference pattern.

    Args:
        video_path: Path to input video file
        checkpoint_path: Path to TAPIR/BootsTAPIR checkpoint (.npy file)
        model_type: 'tapir' or 'bootstapir' (default 'tapir')
        stride: Spacing between grid points in pixels (default 8)
        resize_height: Resize video height (default 256)
        resize_width: Resize video width (default 256)
        chunk_size: Number of query points per chunk (default 32, matches notebook)
        visualize: If True, save visualization video; if False, compute motion score
        output_path: Path to save visualization video (only used if visualize=True)

    Returns:
        If visualize=False: Single scalar motion score (higher = more motion)
        If visualize=True: None (saves video to output_path)
    """
    # ========================================================================
    # PATTERN MATCH: tapir_demo.ipynb cell 12, lines 1-4
    # ========================================================================
    # ORIGINAL:
    #   resize_height = 256
    #   resize_width = 256
    #   frames = media.resize_video(video, (resize_height, resize_width))
    #   frames = model_utils.preprocess_frames(frames[None])

    # Load original video
    # Function: media.read_video()
    # Location: mediapy (external library)
    # Returns: [T, H, W, 3] uint8 [0, 255]
    video = media.read_video(video_path)
    orig_height, orig_width = video.shape[1:3]

    # Resize video (EXACTLY as notebook)
    # Function: media.resize_video()
    # Location: mediapy (external library)
    # Computes: Bilinear resize each frame
    frames = media.resize_video(video, (resize_height, resize_width))

    # Preprocess and add batch dimension (EXACTLY as notebook)
    # Function: model_utils.preprocess_frames()
    # Location: tapnet/utils/model_utils.py:362-373
    # Computes: (frames / 255) * 2 - 1 → converts [0,255] to [-1,1]
    # Then add batch dimension [T,H,W,3] → [1,T,H,W,3]
    frames = model_utils.preprocess_frames(frames[None])

    # ========================================================================
    # PATTERN MATCH: tapir_demo.ipynb cell 6
    # ========================================================================
    # ORIGINAL:
    #   ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
    #   params, state = ckpt_state['params'], ckpt_state['state']
    #   kwargs = dict(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
    #   if MODEL_TYPE == 'bootstapir':
    #       kwargs.update(dict(pyramid_level=1, extra_convs=True, softmax_temperature=10.0))
    #   tapir = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=kwargs)

    ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
    params, state = ckpt_state['params'], ckpt_state['state']

    kwargs = dict(bilinear_interp_with_depthwise_conv=False, pyramid_level=0)
    if model_type == 'bootstapir':
        kwargs.update(
            dict(pyramid_level=1, extra_convs=True, softmax_temperature=10.0)
        )

    # Class: tapir_model.ParameterizedTAPIR
    # Location: tapnet/models/tapir_model.py:1200-1262
    # Computes: Wraps TAPIR model with Haiku state management
    tapir = tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=kwargs)

    # ========================================================================
    # PATTERN MATCH: tapir_demo.ipynb cell 12, lines 5-6
    # ========================================================================
    # ORIGINAL:
    #   feature_grids = tapir.get_feature_grids(frames, is_training=False)
    #   query_points = sample_random_points(0, frames.shape[2], frames.shape[3], num_points)

    # Get feature grids (pre-compute once for all chunks)
    # Method: tapir.get_feature_grids()
    # Location: tapnet/models/tapir_model.py:619-724
    # Computes: ResNet features at multiple resolutions
    feature_grids = tapir.get_feature_grids(frames, is_training=False)

    # Sample grid points on first frame
    # Function: motion_scoring.sample_grid_points()
    # Location: tapnet/utils/motion_scoring.py:22-46
    # Computes: Creates evenly-spaced grid at stride intervals
    # NOTE: frames.shape is [1, T, H, W, 3], so frames.shape[2] = H, frames.shape[3] = W
    query_points = motion_scoring.sample_grid_points(
        frame_idx=0,
        height=frames.shape[2],  # Height of resized video
        width=frames.shape[3],   # Width of resized video
        stride=stride
    )

    # ========================================================================
    # PATTERN MATCH: tapir_demo.ipynb cell 12, lines 8-30
    # EXACT MATCH - NO MODIFICATIONS
    # ========================================================================
    # ORIGINAL:
    #   chunk_size = 32
    #   def chunk_inference(query_points):
    #       query_points = query_points.astype(np.float32)[None]
    #       outputs = tapir(video=frames, query_points=query_points, is_training=False,
    #                       query_chunk_size=chunk_size, feature_grids=feature_grids)
    #       tracks, occlusions, expected_dist = outputs["tracks"], outputs["occlusion"], outputs["expected_dist"]
    #       visibles = model_utils.postprocess_occlusions(occlusions, expected_dist)
    #       return tracks[0], visibles[0]
    #   chunk_inference = jax.jit(chunk_inference)

    # Define chunk_inference function (EXACTLY as notebook)
    def chunk_inference(query_points):
        """Process a chunk of query points through TAPIR.

        EXACT MATCH to tapir_demo.ipynb cell 12 lines 12-23.

        Args:
            query_points: [N, 3] query points in [t, y, x] format

        Returns:
            tracks: [N, T, 2] predicted positions in [x, y] format
            visibles: [N, T] boolean visibility flags
        """
        # Add batch dimension: [N, 3] → [1, N, 3]
        query_points = query_points.astype(np.float32)[None]

        # Method: tapir.__call__()
        # Location: tapnet/models/tapir_model.py:1061-1145
        # Computes:
        #   1. get_query_features(): Extract features for query points
        #   2. estimate_trajectories(): Predict tracks via cost volumes + PIPs refinement
        # Returns: dict with 'tracks', 'occlusion', 'expected_dist'
        outputs = tapir(
            video=frames,
            query_points=query_points,
            is_training=False,
            query_chunk_size=chunk_size,
            feature_grids=feature_grids,
        )
        tracks, occlusions, expected_dist = (
            outputs["tracks"],
            outputs["occlusion"],
            outputs["expected_dist"],
        )

        # Function: model_utils.postprocess_occlusions()
        # Location: tapnet/utils/model_utils.py:376-389
        # Computes: (1 - sigmoid(occ)) * (1 - sigmoid(dist)) > 0.5
        visibles = model_utils.postprocess_occlusions(occlusions, expected_dist)

        # Remove batch dimension: [1, N, T, 2] → [N, T, 2]
        return tracks[0], visibles[0]

    # JIT compile (EXACTLY as notebook)
    chunk_inference = jax.jit(chunk_inference)

    # ========================================================================
    # PATTERN MATCH: tapir_demo.ipynb cell 12, lines 32-39
    # EXACT MATCH - NO PADDING, NO MODIFICATIONS
    # ========================================================================
    # ORIGINAL:
    #   all_tracks = []
    #   all_visibles = []
    #   for chunk in range(0, query_points.shape[0], chunk_size):
    #       tracks, visibles = chunk_inference(query_points[chunk : chunk + chunk_size])
    #       all_tracks.append(np.array(tracks))
    #       all_visibles.append(np.array(visibles))
    #   tracks = np.concatenate(all_tracks, axis=0)
    #   visibles = np.concatenate(all_visibles, axis=0)

    # Process query points in chunks (EXACTLY as notebook - NO PADDING!)
    all_tracks = []
    all_visibles = []
    for chunk in range(0, query_points.shape[0], chunk_size):
        # Direct slice - notebook handles variable chunk sizes naturally
        tracks, visibles = chunk_inference(query_points[chunk : chunk + chunk_size])
        all_tracks.append(np.array(tracks))
        all_visibles.append(np.array(visibles))

    # Concatenate all chunks
    tracks = np.concatenate(all_tracks, axis=0)
    visibles = np.concatenate(all_visibles, axis=0)

    # ========================================================================
    # PATTERN MATCH: tapir_demo.ipynb cell 12, lines 41-46
    # ========================================================================
    # ORIGINAL:
    #   height, width = video.shape[1:3]
    #   tracks = transforms.convert_grid_coordinates(
    #       tracks, (resize_width, resize_height), (width, height)
    #   )
    #   video_viz = viz_utils.paint_point_track(video, tracks, visibles)

    # Convert tracks back to original video resolution
    # Function: transforms.convert_grid_coordinates()
    # Location: tapnet/utils/transforms.py:24-78
    # Computes: Scales coordinates from resized frame to original frame
    #   Formula: coords_out = coords_in * (output_size / input_size)
    tracks = transforms.convert_grid_coordinates(
        tracks,
        (resize_width, resize_height),  # Input: resized resolution
        (orig_width, orig_height)        # Output: original resolution
    )

    if visualize:
        # ====================================================================
        # OPTION 1: VISUALIZATION (matches notebook cell 12 lines 45-46)
        # ====================================================================
        # Function: viz_utils.paint_point_track()
        # Location: tapnet/utils/viz_utils.py:47+
        # Computes: Draws colored tracks on video frames
        video_viz = viz_utils.paint_point_track(video, tracks, visibles)

        # Save visualization video
        if output_path is None:
            output_path = video_path.rsplit('.', 1)[0] + '_tracked.mp4'
        media.write_video(output_path, video_viz, fps=10)
        return None
    else:
        # ====================================================================
        # OPTION 2: MOTION SCORE (new functionality)
        # ====================================================================
        # Convert visibility to occlusion for motion scoring
        occluded = 1.0 - visibles

        # Function: motion_scoring.compute_motion_score()
        # Location: tapnet/utils/motion_scoring.py:49-79
        # Computes: Mean displacement per visible point per frame
        score = motion_scoring.compute_motion_score(tracks, occluded)
        return score


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit(1)

    video_path = sys.argv[1]
    checkpoint_path = sys.argv[2]

    # Parse optional arguments
    model_type = 'tapir'
    visualize = False

    for arg in sys.argv[3:]:
        if arg == '--visualize':
            visualize = True
        elif arg in ['tapir', 'bootstapir']:
            model_type = arg

    result = compute_video_motion_score(
        video_path,
        checkpoint_path,
        model_type=model_type,
        visualize=visualize
    )

    if not visualize:
        print(f"Motion Score: {result}")
