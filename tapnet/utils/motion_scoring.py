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

"""Compute motion score from video without manual point selection."""

import jax.numpy as jnp
import numpy as np


def sample_grid_points(frame_idx, height, width, stride=8):
    """Sample grid points with (time, height, width) order.

    Args:
        frame_idx: Frame index to query from (typically 0)
        height: Video height in pixels
        width: Video width in pixels
        stride: Spacing between points in pixels (default 8)
                stride=8 gives ~32x32 grid for 256x256 video
                stride=1 gives dense per-pixel grid

    Returns:
        Query points array of shape [num_points, 3] with format [t, y, x]
    """
    # Create grid starting at stride//2 with spacing of stride
    points = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
    points = points.transpose(1, 2, 0)
    out_height, out_width = points.shape[0:2]

    # Add frame index as first coordinate
    frame_idx = np.ones((out_height, out_width, 1)) * frame_idx
    points = np.concatenate((frame_idx, points), axis=-1).astype(np.int32)
    points = points.reshape(-1, 3)  # [num_points, 3]

    return points


def compute_motion_score(tracks, occluded):
    """Compute single motion score from predicted tracks.

    Args:
        tracks: Predicted tracks of shape [num_points, num_frames, 2]
        occluded: Occlusion predictions of shape [num_points, num_frames]

    Returns:
        Single scalar motion score (higher = more motion)
    """
    # Compute displacement between consecutive frames
    displacements = jnp.diff(tracks, axis=1)  # [num_points, num_frames-1, 2]

    # Compute magnitude of displacement
    displacement_magnitude = jnp.linalg.norm(displacements, axis=-1)  # [num_points, num_frames-1]

    # Mask out occluded points (consider both current and next frame)
    visible_mask = (1 - occluded[:, :-1]) * (1 - occluded[:, 1:])  # [num_points, num_frames-1]

    # Compute average displacement for visible points only
    masked_displacements = displacement_magnitude * visible_mask
    total_displacement = jnp.sum(masked_displacements)
    total_visible = jnp.sum(visible_mask)

    # Average motion per visible point per frame
    motion_score = total_displacement / jnp.maximum(total_visible, 1.0)

    return float(motion_score)
