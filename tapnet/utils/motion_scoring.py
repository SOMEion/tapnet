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


def create_grid_query_points(num_frames, height, width, grid_size=32):
    """Create a grid of query points at frame 0.

    Args:
        num_frames: Number of frames in video
        height: Video height
        width: Video width
        grid_size: Number of points per dimension (default 32x32 = 1024 points)

    Returns:
        Query points array of shape [grid_size*grid_size, 3] with format [t, y, x]
    """
    # Create normalized grid centers (0 to 1)
    grid_centers = np.arange(grid_size) / grid_size + 1.0 / (2.0 * grid_size)

    # Create meshgrid in pixel coordinates
    y_coords = grid_centers * height
    x_coords = grid_centers * width
    query_y, query_x = np.meshgrid(y_coords, x_coords, indexing='ij')

    # All points start at frame 0
    query_t = np.zeros_like(query_x)

    # Stack and reshape to [num_points, 3]
    query_points = np.stack([query_t, query_y, query_x], axis=-1)
    query_points = query_points.reshape(-1, 3)

    return query_points


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
