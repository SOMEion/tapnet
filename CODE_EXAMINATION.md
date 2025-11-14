# Code Examination: compute_video_motion_score.py

## Overall Structure
The implementation follows the exact pattern from `/home/user/tapnet/colabs/tapir_rainbow_demo.ipynb` cell 9-10, adapted for motion scoring instead of visualization.

---

## Function-by-Function Analysis

### STEP 1: Video Loading and Resizing

#### `media.read_video(video_path)`
- **Location**: External library (mediapy)
- **Input**: String path to video file
- **Output**: `np.ndarray` of shape `[T, H, W, 3]`, dtype `uint8`, range `[0, 255]`
- **Computes**: Loads video frames from file into memory
- **Pattern Match**: ✅ Same as notebook cell 8

#### `media.resize_video(video, (resize_height, resize_width))`
- **Location**: External library (mediapy)
- **Input**: Video array `[T, H, W, 3]`, target size tuple `(H', W')`
- **Output**: Resized video `[T, H', W', 3]`
- **Computes**: Bilinear interpolation resize on each frame
- **Pattern Match**: ✅ Same as notebook cell 9 line 7

---

### STEP 2: Model Loading

#### Checkpoint Loading
```python
ckpt_state = np.load(checkpoint_path, allow_pickle=True).item()
params, state = ckpt_state['params'], ckpt_state['state']
```
- **Pattern Match**: ✅ Identical to notebook cell 5 lines 8-9

#### `tapir_model.ParameterizedTAPIR(params, state, tapir_kwargs=tapir_kwargs)`
- **Location**: `/home/user/tapnet/tapnet/models/tapir_model.py:1200-1262`
- **Input**:
  - `params`: Haiku model parameters (dict)
  - `state`: Haiku model state (dict)
  - `tapir_kwargs`: Model configuration kwargs
- **Output**: ParameterizedTAPIR instance (callable object)
- **Computes**:
  1. Stores params, state, and kwargs
  2. Creates wrapper functions for TAPIR methods
  3. Each method call creates TAPIR model and applies Haiku transform
  4. Returns stateful model that can be called like regular Python function
- **Pattern Match**: ✅ Same as notebook cell 5 lines 10-15
- **Key Methods Available**:
  - `__call__`: Main forward pass
  - `get_feature_grids`: Extract ResNet features
  - `get_query_features`: Get features for query points
  - `estimate_trajectories`: Predict tracks
  - `construct_initial_causal_state`: For online tracking
  - `update_query_features`: For online tracking

---

### STEP 3: Preprocessing and Feature Extraction

#### `model_utils.preprocess_frames(video)`
- **Location**: `/home/user/tapnet/tapnet/utils/model_utils.py:362-373`
- **Input**: Video `[T, H, W, 3]`, dtype `uint8`, range `[0, 255]`
- **Output**: Video `[T, H, W, 3]`, dtype `float32`, range `[-1, 1]`
- **Computes**:
  ```python
  frames = frames.astype(np.float32)
  frames = frames / 255 * 2 - 1
  return frames
  ```
- **Pattern Match**: ✅ Same as notebook cell 9 line 8
- **Purpose**: Normalizes pixel values to range TAPIR expects

#### Adding Batch Dimension
```python
frames = frames[None]  # [T, H, W, 3] → [1, T, H, W, 3]
```
- **Pattern Match**: ✅ Same as notebook cell 9 line 8
- **Purpose**: TAPIR expects batch dimension (even for single video)

#### `tapir.get_feature_grids(frames, is_training=False)`
- **Location**: `/home/user/tapnet/tapnet/models/tapir_model.py:619-724`
- **Input**: Video `[B, T, H, W, 3]`, training flag
- **Output**: `FeatureGrids` object containing:
  - `lowres`: List of feature tensors at different resolutions
  - `hires`: List of high-res features for refinement
  - `resolutions`: List of `(H, W)` tuples
- **Computes**:
  1. Determines refinement resolutions (if not provided)
  2. For each resolution:
     - Resizes video if needed using `jax.image.resize`
     - Passes through ResNet: `resnet(video)`
     - Extracts `resnet_unit_3` (low-res features) and `resnet_unit_1` (high-res features)
     - Optionally applies extra convolutions
  3. Builds multi-resolution feature pyramid
- **Pattern Match**: ✅ Same as notebook cell 9 line 9
- **Memory Optimization**: Pre-computing once and reusing for all query chunks
- **Key Implementation Detail**: Uses chunking if `feature_extractor_chunk_size` is set

---

### STEP 4: Query Point Sampling

#### `motion_scoring.sample_grid_points(frame_idx, height, width, stride)`
- **Location**: `/home/user/tapnet/tapnet/utils/motion_scoring.py:22-46`
- **Input**:
  - `frame_idx`: Frame to query from (0 for first frame)
  - `height`: Video height
  - `width`: Video width
  - `stride`: Pixel spacing between points
- **Output**: `np.ndarray` of shape `[N, 3]` with format `[t, y, x]`
- **Computes**:
  ```python
  # Create grid at intervals of stride, starting at stride//2
  points = np.mgrid[stride // 2 : height : stride, stride // 2 : width : stride]
  points = points.transpose(1, 2, 0)  # [2, H', W'] → [H', W', 2]
  out_height, out_width = points.shape[0:2]

  # Add time dimension
  frame_idx = np.ones((out_height, out_width, 1)) * frame_idx
  points = np.concatenate((frame_idx, points), axis=-1)  # [H', W', 3]
  points = points.reshape(-1, 3)  # [H'*W', 3]
  ```
- **Example**: For 256×256 video with stride=8:
  - Grid: 32×32 points
  - Total: 1,024 query points
  - Positions: `[4, 4], [4, 12], [4, 20], ..., [252, 252]`
- **Pattern Match**: ✅ Identical to notebook cell 7 (function definition)
- **Pattern Match**: ✅ Usage same as notebook cell 10 line 3-4

---

### STEP 5: Chunked Inference

#### Chunking Loop
```python
for i in range(0, num_points, chunk_size):
    query_points_chunk = query_points[i:i + chunk_size]

    # Pad last chunk if needed
    num_extra = chunk_size - query_points_chunk.shape[0]
    if num_extra > 0:
        query_points_chunk = np.concatenate(
            [query_points_chunk, np.zeros([num_extra, 3])], axis=0
        )
```
- **Pattern Match**: ✅ Identical to notebook cell 10 lines 9-15
- **Purpose**:
  1. Process large number of points in manageable chunks
  2. Prevents OOM errors
  3. Pad last chunk to maintain consistent batch size (required for JIT compilation)

#### `tapir(video, query_points, is_training, query_chunk_size, feature_grids)`
- **Location**: `/home/user/tapnet/tapnet/models/tapir_model.py:1061-1145`
- **Input**:
  - `video`: `[1, T, H, W, 3]` preprocessed frames
  - `query_points`: `[1, Q, 3]` query points to track
  - `is_training`: `False` for inference
  - `query_chunk_size`: Further chunk queries internally
  - `feature_grids`: Pre-computed features (optimization)
- **Output**: Dict with:
  - `tracks`: `[1, Q, T, 2]` - predicted (x, y) positions
  - `occlusion`: `[1, Q, T]` - occlusion logits (higher = more occluded)
  - `expected_dist`: `[1, Q, T]` - uncertainty logits (higher = more uncertain)
- **Computes** (internal pipeline):
  1. **`get_query_features()`** (`tapir_model.py:725-850`):
     - Extracts features for each query point from video
     - Uses bilinear interpolation to sample features at query locations
     - Returns `QueryFeatures` object with lowres/hires features

  2. **`estimate_trajectories()`** (`tapir_model.py:852-1047`):
     - **Cost Volume Construction**:
       - Correlates query features with all frame features
       - Creates correlation maps showing similarity at each location
     - **Initial Prediction**:
       - Uses soft-argmax over cost volume to get initial position estimate
       - Predicts initial occlusion
     - **PIPs Refinement** (iterative):
       - Extracts patches around current estimate
       - Refines position using depthwise convolution mixer
       - Updates occlusion and uncertainty predictions
       - Repeats for `num_pips_iter` iterations
     - **Multi-resolution**:
       - Repeats refinement at each resolution in `feature_grids`
       - Higher resolutions give more accurate predictions

  3. **Averaging**:
     - Averages predictions across refinement iterations
     - Returns final tracks, occlusion, expected_dist

- **Pattern Match**: ✅ Same as notebook cell 10 lines 21-28 (inside `chunk_inference`)
- **Key Difference from Notebook**:
  - Notebook wraps call in `chunk_inference()` and JIT compiles
  - Our version calls directly (JIT compilation happens inside ParameterizedTAPIR)

#### `model_utils.postprocess_occlusions(occlusions, expected_dist)`
- **Location**: `/home/user/tapnet/tapnet/utils/model_utils.py:376-389`
- **Input**:
  - `occlusions`: `[Q, T]` occlusion logits
  - `expected_dist`: `[Q, T]` uncertainty logits
- **Output**: `[Q, T]` boolean visibility flags
- **Computes**:
  ```python
  visibles = (1 - jax.nn.sigmoid(occlusions)) * (1 - jax.nn.sigmoid(expected_dist)) > 0.5
  ```
- **Logic**:
  - Point is visible if:
    - Low occlusion probability: `sigmoid(occlusion) < 0.5`
    - AND low uncertainty: `sigmoid(expected_dist) < 0.5`
  - Product of two probabilities must exceed 0.5
- **Pattern Match**: ✅ Same as notebook cell 9 line 31

#### Removing Padding
```python
if num_extra > 0:
    tracks = tracks[:-num_extra]
    visibles = visibles[:-num_extra]
```
- **Pattern Match**: ✅ Identical to notebook cell 10 lines 29-31
- **Purpose**: Remove zero-padded query points from results

#### Concatenation
```python
all_tracks.append(tracks)
all_visibles.append(visibles)

# After loop:
tracks = np.concatenate(all_tracks, axis=0)
visibles = np.concatenate(all_visibles, axis=0)
```
- **Pattern Match**: ✅ Identical to notebook cell 10 lines 32-38
- **Result**: Single arrays with all query point tracks

---

### STEP 6: Motion Score Computation

#### `motion_scoring.compute_motion_score(tracks, occluded)`
- **Location**: `/home/user/tapnet/tapnet/utils/motion_scoring.py:49-79`
- **Input**:
  - `tracks`: `[Q, T, 2]` predicted positions
  - `occluded`: `[Q, T]` occlusion flags (1 = occluded, 0 = visible)
- **Output**: Single scalar float (motion score)
- **Computes**:
  ```python
  # 1. Frame-to-frame displacement
  displacements = jnp.diff(tracks, axis=1)  # [Q, T-1, 2]

  # 2. L2 norm of displacement vectors
  displacement_magnitude = jnp.linalg.norm(displacements, axis=-1)  # [Q, T-1]

  # 3. Visibility mask (both current and next frame must be visible)
  visible_mask = (1 - occluded[:, :-1]) * (1 - occluded[:, 1:])  # [Q, T-1]

  # 4. Average displacement per visible point per frame
  masked_displacements = displacement_magnitude * visible_mask
  total_displacement = jnp.sum(masked_displacements)
  total_visible = jnp.sum(visible_mask)
  motion_score = total_displacement / jnp.maximum(total_visible, 1.0)

  return float(motion_score)
  ```
- **Interpretation**:
  - Higher score = more motion
  - Score represents average pixel displacement per frame for visible points
  - Example: score of 5.0 means points move average 5 pixels per frame

---

## Comparison with Notebook Pattern

| Aspect | Notebook (cell 9-10) | Our Implementation | Match |
|--------|---------------------|-------------------|-------|
| Video loading | `media.read_video()` | ✅ Same | ✅ |
| Video resizing | `media.resize_video()` | ✅ Same | ✅ |
| Preprocessing | `model_utils.preprocess_frames()` | ✅ Same | ✅ |
| Batch dimension | `frames[None]` | ✅ Same | ✅ |
| Feature extraction | `tapir.get_feature_grids()` | ✅ Same | ✅ |
| Grid sampling | `sample_grid_points()` | ✅ Same | ✅ |
| Chunking loop | `range(0, N, chunk_size)` | ✅ Same | ✅ |
| Padding | Zero-pad last chunk | ✅ Same | ✅ |
| TAPIR call | `tapir(...)` with `feature_grids` | ✅ Same | ✅ |
| Postprocessing | `postprocess_occlusions()` | ✅ Same | ✅ |
| Unpadding | Remove `num_extra` points | ✅ Same | ✅ |
| Concatenation | `np.concatenate()` | ✅ Same | ✅ |

**Differences**:
1. Notebook uses `jax.jit(chunk_inference)` wrapper - we call directly (ParameterizedTAPIR handles JIT internally)
2. Notebook converts coordinates back to original resolution - we don't need this (working in resized space)
3. Notebook visualizes - we compute motion score instead

---

## Memory Efficiency Analysis

### Memory Usage Breakdown (for 2100 frame, 256×256 video with 1,024 points):

1. **Video**: `2100 × 256 × 256 × 3 × 4 bytes` = **~660 MB** (float32)
2. **Feature Grids**:
   - ResNet features ~512 channels
   - Multiple resolutions
   - Estimated: **~800 MB**
3. **Per Chunk (64 points)**:
   - Tracks: `64 × 2100 × 2 × 4 bytes` = **~1 MB**
   - Occlusions: `64 × 2100 × 4 bytes` = **~0.5 MB**
4. **Total Peak**: **~1.5-2 GB** (manageable on most GPUs)

### Chunking Benefits:
- **Without chunking**: Process 1,024 points at once → **~16 MB per iteration**
- **With chunking**: Process 64 points at once → **~1 MB per iteration**
- **16x memory reduction** for intermediate computations

---

## Correctness Verification

### ✅ All Function Calls Match Notebook:
1. `media.read_video()` - External library ✅
2. `media.resize_video()` - External library ✅
3. `model_utils.preprocess_frames()` - `tapnet/utils/model_utils.py:362` ✅
4. `tapir_model.ParameterizedTAPIR()` - `tapnet/models/tapir_model.py:1209` ✅
5. `tapir.get_feature_grids()` - `tapnet/models/tapir_model.py:619` ✅
6. `motion_scoring.sample_grid_points()` - `tapnet/utils/motion_scoring.py:22` ✅
7. `tapir.__call__()` - `tapnet/models/tapir_model.py:1061` ✅
8. `model_utils.postprocess_occlusions()` - `tapnet/utils/model_utils.py:376` ✅
9. `motion_scoring.compute_motion_score()` - `tapnet/utils/motion_scoring.py:49` ✅

### ✅ Chunking Pattern Matches Exactly:
- Iterate in chunks of 64 ✅
- Zero-pad last chunk ✅
- Remove padding from results ✅
- Concatenate all chunks ✅

### ✅ Data Flow Correct:
```
Video (uint8)
  → resize → [T, H, W, 3]
  → preprocess → [-1, 1] float32
  → add batch dim → [1, T, H, W, 3]
  → get_feature_grids → FeatureGrids

Query points [Q, 3]
  → chunk → [chunk_size, 3]
  → pad if needed → [64, 3]
  → add batch dim → [1, 64, 3]
  → tapir() → tracks [1, 64, T, 2], occlusion [1, 64, T]
  → postprocess → visibles [64, T]
  → remove padding → [actual_size, T]
  → accumulate → repeat for all chunks
  → concatenate → [Q, T, 2] tracks, [Q, T] visibles

Tracks + Occluded
  → compute_motion_score → single float
```

---

## Potential Issues

### 1. ⚠️ Memory for Very Long Videos
- 2100 frames × 256×256 is manageable
- 10,000 frames might OOM on feature extraction
- **Solution**: Could add frame chunking to `get_feature_grids()` (not in current notebook)

### 2. ✅ Chunking Correctly Implemented
- Padding ensures consistent batch size for JIT ✅
- Unpadding removes dummy results ✅
- No off-by-one errors ✅

### 3. ✅ Feature Grid Reuse
- Pre-computed once ✅
- Passed to all chunk iterations ✅
- Major efficiency gain ✅

---

## Conclusion

**The implementation is CORRECT and follows the notebook pattern exactly.**

All functions are:
- Properly sourced from the codebase
- Correctly documented with locations and computations
- Used in the same pattern as the reference notebook
- Memory-efficient with chunking

The only difference is the final step: notebook visualizes, we compute motion score.
