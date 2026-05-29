# Ultimate Frisbee Player Tracking Project Plan

## Project Summary

This project aims to generate approximate player tracking data from a short, single-camera ultimate frisbee clip. The planned input is about 10 seconds long with little to no camera motion, so the system can likely use one stable camera-to-field mapping across the clip. Given the video, the system should detect players in each frame, estimate where they are on the field, and visualize those positions on a fixed 2D top-down field map.

The core technical idea is to avoid full 3D reconstruction. Instead, the project will treat the field as a planar surface and use computer vision methods such as player detection, manual or assisted field-point annotation, optional field-line estimation, and homography transforms to map image coordinates into field coordinates.

## Goals

### Primary Goal

Build a computer vision pipeline that takes a short ultimate frisbee video and produces a top-down 2D visualization of player locations over time.

### Success Criteria

- Detect active players in video frames with reasonable reliability.
- Filter out sideline people, substitutes, spectators, and other non-field detections.
- Estimate a static clip-level homography using visible field markings, cones, or manually annotated reference points.
- Project each detected player's field contact point, likely the bottom-center of their bounding box, onto a 2D ultimate field diagram.
- Interpolate short missing detection gaps when a player is briefly obscured and then re-detected.
- Generate a visual output showing player movement over time.

### Stretch Goals

- Maintain robust player identities across longer or ambiguous occlusions.
- Separate teams by jersey color.
- Smooth longer noisy player trajectories.
- Export polished tracking data with extra metadata and summary statistics.
- Overlay the projected top-down positions next to the original video for comparison.

## Input and Output

### Input

- A roughly 10-second ultimate frisbee video clip.
- Preferably footage with visible field lines, cones, or boundaries.
- Little to no camera motion during the selected timeframe.
- Ideally a reasonably stable sideline or elevated angle.

### Output

- A 2D top-down field visualization with player markers.
- A per-frame table of projected player coordinates, including interpolated coordinates for short detection gaps.
- Optional annotated video frames showing detections and field geometry.

Example tracking data format:

```csv
frame,timestamp,player_id,x_yards,y_yards,confidence,is_interpolated
0,0.000,player_01,34.2,18.7,0.91,false
1,0.033,player_01,34.4,18.9,,true
```

## Proposed Pipeline

### 1. Video Preprocessing and Lens Distortion

Extract frames from the source video at a manageable frame rate. For early development, processing every 5th or 10th frame may be enough to validate the geometry before optimizing for full video speed. Immediately address lens distortion to ensure field lines are straight.

Key tasks:

- Load video with OpenCV.
- Extract frames and timestamps.
- **Estimate lens distortion coefficients automatically by detecting field lines and fitting them to be straight.** Fallback to manual parameter tuning if automatic approach is unreliable.
- Apply undistortion to all frames.
- Optionally resize frames for faster detection.
- Store intermediate frames for debugging.

### 2. Field Geometry Estimation (Line-Based)

Estimate visible field lines, sidelines, end lines, and other structural cues using undistorted frames. The key insight is to identify sets of parallel and perpendicular lines (e.g., sidelines and end lines, or yard-line markers) that can serve as reference geometry. Because the selected clip has a static camera, the goal is to recover enough field reference lines once and reuse that homography across the 10-second sequence.

Key tasks:

- Detect or manually annotate strong line segments in the undistorted reference frame using edge detection, Hough transforms, or manual annotation.
- Identify pairs or sets of parallel lines (sidelines, end lines) and perpendicular intersections.
- Use line-to-line correspondences and parallel/perpendicular constraints to establish homography reference points.
- Verify that the same line geometry remains valid throughout the clip.

### 3. Player Detection

Use a pretrained object detector, likely YOLO, to detect people in each frame. The detector should return bounding boxes, confidence scores, and class labels.

Key tasks:

- Run person detection on each frame.
- Use the bottom-center of each player bounding box as the approximate ground contact point.
- Filter low-confidence detections.
- Remove detections outside likely field regions.

Main challenge:

Sideline clutter may produce many valid "person" detections that are not active players. Initial filtering can use bounding-box position, confidence, size, and proximity to the estimated field polygon.

### 4. Homography Projection

Compute a planar homography that maps image coordinates to coordinates on a regulation ultimate field. Since the camera is not moving during the selected clip, this homography should only need to be computed once unless the video includes noticeable shake or zoom.

Reference field dimensions:

- Width: 40 yards.
- Total length: 110 yards.
- Central playing field: 70 yards.
- End zones: 20 yards each.

Key tasks:

- Define destination coordinates for the top-down field.
- Use four or more corresponding image-field point pairs.
- Compute homography with OpenCV.
- Project player ground-contact points from every processed frame into field coordinates using the same matrix.
- Reject projections that fall far outside the field.

### 5. Tracking, Gap Filling, and Temporal Smoothing

After per-frame projection works, add lightweight temporal logic to make the output easier to interpret and more robust to brief missed detections. This is important because players may disappear for a few frames when they are obscured by another player, then reappear nearby.

Key tasks:

- Associate detections across nearby frames using projected field distance, bounding-box overlap, or a tracker such as ByteTrack, DeepSORT, or a simple Kalman filter.
- Detect short gaps in a player's track, such as 1-5 missing frames between reliable detections.
- Linearly interpolate the player's projected field coordinates across short gaps.
- Mark interpolated rows in the exported data so measured detections and filled estimates remain distinguishable.
- Smooth projected locations over time if jitter is visually distracting.

For the two-week version, simple nearest-neighbor association plus linear interpolation is likely enough. More advanced tracking can remain a stretch goal.

Tracking design (recommended implementation):

1. Run YOLO detections independently per frame.
2. Project each detection's bottom-center point into field coordinates (yards or meters).
3. Predict each active track's next position with a constant-velocity Kalman filter.
4. Build a track-to-detection cost matrix using field-space distance as the primary term.
5. Solve assignment with Hungarian matching, then reject matches beyond a gating threshold.
6. Keep unmatched tracks in a short "lost" state for `max_gap_frames`.
7. Reactivate a lost track with the same ID if a nearby detection reappears within the gap window.
8. Interpolate missing coordinates only for short gaps and only when both endpoints belong to the same track ID.

Why this works for sports footage:

- YOLO does not persist identity across frames by itself; tracking is required.
- Player orientation can change rapidly, but short-horizon field position and velocity are smoother than appearance.
- Projecting to field coordinates makes association more stable than using raw pixel coordinates.

Suggested defaults to start:

- `match_gate_yards`: 1.5 to 3.0 (tune by FPS and motion speed)
- `max_gap_frames`: 3 to 6
- `interp_max_gap_frames`: 2 to 5
- `min_track_length`: 3 frames
- `new_track_conf`: higher threshold than continuation threshold

Interpolation policy:

- Use linear interpolation for short, confident gaps within a single track.
- Mark each filled row with `is_interpolated = true`.
- Do not interpolate long gaps or ambiguous re-identifications.

### 6. Visualization

Render a top-down ultimate field and plot player positions for each processed frame.

Key tasks:

- Draw field boundaries, end zones, and yard/grid references.
- Plot player markers at projected coordinates.
- Generate a frame-by-frame animation or side-by-side video.
- Optionally color players by team if jersey classification is added.

## MVP Scope

The first working version should be intentionally narrow:

1. Choose one short clip with visible field boundaries.
2. Extract frames from the 10-second static-camera segment.
3. Detect people with a pretrained detector.
4. Manually annotate four or more field reference points in one clear frame.
5. Compute one static homography for the clip.
6. Project detected player points from multiple frames onto a top-down field.
7. Fill short missing detection gaps with linear interpolation.
8. Save a short top-down animation or selected frame sequence.

Once this works, improve detection filtering, refine simple track association, and add validation.

## Technical Approach

Likely tools:

- Python for the main pipeline.
- OpenCV for video I/O, homography, line detection, undistortion, and drawing.
- NumPy for coordinate transforms and geometry.
- Ultralytics YOLO or another pretrained detector for person detection.
- SciPy (`linear_sum_assignment`) for Hungarian assignment.
- FilterPy or a custom Kalman filter implementation for track prediction.
- Matplotlib or OpenCV for visualization.
- Pandas for exporting tracking data.

Possible repository structure:

```text
data/
  film.mp4
  screenshot.jpg
  distortion_params.json
  frames/
    frame_00000.jpg
    ...
    manifest.json
  detections/
    detections_raw.csv
  tracks/
    tracks_projected.csv
  outputs/
    topdown_demo.mp4
    side_by_side.mp4
src/
  extract_frames.py
  lens_distortion.py
  tune_distortion.py
  detect_players.py
  field_geometry.py
  homography.py
  project_points.py
  tracker.py
  interpolate_tracks.py
  export_tracks.py
  visualize.py
  pipeline.py
notebooks/
  lens_distortion_demo.ipynb
  prototype_homography.ipynb
  tracking_debug.ipynb
docs/
  project_plan.md
  progress_update.md
```

Module responsibilities (new files to add next):

- `src/project_points.py`: convert detection anchor points from image space to field space.
- `src/tracker.py`: track state, Kalman prediction, Hungarian assignment, lost/reactivate logic.
- `src/interpolate_tracks.py`: short-gap interpolation with guardrails and `is_interpolated` flagging.
- `src/export_tracks.py`: save per-frame outputs in a consistent schema for analysis and visualization.
- `notebooks/tracking_debug.ipynb`: inspect ID switches, missed detections, and interpolation behavior.

## Validation Plan

Validation should focus on whether the projected positions are plausible rather than perfectly precise.

Suggested checks:

- Confirm projected players stay within the field when they are visibly in bounds.
- Compare projected player positions against manually clicked ground-truth points on a few frames.
- Check that interpolated positions are only used for short gaps and remain plausible between surrounding detections.
- Measure average projection error in yards for manually annotated test points.
- Verify that field corners, sidelines, and end zones align correctly in the top-down view.
- Visually inspect side-by-side original frames and projected maps.

Potential metrics:

- Player detection precision on active-field players.
- Player detection recall on active-field players.
- Mean projection error in yards for manually labeled points.
- Percentage of projected detections inside the legal field boundary.

## Main Risks

### Static Camera Assumption

The selected 10-second clip has little to no camera motion, which makes the project much simpler because one homography can be reused across the sequence. The main risk is that small camera shake, zoom, or rolling shutter effects may still cause projection drift.

Mitigation:

- Use the most stable 10-second segment available.
- Check projected field reference points at the start, middle, and end of the clip.
- If drift is visible, annotate one or two additional keyframes and interpolate or switch homographies.

### Missing Field Boundaries

The camera may not show all four field corners or even all key boundary lines.

Mitigation:

- Use line extrapolation from visible markings.
- Manually annotate key frames for the prototype.
- Limit the first dataset to clips with enough visible geometry.

### Sideline Detections

People on the sideline may be detected as players.

Mitigation:

- Filter detections using the projected field polygon.
- Rank detections by likely active-player position and size.
- Use temporal consistency to remove stationary sideline people.

### Lens Distortion (Critical)

Wide-angle footage will make straight field lines appear curved, breaking line detection and homography accuracy. This is the primary technical challenge for this project.

Mitigation:

- Prioritize lens distortion handling as a **prerequisite** for field geometry estimation, not an optional step.
- **Primary approach: Automatically estimate distortion coefficients by detecting field lines and iteratively solving for k1/k2 that make lines straight.**
- Fallback approach: If automatic line-fitting is unreliable, manually tune distortion parameters (k1, k2) and visually verify that field lines straighten.
- Test undistortion on a representative frame: straight lines should appear visibly straighter in the undistorted output.
- Validate homography accuracy on undistorted frames: this will significantly improve line-based reference point detection.

### Occlusion and Small Players

Players can overlap, blur, or appear small in wide shots. The detector may miss a player for a few frames during an occlusion and then re-detect that player nearby.

Mitigation:

- Use high-resolution input clips.
- Process at a frame rate that captures meaningful movement.
- Use short-gap interpolation when a track disappears briefly and then resumes nearby.
- Avoid interpolating long gaps where player identity or path is ambiguous.

## Two-Week Timeline

With two weeks remaining and a static-camera 10-second clip, the project should prioritize a reliable end-to-end projection pipeline over fully automatic field reconstruction. Manual field annotation is acceptable and probably the fastest path to a strong final demo.

### Week 1: End-to-End MVP

- Select the final 10-second static-camera clip with visible field boundaries or cones.
- Build frame extraction and choose a manageable processing rate.
- Run pretrained person detection and save annotated debug frames.
- Filter obvious sideline detections using confidence, bounding-box location, and an approximate field region.
- Define the top-down ultimate field coordinate system.
- Manually annotate four or more field reference points in one representative frame.
- Compute one static homography for the clip and project detected player ground-contact points.
- Generate a top-down visualization for a small set of frames from the clip.

Deliverable:

- A static-camera demo showing detected players projected onto a top-down field for several frames.

### Week 2: Multi-Frame Demo, Validation, and Report

- Extend the projection pipeline across a short clip.
- Reuse the static homography across the full 10-second segment.
- Add lightweight track association and interpolate short missing detection gaps.
- Export per-frame projected player coordinates as CSV or JSON.
- Add a clear visualization: top-down animation, side-by-side video, or selected frame sequence.
- Validate projection quality against a small set of manually labeled player or field points.
- Document failure cases, assumptions, and project limitations.
- Prepare the final report and presentation materials.

Deliverable:

- Final demo output, tracking data file, validation results, and written report.

### If Time Allows

- Add more robust player identity tracking across nearby frames.
- Separate teams by jersey color.
- Add more advanced trajectory smoothing.
- Try automatic field-line detection and corner extrapolation.
- Test lens undistortion if field-line curvature is visibly affecting results.
- Add extra homography keyframes if small camera drift is noticeable.

## Immediate Next Steps

1. Confirm the exact 10-second static-camera segment to use.
2. Set up a Python environment with OpenCV, NumPy, and a person detector.
3. Create a small frame extraction script and visually inspect a clear frame.
4. **[PRIORITY]** Tackle lens distortion using automatic line-fitting:
   - Extract a clear frame showing field lines (sidelines, yard markers).
   - Detect white field lines using edge detection (e.g., Canny) or Hough transform.
   - Fit lines to the detected segments.
   - **Automatically solve for k1/k2 distortion coefficients that make lines straight.** (Iterative or optimization-based approach.)
   - If automatic estimation doesn't converge or produces poor results, fall back to manual parameter tuning.
   - Apply undistortion and verify that field lines now appear straight.
5. Pick one clear reference frame (post-undistortion) for field-line detection.
6. Detect or manually annotate parallel/perpendicular field lines in the reference frame.
7. Use line correspondences to compute a static homography for the clip.
8. Test the homography by projecting detected players onto a top-down field across the start, middle, and end of the clip.
9. Run person detection on sample frames and inspect false positives and sideline clutter.

## Open Questions

- What type of footage will be used: sideline, elevated sideline, end-zone, or broadcast-style?
- Are field lines clearly visible, or are cones the main field markers?
- Does the 10-second segment have any visible shake, zoom, or rolling shutter distortion?
- Does the project need player identities, or are anonymous player positions sufficient?
- What maximum missing-detection gap should be interpolated: 2 frames, 5 frames, or a time-based threshold?
- Should the final output prioritize visual demonstration, quantitative tracking data, or both?
- How much manual annotation is acceptable for the final course project?

