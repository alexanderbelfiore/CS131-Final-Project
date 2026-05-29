# Project Progress Update

## Technical Progress

The goal of this project is to convert a short, single-camera ultimate frisbee clip into approximate player tracking data on a 2D top-down field. Since the selected clip is short and the camera is mostly static, I have narrowed the implementation around a fixed camera-to-field mapping: correct the input frames, estimate one stable homography, detect players, project their locations into field coordinates, and then fill brief missed detections with interpolation.

The main technical progress so far has been on preprocessing and lens distortion correction, which became more important than expected once I inspected the footage. The camera has noticeable wide-angle distortion, causing field boundaries that should be straight to appear curved. This matters because the later homography step depends on straight field geometry. If the input frame is distorted, then field-line detection, corner estimation, and player projection will all inherit systematic error.

To address this, I implemented a reusable fisheye correction module in `src/lens_distortion.py`. This module builds the OpenCV fisheye camera matrix, applies image undistortion, and can also undistort individual pixel coordinates. I also created `src/tune_distortion.py`, which supports manual visual tuning of fisheye parameters by sweeping over candidate values and saving the selected parameters to `data/distortion_params.json`. The current saved parameters use a fisheye model with `k1 = 0.1`, centered at the 1920 by 1080 frame midpoint.

I then integrated the correction step into the frame extraction pipeline in `src/extract_frames.py`. This script loads the video, extracts frames at a configurable interval, optionally applies the saved lens correction, and writes a manifest with the source video frame number and timestamp for every extracted image. On the current clip, the pipeline processed `film.mp4`, identified a 30 FPS video with 390 frames over 13 seconds, extracted every 5th frame, and saved 78 undistorted frames to `data/frames`. This gives a clean sequence for the next stages: field geometry estimation, player detection, and top-down projection.

This is a slight pivot from the original proposal. The proposal listed player detection as the first implementation stage and treated lens correction as conditional. In practice, lens correction needed to become a prerequisite because the distorted field lines would make the homography less reliable. The project scope has also been simplified around a static 10-second segment, which removes the need for per-frame camera pose estimation unless small camera drift becomes visible.

## Visualization and Intermediate Result

The intermediate visualization is in `notebooks/lens_distortion_demo.ipynb`. The notebook compares a raw frame from the video against the undistorted version using the saved fisheye parameters. The visual result shows that the curved field boundary in the raw image becomes much straighter after correction.

The notebook also includes a more diagnostic visualization: it detects the white field boundary in the lower part of the image, fits a straight line to the detected boundary pixels, and overlays that fitted line on both the raw and corrected frames. In the raw frame, the best-fit line does not align cleanly with the curved boundary, which confirms the distortion problem. In the undistorted frame, the fitted line lies much more closely along the field boundary, showing that the correction is doing what the later geometry pipeline needs.

This result is meaningful because it demonstrates a functional preprocessing stage rather than just a visual filter. The corrected frames preserve the visual content of the original video while making the field geometry more compatible with line fitting and homography estimation. The remaining weakness is that the parameter selection is currently based on visual tuning rather than a fully automatic calibration process. For the final project, this is acceptable if the correction is validated on multiple frames, but automatic or semi-automatic line-based refinement remains a possible improvement.

## Updated Timeline and Objectives

For the next phase, the priority is to turn the corrected frame sequence into an end-to-end tracking prototype. The first objective is field geometry estimation. I will choose a clear undistorted reference frame, identify field lines or manually annotate reference points, and compute a single static homography from image coordinates to a regulation ultimate field. Because the camera is mostly static, this homography should apply across the clip. I will validate it by checking projected field reference points near the start, middle, and end of the sequence.

After that, I will add player detection using a pretrained person detector such as YOLO. For each detected player, I will use the bottom-center of the bounding box as the approximate ground-contact point, undistort or use the corrected frame coordinates, and project that point into top-down field coordinates. I will then filter detections that fall clearly outside the field area to reduce sideline clutter.

The final implementation step will be temporal cleanup. Since the detector may miss players for a few frames when they are occluded, I will associate nearby detections across frames and linearly interpolate short gaps. Interpolated positions will be marked separately from measured detections in the output data so the final visualization is honest about which points are observed and which are estimated.

The remaining schedule is:

- **Next 2-3 days:** finalize undistorted reference frame, annotate or detect field geometry, compute and validate the static homography.
- **Following 3-4 days:** run player detection on the corrected frames, project detections into field coordinates, and remove obvious false positives.
- **Following 2-3 days:** add simple frame-to-frame association, interpolate short occlusion gaps, and export tracking data.
- **Final 2-3 days:** produce the top-down visualization, compare it against selected original frames, document limitations, and prepare the final report.

The realistic final deliverable is a short static-camera demo showing detected players projected onto a 2D field over time, with lens-corrected input frames, a static homography, basic false-positive filtering, and short-gap interpolation. More advanced features such as robust player identity tracking, team classification, and fully automatic field-line detection will remain stretch goals.
