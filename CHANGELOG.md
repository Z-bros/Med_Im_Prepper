# Changelog

## 0.2.0 — 2026-09-28

- Add paired random translation and centered zoom in pixel/voxel coordinates.
- Add image-only intensity gain/offset and Gaussian blur with axis selection.
- Add opt-in experimental in-plane MRI motion using zero-phase Cartesian
  k-space band stitching; report realized poses and preserve reference masks.
- Preserve old defaults and RNG sequences when new operations are disabled.
- Add bounds/axis validation, geometry/mask/reproducibility tests, notebook
  examples, upgrade instructions and a standalone synthetic gallery script.
- No new dependencies, physical-space resampling, or scanner-accuracy claims.

## 0.1.0

Initial scalar array preprocessing, paired flips/rotation, image noise and readers.
