# Changelog

## 0.3.0 — 2026-11-23

- Add `medprep.characterization`: `intensity_stats` (mean, median, std,
  variance, min/max, percentiles, dynamic range, histogram, Shannon entropy),
  `roi_stats` per mask label, and single-region `snr` / `cnr`.
- Supports scalar 2D and 3D arrays, optional label masks, and reuses the
  existing input validation. Statistics are computed in float64.
- Add `glcm_features` (N-D offsets, explicit quantization, scikit-image
  graycoprops definitions), `gradient_stats` (np.gradient magnitude, optional
  spacing and edge fraction) and `roi_morphology` (voxel count, bbox,
  centroid, components, area/volume with explicit spacing).
- Add CHARACTERIZATION.md, an example script, and analytic unit tests.
- Existing API and behavior unchanged; no new dependencies.

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
