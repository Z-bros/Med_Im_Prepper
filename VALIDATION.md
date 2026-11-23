# Validation — 0.3.0

Synthetic verification performed on 2026-11-23.

- 45 unit tests discovered: 43 passed, 2 skipped.
- Skipped: optional DICOM and NIfTI reader tests because pydicom/nibabel are not
  installed in the current runtime. Reader code is unchanged since 0.1.0.
- New characterization checks use analytically known results: moments and
  percentiles of 1..100 (variance (n²−1)/12), entropy of exactly 1 bit and 4 bits,
  zero entropy/std for constant inputs, label vs. nonzero ROIs in 3D, histogram
  edge coverage, SNR = 25 and CNR = 10 on a background of exactly ±2, input
  nonmutation, and rejection of 1D/4D, NaN, float masks, shape mismatch, empty
  ROI and zero background noise.
- GLCM checks: checkerboard (contrast 1, correlation −1, homogeneity 0.5,
  ASM 0.5 for neighbours; contrast 0, correlation 1 on diagonals), constant
  input, 3D stripe direction, ROI pair restriction, default and fixed-range
  quantization, offset/level validation. Per-offset contrast, correlation,
  homogeneity, ASM and energy match scikit-image 0.26.0 `graycoprops` to
  rtol 1e-10 on a random 8-level image (scikit-image installed only in the
  test environment; the test skips without it).
- Gradient checks: linear ramps give exact magnitude (3 per voxel, 6 with
  spacing 0.5, 2 in 3D) with zero std; step-edge fraction 2/6; ROI sampling.
- Morphology checks: voxel count, half-open bbox, centroid, area with
  anisotropic spacing, 3D volume, component count, and rejections.
- Timing on a synthetic 128×256×256 float32 volume with a 60×140×140 ROI:
  GLCM (32 levels, 13 offsets) 1.7 s, gradient 0.5 s, morphology 0.16 s.
- All 0.2.0 tests pass unchanged; `preprocess`/`augment` code is untouched.
- `examples/characterization.py` (noise std 5, contrast 100/60) reports
  SNR ≈ 20.3 and CNR ≈ 8.2 (expected 20 and 8), GLCM correlation ≈ 0 for
  white noise, area 900 × 0.64 = 576 mm², and edge fraction 0.129
  (116 boundary pixels / 900).
- `examples/quickstart.py` runs unchanged.

Characterization functions are descriptive statistics. They have not been
validated against IBSI reference values, scanner phantoms or other software.
No hosted CI has been run for this update.

Runtime: Python 3.14.4, NumPy 2.5.5, SciPy 1.18.1, Pillow 12.3.0.

## Validation — 0.2.0

Synthetic verification performed on 2026-09-28.

- 27 unit tests discovered: 25 passed, 2 skipped (optional DICOM/NIfTI readers).
- Checks covered translation direction/units, centered zoom, mask alignment
  and discrete labels, shared in-plane geometry, blur without slice mixing,
  intensity arithmetic, motion artifacts/reference masks, zero-motion FFT
  roundtrip, arbitrary motion-plane placement, finite outputs, invalid settings,
  input nonmutation, validation identity/RNG preservation, and reseeding.
- 0.1.0 and 0.2.0 flip/rotation/noise outputs matched exactly at seeds
  0, 1, 42, 100 when new transforms were disabled.
- Experimental motion smoke-tested on a synthetic (22, 224, 224) float32 volume.
  Motion simulation has not been validated against scanner acquisitions.

Runtime: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, Pillow 12.3.0.
