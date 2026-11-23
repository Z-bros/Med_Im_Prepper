# Validation — 0.3.0

Synthetic verification performed on 2026-11-23.

- 35 unit tests discovered: 33 passed, 2 skipped.
- Skipped: optional DICOM and NIfTI reader tests because pydicom/nibabel are not
  installed in the current runtime. Reader code is unchanged since 0.1.0.
- New characterization checks use analytically known results: moments and
  percentiles of 1..100 (variance (n²−1)/12), entropy of exactly 1 bit and 4 bits,
  zero entropy/std for constant inputs, label vs. nonzero ROIs in 3D, histogram
  edge coverage, SNR = 25 and CNR = 10 on a background of exactly ±2, input
  nonmutation, and rejection of 1D/4D, NaN, float masks, shape mismatch, empty
  ROI and zero background noise.
- All 0.2.0 tests pass unchanged; `preprocess`/`augment` code is untouched.
- `examples/characterization.py` (noise std 5, contrast 100/60) reports
  SNR ≈ 20.3 and CNR ≈ 8.2, consistent with the expected 20 and 8.
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
