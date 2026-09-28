# Validation — 0.2.0

Synthetic verification performed on 2026-09-28.

- Python 3.11/3.12-compatible source; see runtime versions below.
- 27 unit tests discovered: 25 passed, 2 skipped.
- Skipped: optional DICOM and NIfTI reader tests because pydicom/nibabel are not
  installed in the current runtime. Reader code is unchanged from 0.1.0.
- New checks cover translation direction/units, centered zoom, mask alignment
  and discrete labels, shared in-plane geometry, blur without slice mixing,
  intensity arithmetic, motion artifacts/reference masks, zero-motion FFT
  roundtrip, arbitrary motion-plane placement, finite outputs, invalid settings,
  input nonmutation, validation identity/RNG preservation, and reseeding.
- Compared 0.1.0 and 0.2.0 flip/rotation/noise outputs and histories at seeds
  0, 1, 42, 100: exact matches when new transforms are disabled.
- Experimental motion smoke-tested on a synthetic (22, 224, 224) float32 volume.
- Synthetic gallery generated and visually inspected.

Motion simulation has not been validated against scanner acquisitions, RSNA
images, expert artifact ratings, or model performance. Unit tests establish
mechanics, not clinical suitability. No hosted CI has been run for this update.

Runtime: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, Pillow 12.3.0.

The 0.2.0 wheel built successfully and was installed into a separate target;
version import and a 3D translation pipeline smoke check passed.
