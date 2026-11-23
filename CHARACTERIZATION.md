# Image characterization (0.3.0)

`medprep.characterization` computes first-order intensity statistics, simple
image-quality ratios, GLCM texture, gradient statistics and ROI shape
descriptors for scalar 2D images and 3D volumes. Everything is computed in
**array space**. Intensity statistics and GLCM never use spacing; gradients
and area/volume use it only when you pass `spacing` explicitly. Orientation
and the affine are never used. Inputs are validated with the same rules as
`preprocess` (finite real image; integer/bool mask on the same grid), and
caller arrays are never modified. Results are plain Python numbers and lists
(JSON-serializable).

```python
from medprep import intensity_stats, roi_stats, snr, cnr

s = intensity_stats(image)                    # whole image
s = intensity_stats(image, mask)              # all nonzero mask voxels
s = intensity_stats(image, mask, label=2)     # one label only
per_label = roi_stats(image, mask)            # {label: stats} for every nonzero label
q = snr(image, mask == 1, background)         # boolean or label masks
c = cnr(image, mask == 1, mask == 2, background)
```

A runnable synthetic example is in `examples/characterization.py`.

## `intensity_stats` output

| Key | Definition |
| --- | --- |
| `n` | Number of ROI voxels |
| `mean`, `median`, `min`, `max` | Over ROI voxels, float64 |
| `variance`, `std` | Population values (ddof = 0), matching `preprocess` z-score |
| `percentiles` | `{p: value}` for the requested `percentiles`, NumPy default linear interpolation |
| `dynamic_range` | `max - min` of the ROI, in input units |
| `entropy_bits` | Shannon entropy, base 2, of the normalized histogram: `-Σ pᵢ log₂ pᵢ` over nonempty bins |
| `histogram` | `counts` and `edges` from `np.histogram(values, bins)` |

## Assumptions and caveats

- **Entropy depends on binning.** `bins=256` (the default) divides each ROI's
  own min–max range into 256 bins, so values are *not* comparable across images
  or ROIs. For comparisons pass the same explicit edge array, e.g.
  `bins=np.linspace(-1000, 400, 141)` for HU. Values outside explicit edges raise
  `ValueError` and are never silently dropped. A constant ROI has entropy 0.
- **Units are the input's units.** Statistics computed after `preprocess`
  normalization describe normalized values, not HU or scanner units.
- **Empty ROIs, NaN/inf, 1D/4D arrays, float masks and shape mismatches** raise
  `ValueError`.
- **SNR** = mean(signal ROI) / std(background ROI). **CNR** = |mean(A) − mean(B)| /
  std(background ROI). These single-image, single-region definitions assume the
  background ROI contains only noise around a constant level. Choosing that
  region is up to you. The result is signed if the signal mean is negative
  (e.g. CT in HU). If the background std is zero, `ValueError` is raised.
- **Magnitude MRI**: background noise is Rician/noncentral-χ, not Gaussian, and
  parallel imaging makes noise spatially varying. No correction factor is
  applied; use dual-acquisition or vendor-specific methods for formal SNR
  measurements.

## Texture, gradients and shape

```python
from medprep import glcm_features, gradient_stats, roi_morphology

t = glcm_features(image, mask, label=1, levels=32, value_range=(-1000, 400))
t['mean']['contrast'], t['per_offset']['energy']
g = gradient_stats(image, mask, spacing=(2.0, 0.8, 0.8), edge_threshold=50)
s = roi_morphology(mask, label=1, spacing=(2.0, 0.8, 0.8))   # s['volume'] in spacing units^3
```

### `glcm_features`

- **Quantization.** ROI values are binned into `levels` equal-width bins over
  `value_range`: `floor((v - low) / (high - low) * levels)`, with `high` itself
  in the top bin. By default the range is the ROI's own min/max, so features are
  **not comparable across images** unless you pass a fixed `value_range` (and
  the same `levels`). Values outside an explicit range raise `ValueError`.
  A constant ROI maps to a single level.
- **Offsets** are integer voxel steps, one per array axis, e.g. `(0, 1)` is
  the next column. The default is every unit-step neighbour direction (4 in
  2D, 13 in 3D). They are voxel steps, not millimetres: with anisotropic
  spacing, `(1, 0, 0)` and `(0, 1, 0)` cover different physical distances. Pass
  explicit in-plane offsets (e.g. `[(0, 1, 0), (0, 0, 1), (0, 1, 1), (0, 1, -1)]`)
  when slice thickness differs from in-plane resolution.
- A pair is counted only when **both** voxels lie in the ROI. An offset with no
  pairs raises `ValueError`; it is not silently skipped.
- Each offset's matrix is symmetrized and normalized to sum to 1. Features are
  reported per offset (`per_offset`) and as their unweighted mean (`mean`).
- Definitions follow scikit-image `graycoprops` (verified to 1e-10 in the tests
  when scikit-image is installed; it is **not** a dependency):
  `contrast = Σ P(i,j)(i−j)²`, `homogeneity = Σ P/(1+(i−j)²)`, `asm = Σ P²`,
  `energy = √asm` (some texts call `asm` itself "energy"), and
  `correlation = Σ P(i−μ)(j−μ)/σ²`, set to 1 when σ² = 0, as scikit-image does.

### `gradient_stats`

- Gradient magnitude `√Σ(∂I/∂axis)²` from `np.gradient` (central differences,
  one-sided at array borders). It is exact on linear ramps, and there is no smoothing.
- It is computed over the whole image and then sampled in the ROI, so ROI boundary
  voxels include intensity steps to their neighbours outside the ROI.
- Units are intensity per voxel, or per spacing unit when `spacing` is given.
- Returns the full `intensity_stats` dictionary of the magnitude, plus `units`
  and, with `edge_threshold`, `edge_fraction` (share of ROI voxels with
  magnitude ≥ threshold). The threshold is in the same units.

### `roi_morphology`

- `voxel_count`; `bbox` as half-open `[start, stop)` index pairs (the same form as
  `PreprocessConfig.crop`); `centroid_index` in array indices; `components`
  (face-connected); `area` (2D) or `volume` (3D) as `voxel_count × Πspacing`, or
  `None` without spacing.
- Area/volume come from counting voxels. No perimeter or surface mesh is computed.
- **Spacing must describe the array you pass.** `load_image` reports
  `source_spacing` (NIfTI, in array-axis order) and `source_pixel_spacing`
  (DICOM, row then column). Those are valid only for the unresized source
  array. After `preprocess` resizing or any `augment` geometry, the source
  spacing no longer applies, and this package does not compute a new one.
  Cropping keeps spacing.

## Not radiomics

These are descriptive statistics, **not** IBSI-verified radiomics features.
No compliance with standardized radiomics definitions is claimed. For
standardized features, use a dedicated radiomics package on spatially
calibrated, untransformed images.
