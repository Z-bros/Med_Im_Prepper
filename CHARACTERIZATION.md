# Image characterization (0.3.0)

`medprep.characterization` computes first-order intensity statistics and simple
image-quality ratios for scalar 2D images and 3D volumes. Everything is computed
from voxel values in **array space**: spacing, orientation and the affine are
not used and not needed. Inputs are validated with the same rules as
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
- These are descriptive statistics, **not** IBSI-verified radiomics features.
  No compliance with standardized radiomics definitions is claimed.
