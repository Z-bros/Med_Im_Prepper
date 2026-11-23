"""First-order intensity characterization of scalar 2D/3D arrays.

All quantities are computed in array space from voxel values only; voxel
spacing and orientation do not affect them. See CHARACTERIZATION.md for
definitions and limitations. These are not standardized (e.g. IBSI) radiomics
features.
"""
import numpy as np
from .core import _validate


def _roi_values(image, mask=None, label=None):
    """Validate inputs and return the selected voxel values as float64."""
    dims = np.ndim(image)
    if dims not in (2, 3):
        raise ValueError('characterization expects scalar 2D/3D arrays')
    _validate(image, mask, dims)
    x = np.asarray(image, dtype=np.float64)
    if mask is None:
        if label is not None:
            raise ValueError('label requires a mask')
        return x.ravel()
    m = np.asarray(mask)
    values = x[m != 0 if label is None else m == label]
    if values.size == 0:
        raise ValueError('ROI is empty')
    return values


def intensity_stats(image, mask=None, *, label=None,
                    percentiles=(1, 5, 25, 75, 95, 99), bins=256):
    """Return first-order statistics, histogram and entropy of image values.

    mask: optional integer/bool label array on the image grid. With label=None
    all nonzero voxels form the ROI; otherwise only voxels equal to label.
    Variance/std are population values (ddof=0). bins is passed to
    np.histogram (count or explicit edges); entropy_bits is the Shannon entropy
    of that histogram, so it is only comparable between images analysed with
    the same fixed edges. Values outside explicit edges raise ValueError.
    """
    v = _roi_values(image, mask, label)
    counts, edges = np.histogram(v, bins=bins)
    if counts.sum() != v.size:
        raise ValueError('histogram edges must cover every ROI value')
    p = counts[counts > 0] / v.size
    low, high = float(v.min()), float(v.max())
    return {
        'n': int(v.size),
        'mean': float(v.mean()),
        'median': float(np.median(v)),
        'std': float(v.std()),
        'variance': float(v.var()),
        'min': low,
        'max': high,
        'percentiles': {p_: float(q) for p_, q in zip(percentiles, np.percentile(v, percentiles))},
        'dynamic_range': high - low,
        'entropy_bits': float(-np.sum(p * np.log2(p))) + 0.0,  # + 0.0 turns -0.0 into 0.0
        'histogram': {'counts': counts.tolist(), 'edges': edges.tolist()},
    }


def roi_stats(image, mask, *, labels=None, **kwargs):
    """Return {label: intensity_stats} for each nonzero label (or given labels)."""
    _roi_values(image, mask)
    if labels is None:
        labels = [int(k) for k in np.unique(mask) if k != 0]
    return {k: intensity_stats(image, mask, label=k, **kwargs) for k in labels}


def _noise_std(image, background_mask):
    s = _roi_values(image, background_mask).std()
    if s == 0:
        raise ValueError('background standard deviation is zero; SNR/CNR undefined')
    return s


def snr(image, signal_mask, background_mask):
    """mean(signal ROI) / std(background ROI); masks select nonzero voxels.

    Single-region definition. No Rician/magnitude-MRI correction is applied.
    """
    return float(_roi_values(image, signal_mask).mean() / _noise_std(image, background_mask))


def cnr(image, mask_a, mask_b, background_mask):
    """|mean(ROI a) - mean(ROI b)| / std(background ROI); masks select nonzero voxels."""
    diff = abs(_roi_values(image, mask_a).mean() - _roi_values(image, mask_b).mean())
    return float(diff / _noise_std(image, background_mask))
