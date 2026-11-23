"""Intensity, texture, gradient and ROI-shape characterization of scalar 2D/3D arrays.

Intensity statistics use voxel values only. Texture offsets, gradients and
shape measurements are in array (voxel-index) coordinates unless an explicit
``spacing`` is supplied. See CHARACTERIZATION.md for definitions and
limitations. These are not standardized (e.g. IBSI) radiomics features.
"""
from itertools import product
import numpy as np
from scipy import ndimage
from .core import _validate


def _roi(image, mask=None, label=None):
    """Validate inputs; return (float64 image, boolean ROI)."""
    dims = np.ndim(image)
    if dims not in (2, 3):
        raise ValueError('characterization expects scalar 2D/3D arrays')
    _validate(image, mask, dims)
    x = np.asarray(image, dtype=np.float64)
    if mask is None:
        if label is not None:
            raise ValueError('label requires a mask')
        return x, np.ones(x.shape, bool)
    m = np.asarray(mask)
    roi = m != 0 if label is None else m == label
    if not roi.any():
        raise ValueError('ROI is empty')
    return x, roi


def _roi_values(image, mask=None, label=None):
    x, roi = _roi(image, mask, label)
    return x[roi]


def _spacing(spacing, ndim):
    if spacing is None:
        return None
    s = tuple(float(v) for v in spacing)
    if len(s) != ndim or not all(np.isfinite(v) and v > 0 for v in s):
        raise ValueError('spacing needs one positive finite value per array axis')
    return s


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


def _default_offsets(ndim):
    # Unit-step neighbours with first nonzero component positive: 4 in 2D, 13 in 3D.
    return [o for o in product((-1, 0, 1), repeat=ndim) if any(o) and next(v for v in o if v) > 0]


def glcm_features(image, mask=None, *, label=None, levels=32, value_range=None, offsets=None):
    """Gray-level co-occurrence texture features, averaged over offsets.

    ROI values are quantized into ``levels`` equal-width bins over
    ``value_range`` (default: the ROI's own min/max, which is NOT comparable
    across images; pass a fixed range for comparisons). Values outside an
    explicit range raise ValueError. ``offsets`` are integer voxel steps, one
    component per array axis (default: all unit-step neighbour directions);
    a pair counts only if both voxels lie in the ROI. Each offset's matrix is
    symmetrized and normalized. Features follow the scikit-image graycoprops
    definitions: contrast, correlation (1 when gray-level variance is 0),
    homogeneity (1 / (1 + (i-j)^2) weighting), asm and energy = sqrt(asm).
    """
    x, roi = _roi(image, mask, label)
    if not isinstance(levels, (int, np.integer)) or isinstance(levels, bool) or levels < 2:
        raise ValueError('levels must be an integer >= 2')
    v = x[roi]
    if value_range is None:
        low, high = v.min(), v.max()
    else:
        low, high = (float(b) for b in value_range)
        if not (np.isfinite(low) and np.isfinite(high) and low < high):
            raise ValueError('value_range requires finite increasing bounds')
        if v.min() < low or v.max() > high:
            raise ValueError('ROI values lie outside value_range')
    q = np.full(x.shape, -1, np.int64)
    q[roi] = 0 if high == low else np.minimum(np.floor((v - low) / (high - low) * levels), levels - 1)
    offsets = _default_offsets(x.ndim) if offsets is None else [tuple(o) for o in offsets]
    if not offsets or any(len(o) != x.ndim or not any(o) or
                          any(not isinstance(d, (int, np.integer)) or isinstance(d, bool) for d in o)
                          for o in offsets):
        raise ValueError('offsets need nonzero integer steps, one per array axis')
    i, j = np.indices((levels, levels))
    names = ('contrast', 'correlation', 'homogeneity', 'asm', 'energy')
    per_offset, n_pairs = {k: [] for k in names}, []
    for off in offsets:
        src = tuple(slice(max(0, -d), n - max(0, d)) for d, n in zip(off, x.shape))
        dst = tuple(slice(max(0, d), n - max(0, -d)) for d, n in zip(off, x.shape))
        a, b = q[src].ravel(), q[dst].ravel()
        keep = (a >= 0) & (b >= 0)
        if not keep.any():
            raise ValueError(f'offset {off} has no voxel pairs inside the ROI; pass smaller offsets')
        n_pairs.append(int(keep.sum()))
        P = np.bincount(a[keep] * levels + b[keep], minlength=levels * levels).reshape(levels, levels)
        P = (P + P.T) / (2.0 * keep.sum())
        mu = (i * P).sum()
        var = ((i - mu) ** 2 * P).sum()  # equal for i and j because P is symmetric
        asm = (P ** 2).sum()
        per_offset['contrast'].append(float(((i - j) ** 2 * P).sum()))
        per_offset['correlation'].append(1.0 if var < 1e-15 else float(((i - mu) * (j - mu) * P).sum() / var))
        per_offset['homogeneity'].append(float((P / (1.0 + (i - j) ** 2)).sum()))
        per_offset['asm'].append(float(asm))
        per_offset['energy'].append(float(np.sqrt(asm)))
    return {'levels': int(levels), 'value_range': [float(low), float(high)],
            'offsets': [[int(d) for d in o] for o in offsets], 'n_pairs': n_pairs,
            'per_offset': per_offset, 'mean': {k: float(np.mean(vals)) for k, vals in per_offset.items()}}


def gradient_stats(image, mask=None, *, label=None, spacing=None, edge_threshold=None, **kwargs):
    """intensity_stats of the gradient magnitude, sampled in the ROI.

    Gradients are np.gradient central differences (one-sided at array borders)
    over the whole image, so ROI boundary voxels see neighbouring intensities.
    Units are intensity per voxel, or per spacing unit when spacing is given.
    With edge_threshold, also returns edge_fraction: share of ROI voxels whose
    magnitude is >= the threshold. Extra kwargs go to intensity_stats.
    """
    x, roi = _roi(image, mask, label)
    if min(x.shape) < 2:
        raise ValueError('gradient needs at least 2 voxels along every axis')
    sp = _spacing(spacing, x.ndim)
    magnitude = np.sqrt(sum(g * g for g in np.gradient(x, *(sp or ()))))
    out = intensity_stats(magnitude, mask, label=label, **kwargs)
    out['units'] = 'intensity_per_spacing_unit' if sp else 'intensity_per_voxel'
    if edge_threshold is not None:
        out['edge_fraction'] = float(np.mean(magnitude[roi] >= edge_threshold))
    return out


def roi_morphology(mask, *, label=None, spacing=None):
    """Voxel-count shape descriptors of one ROI.

    Returns voxel_count, half-open bounding box and centroid in array indices,
    face-connected component count, and area (2D) or volume (3D) as
    voxel_count * prod(spacing) when spacing is given, else None. Spacing must
    describe the array as passed (e.g. not the source spacing after resizing).
    """
    m = np.asarray(mask)
    if m.ndim not in (2, 3):
        raise ValueError('roi_morphology expects a 2D/3D mask')
    _validate(np.zeros(m.shape, np.float32), m, m.ndim)
    roi = m != 0 if label is None else m == label
    idx = np.argwhere(roi)
    if not len(idx):
        raise ValueError('ROI is empty')
    sp = _spacing(spacing, m.ndim)
    voxel = float(np.prod(sp)) if sp else None
    return {
        'voxel_count': int(len(idx)),
        'bbox': [[int(lo), int(hi) + 1] for lo, hi in zip(idx.min(0), idx.max(0))],
        'centroid_index': idx.mean(0).tolist(),
        'components': int(ndimage.label(roi)[1]),
        'spacing': list(sp) if sp else None,
        'area' if m.ndim == 2 else 'volume': None if voxel is None else len(idx) * voxel,
    }
