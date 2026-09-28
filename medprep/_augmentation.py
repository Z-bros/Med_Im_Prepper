"""Array-space augmentation helpers; no patient-space geometry is inferred."""
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter


def _range(name, value, *, positive=False, nonnegative=False):
    if (not isinstance(value, (tuple, list)) or len(value) != 2
            or not np.isfinite(value).all() or value[0] > value[1]):
        raise ValueError(f'{name} requires two finite bounds in increasing order')
    if positive and value[0] <= 0:
        raise ValueError(f'{name} must be positive')
    if nonnegative and value[0] < 0:
        raise ValueError(f'{name} must be nonnegative')


def _axes(name, value, dims=None, length=None):
    if value is None:
        return
    if (not isinstance(value, (tuple, list)) or not value
            or (length is not None and len(value) != length)
            or any(not isinstance(a, int) or isinstance(a, bool) or a < 0
                   or (dims is not None and a >= dims) for a in value)
            or len(set(value)) != len(value)):
        raise ValueError(f'{name} requires distinct nonnegative spatial axis indices')


def validate_extra_config(cfg):
    for name in ('translation', 'zoom', 'intensity', 'blur', 'motion'):
        p = getattr(cfg, f'{name}_probability')
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError(f'{name}_probability must lie in [0, 1]')
    _range('zoom_range', cfg.zoom_range, positive=True)
    _range('intensity_scale_range', cfg.intensity_scale_range, positive=True)
    _range('intensity_shift_range', cfg.intensity_shift_range)
    _range('blur_sigma_range', cfg.blur_sigma_range, nonnegative=True)
    for name in ('translation_pixels', 'motion_translation_pixels'):
        v = getattr(cfg, name)
        if name == 'translation_pixels' and v is None:
            continue
        if (not isinstance(v, (tuple, list)) or len(v) not in (2, 3)
                or not np.isfinite(v).all() or any(t < 0 for t in v)):
            raise ValueError(f'{name} requires finite nonnegative per-axis limits')
    if len(cfg.motion_translation_pixels) != 2:
        raise ValueError('motion_translation_pixels requires two in-plane limits')
    if not np.isfinite(cfg.motion_degrees) or not 0 <= cfg.motion_degrees <= 180:
        raise ValueError('motion_degrees must lie in [0, 180]')
    if (not isinstance(cfg.motion_segments, int) or isinstance(cfg.motion_segments, bool)
            or cfg.motion_segments < 2):
        raise ValueError('motion_segments must be an integer >= 2')
    _axes('zoom_axes', cfg.zoom_axes)
    _axes('blur_axes', cfg.blur_axes)
    _axes('motion_axes', cfg.motion_axes, length=2)
    if cfg.motion_phase_axis is not None and (
            not isinstance(cfg.motion_phase_axis, int)
            or isinstance(cfg.motion_phase_axis, bool) or cfg.motion_phase_axis < 0):
        raise ValueError('motion_phase_axis must be a nonnegative axis index')


def _motion_enabled(cfg):
    return bool(cfg.motion_degrees or any(cfg.motion_translation_pixels))


def _motion_axes(cfg, dims):
    axes = cfg.motion_axes if cfg.motion_axes is not None else (dims - 2, dims - 1)
    phase = axes[0] if cfg.motion_phase_axis is None else cfg.motion_phase_axis
    return axes, phase


def validate_extra_input(cfg, x):
    for name in ('zoom_axes', 'blur_axes', 'motion_axes'):
        _axes(name, getattr(cfg, name), x.ndim)
    if cfg.translation_pixels is not None and len(cfg.translation_pixels) != x.ndim:
        raise ValueError('translation_pixels needs one limit per input spatial axis')
    axes, phase = _motion_axes(cfg, x.ndim)
    if phase not in axes:
        raise ValueError('motion_phase_axis must be one of motion_axes')
    if _motion_enabled(cfg) and cfg.motion_probability > 0:
        if cfg.motion_segments > x.shape[phase]:
            raise ValueError('motion_segments cannot exceed the phase-axis length')
        if x.min() < 0:
            raise ValueError('experimental MRI motion requires nonnegative magnitude input')


def _affine(x, matrix, translation, order):
    # matrix maps output coordinates back to input; center stays fixed at zero shift.
    center = (np.asarray(x.shape, dtype=float) - 1) / 2
    offset = center - matrix @ (center + translation)
    return affine_transform(x, matrix, offset=offset, output_shape=x.shape,
                            order=order, mode='constant', cval=0, prefilter=False)


def _motion(x, cfg, rng):
    """Toy Cartesian 2D k-space stitching, shared across slices in a 3D array.

    Treats the real nonnegative input as a zero-phase magnitude image. Adjacent
    bands of centered k-space come from different in-plane rigid poses. The
    band containing DC stays at the reference pose. Reconstruction is magnitude
    IFFT; masks remain reference targets. This is not a scanner forward model.
    """
    if x.min() < 0:
        raise ValueError('MRI motion requires nonnegative input after intensity adjustments')
    axes, phase = _motion_axes(cfg, x.ndim)
    n = x.shape[phase]
    cuts = np.sort(rng.choice(np.arange(1, n), cfg.motion_segments - 1, replace=False))
    bounds = [0, *cuts.tolist(), n]
    spectrum = np.fft.fftshift(np.fft.fftn(x, axes=axes), axes=axes)
    mixed = spectrum.copy()
    poses = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        reference = start <= n // 2 < stop
        angle = 0.0
        shifts = np.zeros(x.ndim)
        if not reference:
            angle = float(rng.uniform(-cfg.motion_degrees, cfg.motion_degrees))
            shifts[list(axes)] = rng.uniform(-np.asarray(cfg.motion_translation_pixels),
                                             cfg.motion_translation_pixels)
            theta = np.deg2rad(angle)
            c, s = np.cos(theta), np.sin(theta)
            matrix = np.eye(x.ndim)
            matrix[np.ix_(axes, axes)] = [[c, s], [-s, c]]
            pose = _affine(x, matrix, shifts, order=1)
            pose_spectrum = np.fft.fftshift(np.fft.fftn(pose, axes=axes), axes=axes)
            selection = [slice(None)] * x.ndim
            selection[phase] = slice(start, stop)
            mixed[tuple(selection)] = pose_spectrum[tuple(selection)]
        poses.append({'lines': [start, stop], 'reference': reference,
                      'rotation_degrees': angle,
                      'translation_pixels': shifts.tolist()})
    out = np.abs(np.fft.ifftn(np.fft.ifftshift(mixed, axes=axes), axes=axes))
    return out.astype(np.float32), {
        'experimental_mri_motion': {'axes': list(axes), 'phase_axis': phase,
                                    'model': '2d_cartesian_zero_phase_band_stitching',
                                    'segments': poses}}


def apply_extras(x, m, cfg, rng, history):
    # Identity/disabled transforms consume no random draws, preserving 0.1 behavior.
    translation = np.zeros(x.ndim)
    scale = 1.0
    translated = (cfg.translation_pixels is not None and any(cfg.translation_pixels)
                  and rng.random() < cfg.translation_probability)
    if translated:
        limits = np.asarray(cfg.translation_pixels)
        translation = rng.uniform(-limits, limits)
        history.append({'translation_pixels': translation.tolist()})
    zoomed = (tuple(cfg.zoom_range) != (1.0, 1.0)
              and rng.random() < cfg.zoom_probability)
    zoom_axes = tuple(range(x.ndim)) if cfg.zoom_axes is None else cfg.zoom_axes
    if zoomed:
        scale = float(rng.uniform(*cfg.zoom_range))
        history.append({'zoom_factor': scale, 'axes': list(zoom_axes)})
    if translated or zoomed:
        factors = np.ones(x.ndim)
        factors[list(zoom_axes)] = scale
        matrix = np.diag(1 / factors)
        # One interpolation: centered zoom followed by translation.
        x = _affine(x, matrix, translation, order=1)
        m = None if m is None else _affine(m, matrix, translation, order=0)
    intensity_on = (tuple(cfg.intensity_scale_range) != (1.0, 1.0)
                    or tuple(cfg.intensity_shift_range) != (0.0, 0.0))
    if intensity_on and rng.random() < cfg.intensity_probability:
        gain = float(rng.uniform(*cfg.intensity_scale_range))
        offset = float(rng.uniform(*cfg.intensity_shift_range))
        x = (x * gain + offset).astype(np.float32)
        history.append({'intensity_scale': gain, 'intensity_shift': offset})
    if cfg.blur_sigma_range[1] > 0 and rng.random() < cfg.blur_probability:
        sigma = float(rng.uniform(*cfg.blur_sigma_range))
        blur_axes = tuple(range(x.ndim)) if cfg.blur_axes is None else cfg.blur_axes
        sigmas = np.zeros(x.ndim)
        sigmas[list(blur_axes)] = sigma
        x = gaussian_filter(x, sigma=sigmas, mode='reflect')
        history.append({'blur_sigma_pixels': sigmas.tolist()})
    if _motion_enabled(cfg) and rng.random() < cfg.motion_probability:
        x, entry = _motion(x, cfg, rng)
        history.append(entry)
    return x, m
