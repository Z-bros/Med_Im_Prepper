"""Explicit, array-space preprocessing for scalar medical images."""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.ndimage import zoom, rotate
from ._augmentation import validate_extra_config, validate_extra_input, apply_extras


@dataclass(frozen=True)
class PreprocessConfig:
    spatial_dims: int = 2
    target_shape: tuple[int, ...] | None = None
    resize_mode: str = 'pad'  # pad preserves voxel-array aspect ratio; stretch does not
    normalization: str = 'none'  # none, minmax, zscore
    percentiles: tuple[float, float] | None = None
    window: tuple[float, float] | None = None  # lower, upper in input units (HU for CT)
    crop: tuple[tuple[int, int], ...] | None = None  # half-open bounds
    nonzero_stats: bool = False  # preserve original zero background during normalization

    def __post_init__(self):
        if self.spatial_dims not in (2, 3):
            raise ValueError('spatial_dims must be 2 or 3; channels are not spatial axes')
        if self.target_shape is not None and (
            len(self.target_shape) != self.spatial_dims or
            any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in self.target_shape)
        ):
            raise ValueError('target_shape needs one positive integer per spatial axis')
        if self.resize_mode not in ('pad', 'stretch'):
            raise ValueError('resize_mode must be pad or stretch')
        if self.normalization not in ('none', 'minmax', 'zscore'):
            raise ValueError('unknown normalization')
        for name, bounds in [('window', self.window), ('percentiles', self.percentiles)]:
            if bounds is not None and (len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]):
                raise ValueError(f'{name} requires finite increasing bounds')
        if self.percentiles is not None and not 0 <= self.percentiles[0] < self.percentiles[1] <= 100:
            raise ValueError('percentiles must lie in [0, 100]')
        if self.window is not None and self.percentiles is not None:
            raise ValueError('choose window OR percentiles, not both')
        if self.crop is not None and (len(self.crop) != self.spatial_dims or any(
            len(b) != 2 or any(not isinstance(v, int) or isinstance(v, bool) for v in b)
            or b[0] < 0 or b[1] <= b[0] for b in self.crop
        )):
            raise ValueError('crop needs valid integer (start, stop) pairs')


@dataclass(frozen=True)
class AugmentConfig:
    flip_axes: tuple[int, ...] = ()  # opt-in: laterality may matter
    flip_probability: float = 0.5
    rotation_degrees: float = 0.0
    rotation_axes: tuple[int, int] = (0, 1)
    rotation_probability: float = 0.5
    noise_std: float = 0.0  # output intensity units, applied after normalization
    noise_probability: float = 0.5

    # New in 0.2.0. All transforms are disabled by their identity defaults.
    translation_pixels: tuple[float, ...] | None = None  # per-axis maximum absolute shift
    translation_probability: float = 0.5
    zoom_range: tuple[float, float] = (1.0, 1.0)
    zoom_axes: tuple[int, ...] | None = None  # None = all spatial axes
    zoom_probability: float = 0.5
    intensity_scale_range: tuple[float, float] = (1.0, 1.0)
    intensity_shift_range: tuple[float, float] = (0.0, 0.0)
    intensity_probability: float = 0.5
    blur_sigma_range: tuple[float, float] = (0.0, 0.0)  # pixel/voxel units
    blur_axes: tuple[int, ...] | None = None
    blur_probability: float = 0.5
    motion_degrees: float = 0.0
    motion_translation_pixels: tuple[float, float] = (0.0, 0.0)
    motion_axes: tuple[int, int] | None = None  # None = last two array axes
    motion_phase_axis: int | None = None  # None = first selected motion axis
    motion_segments: int = 3
    motion_probability: float = 0.5

    def __post_init__(self):
        for p in (self.flip_probability, self.rotation_probability, self.noise_probability):
            if not np.isfinite(p) or not 0 <= p <= 1:
                raise ValueError('probabilities must lie in [0, 1]')
        if not np.isfinite(self.rotation_degrees) or not 0 <= self.rotation_degrees <= 180:
            raise ValueError('rotation_degrees must lie in [0, 180]')
        if not np.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError('noise_std must be finite and nonnegative')
        if len(set(self.flip_axes)) != len(self.flip_axes):
            raise ValueError('flip_axes must be unique')
        validate_extra_config(self)


def _validate(image, mask, dims):
    x = np.asarray(image)
    if x.ndim != dims or any(n == 0 for n in x.shape):
        raise ValueError(f'expected nonempty scalar {dims}D array, got {x.shape}')
    if x.dtype.kind not in 'biuf' or not np.isfinite(x).all():
        raise ValueError('image must contain finite real numbers')
    x = x.astype(np.float32, copy=True)
    if not np.isfinite(x).all():
        raise ValueError('image values exceed float32 range')
    if mask is None:
        return x, None
    m = np.asarray(mask)
    if m.shape != x.shape or m.dtype.kind not in 'biu':
        raise ValueError('mask must be an integer/bool label array with the same shape')
    if m.min() < 0 or m.max() > 2147483647:
        raise ValueError('mask labels must be in [0, 2**31 - 1]')
    return x, m.astype(np.int32, copy=True)


def preprocess(image, mask=None, *, config=None):
    """Return {'image', 'mask', 'metadata'}; never modifies caller arrays.

    Mask must already share the image's spatial grid. Shape alone cannot verify
    registration. Outputs are contiguous float32 and optional int32, with no
    channel dimension. All geometry is in voxel/array coordinates, not mm.
    """
    cfg = config or PreprocessConfig()
    x, m = _validate(image, mask, cfg.spatial_dims)
    meta = {'original_shape': list(x.shape), 'config': asdict(cfg),
            'coordinate_space': 'array', 'operations': [], 'augmentation': []}
    if cfg.crop is not None:
        if any(stop > n for (_, stop), n in zip(cfg.crop, x.shape)):
            raise ValueError('crop lies outside image')
        slices = tuple(slice(start, stop) for start, stop in cfg.crop)
        x = x[slices].copy()
        m = None if m is None else m[slices].copy()
        meta['operations'].append({'crop': cfg.crop})
    foreground = x != 0 if cfg.nonzero_stats else np.ones(x.shape, dtype=bool)
    values = x[foreground]
    if values.size:
        bounds = cfg.window
        if cfg.percentiles is not None:
            bounds = tuple(float(v) for v in np.percentile(values, cfg.percentiles))
        if bounds is not None:
            x[foreground] = np.clip(x[foreground], *bounds)
            meta['operations'].append({'clip_bounds': list(bounds)})
        values = x[foreground].astype(np.float64)
        if cfg.normalization == 'minmax':
            # Explicit CT windows retain the same intensity scale across images.
            low, high = cfg.window if cfg.window is not None else (values.min(), values.max())
            scale = high - low
            x[foreground] = (values - low) / scale if scale > 0 else 0
            meta['operations'].append({'normalization': 'minmax', 'offset': float(low), 'scale': float(scale)})
        elif cfg.normalization == 'zscore':
            mean, std = values.mean(), values.std()
            x[foreground] = (values - mean) / std if std > 0 else 0
            meta['operations'].append({'normalization': 'zscore', 'offset': float(mean), 'scale': float(std)})
    if cfg.target_shape is not None:
        before = np.asarray(x.shape)
        target = np.asarray(cfg.target_shape)
        if cfg.resize_mode == 'pad':
            resized = np.minimum(target, np.maximum(1, np.rint(before * np.min(target / before)).astype(int)))
        else:
            resized = target
        factors = resized / before
        x = zoom(x, factors, order=1, mode='nearest', prefilter=False, grid_mode=False)
        m = None if m is None else zoom(m, factors, order=0, mode='nearest', prefilter=False, grid_mode=False)
        pads = tuple((int((t - s) // 2), int(t - s - (t - s) // 2)) for t, s in zip(target, x.shape))
        x = np.pad(x, pads, constant_values=0)
        m = None if m is None else np.pad(m, pads, constant_values=0)
        meta['operations'].append({'resize_from': before.tolist(), 'resize_to': resized.tolist(), 'pad': pads})
    meta['output_shape'] = list(x.shape)
    return {'image': np.ascontiguousarray(x), 'mask': None if m is None else np.ascontiguousarray(m), 'metadata': meta}


def augment(image, mask=None, *, config=None, seed=None):
    """Apply paired geometry and image-only intensity/artifact transforms.

    seed accepts an integer or numpy Generator. Reusing an integer repeats the
    exact augmentation; use a persistent Generator for a random sequence.
    """
    cfg = config or AugmentConfig()
    dims = np.ndim(image)
    if dims not in (2, 3):
        raise ValueError('augment expects scalar 2D/3D arrays')
    x, m = _validate(image, mask, dims)
    axes = (*cfg.flip_axes, *cfg.rotation_axes)
    if any(not isinstance(a, int) or isinstance(a, bool) or not 0 <= a < dims for a in axes):
        raise ValueError('augmentation axes must be spatial axis indices')
    if len(cfg.rotation_axes) != 2 or len(set(cfg.rotation_axes)) != 2:
        raise ValueError('rotation_axes must contain two different axes')
    validate_extra_input(cfg, x)
    rng = np.random.default_rng(seed)
    history = []
    for axis in cfg.flip_axes:
        if rng.random() < cfg.flip_probability:
            x = np.flip(x, axis)
            m = None if m is None else np.flip(m, axis)
            history.append({'flip_axis': axis})
    if cfg.rotation_degrees and rng.random() < cfg.rotation_probability:
        angle = float(rng.uniform(-cfg.rotation_degrees, cfg.rotation_degrees))
        kwargs = dict(angle=angle, axes=cfg.rotation_axes, reshape=False, mode='constant', cval=0, prefilter=False)
        x = rotate(x, order=1, **kwargs)
        m = None if m is None else rotate(m, order=0, **kwargs)
        history.append({'rotation_degrees': angle, 'axes': cfg.rotation_axes})
    x, m = apply_extras(x, m, cfg, rng, history)
    if cfg.noise_std and rng.random() < cfg.noise_probability:
        x = x + rng.normal(0, cfg.noise_std, x.shape).astype(np.float32)
        history.append({'noise_std': cfg.noise_std})
    if not np.isfinite(x).all():
        raise ValueError('augmentation produced nonfinite values; reduce augmentation strength')
    return {'image': np.ascontiguousarray(x, dtype=np.float32),
            'mask': None if m is None else np.ascontiguousarray(m),
            'metadata': {'augmentation': history}}


class MedicalPipeline:
    """Reusable deterministic preprocessing + training-only augmentation."""
    def __init__(self, preprocessing=None, augmentation=None, *, seed=None):
        self.preprocessing = preprocessing or PreprocessConfig()
        self.augmentation = augmentation
        self.rng = np.random.default_rng(seed)

    def reseed(self, seed):
        """Call per DataLoader worker to avoid copied RNG streams."""
        self.rng = np.random.default_rng(seed)

    def __call__(self, image, mask=None, *, training=False):
        result = preprocess(image, mask, config=self.preprocessing)
        if training and self.augmentation is not None:
            transformed = augment(result['image'], result['mask'], config=self.augmentation, seed=self.rng)
            result['image'], result['mask'] = transformed['image'], transformed['mask']
            result['metadata'].update(transformed['metadata'])
        return result
