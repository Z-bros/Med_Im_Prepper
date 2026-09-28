# medprep-starter

Importable preprocessing and augmentation for **scalar 2D images and 3D volumes**.
A small, readable starting package for classification and segmentation experiments.
No notebook required. Python 3.10+. Version 0.2.0, MIT licensed.

The interface is reusable; the preprocessing policy must be chosen for the dataset.
The default preserves intensity values and performs no random augmentation.
This is an array-space ML utility, not a complete medical-imaging pipeline.

## Install and call

Unzip, open a terminal in the `medprep` directory (the one containing
`pyproject.toml`), then:

```bash
python -m pip install -e ".[medical]"
python examples/quickstart.py
python -m unittest discover -s tests -v
```

The `medical` extra installs NIfTI and DICOM readers. For NumPy and grayscale
PNG/TIFF/JPEG alone, use `python -m pip install -e .`.
This project has not been published to PyPI; do not use `pip install medprep-starter`
as though that package name were a published release.

```python
from medprep import (
    load_image, MedicalPipeline, PreprocessConfig, AugmentConfig,
)

image, source_metadata = load_image("knee.png")
pipe = MedicalPipeline(
    PreprocessConfig(
        spatial_dims=2,
        target_shape=(256, 256),
        normalization="minmax",
        percentiles=(1, 99),
    ),
    AugmentConfig(rotation_degrees=7, noise_std=0.01),
    seed=42,
)

train = pipe(image, training=True)
valid = pipe(image)  # preprocessing only: also use for test/inference
x = train["image"]  # float32, (256, 256)
```

These settings demonstrate the API, not a validated knee protocol. Inspect images
before choosing percentile clipping: it can remove subtle signal. Rotation can
crop corners. Noise has standard deviation 0.01 in the normalized output units.
No output clipping is applied after noise.

## Paired segmentation

```python
import numpy as np
image = np.load("image.npy", allow_pickle=False)
mask = np.load("mask.npy", allow_pickle=False)  # integer labels, zero = background
sample = pipe(image, mask, training=True)
x, y = sample["image"], sample["mask"]
```

Image and mask must already share the same grid, orientation and physical
alignment; equal array shapes do not prove registration. Geometric operations
are paired. Images use linear interpolation; masks use nearest neighbor and
remain int32. Intensity adjustment, blur, experimental MRI motion artifacts, and noise only affect the image. Motion-artifact masks remain targets in the reference pose, not labels of ghosted structures. Negative labels and float masks are
rejected; supported label values are 0 through 2**31 - 1.

## Functions and outputs

| API | Purpose |
| --- | --- |
| `load_image(path)` | Return array and source-only metadata |
| `preprocess(image, mask=None, config=...)` | Deterministic crop, clipping, normalization, resize/pad |
| `augment(image, mask=None, config=..., seed=...)` | Paired flips/rotation/translation/zoom plus image-only intensity, blur, motion artifacts and noise |
| `MedicalPipeline(...)(image, mask=None, training=False)` | Combine both; augmentation is training-only |
| `pipe.reseed(seed)` | Reset the augmentation random generator |

Processing calls return a dictionary with `image`, `mask` (None if absent), and
`metadata`. Arrays are contiguous, have no channel dimension, and inputs are
not mutated. Metadata records configuration and applied transforms, but does
not provide inverse transforms, transformed affine matrices, or exact replay of
noise from metadata alone. Reproduce a run with the same seed, software versions,
input ordering, and number of calls.

## Configure preprocessing

Order: validate → explicit crop → clipping → normalization → resize and zero pad.

- `spatial_dims`: 2 for `(H, W)`, 3 for a scalar volume with three spatial axes.
  Array axis order is preserved. Never pass HWC/RGB arrays as volumes.
- `crop`: explicit half-open bounds, e.g. `((20, 220), (30, 230))`.
  No automatic lesion detection, foreground crop or mask-derived crop.
- `window=(low, high)`: fixed intensity limits, in the input's units.
  With minmax, these exact bounds define the output scale across images.
- `percentiles=(low, high)`: per-image clipping based on the selected pixels.
  Cannot be combined with `window`.
- `normalization`: `none`, `minmax`, or `zscore`. Constant inputs become zero
  under minmax/zscore unless an explicit fixed minmax window defines a scale.
- `nonzero_stats=True`: select original nonzero pixels for both statistics and
  intensity operations, and keep original zeros at zero. Only enable if zero
  really denotes background. **Do not assume this for CT: 0 HU is meaningful.**
- `target_shape`: optional model input dimensions.
- `resize_mode="pad"`: retain voxel-array aspect ratio approximately (integer
  rounding), then center-pad to target shape; `stretch` scales axes independently.
  Zero padding occurs after normalization and may not represent background in
  every intensity system. Array aspect ratio is not physical aspect ratio when
  voxel spacing is anisotropic.

Minimal direct call:

```python
from medprep import preprocess, PreprocessConfig
sample = preprocess(image, config=PreprocessConfig(normalization="zscore"))
```

A scalar 3D volume:

```python
volume, source = load_image("scan.nii.gz")
volume_pipe = MedicalPipeline(
    PreprocessConfig(spatial_dims=3, normalization="zscore")
)
sample = volume_pipe(volume)
```

An explicit CT window (example only; choose it for the anatomy and task):

```python
ct_pipe = MedicalPipeline(PreprocessConfig(
    spatial_dims=3, window=(-1000, 400), normalization="minmax",
))
sample = ct_pipe(hu_volume)  # supplied volume must already be calibrated to HU
```

## Extended augmentation (0.2.0)

See [AUGMENTATION.md](AUGMENTATION.md) for every new setting, axis conventions,
experimental MRI motion limitations, and a step-by-step Notebook 05 example.
See [UPGRADE.md](UPGRADE.md) for GitHub/Kaggle upgrade steps.
No new runtime dependencies are required. Existing configurations still work;
new operations are disabled by default and do not consume RNG draws when disabled.

## Configure augmentation and reproducibility

Flips are disabled unless `flip_axes` is set. Axis indices refer to your array,
not anatomical directions; inspect orientation and laterality first.
`rotation_axes=(0, 1)` selects the plane for rotation, including in 3D. Rotations
are in voxel coordinates and can be inappropriate for anisotropic volumes.
No elastic deformation, arbitrary 3D rotations, channel mixing or label remapping
is performed.

```python
from medprep import augment, AugmentConfig
aug = AugmentConfig(flip_axes=(1,), flip_probability=0.5)
a = augment(image, config=aug, seed=42)
b = augment(image, config=aug, seed=42)  # identical
```

Use a persistent `MedicalPipeline` or NumPy Generator for varied samples. Do not
pass the same integer seed for every sample unless identical random choices are
intentional. Validation calls do not consume pipeline RNG state.

For PyTorch (optional, installed separately):

```python
import torch
x = torch.from_numpy(sample["image"]).unsqueeze(0)  # C,H,W or C,D,H,W
if sample["mask"] is not None:
    y = torch.from_numpy(sample["mask"]).long()
```

A copied pipeline in each DataLoader worker otherwise inherits the same RNG
state. If your Dataset stores it as `self.pipeline`, use a worker initializer:

```python
def worker_init_fn(worker_id):
    import torch
    info = torch.utils.data.get_worker_info()
    info.dataset.pipeline.reseed(info.seed)
```

Set the DataLoader generator seed for reproducibility, and use separate pipeline
instances for concurrent threads. Stateless per-sample seeds are preferable
when results must be independent of worker scheduling.

## File-format boundaries

| Input | Behavior |
| --- | --- |
| NumPy `.npy` | Load without pickle; caller specifies dimensions and units |
| Grayscale PNG/TIFF/JPEG | Preserve pixel values/bit depth; reject color and multipage files |
| NIfTI `.nii` / `.nii.gz` | Load scalar 3D data with scaling; preserve voxel order and source affine |
| Classic `.dcm` / `.dicom` | Load one monochrome 2D frame; apply modality LUT/rescale |
| DICOM series/enhanced/multiframe | Rejected; reconstruct with an appropriate series-aware reader |

DICOM VOI LUT/windowing and MONOCHROME1 display inversion are not applied. Stored
quantitative values and display appearance are separate concerns. The reader
reports photometric interpretation; use an explicit display policy if needed.
DICOM files with pixel-padding tags are rejected until a valid-pixel policy is
implemented. Compressed DICOM may require pydicom decoder plugins, installed
separately. Extensionless DICOM is not auto-detected.

NIfTI source spacing/affine describes the input only. The loader does not
reorient images or masks. DICOM loading here does not reconstruct patient-space
geometry. Resizing, cropping and augmentation do not produce a new physical
affine. Do not export transformed arrays with the original affine or use them
as radiomics-ready measurements. Physical-space resampling, registration,
orientation harmonization and radiomics discretization need a separate pipeline.

For 3D arrays, the caller guarantees that all axes are spatial. The library
cannot infer whether a third axis represents slices, channels, or time.

## Use from GitHub or Kaggle

1. Create a repository such as `medical-image-preprocessing`.
2. Upload this directory's contents with `pyproject.toml` at repository root.
3. Commit the code. Other users can fork the repo and install their fork.
4. From Kaggle (Internet enabled) or another Python environment:

```python
# Replace these placeholders with your real repository and a commit SHA or tag.
%pip install "medprep-starter[medical] @ git+https://github.com/YOUR_USERNAME/medical-image-preprocessing.git@COMMIT_SHA"
from medprep import MedicalPipeline, PreprocessConfig
```

Pin a commit for reproducible experiments. If Kaggle Internet is disabled,
attach the source and required dependency wheels as a Kaggle Dataset and install
from those local files. A notebook may call the package but never needs to hold
its implementation. No GitHub repository was created by generating this archive.

## Experiment hygiene and extensions

Split by patient (and study where appropriate) before making slices, patches,
augmented copies or fitted dataset-wide statistics. This package uses per-image
statistics and cannot enforce your split policy. Avoid mask-derived crops at
validation when masks would not be available at deployment.

Inspect representative transformed images and overlays for each new dataset.
Synthetic unit tests validate mechanics, not clinical suitability or model gains.
The tests cover deterministic validation, reproducibility, geometry alignment,
label preservation, normalization, 3D shape handling, and synthetic file readers.

Extend `medprep/io.py` for readers and `medprep/core.py` / `medprep/_augmentation.py` for transforms. Before
adding a spatial transform, test paired mask alignment and its coordinate
convention. For full physical-space 3D workflows, integrate a geometry-aware
framework instead of stacking more array operations into this function.

References consulted for implementation:
- [pydicom modality LUT](https://pydicom.github.io/pydicom/stable/reference/generated/pydicom.pixels.apply_modality_lut.html)
- [NiBabel coordinate systems](https://nipy.org/nibabel/coordinate_systems.html)
- [SciPy interpolation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html)
