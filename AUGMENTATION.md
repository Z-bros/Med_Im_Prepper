# Augmentation in 0.2.0

All additions are opt-in. Defaults preserve the previous flip/rotation/noise
behavior. Scalar input is `(H, W)` or a three-spatial-axis array; no channel axis.
For a knee stack `(D, H, W)`, axes 1 and 2 are the in-plane dimensions.

## Operations and units

| Setting | Identity default | Meaning when enabled |
| --- | --- | --- |
| `translation_pixels` | `None` | One maximum absolute displacement per axis; each draw is uniform in ±limit. `(0, 5, 5)` preserves slice index in a stack. Positive shifts move content toward larger array indices. |
| `translation_probability` | `0.5` | Chance to apply translation. |
| `zoom_range` | `(1, 1)` | Draw one factor uniformly. Greater than 1 magnifies around array center; less than 1 shrinks. Output shape stays fixed. |
| `zoom_axes` | `None` | Apply the same factor on selected axes; `None` selects all axes. Use `(1, 2)` for in-plane stack zoom. |
| `zoom_probability` | `0.5` | Chance to apply zoom. |
| `intensity_scale_range` | `(1, 1)` | Uniform positive gain in `image * gain + offset`. This is zero-anchored gain, not mean-centered contrast. |
| `intensity_shift_range` | `(0, 0)` | Uniform additive offset in current image units. Gain and offset share one application decision. |
| `intensity_probability` | `0.5` | Chance to apply intensity adjustment. |
| `blur_sigma_range` | `(0, 0)` | Uniform Gaussian sigma in pixels/voxels, not mm. |
| `blur_axes` | `None` | Filter selected axes; use `(1, 2)` to avoid blurring between slices. |
| `blur_probability` | `0.5` | Chance to apply blur. |
| `motion_degrees` | `0` | Maximum absolute in-plane angle for experimental motion poses. |
| `motion_translation_pixels` | `(0, 0)` | Maximum absolute displacement along the two `motion_axes`. |
| `motion_axes` | `None` | Two FFT/rigid-pose axes; defaults to the last two array axes. |
| `motion_phase_axis` | `None` | Axis divided into simulated acquisition bands; defaults to the first motion axis. Must belong to `motion_axes`. This does not infer the DICOM phase-encoding direction. |
| `motion_segments` | `3` | Number of k-space bands, including one reference-pose band; integer >=2 and no larger than the phase-axis length. |
| `motion_probability` | `0.5` | Chance to apply experimental motion when at least one motion limit is nonzero. |

Equal range bounds are permitted for fixed settings. All probabilities are
independent between operations (gain and offset together form one operation).
Intensity/blur/motion are applied once per volume with shared parameters. Motion
uses the same pose schedule for all slices. Gaussian noise still has independent
voxel samples. No settings identify anatomical directions automatically.

Order: flips → rotation → centered zoom then translation (one affine resampling)
→ intensity gain/offset → Gaussian blur → experimental MRI motion → Gaussian noise.
Translation and zoom are sampled independently; both use a single resampling
when selected together. History entries list sampled translation then zoom, and
this documented order defines their combined effect.

Images use linear geometric interpolation, masks use nearest neighbor. Zero is
the fill value for geometric boundaries and mask background. Zoom and translation
can crop anatomy; zero padding cannot recover it. Intensity offsets also change
zero background. Blur uses reflective boundaries. No output intensity clipping
is performed. Anti-alias filtering is not automatically applied before spatial
downsampling; inspect smaller zooms/resizes for aliasing as well as lost detail.

## Notebook 05: begin with translation alone

First install the updated package and restart the notebook session (see UPGRADE.md).
Load `volume_resized.npy` from its exact attached path, as before.

```python
import numpy as np
import matplotlib.pyplot as plt
import medprep
from medprep import MedicalPipeline, PreprocessConfig, AugmentConfig

print(medprep.__version__)  # 0.2.0
# Replace with the path copied from your Kaggle input panel; do not scan all inputs.
volume_resized = np.load(
    '/kaggle/input/YOUR-DATASET/volume_resized.npy', allow_pickle=False
)

translation_pipe = MedicalPipeline(
    preprocessing=PreprocessConfig(spatial_dims=3),
    augmentation=AugmentConfig(
        translation_pixels=(0, 5, 5),
        translation_probability=1.0,  # force it only for inspection
    ),
    seed=42,
)
sample = translation_pipe(volume_resized, training=True)
print(sample['metadata']['augmentation'])

i = len(volume_resized) // 2
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, array, title in zip(axes,
        [volume_resized, sample['image']], ['Original', 'Translated']):
    ax.imshow(array[i], cmap='gray', vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Inspect edge loss and all slices before choosing training strengths. Five pixels
at 224×224 is not the same physical displacement as five at 512×512.

## Combined array-space example

```python
settings = AugmentConfig(
    rotation_degrees=7, rotation_axes=(1, 2),
    translation_pixels=(0, 5, 5),
    zoom_range=(0.95, 1.05), zoom_axes=(1, 2),
    intensity_scale_range=(0.9, 1.1),
    intensity_shift_range=(-0.02, 0.02),
    blur_sigma_range=(0.3, 0.8), blur_axes=(1, 2),
    noise_std=0.02,
)
pipe = MedicalPipeline(PreprocessConfig(spatial_dims=3), settings, seed=42)
train = pipe(volume_resized, training=True)
valid = pipe(volume_resized, training=False)  # augmentation off; RNG not advanced
```

These strengths are teaching examples, not a validated knee protocol. Compare
options using the same held-out studies. Do not automatically enable everything.
Existing flips remain available; choose axes only after considering orientation,
laterality and whether labels remain valid.

## Experimental MRI motion: separate from translation or blur

```python
motion_pipe = MedicalPipeline(
    PreprocessConfig(spatial_dims=3),
    AugmentConfig(
        motion_degrees=3,
        motion_translation_pixels=(2, 2),
        motion_axes=(1, 2),
        motion_phase_axis=1,  # illustrative array axis, not inferred from metadata
        motion_segments=3,
        motion_probability=1.0,
    ),
    seed=42,
)
motion_sample = motion_pipe(volume_resized, training=True)
print(motion_sample['metadata']['augmentation'])
```

The implementation treats nonnegative input as a zero-phase magnitude image,
computes a centered 2D FFT for each plane, and assembles contiguous frequency
bands from independently sampled rigid poses. The band containing the DC line
uses the untransformed reference. An inverse FFT followed by magnitude produces
the output. Bands, axes and pose parameters are included in metadata. For 3D
stacks the selected plane and schedule are shared; through-plane motion and
slice timing are not simulated. No native acquisition data are loaded.

This is a **toy Cartesian k-space corruption experiment**, not a scanner-accurate
model or a validated simulation for the RSNA sequences. It ignores actual phase,
coil sensitivities, pulse sequences, multi-shot/echo ordering, spin history,
non-Cartesian sampling, and physical voxel geometry. Cropped poses can introduce
additional boundary artifacts. Processed 224×224 inputs are useful for teaching
but make no claim to represent the original acquisition's k-space.

Motion requires nonnegative magnitude input; z-scored/negative images are
rejected when motion can run. A preceding negative intensity offset can also
cause rejection. Do not fix that by silently taking absolute values. Keep the
motion demo separate from signed intensity augmentation, as above. Gaussian
noise runs afterward and can still produce negative final values. Motion
reconstruction can exceed the original maximum; no clipping is added.

Masks are unchanged by motion corruption, just as for image noise: they represent
the reference anatomy, not shifted or ghosted features. This is a corruption
robustness target assumption that must suit the task. Earlier geometric
augmentation still transforms masks alongside the image.

## Reproducibility

New disabled operations consume no RNG draws, preserving old outputs for the
same old settings, inputs and dependency versions. Enabled settings can alter
later random draws. Use `pipe.reseed(42)` to reproduce a run; do not reseed for
every training example. Metadata reports realized settings, but does not store
noise arrays or enough RNG state for arbitrary exact replay.

## Technical references

- SciPy backward affine resampling convention:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html
- SciPy Gaussian filtering:
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
- TorchIO's description of segmented k-space motion provides context for this
  simplified independent implementation; TorchIO is not a dependency:
  https://docs.torchio.org/2.0/reference/transforms/motion/
