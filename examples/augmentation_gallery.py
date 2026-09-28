"""Synthetic demonstration: python examples/augmentation_gallery.py.

Requires matplotlib separately. Creates augmentation_gallery.png in the current
directory. This is a phantom, not patient data or validation of clinical realism.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from medprep import augment, AugmentConfig

y, x = np.mgrid[:128, :128]
image = .6 * ((((x - 63)/35)**2 + ((y - 64)/49)**2) < 1).astype(np.float32)
image[((x - 52)**2 + (y - 42)**2) < 11**2] = 1
image[68:86, 65:77] = .15
image[28:97:6, 48:52] = .85

settings = [
    ('Original', AugmentConfig()),
    ('Translation', AugmentConfig(translation_pixels=(10, 10), translation_probability=1)),
    ('Zoom', AugmentConfig(zoom_range=(1.15, 1.15), zoom_probability=1)),
    ('Intensity gain/offset', AugmentConfig(intensity_scale_range=(1.15, 1.15),
                                        intensity_shift_range=(.03, .03), intensity_probability=1)),
    ('Gaussian blur', AugmentConfig(blur_sigma_range=(1.2, 1.2), blur_probability=1)),
    ('Experimental MRI motion', AugmentConfig(motion_degrees=5,
        motion_translation_pixels=(4, 4), motion_probability=1)),
]
fig, axes = plt.subplots(2, 3, figsize=(10, 7))
for ax, (name, config) in zip(axes.flat, settings):
    result = augment(image, config=config, seed=42)
    ax.imshow(result['image'], cmap='gray', vmin=0, vmax=1)
    ax.set_title(name)
    ax.axis('off')
fig.suptitle('Synthetic phantom — new medprep augmentation options')
plt.tight_layout()
plt.savefig('augmentation_gallery.png', dpi=140)
print('Saved augmentation_gallery.png')
