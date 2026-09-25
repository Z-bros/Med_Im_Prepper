"""Run with python examples/quickstart.py after pip install -e ."""
import numpy as np
from medprep import MedicalPipeline, PreprocessConfig, AugmentConfig

# Replace this synthetic data with load_image(...) or an existing NumPy array.
image = np.zeros((80, 120), dtype=np.float32)
image[20:60, 40:80] = 100
mask = (image > 0).astype(np.int32)
pipe = MedicalPipeline(
    PreprocessConfig(target_shape=(128, 128), normalization='minmax'),
    AugmentConfig(rotation_degrees=7, noise_std=0.01), seed=42,
)
train = pipe(image, mask, training=True)
valid = pipe(image, mask)  # no augmentation by default
print('train:', train['image'].shape, 'labels:', np.unique(train['mask']))
print('valid:', valid['image'].shape)
