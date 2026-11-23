"""Run with python examples/characterization.py after pip install -e ."""
import numpy as np
from medprep import cnr, intensity_stats, roi_stats, snr

# Synthetic phantom: noisy background with two labelled regions.
rng = np.random.default_rng(0)
image = rng.normal(0, 5, (128, 128))
mask = np.zeros(image.shape, np.int32)
mask[30:60, 30:60] = 1
mask[70:100, 70:100] = 2
image[mask == 1] += 100
image[mask == 2] += 60
background = np.zeros(image.shape, bool)
background[:20, :20] = True  # noise-only corner; choose this ROI deliberately

whole = intensity_stats(image)
print('whole image: mean %.2f  std %.2f  entropy %.2f bits' % (whole['mean'], whole['std'], whole['entropy_bits']))
edges = np.linspace(-50, 150, 65)  # fixed edges -> entropy comparable across ROIs
for label, s in roi_stats(image, mask, bins=edges).items():
    print(f"label {label}: n={s['n']} mean={s['mean']:.2f} median={s['median']:.2f} "
          f"p95={s['percentiles'][95]:.2f} entropy={s['entropy_bits']:.2f} bits")
print('SNR(label 1): %.1f' % snr(image, mask == 1, background))
print('CNR(1 vs 2): %.1f' % cnr(image, mask == 1, mask == 2, background))
