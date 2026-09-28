import json
import unittest
import numpy as np
from scipy.ndimage import center_of_mass
from medprep import AugmentConfig, MedicalPipeline, PreprocessConfig, augment
from medprep._augmentation import _motion


def phantom():
    y, x = np.mgrid[:65, :65]
    a = (((y - 32) / 22)**2 + ((x - 30) / 16)**2 < 1).astype(np.float32)
    a[18:26, 24:33] = 2
    a[37:43, 30:38] = .2
    return a


class ExtendedAugmentationTests(unittest.TestCase):
    def test_translation_direction_and_units(self):
        x = np.zeros((65, 65), np.float32)
        x[32, 32] = 1
        r = augment(x, config=AugmentConfig(
            translation_pixels=(4, 7), translation_probability=1), seed=4)
        displacement = r['metadata']['augmentation'][0]['translation_pixels']
        np.testing.assert_allclose(np.asarray(center_of_mass(r['image'])) - 32,
                                   displacement, atol=1e-5)
        self.assertEqual(r['image'].shape, x.shape)

    def test_geometry_mask_alignment_and_labels(self):
        m = np.zeros((65, 65), np.int32)
        m[15:49, 18:40] = 7
        cfg = AugmentConfig(translation_pixels=(4, 5), translation_probability=1,
                            zoom_range=(1.15, 1.15), zoom_probability=1)
        r = augment(m, m, config=cfg, seed=8)
        self.assertEqual(set(np.unique(r['mask'])), {0, 7})
        self.assertEqual(r['mask'].dtype, np.int32)
        self.assertGreater(np.mean((r['image'] > 3.5) == (r['mask'] == 7)), .99)
        self.assertFalse(np.array_equal(r['mask'], m))

    def test_centered_zoom_magnifies_and_keeps_shape(self):
        x = np.zeros((65, 65), np.float32)
        x[28:37, 28:37] = 1
        r = augment(x, config=AugmentConfig(zoom_range=(2, 2), zoom_probability=1), seed=1)
        np.testing.assert_allclose(center_of_mass(r['image']), (32, 32))
        self.assertGreater(np.count_nonzero(r['image'] > .5), 3 * np.count_nonzero(x))
        self.assertEqual(r['image'].shape, x.shape)

    def test_in_plane_geometry_keeps_slice_alignment(self):
        a = phantom()
        x = np.stack([a, a, a])
        cfg = AugmentConfig(translation_pixels=(0, 3, 4), translation_probability=1,
                            zoom_range=(.9, 1.1), zoom_axes=(1, 2), zoom_probability=1)
        r = augment(x, config=cfg, seed=11)
        np.testing.assert_array_equal(r['image'][0], r['image'][1])
        np.testing.assert_array_equal(r['image'][1], r['image'][2])

    def test_intensity_formula_mask_and_no_clipping(self):
        x = np.array([[0, .5], [1, 2]], np.float32)
        m = np.ones((2, 2), np.int32)
        r = augment(x, m, config=AugmentConfig(
            intensity_scale_range=(2, 2), intensity_shift_range=(-.2, -.2),
            intensity_probability=1), seed=3)
        np.testing.assert_allclose(r['image'], x * 2 - .2)
        np.testing.assert_array_equal(r['mask'], m)
        self.assertLess(r['image'].min(), 0)
        self.assertGreater(r['image'].max(), 1)

    def test_blur_reduces_peak_without_mixing_slices(self):
        x = np.zeros((3, 33, 33), np.float32)
        x[1, 16, 16] = 1
        m = (x > 0).astype(np.int32)
        r = augment(x, m, config=AugmentConfig(
            blur_sigma_range=(1, 1), blur_axes=(1, 2), blur_probability=1), seed=1)
        self.assertLess(r['image'].max(), 1)
        self.assertGreater(r['image'][1, 16, 17], 0)
        self.assertAlmostEqual(float(r['image'].sum()), 1, places=5)
        np.testing.assert_array_equal(r['image'][[0, 2]], 0)
        np.testing.assert_array_equal(r['mask'], m)

    def test_motion_changes_image_preserves_mask_and_shared_slices(self):
        a = phantom()
        x = np.stack([a, a, a])
        m = (x > 0).astype(np.int32)
        cfg = AugmentConfig(motion_degrees=4, motion_translation_pixels=(3, 3),
                            motion_probability=1, motion_axes=(1, 2), motion_phase_axis=1)
        r = augment(x, m, config=cfg, seed=9)
        self.assertGreater(float(np.mean(np.abs(r['image'] - x))), .001)
        self.assertTrue(np.isfinite(r['image']).all())
        self.assertGreaterEqual(r['image'].min(), 0)
        np.testing.assert_array_equal(r['mask'], m)
        np.testing.assert_allclose(r['image'][0], r['image'][2], atol=1e-6)
        info = r['metadata']['augmentation'][0]['experimental_mri_motion']
        segments = info['segments']
        self.assertEqual(segments[0]['lines'][0], 0)
        self.assertEqual(segments[-1]['lines'][1], 65)
        self.assertEqual(sum(s['reference'] for s in segments), 1)
        reference = next(s for s in segments if s['reference'])
        self.assertTrue(reference['lines'][0] <= 32 < reference['lines'][1])
        json.dumps(r['metadata'])

    def test_zero_pose_kspace_roundtrip(self):
        x = phantom()
        y, _ = _motion(x, AugmentConfig(motion_segments=4), np.random.default_rng(2))
        np.testing.assert_allclose(y, x, atol=1e-6)

    def test_motion_axis_permutation(self):
        x = np.stack([phantom()] * 3)
        cfg = AugmentConfig(motion_degrees=3, motion_probability=1)
        a = augment(x, config=cfg, seed=5)['image']
        permuted_cfg = AugmentConfig(motion_degrees=3, motion_probability=1,
                                    motion_axes=(0, 2), motion_phase_axis=0)
        b = augment(x.transpose(1, 0, 2), config=permuted_cfg, seed=5)['image']
        np.testing.assert_allclose(a, b.transpose(1, 0, 2), atol=1e-6)

    def test_combined_reseed_validation_and_nonmutation(self):
        x = phantom()
        original = x.copy()
        cfg = AugmentConfig(translation_pixels=(2, 3), translation_probability=1,
                            zoom_range=(.95, 1.05), zoom_probability=1,
                            intensity_scale_range=(.9, 1.1), intensity_probability=1,
                            blur_sigma_range=(.2, .6), blur_probability=1,
                            motion_degrees=3, motion_probability=1,
                            noise_std=.01, noise_probability=1)
        p = MedicalPipeline(augmentation=cfg, seed=42)
        a = p(x, training=True)
        b = p(x, training=True)
        self.assertFalse(np.array_equal(a['image'], b['image']))
        p.reseed(42)
        valid = p(x, training=False)
        np.testing.assert_array_equal(valid['image'], x)
        self.assertEqual(valid['metadata']['augmentation'], [])
        c = p(x, training=True)
        np.testing.assert_array_equal(a['image'], c['image'])
        self.assertEqual(a['metadata'], c['metadata'])
        np.testing.assert_array_equal(x, original)
        self.assertEqual(a['image'].dtype, np.float32)
        self.assertTrue(a['image'].flags.c_contiguous)

    def test_zero_probability_is_identity(self):
        cfg = AugmentConfig(translation_pixels=(3, 3), translation_probability=0,
                            zoom_range=(.5, 2), zoom_probability=0,
                            intensity_scale_range=(.5, 2), intensity_probability=0,
                            blur_sigma_range=(1, 2), blur_probability=0,
                            motion_degrees=5, motion_probability=0)
        x = phantom()
        r = augment(x, config=cfg, seed=1)
        np.testing.assert_array_equal(r['image'], x)
        self.assertEqual(r['metadata']['augmentation'], [])

    def test_config_errors(self):
        bad = [dict(translation_pixels=(-1, 2)), dict(zoom_range=(0, 1)),
               dict(zoom_range=(2, 1)), dict(zoom_axes=(1, 1)),
               dict(intensity_scale_range=(-1, 1)), dict(blur_sigma_range=(-1, 2)),
               dict(motion_degrees=float('nan')), dict(motion_segments=1),
               dict(motion_segments=2.5), dict(motion_axes=(0,)),
               dict(motion_translation_pixels=(1, 2, 3)), dict(blur_probability=2)]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AugmentConfig(**kwargs)

    def test_input_dependent_errors(self):
        bad = [dict(translation_pixels=(0, 1, 2)), dict(zoom_axes=(2,)),
               dict(blur_axes=(2,)), dict(motion_phase_axis=2),
               dict(motion_degrees=3, motion_segments=100)]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                augment(phantom(), config=AugmentConfig(**kwargs))
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            augment(-phantom(), config=AugmentConfig(motion_degrees=3))
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            augment(phantom(), config=AugmentConfig(
                intensity_shift_range=(-2, -2), intensity_probability=1,
                motion_degrees=3, motion_probability=1))


if __name__ == '__main__':
    unittest.main()
