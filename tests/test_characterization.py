import importlib.util
import json
import unittest
import numpy as np
from medprep import (cnr, glcm_features, gradient_stats, intensity_stats,
                     roi_morphology, roi_stats, snr)


class IntensityStatsTests(unittest.TestCase):
    def test_known_moments_and_order_statistics(self):
        x = np.arange(1, 101).reshape(10, 10)
        s = intensity_stats(x, percentiles=(0, 50, 100))
        self.assertEqual(s['n'], 100)
        self.assertAlmostEqual(s['mean'], 50.5)
        self.assertAlmostEqual(s['median'], 50.5)
        self.assertAlmostEqual(s['variance'], (100**2 - 1) / 12)
        self.assertAlmostEqual(s['std'], np.sqrt((100**2 - 1) / 12))
        self.assertEqual((s['min'], s['max'], s['dynamic_range']), (1, 100, 99))
        self.assertEqual(s['percentiles'], {0: 1, 50: 50.5, 100: 100})
        self.assertEqual(sum(s['histogram']['counts']), 100)
        json.dumps(s)

    def test_entropy_exact_values(self):
        self.assertAlmostEqual(intensity_stats(np.array([[0, 1], [0, 1]]), bins=2)['entropy_bits'], 1)
        x = np.arange(64).reshape(8, 8)
        self.assertAlmostEqual(intensity_stats(x, bins=16)['entropy_bits'], 4)
        c = intensity_stats(np.full((4, 4), 7.5))
        self.assertEqual((c['entropy_bits'], c['std'], c['dynamic_range']), (0, 0, 0))

    def test_fixed_edges_must_cover_values(self):
        x = np.array([[0, 1], [2, 3]])
        self.assertEqual(intensity_stats(x, bins=[0, 2, 4])['histogram']['counts'], [2, 2])
        with self.assertRaises(ValueError): intensity_stats(x, bins=[0, 1, 2])

    def test_roi_by_nonzero_and_label_in_3d(self):
        x = np.arange(27).reshape(3, 3, 3)
        m = np.zeros(x.shape, np.int32)
        m[0] = 1
        m[2] = 5
        self.assertAlmostEqual(intensity_stats(x, m)['mean'], (4 + 22) / 2)
        self.assertAlmostEqual(intensity_stats(x, m, label=5)['mean'], 22)
        r = roi_stats(x, m, percentiles=(50,))
        self.assertEqual(list(r), [1, 5])
        self.assertEqual(r[1]['n'], 9)
        self.assertAlmostEqual(r[5]['percentiles'][50], 22)
        self.assertEqual(intensity_stats(x, m.astype(bool))['n'], 18)

    def test_input_not_mutated(self):
        x = np.arange(16.).reshape(4, 4)
        m = (x > 5).astype(np.uint8)
        xc, mc = x.copy(), m.copy()
        intensity_stats(x, m)
        snr(x, m, 1 - m)
        np.testing.assert_array_equal(x, xc)
        np.testing.assert_array_equal(m, mc)

    def test_rejections(self):
        x = np.ones((3, 3))
        bad = [
            lambda: intensity_stats(np.ones((2, 2, 2, 2))),
            lambda: intensity_stats(np.ones(4)),
            lambda: intensity_stats(np.array([[np.nan, 1]])),
            lambda: intensity_stats(x, np.ones((2, 2), int)),
            lambda: intensity_stats(x, np.ones((3, 3))),  # float mask
            lambda: intensity_stats(x, np.zeros((3, 3), int)),  # empty ROI
            lambda: intensity_stats(x, np.ones((3, 3), int), label=2),
            lambda: intensity_stats(x, label=1),  # label without mask
        ]
        for f in bad:
            with self.assertRaises(ValueError): f()


class QualityTests(unittest.TestCase):
    def setUp(self):
        # background alternates +-2 -> population std exactly 2
        self.x = np.where(np.indices((8, 8)).sum(0) % 2 == 0, 2.0, -2.0)
        self.bg = np.zeros((8, 8), bool)
        self.bg[:4] = True
        self.a = np.zeros((8, 8), np.int32)
        self.a[4:6, 2:4] = 1
        self.b = np.zeros((8, 8), np.int32)
        self.b[6:8, 4:6] = 1
        self.x[self.a == 1] = 50
        self.x[self.b == 1] = 30

    def test_snr_and_cnr(self):
        self.assertAlmostEqual(snr(self.x, self.a, self.bg), 25)
        self.assertAlmostEqual(cnr(self.x, self.a, self.b, self.bg), 10)
        self.assertAlmostEqual(cnr(self.x, self.b, self.a, self.bg), 10)

    def test_zero_noise_rejected(self):
        x = np.ones((4, 4))
        m = np.eye(4, dtype=int)
        with self.assertRaises(ValueError): snr(x, m, 1 - m)
        with self.assertRaises(ValueError): cnr(x, m, m, 1 - m)


class GlcmTests(unittest.TestCase):
    board = np.indices((6, 6)).sum(0) % 2  # 0/1 checkerboard

    def test_checkerboard_neighbours_differ_diagonals_match(self):
        m = glcm_features(self.board, levels=2, offsets=[(0, 1)])['mean']
        self.assertAlmostEqual(m['contrast'], 1)
        self.assertAlmostEqual(m['correlation'], -1)
        self.assertAlmostEqual(m['homogeneity'], .5)
        self.assertAlmostEqual(m['asm'], .5)
        self.assertAlmostEqual(m['energy'], np.sqrt(.5))
        d = glcm_features(self.board, levels=2, offsets=[(1, 1)])['mean']
        self.assertEqual((d['contrast'], d['correlation'], d['homogeneity']), (0, 1, 1))

    def test_constant_and_default_offsets(self):
        r = glcm_features(np.full((5, 5), 3.0), levels=8)
        self.assertEqual(len(r['offsets']), 4)
        self.assertEqual(r['mean'], {'contrast': 0, 'correlation': 1, 'homogeneity': 1, 'asm': 1, 'energy': 1})
        self.assertEqual(len(glcm_features(np.ones((3, 3, 3)))['offsets']), 13)
        json.dumps(r)

    def test_3d_stripes_direction(self):
        x = np.zeros((4, 4, 4))
        x[1::2] = 1  # alternates along axis 0 only
        r = glcm_features(x, levels=2, offsets=[(1, 0, 0), (0, 1, 0)])
        self.assertEqual(r['per_offset']['contrast'], [1, 0])

    def test_roi_restricts_pairs_and_quantization(self):
        x = np.tile(np.arange(4), (4, 1))  # every row is 0, 1, 2, 3
        m = np.zeros((4, 4), int)
        m[:, :2] = 1  # ROI holds values 0 and 1 only
        r = glcm_features(x, m, levels=2, offsets=[(0, 1)])
        self.assertEqual((r['n_pairs'], r['value_range']), ([4], [0, 1]))
        self.assertAlmostEqual(r['mean']['contrast'], 1)
        # default range (0, 3), 4 levels: identity quantization, every neighbour differs by 1
        self.assertAlmostEqual(glcm_features(x, levels=4, offsets=[(0, 1)])['mean']['contrast'], 1)
        # fixed range (0, 8), 4 levels: {0,1}->0, {2,3}->1, so 1 of 3 pairs per row differs
        r = glcm_features(x, levels=4, value_range=(0, 8), offsets=[(0, 1)])
        self.assertAlmostEqual(r['mean']['contrast'], 1 / 3)

    def test_glcm_rejections(self):
        x = np.ones((4, 4))
        for kw in ({'levels': 1}, {'levels': 2.0}, {'value_range': (2, 3)}, {'value_range': (1, 1)},
                   {'offsets': [(0, 0)]}, {'offsets': [(1,)]}, {'offsets': [(0, 4)]}, {'offsets': []}):
            with self.assertRaises(ValueError): glcm_features(x, **kw)

    @unittest.skipUnless(importlib.util.find_spec('skimage'), 'scikit-image not installed')
    def test_matches_scikit_image(self):
        from skimage.feature import graycomatrix, graycoprops
        x = np.random.default_rng(3).integers(0, 8, (40, 40))
        ours = glcm_features(x, levels=8, value_range=(0, 8), offsets=[(0, 1), (1, 0)])['per_offset']
        P = graycomatrix(x.astype(np.uint8), [1], [0, np.pi / 2], levels=8, symmetric=True, normed=True)
        for name in ('contrast', 'correlation', 'homogeneity', 'ASM', 'energy'):
            np.testing.assert_allclose(sorted(ours[name.lower()]), sorted(graycoprops(P, name)[0]), rtol=1e-10)


class GradientMorphologyTests(unittest.TestCase):
    def test_ramp_gradient_and_spacing(self):
        x = np.tile(3.0 * np.arange(8), (5, 1))  # slope 3 per voxel along axis 1
        g = gradient_stats(x)
        self.assertAlmostEqual(g['mean'], 3)
        self.assertAlmostEqual(g['std'], 0)
        self.assertEqual(g['units'], 'intensity_per_voxel')
        self.assertAlmostEqual(gradient_stats(x, spacing=(1, .5))['mean'], 6)
        v = 2.0 * np.indices((4, 5, 6))[0]
        self.assertAlmostEqual(gradient_stats(v, spacing=(1, 1, 1))['max'], 2)

    def test_edge_fraction_and_roi(self):
        x = np.zeros((6, 6))
        x[:, 3:] = 10  # central differences give 5 in columns 2 and 3
        self.assertAlmostEqual(gradient_stats(x, edge_threshold=1)['edge_fraction'], 2 / 6)
        m = np.zeros((6, 6), int)
        m[:, :2] = 1
        self.assertEqual(gradient_stats(x, m, edge_threshold=1)['edge_fraction'], 0)
        with self.assertRaises(ValueError): gradient_stats(np.ones((1, 5)))
        with self.assertRaises(ValueError): gradient_stats(x, spacing=(1, 0))

    def test_morphology_2d_3d(self):
        m = np.zeros((8, 8), np.int32)
        m[2:5, 1:5] = 3
        r = roi_morphology(m)
        self.assertEqual((r['voxel_count'], r['bbox'], r['centroid_index']), (12, [[2, 5], [1, 5]], [3, 2.5]))
        self.assertIsNone(r['area'])
        self.assertAlmostEqual(roi_morphology(m, spacing=(.5, 2))['area'], 12)
        v = np.zeros((5, 5, 6), bool)
        v[:2, :3, :4] = True
        v[4, 4, 5] = True
        r = roi_morphology(v, spacing=(1, 2, 3))
        self.assertEqual((r['voxel_count'], r['components']), (25, 2))
        self.assertAlmostEqual(r['volume'], 25 * 6)
        json.dumps(r)

    def test_morphology_rejections(self):
        m = np.ones((3, 3), int)
        for f in (lambda: roi_morphology(np.zeros((3, 3), int)), lambda: roi_morphology(m, label=2),
                  lambda: roi_morphology(m, spacing=(1,)), lambda: roi_morphology(m, spacing=(1, -1)),
                  lambda: roi_morphology(np.ones(3, int)), lambda: roi_morphology(np.ones((3, 3)))):
            with self.assertRaises(ValueError): f()


if __name__ == '__main__':
    unittest.main()
