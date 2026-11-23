import json
import unittest
import numpy as np
from medprep import cnr, intensity_stats, roi_stats, snr


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


if __name__ == '__main__':
    unittest.main()
