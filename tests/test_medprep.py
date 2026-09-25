import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
from medprep import *

class CoreTests(unittest.TestCase):
    def test_default_identity_and_no_mutation(self):
        x = np.arange(24).reshape(4, 6)
        original = x.copy()
        r = preprocess(x)
        np.testing.assert_array_equal(r['image'], x)
        r['image'][0, 0] = 50
        np.testing.assert_array_equal(x, original)

    def test_pad_shape_and_mask_alignment(self):
        m = np.zeros((10, 20), np.int32)
        m[2:8, 5:15] = 4
        r = preprocess(m, m, config=PreprocessConfig(target_shape=(40, 40)))
        self.assertEqual(r['image'].shape, (40, 40))
        self.assertEqual(set(np.unique(r['mask'])), {0, 4})
        self.assertGreater(np.mean((r['image'] > 2) == (r['mask'] == 4)), 0.98)
        self.assertTrue(np.all(r['mask'][:10] == 0))

    def test_constant_and_zero(self):
        for n in ('minmax', 'zscore'):
            for v in (0, 7):
                r = preprocess(np.full((5, 5), v), config=PreprocessConfig(normalization=n))
                np.testing.assert_array_equal(r['image'], 0)

    def test_window_scale_is_fixed_across_images(self):
        cfg = PreprocessConfig(window=(-1000, 1000), normalization='minmax')
        x = np.array([[-2000, 0, 2000], [-1000, 500, 1000]])
        r = preprocess(x, config=cfg)
        np.testing.assert_allclose(r['image'], [[0, .5, 1], [0, .75, 1]])
        self.assertEqual(preprocess(np.full((2, 2), 500), config=cfg)['image'][0, 0], .75)

    def test_nonzero_stats_preserve_background(self):
        x = np.array([[0, 2], [0, 4]])
        r = preprocess(x, config=PreprocessConfig(normalization='zscore', nonzero_stats=True))
        np.testing.assert_array_equal(r['image'], [[0, -1], [0, 1]])

    def test_flip_pair_and_contiguity(self):
        m = np.arange(24).reshape(4, 6)
        r = augment(m, m, config=AugmentConfig(flip_axes=(1,), flip_probability=1), seed=1)
        np.testing.assert_array_equal(r['image'], m[:, ::-1])
        np.testing.assert_array_equal(r['image'], r['mask'])
        self.assertTrue(r['image'].flags.c_contiguous)

    def test_rotation_alignment_and_mask_labels(self):
        m = np.zeros((60, 60), np.int32)
        m[12:45, 20:38] = 2
        cfg = AugmentConfig(rotation_degrees=15, rotation_probability=1)
        r = augment(m, m, config=cfg, seed=4)
        self.assertGreater(np.mean((r['image'] > 1) == (r['mask'] == 2)), .99)
        self.assertEqual(set(np.unique(r['mask'])), {0, 2})

    def test_reproducibility_and_rng_progression(self):
        cfg = AugmentConfig(noise_std=.1, noise_probability=1)
        x = np.ones((10, 10))
        a = MedicalPipeline(augmentation=cfg, seed=42)
        b = MedicalPipeline(augmentation=cfg, seed=42)
        first = a(x, training=True)['image']
        np.testing.assert_array_equal(first, b(x, training=True)['image'])
        self.assertFalse(np.array_equal(first, a(x, training=True)['image']))
        np.testing.assert_array_equal(a(x)['image'], x)
        a.reseed(42)
        np.testing.assert_array_equal(first, a(x, training=True)['image'])

    def test_noise_does_not_change_mask(self):
        m = np.ones((5, 5), np.int32)
        r = augment(m, m, config=AugmentConfig(noise_std=1, noise_probability=1), seed=2)
        np.testing.assert_array_equal(r['mask'], m)

    def test_volume_and_crop(self):
        x = np.arange(6*8*10).reshape(6, 8, 10)
        cfg = PreprocessConfig(spatial_dims=3, crop=((1, 5), (2, 7), (0, 10)))
        np.testing.assert_array_equal(preprocess(x, config=cfg)['image'], x[1:5, 2:7, :])
        cfg = PreprocessConfig(spatial_dims=3, target_shape=(8, 8, 8))
        self.assertEqual(preprocess(x, config=cfg)['image'].shape, (8, 8, 8))

    def test_reject_bad_inputs(self):
        for x in (np.zeros((3, 3, 3)), np.array([[np.nan]]), np.empty((0, 3))):
            with self.assertRaises(ValueError): preprocess(x)
        with self.assertRaises(ValueError): preprocess(np.ones((2, 2)), np.ones((2, 2)))
        with self.assertRaises(ValueError): PreprocessConfig(normalization='oops')
        with self.assertRaises(ValueError): PreprocessConfig(target_shape=(0, 4))
        with self.assertRaises(ValueError): PreprocessConfig(window=(0, 1), percentiles=(1, 99))
        with self.assertRaises(ValueError): augment(np.ones((2, 2)), config=AugmentConfig(flip_axes=(2,)))
        with self.assertRaises(ValueError): preprocess(np.ones((2, 2)), config=PreprocessConfig(crop=((0, 3), (0, 2))))

    def test_npy_and_16bit_png(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            x = np.array([[0, 1000], [30000, 65535]], np.uint16)
            p = Path(d) / 'x.npy'
            np.save(p, x)
            np.testing.assert_array_equal(load_image(p)[0], x)
            p = Path(d) / 'x.png'
            Image.fromarray(x).save(p)
            np.testing.assert_array_equal(load_image(p)[0], x)
            p = Path(d) / 'rgb.png'
            Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(p)
            with self.assertRaises(ValueError): load_image(p)

    @unittest.skipUnless(importlib.util.find_spec('nibabel'), 'nibabel not installed')
    def test_nifti_preserves_source_geometry(self):
        import nibabel as nib
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'test.nii.gz'
            x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            affine = np.diag([-2., 3., 4., 1.])
            nib.save(nib.Nifti1Image(x, affine), p)
            a, meta = load_image(p)
            np.testing.assert_array_equal(a, x)
            np.testing.assert_array_equal(meta['source_affine'], affine)

    @unittest.skipUnless(importlib.util.find_spec('pydicom'), 'pydicom not installed')
    def test_dicom_rescale_and_multiframe_rejection(self):
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, CTImageStorage, generate_uid
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'test.dcm'
            fm = FileMetaDataset()
            fm.TransferSyntaxUID = ExplicitVRLittleEndian
            fm.MediaStorageSOPClassUID = CTImageStorage
            fm.MediaStorageSOPInstanceUID = generate_uid()
            ds = FileDataset(str(p), {}, file_meta=fm, preamble=b'\0'*128)
            ds.Rows, ds.Columns = 2, 2
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = 'MONOCHROME2'
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit, ds.PixelRepresentation = 15, 0
            ds.RescaleSlope, ds.RescaleIntercept = 2, -1000
            ds.Modality = 'CT'
            x = np.array([[0, 100], [500, 1000]], dtype='<u2')
            ds.PixelData = x.tobytes()
            ds.save_as(p, enforce_file_format=True)
            a, meta = load_image(p)
            np.testing.assert_array_equal(a, x.astype(float)*2 - 1000)
            self.assertTrue(meta['modality_lut_applied'])
            ds.NumberOfFrames = 2
            ds.save_as(p, enforce_file_format=True)
            with self.assertRaises(ValueError): load_image(p)

if __name__ == '__main__':
    unittest.main()
