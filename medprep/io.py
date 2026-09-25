"""Small explicit readers; returned geometry describes the SOURCE only."""
from pathlib import Path
import numpy as np


def load_image(path):
    """Return (array, source_metadata). No normalization or display LUT.

    Supported: grayscale PNG/TIFF/JPEG, NPY, scalar 3D NIfTI, classic single-frame
    grayscale DICOM. DICOM folders, enhanced/multiframe and color are rejected.
    """
    path = Path(path)
    name = path.name.lower()
    if path.is_dir():
        raise ValueError('DICOM series reconstruction is outside this reader; provide a validated volume')
    if name.endswith('.npy'):
        return np.load(path, allow_pickle=False), {'format': 'npy', 'geometry': 'unknown'}
    if name.endswith(('.nii', '.nii.gz')):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError('Install medprep-starter[medical] for NIfTI/DICOM') from exc
        img = nib.load(path)
        if len(img.shape) != 3:
            raise ValueError('only scalar 3D NIfTI is supported')
        # Preserve original voxel order and affine, avoiding unpaired mask reorientation.
        return img.get_fdata(dtype=np.float32), {
            'format': 'nifti', 'source_affine': img.affine.tolist(),
            'source_spacing': list(map(float, img.header.get_zooms())),
            'source_axes': list(nib.aff2axcodes(img.affine)),
            'geometry': 'source_only',
        }
    if name.endswith(('.dcm', '.dicom')):
        try:
            import pydicom
            from pydicom.pixels import apply_modality_lut
        except ImportError as exc:
            raise ImportError('Install medprep-starter[medical] for NIfTI/DICOM') from exc
        ds = pydicom.dcmread(path)
        if (int(getattr(ds, 'NumberOfFrames', 1)) != 1 or
                'PerFrameFunctionalGroupsSequence' in ds or 'SharedFunctionalGroupsSequence' in ds):
            raise ValueError('enhanced/multiframe DICOM is unsupported')
        photo = str(getattr(ds, 'PhotometricInterpretation', ''))
        if int(getattr(ds, 'SamplesPerPixel', 1)) != 1 or photo not in ('MONOCHROME1', 'MONOCHROME2'):
            raise ValueError('only monochrome DICOM is supported')
        x = ds.pixel_array
        if x.ndim != 2:
            raise ValueError('expected a single 2D DICOM frame')
        if 'PixelPaddingValue' in ds:
            raise ValueError('DICOM pixel padding requires an explicit valid-pixel policy before normalization')
        x = apply_modality_lut(x, ds).astype(np.float32)
        return x, {'format': 'dicom', 'modality': str(getattr(ds, 'Modality', '')),
                   'photometric_interpretation': photo, 'modality_lut_applied': True,
                   'display_inversion_applied': False, 'geometry': 'source_only',
                   'source_pixel_spacing': list(map(float, ds.PixelSpacing)) if 'PixelSpacing' in ds else None}
    if path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
        from PIL import Image
        with Image.open(path) as img:
            if getattr(img, 'n_frames', 1) != 1:
                raise ValueError('multipage images are unsupported')
            if img.mode not in ('1', 'L', 'I', 'F', 'I;16', 'I;16B', 'I;16L'):
                raise ValueError('expected grayscale image; convert color explicitly outside this reader')
            return np.array(img), {'format': path.suffix[1:].lower(), 'geometry': 'unknown'}
    raise ValueError('unsupported extension; use .npy, .nii[.gz], .dcm/.dicom or grayscale image')
