# Upgrade to medprep-starter 0.2.0

The Python import stays `medprep`; the distribution version becomes 0.2.0.
No new runtime dependency is introduced. Existing preprocessing and augmentation
calls remain supported. No repository has been edited or release published by
producing this archive.

## Update your GitHub package repository

Unzip this archive. From the inner `medprep` directory (containing
`pyproject.toml`), copy these files into the same locations of your existing
package repository:

- `medprep/core.py` (updated config and dispatch)
- `medprep/_augmentation.py` (new implementation; required)
- `medprep/__init__.py` and `pyproject.toml` (version)
- `tests/test_extended_augmentation.py` (new tests)
- `README.md`, `AUGMENTATION.md`, `UPGRADE.md`, `CHANGELOG.md`, `VALIDATION.md`
- `examples/augmentation_gallery.py`

Keep your existing unrelated files. If you've changed core.py since the original
starter, merge these changes instead of overwriting those customizations.
Run `python -m unittest discover -s tests -v`, then commit. Copy the commit SHA
to pin the notebook installation. The archive is based on the original saved
starter; the live GitHub repository was not inspected.

## Kaggle install from your repository

Use the same repository URL as your earlier installation, updated to the new
commit. Substitute your actual URL and SHA below:

```python
%pip install --upgrade "medprep-starter[medical] @ git+https://github.com/YOUR_USERNAME/YOUR_PACKAGE_REPO.git@NEW_COMMIT_SHA"
```

Restart the notebook session after installation, then rerun imports and the
direct `.npy` load cell. Objects imported before the upgrade still refer to the
old class definitions until restart. Check:

```python
import medprep
from medprep import AugmentConfig
print(medprep.__version__)  # 0.2.0
print(medprep.__file__)
print(AugmentConfig(translation_pixels=(0, 5, 5)))
```

If you see an unexpected keyword error, verify that the notebook imports the
new installation rather than an older local `medprep` folder.

## Install directly from this archive

Extract it locally and run from the directory containing pyproject.toml:

```bash
python -m pip install --upgrade ".[medical]"
```

For Notebook 05's `.npy` input, base dependencies are enough; `medical` is only
needed for DICOM/NIfTI readers. For an offline Kaggle session, attach the extracted
source and dependencies, then install from its exact path. This does not publish
the package to PyPI.
