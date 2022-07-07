import numpy as np

from benchopt import safe_import_context, BaseDataset


with safe_import_context() as import_ctx:
    from pysap.data import get_sample_data

class Dataset(BaseDataset):

    name = "2D_MRI"

    def get_data(self):

        image = get_sample_data('2d-mri')
        radial_mask = get_sample_data("mri-radial-samples")

        # `data` holds the keyword arguments for the `set_data` method of the
        # objective.
        # They are customizable.
        return dict(image=image, mask=radial_mask)
