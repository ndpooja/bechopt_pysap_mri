import numpy as np

from benchopt import safe_import_context, BaseDataset


with safe_import_context() as import_ctx:
    from pysap.data import get_sample_data
    from mri.operators.utils import convert_mask_to_locations
    from mri.operators import FFT

class Dataset(BaseDataset):

    name = "Cartesian_MRI_Data"

    def get_data(self):

        image = get_sample_data('2d-mri')
        cartesian_mask = get_sample_data("cartesian-mri-mask")
        kspace_loc = convert_mask_to_locations(cartesian_mask.data)
        fourier_op = FFT(samples= kspace_loc, shape=image.shape)

        # `data` holds the keyword arguments for the `set_data` method of the
        # objective.
        # They are customizable.
        return dict(image=image, fourier_op=fourier_op)
