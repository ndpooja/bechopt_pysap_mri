import numpy as np

from benchopt import safe_import_context, BaseDataset


with safe_import_context() as import_ctx:
    from pysap.data import get_sample_data
    from mri.operators import NonCartesianFFT
    from mri.operators.utils import convert_mask_to_locations

class Dataset(BaseDataset):

    name = "Non_Cartesian_MRI_Data"

    def get_data(self):

        image = get_sample_data('2d-mri')
        mask = get_sample_data("mri-radial-samples")
        kspace_loc = mask.data
        fourier_op = NonCartesianFFT(samples=kspace_loc, shape=image.shape, implementation='gpuNUFFT')

        # `data` holds the keyword arguments for the `set_data` method of the
        # objective.
        # They are customizable.
        return dict(image=image, fourier_op = fourier_op)
