
from benchopt import safe_import_context, BaseDataset


with safe_import_context() as import_ctx:
    from pysap.data import get_sample_data
    from mri.operators import NonCartesianFFT
    from mri.operators.utils import normalize_frequency_locations
    from sparkling.utils.shots import convert_NCxNSxD_to_NCNSxD
    from sparkling.utils.gradient import get_kspace_loc_from_gradfile

class Dataset(BaseDataset):

    name = "SPARKLING_Data"

    def get_data(self):

        image = get_sample_data("mri-slice-nifti")
        traj_file = "/neurospin/optimed/KPooja/Sparkling/Output/dim2_i_RadialIO_TRW0.0_N512x512_FOV0.2x0.2_Nc34_Ns2048_OSF5_c6.64_d0.62__D8M12Y2022T237.bin"


        def get_traj_params(traj_file):
            dico_params = get_kspace_loc_from_gradfile(traj_file)[1]
            FOV = dico_params['FOV']
            M = dico_params['img_size']
            Ns = dico_params['num_samples_per_shot']
            OS = dico_params['min_osf']
            return(FOV, M, Ns, OS)

        def get_norm_samples(filename, dwell_t, num_adc_samples, kmax):
            sample_locations = convert_NCxNSxD_to_NCNSxD(get_kspace_loc_from_gradfile(filename, dwell_t, num_adc_samples)[0])
            sample_locations = normalize_frequency_locations(sample_locations, Kmax= kmax)
            return sample_locations

        FOV, M, Ns, OS = get_traj_params(traj_file)
        dwell_time = 10e-6 / OS
        kspace_loc = get_norm_samples(traj_file, dwell_t = dwell_time * 1e3, num_adc_samples = Ns*OS, kmax=(M[0]/FOV[0], M[1]/FOV[1]))
        fourier_op = NonCartesianFFT(samples=kspace_loc, shape=image.shape, implementation='gpuNUFFT')

        # `data` holds the keyword arguments for the `set_data` method of the
        # objective.
        # They are customizable.
        return dict(image=image, fourier_op = fourier_op)
