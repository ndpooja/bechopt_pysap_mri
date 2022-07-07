from benchopt import BaseObjective, safe_import_context

# Protect import to allow manipulating objective without importing library
# Useful for autocompletion and install commands
with safe_import_context() as import_ctx:
    import numpy as np
    from mri.operators import NonCartesianFFT, WaveletN
    

class Objective(BaseObjective):
    name = "FISTA"

    # All parameters 'p' defined here are available as 'self.p'


    def set_data(self, image, mask):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        self.image, self.mask = image, mask
        self.kspace_loc = self.mask.data
        self.fourier_op = NonCartesianFFT(samples=self.kspace_loc, shape=image.shape, implementation='gpuNUFFT')
        self.kspace_obs = self.fourier_op.op(image.data)
        self.mu = 1e-4
        self.linear_op = WaveletN(wavelet_name='sym8',
                         nb_scale=3,
                         dim=2,
                         padding='periodization')
        



    def compute(self, x_final):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        sparsity = self.mu * np.sum(np.abs(x_final))
        data_fidelity = 0.5 * np.linalg.norm(self.fourier_op.op(x_final) - self.kspace_obs)**2
        cost = sparsity + data_fidelity
        
        return dict(value=cost)

    def to_dict(self):
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver.
        # They are customizable.
        return dict(fourier_op=self.fourier_op, kspace_obs=self.kspace_obs, linear_op=self.linear_op, mu = self.mu)
