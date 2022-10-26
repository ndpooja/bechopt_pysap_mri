import numpy as np


from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from mri.reconstructors import SingleChannelReconstructor
    from modopt.opt.linear import Identity
    from modopt.opt.proximity import SparseThreshold



class Solver(BaseSolver):
    """single channel reconstructor."""
    name = 'pysap_mri_fista'
    stopping_strategy = "iteration"

    # any parameter defined here is accessible as a class attribute


    def set_objective(self, image, fourier_op, kspace_obs, linear_op, mu):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.image = image
        self.fourier_op, self.kspace_obs = fourier_op, kspace_obs
        self.linear_op = linear_op
        self.mu = mu
        self.regularizer_op = SparseThreshold(Identity(), self.mu, thresh_type="soft")

    def run(self, n_iter):
        reconstructor = SingleChannelReconstructor(
            fourier_op=self.fourier_op,
            linear_op=self.linear_op,
            regularizer_op=self.regularizer_op,
            gradient_formulation='synthesis',
            verbose=0,
        )

        x_final, _, _ = reconstructor.reconstruct(
            kspace_data=self.kspace_obs,
            optimization_alg='fista',
            num_iterations= max(n_iter, 1),
        )
        self.x = np.abs(x_final) 

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.x