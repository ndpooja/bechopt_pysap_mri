import numpy as np


from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from mri.reconstructors import SingleChannelReconstructor
    from modopt.opt.linear import Identity
    from modopt.opt.proximity import SparseThreshold
    from mri.operators import WaveletN
    from modopt.math.metrics import psnr, ssim



class Solver(BaseSolver):
    """single channel reconstructor."""
    name = 'FISTA'
    stopping_strategy = "iteration"
    requirements = [
        'pip:modopt',
        'pip:pysap',
        'pip:pysap-mri'
    ]

    # any parameter defined here is accessible as a class attribute

    parameters = {
        'param_name': ['greedy FISTA', 'FISTA-CD', 'FISTA-BT', 'Rada-FISTA' ],
    }


    def set_objective(self, image, fourier_op, kspace_obs, mu):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.image = image
        self.fourier_op, self.kspace_obs = fourier_op, kspace_obs
        self.mu = mu
        self.linear_op = WaveletN(wavelet_name="db4",
                         nb_scale=4,
                         dim=2,
                         padding='periodization')
        self.regularizer_op = SparseThreshold(Identity(), self.mu, thresh_type="soft")
        
        self.reconstructor = SingleChannelReconstructor(
            fourier_op=self.fourier_op,
            linear_op=self.linear_op,
            regularizer_op=self.regularizer_op,
            gradient_formulation='synthesis',
            verbose=0,
        )

        x_ref, _, _ = self.reconstructor.reconstruct(
            kspace_data=self.kspace_obs,
            optimization_alg='fista',
            num_iterations= 1000,
        )
        self.ref = np.abs(x_ref) 

        print(ssim(x_ref, self.image)) 


    def run(self, n_iter):

        if self.param_name == 'FISTA-BT':
            param = {}
        if self.param_name == 'FISTA-CD':
            param = {'a_cd': 20}
        if self.param_name == 'Rada-FISTA':
            param = {'p_lazy': 0.03333333333333333, 'q_lazy': 0.1, 'restart_strategy': 'adaptive', 'xi_restart': 0.96}
        if self.param_name == 'greedy FISTA':
            param = {'restart_strategy': 'greedy', 'xi_restart': 0.96, 's_greedy': 1.1}

        x_final, _, _ = self.reconstructor.reconstruct(
            kspace_data=self.kspace_obs,
            optimization_alg='fista',
            num_iterations= max(n_iter, 1),
            **param
        )
        self.x = np.abs(x_final) 

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.x, self.ref