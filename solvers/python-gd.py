import numpy as np


from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from mri.operators import WaveletN, GradSynthesis
    from modopt.opt.algorithms import ForwardBackward
    from modopt.opt.proximity import SparseThreshold
    from modopt.opt.linear import Identity




class Solver(BaseSolver):
    """single channel reconstructor."""
    name = 'pysap_mri_fista'
    stopping_strategy = 'callback'

    # any parameter defined here is accessible as a class attribute
    

    def set_objective(self, image, mask, fourier_op, kspace_obs):
        # The arguments of this function are the results of the
        # `to_dict` method of the objective.
        # They are customizable.
        self.image, self.mask = image, mask
        self.fourier_op, self.kspace_obs = fourier_op, kspace_obs
        mu = 1e-4
        self.linear_op = WaveletN(wavelet_name='sym8',
                         nb_scale=3,
                         dim=2,
                         padding='periodization')
        self.regularizer_op = SparseThreshold(Identity(), mu, thresh_type="soft")
        

    def run(self, callback):

        self.grad_class = GradSynthesis
        self.gradient_op = self.grad_class(
            fourier_op=self.fourier_op,
            linear_op= self.linear_op,
        )
        self.gradient_op.obs_data = self.kspace_obs
        x_init = np.squeeze(np.zeros((1,*self.gradient_op.fourier_op.shape),
                                     dtype=np.complex))
        alpha_init = self.linear_op.op(x_init)

        self.opt = ForwardBackward(
        x=alpha_init,
        grad=self.gradient_op,
        prox=self.regularizer_op,
        metric_call_period=None,
        linear=self.linear_op,
        auto_iterate=False,
        progress=True,
        cost=None,
        )

        self.opt.iterate(max_iter=1)
        while callback(self.opt.x_final):
            self.opt.iterate(max_iter=200)



    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.opt.x_final.copy()