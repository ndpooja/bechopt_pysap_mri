import numpy as np


from benchopt import BaseSolver
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    from mri.operators import GradSynthesis
    from mri.optimizers.utils.cost import GenericCost
    from modopt.opt.algorithms import ForwardBackward
    from modopt.opt.proximity import SparseThreshold
    from modopt.opt.linear import Identity
    




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
        grad_class = GradSynthesis
        self.gradient_op = grad_class(
            fourier_op=self.fourier_op,
            linear_op= self.linear_op,
        )
        self.gradient_op.obs_data = self.kspace_obs
        x_init = np.squeeze(np.zeros((1,*self.gradient_op.fourier_op.shape),
                                     dtype=np.complex))
        self.alpha_init = self.linear_op.op(x_init)
        self.regularizer_op = SparseThreshold(Identity(), self.mu, thresh_type="soft")
        self.beta_param = self.gradient_op.inv_spec_rad
        self.lambda_init = 1.0

        self.cost_op = GenericCost(
                    gradient_op=self.gradient_op,
                    linear_op=self.linear_op,
                    prox_op=self.regularizer_op,
                    verbose = 20,
                    optimizer_type='forward_backward',
                )
        def sparsity(x):
            return self.mu * np.sum(np.abs(x))

        def data_fidelity(x):
            return 0.5 * np.linalg.norm(self.fourier_op.op(x) - self.kspace_obs)**2

        def objective_cost(x):
            return data_fidelity(x) + sparsity(x)

        def nrmse(x):
            return np.linalg.norm(x - self.image) / np.mean(self.image)  

        metrics_ = {
            "cost": {"metric": objective_cost, "mapping": {"z_new": "x"}, "cst_kwargs": {}, "early_stopping": False},
            "nrmse": {"metric": nrmse, "mapping": {"x_new": "x"}, "cst_kwargs": {}, "early_stopping": False},
        }  

        self.opt = ForwardBackward(
        x=self.alpha_init,
        grad=self.gradient_op,
        prox=self.regularizer_op,
        cost=self.cost_op,
        auto_iterate=False,
        metric_call_period=1,
        metrics=metrics_,
        linear=self.linear_op,
        lambda_param=self.lambda_init,
        beta_param=self.beta_param,
        )

        self.run(1)

        

    def run(self, n_iter):

        
        #print('n_iter =', n_iter)
        self.opt.iterate(max_iter=n_iter)
        if hasattr(self.opt._grad, "linear_op"):
            self.x_final = self.opt._grad.linear_op.adj_op(self.opt.x_final)
        else:
            self.x_final = self.opt.x_final   

        
        #print(self.x_final.shape)

    def get_result(self):
        # The outputs of this function are the arguments of the
        # `compute` method of the objective.
        # They are customizable.
        return self.x_final