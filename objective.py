from benchopt import BaseObjective, safe_import_context

# Protect import to allow manipulating objective without importing library
# Useful for autocompletion and install commands
with safe_import_context() as import_ctx:
    import numpy as np
    from mri.operators import NonCartesianFFT

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



    def compute(self, x_final):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        mu = 1e-4
        def sparsity(x):
            return mu * np.sum(np.abs(x))

        def data_fidelity(x):
            return 0.5 * np.linalg.norm(self.fourier_op.op(x) - self.kspace_obs)**2

        def objective_cost(x):
            return data_fidelity(x) + sparsity(x)

        cost = objective_cost(x_final)
        
        return dict(value=cost)

    def to_dict(self):
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver.
        # They are customizable.
        return dict(image= self.image, mask = self.mask, fourier_op = self.fourier_op, kspace_obs = self.kspace_obs)
