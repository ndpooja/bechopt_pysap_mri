from benchopt import BaseObjective, safe_import_context

# Protect import to allow manipulating objective without importing library
# Useful for autocompletion and install commands
with safe_import_context() as import_ctx:
    import numpy as np
    from modopt.math.metrics import psnr, ssim


class Objective(BaseObjective):
    name = "MRI Reconstruction"


    def set_data(self, image, fourier_op):
        # The keyword arguments of this function are the keys of the `data`
        # dict in the `get_data` function of the dataset.
        # They are customizable.
        self.image = image
        self.fourier_op = fourier_op
        self.kspace_obs = self.fourier_op.op(image.data)
        self.mu = 0.8

    def compute(self, X):
        # The arguments of this function are the outputs of the
        # `get_result` method of the solver.
        # They are customizable.
        def sparsity(x):
            return self.mu * np.sum(np.abs(x))

        def data_fidelity(x):
            return 0.5 * np.linalg.norm(self.fourier_op.op(x) - self.kspace_obs)**2

        def NRMSE(x):
            return np.linalg.norm(self.image- x) / np.mean(self.image)

        x_final, x_ref = X
        cost = sparsity(x_final) + data_fidelity(x_final)
        nrmse = NRMSE(x_final)

        cost_ref = sparsity(x_ref) + data_fidelity(x_ref)
        nrmse_ref = NRMSE(x_ref)
        
        print(ssim(x_ref, self.image)) 

        return dict(value= np.log10(abs(cost-cost_ref)), value_cost=np.log10(abs(cost-cost_ref)), value_nrmse = abs(nrmse-nrmse_ref), value_ssim=ssim(x_final, self.image), value_psnr=psnr(x_final, self.image))

    def to_dict(self):
        # The output of this function are the keyword arguments
        # for the `set_objective` method of the solver.
        # They are customizable.
        return dict(image=self.image, fourier_op=self.fourier_op, kspace_obs=self.kspace_obs, mu=self.mu)
