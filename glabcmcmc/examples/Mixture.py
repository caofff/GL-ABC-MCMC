import torch
import numpy as np
import glabcmcmc.distribution as distribution

class Mixture_set:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.theta_dim = 2
        self.y_obs = torch.tensor([[1.5, 1.5]])
        self.y_dim = self.y_obs.shape[1]
        # Initialize other necessary variables or constants here

    def generate_samples(self, theta, num_samples=1):
        # cov_matrix = torch.eye(2, dtype=torch.float32) * 0.05
        if theta.dim() == 1:
            num_theta = 1
        else:
            num_theta, dim_theta = theta.shape
        likelihood = distribution.DiagGaussian(self.theta_dim, torch.tensor([0.0, 0.0]), torch.log(torch.tensor([0.05, 0.05]).sqrt()))
        if num_theta == 1:
            samples = torch.abs(theta) + likelihood.sample(num_samples)
        elif num_samples == 1:
            samples = torch.abs(theta) + likelihood.sample(num_theta)
        else:
            samples = torch.abs(theta).unsqueeze(1).repeat(1, num_samples, 1) + likelihood.sample(num_samples * num_theta).view(num_theta, num_samples, dim_theta)
        return samples

    def prior_log_prob(self, samples):
        samples = samples.view(-1, self.theta_dim)
        Priord = distribution.DiagGaussian(self.theta_dim, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
        return Priord.log_prob(samples)

    def discrepancy(self, y):
        y = y.view(-1, self.y_dim)
        self.y_obs = self.y_obs.view(-1, self.y_dim)
        return torch.sqrt(torch.sum((y - self.y_obs) ** 2, dim=1))

    def calculate_log_kernel(self, y):
        dis = self.discrepancy(y)
        log_value = distribution.DiagGaussian(1, loc=torch.tensor([0.0]),
                                              log_scale=torch.log(torch.tensor([self.epsilon]))).log_prob(
            dis.view(-1, 1))
        return log_value

# Example usage
if __name__ == "__main__":
    from glabcmcmc.MCMCRunner import MCMCRunner
    import glabcmcmc.distribution as distribution
    import normflows as nf
    from glabcmcmc.ESJD import esjd
    Model = Mixture_set(epsilon=0.05)
    torch.manual_seed(0)
    np.random.seed(0)
    num_ite = 1000000
    theta0 = torch.tensor([0.0, 0.0])
    y0 = Model.generate_samples(theta0)
    lp = distribution.DiagGaussian(2, loc=torch.zeros(1, 2), log_scale=torch.log(torch.tensor([0.35, 0.35])))
    ip = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
    gp = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
    gp_base = nf.distributions.base.DiagGaussian(2)
    runner = MCMCRunner(Model, output_dir='./')
    # chain_global = runner.run_global_mcmc(num_ite, theta0, y0, 0.5, lp, gp, output_file='global_mcmc_results.csv')
    chain_glmcmc = runner.run_glmcmc(num_ite, theta0, y0, 0.9, lp, ip, 5, output_file='glmcmc_results.csv')
    print(esjd(chain_glmcmc))
    # chain_glmala =runner.run_glmala(num_ite, theta0, y0, 0.8, ip, 5, 0.3, 100, output_file='glmala_results.csv')
    # chain_glmcmc_nf =runner.run_glmcmc_nf(num_ite, 0.5, lp, 5, gp_base, 200, 50, output_file='glmcmc_nf_results.csv')
