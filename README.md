# GL-ABC-MCMC

GL-ABC-MCMC is a Python package, whichoffers a variety of GL-ABC-MCMC sampling methods, including the usage of a distribution directly as a proposal, employing iterative sampling importance resampling (iSIR) to construct the global proposal, utilizing the gradient-based Metropolis-Adjusted Langevin algorithm (MALA) as a local proposal, and enhancing the global proposal with normalizing flows. Additionally, the package provides functionality to evaluate the expected square jump distance (ESJD), a criterion that used to select appropriate hyperparameters. 


## Key Features

- **Multiple MCMC Sampling Algorithms**:
  - `GlobalMCMC`: This combines parametric global and local proposal distributions.
  - `GLMCMC`: The global proposal of `GlobalMCMC` is replaced with one constructed using iSIR.
  - `GLMALA`: This version uses MALA as the local proposal in the  `GLMCMC` algorithm.
  -`GLMCMC-NFs`: Building upon GLMCMC, this implementation utilizes normalizing flows to enhance the global proposal distribution. 

- **Built-in Probability Distributions**:
  - Uniform distribution
  - Gamma distribution
  - Diagonal Gaussian distribution
  - Gaussian Mixture distribution

- **Utility Tools**:
  - MCMCRunner for easy execution of sampling algorithms
  - ESJD (Expected Squared Jumping Distance) calculation for MCMC diagnostics

## Installation

You can install the package using pip:

```bash
pip install glabcmcmc
```

Or install from source:

```bash
git clone https://github.com/caofff/GL-ABC-MCMC.git
cd GL-ABC-MCMC
conda activate pytorch
pip install -e .
```


## Dependencies

- Python >= 3.8
- PyTorch >= 1.12.1
- NumPy < 2
- tqdm >= 4.64.0
- normflows >= 1.7.2
- scipy >= 1.8.1
- statsmodels >= 0.14.4
- pandas>=2.2.2
- matplotlib>=3.8.2
- seaborn>=0.13.2

## Quick Start


Here's a simple example of how to use the package:

```python
import torch
import numpy as np
import glabcmcmc.distribution as distribution
# Define your ABC set
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

from glabcmcmc import MCMCRunner
# Initialize problem
Model = Mixture_set(epsilon=0.05)
theta0 = torch.tensor([0.0, 0.0])
y0 = Model.generate_samples(theta0)
#create MCMCRunner
runner = MCMCRunner(Model, output_dir='./')

## Run different methods
import glabcmcmc.distribution as distribution
import normflows as nf
from glabcmcmc import esjd

num_ite = 1000000
lp = distribution.DiagGaussian(2, loc=torch.zeros(1, 2), log_scale=torch.log(torch.tensor([0.35, 0.35])))
gp = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
chain_global = runner.run_global_mcmc(num_ite, theta0, y0, 0.5, lp, gp, output_file='global_mcmc_results.csv')


ip = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
chain_glmcmc = runner.run_glmcmc(num_ite, theta0, y0, 0.9, lp, ip, 5, output_file='glmcmc_results.csv')
    
chain_glmala =runner.run_glmala(num_ite, theta0, y0, 0.8, ip, 5, 0.3, 100, output_file='glmala_results.csv')

gp_base = nf.distributions.base.DiagGaussian(2)
chain_glmcmc_nf =runner.run_glmcmc_nf(num_ite, 0.5, lp, 5, gp_base, 200, 50, output_file='glmcmc_nf_results.csv')


## Calculate ESJD
esjd_global = esjd(chain_global)
```

Or run the code in the terminal
```bash
cd glabcmcmc/examples
python3 Mixture_hyper.py
python3 Mixture.py
python3 plot.py
```

## Documentation

Check the `examples` directory for detailed usage examples and tutorials.

## Author

- **Xuefei Cao, Shijia Wang, and Yongdao Zhou**
- Email: xuefeic@mail.nankai.edu.cn, Wangshj1@shanghaitech.edu.cn, and ydzhou@nankai.edu.cn.
- GitHub: [caofff/GL-ABC-MCMC](https://github.com/caofff/GL-ABC-MCMC)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{glabcmcmc,
    title = {glabcmcmc: A Python package for ABC-MCMC with local and global moves},
    author = {Xuefei Cao, Shijia Wang, and Yongdao Zhou},
    year = {2025},
    url = {https://github.com/caofff/GL-ABC-MCMC}
}
```

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
