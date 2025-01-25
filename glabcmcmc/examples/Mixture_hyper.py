import numpy as np

from glabcmcmc.MCMCRunner import MCMCRunner
import glabcmcmc.distribution as distribution
import normflows as nf
from glabcmcmc.ESJD import esjd
from Mixture import Mixture_set
import torch
import time
import random

Model = Mixture_set(epsilon=0.05)
torch.manual_seed(0)
num_ite = 1000000
theta0 = torch.tensor([0.0, 0.0])
y0 = Model.generate_samples(theta0)
torch.manual_seed(0)
lp = distribution.DiagGaussian(2, loc=torch.zeros(1, 2), log_scale=torch.log(torch.tensor([0.35, 0.35])))
ip = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
gp = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
gp_base = nf.distributions.base.DiagGaussian(2)
runner = MCMCRunner(Model, output_dir='./')
seeds = [1,2,3,4,5,6,7,8,9,10]
global_frequencies = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
id = list(range(len(global_frequencies)))
resjd_value = [[0 for _ in id] for _ in range(len(seeds))]
num_ite2 = 1000
for i in range(len(seeds)):
    torch.manual_seed(seeds[i])
    for j in id:
        gf = global_frequencies[j]
        start_time = time.time()
        chain = runner.run_glmcmc(num_ite2, theta0, y0, gf, lp,
                ip, 5, output_file=None)
        end_time = time.time()
        time_mean =(end_time - start_time)/num_ite2
        resjd_value[i][j] = esjd(chain)/time_mean
resjd_mean = np.mean(resjd_value, axis=0)
best_gf = global_frequencies[np.argmax(resjd_mean)]
print('*****************************')
print(f"The best global frequency: {best_gf}")