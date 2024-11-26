from GLMCMC import GLMCMC
import torch
import distribution
from Mixabs import Mixture_set
import multiprocessing
from GlobalMCMC import GlobalMCMC
import time
import csv

import numpy as np
Model = Mixture_set(epsilon=0.05)
y_obs = torch.tensor([[1.5, 1.5]])
torch.manual_seed(0)
num_ite = 1000000
theta0 = torch.tensor([0.0, 0.0])
Initial_y = Model.generate_samples(theta0)
Local_Proposal = distribution.DiagGaussian(2, loc=torch.zeros(1, 2), log_scale=torch.log(torch.tensor([0.2, 0.2])))
def run_simulation(Initial_theta0, Initial_y0,filelocation,global_frequency, Global_Proposal, batch_size):
    GLMCMC(Model, y_obs, num_ite, Initial_theta0, Initial_y0, Local_Proposal, filelocation, global_frequency,
           Global_Proposal, batch_size)

def run_simulation2(Initial_theta0, Initial_y0, filelocation,Proposal,global_frequency=1):
    # 这个函数包装了您要并行运行的循环迭代的主要逻辑
    GlobalMCMC(Model, y_obs, num_ite, Initial_theta0, Initial_y0,  Proposal,
               filelocation,global_frequency,Local_Proposal)

def main():
    torch.manual_seed(0)
    Global_Proposal = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.log(
        torch.tensor([1.0, 1.0])))
    folder = "./Mixabs/"
    seeds = [1,2,3,4,5,6,7,8,9,10]
    batch_size = [2,5]
    for seed in seeds:
        torch.manual_seed(seed)
        Initial_theta0 = Model.prior_sample(1, ).view(-1)
        Initial_y0 = Model.generate_samples(Initial_theta0)
        for bs in batch_size:
            processes = []
            for gf in [0.1, 0.2,0.5, 1]:
                filelocation = folder + 'GLMCMC_bs' + str(bs) +'_gf' + str(gf) + '_' + str(seed) + ".csv"
                p = multiprocessing.Process(target=run_simulation,
                                        args=(Initial_theta0, Initial_y0, filelocation,gf,Global_Proposal,
                                              bs))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()

if __name__ == '__main__':
    main()




