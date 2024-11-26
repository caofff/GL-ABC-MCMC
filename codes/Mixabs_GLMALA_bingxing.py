from GLMALA import GLMALA
import distribution
from Mixabs import Mixture_set
import torch
import numpy as np
import multiprocessing

Model = Mixture_set(epsilon=0.05)
y_obs = torch.tensor([[1.5, 1.5]])
torch.manual_seed(0)
num_ite = 1000000
theta0 = torch.tensor([0.0, 0.0])
Initial_y = Model.generate_samples(theta0)
Local_Proposal = distribution.DiagGaussian(2, loc=torch.zeros(1, 2), log_scale=torch.log(torch.tensor([0.2, 0.2])))
Global_Proposal = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.log(
    torch.tensor([1.0, 1.0])))
def run_simulation(theta0, Initial_y, tau, num_grad, filelocation,gf=0,bs=0):
    # 这个函数包装了您要并行运行的循环迭代的主要逻辑
    GLMALA(Model, y_obs, num_ite, theta0, Initial_y, tau, num_grad, filelocation, global_frequency=gf,
           Global_Proposal=Global_Proposal, batch_size=bs)

if __name__ == '__main__':
    # 设置要运行的全局频率和批量大小的列表

    # # 循环创建进程
    # for seed in seeds:
    #     torch.manual_seed(seed)
    #     np.random.seed(seed)
    #     Initial_theta0 = theta0 + Local_Proposal.sample(1)
    #     Initial_y0 = Model.generate_samples(Initial_theta0)
    #     processes = []
    #     num_grad = 100
    #     gf = 0
    #     for tau in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    #         for bs in [0]:
    #             filelocation = folder + "tau/GLMALA_tau" + str(tau) + '_num' + str(num_grad) +'_seed' + str(seed) + ".csv"
    #             p = multiprocessing.Process(target=run_simulation,
    #                                         args=(Initial_theta0, Initial_y0, tau, num_grad, filelocation, gf, bs))
    #             processes.append(p)
    #             p.start()
    #     for p in processes:
    #         p.join()
    # # # 等待所有进程完成


    seeds = [1,2,3,4,5,6,7,8,9,10]

    # 准备进程列表
    folder = "./Mixabs/"
    # 循环创建进程
    for gf in [0.6,0.8,1]:
        for bs in [5,10]:
            processes = []
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                Initial_theta0 = theta0 + Local_Proposal.sample(1)
                Initial_y0 = Model.generate_samples(Initial_theta0)
                tau = 0.3
                num_grad = 100
                filelocation = folder + "GLMCMC_xiuzheng/GLMALA_tau" + str(tau) + '_num' + str(num_grad) + '_bs' + str(
                    bs) + '_gf' + str(gf) + '_seed' + str(seed) + ".csv"
                p = multiprocessing.Process(target=run_simulation, args=(Initial_theta0, Initial_y0, tau, num_grad, filelocation,gf,bs))
                processes.append(p)
                p.start()
            for p in processes:
                p.join()
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    processes = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        Initial_theta0 = theta0 + Local_Proposal.sample(1)
        Initial_y0 = Model.generate_samples(Initial_theta0)
        tau = 0.3
        num_grad = 100
        for gf in [0]:
            for bs in [0]:
                if gf==0:
                    filelocation = folder + "GLMCMC_xiuzheng/MALA_tau" + str(tau) + '_num' + str(num_grad) + '_seed' + str(seed) + ".csv"
                else:
                    filelocation = folder + "GLMCMC_xiuzheng/GLMALA_tau" + str(tau) + '_num' + str(num_grad) + '_bs' + str(
                        bs) + '_gf' + str(gf) + '_seed' + str(seed) + ".csv"
                p = multiprocessing.Process(target=run_simulation, args=(Initial_theta0, Initial_y0, tau, num_grad, filelocation,gf,bs))
                processes.append(p)
                p.start()
    for p in processes:
        p.join()



