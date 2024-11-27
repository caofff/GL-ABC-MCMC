from GLMCMC_NFs import GLMCMC_NF
import torch
import distribution
import normflows as nf
from Mixabs import Mixture_set
import multiprocessing
from globalMCMC_NFs import GlobalMCMC_NF
Model = Mixture_set(epsilon=0.05)
y_obs = torch.tensor([[1.5, 1.5]])
torch.manual_seed(0)
num_ite = 1000000
theta0 = torch.tensor([0.0, 0.0])
Initial_y = Model.generate_samples(theta0)
Local_Proposal = distribution.DiagGaussian(2, loc=torch.zeros(1, 2), log_scale=torch.log(torch.tensor([0.1, 0.1])))
base = nf.distributions.base.DiagGaussian(2)  # Uniform(2, torch.tensor([-30.0, -20.0]), torch.tensor([30.0, 10.0]))

def run_simulation(step_size, batch_size, train_step,filelocation):
    # 这个函数包装了您要并行运行的循环迭代的主要逻辑
    GLMCMC_NF(Model, y_obs, num_ite, theta0, Initial_y, Local_Proposal,
              filelocation,1, step_size,batch_size,base,train_step)

def run_simulation2(step_size, train_step, filelocation):
    # 这个函数包装了您要并行运行的循环迭代的主要逻辑
    GlobalMCMC_NF(Model, y_obs, num_ite, theta0, Initial_y, Local_Proposal,
                  filelocation, 1, step_size, base, train_step)

def main():
    torch.manual_seed(0)
    # 设置要运行的全局频率和批量大小的列表
    Step_sizes = [2000]
    Train_steps = [50]#, 500, 1000]
    # 准备进程列表
    processes = []

    folder = '/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/Mixabs_mu2/'
    # 循环创建进程
    # for step_size in Step_sizes:
    #     for train_step in Train_steps:
    #         filelocation = folder + "NF_GlobalMCMC_ss" + str(step_size) + 'ts' + str(train_step) + ".csv"
    #         print(filelocation)
    #         p = multiprocessing.Process(target=run_simulation2,
    #                                     args=(step_size, train_step, filelocation))
    #         processes.append(p)
    #         p.start()

    Step_sizes = [100]#, 500]
    Batch_sizes = [10]#, 11, 21]
    Train_steps = [50]#, 500, 1000]
    # 准备进程列表
    processes = []

    folder = '/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/Mixabs_mu2/'
    # 循环创建进程
    for step_size in Step_sizes:
        for batch_size in Batch_sizes:
            for train_step in Train_steps:
                # filelocation = folder + "NF_iSIR_ss" + str(step_size) + 'bs' + str(batch_size) + 'ts' + str(
                #     train_step) + ".csv"
                filelocation = folder + "xxxx.csv"
                print(filelocation)
                p = multiprocessing.Process(target=run_simulation,
                                            args=(step_size, batch_size, train_step, filelocation))
                processes.append(p)
                p.start()
    # 等待所有进程完成
    # filelocation = folder + "NF_iSIR_ss" + str(200) + 'bs' + str(5) + 'ts' + str(
    #     50) + ".csv"
    # print(filelocation)
    # p = multiprocessing.Process(target=run_simulation,
    #                             args=(200, 5, 50, filelocation))
    # processes.append(p)
    # p.start()
    for p in processes:
        p.join()
if __name__ == '__main__':
    main()
# Model = Mixture_set(epsilon=0.05)
# y_obs = torch.tensor([[0.5, 0.5]])
# torch.manual_seed(0)
# num_ite = 1000000
# theta0 = torch.tensor([0.0, 0.0])
# Initial_y = Model.generate_samples(theta0)
# Local_Proposal = distribution.DiagGaussian(2, loc=torch.zeros(1, 2), log_scale=torch.log(torch.tensor([0.1, 0.1])))
# filelocation = "./GLMCMC_NF/Mixture_ISIR_200_5_100.csv"
# base = nf.distributions.base.DiagGaussian(2)  # Uniform(2, torch.tensor([-30.0, -20.0]), torch.tensor([30.0, 10.0]))
# GLMCMC_NF(Model, y_obs, num_ite, theta0, Initial_y, Local_Proposal,
#           filelocation, 1, 200, 5, base,100)