import torch
from tqdm import tqdm
import numpy as np
import csv
import distribution

def weight_sampling(w_list):
    ran = np.random.uniform(0, 1)
    s_wei = 0
    for j in range(len(w_list)):
        s_wei += w_list[j]
        if ran < s_wei:
            return j


def GlobalMCMC(ABCset,y_obs,num_ite,Initial_theta,Initial_y,
               Global_Proposal,filelocation,global_frequency=1,Local_Proposal=None):
    # ABCset: the set of ABC samples
    # num_ite: the number of iterations
    # Initial_theta: the initial theta
    # Initial_y: the initial y
    # Local_Proposal: the local proposal
    # folderlocation: the folder location to save the results
    # global_frequency: the global frequency
    # Global_Proposal: the global proposal
    # batch_size: the batch size
    print(filelocation)
    with open(filelocation, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入初始Theta
        writer.writerow(Initial_theta.detach().numpy())
    Theta_old = Initial_theta.view(1, -1)
    y_old = Initial_y
    num_acc = 0
    for i in tqdm(range(1, num_ite)):
        if torch.rand(1) < global_frequency:
            Theta_prop, prop_log_prob = Global_Proposal.forward(1)
            y = ABCset.generate_samples(Theta_prop, 1)
            y = y.view(1, -1)
            log_acc = ABCset.prior_log_prob(Theta_prop) + ABCset.calculate_log_kernel(y, y_obs) + \
                      Global_Proposal.log_prob(Theta_old.view(1, -1)) - prop_log_prob - \
                      ABCset.prior_log_prob(Theta_old.view(1, -1)) - ABCset.calculate_log_kernel(y_old, y_obs)
            log_w = torch.log(torch.rand(1))
            if log_w < log_acc:
                num_acc += 1
                Theta_old = Theta_prop.clone()
                y_old = y.clone()
                # print(Theta_Re[i, :], num_acc / (i + 1))

        else:
            Theta_prop = Local_Proposal.sample(1) + Theta_old
            y = ABCset.generate_samples(Theta_prop, 1)
            y = y[0,].clone()
            log_acc = ABCset.prior_log_prob(Theta_prop) + ABCset.calculate_log_kernel(y, y_obs) - \
                      ABCset.prior_log_prob(Theta_old) - ABCset.calculate_log_kernel(y_old,
                                                                                     y_obs)
            log_w = torch.log(torch.rand(1))
            if log_w < log_acc:
                num_acc += 1
                Theta_old = Theta_prop.clone()
                y_old = y.clone()
                # print(Theta_old, num_acc / (i + 1))
        with open(filelocation, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(Theta_old.view(-1).detach().numpy())



# Example usage
if __name__ == "__main__":
    from Mixture import Mixture_set
    from GLMCMC import GLMCMC
    Model = Mixture_set(epsilon=0.05)
    y_obs = torch.tensor([[0.0, 0.0]])
    torch.manual_seed(0)
    num_ite = 1000000
    theta0 = torch.tensor([0, 0])
    Initial_y = Model.generate_samples(theta0)
    Local_Proposal = None
    epsilon2 = 0.05
    # Pro_loc = torch.tensor([[0.5 / (1 + epsilon2 ** 2 + 0.01), 0.5 / (1 + epsilon2 ** 2 + 0.01)],
    #                         [-0.5 / (1 + epsilon2 ** 2 + 0.01), 0.5 / (1 + epsilon2 ** 2 + 0.01)],
    #                         [0.5 / (1 + epsilon2 ** 2 + 0.01), -0.5 / (1 + epsilon2 ** 2 + 0.01)],
    #                         [-0.5 / (1 + epsilon2 ** 2 + 0.01), -0.5 / (1 + epsilon2 ** 2 + 0.01)]])
    # Pro_scale = torch.tensor([[0.116] * 2] * 4)
    # Global_Proposal = distribution.GaussianMixture(4, 2, Pro_loc.tolist(), Pro_scale.tolist())
    # filelocation = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/MixtureglobalMCMC_optimal.csv"
    # print(filelocation, '\n')
    # GlobalMCMC(Model, y_obs, num_ite, theta0, Initial_y, Global_Proposal, filelocation)
    # folder = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/Mixture"
    # for Batch_size in [1, 10, 50]:
    #     filelocation = folder + "iSIROptimal_bs" + str(Batch_size) + ".csv"
    #     print(filelocation, '\n')
    #     GLMCMC(Model, y_obs, num_ite, theta0, Initial_y, Local_Proposal, filelocation, global_frequency=1,
    #            Global_Proposal=Global_Proposal, batch_size=Batch_size)
    #
    Global_Proposal = distribution.Uniform(2, torch.tensor([-3.0, -3.0]), torch.tensor([3.0, 3.0]))
    filelocation = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/MixtureglobalMCMC3_Uniform.csv"
    print(filelocation, '\n')
    GlobalMCMC(Model, y_obs, num_ite, theta0, Initial_y, Global_Proposal, filelocation)
    folder = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/Mixture"
    for Batch_size in [1, 10, 50]:
        filelocation = folder + "iSIRUniform3_bs" + str(Batch_size) + ".csv"
        GLMCMC(Model, y_obs, num_ite, theta0, Initial_y, Local_Proposal, filelocation, global_frequency=1,
               Global_Proposal=Global_Proposal, batch_size=Batch_size)

    # Global_Proposal = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.tensor([np.log(), np.log(0.5)]))
    # filelocation = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/MixtureglobalMCMC_Gaussian5.csv"
    # print(filelocation, '\n')
    # GlobalMCMC(Model, y_obs, num_ite, theta0, Initial_y, Global_Proposal, filelocation)
    # folder = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/Mixture"
    # for Batch_size in [1, 10, 50]:
    #     filelocation = folder + "iSIRGaussian5_bs" + str(Batch_size) + ".csv"
    #     GLMCMC(Model, y_obs, num_ite, theta0, Initial_y, Local_Proposal, filelocation, global_frequency=1,
    #            Global_Proposal=Global_Proposal, batch_size=Batch_size)