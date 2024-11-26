import csv
import torch
import numpy as np

import normflows as nf
import distribution
from matplotlib import pyplot as plt
from tqdm import tqdm

def weight_sampling(w_list):
    ran = np.random.uniform(0, 1)
    s_wei = 0
    for j in range(len(w_list)):
        s_wei += w_list[j]
        if ran < s_wei:
            return j

def Local_proposal_forward(theta, grad_logABC_theta, tau):
    """
    Proposal distribution.

    Args:
    theta (torch.Tensor): The latent variable.
    grad_logABC (torch.Tensor): The gradient of the log-ABC.
    tau (float): The temperature.

    Returns:
    torch.Tensor: The proposal distribution.
    """
    if theta.ndim == 1:
        latent_size = int(theta.shape[0])
    else:
        _, latent_size = theta.shape
    pro = distribution.DiagGaussian(latent_size, loc=torch.tensor([0.0]), log_scale=torch.tensor([0.0]))
    z, log_pro = pro.forward(1)
    x = z*tau + theta + grad_logABC_theta * tau ** 2 / 2
    return x, log_pro


def log_proposal(theta_old, grad_logABC_theta_old, theta_prop, tau):
    if theta_old.ndim == 1:
        latent_size = int(theta_old.shape[0])
    else:
        _, latent_size = theta_old.shape
    pro = distribution.DiagGaussian(latent_size, loc=torch.tensor([0.0]), log_scale=torch.tensor([0.0]))
    re = pro.log_prob((theta_prop-theta_old-grad_logABC_theta_old*tau**2/2)/tau)
    return re


def GLMALA(ABCset,y_obs,num_ite,Initial_theta,Initial_y,tau,num_grad,
           filelocation,global_frequency=0,Global_Proposal=None,batch_size=None):
    # ABCset: the set of ABC samples
    # num_ite: the number of iterations
    # Initial_theta: the initial theta
    # Initial_y: the initial y
    # Local_Proposal: the local proposal
    # folderlocation: the folder location to save the results
    # global_frequency: the global frequency
    # Global_Proposal: the global proposal
    # batch_size: the batch size

    # Initialize theta and y
    print(filelocation)
    with open(filelocation, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入初始Theta
        writer.writerow(Initial_theta.view(-1).detach().numpy())
    Theta_old = Initial_theta.view(1, -1)
    y_old = Initial_y
    num_acc = 0
    grad_logABC_Theta_old = None
    local = True
    for i in tqdm(range(1, num_ite)):
        if torch.rand(1) < global_frequency:
            if local:
                log_prob_like = ABCset.calculate_log_kernel(y_old, y_obs)
                log_weight_old = (
                        ABCset.prior_log_prob(Theta_old) + log_prob_like - Global_Proposal.log_prob(
                    Theta_old)).view(-1)
            local = False
            Theta_prop0, Theta_prop_log_prob0 = Global_Proposal.forward(batch_size)
            has_nans = torch.isnan(Theta_prop0)
            no_nan_in_row = torch.all(~has_nans, dim=1)
            Theta_prop0 = Theta_prop0[no_nan_in_row].clone()
            Theta_prop_log_prob0 = Theta_prop_log_prob0[no_nan_in_row].clone()
            x = ABCset.generate_samples(Theta_prop0, 1)
            log_prob_like = ABCset.calculate_log_kernel(x, y_obs)
            # print(ABCset.discrepancy(x,y_obs))
            log_weight0 = ABCset.prior_log_prob(Theta_prop0) + log_prob_like - Theta_prop_log_prob0
            log_weight = torch.cat((log_weight_old, log_weight0))
            Theta_prop = torch.cat((Theta_old, Theta_prop0), dim=0)
            x = torch.cat((y_old.view(1, -1), x), dim=0)
            weight = torch.exp(log_weight)

            has_nans = torch.isnan(weight)
            weight[has_nans] = 0.0
            weight = weight / torch.sum(weight)
            ind = weight_sampling(weight.tolist())
            if ind is not None and ind != 0:
                Theta_old = Theta_prop[ind, :].clone().view(1, -1)
                log_weight_old = log_weight[ind].clone().view(-1)
                y_old = x[ind, :].clone()
                num_acc += 1
                # print('*******************', i, num_acc, num_acc / (i + 1))

        else:
            # print(grad_logABC_Theta_old is None)
            if grad_logABC_Theta_old is None:
                grad_logABC_Theta_old = ABCset.numberical_gradient_logABC(Theta_old,y_obs, num_grad)
                # print(grad_logABC_Theta_old)
            # print(i, num_acc / (i + 1),Theta_old, grad_logABC_Theta_old, tau)
            Theta_prop, Theta_prop_log_prob = Local_proposal_forward(Theta_old, grad_logABC_Theta_old, tau)
            # print(Theta_prop, Theta_prop_log_prob)
            grad_logABC_prop = ABCset.numberical_gradient_logABC(Theta_prop,y_obs, num_grad)
            y = ABCset.generate_samples(Theta_prop, 1)
            y = y[0,]
            # print('grad',grad_logABC_Theta_old,Theta_prop)
            log_acc = ABCset.prior_log_prob(Theta_prop) + ABCset.calculate_log_kernel(y,y_obs) + \
                      log_proposal(Theta_prop, grad_logABC_prop, Theta_old, tau) - \
                      ABCset.prior_log_prob(Theta_old.view(1, -1)) - ABCset.calculate_log_kernel(y_old,y_obs) - \
                      Theta_prop_log_prob
            # print(ABCset.prior_log_prob(Theta_prop),ABCset.calculate_log_kernel(y,y_obs) ,
            #           log_proposal(Theta_prop, grad_logABC_prop, Theta_old, tau) ,
            #           ABCset.prior_log_prob(Theta_old.view(1, -1)) , ABCset.calculate_log_kernel(y_old,y_obs) ,
            #           Theta_prop_log_prob)
            log_w = torch.log(torch.rand(1))
            # print('----------------',log_w, log_acc)
            if log_w < log_acc:
                num_acc += 1
                Theta_old = Theta_prop
                y_old = y
                grad_logABC_Theta_old = grad_logABC_prop
                # print('**********************',i, num_acc / (i + 1))
        with open(filelocation, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(Theta_old.view(-1).detach().numpy())

if __name__ == "__main__":
    import torch
    import distribution
    from rgammaSDE5 import SDE_set
    torch.manual_seed(4)
    np.random.seed(4)
    Model = SDE_set(epsilon=0.15)
    theta_t = torch.tensor([1.0, 1.0 / 2.0, 1.0 / 10.0, np.pi / 5.0, 1.0 / 100])
    y_obs = Model.generate_samples(theta_t)
    num_ite = 50000
    theta0 = torch.tensor([1.0, 1.0 / 2.0, 1.0 / 10.0, np.pi / 5.0, 1.0 / 100])
    Initial_y = Model.generate_samples(theta0)
    Global_Proposal = distribution.Gamma(torch.tensor([5.0, 3.0, 5.0, 5.0, 2.0]),
                                         torch.tensor([1.0, 5.0, 15.0, 10.0, 15.0]))
    Local_Proposal = distribution.DiagGaussian(5, loc=torch.zeros(1, 5),
                                               log_scale=torch.log(torch.tensor([0.2, 0.2, 0.1, 0.3, 0.05])))
    tau = 0.05
    num_grad = 5
    filelocation = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/Result/SDE5/xxxxx2.csv"
    GLMALA(Model, y_obs, num_ite, theta0, Initial_y, tau, num_grad, filelocation, global_frequency=0,
               Global_Proposal=Global_Proposal, batch_size=0)



    # from Mixabs import Mixture_set
    # import time
    # Model = Mixture_set(epsilon=0.05)
    # y_obs = torch.tensor([[0.0, 0.0]])
    # torch.manual_seed(0)
    # num_ite = 1000
    # theta0 = torch.tensor([0, 0])
    # Initial_y = Model.generate_samples(theta0)
    # Local_Proposal = None
    # epsilon2 = 0.05
    # Global_Proposal = distribution.Uniform(2, torch.tensor([-5.0, -5.0]), torch.tensor([5.0, 5.0]))
    # folder = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/xxxxx"
    # for Batch_size in [5]:
    #     tau = 0.1
    #     num_grad = 5
    #     start_time = time.time()  # 任务开始时间
    #     filelocation = folder + "iSIRUniform_bs" + str(Batch_size) + ".csv"
    #     GLMALA(Model, y_obs, num_ite, theta0, Initial_y, tau, num_grad, filelocation, global_frequency=0,
    #            Global_Proposal=Global_Proposal, batch_size=Batch_size)
    #     end_time = time.time()  # 任务结束时间
    #     elapsed_time0 = end_time - start_time  # 经过的时间
    #
    #     start_time = time.time()  # 任务开始时间
    #     filelocation = folder + "iSIRUniform_bs" + str(Batch_size) + ".csv"
    #     GLMALA(Model, y_obs, num_ite, theta0, Initial_y, tau, num_grad, filelocation, global_frequency=0.5,
    #            Global_Proposal=Global_Proposal, batch_size=Batch_size)
    #     end_time = time.time()  # 任务结束时间
    #     elapsed_time5 = end_time - start_time  # 经过的时间
    #
    #     start_time = time.time()  # 任务开始时间
    #     filelocation = folder + "iSIRUniform_bs" + str(Batch_size) + ".csv"
    #     GLMALA(Model, y_obs, num_ite, theta0, Initial_y, tau, num_grad, filelocation, global_frequency=1,
    #            Global_Proposal=Global_Proposal, batch_size=Batch_size)
    #     end_time = time.time()  # 任务结束时间
    #     elapsed_time10 = end_time - start_time  # 经过的时间
    #     print(f"gf0: {elapsed_time0} seconds")
    #     print(f"gf0.5: {elapsed_time5} seconds")
    #     print(f"gf1: {elapsed_time10} seconds")
    # Global_Proposal = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.tensor([np.log(0.5), np.log(0.5)]))
    # folder = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/Mixture"
    # for Batch_size in [1, 10, 50]:
    #     filelocation = folder + "iSIRGaussian_bs" + str(Batch_size) + ".csv"
    #     GLMCMC(Model, y_obs, num_ite, theta0, Initial_y, Local_Proposal, filelocation, global_frequency=1,
    #            Global_Proposal=Global_Proposal, batch_size=Batch_size)
    #
    # # from GlobalMCMC import GlobalMCMC
    # # GlobalMCMC(Model, y_obs, num_ite, theta0, Initial_y, Global_Proposal, filelocation)
    # Pro_loc = torch.tensor([[0.7 / (1 + epsilon2 ** 2 + 0.01), 0.7 / (1 + epsilon2 ** 2 + 0.01)],
    #                         [-0.7 / (1 + epsilon2 ** 2 + 0.01), 0.7 / (1 + epsilon2 ** 2 + 0.01)],
    #                         [0.7 / (1 + epsilon2 ** 2 + 0.01), -0.7 / (1 + epsilon2 ** 2 + 0.01)],
    #                         [-0.7 / (1 + epsilon2 ** 2 + 0.01), -0.7 / (1 + epsilon2 ** 2 + 0.01)]])
    # Pro_scale = torch.tensor([[1 / (1 + 1 / (epsilon2 ** 2 + 0.01)) ** (1 / 2)] * 2] * 4)
    # Global_Proposal = distribution.GaussianMixture(4, 2, Pro_loc.tolist(), Pro_scale.tolist())

    # folder = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/Mixture"
    # for Batch_size in [5, 11, 21, 31, 51, 81, 101]:
    #     filelocation = folder + "iSIR_bs" + str(Batch_size) + ".csv"
    #     GLMCMC(Model, y_obs, num_ite, theta0, Initial_y, Local_Proposal, filelocation, global_frequency=1,
    #            Global_Proposal=Global_Proposal, batch_size=Batch_size)

    # for Batch_size in [5, 11, 21, 31, 51, 81, 101]:
    #     filelocation = folder + "iSIRUD_bs" + str(Batch_size) + ".csv"
    #     GLMCMC_UD(Model, y_obs, num_ite, theta0, Initial_y, None, filelocation, global_frequency=1,
    #               range_min=torch.tensor([-5.0, -5.0]), range_max=torch.tensor([5.0, 5.0]), batch_size=Batch_size)
    # #
