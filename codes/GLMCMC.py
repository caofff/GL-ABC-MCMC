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


def GLMCMC(ABCset,y_obs,num_ite,Initial_theta,Initial_y,Local_Proposal,
           filelocation,global_frequency=0,Global_Proposal=None,batch_size=None):



    # ABCset: the set of ABC samples
    # num_ite: the number of iterations
    # Initial_theta: the initial theta
    # Initial_y: the initial y
    # Local_Proposal: the local proposal
    # folderlocation: the folder location to save the results
    # global_frequency: the global frequency
    # Global_Proposal: the global proposal distribution
    # batch_size: the batch size of ABC-i-SIR

    # Initialize theta and y
    print(filelocation,Initial_theta.view(-1))

    with open(filelocation, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入初始Theta
        writer.writerow(Initial_theta.view(-1).detach().numpy())
    Theta_old = Initial_theta.view(1, -1)
    y_old = Initial_y
    local = True
    num_acc = 0
    log_prob_like = ABCset.calculate_log_kernel(y_old, y_obs)
    log_weight_old = (
            ABCset.prior_log_prob(Theta_old) + log_prob_like - Global_Proposal.log_prob(
        Theta_old)).view(-1)
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
            x = torch.cat((y_old.view(1,-1), x), dim=0)
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
            Theta_prop = Local_Proposal.sample(1) + Theta_old
            while ABCset.prior_log_prob(Theta_prop) == 7*torch.tensor(10**(-10)).log():
                Theta_prop = Local_Proposal.sample(1) + Theta_old
            y = ABCset.generate_samples(Theta_prop, 1)
            y = y[0,].clone()
            log_acc = ABCset.prior_log_prob(Theta_prop) + ABCset.calculate_log_kernel(y, y_obs) - \
                      ABCset.prior_log_prob(Theta_old) - ABCset.calculate_log_kernel(y_old,
                                                                                     y_obs)
            log_w = torch.log(torch.rand(1))
            if log_w < log_acc:
                local = True
                num_acc += 1
                Theta_old = Theta_prop.clone()
                y_old = y.clone()
                # print(i,num_acc, num_acc / (i + 1))
        with open(filelocation, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(Theta_old.view(-1).detach().numpy())

# Example usage
if __name__ == "__main__":
    # from Banana import Banana_set
    # # 创建一个具有某些观测值和epsilon值的ABCset对象
    # Model = Banana_set(epsilon=0.05)
    # torch.manual_seed(0)
    # y_obs = torch.tensor([[0.0, 0.0]])
    # theta0 = Model.prior_sample(1)
    # Initial_y = Model.generate_samples(theta0)
    # Local_Proposal = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
    # filelocation = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/Banana/GLMCMC0.csv"
    # GLMCMC(Model,y_obs, 1000, theta0, Initial_y, Local_Proposal, filelocation, global_frequency=0,
    #        Global_Proposal=None, batch_size=None)
    # print(Model.prior_log_prob(theta0))
    # # 使用abc_instance对象调用generate_samples方法
    # y_samples = Model.generate_samples(theta0)
    # print(y_samples)
    # log_kernel = Model.calculate_log_kernel(y_samples, y_obs)
    # print(Model.discrepancy(y_samples, y_obs))
    # print(torch.exp(log_kernel))

    import torch
    from GLMCMC import GLMCMC
    import distribution
    from GLMCMC_UD import GLMCMC_UD
    from Mixture import Mixture_set
    Model = Mixture_set(epsilon=0.05)
    y_obs = torch.tensor([[0.0, 0.0]])
    torch.manual_seed(0)
    num_ite = 10
    theta0 = torch.tensor([0, 0])
    Initial_y = Model.generate_samples(theta0)
    Local_Proposal = None
    epsilon2 = 0.05
    Global_Proposal = distribution.Uniform(2, torch.tensor([-5.0, -5.0]), torch.tensor([5.0, 5.0]))
    folder = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/xxxxx"
    for Batch_size in [1]:
        filelocation = folder + "iSIRUniform_bs" + str(Batch_size) + ".csv"
        GLMCMC(Model, y_obs, num_ite, theta0, Initial_y, Local_Proposal, filelocation, global_frequency=1,
               Global_Proposal=Global_Proposal, batch_size=Batch_size)

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
