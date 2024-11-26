import csv
import torch
import numpy as np
import secrets
import normflows as nf
import distribution
from matplotlib import pyplot as plt
from tqdm import tqdm


# 模块化函数
def generate_samples(theta, num_samples):
    cov_matrix = torch.eye(2, dtype=torch.float32) * 0.01
    weights = torch.ones(4, dtype=torch.float32) / 4.0
    location = torch.tensor([[-0.5, 0.5], [-0.5, -0.5], [0.5, 0.5], [0.5, -0.5]])
    if theta.dim() == 1:
        num_theta = 1
    else:
        num_theta, dim_theta = theta.shape
    likelihood = torch.distributions.MultivariateNormal(torch.tensor([0.0, 0.0]), cov_matrix)
    if num_theta == 1:
        mode = torch.multinomial(weights, num_samples, replacement=True)
        samples = theta + location[mode, ] + likelihood.sample((num_samples,))
    elif num_samples == 1:
        mode = torch.multinomial(weights, num_theta, replacement=True)
        samples = theta + location[mode, ] + likelihood.sample((num_theta,))
    else:
        samples = torch.empty((num_theta, num_samples, dim_theta))
        for k in range(num_theta):
            mode = torch.multinomial(weights, num_samples, replacement=True)
            samples[k, :, :] = theta[k, :] + location[mode, ] + likelihood.sample((num_samples, ))
    return samples


def calculate_log_kernel(xx):
    cov_matrix = torch.eye(2, dtype=torch.float32) * epsilon ** 2
    log_value = torch.distributions.MultivariateNormal(y_obs, cov_matrix).log_prob(xx)
    return log_value


def weight_sampling(w_list):
    ran = np.random.uniform(0, 1)
    s_wei = 0
    for j in range(len(w_list)):
        s_wei += w_list[j]
        if ran < s_wei:
            return j


def numerical_gradient_logABC(x, num, d=1e-10):
    grad = torch.zeros_like(x)
    if x.ndim == 1:
        x = x.view(1, -1)
    x_num, x_dim = x.shape
    xi_plus = x.clone()[:, None, :].repeat(1, x_dim, 1)
    xi_minus = x.clone()[:, None, :].repeat(1, x_dim, 1)
    for j in range(x_dim):
        xi_plus[:, j, j] += d
        xi_minus[:, j, j] -= d
    xi_plus = xi_plus.view(-1, x_dim)
    xi_minus = xi_minus.view(-1, x_dim)
    random_seeds_int = torch.tensor([secrets.randbelow(2 ** 32) for _ in range(num)], dtype=torch.long)
    pi_plus = torch.zeros(x_num, x_dim)
    pi_minus = torch.zeros(x_num, x_dim)
    for k in range(num):
        for xi, fx in zip([xi_plus, xi_minus], [pi_plus, pi_minus]):
            torch.manual_seed(random_seeds_int[k])
            y_theta = generate_samples(xi, 1).view(-1, 2)
            fx += torch.exp(calculate_log_kernel(y_theta))
    grad += ((torch.log(pi_plus/torch.tensor([num])) - torch.log(pi_minus/torch.tensor([num])))/(2 * d)).squeeze()
    return grad-x


def proposal(theta, grad_logABC_theta, tau):
    """
    Proposal distribution.

    Args:
    theta (torch.Tensor): The latent variable.
    grad_logABC (torch.Tensor): The gradient of the log-ABC.
    tau (float): The temperature.

    Returns:
    torch.Tensor: The proposal distribution.
    """
    pro = distribution.DiagGaussian(latent_size, loc=torch.tensor([0.0]), log_scale=torch.tensor([0.0]))
    z, log_pro = pro.forward(1)
    x = z*tau + theta + grad_logABC_theta * (tau ** 2) / 2
    return x, log_pro


def log_proposal(theta_old, grad_logABC_theta_old, theta_prop, tau):
    pro = distribution.DiagGaussian(latent_size, loc=torch.tensor([0.0]), log_scale=torch.tensor([0.0]))
    re = pro.log_prob((theta_prop-theta_old-grad_logABC_theta_old*(tau**2)/2)/tau)
    return re


latent_size = 2
Proposal_G = distribution.Uniform(2, -1.5, 1.5)
Prior = distribution.DiagGaussian(2, loc=torch.zeros(1, 2), log_scale=torch.log(torch.tensor([1.0, 1.0])))
Local_proposal = distribution.DiagGaussian(2, loc=torch.zeros(1, 2), log_scale=torch.log(torch.tensor([0.2, 0.2])))
y_obs = torch.tensor([[0, 0]])
epsilon = 0.05
num_ite = 100000
batch_size = 20
Initial = torch.tensor([0, 0])
Theta_Re = torch.empty(num_ite+1, 2)
y_Re = torch.empty(num_ite+1, 2)
Theta_Re[0, :] = Initial
y_old = generate_samples(Theta_Re[0, :], 1)
tau = 0.2
grad_logABC_old = numerical_gradient_logABC(Theta_Re[0, ].view(1, -1), 50)
num_acc=0
global_frequency = 0.1
MC='MHC'
if __name__ == '__main__':
    if MC == 'MCMC':
        for i in tqdm(range(0, num_ite)):
            Theta_Re[i + 1, :] = Theta_Re[i, :]
            if torch.rand(1) < global_frequency:
                Theta_prop, Theta_prop_log_prob = Proposal_G.forward(batch_size)
                Theta_prop[0, :] = Theta_Re[i, :]
                Theta_prop_log_prob[0,] = Proposal_G.log_prob(Theta_prop[0, :].view(1, -1))
                has_nans = torch.isnan(Theta_prop)
                no_nan_in_row = torch.all(~has_nans, dim=1)
                Theta_prop = Theta_prop[no_nan_in_row]
                Theta_prop_log_prob = Theta_prop_log_prob[no_nan_in_row]
                x = generate_samples(Theta_prop, 1)
                x[0, :] = y_old
                log_prob_like = calculate_log_kernel(x)
                log_weight = Prior.log_prob(Theta_prop) + log_prob_like - Theta_prop_log_prob
                weight = torch.exp(log_weight)
                weight = weight / torch.sum(weight)
                ind = weight_sampling(weight.tolist())
                if ind is not None and ind !=0:
                    Theta_Re[i + 1, :] = Theta_prop[ind, :]
                    y_old = x[ind,]
                    num_acc += 1
                    print(num_acc / (i + 1))
                if global_frequency != 1:
                    grad_logABC_old = numerical_gradient_logABC(Theta_prop[ind, :], 50)
            else:
                Theta_prop = Local_proposal.sample(1) + Theta_Re[i - 1, :].view(1, -1)
                x = generate_samples(Theta_prop, 1)[0,]
                log_acc = Prior.log_prob(Theta_prop) + calculate_log_kernel(x) - \
                          Prior.log_prob(Theta_Re[i - 1, :].view(1, -1)) - calculate_log_kernel(y_old)
                log_w = torch.log(torch.rand(1))
                # Theta_prop, Theta_prop_log_prob = proposal(Theta_Re[i, :], grad_logABC_old, tau)
                # grad_logABC_prop = numerical_gradient_logABC(Theta_prop, 20)
                # y = generate_samples(Theta_prop, 1)
                # y = y[0,]
                # log_acc = Prior.log_prob(Theta_prop) + calculate_log_kernel(y) + \
                #           log_proposal(Theta_prop, grad_logABC_prop, Theta_Re[i, :], tau) - \
                #           Prior.log_prob(Theta_Re[i, :].view(1, -1)) - calculate_log_kernel(y_old) - \
                #           Theta_prop_log_prob
                # log_w = torch.log(torch.rand(1))
                if log_w < log_acc:
                    num_acc += 1
                    print(num_acc / (i + 1))
                    Theta_Re[i + 1, :] = Theta_prop
                    y_old = x
                    # grad_logABC_old = grad_logABC_prop
        Theta_Re = Theta_Re.view(-1, latent_size)
        my_list = Theta_Re.detach().numpy()
        file = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/Result/Mix4"
        csv_file = file + "/Mix_GLMCMC" + str(global_frequency) + ".csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in my_list:
                writer.writerow(row)
    else:
        for i in tqdm(range(0, num_ite)):
            Theta_Re[i + 1, :] = Theta_Re[i, :]
            if torch.rand(1) < global_frequency:
                Theta_prop, Theta_prop_log_prob = Proposal_G.forward(batch_size)
                Theta_prop[0, :] = Theta_Re[i, :]
                Theta_prop_log_prob[0,] = Proposal_G.log_prob(Theta_prop[0, :].view(1, -1))
                has_nans = torch.isnan(Theta_prop)
                no_nan_in_row = torch.all(~has_nans, dim=1)
                Theta_prop = Theta_prop[no_nan_in_row]
                Theta_prop_log_prob = Theta_prop_log_prob[no_nan_in_row]
                x = generate_samples(Theta_prop, 1)
                x[0, :] = y_old
                log_prob_like = calculate_log_kernel(x)
                log_weight = Prior.log_prob(Theta_prop) + log_prob_like - Theta_prop_log_prob
                weight = torch.exp(log_weight)
                weight = weight / torch.sum(weight)
                ind = weight_sampling(weight.tolist())
                if ind is not None:
                    Theta_Re[i + 1, :] = Theta_prop[ind, :]
                    y_old = x[ind,]
                    num_acc += 1
                    print(num_acc / (i + 1))
                if global_frequency != 1:
                    grad_logABC_old = numerical_gradient_logABC(Theta_prop[ind, :], 50)
            else:
                Theta_prop, Theta_prop_log_prob = proposal(Theta_Re[i, :], grad_logABC_old, tau)
                grad_logABC_prop = numerical_gradient_logABC(Theta_prop, 20)
                y = generate_samples(Theta_prop, 1)
                y = y[0,]
                log_acc = Prior.log_prob(Theta_prop) + calculate_log_kernel(y) + \
                          log_proposal(Theta_prop, grad_logABC_prop, Theta_Re[i, :], tau) - \
                          Prior.log_prob(Theta_Re[i, :].view(1, -1)) - calculate_log_kernel(y_old) - \
                          Theta_prop_log_prob
                log_w = torch.log(torch.rand(1))
                if log_w < log_acc:
                    num_acc += 1
                    print(num_acc / (i + 1))
                    Theta_Re[i + 1, :] = Theta_prop
                    y_old = y
                    grad_logABC_old = grad_logABC_prop
        Theta_Re = Theta_Re.view(-1, latent_size)
        my_list = Theta_Re.detach().numpy()
        file = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/Result/Mix4"
        csv_file = file + "/Mix_GLMHC" + str(global_frequency) + ".csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in my_list:
                writer.writerow(row)
