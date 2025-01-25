import csv
import torch
import numpy as np
from tqdm import tqdm
import secrets
import glabcmcmc.distribution as distribution

def weight_sampling(w_list):
    """
    Perform weighted sampling based on the provided weight list.

    Args:
    w_list (list): List of weights.

    Returns:
    int: The sampled index.
    """
    ran = np.random.uniform(0, 1)
    s_wei = 0
    for j in range(len(w_list)):
        s_wei += w_list[j]
        if ran < s_wei:
            return j

def Local_proposal_forward(theta, grad_logABC_theta, tau):
    """
    Generate a proposal distribution for the next step in the sampling process.

    Args:
    theta (torch.Tensor): The current latent variable.
    grad_logABC_theta (torch.Tensor): The gradient of the log-ABC at the current theta.
    tau (float): The temperature parameter.

    Returns:
    tuple: A tuple containing the proposed new theta and the log probability of the proposal.
    """
    if theta.ndim == 1:
        latent_size = int(theta.shape[0])
    else:
        _, latent_size = theta.shape
    pro = distribution.DiagGaussian(latent_size, loc=torch.tensor([0.0]), log_scale=torch.tensor([0.0]))
    z, log_pro = pro.forward(1)
    x = z*tau + theta + grad_logABC_theta * tau ** 2 / 2
    return x, log_pro

def numberical_gradient_logABC(ABCset,theta,num, d=1e-1):
    """
    Compute the numerical gradient of the log-ABC function.

    Args:
    ABCset: The set of ABC samples.
    theta (torch.Tensor): The current latent variable.
    num (int): Number of samples to generate for discrepancy calculation.
    d (float): Small perturbation used for numerical differentiation.

    Returns:
    torch.Tensor: The computed gradient of the log-ABC function.
    """
    if theta.ndim == 1:
        theta = theta.view(1, -1)
    theta_num, theta_dim = theta.shape
    theta = theta.float()
    thetai_plus = theta.clone().repeat(1, 1, theta_dim).view(-1, theta_dim, theta_dim).float()
    thetai_minus = theta.clone().repeat(1, 1, theta_dim).view(-1, theta_dim, theta_dim).float()
    eye_matrix = d * torch.eye(theta_dim)
    eye = torch.eye(theta_dim)
    thetai_plus += eye_matrix.repeat(theta_num,1,1)
    thetai_minus -= eye_matrix.repeat(theta_num,1,1)
    y_dim = ABCset.y_obs.shape[1]
    deltai_plus = torch.zeros(theta_num, theta_dim, num).to(dtype=torch.float64)
    deltai_minus = torch.zeros(theta_num, theta_dim, num).to(dtype=torch.float64)
    grad_log_prior = torch.zeros(theta_num, theta_dim).to(dtype=torch.float64)
    for i in range(theta_num):
        random_seeds_int = torch.tensor([secrets.randbelow(2 ** 32) for _ in range(theta_dim)], dtype=torch.long)
        for k in range(theta_dim):
            torch.manual_seed(random_seeds_int[k])
            np.random.seed(random_seeds_int[k])
            deltai_plus[i, k, :] = ABCset.discrepancy(ABCset.generate_samples(thetai_plus[i, k, ].repeat(num, 1), 1).
                                                      view(-1, y_dim))
            torch.manual_seed(random_seeds_int[k])
            np.random.seed(random_seeds_int[k])
            deltai_minus[i, k, :] = ABCset.discrepancy(ABCset.generate_samples(thetai_minus[i, k, ].repeat(num, 1), 1).
                                                       view(-1, y_dim))
            grad_log_prior[i, k] = (ABCset.prior_log_prob(theta[i, :] + eye[k, :] * 0.00001) - ABCset.prior_log_prob(
                theta[i, :] - eye[k, :] * 0.00001)) / (2 * 0.00001)
    mu_plus = torch.mean(deltai_plus, dim=2).view(theta_num, -1)
    mu_minus = torch.mean(deltai_minus, dim=2).view(theta_num, -1)
    Sigma_plus = torch.var(deltai_plus, dim=2).view(theta_num, -1)
    Sigma_minus = torch.var(deltai_minus, dim=2).view(theta_num, -1)
    logp_plus = -1 / 2 * torch.log(Sigma_plus + ABCset.epsilon ** 2) \
                - 1 / 2 * (mu_plus) ** 2 / (Sigma_plus + ABCset.epsilon ** 2)
    logp_minus = -1 / 2 * torch.log(Sigma_minus + ABCset.epsilon ** 2) \
                 - 1 / 2 * (mu_minus) ** 2 / (Sigma_minus + ABCset.epsilon ** 2)
    grad_log_likelihood = (logp_plus - logp_minus) / (2 * d)
    return grad_log_likelihood.view(-1, theta_dim) + grad_log_prior.view(-1, theta_dim)

def log_proposal(theta_old, grad_logABC_theta_old, theta_prop, tau):
    """
    Compute the log probability of the proposal distribution.

    Args:
    theta_old (torch.Tensor): The old latent variable.
    grad_logABC_theta_old (torch.Tensor): The gradient of the log-ABC at the old theta.
    theta_prop (torch.Tensor): The proposed new latent variable.
    tau (float): The temperature parameter.

    Returns:
    torch.Tensor: The log probability of the proposal.
    """
    if theta_old.ndim == 1:
        latent_size = int(theta_old.shape[0])
    else:
        _, latent_size = theta_old.shape
    pro = distribution.DiagGaussian(latent_size, loc=torch.tensor([0.0]), log_scale=torch.tensor([0.0]))
    re = pro.log_prob((theta_prop-theta_old-grad_logABC_theta_old*tau**2/2)/tau)
    return re

def GLMALA(ABCset,num_ite,Initial_theta,Initial_y,tau,num_grad,
           filelocation,global_frequency=0,Importance_Proposal=None,batch_size=None):
    """
    Perform Generalized Metropolis Adjusted Langevin Algorithm (GLMALA).

    Args:
    ABCset: The set of ABC samples.
    num_ite (int): The number of iterations.
    Initial_theta (torch.Tensor): The initial theta.
    Initial_y (torch.Tensor): The initial y.
    tau (float): The temperature parameter.
    num_grad (int): Number of samples for gradient estimation.
    filelocation (str): File location to save results.
    global_frequency (float): Frequency of global moves.
    Importance_Proposal: Proposal distribution for importance sampling.
    batch_size (int): Batch size for importance sampling.

    Returns:
    torch.Tensor: The chain of sampled thetas.
    """
    if filelocation is not None:
        with open(filelocation, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入初始Theta
            writer.writerow(Initial_theta.view(-1).detach().numpy())
    Theta_old = Initial_theta.view(1, -1)
    y_old = Initial_y
    num_acc = 0
    grad_logABC_Theta_old = None
    local = True
    Theta_Re = torch.zeros(num_ite, ABCset.theta_dim)
    Theta_Re[0,:] = Theta_old.clone()
    for i in tqdm(range(1, num_ite)):
        if torch.rand(1) < global_frequency:
            if local:
                log_prob_like = ABCset.calculate_log_kernel(y_old)
                log_weight_old = (
                        ABCset.prior_log_prob(Theta_old) + log_prob_like - Importance_Proposal.log_prob(
                    Theta_old)).view(-1)
            local = False
            Theta_prop0, Theta_prop_log_prob0 = Importance_Proposal.forward(batch_size)
            has_nans = torch.isnan(Theta_prop0)
            no_nan_in_row = torch.all(~has_nans, dim=1)
            Theta_prop0 = Theta_prop0[no_nan_in_row].clone()
            Theta_prop_log_prob0 = Theta_prop_log_prob0[no_nan_in_row].clone()
            x = ABCset.generate_samples(Theta_prop0, 1)
            log_prob_like = ABCset.calculate_log_kernel(x)
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
            Theta_Re[i, :] = Theta_old.clone()

        else:
            if grad_logABC_Theta_old is None:
                grad_logABC_Theta_old = numberical_gradient_logABC(ABCset,Theta_old,num_grad)

            Theta_prop, Theta_prop_log_prob = Local_proposal_forward(Theta_old, grad_logABC_Theta_old, tau)
            grad_logABC_prop = numberical_gradient_logABC(ABCset,Theta_prop,num_grad)
            y = ABCset.generate_samples(Theta_prop, 1)
            y = y[0,]
            log_acc = ABCset.prior_log_prob(Theta_prop) + ABCset.calculate_log_kernel(y) + \
                      log_proposal(Theta_prop, grad_logABC_prop, Theta_old, tau) - \
                      ABCset.prior_log_prob(Theta_old.view(1, -1)) - ABCset.calculate_log_kernel(y_old) - \
                      Theta_prop_log_prob
            log_w = torch.log(torch.rand(1))
            if log_w < log_acc:
                num_acc += 1
                Theta_old = Theta_prop
                y_old = y
                grad_logABC_Theta_old = grad_logABC_prop
            Theta_Re[i, :] = Theta_old.clone()
        if filelocation is not None:
            if i % 10000 == 0 or i == num_ite - 1:
                k = i // 10000
                with open(filelocation, 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    for j in range(max(1, (k - 1) * 10000 + 1), i + 1):
                        writer.writerow(Theta_Re[j, :].view(-1).detach().numpy())

    means = torch.mean(Theta_Re, dim=0)
    variances = torch.var(Theta_Re, dim=0)

    confidence_intervals = []
    alpha = 0.05
    z_score = 1.96

    for i in range(Theta_Re.size(1)):
        mean = means[i].item()
        std_err = torch.std(Theta_Re[:, i])
        margin_of_error = z_score * std_err
        ci_lower = mean - margin_of_error
        ci_upper = mean + margin_of_error
        confidence_intervals.append((ci_lower, ci_upper))


    for i in range(Theta_Re.size(1)):
        print(f"Theta_Re {i + 1}:")
        print(f"  Mean: {means[i].item():.4f}")
        print(f"  Variance: {variances[i].item():.4f}")
        print(f"  95% Confidence Interval: {confidence_intervals[i]}")
    return Theta_Re
