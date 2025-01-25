import torch
from tqdm import tqdm
import numpy as np
import csv


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

def GLMCMC(ABCset,num_ite,Initial_theta,Initial_y,Local_Proposal,
           filelocation,global_frequency=0,Importance_Proposal=None,batch_size=None):
    """
    Implement the Generalized Likelihood Monte Carlo (GLMCMC) algorithm.

    Args:
    ABCset (object): ABC method set.
    num_ite (int): Number of iterations.
    Initial_theta (torch.Tensor): Initial parameter values.
    Initial_y (torch.Tensor): Initial observed data.
    Local_Proposal (object): Local proposal distribution.
    filelocation (str): Path to save samples.
    global_frequency (float, optional): Probability of executing a global proposal. Defaults to 0.
    Importance_Proposal (object, optional): Importance proposal distribution. Defaults to None.
    batch_size (int, optional): Batch size for generating samples. Defaults to None.

    Returns:
    torch.Tensor: Collected parameter samples.
    """
    if filelocation is not None:
        with open(filelocation, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write initial Theta
            writer.writerow(Initial_theta.view(-1).detach().numpy())
    Theta_old = Initial_theta.view(1, -1)
    y_old = Initial_y
    local = True
    num_acc = 0
    log_prob_like = ABCset.calculate_log_kernel(y_old)
    log_weight_old = (
            ABCset.prior_log_prob(Theta_old) + log_prob_like - Importance_Proposal.log_prob(
        Theta_old)).view(-1)
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
            # print(ABCset.discrepancy(x))
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
            Theta_Re[i, :] = Theta_old.clone()
        else:
            Theta_prop = Local_Proposal.sample(1) + Theta_old
            while ABCset.prior_log_prob(Theta_prop) == 7*torch.tensor(10**(-10)).log():
                Theta_prop = Local_Proposal.sample(1) + Theta_old
            y = ABCset.generate_samples(Theta_prop, 1)
            y = y[0,].clone()
            log_acc = ABCset.prior_log_prob(Theta_prop) + ABCset.calculate_log_kernel(y) - \
                      ABCset.prior_log_prob(Theta_old) - ABCset.calculate_log_kernel(y_old)
            log_w = torch.log(torch.rand(1))
            if log_w < log_acc:
                local = True
                num_acc += 1
                Theta_old = Theta_prop.clone()
                y_old = y.clone()
            Theta_Re[i, :] = Theta_old.clone()
        if filelocation is not None:
            if i % 10000 == 0 or i == num_ite - 1:
                k = (i-1)//10000
                with open(filelocation, 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    for j in range(max(1,k*10000+1), i+1):
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
        # print(f"  Effective Sample Size: {effective_sample_sizes[i]:.2f}\n")
    return Theta_Re
