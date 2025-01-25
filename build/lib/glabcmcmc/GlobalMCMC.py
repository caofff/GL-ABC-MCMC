import torch
from tqdm import tqdm
import numpy as np
import csv

def GlobalMCMC(ABCset, num_ite,Initial_theta,Initial_y,
               Global_Proposal,filelocation,global_frequency,Local_Proposal=None):
    """
    Implement the Generalized Likelihood Monte Carlo (GLMCMC) algorithm.

    Args:
    ABCset (object): ABC method set.
    num_ite (int): Number of iterations.
    Initial_theta (torch.Tensor): Initial parameter values.
    Initial_y (torch.Tensor): Initial observed data.
    Local_Proposal (object): Local proposal distribution.
    Global_Proposal (object): Local proposal distribution.
    filelocation (str): Path to save samples.
    global_frequency (float, optional): Probability of executing a global proposal. Defaults to 0.
    batch_size (int, optional): Batch size for generating samples. Defaults to None.

    Returns:
    torch.Tensor: Collected parameter samples.
    """
    # Initialize the CSV file with the initial theta value

    if filelocation is not None:
        with open(filelocation, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(Initial_theta.detach().numpy())
    Theta_old = Initial_theta.view(1, -1)
    y_old = Initial_y
    num_acc = 0
    Theta_Re = torch.zeros(num_ite, ABCset.theta_dim)
    Theta_Re[0,:] = Theta_old.clone()
    # Start MCMC iterations
    for i in tqdm(range(1, num_ite)):
        # Decide whether to use global proposal or local proposal
        if torch.rand(1) < global_frequency:
            Theta_prop, prop_log_prob = Global_Proposal.forward(1)
            y = ABCset.generate_samples(Theta_prop, 1)
            y = y.view(1, -1)
            # Calculate the acceptance probability
            log_acc = ABCset.prior_log_prob(Theta_prop) + ABCset.calculate_log_kernel(y) + \
                      Global_Proposal.log_prob(Theta_old.view(1, -1)) - prop_log_prob - \
                      ABCset.prior_log_prob(Theta_old.view(1, -1)) - ABCset.calculate_log_kernel(y_old)
            log_w = torch.log(torch.rand(1))
            # Decide whether to accept the proposal
            if log_w < log_acc:
                num_acc += 1
                Theta_old = Theta_prop.clone()
                y_old = y.clone()
            Theta_Re[i, :] = Theta_old.clone()

        else:
            Theta_prop = Local_Proposal.sample(1) + Theta_old
            y = ABCset.generate_samples(Theta_prop, 1)
            y = y[0,].clone()
            # Calculate the acceptance probability for local proposal
            log_acc = ABCset.prior_log_prob(Theta_prop) + ABCset.calculate_log_kernel(y) - \
                      ABCset.prior_log_prob(Theta_old) - ABCset.calculate_log_kernel(y_old)
            log_w = torch.log(torch.rand(1))
            # Decide whether to accept the proposal
            if log_w < log_acc:
                num_acc += 1
                Theta_old = Theta_prop.clone()
                y_old = y.clone()
            Theta_Re[i, :] = Theta_old.clone()
        # Save the sampling results
        if filelocation is not None:
            if i % 10000 == 0 or i == num_ite - 1:
                k = i//10000
                with open(filelocation, 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    for j in range(max(1, (k-1)*10000+1), i+1):
                        writer.writerow(Theta_Re[j, :].view(-1).detach().numpy())
    # Calculate the mean, variance, and confidence interval of the sampling results
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
    # Print the mean, variance, and confidence interval of the sampling results
    for i in range(Theta_Re.size(1)):
        print(f"Theta_Re {i + 1}:")
        print(f"  Mean: {means[i].item():.4f}")
        print(f"  Variance: {variances[i].item():.4f}")
        print(f"  95% Confidence Interval: {confidence_intervals[i]}")
    return Theta_Re
