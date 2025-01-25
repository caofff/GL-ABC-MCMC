# import torch
# from tqdm import tqdm
# import numpy as np
# import csv
# import secrets
# import glabcmcmc.distribution as distribution
#
# def weight_sampling(w_list):
#     ran = np.random.uniform(0, 1)
#     s_wei = 0
#     for j in range(len(w_list)):
#         s_wei += w_list[j]
#         if ran < s_wei:
#             return j
#
# def Local_proposal_forward(theta, grad_logABC_theta, tau):
#     """
#     Proposal distribution.
#
#     Args:
#     theta (torch.Tensor): The latent variable.
#     grad_logABC (torch.Tensor): The gradient of the log-ABC.
#     tau (float): The temperature.
#
#     Returns:
#     torch.Tensor: The proposal distribution.
#     """
#     if theta.ndim == 1:
#         latent_size = int(theta.shape[0])
#     else:
#         _, latent_size = theta.shape
#     pro = distribution.DiagGaussian(latent_size, loc=torch.tensor([0.0]), log_scale=torch.tensor([0.0]))
#     z, log_pro = pro.forward(1)
#     x = z*tau + theta + grad_logABC_theta * tau ** 2 / 2
#     return x, log_pro
#
# def numberical_gradient_logABC(ABCset,theta,y_obs,num, d=1e-1):
#     # 计算数值梯度
#     if theta.ndim == 1:
#         theta = theta.view(1, -1)
#     theta_num, theta_dim = theta.shape
#     theta = theta.float()
#     thetai_plus = theta.clone().repeat(1, 1, theta_dim).view(-1, theta_dim, theta_dim).float()
#     thetai_minus = theta.clone().repeat(1, 1, theta_dim).view(-1, theta_dim, theta_dim).float()
#     eye_matrix = d * torch.eye(theta_dim)
#     eye = torch.eye(theta_dim)
#     thetai_plus += eye_matrix.repeat(theta_num,1,1)
#     thetai_minus -= eye_matrix.repeat(theta_num,1,1)
#     y_dim = y_obs.shape[1]
#     deltai_plus = torch.zeros(theta_num, theta_dim, num).to(dtype=torch.float64)
#     deltai_minus = torch.zeros(theta_num, theta_dim, num).to(dtype=torch.float64)
#     grad_log_prior = torch.zeros(theta_num, theta_dim).to(dtype=torch.float64)
#     for i in range(theta_num):
#         random_seeds_int = torch.tensor([secrets.randbelow(2 ** 32) for _ in range(theta_dim)], dtype=torch.long)
#         for k in range(theta_dim):
#             torch.manual_seed(random_seeds_int[k])
#             np.random.seed(random_seeds_int[k])
#             deltai_plus[i, k, :] = ABCset.discrepancy(ABCset.generate_samples(thetai_plus[i, k, ].repeat(num, 1), 1).
#                                                       view(-1, y_dim), y_obs)
#             torch.manual_seed(random_seeds_int[k])
#             np.random.seed(random_seeds_int[k])
#             deltai_minus[i, k, :] = ABCset.discrepancy(ABCset.generate_samples(thetai_minus[i, k, ].repeat(num, 1), 1).
#                                                        view(-1, y_dim), y_obs)
#             grad_log_prior[i, k] = (ABCset.prior_log_prob(theta[i, :] + eye[k, :] * 0.00001) - ABCset.prior_log_prob(
#                 theta[i, :] - eye[k, :] * 0.00001)) / (2 * 0.00001)
#     mu_plus = torch.mean(deltai_plus, dim=2).view(theta_num, -1)
#     mu_minus = torch.mean(deltai_minus, dim=2).view(theta_num, -1)
#     Sigma_plus = torch.var(deltai_plus, dim=2).view(theta_num, -1)
#     Sigma_minus = torch.var(deltai_minus, dim=2).view(theta_num, -1)
#     logp_plus = -1 / 2 * torch.log(Sigma_plus + ABCset.epsilon ** 2) \
#                 - 1 / 2 * (mu_plus) ** 2 / (Sigma_plus + ABCset.epsilon ** 2)
#     logp_minus = -1 / 2 * torch.log(Sigma_minus + ABCset.epsilon ** 2) \
#                  - 1 / 2 * (mu_minus) ** 2 / (Sigma_minus + ABCset.epsilon ** 2)
#     grad_log_likelihood = (logp_plus - logp_minus) / (2 * d)
#     return grad_log_likelihood.view(-1, theta_dim) + grad_log_prior.view(-1, theta_dim)
#
# def log_proposal(theta_old, grad_logABC_theta_old, theta_prop, tau):
#     if theta_old.ndim == 1:
#         latent_size = int(theta_old.shape[0])
#     else:
#         _, latent_size = theta_old.shape
#     pro = distribution.DiagGaussian(latent_size, loc=torch.tensor([0.0]), log_scale=torch.tensor([0.0]))
#     re = pro.log_prob((theta_prop-theta_old-grad_logABC_theta_old*tau**2/2)/tau)
#     return re
#
# def GLMCMC(ABCset,y_obs,num_ite,Initial_theta,Initial_y,
#            filelocation,Local_Proposal=None,global_frequency=0,Global_Proposal=None,iSIR=False,
#            batch_size=None,Importance_Proposal=None, MALA=False,tau=None,num_grad=None,NF=False,
#            step_size=None,base=None, Train_step=None):
#     # ABCset: the set of ABC samples
#     # num_ite: the number of iterations
#     # Initial_theta: the initial theta
#     # Initial_y: the initial y
#     # Local_Proposal: the local proposal
#     # folderlocation: the folder location to save the results
#     # global_frequency: the global frequency
#     # Global_Proposal: the global proposal distribution
#     # batch_size: the batch size of ABC-i-SIR
#
#     # Initialize theta and y
#     if iSIR:
#         if batch_size is None:
#             raise ValueError("When iSIR=True, the parameter 'batch_size' must be provided.")
#     if MALA:
#         if tau is None or num_grad is None:
#             raise ValueError("When MALA=True, the parameters 'tau' and 'num_grad' must be provided.")
#     else:
#         if Local_Proposal is None:
#             raise ValueError("When MALA=False, 'Local_Proposal' must be provided.")
#
#     with open(filelocation, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.writer(f)
#         # 写入初始Theta
#         writer.writerow(Initial_theta.view(-1).detach().numpy())
#     Theta_old = Initial_theta.view(1, -1)
#     y_old = Initial_y
#     local = True
#     grad_logABC_Theta_old = None
#     num_acc = 0
#     if iSIR:
#         Method = iSIR(local,A=c,B=c)
#     for i in tqdm(range(1, num_ite)):
#         if torch.rand(1) < global_frequency:
#             Method(,,,,)
#             if iSIR:
#                 if local:
#                     log_prob_like = ABCset.calculate_log_kernel(y_old, y_obs)
#                     log_weight_old = (
#                             ABCset.prior_log_prob(Theta_old) + log_prob_like - Importance_Proposal.log_prob(
#                         Theta_old)).view(-1)
#                 local = False
#                 Theta_prop0, Theta_prop_log_prob0 = Importance_Proposal.forward(batch_size)
#                 has_nans = torch.isnan(Theta_prop0)
#                 no_nan_in_row = torch.all(~has_nans, dim=1)
#                 Theta_prop0 = Theta_prop0[no_nan_in_row].clone()
#                 Theta_prop_log_prob0 = Theta_prop_log_prob0[no_nan_in_row].clone()
#                 x = ABCset.generate_samples(Theta_prop0, 1)
#                 log_prob_like = ABCset.calculate_log_kernel(x, y_obs)
#                 # print(ABCset.discrepancy(x,y_obs))
#                 log_weight0 = ABCset.prior_log_prob(Theta_prop0) + log_prob_like - Theta_prop_log_prob0
#                 log_weight = torch.cat((log_weight_old, log_weight0))
#                 Theta_prop = torch.cat((Theta_old, Theta_prop0), dim=0)
#                 x = torch.cat((y_old.view(1, -1), x), dim=0)
#                 weight = torch.exp(log_weight)
#
#                 has_nans = torch.isnan(weight)
#                 weight[has_nans] = 0.0
#                 weight = weight / torch.sum(weight)
#                 ind = weight_sampling(weight.tolist())
#                 if ind is not None and ind != 0:
#                     Theta_old = Theta_prop[ind, :].clone().view(1, -1)
#                     log_weight_old = log_weight[ind].clone().view(-1)
#                     y_old = x[ind, :].clone()
#                     num_acc += 1
#                     # print('*******************', i, num_acc, num_acc / (i + 1))
#             else:
#                 Theta_prop, prop_log_prob = Global_Proposal.forward(1)
#                 y = ABCset.generate_samples(Theta_prop, 1)
#                 y = y.view(1, -1)
#                 log_acc = ABCset.prior_log_prob(Theta_prop) + ABCset.calculate_log_kernel(y, y_obs) + \
#                           Global_Proposal.log_prob(Theta_old.view(1, -1)) - prop_log_prob - \
#                           ABCset.prior_log_prob(Theta_old.view(1, -1)) - ABCset.calculate_log_kernel(y_old, y_obs)
#                 log_w = torch.log(torch.rand(1))
#                 if log_w < log_acc:
#                     num_acc += 1
#                     Theta_old = Theta_prop.clone()
#                     y_old = y.clone()
#
#         else:
#             if MALA:
#                 if grad_logABC_Theta_old is None:
#                     grad_logABC_Theta_old = numberical_gradient_logABC(ABCset, Theta_old, y_obs, num_grad)
#
#                 Theta_prop, Theta_prop_log_prob = Local_proposal_forward(Theta_old, grad_logABC_Theta_old, tau)
#                 grad_logABC_prop = numberical_gradient_logABC(ABCset, Theta_prop, y_obs, num_grad)
#                 y = ABCset.generate_samples(Theta_prop, 1)
#                 y = y[0,]
#                 log_acc = ABCset.prior_log_prob(Theta_prop) + ABCset.calculate_log_kernel(y, y_obs) + \
#                           log_proposal(Theta_prop, grad_logABC_prop, Theta_old, tau) - \
#                           ABCset.prior_log_prob(Theta_old.view(1, -1)) - ABCset.calculate_log_kernel(y_old, y_obs) - \
#                           Theta_prop_log_prob
#                 log_w = torch.log(torch.rand(1))
#                 if log_w < log_acc:
#                     num_acc += 1
#                     Theta_old = Theta_prop
#                     y_old = y
#                     grad_logABC_Theta_old = grad_logABC_prop
#             else:
#                 if grad_logABC_Theta_old is None:
#                     grad_logABC_Theta_old = numberical_gradient_logABC(ABCset, Theta_old, y_obs, num_grad)
#
#                 Theta_prop, Theta_prop_log_prob = Local_proposal_forward(Theta_old, grad_logABC_Theta_old, tau)
#                 grad_logABC_prop = numberical_gradient_logABC(ABCset, Theta_prop, y_obs, num_grad)
#                 y = ABCset.generate_samples(Theta_prop, 1)
#                 y = y[0,]
#                 log_acc = ABCset.prior_log_prob(Theta_prop) + ABCset.calculate_log_kernel(y, y_obs) + \
#                           log_proposal(Theta_prop, grad_logABC_prop, Theta_old, tau) - \
#                           ABCset.prior_log_prob(Theta_old.view(1, -1)) - ABCset.calculate_log_kernel(y_old, y_obs) - \
#                           Theta_prop_log_prob
#                 log_w = torch.log(torch.rand(1))
#                 if log_w < log_acc:
#                     num_acc += 1
#                     Theta_old = Theta_prop
#                     y_old = y
#                     grad_logABC_Theta_old = grad_logABC_prop
#         with open(filelocation, 'a', newline='', encoding='utf-8') as file:
#             writer = csv.writer(file)
#             writer.writerow(Theta_old.view(-1).detach().numpy())
#
#
#
