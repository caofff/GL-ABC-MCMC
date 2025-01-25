import torch
import torch.nn.parallel
import torch.optim
from tqdm import tqdm
import numpy as np
import csv
import normflows as nf



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


def resample(W, N):
    n_re = torch.zeros(len(W), dtype=torch.int)  # 初始化重采样计数器
    u = (torch.rand(1).item() + torch.arange(N)) / N  # 生成随机数序列
    Psum = torch.cumsum(W, dim=0)  # 权重的累积和
    i = 0
    for j in range(len(W)):
        while i < N and Psum[j] > u[i]:
            i += 1
            n_re[j] += 1
    # 重复索引以匹配重采样计数器
    id = torch.repeat_interleave(torch.arange(0, len(W)), n_re)
    return id


def GLMCMC_NF(ABCset,num_ite,Initial_theta,Initial_y,Local_Proposal,
           filelocation,global_frequency,step_size,batch_size,base, Train_step):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    Theta_old = Initial_theta
    y_old = Initial_y
    # num_acc = 0
    # Normalizing flows
    num_layers = 32
    flows = []
    for i in range(num_layers):
        # Neural network with two hidden layers having 64 units each
        # Last layer is initialized by zeros making training more stable
        param_map = nf.nets.MLP([1, 128, 128, 2], init_zeros=True)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        flows.append(nf.flows.Permute(ABCset.theta_dim, mode='swap'))
    NF_model = nf.NormalizingFlow(base, flows)
    NF_model = NF_model.to(device)
    optimizer = torch.optim.Adam(NF_model.parameters(), lr=5e-4, weight_decay=1e-5)
    # Move model on GPU if available
    loss_hist = np.array([])
    if filelocation is not None:
        with open(filelocation, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(Initial_theta.detach().numpy())
    with torch.no_grad():
        NF_model.eval()
        Theta_prop0, Theta_prop_log_prob0 = NF_model.sample(batch_size * step_size)
    has_nans = torch.isnan(Theta_prop0)
    no_nan_in_row = torch.all(~has_nans, dim=1)
    nan_in_row = torch.all(has_nans, dim=1)
    Theta_prop0 = Theta_prop0.cpu()
    Theta_prop_log_prob0 = Theta_prop_log_prob0.cpu()
    x0 = torch.zeros(batch_size * step_size, Initial_y.shape[1])
    x0[no_nan_in_row] = ABCset.generate_samples(Theta_prop0[no_nan_in_row], 1)
    like_log_prob0 = ABCset.calculate_log_kernel(x0)
    log_weight0 = ABCset.prior_log_prob(Theta_prop0) + like_log_prob0 - Theta_prop_log_prob0
    weight0 = torch.exp(log_weight0)
    weight0[nan_in_row] = 0.0
    has_nans = torch.isnan(weight0)
    weight0[has_nans] = 0.0
    kk = 0
    num_train = 0
    Theta_Re = torch.zeros(num_ite, ABCset.theta_dim)
    Theta_Re[0, :] = Theta_old.clone()
    for i in tqdm(range(1, num_ite)):
        a = torch.rand(1)
        if a < global_frequency:
            iiid = range(kk * batch_size, (kk + 1) * batch_size)
            Theta_prop = torch.cat((Theta_old.view(1, -1), Theta_prop0[iiid, :]), dim=0)
            x = torch.cat((y_old.view(1, -1), x0[iiid,:]), dim=0)
            with torch.no_grad():
                NF_model.eval()
                Theta_old_prop_log_prob = NF_model.log_prob(Theta_old.view(1, -1).to(device)).cpu()
            Theta_old_like_log_prob = ABCset.calculate_log_kernel(y_old)
            Theta_old_weight = torch.exp(ABCset.prior_log_prob(Theta_old.view(1, -1)) + Theta_old_like_log_prob -
                                         Theta_old_prop_log_prob)
            weight = torch.cat((Theta_old_weight, weight0[iiid]), dim=0)
            weight = weight / torch.sum(weight)
            ind = weight_sampling(weight.tolist())
            if ind is not None and ind != 0:
                Theta_old = Theta_prop[ind, :].clone()
                y_old = x[ind, :].clone().view(1, -1)
                # num_acc += 1
                # print(Theta_Re[i, :],num_acc/(i+1))
            Theta_Re[i, :] = Theta_old.clone()
            kk += 1
            if kk == step_size:
                kk = 0
                if num_train < Train_step:
                    optimizer.zero_grad()
                    Train_weight = weight0 / torch.sum(weight0)
                    id = resample(Train_weight, step_size * batch_size)
                    Train_t = Theta_prop0[id, :].to(device)
                    loss = NF_model.forward_kld(Train_t.detach().float())
                    if ~(torch.isnan(loss) | torch.isinf(loss)):
                        loss.backward()
                    optimizer.step()
                    num_train += 1
                    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
                with torch.no_grad():
                    NF_model.eval()
                    Theta_prop0, Theta_prop_log_prob0 = NF_model.sample(batch_size * step_size)
                has_nans = torch.isnan(Theta_prop0)
                no_nan_in_row = torch.all(~has_nans, dim=1)
                nan_in_row = torch.all(has_nans, dim=1)
                Theta_prop0 = Theta_prop0.cpu()
                Theta_prop_log_prob0 = Theta_prop_log_prob0.cpu()
                x0 = torch.zeros(batch_size * step_size, Initial_y.shape[1])
                x0[no_nan_in_row] = ABCset.generate_samples(Theta_prop0[no_nan_in_row], 1)
                like_log_prob0 = ABCset.calculate_log_kernel(x0)
                log_weight0 = ABCset.prior_log_prob(Theta_prop0) + like_log_prob0 - Theta_prop_log_prob0
                weight0 = torch.exp(log_weight0)
                weight0[nan_in_row] = 0.0
                has_nans = torch.isnan(weight0)
                weight0[has_nans] = 0.0
        else:
            Theta_prop = Local_Proposal.sample(1) + Theta_old.view(1, -1)
            y = ABCset.generate_samples(Theta_prop, 1)
            y = y[0,].to(device)
            log_acc = ABCset.prior_log_prob(Theta_prop) + ABCset.calculate_log_kernel(y) - \
                      ABCset.prior_log_prob(Theta_old.view(1, -1)) - ABCset.calculate_log_kernel(y_old)
            log_w = torch.log(torch.rand(1))
            if log_w < log_acc:
                # num_acc += 1
                Theta_old = Theta_prop.clone()
                y_old = y.clone()
            Theta_Re[i, :] = Theta_old.clone()
        if filelocation is not None:
            if i % 10000 == 0 or i == num_ite - 1:
                k = i // 10000
                with open(filelocation, 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    for j in range(max(1, (k - 1) * 10000 + 1), i + 1):
                        writer.writerow(Theta_Re[j, :].view(-1).detach().numpy())
        torch.cuda.empty_cache()

    # 为每一列计算均值和方差
    means = torch.mean(Theta_Re, dim=0)
    variances = torch.var(Theta_Re, dim=0)

    # 计算每一列的置信区间（95%）
    confidence_intervals = []
    alpha = 0.05
    z_score = 1.96  # 对于95%置信区间，假设正态分布

    for i in range(Theta_Re.size(1)):
        mean = means[i].item()
        std_err = torch.std(Theta_Re[:, i])
        margin_of_error = z_score * std_err
        ci_lower = mean - margin_of_error
        ci_upper = mean + margin_of_error
        confidence_intervals.append((ci_lower, ci_upper))

    # 输出结果
    for i in range(Theta_Re.size(1)):
        print(f"Theta_Re {i + 1}:")
        print(f"  Mean: {means[i].item():.4f}")
        print(f"  Variance: {variances[i].item():.4f}")
        print(f"  95% Confidence Interval: {confidence_intervals[i]}")
        # print(f"  Effective Sample Size: {effective_sample_sizes[i]:.2f}\n")
    return Theta_Re
