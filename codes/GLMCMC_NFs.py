import torch
import torch.nn.parallel
import torch.optim
from tqdm import tqdm
import numpy as np
import csv
import distribution
import normflows as nf
from matplotlib import pyplot as plt


def weight_sampling(w_list):
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


def GLMCMC_NF(ABCset,y_obs,num_ite,Initial_theta,Initial_y,Local_Proposal,
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
        flows.append(nf.flows.Permute(2, mode='swap'))
    NF_model = nf.NormalizingFlow(base, flows)
    NF_model = NF_model.to(device)
    optimizer = torch.optim.Adam(NF_model.parameters(), lr=5e-4, weight_decay=1e-5)
    xx, yy = torch.meshgrid(torch.linspace(-2, 2, 500), torch.linspace(-2, 2, 500), indexing='ij')
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    NF_model.eval()
    log_prob = NF_model.log_prob(zz).to('cpu').view(*xx.shape)
    NF_model.train()
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0
    plt.figure(figsize=(4, 4))
    plt.pcolormesh(xx.detach().numpy(), yy.detach().numpy(), prob.detach().numpy(), cmap='coolwarm')
    plt.gca().set_aspect('equal', 'box')
    plt.savefig('/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/NF_Mixabs_prior.png')
    # Move model on GPU if available
    loss_hist = np.array([])
    with open(filelocation, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(Initial_theta.detach().numpy())
    with torch.no_grad():
        NF_model.eval()
        Theta_prop0, Theta_prop_log_prob0 = NF_model.sample(batch_size * step_size)
    has_nans = torch.isnan(Theta_prop0)
    no_nan_in_row = torch.all(~has_nans, dim=1)
    nan_in_row = torch.all(has_nans, dim=1)
    # Theta_prop0 = Theta_prop0[no_nan_in_row]
    # Theta_prop_log_prob0 = Theta_prop_log_prob0[no_nan_in_row]
    Theta_prop0 = Theta_prop0.cpu()
    Theta_prop_log_prob0 = Theta_prop_log_prob0.cpu()
    x0 = torch.zeros(batch_size * step_size, Initial_y.shape[1])
    x0[no_nan_in_row] = ABCset.generate_samples(Theta_prop0[no_nan_in_row], 1)
    like_log_prob0 = ABCset.calculate_log_kernel(x0, y_obs)
    log_weight0 = ABCset.prior_log_prob(Theta_prop0) + like_log_prob0 - Theta_prop_log_prob0
    weight0 = torch.exp(log_weight0)
    weight0[nan_in_row] = 0.0
    has_nans = torch.isnan(weight0)
    weight0[has_nans] = 0.0
    kk = 0
    num_train = 0
    for i in tqdm(range(1, num_ite)):
        # if i % 500 == 0:
        #     optimizer.zero_grad()
        #     loss = NF_model.forward_kld(Theta_Re[range(0, i), :].detach().float())
        #     if ~(torch.isnan(loss) | torch.isinf(loss)):
        #         loss.backward()
        #         optimizer.step()
        #         loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        a = torch.rand(1)
        if a < global_frequency:
            iiid = range(kk * batch_size, (kk + 1) * batch_size)
            Theta_prop = torch.cat((Theta_old.view(1, -1), Theta_prop0[iiid, :]), dim=0)
            x = torch.cat((y_old, x0[iiid]), dim=0)
            with torch.no_grad():
                NF_model.eval()
                Theta_old_prop_log_prob = NF_model.log_prob(Theta_old.view(1, -1).to(device)).cpu()
            Theta_old_like_log_prob = ABCset.calculate_log_kernel(y_old, y_obs)
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
            kk += 1
            if kk == step_size:
                kk = 0
                if num_train < Train_step:
                    optimizer.zero_grad()
                    Train_weight = weight0 / torch.sum(weight0)
                    id = resample(Train_weight, step_size * batch_size)
                    Train_t = Theta_prop0[id, :].to(device)
                # plt.figure(figsize=(6, 6))
                # plt.plot(Train_t[:,0].cpu().detach().numpy(),Train_t[:,1].cpu().detach().numpy(), 'b.')
                # plt.show()
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
                # Theta_prop0 = Theta_prop0[no_nan_in_row]
                # Theta_prop_log_prob0 = Theta_prop_log_prob0[no_nan_in_row]
                Theta_prop0 = Theta_prop0.cpu()
                Theta_prop_log_prob0 = Theta_prop_log_prob0.cpu()
                x0 = torch.zeros(batch_size * step_size, Initial_y.shape[1])
                x0[no_nan_in_row] = ABCset.generate_samples(Theta_prop0[no_nan_in_row], 1)
                like_log_prob0 = ABCset.calculate_log_kernel(x0, y_obs)
                log_weight0 = ABCset.prior_log_prob(Theta_prop0) + like_log_prob0 - Theta_prop_log_prob0
                weight0 = torch.exp(log_weight0)
                weight0[nan_in_row] = 0.0
                has_nans = torch.isnan(weight0)
                weight0[has_nans] = 0.0
                if (num_train+1) % 10 == 0:
                    NF_model.eval()
                    log_prob = NF_model.log_prob(zz).to('cpu').view(*xx.shape)
                    NF_model.train()
                    prob = torch.exp(log_prob)
                    prob[torch.isnan(prob)] = 0
                    plt.figure(figsize=(4, 4))
                    plt.pcolormesh(xx.detach().numpy(), yy.detach().numpy(), prob.detach().numpy(), cmap='coolwarm')
                    plt.gca().set_aspect('equal', 'box')
                    plt.savefig('/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/NF_Mixabs_' +
                                str(num_train) + '.png')

        else:
            Theta_prop = Local_Proposal.sample(1) + Theta_old.view(1, -1)
            y = ABCset.generate_samples(Theta_prop, 1)
            y = y[0,].to(device)
            log_acc = ABCset.prior_log_prob(Theta_prop) + ABCset.calculate_log_kernel(y, y_obs) - \
                      ABCset.prior_log_prob(Theta_old.view(1, -1)) - ABCset.calculate_log_kernel(y_old, y_obs)
            log_w = torch.log(torch.rand(1))
            if log_w < log_acc:
                # num_acc += 1
                Theta_old = Theta_prop.clone()
                y_old = y.clone()
                # print(Theta_Re[i, :], num_acc / (i + 1))
        with open(filelocation, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # 写入新数据
            writer.writerow(Theta_old.detach().numpy())
        torch.cuda.empty_cache()
    # plt.figure(figsize=(10, 10))
    # plt.plot(loss_hist, label='loss')
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    from Mixabs import Mixture_set
    Model = Mixture_set(epsilon=0.05)
    y_obs = torch.tensor([[0.5, 0.5]])
    torch.manual_seed(0)
    num_ite = 1000000
    theta0 = torch.tensor([0.0, 0.0])
    Initial_y = Model.generate_samples(theta0)
    Local_Proposal = distribution.DiagGaussian(2, loc=torch.zeros(1, 2), log_scale=torch.log(torch.tensor([0.1, 0.1])))
    # Global_Proposal = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
    # Global_Proposal = distribution.DiagGaussian(2, loc=torch.zeros(1, 2), log_scale=torch.log(torch.tensor([1, 1])))
    filelocation = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/test.csv"
    base = nf.distributions.base.DiagGaussian(2)  # Uniform(2, torch.tensor([-30.0, -20.0]), torch.tensor([30.0, 10.0]))
    GLMCMC_NF(Model, y_obs, num_ite, theta0, Initial_y, Local_Proposal,
              filelocation, 1, 200, 5, base,100)


    # from Moon2 import Moon_set
    # Model = Moon_set(epsilon=0.05)
    # y_obs = torch.tensor([[0.0, 0.0]])
    # torch.manual_seed(0)
    # num_ite = 1000000
    # theta0 = torch.tensor([0.0, 0.0])
    # Initial_y = Model.generate_samples(theta0)
    # Local_Proposal = distribution.DiagGaussian(2, loc=torch.zeros(1, 2), log_scale=torch.log(torch.tensor([0.1, 0.1])))
    # # Global_Proposal = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
    # # Global_Proposal = distribution.DiagGaussian(2, loc=torch.zeros(1, 2), log_scale=torch.log(torch.tensor([1, 1])))
    # filelocation = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/NF_Moon2        _ISIR_100_5.csv"
    # base = nf.distributions.base.DiagGaussian(2)  # Uniform(2, torch.tensor([-30.0, -20.0]), torch.tensor([30.0, 10.0]))
    # GLMCMC_NF(Model, y_obs, num_ite, theta0, Initial_y, Local_Proposal,
    #           filelocation, 1, 200,5, base)

    #
    # from Banana import Banana_set
    # Model = Banana_set(epsilon=0.03)
    # torch.manual_seed(0)
    # y_obs = torch.tensor([[0.0, 0.0]])
    # num_ite = 1000000
    # theta0 = torch.tensor([0.0, 0.0])
    # Initial_y = Model.generate_samples(theta0)
    # Local_Proposal = distribution.DiagGaussian(2, loc=torch.zeros(1, 2), log_scale=torch.log(torch.tensor([0.25, 0.25])))
    # # Global_Proposal = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
    # # Global_Proposal = distribution.DiagGaussian(2, loc=torch.zeros(1, 2), log_scale=torch.log(torch.tensor([1, 1])))
    # filelocation = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/NF_Banana_ISIR_100.csv"
    # base = nf.distributions.base.DiagGaussian(2)  # Uniform(2, torch.tensor([-30.0, -20.0]), torch.tensor([30.0, 10.0]))
    # GLMCMC_NF(Model, y_obs, num_ite, theta0, Initial_y, Local_Proposal,
    #           filelocation, 1, 100, 5, base)