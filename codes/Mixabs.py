import torch
import numpy as np
import distribution
import secrets

class Mixture_set:
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.theta_dim = 2
        # Initialize other necessary variables or constants here

    def generate_samples(self, theta, num_samples=1):
        # cov_matrix = torch.eye(2, dtype=torch.float32) * 0.05
        if theta.dim() == 1:
            num_theta = 1
        else:
            num_theta, dim_theta = theta.shape
        likelihood = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.log(torch.tensor([0.05, 0.05]).sqrt()))
        if num_theta == 1:
            samples = torch.abs(theta) + likelihood.sample(num_samples)
        elif num_samples == 1:
            samples = torch.abs(theta) + likelihood.sample(num_theta)
        else:
            samples = torch.abs(theta).unsqueeze(1).repeat(1, num_samples, 1) + likelihood.sample(num_samples * num_theta).view(num_theta, num_samples, dim_theta)
        return samples

    def prior_sample(self, num):
        Priord = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
        samples, _ = Priord.forward(num)
        return samples

    def prior_log_prob(self, samples):
        samples = samples.view(-1, 2)
        Priord = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
        return Priord.log_prob(samples)

    def discrepancy(self, y, y_obs):
        y = y.view(-1, 2)
        y_obs = y_obs.view(-1, 2)
        return torch.sqrt(torch.sum((y - y_obs) ** 2, dim=1))

    def calculate_log_kernel(self, y, y_obs):
        # cov_matrix = torch.eye(2, dtype=torch.float32) * self.epsilon ** 2
        # log_value = torch.distributions.MultivariateNormal(y_obs, cov_matrix).log_prob(y)
        # return log_value
        dis = self.discrepancy(y, y_obs).cpu()
        log_value = distribution.DiagGaussian(1, loc=torch.tensor([0.0]),
                                              log_scale=torch.log(torch.tensor([self.epsilon]))).log_prob(
            dis.view(-1, 1))
        return log_value

    def numberical_gradient_logABC(self, theta, y_obs, num, d=1e-1):
        # 计算数值梯度
        if theta.ndim == 1:
            theta = theta.view(1, -1)
        theta_num, theta_dim = theta.shape
        theta = theta.float()
        thetai_plus = theta.clone().repeat(theta_dim, 1).float()
        thetai_minus = theta.clone().repeat(theta_dim, 1).float()
        eye_matrix = d * torch.eye(theta_dim)
        # Randomly select an index for perturbation
        # ind = torch.randint(0, theta_dim, (theta_num,)).item()  # Ensure index is within [0, theta_num)
        # Create perturbed versions
        # thetai_plus = theta.clone()
        # thetai_minus = theta.clone()
        # Add and subtract d along the diagonal
        thetai_plus += eye_matrix
        thetai_minus -= eye_matrix
        # print(thetai_plus,thetai_minus)
        y_dim = y_obs.shape[1]
        deltai_plus = torch.zeros(theta_dim, num).to(dtype=torch.float64)
        deltai_minus = torch.zeros(theta_dim, num).to(dtype=torch.float64)
        random_seeds_int = torch.tensor([secrets.randbelow(2 ** 32) for _ in range(theta_dim)], dtype=torch.long)
        for k in range(theta_dim):
            torch.manual_seed(random_seeds_int[k])
            np.random.seed(random_seeds_int[k])
            deltai_plus[k, :] = self.discrepancy(
                self.generate_samples(thetai_plus[k,].repeat(num, 1), 1).view(-1, y_dim), y_obs)
            torch.manual_seed(random_seeds_int[k])
            np.random.seed(random_seeds_int[k])
            deltai_minus[k, :] = self.discrepancy(
                self.generate_samples(thetai_minus[k,].repeat(num, 1), 1).view(-1, y_dim), y_obs)
        mu_plus = torch.mean(deltai_plus, dim=1).view(-1)
        mu_minus = torch.mean(deltai_minus, dim=1).view(-1)
        Sigma_plus = torch.var(deltai_plus, dim=1).view(-1)
        Sigma_minus = torch.var(deltai_minus, dim=1).view(-1)
        grad = torch.zeros(theta_dim).to(dtype=torch.float64)
        for i in range(theta_dim):
            logp_plus = -1 / 2 * torch.log(Sigma_plus[i] + self.epsilon ** 2) \
                        - 1 / 2 * (mu_plus[i]) ** 2 / (Sigma_plus[i] + self.epsilon ** 2)
            logp_minus = -1 / 2 * torch.log(Sigma_minus[i] + self.epsilon ** 2) \
                         - 1 / 2 * (mu_minus[i]) ** 2 / (Sigma_minus[i] + self.epsilon ** 2)
            # print(logp_plus, logp_minus)
            grad[i] = (logp_plus - logp_minus) / (2 * d)
        return grad.view(-1, theta_dim)-theta.view(-1, theta_dim)

# Example usage
if __name__ == "__main__":
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
    # 创建一个具有某些观测值和epsilon值的ABCset对象
    # Model = Mixture_set(epsilon=0.05)
    # torch.manual_seed(0)
    # y_obs = torch.tensor([[0.0, 0.0]])
    # theta0 = Model.prior_sample(5)
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
    Model = Mixture_set(epsilon=0.05)
    y_obs = torch.tensor([[1.5, 1.5]])
    torch.manual_seed(0)
    # import csv
    # theta0 = Model.prior_sample(100000000)
    # print('theta0',theta0)
    # # 使用abc_instance对象调用generate_samples方法
    # y_samples = Model.generate_samples(theta0)
    # print('y_samples',y_samples)
    # log_kernel = Model.calculate_log_kernel(y_samples, y_obs).view(-1,1)
    # weight = torch.exp(log_kernel)
    # weight = weight/torch.sum(weight)
    # id = resample(weight,500000)
    # Result = theta0[id,]
    # print("Result",Result)
    # my_list = Result.detach().numpy()
    # csv_file = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/Mixabs_mu2/MixabsTrue_500000_0.05_mu1.5.csv"
    # with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     for row in my_list:
    #         writer.writerow(row)

    #
    num_ite = 1000000
    theta0 = torch.tensor([0, 0])
    Initial_y = Model.generate_samples(theta0)
    torch.manual_seed(0)
    Local_Proposal = distribution.DiagGaussian(2, loc=torch.zeros(1, 2), log_scale=torch.log(torch.tensor([0.35, 0.35])))
    Global_Proposal = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))#distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0]))
    folder = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/Mixabs_mu2/Mixabs"
    # filelocation = folder + "MCMCtest.csv"
    # torch.manual_seed(10)
    # Initial_theta0 = Model.prior_sample(1, ).view(-1)
    # Initial_y0 = Model.generate_samples(Initial_theta0)
    # print(filelocation)
    # GLMCMC(Model, y_obs, num_ite, Initial_theta0, Initial_y0, Local_Proposal, filelocation, 0,
    #        Global_Proposal=Global_Proposal, batch_size=0)
    #
    # for Batch_size in [5, 21, 31, 51, 81, 101]:
    #     filelocation = folder + "iSIR5_Gaussian_bs" + str(Batch_size) + ".csv"
    #     print(filelocation)
    #     GLMCMC(Model, y_obs, num_ite, theta0, Initial_y, None, filelocation, global_frequency=1,
    #            Global_Proposal=Global_Proposal, batch_size=Batch_size)
    # Global_Proposal = distribution.Uniform(2, torch.tensor([-3.0, -3.0]), torch.tensor([3.0, 3.0]))
    # for Batch_size in [5,11, 31, 51, 81, 101]:
    #     filelocation = folder + "iSIR_Uniform_bs" + str(Batch_size) + ".csv"
    #     print(filelocation)
    #     GLMCMC(Model, y_obs, num_ite, theta0, Initial_y, None, filelocation, global_frequency=1,
    #            Global_Proposal=Global_Proposal, batch_size=Batch_size)
    #
    # # for Batch_size in [5]:
    # #     for global_frequency in [0.1, 0.3, 0.5, 0.7]:
    # #         filelocation = folder + "GLMCMC_Gaussian_bs" + str(Batch_size) + '_gf' + str(global_frequency) + ".csv"
    # #         print(filelocation)
    # #         GLMCMC(Model, y_obs, num_ite, theta0, Initial_y, Local_Proposal, filelocation, global_frequency,
    # #                Global_Proposal=Global_Proposal, batch_size=Batch_size)





    # # from GlobalMCMC import GlobalMCMC
    # Global_Proposal = distribution.Uniform(2, torch.tensor([-5.0, -5.0]), torch.tensor([5.0, 5.0]))
    # # filelocation = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/Mixabs_mu2/MixabsglobalMCMC_Uniform.csv"
    # # print(filelocation)
    # # GlobalMCMC(Model, y_obs, num_ite, theta0, Initial_y, Global_Proposal, filelocation)
    folder = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/Mixabs_mu2/Mixabs"
    for Batch_size in [2,3]:
        filelocation = folder + "iSIRUniform_bs" + str(Batch_size) + ".csv"
        print(filelocation)
        GLMCMC(Model, y_obs, num_ite, theta0, Initial_y, Local_Proposal, filelocation, global_frequency=1,
               Global_Proposal=Global_Proposal, batch_size=Batch_size)
    Pro_loc = torch.tensor([[1.425, 1.425],
                            [1.425, -1.425],
                            [-1.425, 1.425],
                            [-1.425, -1.425]])
    Pro_scale = torch.tensor([[0.28] * 2] * 4)
    Global_Proposal = distribution.GaussianMixture(4, 2, Pro_loc.tolist(), Pro_scale.tolist())
    folder = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/Mixabs_mu2/Mixabs"
    for Batch_size in [2,3]:
        filelocation = folder + "iSIROptimal_bs" + str(Batch_size) + ".csv"
        print(filelocation)
        GLMCMC(Model, y_obs, num_ite, theta0, Initial_y, Local_Proposal, filelocation, global_frequency=1,
               Global_Proposal=Global_Proposal, batch_size=Batch_size)
    # filelocation = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/Mixabs_mu2/MixabsglobalMCMC_optimal3.csv"
    # print(filelocation)
    # GlobalMCMC(Model, y_obs, num_ite, theta0, Initial_y, Global_Proposal, filelocation)
    #
    #
    # Global_Proposal = distribution.DiagGaussian(2, torch.tensor([0.0, 0.0]), torch.tensor([np.log(1), np.log(1)]))
    # filelocation = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/Mixabs_mu2/MixabsglobalMCMC_Gaussian.csv"
    # print(filelocation)
    # GlobalMCMC(Model, y_obs, num_ite, theta0, Initial_y, Global_Proposal, filelocation)
    # folder = "/Users/caoxuefei/Desktop/Adaptive ABC-GL-MCMC/systhetic_example/Mixabs_mu2/Mixabs"
    # for Batch_size in [1, 10, 50]:
    #     filelocation = folder + "iSIRGaussian_bs" + str(Batch_size) + ".csv"
    #     print(filelocation)
    #     GLMCMC(Model, y_obs, num_ite, theta0, Initial_y, Local_Proposal, filelocation, global_frequency=1,
    #            Global_Proposal=Global_Proposal, batch_size=Batch_size)
    #
    #
    # for Batch_size in [5,11, 21, 31, 51, 81, 101]:
    #     filelocation = folder + "iSIRUD3_bs" + str(Batch_size) + ".csv"
    #     GLMCMC_UD(Model, y_obs, num_ite, theta0, Initial_y, None, filelocation, global_frequency=1,
    #               range_min=torch.tensor([-4.0, -4.0]), range_max=torch.tensor([4.0, 4.0]), batch_size=Batch_size)