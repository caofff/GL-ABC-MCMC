import torch
import torch.nn.parallel
import torch.optim
from sympy.physics.units import length
from tqdm import tqdm
import numpy as np
import csv
from matplotlib import pyplot as plt
from glabcmcmc.kernel_density import KernelDensity


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


def AGLMCMC(ABCset, num_ite, Initial_theta, Initial_y, Local_Proposal, Initial_ISIR_prop,
             filelocation, global_frequency, step_size, batch_size, alpha, hat_eps_T, device=None):
    """
    Adaptive ABC-GL-MCMC algorithm implementation.

    Args:
        ABCset: Object containing ABC problem setup with required methods
        num_ite: Number of MCMC iterations
        Initial_theta: Initial parameter values (tensor)
        Initial_y: Initial observations (tensor)
        Local_Proposal: Local proposal distribution object with sample() method
        Initial_ISIR_prop: Initial ISIR proposal distribution object with forward() and log_prob() methods
        filelocation: Path to save results (optional, if None no saving)
        global_frequency: Probability of making a global move (between 0 and 1)
        step_size: Number of steps between adaptations of proposal
        batch_size: Number of samples per batch used in ISIR/global moves
        alpha: Adaptation rate controlling threshold update
        hat_eps_T: Target auxiliary threshold for discrepancy
        device: Device to run computations on (default: 'cuda' if available, else 'cpu')

    Returns:
        Dictionary containing MCMC chain and statistics (currently returns None explicitly)
    """
    # Determine device for tensor computations
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_dtype(torch.float32)  # Set default dtype for torch tensors

    # Initialize state variables
    Theta_old = Initial_theta  # Current parameter estimate
    y_old = Initial_y  # Current simulated observation
    ISIR_prop = Initial_ISIR_prop  # Initial global proposal distribution

    # Save initial parameter vector if file saving is enabled
    if filelocation is not None:
        with open(filelocation, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(Initial_theta.view(-1).detach().numpy())

    # Generate initial global proposals using ISIR proposal
    Theta_prop0, Theta_prop_log_prob0 = ISIR_prop.forward(
        batch_size * step_size)  # Shape: (batch_size*step_size, param_dim)

    # Filter out any proposals with NaNs
    has_nans = torch.isnan(Theta_prop0)
    no_nan_in_row = torch.all(~has_nans, dim=1)  # Keep rows without NaNs
    Theta_prop0 = Theta_prop0[no_nan_in_row].clone()
    Theta_prop_log_prob0 = Theta_prop_log_prob0[no_nan_in_row].clone()

    # Generate samples from model given proposed parameters
    x0 = ABCset.generate_samples(Theta_prop0, 1)  # Simulated data for each Theta_prop0

    # Compute discrepancy between simulated data and observed data
    dis0 = ABCset.discrepancy(x0)  # Should be shape (num_samples, )

    # Handle rows where discrepancy is NaN (set to large value slightly less than 1,000,000)
    nan_in_row = torch.all(torch.isnan(dis0))
    dis0[nan_in_row] = 1000000 - 5

    # Calculate log kernel (approximate likelihood) based on discrepancy
    like_log_prob0 = ABCset.calculate_log_kernel_dis(dis0)

    # Compute log weights for importance sampling: prior + likelihood - proposal log prob
    log_weight0 = ABCset.prior_log_prob(Theta_prop0) + like_log_prob0 - Theta_prop_log_prob0

    # Convert log weights to normalized weights
    weight0 = torch.exp(log_weight0)
    has_nans = torch.isnan(weight0)
    weight0[has_nans] = 0.0  # Replace any NaNs with zero weight

    # Initialize counters and result tensor
    kk = 0  # Counter for batches processed
    num_train = 0  # Number of KDE retrainings performed
    Theta_Re = torch.zeros(10000, ABCset.theta_dim)  # Store chain of parameter samples

    hat_eps = 1000000  # Large initial threshold for discrepancy
    eps_num = 0  # Counter for threshold adaptations

    # Main MCMC loop

    for i in tqdm(range(1, num_ite)):
        a = torch.rand(1)  # Random number to decide global or local step
        if a < global_frequency:
            # --------- Global move (importance sampling + KDE adaptation) ---------

            # Determine indices for current batch of proposals
            iiid = range(kk * batch_size, (kk + 1) * batch_size)

            # Concatenate current state with batch of proposed parameters and their simulated data
            Theta_prop = torch.cat((Theta_old.view(1, -1), Theta_prop0[iiid, :]), dim=0)  # (batch_size + 1, param_dim)
            x = torch.cat((y_old.view(1, -1), x0[iiid, :]), dim=0)  # Corresponding simulated data

            # Compute proposal log probability for current state using ISIR or KDE proposal (if available)
            if (num_train == 0):
                Theta_old_prop_log_prob = ISIR_prop.log_prob(Theta_old.view(1, -1).to(device)).cpu()
            else:
                Theta_old_prop_log_prob = KDE.log_prob(Theta_old.view(1, -1).to(device)).cpu()

            # Compute log likelihood (approx kernel) for current state
            Theta_old_like_log_prob = ABCset.calculate_log_kernel(y_old)

            # Calculate importance weight for current state
            Theta_old_weight = torch.exp(ABCset.prior_log_prob(Theta_old.view(1, -1)) + Theta_old_like_log_prob -
                                         Theta_old_prop_log_prob)

            # Concatenate weights for current state and the batch of proposals
            weight = torch.cat((Theta_old_weight, weight0[iiid]), dim=0)

            # Normalize weights to sum to 1
            weight = weight / torch.sum(weight)

            # Sample an index according to importance weights
            ind = weight_sampling(weight.tolist())  # weight_sampling presumably returns index or None

            # If sampled index is not the current state (index 0), update current state
            if ind is not None and ind != 0:
                Theta_old = Theta_prop[ind, :].clone()
                y_old = x[ind, :].clone().view(1, -1)

            # Record current parameter in chain
            Theta_Re[i, :] = Theta_old.clone()

            # Increment batch counter
            kk += 1

            # After processing 'step_size' batches, perform adaptation step
            if kk == step_size:
                kk = 0  # Reset batch counter

                # Adapt threshold hat_eps if above target hat_eps_T
                if hat_eps > hat_eps_T:
                    eps_num = eps_num + 1

                    # Number of discrepancies less than current threshold
                    num_a = torch.sum(dis0 < hat_eps)

                    # Filter out NaN discrepancies
                    dis0_valid = dis0[~torch.isnan(dis0)]

                    if len(dis0_valid) > 0:
                        # Update quantile q for threshold adjustment
                        value = alpha * num_a / dis0_valid.shape[0]
                        if isinstance(value, torch.Tensor):
                            q = value.clone().detach().to(dtype=dis0.dtype)
                        else:
                            q = torch.tensor(value, dtype=dis0.dtype, device=dis0.device)
                        q = torch.clamp(q, 0.0, 1.0)

                        # Update hat_eps to the q-th quantile of discrepancies
                        hat_eps = torch.quantile(dis0_valid, q)

                    # Ensure hat_eps doesn't go below target threshold
                    hat_eps = torch.max(hat_eps, torch.tensor(hat_eps_T))

                # Compute training log weights with updated threshold
                Train_like_log_prob = ABCset.calculate_log_kernel_dis(dis0, hat_eps)
                Train_log_weight = ABCset.prior_log_prob(Theta_prop0) + Train_like_log_prob - Theta_prop_log_prob0
                Train_weight = torch.exp(Train_log_weight)

                # Zero out weights corresponding to NaN discrepancies
                Train_weight[nan_in_row] = 0.0

                # Select training samples with positive weights
                Train_Theta = Theta_prop0[Train_weight > 0]
                Train_weight = Train_weight[Train_weight > 0]

                # Normalize training weights
                Train_weight = Train_weight / torch.sum(Train_weight)

                # Fit KDE to training weighted samples (KernelDensity assumed defined elsewhere)
                KDE = KernelDensity(bandwidth='silverman', device=device)
                KDE.fit(Train_Theta, Train_weight)

                num_train = num_train + 1  # Increment training counter

                # Sample new proposals from KDE (oversample to ensure enough valid samples)
                Theta_prop00 = KDE.sample(batch_size * step_size * 4)

                # Filter samples using prior density threshold
                prior_density = ABCset.prior_log_prob(Theta_prop00)
                Ind = prior_density > np.log(10 ** (-10))
                Theta_prop0 = Theta_prop00[Ind,][
                    range(batch_size * step_size),]  # Select first batch_size*step_size valid

                # Calculate proposal log probabilities for filtered samples
                Theta_prop_log_prob0 = KDE.log_prob(Theta_prop0)

                # Generate new simulated data for proposals
                x0 = ABCset.generate_samples(Theta_prop0, 1)

                # Compute discrepancies for new simulations
                dis0 = ABCset.discrepancy(x0)

                # Handle NaN discrepancies
                nan_in_row = torch.all(torch.isnan(dis0))
                dis0[nan_in_row] = 1000000 - 5

                # Compute log kernel for new discrepancies
                like_log_prob0 = ABCset.calculate_log_kernel_dis(dis0)

                # Compute new log weights
                log_weight0 = ABCset.prior_log_prob(Theta_prop0) + like_log_prob0 - Theta_prop_log_prob0

                # Convert to weights and handle NaNs
                weight0 = torch.exp(log_weight0)
                weight0[nan_in_row] = 0.0

        else:
            # --------- Local move (Metropolis-Hastings step) ---------

            # Propose new parameter by adding local perturbation
            Theta_prop = Local_Proposal.sample(1) + Theta_old.view(1, -1)

            # Generate simulated data for proposed parameter
            y = ABCset.generate_samples(Theta_prop, 1)
            y = y[0,].to(device)

            # Compute log acceptance ratio (prior + kernel - old prior - old kernel)
            log_acc = ABCset.prior_log_prob(Theta_prop) + ABCset.calculate_log_kernel(y) - \
                      ABCset.prior_log_prob(Theta_old.view(1, -1)) - ABCset.calculate_log_kernel(y_old)

            # Accept or reject the proposal with MH probability
            log_w = torch.log(torch.rand(1))
            if log_w < log_acc:
                Theta_old = Theta_prop.clone()
                y_old = y.clone()

            # Store current parameter sample
            Theta_Re[i, :] = Theta_old.clone()


        if filelocation is not None:
            if i % 10000 == 0 and  i+1 < num_ite:
                k = i // 10000
                with open(filelocation, 'a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerows(Theta_Re[range(max(1, (k - 1) * 10000 + 1), i + 1), :].view(-1,ABCset.theta_dim).detach().numpy())
        torch.cuda.empty_cache()
        # Clear CUDA cache to free memory
    if filelocation is not None:
        if i % 10000 != 0 and i + 1 == num_ite:
            k = i // 10000
            with open(filelocation, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(Theta_Re[range(max(1, (k) * 10000 + 1), i + 1), :].view(-1,ABCset.theta_dim).detach().numpy())

