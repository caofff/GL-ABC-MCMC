import torch
import glabcmcmc.distribution as distribution

class KernelDensity:
    def __init__(self, bandwidth='silverman', device=None):
        """
        Initialize the KernelDensity estimator.

        Args:
            bandwidth (float, str, or torch.Tensor): Bandwidth for the Gaussian kernels.
                Can be a scalar, 'silverman', 'scott', or a vector of length n_features.
            device (str, optional): Device to perform computations on ('cuda' or 'cpu').
        """
        self.bandwidth = bandwidth
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.X = None
        self.weights = None
        self.n_samples = 0
        self.dim = None
        self._fitted = False

    def _compute_bandwidth(self, X,weights):
        """Compute bandwidth if using automatic methods."""
        if isinstance(self.bandwidth, str):
            n = len(X)
            if self.bandwidth == 'silverman':
                # Silverman's rule of thumb
                h = (n * (self.dim + 2) / 4.) ** (-1. / (self.dim + 4))
            elif self.bandwidth == 'scott':
                # Scott's rule
                h = n ** (-1. / (self.dim + 4))
            else:
                raise ValueError("bandwidth should be 'silverman', 'scott' or a float")
            # Use the standard deviation of the data
            std = self.weighted_std(X, weights, unbiased=True, dim=0)
            return h * std
        return self.bandwidth

    def weighted_std(self,X, weights, unbiased=True, dim=0):
        """
        计算带权重的标准差。

        Args:
            X: Tensor，形状 (n_samples, n_features)
            weights: Tensor，形状 (n_samples,)
            unbiased: 是否无偏（默认True）
            dim: 计算维度（默认特征维度0）

        返回:
            Tensor，形状 (n_features,)
        """
        # 归一化权重
        w = weights / weights.sum()

        # 计算加权均值
        mean = (w.unsqueeze(-1) * X).sum(dim=dim)

        # 计算加权方差
        diff = X - mean

        weighted_var = (w.unsqueeze(-1) * diff ** 2).sum(dim=dim)

        if unbiased:
            # 有偏校正因子
            correction = 1 - (w ** 2).sum()
            weighted_var = weighted_var / correction.clamp(min=1e-10)  # 防止除零

        return weighted_var.sqrt()

    def fit(self, X, weights=None):
        """
        Fit the kernel density estimator to the data.

        Args:
            X (torch.Tensor): Input samples of shape (n_samples, n_features).
            weights (torch.Tensor, optional): Weights for each sample.
        """
        self.X = X.to(self.device)
        self.n_samples, self.dim = X.shape

        # Handle weights
        if weights is None:
            self.weights = torch.ones(self.n_samples, device=self.device) / self.n_samples
        else:
            weights = weights.to(self.device)
            self.weights = (weights / weights.sum()).to(self.device)

        # Compute bandwidth
        self.bandwidth = self._compute_bandwidth(self.X,self.weights)
        if isinstance(self.bandwidth, torch.Tensor):
            self.bandwidth = self.bandwidth.to(self.device)

        self._fitted = True
        return self

    def log_prob(self, x):
        """
        Compute the log probability density at x.

        Args:
            x (torch.Tensor): Points at which to evaluate the log density.
                Shape (n_points, n_features).
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before computing probabilities")

        x = x.to(self.device)
        n_points = x.shape[0]

        # Handle bandwidth
        if isinstance(self.bandwidth, (int, float)):
            bandwidth = torch.ones(self.dim, device=self.device) * self.bandwidth
        else:
            bandwidth = self.bandwidth.to(self.device)

        # Vectorized computation
        x = x.unsqueeze(1)  # (n_points, 1, n_features)
        X = self.X.unsqueeze(0)  # (1, n_samples, n_features)
        diff = (x - X) / bandwidth  # (n_points, n_samples, n_features)
        log_kernel = -0.5 * (diff ** 2).sum(dim=-1)  # (n_points, n_samples)
        log_kernel -= 0.5 * self.dim * torch.log(torch.tensor(2 * torch.pi, device=self.device))
        log_kernel -= torch.log(bandwidth).sum()

        # Weighted sum
        log_weighted = log_kernel + torch.log(self.weights + 1e-10)
        log_densities = torch.logsumexp(log_weighted, dim=1)

        return log_densities

    def sample(self, n_samples=1, return_log_prob=False):
        """
        Draw samples from the kernel density estimate.

        Args:
            n_samples (int): Number of samples to generate.
            return_log_prob (bool): If True, also return the log probability of each sample.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before sampling")

        # Sample indices according to weights
        indices = torch.multinomial(self.weights, n_samples, replacement=True)

        # Sample from the selected kernels
        if isinstance(self.bandwidth, (int, float)):
            bandwidth = torch.ones(self.dim, device=self.device) * self.bandwidth
        else:
            bandwidth = self.bandwidth.to(self.device)

        noise = torch.randn(n_samples, self.dim, device=self.device) * bandwidth
        samples = self.X[indices] + noise

        if return_log_prob:
            log_probs = self.log_prob(samples)
            return samples, log_probs
        return samples

    def forward(self, n_samples=1):
        if not self._fitted:
            raise RuntimeError("Must call fit() before sampling")

        indices = torch.multinomial(self.weights, n_samples, replacement=True)

        if isinstance(self.bandwidth, (int, float)):
            bandwidth = torch.ones(self.dim, device=self.device) * self.bandwidth
        else:
            bandwidth = self.bandwidth.to(self.device)

        Prop = distribution.DiagGaussian(self.dim, torch.zeros(self.dim, device=self.device),
                                         log_scale=torch.log(bandwidth))
        noise = Prop.sample(n_samples)
        samples = self.X[indices] + noise

        # 计算采样点的log_prob
        log_prob = self.log_prob(samples)

        return samples, log_prob
