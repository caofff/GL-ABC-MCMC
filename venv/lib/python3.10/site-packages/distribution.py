import torch
import torch.nn as nn
import numpy as np
from scipy.stats import gamma
from torch.distributions import Normal

class BaseDistribution:
    """
    Base distribution of a flow-based model
    Parameters do not depend of target variable (as is the case for a VAE encoder)
    """

    def __init__(self):
        super().__init__()

    def forward(self, num_samples=1):
        """Samples from base distribution and calculates log probability

        Args:
          num_samples: Number of samples to draw from the distribution

        Returns:
          Samples drawn from the distribution, log probability
        """
        raise NotImplementedError

    def log_prob(self, z):
        """Calculate log probability of batch of samples

        Args:
          z: Batch of random variables to determine log probability for

        Returns:
          log probability for each batch element
        """
        raise NotImplementedError

    def sample(self, num_samples=1, **kwargs):
        """Samples from base distribution

        Args:
          num_samples: Number of samples to draw from the distribution

        Returns:
          Samples drawn from the distribution
        """
        z, _ = self.forward(num_samples, **kwargs)
        return z

class Uniform(BaseDistribution):
    """
    Multivariate uniform distribution
    """

    def __init__(self, shape, low=torch.tensor([-2.0]), high=torch.tensor([2.0])):
        """Constructor

        Args:
          shape: Tuple with shape of data, if int shape has one dimension
          low: Lower bound of uniform distribution
          high: Upper bound of uniform distribution
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.low = low
        self.high = high
        self.log_prob_val = - torch.log(torch.prod(self.high - self.low))

    def forward(self, num_samples=1, context=None):
        eps = torch.rand(
            (num_samples,) + self.shape, dtype=self.low.dtype, device=self.low.device
        )
        z = self.low + (self.high - self.low) * eps
        log_p = self.log_prob_val * torch.ones(num_samples, device=self.low.device)
        return z, log_p

    def log_prob(self, z, context=None):
        log_p = self.log_prob_val * torch.ones(z.shape[0], device=z.device)
        out_range = torch.logical_or(z < self.low, z > self.high)
        ind_inf = torch.any(torch.reshape(out_range, (z.shape[0], -1)), dim=-1)
        log_p[ind_inf] = -np.inf
        return log_p



class Gamma(BaseDistribution):
    """
    Multivariate independent Gamma distribution.
    """

    def __init__(self, Shape, Rate):
        """Constructor

        Args:
          shape: Shape parameter of the Gamma distribution, can be a scalar or a vector.
          rate: Rate parameter of the Gamma distribution (1/scale), can be a scalar or a vector.
        """
        super().__init__()
        self.Shape = Shape.numpy()
        self.Rate = Rate.numpy()

    def forward(self, num_samples=1, context=None):
        """Generate samples from the Gamma distribution.

        Args:
          num_samples: Number of samples to generate.
          context: Additional context (not used in this implementation).

        Returns:
          z: Generated samples.
          log_p: Log probability of the generated samples.
        """
        size = (num_samples,) + tuple(self.Shape.shape)
        z = gamma.rvs(self.Shape, scale=1/self.Rate, size=size)
        z = torch.tensor(z)
        log_p = self.log_prob(z)
        return z, log_p

    def log_prob(self, z, context=None):
        """Compute the log probability of the samples.

        Args:
          z: Samples from the Gamma distribution.
          context: Additional context (not used in this implementation).

        Returns:
          log_p: Log probability of the samples.
        """
        p = gamma.pdf(z.numpy(), self.Shape, scale=1 / self.Rate)
        epsilon = 0
        # 使用 epsilon 作为阈值来避免除以零的问题
        log_p = np.where(p > epsilon, np.log(p), -np.inf)
        return torch.sum(torch.tensor(log_p),dim=1)





class DiagGaussian(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """

    def __init__(self, shape, loc, log_scale):
        """Constructor

        Args:
          shape: Tuple with shape of data, if int shape has one dimension
          trainable: Flag whether to use trainable or fixed parameters
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        self.loc = loc
        self.log_scale = log_scale

    def forward(self, num_samples=1, context=None):
        eps = torch.randn(
            (num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device
        )
        z = self.loc + torch.exp(self.log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            self.log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1))
        )
        return z, log_p

    def log_prob(self, z, context=None):
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            self.log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(self.log_scale), 2),
            list(range(1, self.n_dim + 1)),
        )
        return log_p

    def cdf(self, z):
        """
        Cumulative distribution function (CDF) for the Diagonal Gaussian distribution.

        Args:
          z: Input tensor for which to compute the CDF.

        Returns:
          cdf_value: The CDF values at the input tensor z.
        """
        normal_dist = Normal(self.loc, torch.exp(self.log_scale))
        cdf_value = normal_dist.cdf(z)

        # For multivariate, we can either return the product or the joint cdf.
        # Here we return the joint cdf assuming independence.
        joint_cdf = torch.prod(cdf_value, dim=-1)

        return joint_cdf

    def register_buffer(self, param, param1):
        pass


class GaussianMixture(BaseDistribution):
    """
    Mixture of Gaussians with diagonal covariance matrix
    """

    def __init__(
        self, n_modes, dim, loc=None, scale=None, weights=None
    ):
        """Constructor

        Args:
          n_modes: Number of modes of the mixture model
          dim: Number of dimensions of each Gaussian
          loc: List of mean values
          scale: List of diagonals of the covariance matrices
          weights: List of mode probabilities
        """
        super().__init__()

        self.n_modes = n_modes
        self.dim = dim

        if loc is None:
            loc = np.random.randn(self.n_modes, self.dim)
        loc = np.array(loc)[None, ...]
        if scale is None:
            scale = np.ones((self.n_modes, self.dim))
        scale = np.array(scale)[None, ...]
        if weights is None:
            weights = np.ones(self.n_modes)
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)
        self.loc = nn.Parameter(torch.tensor(1.0 * loc))
        self.log_scale = nn.Parameter(torch.tensor(np.log(1.0 * scale)))
        self.weight_scores = nn.Parameter(torch.tensor(np.log(1.0 * weights)))

    def forward(self, num_samples=1):
        # Get weights
        weights = torch.softmax(self.weight_scores, 1)

        # Sample mode indices
        mode = torch.multinomial(weights[0, :], num_samples, replacement=True)
        mode_1h = nn.functional.one_hot(mode, self.n_modes)
        mode_1h = mode_1h[..., None]

        # Get samples
        eps_ = torch.randn(
            num_samples, self.dim, dtype=self.loc.dtype, device=self.loc.device
        )
        scale_sample = torch.sum(torch.exp(self.log_scale) * mode_1h, 1)
        loc_sample = torch.sum(self.loc * mode_1h, 1)
        z = eps_ * scale_sample + loc_sample
        # Compute log probability
        eps = (z[:, None, :] - self.loc) / torch.exp(self.log_scale)
        log_p = (
            -0.5 * self.dim * np.log(2 * np.pi)
            + torch.log(weights)
            - 0.5 * torch.sum(torch.pow(eps, 2), 2)
            - torch.sum(self.log_scale, 2)
        )
        log_p = torch.logsumexp(log_p, 1)

        return z, log_p

    def log_prob(self, z):
        # Get weights
        weights = torch.softmax(self.weight_scores, 1)

        # Compute log probability
        if self.dim == 1:
            eps = (z[:, None] - self.loc) / torch.exp(self.log_scale)
            log_p = (
                    -0.5 * self.dim * np.log(2 * np.pi)
                    + torch.log(weights)
                    - 0.5 * torch.sum(torch.pow(eps, 2), 1)
                    - torch.sum(self.log_scale, 1)
            )
        else:
            eps = (z[:, None, :] - self.loc) / torch.exp(self.log_scale)
            log_p = (
                    -0.5 * self.dim * np.log(2 * np.pi)
                    + torch.log(weights)
                    - 0.5 * torch.sum(torch.pow(eps, 2), 2)
                    - torch.sum(self.log_scale, 2)
            )
        log_p = torch.logsumexp(log_p, 1)

        return log_p

