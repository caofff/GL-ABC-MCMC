import torch
def esjd(data):
    """
    Calculate the Expected Square Jump Distance (ESJD) of the given data.

    Args:
    data (torch.Tensor): The input data with shape (N, D), where N is the number of samples and D is the dimension.

    Returns:
    torch.Tensor: The ESJD value.
    """
    # Calculate dimensions
    n_data = data.shape[0]
    dim_theta = data.shape[1]

    # Calculate differences between consecutive rows
    delta = data[1:, :] - data[:-1, :]
    n_delta = n_data - 1

    # Vectorized calculation of the covariance matrix equivalent
    re = torch.matmul(delta.T, delta) / n_delta

    # Calculate the determinant and normalize
    re = torch.det(re) ** (1 / dim_theta)

    return re.numpy()