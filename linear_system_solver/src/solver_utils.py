import torch

# Assert that CUDA is available since we require a CUDA device for all operations
assert torch.cuda.is_available(), "CUDA device is required but not found. Please check your GPU installation."

# Get CUDA device globally
device = torch.device("cuda:0")

# Type alias for tensors
TensorType = torch.Tensor | list[float]


def ensure_tensor_on_device(tensor: TensorType, data_type: torch.dtype = torch.float64) -> torch.Tensor:
    """
    Ensures input is a PyTorch tensor on the CUDA device with the specified data type.

    Parameters:
    -----------
    tensor : torch.Tensor or list
        The input to convert to a CUDA tensor.
    data_type : torch.dtype
        The desired data type for the tensor.

    Returns:
    --------
    torch.Tensor
        The input as a PyTorch tensor on the CUDA device.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.to(device=device, dtype=data_type)
    return torch.tensor(tensor, dtype=data_type, device=device)


def generate_test_matrix(
    size: int, condition_number: float | None = None, data_type: torch.dtype = torch.float64
) -> torch.Tensor:
    """
    Generate a random symmetric positive definite test matrix.

    Parameters:
    -----------
    size : int
        Size of the matrix (size x size).
    condition_number : float, optional
        Desired condition number of the matrix.
        If None, a random condition number will be used.
    data_type : torch.dtype
        Data type of the generated matrix.

    Returns:
    --------
    torch.Tensor
        Random symmetric positive definite matrix.
    """
    random_matrix = torch.randn(size, size, dtype=data_type, device=device)
    orthogonal_matrix, upper_triangular = torch.linalg.qr(random_matrix)

    if condition_number is None:
        eigenvalues = torch.rand(size, dtype=data_type, device=device) * 999 + 1
    else:
        eigenvalues = torch.linspace(1, condition_number, size, dtype=data_type, device=device)

    diagonal_matrix = torch.diag(eigenvalues)
    matrix = orthogonal_matrix @ diagonal_matrix @ orthogonal_matrix.T
    matrix = 0.5 * (matrix + matrix.T)

    return matrix
