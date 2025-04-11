import torch
import ctypes

# Load the TensorCore library
lib = ctypes.cdll.LoadLibrary("./libtensorcore_matmul.so")
lib.tensorcore_matmul.argtypes = [
    ctypes.c_void_p,  # pointer to A
    ctypes.c_void_p,  # pointer to B
    ctypes.c_void_p,  # pointer to C
    ctypes.c_int,     # M_total
    ctypes.c_int,     # K_total
    ctypes.c_int      # N_total
]

# TensorCore configuration (must match CUDA kernel)
M = 16
N = 16
K = 8
M_TILES = 16
N_TILES = 16
K_TILES = 32
M_TOTAL = M * M_TILES  # 256
N_TOTAL = N * N_TILES  # 256
K_TOTAL = K * K_TILES  # 256

def tensorcore_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication using TensorCore operations.
    Note: This implementation requires specific matrix dimensions due to TensorCore requirements:
    - Input A must be [M_TOTAL x K_TOTAL] = [256 x 256]
    - Input B must be [K_TOTAL x N_TOTAL] = [256 x 256]
    - Output C will be [M_TOTAL x N_TOTAL] = [256 x 256]
    
    For matrix-vector products, the vector will be padded to match requirements.
    
    Args:
        a: First input tensor (M_TOTAL x K_TOTAL)
        b: Second input tensor (K_TOTAL x N_TOTAL)
        
    Returns:
        Output tensor (M_TOTAL x N_TOTAL)
        
    Raises:
        ValueError: If tensor dimensions exceed TensorCore limits
    """
    # Input validation
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        raise ValueError("Inputs must be PyTorch tensors")
    
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError(f"Inputs must be 2D tensors. Got shapes {a.shape} and {b.shape}")
    
    # Get original dimensions
    M_orig, K_orig = a.shape
    K_check, N_orig = b.shape
    
    # Check if dimensions match TensorCore requirements
    if M_orig > M_TOTAL or K_orig > K_TOTAL or N_orig > N_TOTAL:
        raise ValueError(
            f"Input dimensions exceed TensorCore limits. "
            f"Got A: {a.shape}, B: {b.shape}. "
            f"Maximum allowed: A: ({M_TOTAL}, {K_TOTAL}), B: ({K_TOTAL}, {N_TOTAL})"
        )
    
    if K_orig != K_check:
        raise ValueError(f"Inner dimensions must match: {K_orig} != {K_check}")
    
    # Create padded tensors with exact TensorCore dimensions
    try:
        a_padded = torch.zeros(M_TOTAL, K_TOTAL, device="cuda", dtype=torch.float32)
        b_padded = torch.zeros(K_TOTAL, N_TOTAL, device="cuda", dtype=torch.float32)
        c_padded = torch.zeros(M_TOTAL, N_TOTAL, device="cuda", dtype=torch.float32)
        
        # Copy original data
        a_padded[:M_orig, :K_orig] = a.to("cuda").float()
        b_padded[:K_orig, :N_orig] = b.to("cuda").float()
    except Exception as e:
        raise RuntimeError(f"Failed to allocate padded tensors: {e}")
    
    # Ensure tensors are contiguous
    a_padded = a_padded.contiguous()
    b_padded = b_padded.contiguous()
    c_padded = c_padded.contiguous()
    
    # Verify tensors are on GPU and contiguous
    if not (a_padded.is_cuda and b_padded.is_cuda and c_padded.is_cuda):
        raise RuntimeError("Tensors must be on CUDA device")
    
    if not (a_padded.is_contiguous() and b_padded.is_contiguous() and c_padded.is_contiguous()):
        raise RuntimeError("Tensors must be contiguous")
    
    # Call the custom multiplication
    try:
        lib.tensorcore_matmul(
            a_padded.data_ptr(), b_padded.data_ptr(), c_padded.data_ptr(),
            M_TOTAL, K_TOTAL, N_TOTAL
        )
    except Exception as e:
        raise RuntimeError(f"TensorCore multiplication failed: {e}")
    
    # Extract the relevant portion of the result
    c = c_padded[:M_orig, :N_orig].clone()
    
    # Verify output
    if torch.isnan(c).any():
        raise RuntimeError("NaN values detected in output")
    
    if torch.isinf(c).any():
        raise RuntimeError("Inf values detected in output")
    
    return c 