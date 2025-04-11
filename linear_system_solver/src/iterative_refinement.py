import ctypes
import torch
from typing import Literal, Sequence

from .solver_utils import ensure_tensor_on_device, device, generate_test_matrix, TensorType
from .pytorch_solver import pytorch_direct_solve
from .cupy_solver import direct_solve, has_cupy

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

def tensorcore_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Perform matrix multiplication using TensorCore operations.
    
    Parameters:
    -----------
    a : torch.Tensor
        First input matrix
    b : torch.Tensor
        Second input matrix
        
    Returns:
    --------
    torch.Tensor
        Result of matrix multiplication
    """
    # Ensure the input tensors are contiguous, on the GPU, and in float format
    a = a.contiguous().to("cuda").float()
    b = b.contiguous().to("cuda").float()
    
    # Determine matrix dimensions
    M_total, K_total = a.shape
    N_total = b.shape[1]
    
    # Create an output tensor on the GPU
    c = torch.zeros(M_total, N_total, device="cuda", dtype=torch.float32)
    
    # Call the custom multiplication
    lib.tensorcore_matmul(
        a.data_ptr(), b.data_ptr(), c.data_ptr(),
        M_total, K_total, N_total
    )
    
    return c

# Define available precision types
PrecisionType = Literal["float16", "bfloat16", "float32", "float64"]
PRECISION_MAP: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}

def iterative_refinement(
    coefficient_matrix: TensorType,
    target_vector: TensorType,
    precision_sequence: list[torch.dtype],
    initial_guess: TensorType | None = None,
    maximum_iterations: int | None = None,
    inner_tolerance: float = 1e-5,
    outer_tolerance: float = 1e-10,
) -> tuple[torch.Tensor, dict[str, any]]:
    """
    Solve the linear system Ax = b using multi-level iterative refinement.

    Parameters:
    -----------
    coefficient_matrix : torch.Tensor or compatible tensor-like object
        The coefficient matrix A in the equation Ax = b.
    target_vector : torch.Tensor or compatible tensor-like object
        The target vector b in the equation Ax = b.
    precision_sequence : list[torch.dtype]
        Sequence of precisions to use for each refinement level.
    initial_guess : torch.Tensor or compatible tensor-like object, optional
        Initial guess for the solution.
    maximum_iterations : int, optional
        Maximum number of iterations per refinement level.
    inner_tolerance : float, optional
        Tolerance for convergence of inner solver.
    outer_tolerance : float, optional
        Tolerance for convergence of outer refinement loop.

    Returns:
    --------
    torch.Tensor
        Solution vector.
    dict
        Information about the solution process including:
        - 'iterations': number of refinement iterations
        - 'total_inner_iterations': total number of inner solver iterations
        - 'residual_norm': final residual norm
        - 'converged': whether the method converged
    """
    if isinstance(coefficient_matrix, torch.Tensor):
        data_type = coefficient_matrix.dtype
    else:
        data_type = torch.float32  # Default to float32

    coefficient_matrix = ensure_tensor_on_device(coefficient_matrix, data_type=data_type)
    target_vector = ensure_tensor_on_device(target_vector, data_type=data_type)

    if initial_guess is None:
        solution = torch.zeros_like(target_vector)
    else:
        solution = ensure_tensor_on_device(initial_guess, data_type=data_type)

    total_inner_iterations = 0
    iterations = 0
    converged = False
    residual_norm = float("inf")
    best_solution = solution
    best_residual_norm = float("inf")

    print("\nStarting Iterative Refinement:")
    while not converged and iterations < len(precision_sequence):
        current_precision = precision_sequence[iterations]
        print(f"\nIteration {iterations + 1}, Precision: {current_precision}")

        # Convert to current precision
        coefficient_matrix_current = coefficient_matrix.to(current_precision)
        target_vector_current = target_vector.to(current_precision)
        solution_current = solution.to(current_precision)

        # Compute residual in current precision
        mat_mul_re = tensorcore_matmul(coefficient_matrix_current, solution_current.unsqueeze(1)).squeeze(1)
        residual_current = target_vector_current - mat_mul_re
        print(f"  Current residual norm: {torch.norm(residual_current).item():.2e}")

        # Solve correction equation in current precision
        correction, info = pytorch_direct_solve(
            coefficient_matrix_current,
            residual_current,
            max_iterations=maximum_iterations,
            tolerance=inner_tolerance,
        )

        if not info["converged"]:
            print("  Direct solver failed to converge, trying smaller correction")
            correction = correction * 0.1  # Damping factor

        total_inner_iterations += info["iterations"]

        # Update solution in current precision
        solution_current = solution_current + correction
        print(f"  Correction norm: {torch.norm(correction).item():.2e}")

        # Convert back to original precision
        solution = solution_current.to(data_type)

        # Check convergence
        mat_mul_con = tensorcore_matmul(coefficient_matrix_current, solution.unsqueeze(1)).squeeze(1)
        residual = target_vector_current - mat_mul_con
        residual_norm = torch.norm(residual).item()
        
        # Keep track of best solution
        if residual_norm < best_residual_norm:
            best_solution = solution.clone()
            best_residual_norm = residual_norm

        print(f"  New residual norm: {residual_norm:.2e}")
        converged = residual_norm < outer_tolerance

        iterations += 1

    # Return the best solution found
    if not converged:
        print("\nWarning: Using best solution found as fallback")
        solution = best_solution
        residual_norm = best_residual_norm

    return solution, {
        "iterations": iterations,
        "total_inner_iterations": total_inner_iterations,
        "residual_norm": residual_norm,
        "converged": converged,
    }


def test_iterative_refinement() -> None:
    """Test the iterative refinement implementation with single precision."""
    size = 256

    # Create a well-conditioned matrix in float32
    coefficient_matrix = generate_test_matrix(size, condition_number=2).to(torch.float32)
    
    # Debug prints
    print("\nMatrix properties:")
    print(f"  Matrix shape: {coefficient_matrix.shape}")
    print(f"  Matrix dtype: {coefficient_matrix.dtype}")
    print(f"  Matrix device: {coefficient_matrix.device}")
    print(f"  Matrix condition number: {torch.linalg.cond(coefficient_matrix).item():.2e}")
    print(f"  Matrix norm: {torch.linalg.norm(coefficient_matrix).item():.2e}")
    
    true_solution = torch.randn(size, dtype=torch.float32, device=device)
    print(f"\nTrue solution properties:")
    print(f"  Solution norm: {torch.linalg.norm(true_solution).item():.2e}")
    
    # Compute target vector
    target_vector = tensorcore_matmul(coefficient_matrix, true_solution.unsqueeze(1)).squeeze(1)
    print(f"\nTarget vector properties:")
    print(f"  Target vector norm: {torch.linalg.norm(target_vector).item():.2e}")
    
    # Verify the system
    print("\nVerifying system:")
    verify_mul = tensorcore_matmul(coefficient_matrix, true_solution.unsqueeze(1)).squeeze(1)
    verify_error = torch.norm(verify_mul - target_vector) / torch.norm(target_vector)
    print(f"  Initial system error: {verify_error.item():.2e}")

    # Test with float32 only
    precision_sequence = [torch.float32]  # Use only float32
    print(f"\nTesting with single precision (float32)")
    
    solution, info = iterative_refinement(
        coefficient_matrix,
        target_vector,
        precision_sequence=precision_sequence,
        outer_tolerance=1e-5,  # Relaxed tolerance for float32
        inner_tolerance=1e-4,  # Relaxed inner tolerance
    )

    error = torch.norm(solution - true_solution) / torch.norm(true_solution)
    print(f"  Refinement iterations: {info['iterations']}")
    print(f"  Total inner iterations: {info['total_inner_iterations']}")
    print(f"  Final residual norm: {info['residual_norm']:.4e}")
    print(f"  Relative error: {error.item():.4e}")
    print(f"  Converged: {info['converged']}")
    
    # Debug the solution
    print("\nSolution properties:")
    print(f"  Solution norm: {torch.linalg.norm(solution).item():.2e}")
    verify_sol = tensorcore_matmul(coefficient_matrix, solution.unsqueeze(1)).squeeze(1)
    final_residual = torch.norm(verify_sol - target_vector) / torch.norm(target_vector)
    print(f"  Final system error: {final_residual.item():.2e}")

    # Only check if error is reasonable
    assert error < 1e-3, f"Error too large: {error.item():.4e}"
    print("\nIterative refinement test passed!")


if __name__ == "__main__":
    test_iterative_refinement()
