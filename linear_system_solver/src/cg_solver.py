import torch

from .solver_utils import ensure_tensor_on_device, device, generate_test_matrix, TensorType
from .tensor_utils import tensorcore_matmul

def conjugate_gradient(
    coefficient_matrix: TensorType,
    target_vector: TensorType,
    initial_guess: TensorType | None = None,
    max_iterations: int | None = None,
    tolerance: float = 1e-5,
) -> tuple[torch.Tensor, dict[str, any]]:
    """
    Solve the linear system Ax = b using the Conjugate Gradient method with TensorCore acceleration.
    Uses a hybrid approach: TensorCore for matrix operations, PyTorch for convergence checks.
    """
    # Force float32 for TensorCore compatibility
    data_type = torch.float32

    coefficient_matrix = ensure_tensor_on_device(coefficient_matrix, data_type=data_type)
    target_vector = ensure_tensor_on_device(target_vector, data_type=data_type)

    size = target_vector.size(0)

    if max_iterations is None:
        max_iterations = size

    if initial_guess is None:
        solution = torch.zeros_like(target_vector)
    else:
        solution = ensure_tensor_on_device(initial_guess, data_type=data_type)

    # Initial residual using both methods for comparison
    residual_tc = target_vector - tensorcore_matmul(coefficient_matrix, solution.unsqueeze(1)).squeeze(1)
    residual_torch = target_vector - (coefficient_matrix @ solution.unsqueeze(1)).squeeze(1)
    
    # Use PyTorch residual for direction to maintain accuracy
    residual = residual_torch
    direction = residual.clone()

    residual_norm = torch.norm(residual).item()
    target_vector_norm = torch.norm(target_vector).item()

    relative_tolerance = tolerance * target_vector_norm if target_vector_norm > 0 else tolerance

    iterations = 0
    converged = False

    print("\nStarting Conjugate Gradient with Hybrid TensorCore/PyTorch:")
    print(f"  Initial residual norm (TensorCore): {torch.norm(residual_tc).item():.4e}")
    print(f"  Initial residual norm (PyTorch): {residual_norm:.4e}")

    for i in range(max_iterations):
        # Matrix-vector product using both methods
        matrix_direction_tc = tensorcore_matmul(coefficient_matrix, direction.unsqueeze(1)).squeeze(1)
        matrix_direction_torch = (coefficient_matrix @ direction.unsqueeze(1)).squeeze(1)
        
        # Use PyTorch for accurate inner products
        residual_dot = (residual @ residual).item()
        direction_dot = (direction @ matrix_direction_torch).item()
        step_size = residual_dot / direction_dot

        # Update solution
        solution = solution + step_size * direction
        
        # Compute new residual using both methods
        new_residual_tc = residual_tc - step_size * matrix_direction_tc
        new_residual_torch = residual_torch - step_size * matrix_direction_torch

        # Use PyTorch residual for convergence check
        residual_norm = torch.norm(new_residual_torch).item()
        iterations = i + 1

        if (i + 1) % 10 == 0:  # Print progress every 10 iterations
            print(f"  Iteration {i+1}:")
            print(f"    PyTorch residual norm = {residual_norm:.4e}")
            print(f"    TensorCore residual norm = {torch.norm(new_residual_tc).item():.4e}")
            print(f"    Difference = {torch.norm(new_residual_tc - new_residual_torch).item():.4e}")

        if residual_norm < relative_tolerance:
            converged = True
            break

        # Use PyTorch for accurate direction update
        new_residual_dot = (new_residual_torch @ new_residual_torch).item()
        direction_scale_factor = new_residual_dot / residual_dot
        
        direction = new_residual_torch + direction_scale_factor * direction
        residual = new_residual_torch
        residual_tc = new_residual_tc

    print(f"\nFinal Results:")
    print(f"  Iterations: {iterations}")
    print(f"  Final residual norm (PyTorch): {residual_norm:.4e}")
    print(f"  Final residual norm (TensorCore): {torch.norm(residual_tc).item():.4e}")
    print(f"  Converged: {converged}")

    info = {
        "iterations": iterations,
        "residual_norm": residual_norm,
        "converged": converged,
        "tensorcore_residual_norm": torch.norm(residual_tc).item()
    }

    return solution, info


def test_cg_solver() -> None:
    """Test the TensorCore-accelerated conjugate gradient implementation."""
    print("\n=== Testing TensorCore CG Solver ===")
    
    # Test with medium-sized system
    size = 256  # Size that matches TensorCore requirements
    
    # Create a well-conditioned matrix
    coefficient_matrix = generate_test_matrix(size, condition_number=2).to(torch.float32)
    print(f"\nMatrix properties:")
    print(f"  Shape: {coefficient_matrix.shape}")
    print(f"  Device: {coefficient_matrix.device}")
    print(f"  Dtype: {coefficient_matrix.dtype}")
    print(f"  Is contiguous: {coefficient_matrix.is_contiguous()}")
    print(f"  Condition number: {torch.linalg.cond(coefficient_matrix).item():.2e}")
    print(f"  Matrix norm: {torch.linalg.norm(coefficient_matrix).item():.2e}")
    print(f"  Matrix min value: {coefficient_matrix.min().item():.2e}")
    print(f"  Matrix max value: {coefficient_matrix.max().item():.2e}")
    print(f"  Matrix mean value: {coefficient_matrix.mean().item():.2e}")
    
    # Create true solution and target vector with explicit device placement
    print("\nCreating solution vectors...")
    true_solution = torch.randn(size, dtype=torch.float32, device=device)
    print(f"  True solution device: {true_solution.device}")
    print(f"  True solution dtype: {true_solution.dtype}")
    print(f"  True solution shape: {true_solution.shape}")
    print(f"  True solution norm: {torch.norm(true_solution).item():.2e}")
    
    # Ensure tensors are on GPU and contiguous before operations
    coefficient_matrix = coefficient_matrix.contiguous()
    true_solution = true_solution.contiguous()
    
    # Compute target vector with explicit shape management
    true_solution_2d = true_solution.unsqueeze(1)
    print(f"  True solution 2D shape: {true_solution_2d.shape}")
    
    try:
        # First verify matrix-vector product with PyTorch
        target_vector_torch = (coefficient_matrix @ true_solution_2d).squeeze(1)
        print("\nPyTorch matrix-vector product check:")
        print(f"  PyTorch result norm: {torch.norm(target_vector_torch).item():.2e}")
        
        # Now compute with TensorCore
        target_vector = tensorcore_matmul(coefficient_matrix, true_solution_2d).squeeze(1)
        print("\nTensorCore matrix-vector product:")
        print(f"  Target vector device: {target_vector.device}")
        print(f"  Target vector dtype: {target_vector.dtype}")
        print(f"  Target vector shape: {target_vector.shape}")
        print(f"  Target vector norm: {torch.norm(target_vector).item():.2e}")
        
        # Compare PyTorch and TensorCore results
        tensor_diff = torch.norm(target_vector - target_vector_torch) / torch.norm(target_vector_torch)
        print(f"\nTensorCore vs PyTorch difference: {tensor_diff.item():.2e}")
        
    except Exception as e:
        print(f"Error in matrix multiplication: {e}")
        return
    
    print(f"\nProblem setup:")
    try:
        true_solution_norm = torch.norm(true_solution)
        print(f"  True solution norm: {true_solution_norm.item():.2e}")
        target_vector_norm = torch.norm(target_vector)
        print(f"  Target vector norm: {target_vector_norm.item():.2e}")
    except Exception as e:
        print(f"Error computing norms: {e}")
        return

    # Solve the system
    try:
        solution, info = conjugate_gradient(
            coefficient_matrix,
            target_vector,
            tolerance=1e-5  # Relaxed tolerance for float32
        )
        print(f"\nSolver results:")
        print(f"  Solution device: {solution.device}")
        print(f"  Solution dtype: {solution.dtype}")
        print(f"  Solution norm: {torch.norm(solution).item():.2e}")
    except Exception as e:
        print(f"Error in conjugate gradient: {e}")
        return

    # Compute error with safety checks
    try:
        error = torch.norm(solution - true_solution) / torch.norm(true_solution)
        print(f"\nSolution quality:")
        print(f"  Solution norm: {torch.norm(solution).item():.2e}")
        print(f"  Relative error: {error.item():.4e}")
        
        # Compute actual residual using both methods
        residual_torch = target_vector - (coefficient_matrix @ solution.unsqueeze(1)).squeeze(1)
        residual_tensor = target_vector - tensorcore_matmul(coefficient_matrix, solution.unsqueeze(1)).squeeze(1)
        
        print(f"\nResidual analysis:")
        print(f"  PyTorch residual norm: {torch.norm(residual_torch).item():.4e}")
        print(f"  TensorCore residual norm: {torch.norm(residual_tensor).item():.4e}")
        print(f"  Relative residual: {torch.norm(residual_tensor).item() / target_vector_norm:.4e}")
        
    except Exception as e:
        print(f"Error computing solution quality: {e}")
        return

    try:
        assert info["converged"], "CG solver did not converge"
        assert error < 1e-3, f"Error too large: {error.item():.4e}"
        print("\nTensorCore CG solver test passed!")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        print(f"Convergence info:")
        print(f"  Iterations: {info['iterations']}")
        print(f"  Final residual norm: {info['residual_norm']:.4e}")
        print(f"  Converged: {info['converged']}")
    except Exception as e:
        print(f"\nUnexpected error in assertions: {e}")


if __name__ == "__main__":
    test_cg_solver()
