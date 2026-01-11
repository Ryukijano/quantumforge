from .conversions import (
    complex_matrix_to_ri,
    complex_statevector_to_ri,
    kraus_1q_to_ri,
    ri_to_complex_matrix,
    ri_to_complex_statevector,
)
from .noise import (
    apply_kraus_1q_density_matrix,
    apply_pauli_channel_statevector,
    apply_correlated_pauli_noise_statevector,
    apply_cnot_error_statevector,
    expectation_value_pauli_string_py,
)
from .qiskit_adapter import (
    kraus_1q_from_qiskit,
    apply_qiskit_kraus_1q_to_density_matrix,
)
from .circuit import qiskit_to_qhybrid_json

# Import circuit execution functions from rust_kernels extension
try:
    from .rust_kernels import (
        execute_quantum_circuit,
        apply_correlated_pauli_noise_statevector as _apply_correlated_pauli_noise_statevector,
        apply_cnot_error_statevector as _apply_cnot_error_statevector,
        expectation_value_pauli_string_py as _expectation_value_pauli_string_py,
        QuantumCircuit,
    )
    # Re-export with same names for consistency
    apply_correlated_pauli_noise_statevector = _apply_correlated_pauli_noise_statevector
    apply_cnot_error_statevector = _apply_cnot_error_statevector
    expectation_value_pauli_string_py = _expectation_value_pauli_string_py
except ImportError:
    # Fallback if extension not built
    QuantumCircuit = None
    pass

__all__ = [
    "complex_statevector_to_ri",
    "ri_to_complex_statevector",
    "complex_matrix_to_ri",
    "ri_to_complex_matrix",
    "kraus_1q_to_ri",
    "apply_pauli_channel_statevector",
    "apply_kraus_1q_density_matrix",
    "apply_correlated_pauli_noise_statevector",
    "apply_cnot_error_statevector",
    "expectation_value_pauli_string_py",
    "kraus_1q_from_qiskit",
    "apply_qiskit_kraus_1q_to_density_matrix",
    "execute_quantum_circuit",
]

