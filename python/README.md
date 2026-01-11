# qhybrid-kernels: Python Bindings for qhybrid

This package provides high-performance Rust kernels for Qiskit-based quantum simulations.

## Installation

You can build and install the kernels directly from source using `maturin`:

```bash
pip install maturin
cd rust/qhybrid/python
maturin develop
```

## Features

- **Fast Circuit Execution**: Replace slow statevector simulations with optimized Rust kernels.
- **Advanced Noise Modeling**:
  - Pauli Channel (Monte Carlo trajectories)
  - Kraus Operator (Density Matrix)
  - Correlated Pauli Noise
- **Qiskit Adapter**: Convert `qiskit.QuantumCircuit` objects directly to `qhybrid` format.

## Example Usage

```python
from qiskit import QuantumCircuit
from qhybrid_kernels import qiskit_to_qhybrid_json, execute_quantum_circuit

# 1. Create a Qiskit circuit
qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

# 2. Convert to qhybrid JSON
circuit_json = qiskit_to_qhybrid_json(qc)

# 3. Execute in Rust
# Returns a numpy array with shape (2^n, 2) [real, imag]
statevector_ri = execute_quantum_circuit(circuit_json)

# Convert to complex
statevector = statevector_ri[:, 0] + 1j * statevector_ri[:, 1]
print(statevector)
```

## Benchmarking

To run the performance comparison:

```bash
python benchmarks/compare_simulators.py
```