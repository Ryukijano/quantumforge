from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qiskit import QuantumCircuit as QiskitCircuit


def qiskit_to_qhybrid_json(qc: QiskitCircuit) -> str:
    """Convert a Qiskit circuit to qhybrid JSON format.

    Args:
        qc: Qiskit QuantumCircuit

    Returns:
        JSON string representing the circuit
    """
    # Transpile to a small basis we can simulate in Rust.
    #
    # Qiskit will decompose most gates into {"u", "cx"} which maps cleanly onto
    # our `GateType::{U3, CX}`.
    try:
        from qiskit import transpile  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("qiskit is required for qiskit_to_qhybrid_json") from e

    qc = transpile(qc, basis_gates=["u", "cx"], optimization_level=0)

    n_qubits = qc.num_qubits
    gates = []

    for instruction in qc.data:
        operation = instruction.operation
        qubits = [qc.find_bit(q).index for q in instruction.qubits]

        # Ignore non-unitary/metadata ops for statevector execution.
        if operation.name in {"barrier", "measure"}:
            continue

        gate_type = operation.name.lower()

        # Map Qiskit names to our GateType names (serde enum names).
        name_map = {
            "id": "I",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "h": "H",
            "s": "S",
            "sdg": "Sdg",
            "t": "T",
            "tdg": "Tdg",
            "rx": "RX",
            "ry": "RY",
            "rz": "RZ",
            "p": "P",
            "u": "U3",
            "u3": "U3",
            "cx": "CX",
            "cy": "CY",
            "cz": "CZ",
            "ccx": "CCX",
        }

        if gate_type not in name_map:
            raise NotImplementedError(f"Unsupported gate for qhybrid: {operation.name}")

        mapped_type = name_map[gate_type]
        params = [float(p) for p in operation.params]
        
        # Adjust params for U3 if needed
        if mapped_type == "U3" and len(params) == 3:
            gate_data = {
                "gate_type": {"U3": params},
                "qubits": qubits,
                "parameters": params
            }
        elif mapped_type in ["RX", "RY", "RZ", "P"]:
            gate_data = {
                "gate_type": {mapped_type: params[0]},
                "qubits": qubits,
                "parameters": params
            }
        else:
            gate_data = {
                "gate_type": mapped_type,
                "qubits": qubits,
                "parameters": []
            }
            
        gates.append(gate_data)

    circuit_data = {
        "n_qubits": n_qubits,
        "gates": gates,
        "name": qc.name
    }

    return json.dumps(circuit_data)

