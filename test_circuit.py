"""Small generalized Shor demonstration. Local simulation is the default.

The reversible permutation construction avoids dense matrices, but still scales
exponentially in work qubits. Deliberately limited to N <= 31 for quantum runs.
"""
import argparse
from fractions import Fraction
from math import gcd
import random

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import QFTGate
from qiskit.transpiler import generate_preset_pass_manager


# ======================================
# 1. USER INPUT AND INITIALIZATION
# ======================================
def classical_factors(NS, A):
    if NS <= 2:
        raise ValueError("Enter an integer greater than 2.")
    if not 2 <= A < NS:
        raise ValueError("a must satisfy 2 <= a < N.")
    if NS % 2 == 0:
        return (2, NS // 2)
    d = gcd(A, NS)
    return (d, NS // d) if d > 1 else None


# ======================================
# 2. MODULAR MULTIPLICATION GATE
# ======================================
def _swap_states(circuit, x, y):
    """Swap two basis states via a Gray path, leaving all others unchanged."""
    path = [x]
    for bit in range(circuit.num_qubits):
        if (x ^ y) >> bit & 1:
            path.append(path[-1] ^ (1 << bit))
    edges = list(zip(path, path[1:]))
    for left, right in edges + edges[-2::-1]:
        target = (left ^ right).bit_length() - 1
        controls = [i for i in range(circuit.num_qubits) if i != target]
        zeros = [i for i in controls if not (left >> i & 1)]
        for i in zeros:
            circuit.x(i)
        circuit.mcx(controls, target)
        for i in zeros:
            circuit.x(i)


def MBmodNS(a, k, NS):
    """Multiply by a^(2^k) mod N, with states x >= N left unchanged."""
    if not 3 <= NS <= 31:
        raise ValueError("Quantum demonstration supports 3 <= N <= 31.")
    if k < 0 or gcd(a, NS) != 1:
        raise ValueError("Require k >= 0 and gcd(a, N) = 1 for reversibility.")
    B = pow(a, 2**k, NS)
    n = (NS - 1).bit_length()
    circuit = QuantumCircuit(n, name=f"M_{B}")
    # For N = 2^n - 1, multiplication by a power of two is bit rotation.
    # Both 0 and the unused all-ones state remain fixed.
    if NS == 2**n - 1 and B & (B - 1) == 0:
        for _ in range(B.bit_length() - 1):
            for i in reversed(range(n - 1)):
                circuit.swap(i, i + 1)
        return circuit.to_gate()
    visited = set()
    # Decompose the permutation into cycles, then basis-state swaps.
    # This constructs the multiplication map, not the order of a from |1>.
    for start in range(NS):
        if start in visited:
            continue
        cycle = []
        x = start
        while x not in visited:
            visited.add(x)
            cycle.append(x)
            x = B * x % NS
        for y in cycle[1:]:
            _swap_states(circuit, cycle[0], y)
    return circuit.to_gate()


def controlled_MBmodNS(a, k, NS):
    return MBmodNS(a, k, NS).control(1)


# ======================================
# 3. QUANTUM PHASE ESTIMATION
# ======================================
def build_circuit(NS, A):
    if not 3 <= NS <= 31 or gcd(A, NS) != 1:
        raise ValueError("Require N in 3..31 and a coprime to N.")
    NQ = (NS - 1).bit_length()
    CNQ = 2 * NQ
    control = QuantumRegister(CNQ, name="C")
    target = QuantumRegister(NQ, name="T")
    output = ClassicalRegister(CNQ, name="out")
    circuit = QuantumCircuit(control, target, output)
    circuit.x(target[0])
    for k, qubit in enumerate(control):
        circuit.h(qubit)
        if pow(A, 2**k, NS) != 1:
            circuit.append(controlled_MBmodNS(A, k, NS), [qubit] + list(target))

    # ======================================
    # 4. INVERSE QFT AND MEASUREMENT
    # ======================================
    circuit.append(QFTGate(CNQ).inverse(), control)
    circuit.measure(control, output)
    # Full QFT includes swaps. out[0] is least significant: do NOT reverse strings.
    return circuit


# ======================================
# 5. TRANSPILATION / 6. EXECUTION
# ======================================
def run_circuit(circuit, shots=1024, seed=7, backend_name=None,
                submit_ibm=False, noise_suppression=False):
    if shots <= 0:
        raise ValueError("shots must be positive.")
    if backend_name and not submit_ibm:
        raise ValueError("IBM execution requires explicit --submit-ibm approval.")
    if submit_ibm and not backend_name:
        raise ValueError("Specify --backend for IBM execution.")
    if backend_name:
        from qiskit_ibm_runtime import QiskitRuntimeService
        from qiskit_ibm_runtime.executor_sampler import Sampler
        # Uses a previously saved account. No credentials are written or embedded.
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
        sampler = Sampler(mode=backend)
        if noise_suppression:
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XpXm"
            sampler.options.twirling.enable_gates = True
        sampler.options.environment.job_tags = ["SHORS_GENERALIZED"]
    else:
        from qiskit_aer import AerSimulator
        from qiskit.primitives import BackendSamplerV2
        backend = AerSimulator(method="statevector")
        sampler = BackendSamplerV2(backend=backend, options={"seed_simulator": seed})
    pm = generate_preset_pass_manager(optimization_level=2, backend=backend,
                                      seed_transpiler=seed)
    transpiled_circuit = pm.run(circuit)
    print("Total qubits:", circuit.num_qubits)
    print("2q-depth:", transpiled_circuit.depth(lambda x: x.operation.num_qubits == 2))
    print("2q-size:", transpiled_circuit.size(lambda x: x.operation.num_qubits == 2))
    job = sampler.run([transpiled_circuit], shots=shots)
    if backend_name:
        print("Job ID:", job.job_id())
    return job.result()[0].data.out.get_counts()


# ======================================
# 7. PHASE ESTIMATION RESULTS
# ======================================
def phase_rows(counts, CNQ, NS):
    rows = []
    for bitstring, count in sorted(counts.items(), key=lambda item: item[1], reverse=True):
        phase = Fraction(int(bitstring, 2), 2**CNQ)
        frac = phase.limit_denominator(NS - 1)
        rows.append((bitstring, count, phase, frac, frac.denominator))
    return rows


# ======================================
# 8. CLASSICAL FACTOR EXTRACTION
# ======================================
def extract_factors(counts, CNQ, NS, A):
    for bitstring, count, phase, frac, r in phase_rows(counts, CNQ, NS):
        if count <= 0 or phase == 0 or r <= 1:
            continue
        # Require a fraction consistent with rounding to this measurement bin.
        if abs(phase - frac) > Fraction(1, 2 * 2**CNQ):
            continue
        # Only test the measured denominator. Never scan its multiples.
        if pow(A, r, NS) != 1 or r % 2:
            continue
        half_power = pow(A, r // 2, NS)
        if half_power in (1, NS - 1):
            continue
        for d in (gcd(half_power - 1, NS), gcd(half_power + 1, NS)):
            if 1 < d < NS:
                return tuple(sorted((d, NS // d))), r
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("N", type=int, nargs="?")
    parser.add_argument("--a", type=int)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backend", help="IBM backend name; omitted means local")
    parser.add_argument("--submit-ibm", action="store_true", help="Explicitly authorize an IBM job")
    parser.add_argument("--noise-suppression", action="store_true")
    args = parser.parse_args()
    NS = args.N if args.N is not None else int(input("Enter a number to factor: "))
    if NS <= 2:
        parser.error("Enter an integer greater than 2.")
    A = args.a if args.a is not None else random.Random(args.seed).randrange(2, NS)
    try:
        factors = classical_factors(NS, A)
        print("Selected a:", A)
        if factors:
            print("Classical shortcut. Factors:", *factors)
            return
        circuit = build_circuit(NS, A)
        counts = run_circuit(circuit, args.shots, args.seed, args.backend,
                             args.submit_ibm, args.noise_suppression)
    except ValueError as error:
        parser.error(str(error))
    print("Bitstring  Counts  Phase  Fraction  Candidate r (top 20)")
    for bits, count, phase, frac, r in phase_rows(counts, 2 * (NS - 1).bit_length(), NS)[:20]:
        print(bits, count, float(phase), str(frac), r)
    result = extract_factors(counts, 2 * (NS - 1).bit_length(), NS, A)
    if result:
        factors, r = result
        print("Verified order candidate:", r)
        print("Factors:", *factors)
    else:
        print("No useful order recovered. Try another a or more shots.")
        print("Prime inputs and some composite/base choices cannot yield factors this way.")


if __name__ == "__main__":
    main()
