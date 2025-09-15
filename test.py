import math
from fractions import Fraction
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFT
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

QiskitRuntimeService(instance="my_instance_CRN")

# ---- Helper functions ----
def c_amod15(a, power):
    """Controlled multiplication by a^power mod 15."""
    U = QuantumCircuit(4)
    for _ in range(power):
        if a in [2, 13]:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)
        if a in [7, 8]:
            U.swap(1, 2)
            U.swap(2, 3)
            U.swap(0, 1)
        if a in [4, 11]:
            U.swap(1, 3)
            U.swap(0, 2)
        if a in [7, 11, 13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = f"{a}^{power} mod 15"
    c_U = U.control()
    return c_U

def order_finding_circuit(a):
    """Construct the order finding circuit for base a mod 15."""
    n_count = 8  # number of counting qubits
    qc = QuantumCircuit(n_count + 4, n_count)

    # Initialize counting qubits in superposition
    qc.h(range(n_count))

    # Initialize auxiliary register in |1>
    qc.x(n_count)

    # Apply controlled-U operations
    for q in range(n_count):
        qc.append(c_amod15(a, 2**q), [q] + list(range(n_count, n_count + 4)))

    # Apply inverse QFT
    qc.append(QFT(num_qubits=n_count, inverse=True), range(n_count))

    # Measure
    qc.measure(range(n_count), range(n_count))
    return qc

def get_order(measurement, n_count=8):
    """Post-process measurement result to extract order r."""
    phase = int(measurement, 2) / (2**n_count)
    frac = Fraction(phase).limit_denominator(15)
    r = frac.denominator
    return r

# ---- Main Shor routine ----
def shor_factor(N=15, a=7):
    # Connect to IBM Runtime (must be logged in with `qiskit-ibm-runtime login`)
    service = QiskitRuntimeService()
    backend = service.backend("ibmq_qasm_simulator")
    sampler = Sampler(backend)

    qc = order_finding_circuit(a)
    tqc = transpile(qc, backend)

    # Run circuit
    job = sampler.run([tqc])
    result = job.result()
    counts = result[0].data.c.get_counts()

    # Pick the most probable outcome
    measurement = max(counts, key=counts.get)
    r = get_order(measurement)

    print(f"Order found: r = {r}")

    # Try to compute gcd for factors
    if r % 2 != 0:
        return None

    guess1 = math.gcd(a ** (r // 2) - 1, N)
    guess2 = math.gcd(a ** (r // 2) + 1, N)
    return guess1, guess2


# ---- Run example ----
print("\nShor’s Algorithm (N=15)")
print("--------------------------")
factors = shor_factor(N=15, a=7)
print("Factors:", factors)
