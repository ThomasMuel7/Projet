from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit

from qiskit_ibm_runtime import QiskitRuntimeService
from dotenv import load_dotenv
from os import getenv

load_dotenv()

TOKEN = getenv('TOKEN')
INSTANCE = None  # OPTIONAL: e.g., "crn:v1:bluemix:public:quantum-computing:us-east:...:..."

# Safety check to avoid empty tokens
if not TOKEN or TOKEN.strip() in {"", "<PASTE-YOUR-IBM-QUANTUM-API-KEY-HERE>"}:
    raise ValueError("Please paste your IBM Quantum API key into TOKEN (between quotes) and run again.")

# Create the service directly (no saved account needed)
service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=TOKEN.strip(),
    instance=(INSTANCE.strip() if isinstance(INSTANCE, str) and INSTANCE.strip() else None),
)

### 1 ###
cands = service.backends(simulator=False, operational=True, min_num_qubits=6)
for b in cands: print(b.name, b.num_qubits)

A = service.least_busy(simulator=False, operational=True, min_num_qubits=6)
B = next(b for b in cands if b.name != A.name)

print(f'A: {A}', f'B: {B}')

# k‑bit QRNG
def qrng(k: int):
    qc = QuantumCircuit(k)
    for q in range(k):
        qc.h(q)          # one coin‑flip per qubit
    qc.measure_all()
    return qc

k = 8
qc = qrng(k)

pmA = generate_preset_pass_manager(optimization_level=3, backend=A)
isaA = pmA.run(qc)

pmB = generate_preset_pass_manager(optimization_level=3, backend=B)
isaB = pmB.run(qc)

from qiskit.primitives import BackendSamplerV2
from qiskit_aer import AerSimulator

# Sampled counts
sim_sampler = BackendSamplerV2(backend=AerSimulator())
sim_counts = sim_sampler.run([isaA], shots=4000).result()[0].data.meas.get_counts() #Note: We run a circuit isaA, which is optimized to run on backend A, on the simulator. This could be good to keep in mind when comparing the simulator with A and B. However, in our case the circuit is simple and there should not be a lot of optimization specific for each backend.

from qiskit_ibm_runtime import SamplerV2 as Sampler
resultA = Sampler(mode=A).run([isaA], shots=4000).result()
countsA = resultA[0].data.meas.get_counts()

import numpy as np, matplotlib.pyplot as plt

def per_qubit_p1(counts, n):
    shots = sum(counts.values())
    p = np.zeros(n, dtype=float)
    for s, c in counts.items():                 # s like '0101' (qubit 0 is rightmost)
        for j, ch in enumerate(reversed(s)):    # map column 0 -> qubit 0
            if ch == '1':
                p[j] += c
    return p / max(shots, 1)

k = qc.num_qubits  # or isaA.num_qubits
p_sim = per_qubit_p1(sim_counts, k)
p_A   = per_qubit_p1(countsA,    k)

print('p_sim', p_sim)
print('p_A', p_A)

x = np.arange(k); w = 0.42
plt.figure()
plt.bar(x - w/2, p_sim, width=w, label="Aer (sampled)")
plt.bar(x + w/2, p_A,   width=w, label=A.name)
plt.xlabel("Qubit index"); plt.ylabel("Fraction of 1s (P(1))"); plt.title("Per-qubit bias")
plt.xticks(x, [f"q{j}" for j in range(k)]); plt.ylim(0, 1); plt.legend(); plt.tight_layout(); plt.show()