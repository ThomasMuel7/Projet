from qiskit_ibm_runtime import SamplerV2 as Sampler
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService
from dotenv import load_dotenv
from os import getenv
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit

def qrng(k: int):
    qc = QuantumCircuit(k)
    for q in range(k):
        qc.h(q)          # one coin‑flip per qubit
    qc.measure_all()
    return qc

def monobit_summary(counts, result):
    shots = sum(counts.values())
    bitstrings = result[0].data.meas.get_bitstrings()
    M = np.array([[int(b) for b in s[::-1]] for s in bitstrings], dtype=int)
    p = M.mean(axis=0)                 # per-qubit fraction of 1s
    overall = float(p.mean())
    se = np.sqrt(0.25/shots)           # rough expected fluctuation for a fair coin
    suspect = np.abs(p - 0.5) > 3*se   # rule-of-thumb: outside ±3·SE
    return p, overall, se, suspect

load_dotenv()

TOKEN = getenv('TOKEN')
INSTANCE = None  # OPTIONAL: e.g., "crn:v1:bluemix:public:quantum-computing:us-east:...:..."

# Safety check to avoid empty tokens
if not TOKEN or TOKEN.strip() in {"", "<PASTE-YOUR-IBM-QUANTUM-API-KEY-HERE>"}:
    raise ValueError("Please paste your IBM Quantum API key into TOKEN (between quotes) and run again.")

service = QiskitRuntimeService(
    channel="ibm_quantum_platform",
    token=TOKEN.strip(),
    instance=(INSTANCE.strip() if isinstance(INSTANCE, str) and INSTANCE.strip() else None),
)

k=8
qc = qrng(k)
A = service.least_busy(simulator=False, operational=True, min_num_qubits=6)
pmA = generate_preset_pass_manager(optimization_level=3, backend=A)
isaA = pmA.run(qc)

resultA = Sampler(mode=A).run([isaA], shots=4000).result()
countsA = resultA[0].data.meas.get_counts()

print(countsA, resultA)

pA, overallA, seA, flagA = monobit_summary(countsA, resultA)
print("A per-qubit P(1):", np.round(pA, 3), "overall:", round(overallA, 3), "SE~", round(seA, 4))
print("A suspect qubits:", np.where(flagA)[0].tolist())