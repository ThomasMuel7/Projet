from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import QuantumCircuit

from qiskit_ibm_runtime import QiskitRuntimeService
from dotenv import load_dotenv
from os import getenv

import itertools

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
sim_results = sim_sampler.run([isaA], shots=4000).result()
sim_counts = sim_results[0].data.meas.get_counts()


from qiskit_ibm_runtime import SamplerV2 as Sampler
resultA = Sampler(mode=A).run([isaA], shots=4000).result()
countsA = resultA[0].data.meas.get_counts()
resultB = Sampler(mode=B).run([isaB], shots=4000).result()
countsB = resultB[0].data.meas.get_counts()

import numpy as np

def monobit_summary(counts, result):
    shots = sum(counts.values())
    bitstrings = result[0].data.meas.get_bitstrings()
    M = np.array([[int(b) for b in s[::-1]] for s in bitstrings], dtype=int)
    p = M.mean(axis=0)                 # per-qubit fraction of 1s
    overall = float(p.mean())
    se = np.sqrt(0.25/shots)           # rough expected fluctuation for a fair coin
    suspect = np.abs(p - 0.5) > 3*se   # rule-of-thumb: outside ±3·SE
    return p, overall, se, suspect

pSim, overallSim, seSim, flagSim = monobit_summary(sim_counts, sim_results)
print("Sim (A) per-qubit P(1):", np.round(pSim, 3), "overall:", round(overallSim, 3), "SE~", round(seSim, 4))
print("Sim (A) suspect qubits:", np.where(flagSim)[0].tolist())

pA, overallA, seA, flagA = monobit_summary(countsA, resultA)
print("A per-qubit P(1):", np.round(pA, 3), "overall:", round(overallA, 3), "SE~", round(seA, 4))
print("A suspect qubits:", np.where(flagA)[0].tolist())

pB, overallB, seB, flagB = monobit_summary(countsB, resultB)
print("B per-qubit P(1):", np.round(pB, 3), "overall:", round(overallB, 3), "SE~", round(seB, 4))
print("B suspect qubits:", np.where(flagB)[0].tolist())

def runs_fraction_per_qubit(result):
    bitstrings = result[0].data.meas.get_bitstrings()
    M = np.array([[int(b) for b in s[::-1]] for s in bitstrings], dtype=int)
    flips = (M[1:] != M[:-1]).mean(axis=0)   # fraction of shot-to-shot flips per qubit
    return flips

flipsA = runs_fraction_per_qubit(resultA)
print("A runs (flip fraction) per qubit:", np.round(flipsA, 3))

flipsB = runs_fraction_per_qubit(resultB)
print("B runs (flip fraction) per qubit:", np.round(flipsB, 3))

flipsSim = runs_fraction_per_qubit(sim_results)
print("Sim runs (flip fraction) per qubit:", np.round(flipsSim, 3))

def autocorr_lag1(result):
    bitstrings = result[0].data.meas.get_bitstrings()
    M = np.array([[int(b) for b in s[::-1]] for s in bitstrings], dtype=int)
    X = M - M.mean(axis=0, keepdims=True)
    num = (X[1:]*X[:-1]).sum(axis=0)
    den = (X[:-1]**2).sum(axis=0)
    ac1 = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den>0)
    return ac1

ac1A = autocorr_lag1(resultA)
print("A lag-1 autocorr per qubit:", np.round(ac1A, 3))

ac1B = autocorr_lag1(resultB)
print("B lag-1 autocorr per qubit:", np.round(ac1B, 3))

ac1Sim = autocorr_lag1(sim_results)
print("Sim lag-1 autocorr per qubit:", np.round(ac1Sim, 3))


def interqubit_corr(result):
    bitstrings = result[0].data.meas.get_bitstrings()
    M = np.array([[int(b) for b in s[::-1]] for s in bitstrings], dtype=int)
    X = M - M.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / (len(M)-1)
    std = X.std(axis=0, ddof=1)
    R = cov / (std[:,None]*std[None,:])
    np.fill_diagonal(R, 1.0)
    return R

R_A = interqubit_corr(resultA)
flags = [(i,j,float(R_A[i,j])) for i,j in itertools.combinations(range(k),2) if abs(R_A[i,j])>0.1]
print("A Inter-qubit correlation", np.round(R_A, 3))
print("A suspicious pairs:", flags[:10])

R_B = interqubit_corr(resultB)
flags = [(i,j,float(R_B[i,j])) for i,j in itertools.combinations(range(k),2) if abs(R_B[i,j])>0.1]
print("B Inter-qubit correlation", np.round(R_B, 3))
print("B suspicious pairs:", flags[:10])

R_Sim = interqubit_corr(sim_results)
flags = [(i,j,float(R_Sim[i,j])) for i,j in itertools.combinations(range(k),2) if abs(R_Sim[i,j])>0.1]
print("Sim Inter-qubit correlation", np.round(R_Sim, 3))
print("Sim suspicious pairs:", flags[:10])