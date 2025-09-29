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

# Enumerate candidates
cands = service.backends(simulator=False, operational=True, min_num_qubits=6)
for b in cands:
    print(b.name, b.num_qubits)

# Select A (least busy) and B (another candidate)
A = service.least_busy(simulator=False, operational=True, min_num_qubits=6)
B = next(b for b in cands if b.name != A.name)

# Configurations
cfgA, cfgB = A.configuration(), B.configuration()
print(A.name, "basis:", cfgA.basis_gates)
print(B.name, "basis:", cfgB.basis_gates)

# Coupling maps
from qiskit.visualization import plot_coupling_map
cmapA = A.coupling_map; cmapB = B.coupling_map
plot_coupling_map(A.num_qubits, None, cmapA.get_edges())
plot_coupling_map(B.num_qubits, None, cmapB.get_edges())

# Properties (median single-qubit and readout errors)
import numpy as np
propsA, propsB = A.properties(), B.properties()

def summarize_props(props):
    readout = []
    t1s = []
    t2s = []
    freqs = []
    for q in props.qubits:
        for entry in q:
            if entry.name == "readout_error":
                readout.append(entry.value)
            elif entry.name == "T1":
                t1s.append(entry.value)  # in microseconds
            elif entry.name == "T2":
                t2s.append(entry.value)  # in microseconds
            elif entry.name == "frequency":
                freqs.append(entry.value)
    return (
        np.median(readout),
        np.median(t1s),
        np.median(t2s),
        np.median(freqs),
    )


# Queue lengths
queueA = A.status().pending_jobs
queueB = B.status().pending_jobs

readA, t1A, t2A, freqsA = summarize_props(propsA)
readB, t1B, t2B, freqsB = summarize_props(propsB)

import pandas as pd

summary = pd.DataFrame({
    "Backend": [A.name, B.name],
    "Qubits": [A.configuration().num_qubits, B.configuration().num_qubits],
    "Basis gates": [cfgA.basis_gates, cfgB.basis_gates],
    "Median readout err": [readA, readB],
    "Median T1 (µs)": [t1A , t1B],
    "Median T2 (µs)": [t2A , t2B],
    "Frequencies " : [freqsA, freqsB],
    "Queue length": [queueA, queueB]
})
print(summary)
