from qiskit import QuantumCircuit
import matplotlib

# k‑bit QRNG
def qrng(k: int):
    qc = QuantumCircuit(k)
    for q in range(k):
        qc.h(q)          # one coin‑flip per qubit
    qc.measure_all()
    return qc

k = 8
qc = qrng(k)
print(qc.draw())
fig = qc.draw(output="mpl")
fig.savefig("qrng_8qubit_circuit.png")