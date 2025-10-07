from qiskit import QuantumCircuit
import numpy as np

qc = QuantumCircuit(4)

qc.barrier()

qc.h(3)
qc.cp(np.pi/2, 2, 3)
qc.cp(np.pi/4, 1, 3)
qc.cp(np.pi/8, 0, 3)

qc.barrier()

qc.h(2)
qc.cp(np.pi/2, 2, 1)
qc.cp(np.pi/4, 2, 0)

qc.barrier()

qc.h(1)
qc.cp(np.pi/2, 1, 0)

qc.barrier()

qc.h(0)

qc.barrier()

qc.swap(0, 3)
qc.swap(1, 2)

print(qc)
qc.draw('mpl', filename='images/prep_qft_circuit.png')