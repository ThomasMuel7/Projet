"""
shors_qiskit.py

Qiskit-adapted version of the user's shors.py (classical + quantum parts).

This script implements a small-scale, educational version of Shor's algorithm using
Qiskit. It follows the textbook approach for order-finding:
 - Construct the modular multiplication unitary U_a on the output register
 - Use quantum phase estimation on U_a to estimate a phase s/r
 - Use continued fractions to extract the order r and classical post-processing

Limitations & notes:
 - This implementation builds permutation matrices for the modular multiplication
   unitary and therefore only scales to small N (N <= ~15..31 depending on memory).
 - Qiskit removed the built-in Shor factorizer from qiskit.algorithms; this file
   provides a runnable example that works with Qiskit Aer (simulator).
 - Requires qiskit and numpy. Run with: pip install qiskit numpy

References:
 - Qiskit Shor tutorial (textbook / IBM docs)
 - The original shors.py provided by the user was used as the classical backbone.
"""

import argparse
import math
import random
import numpy as np
from fractions import Fraction

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Operator, Statevector
from qiskit_aer import AerSimulator

# --------------------------- Classical helper functions ---------------------------

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def mod_exp(a, e, m):
    return pow(a, e, m)


def continued_fraction(x, q, max_den):
    # Use Fraction for simplicity; returns denominator candidate
    frac = Fraction(x, q).limit_denominator(max_den)
    return frac.denominator

# --------------------------- Quantum helpers (Qiskit) -----------------------------

def next_power_of_two(x):
    return 1 << (x - 1).bit_length()


def build_Ua_matrix(a, N):
    """
    Build the unitary matrix for U_a acting on n = ceil(log2(N)) qubits
    defined by U_a|y> = |(a*y) mod N> for y in [0..N-1], and acts as identity
    on the unused basis states [N..2^n-1].
    """
    n = math.ceil(math.log2(N))
    dim = 1 << n
    U = np.zeros((dim, dim), dtype=complex)
    for y in range(dim):
        if y < N:
            U[(a * y) % N, y] = 1.0
        else:
            # leave unused states unchanged
            U[y, y] = 1.0
    return U


def controlled_powered_unitary_gate(a, N, power):
    """
    Return a controlled gate implementing U_a^{power} where U_a is modular multiplication
    by a (mod N). We create the base matrix and raise it to the requested integer power
    (matrix power), then wrap as a ControlledGate acting on n target qubits.
    """
    U = build_Ua_matrix(a, N)
    Upow = np.linalg.matrix_power(U, power)
    n = int(math.log2(U.shape[0]))
    # Create an Operator and convert to a Gate
    op = Operator(Upow)
    # Build a QuantumCircuit that implements op on n qubits
    qc = QuantumCircuit(n)
    qc.unitary(op, qc.qubits)
    gate = qc.to_gate()
    gate.name = f"U_a^{power}"
    # Controlled gate will be constructed in caller using gate.control()
    return gate


def phase_estimation_circuit(a, N, t):
    """
    Build a phase estimation circuit for unitary U_a with t counting qubits.
    The output (phase) register has t qubits and the target register has n qubits
    (enough to hold numbers modulo N).
    """
    n = math.ceil(math.log2(N))
    counting = QuantumRegister(t, name="count")
    target = QuantumRegister(n, name="target")
    cr = ClassicalRegister(t, name="c")
    qc = QuantumCircuit(counting, target, cr)

    # Initialize target register to |1> (we want eigenstate of U_a starting from 1)
    qc.x(target[0])

    # Apply Hadamards on counting register
    qc.h(counting)

    # For each counting qubit, apply controlled-U^{2^j}
    for j in range(t):
        power = 1 << j
        gate = controlled_powered_unitary_gate(a, N, power)
        cgate = gate.control()
        # cgate expects 1 control + n targets
        qc.append(cgate, [counting[j]] + target[:] )

    # Inverse QFT on counting register
    qc.append(inverse_qft(t), counting[:])

    # Measurement
    qc.measure(counting, cr)

    return qc


def inverse_qft(n):
    qc = QuantumCircuit(n, name="iqft")
    # standard textbook inverse QFT (no swaps because we'll read in binary order)
    for j in range(n//2):
        qc.swap(j, n - j - 1)
    for j in range(n):
        for k in range(j):
            qc.cp(-2 * math.pi / (2 ** (j - k + 1)), k, j)
        qc.h(j)
    return qc.to_gate()

# --------------------------- Shor high-level routine --------------------------------
def find_period_qiskit(a, N, shots=1024):
    """
    Use Qiskit-based phase estimation to estimate the period r of f(x)=a^x mod N.
    Returns a candidate r or None on failure.
    """
    if gcd(a, N) != 1:
        return 1  # trivial

    # number of counting qubits: t ~ 2 * n (textbook) where n = ceil(log2 N)
    n = math.ceil(math.log2(N))
    t = 2 * n
    # build circuit
    qc = phase_estimation_circuit(a, N, t)

    sim = AerSimulator()
    t_qc = transpile(qc, sim)
    t_qc.save_statevector()

    job = sim.run(t_qc, shots=shots)
    result = job.result()

    try:
        sv = result.get_statevector(0)
    except Exception:
        sv = Statevector.from_instruction(qc)

    # get measurement results from the statevector
    counts = sv.sample_counts(shots)

    # pick the most frequent outcome
    measured = max(counts.items(), key=lambda kv: kv[1])[0]
    measured_int = int(measured[::-1], 2)

    # estimate phase = measured_int / 2^t
    r_candidate = continued_fraction(measured_int, 1 << t, N)
    return r_candidate


def shor_qiskit(N, attempts=5, shots=2048):
    """
    High-level Shor wrapper using the Qiskit order finder above.
    Returns non-trivial factors or None.
    """
    if N % 2 == 0:
        return [2, N // 2]
    for attempt in range(attempts):
        a = random.randrange(2, N - 1)
        d = gcd(a, N)
        if d > 1:
            return [d, N // d]

        r = find_period_qiskit(a, N, shots=shots)
        if r is None or r % 2 == 1:
            continue

        # compute potential factors
        x = pow(a, r // 2, N)
        if x == N - 1 or x == 1:
            continue

        p = gcd(x + 1, N)
        q = gcd(x - 1, N)
        if p in (1, N) or q in (1, N):
            continue
        return [p, q]
    return None

# --------------------------- CLI interface ----------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run a small-scale Shor's algorithm using Qiskit")
    parser.add_argument('N', type=int, help='Integer to factor (small, e.g., 15)')
    parser.add_argument('-a', '--attempts', type=int, default=5, help='Number of attempts')
    parser.add_argument('--shots', type=int, default=1024, help='Shots for simulator')
    return parser.parse_args()


def main():
    args = parse_args()
    N = args.N
    print(f"Factoring N={N} (this implementation is for small N)\n")

    factors = shor_qiskit(N, attempts=args.attempts, shots=args.shots)
    if factors:
        print(f"Found factors: {factors[0]} and {factors[1]}")
    else:
        print("Failed to find factors. Try increasing attempts or use another 'a' value.")

if __name__ == '__main__':
    main()
