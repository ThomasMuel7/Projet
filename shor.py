from math import gcd, log, floor
import numpy as np
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, transpile
from fractions import Fraction
from qiskit_aer import AerSimulator

def mod_gate(a, N):
    """
    Modular multiplication gate from permutation matrix.
    """
    n = floor(log(N - 1, 2)) + 1
    U = np.full((2**n, 2**n), 0)
    for x in range(N):
        U[a * x % N][x] = 1
    for x in range(N, 2**n):
        U[x][x] = 1
    G = UnitaryGate(U)
    G.name = f"U_{a}mod{N}"
    return G

def QPE(control, target, circuit, a, N):
    U = mod_gate(a, N)
    for k, qubit in enumerate(control):
        U_controlled = U.power(2**k).control()
        circuit.compose(U_controlled, qubits=[qubit] + list(target), inplace=True)
        circuit.barrier()
    return circuit

def QFT(num_control, control, circuit):
    for j in range(num_control):
        circuit.h(control[num_control-j-1])
        for k in range(j+1, num_control):
            # Apply controlled phase rotation
            circuit.cp(np.pi/2**(k-j), control[num_control-k-1], control[num_control-1-j])
        circuit.barrier()
    # Swap qubits to reverse order
    for i in range(num_control//2):
        circuit.swap(control[i], control[num_control-i-1])
    circuit.barrier()
    return circuit

def initialize_circuit(a, N, num_target, num_control):
    # Initialize the circuit
    control = QuantumRegister(num_control, name="C")
    target = QuantumRegister(num_target, name="T")
    output = ClassicalRegister(num_control, name="out")
    circuit = QuantumCircuit(control, target, output)
    circuit.x(target[0])
    circuit.barrier()
    for qubit in control:
        circuit.h(qubit)
    circuit.barrier()
    circuit = QPE(control, target, circuit, a, N)
    circuit = QFT(num_control, control, circuit)
    circuit.measure(control, output)
    return circuit

def simulate_shor(qc):
    simulator = AerSimulator()
    # Transpile the circuit for the simulator
    compiled_circuit = transpile(qc, simulator)
    # Run the simulation
    job = simulator.run(compiled_circuit, shots=20)
    result = job.result()
    # Get counts (number of times each outcome occurs)
    counts = result.get_counts()
    return counts

def estimate_period(N, value, nb_precision_qubits):
    phase = value / 2**nb_precision_qubits 
    frac = Fraction(phase).limit_denominator(N)
    r = frac.denominator
    return r

def calculate_factors(a, r, N):
    if r % 2 == 0:
    # Calculate potential factors
        factor1 = gcd(pow(a, r//2) - 1, N)
        factor2 = gcd(pow(a, r//2) + 1, N)
        if factor1 not in [1, N] or factor2 not in [1, N]:
            return [factor1, factor2]
    return None

def print_factors(factors, N): 
    factor1 = factors[0]
    factor2 = factors[1]
    if factor1 in [1,N] :
        print(f"Found the 2 factors : {factor2} and {N//factor2}")
    elif factor2 in [1,N] :
        print(f"Found the 2 factors : {factor1} and {N//factor1}")
    else :
        print(f"Found the 2 factors : {factor1} and {factor2}")
        
def shor(a, N):
    if gcd(a, N) > 1:
        print(f"Error: gcd({a},{N}) > 1")
        return None
    # Number of qubits
    num_target = floor(log(N - 1, 2)) + 1  # for modular exponentiation operators
    num_control = 2 * num_target  # for enough precision of estimation
    print("Initializing the circuit...")
    circuit = initialize_circuit(a, N, num_target, num_control)
    print("Circuit initialized")
    print("Simulating the circuit...")
    counts = simulate_shor(circuit)
    print("Simulations done")
    decimal_values = sorted([int(bitstr, 2) for bitstr in counts.keys()])
    for value in decimal_values:
        if value == 0:
            continue
        r = estimate_period(N, value, num_control)
        print(f"Estimated period r = {r} from value {value}")
        print("Trying to calculate factors...")
        factors = calculate_factors(a, r, N)
        if factors:
            print("Factors found!")
            print_factors(factors, N)
            return
        print("Failed to calculate factors with this r, trying next value...")
    print("Failed to find factors.")
    return

shor(3, 22)