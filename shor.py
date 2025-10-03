from qiskit import QuantumCircuit, transpile
from math import pi, gcd
from fractions import Fraction
from qiskit_aer import AerSimulator
# Create Shor for N=15 and a=2
def shor_circuit():
    qc = QuantumCircuit(8, 4)
    #Initilization
    qc.x(0)
    qc.barrier()
    #Modular exponentiation
    qc.h(4)
    qc.h(5)
    qc.h(6)
    qc.h(7)
    qc.barrier()
    #x2 mod 15
    qc.cswap(4,2,3)
    qc.cswap(4,1,2)
    qc.cswap(4,0,1)
    qc.barrier()
    #x4 mod 15
    qc.cswap(5,3,1)
    qc.cswap(5,2,0)
    qc.barrier()
    #apply QFT
    #iter 1
    qc.h(7)
    qc.cp(pi/2,6,7)
    qc.cp(pi/4,5,7)
    qc.cp(pi/8,4,7)
    qc.barrier()
    #iter 2
    qc.h(6)
    qc.cp(pi/2,5,6)
    qc.cp(pi/4,4,6)
    qc.barrier()
    #iter 3
    qc.h(5)
    qc.cp(pi/2,4,5)
    qc.barrier()
    #iter 4
    qc.h(4)
    qc.barrier()
    #swap
    qc.swap(4,7)
    qc.swap(5,6)
    qc.barrier()
    #measure
    qc.measure([4,5,6,7],[0,1,2,3])
    qc.draw('mpl', filename='shor_circuit.png')
    return qc

def simulate_shor(qc):
    simulator = AerSimulator()
    # Transpile the circuit for the simulator
    compiled_circuit = transpile(qc, simulator)
    # Run the simulation
    job = simulator.run(compiled_circuit, shots=1024)
    result = job.result()
    # Get counts (number of times each outcome occurs)
    counts = result.get_counts()
    return counts

def post_processing(counts):
    decimal_values = sorted([int(bitstr, 2) for bitstr in counts.keys()])
    for value in decimal_values:
        if value != 0:
            phase = value / 16  # Since we have 4 qubits in the counting register
            frac = Fraction(phase).limit_denominator(15)
            r = frac.denominator
            if r % 2 == 0:
                # Calculate potential factors
                factor1 = gcd(pow(2, r//2) - 1, 15)
                factor2 = gcd(pow(2, r//2) + 1, 15)
                if factor1 not in [1, 15] or factor2 not in [1, 15]:
                    return [factor1, factor2]
    return None
        
if __name__ == "__main__":
    qc = shor_circuit()
    counts = simulate_shor(qc)
    factors = post_processing(counts)
    print(factors)
    if factors == None:
        print("Failed to find factors.")
    else : 
        factor1 = factors[0]
        factor2 = factors[1]
        if factor1 in [1,15] :
            print(f"Found the 2 factors : {factor2} and {15//factor2}")
        elif factor2 in [1,15] :
            print(f"Found the 2 factors : {factor1} and {15//factor1}")
        else :
            print(f"Found the 2 factors : {factor1} and {factor2}")