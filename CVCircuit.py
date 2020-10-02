import os
import sys
import qiskit
import matplotlib.pyplot as plt
import numpy as np
from qiskit.visualization.circuit_visualization import circuit_drawer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute

from CVGates import CVGates

class CVCircuit:
    def __init__(self, circuit, qr, n_qubits_per_mode=2):
        self.n_qubits_per_mode = n_qubits_per_mode
        self.n_qumodes = len(circuit)
        self.circuit = circuit
        self.qr = qr
        self.gates = CVGates(n_qubits_per_mode)

    def initialize(self, fock_states):
        for qumode, n in enumerate(fock_states):
            if n >= 2**self.n_qubits_per_mode:
                raise ValueError("The parameter n should be lower than the cutoff")
            vector = np.zeros((2**self.n_qubits_per_mode,))
            vector[n] = 1
            start_qumode = self.n_qubits_per_mode * qumode
            self.circuit.initialize(vector, [self.qr[i] for i in range(start_qumode, start_qumode+self.n_qubits_per_mode)])


    def DGate(self, alpha, qumode):
        start_qumode = self.n_qubits_per_mode * qumode
        qubits = self.qr[start_qumode:start_qumode+self.n_qubits_per_mode]
        self.circuit.unitary(self.gates.D(alpha), qubits)

    def SGate(self, z, qumode):
        start_qumode = self.n_qubits_per_mode * qumode
        qubits = self.qr[start_qumode:start_qumode+self.n_qubits_per_mode]
        self.circuit.unitary(self.gates.S(z), qubits)

    def RGate(self, phi, qumode):
        start_qumode = self.n_qubits_per_mode * qumode
        qubits = self.qr[start_qumode:start_qumode+self.n_qubits_per_mode]
        self.circuit.unitary(self.gates.R(phi), qubits)

    def KGate(self, kappa, qumode):
        start_qumode = self.n_qubits_per_mode * qumode
        qubits = self.qr[start_qumode:start_qumode+self.n_qubits_per_mode]
        self.circuit.unitary(self.gates.K(kappa), qubits)

    def BSGate(self, phi, qumodes):
        start_qumodes = [self.n_qubits_per_mode * qumodes[0], self.n_qubits_per_mode * qumodes[1]]
        qubits = self.qr[start_qumodes[0]:start_qumodes[0]+self.n_qubits_per_mode] + self.qr[start_qumodes[1]:start_qumodes[1]+self.n_qubits_per_mode]
        self.circuit.unitary(self.gates.BS(phi), qubits)

    def S2Gate(self, z, qumodes):
        start_qumodes = [self.n_qubits_per_mode * qumodes[0], self.n_qubits_per_mode * qumodes[1]]
        qubits = self.qr[start_qumodes[0]:start_qumodes[0]+self.n_qubits_per_mode] + self.qr[start_qumodes[1]:start_qumodes[1]+self.n_qubits_per_mode]
        self.circuit.unitary(self.gates.S2(z), qubits)
 
if __name__ == '__main__':
    # ===== Constants =====

    n_qubits_per_mode = 3
    n_qumodes = 2
    alpha = 1
    phi = np.pi/2

    # ==== Initialize circuit =====

    qr = QuantumRegister(n_qubits_per_mode*n_qumodes)
    cr = ClassicalRegister(n_qubits_per_mode*n_qumodes)
    circuit = QuantumCircuit(qr, cr)
    cv_circuit = CVCircuit(circuit, qr, n_qubits_per_mode)

    # ==== Build circuit ====

    cv_circuit.initialize([0,0])
    cv_circuit.DGate(alpha, 0)
    cv_circuit.BSGate(phi, (0,1))
    # cv_circuit.BSGate(theta, phi, 0, 1)
    # circuit.measure(qr, cr)
    print(circuit)

    # ==== Compilation =====

    backend = Aer.get_backend('statevector_simulator')
    job = execute(circuit, backend)
    result = job.result()
    state = job.result().get_statevector(circuit)

    # ==== Tests ====

    print(state)

    prob = np.abs(state)**2
    plt.plot(prob,'-o')
    plt.savefig('CVCircuit.png')