import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
#from qiskit.visualization import plot_histogram, plot_state_city
from scipy.io import savemat, loadmat
from qiskit.circuit.library import StatePreparation

def Chebyshev_gate(circuit, p, N_qubit):
    N=2**(N_qubit-1)
    theta=p*np.pi/N
    phi=theta/2

    for i in range(N_qubit-1):
        circuit.h(i)

    if(p==0):
        return circuit
    
    circuit.h(N_qubit-1)
    
    for i in range(0,N_qubit-1):
        circuit.cp((2**i)*theta,N_qubit-1,i)
    circuit.x(N_qubit-1)
    for i in range(0,N_qubit-1):
        circuit.cp(-(2**i)*theta,N_qubit-1,i)
    circuit.x(N_qubit-1)

    circuit.p(2*phi,N_qubit-1)
    circuit.h(N_qubit-1)
    return circuit 



q_num = 10
N=2**q_num
normalize=np.sqrt(N/2)
result_list=[]
coefs=np.zeros(100)
addup=0.0

X=np.linspace(-1,1,N)
target=X**2+np.sin(10*X)
#+X**3

#data=loadmat('input_v2_10.mat')
#target=data["target"].reshape(N)

norm=np.sqrt(np.sum(target**2))
initial_state1=target/norm

controlled_gate1=StatePreparation(initial_state1).control(1,None,'0')

p=0

while(addup<0.85 and p<20):
    phi=p*np.pi/N/2

    circ = QuantumCircuit(QuantumRegister(q_num+2),ClassicalRegister(2))
    #circ = QuantumCircuit(q_num+2)

    circuit1=QuantumCircuit(q_num+1)
    circuit1=Chebyshev_gate(circuit1,p,q_num+1)
    controlled_gate2=circuit1.to_gate().control(1,None,'1')

    circ.h(q_num+1)

    #circ.s(q_num+1)

    circ.append(controlled_gate1,[q_num+1]+list(range(0,q_num)))
    circ.append(controlled_gate2,[q_num+1]+list(range(0,q_num+1)))
    circ.measure(q_num,0)

    circ.p(-phi+p*np.pi,q_num+1)
    circ.h(q_num+1)
    circ.measure(q_num+1,1)


    backend_qasm = AerSimulator(method='statevector')


    results=backend_qasm.run(transpile(circ,backend_qasm), shots=500).result()

    answer=results.results

    t0=answer[0].data.counts['0x0']
    t1=answer[0].data.counts['0x2']
    prob_ori=(t0/(t0+t1)*2-1)
    prob=prob_ori/normalize
    if(p==0):
        prob=prob/np.sqrt(2)

    coefs[p]=prob
    addup=addup+prob_ori**2
    result_list +=(t0,t1,prob_ori,prob)
    print("{}:{}".format(p,addup))
    p=p+1


approx=np.zeros(N)
for i in range(p):
    approx=approx+coefs[i]*np.cos((np.arange(N,0,-1)*2-1)/2/N*np.pi*i)

approx=approx/np.sqrt(np.sum(approx**2))

savemat('polynomial//coefs_1d_cheb.mat',{'coefs':coefs,'target':initial_state1,'result':approx,'result_list':result_list})