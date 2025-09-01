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



q_num_x = 6
q_num_y=6
Nx=2**q_num_x
Ny=2**q_num_y
q_num=q_num_x+q_num_y
normalize_x=np.sqrt(Nx/2)
normalize_y=np.sqrt(Ny/2)

X=np.arange(Nx,0,-1)
Y=np.arange(Ny,0,-1)
[XX,YY]=np.meshgrid(X,Y)
#target=np.sin(XX+YY)*(np.cos(YY))**2

#Xcor=np.linspace(-1,1,Nx)
#Ycor=np.linspace(-1,1,Ny)
#[XXcor,YYcor]=np.meshgrid(Xcor,Ycor)
#target=XXcor**3

data=loadmat('polynomial//input_DNS_filter2.mat')
target=data["target"].reshape(Nx,Ny)

norm=np.sqrt(np.sum(target*np.conj(target)))
initial_state1=(target/norm).transpose()/np.sqrt(2)
initial_state1=initial_state1.reshape(Nx*Ny)

initial_state=np.zeros(Nx*Ny*2,dtype=np.complex128)
initial_state[:Nx*Ny]=initial_state1
initial_state[Nx*Ny]=np.sqrt(2)/2

savemat('polynomial//temp.mat',{'init':initial_state})
#controlled_gate1=StatePreparation(initial_state1).control(1,None,'0')
coefs=np.zeros([70,70],dtype=np.complex128)
addup=0.0

approx=np.zeros([Ny,Nx],dtype=np.complex128)

sum=0
times=0
while(addup<0.5 and times<1000):
    for p in range(sum+1):
        for flag in range(2):
            q=sum-p
            phix=p*np.pi/Nx/2
            phiy=q*np.pi/Ny/2

            circ = QuantumCircuit(QuantumRegister(q_num+3),ClassicalRegister(3))

            circuitx=QuantumCircuit(q_num_x+1)
            circuity=QuantumCircuit(q_num_y+1)
            circuitx=Chebyshev_gate(circuitx,p,q_num_x+1)
            circuity=Chebyshev_gate(circuity,q,q_num_y+1)
            controlled_gate2=circuitx.to_gate().control(1,None,'1')
            controlled_gate3=circuity.to_gate().control(1,None,'1')

            #circ.h(q_num+2)

            circ.initialize(initial_state,list(range(0,q_num+1)))

            if(flag==1):
                circ.s(q_num)

            #circ.append(controlled_gate1,[12,0,1,2,3,4,5,6,7,8,9])
            #circ.append(controlled_gate2,[12,0,1,2,3,4,10])
            #circ.append(controlled_gate3,[12,5,6,7,8,9,11])
            #circ.append(controlled_gate1,[q_num+2]+list(range(0,q_num)))
            circ.append(controlled_gate2,[q_num]+list(range(0,q_num_x))+[q_num+1])
            circ.append(controlled_gate3,[q_num]+list(range(q_num_x,q_num))+[q_num+2])
            circ.measure(q_num+1,0)
            circ.measure(q_num+2,1)

            circ.p(-phix-phiy+(p+q)*np.pi,q_num)
            circ.h(q_num)
            circ.measure(q_num,2)
            #print(circ)


            backend_qasm = AerSimulator(method='statevector')



            results=backend_qasm.run(transpile(circ,backend_qasm), shots=5000).result()
            #answer=results.get_statevector(circ,decimals=10).data

            answer=results.results
            t0=answer[0].data.counts['0x0']
            t1=answer[0].data.counts['0x4']
            prob_ori=(t0/(t0+t1)*2-1)
            prob=prob_ori/normalize_x/normalize_y
            if(p==0):
                prob=prob/np.sqrt(2)
            if(q==0):
                prob=prob/np.sqrt(2)

            if(flag==1):
                prob=prob*1j
            coefs[p,q]+=prob
            addup+=prob_ori**2
            

            print("({},{}):{},{},{}".format(p,q,t0,t1,addup))

            approx+= prob*np.cos((XX*2-1)/2/Nx*np.pi*p)*np.cos((YY*2-1)/2/Ny*np.pi*q)
        times=times+1
    sum=sum+1

approx=approx/np.sqrt(np.sum(approx*np.conj(approx)))
savemat('polynomial//coefs_2d_cheb.mat',{'result':approx,'coefs':coefs,'target':target})