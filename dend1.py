import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
D = 4 # diameter
Rm = 100 # membrance resistance
Cm = 100 # membrane capacitance
g_conductance = 5 # synaptic conductance
E = -5 # reversal potential
y0 = 0 # intializ of neuron
V_thresh = 65  # spike threshold soma
V_reset = -50  # reset potential soma
dt = 0.5  # time step (ms) soma
t_total = 50  # total simulation time (ms) soma
t_s = np.arange(0, t_total, dt)  # time vector soma
V = V_reset * np.ones(t_s.shape)  # membrane potential vector soma
tau = 30  # membrane time constant (ms soma)

# Initialize presynaptic and postsynaptic activity arrays
pre = np.random.rand(50)
post = np.random.rand(50)
weights = np.zeros(len(pre))
alpa = 2.5 # learning rate
# Define a single dendrite with multiple branches
dendrite = [
    [ # Branch 1
        [0.5, 1],
        [1, 1.4],
        [1.3, 0]
    ],
    [ # Branch 2
        [1.5, 6],
        [1.6, 0],
        [2.3, 4.8]
    ],
    [ # Branch 3
        [2.5, 3],
        [2.6, 3],
        [3.3, 4]
    ]
]
def calculate_membrane_potential(dendrite, synaptic_response , I):
    """
    Calculate the membrane potential at the soma due to synaptic inputs.
    Parameters
    ----------
    dendrite : list
        A list of branches, where each branch is a list of synapses.
    synaptic_response : float
        The synaptic response at each synapse.
    Returns
    -------
    potential : float
        The membrane potential at the soma.
    """
    potential = 0
    for branch in dendrite:
        for synapse in branch:
            distance = synapse
            weight = distance [1] / (distance [0] + 1) ** 2
            potential += weight * synaptic_response * np.exp(distance) * I       
    return potential
# define the synaptic current
def s_current(time):
    if time.any() < 0.1:
        return g_conductance * (time - 0.001)
    else:
        return g_conductance  * (time + 0.001)
def calcluter_ion(Rm , g_conductance , E):
    return g_conductance * ((Rm + E) / Cm) * 2
# Define the synaptic responses
time = np.linspace(0 , 10)
t = s_current(time)
A = D / 2
R = Rm * (0.5 / A)
responses = np.heaviside(np.cos(6 * np.pi * ((D / 2) * R) * t), 4)
#calcluter ion chanel dendrites
I = calcluter_ion(Rm , g_conductance , E)
#print(I)
# Simulate the model soma
for i in range(len(t)):
    V[i+1] = V[i] + (-(V[i] - V_reset) / tau + Rm) * dt
soma = V[i+1] # Store the membrane potentials of soma
# Calculate the total signal along the dendrite
potentials = [calculate_membrane_potential(dendrite, response , I) for response in responses]
total_signal = np.cumsum(potentials)
#merg soma and dendrite
s_d = potentials + soma
print(s_d)
# drawing merg soma and dendrites
plt.plot(time , s_d)
plt.title('merg soma and dendrites')
plt.show()
print(total_signal)
#print(potentials)
sns.set()
plot = sns.scatterplot(x = I , y = t)
plt.title('ion chanel dendrite')
plt.show()
# Define Hebbey's learning law
for i in range(len(pre)):
    weights[i] += alpa * pre[i] * post[i]
sns.histplot(weights, bins=10)
plt.title('rate learning dendrite for processing')
plt.show()
#initializ of neuron
y = np.zeros_like(t)
y[0] = y0
for i in range(1, len(t)):
    y[i] = y[i-1] + I * (Rm + E) * (t[i] - t[i-1])
plt.plot(t, y)
plt.xlabel('Time (ms)')
plt.ylabel('Vm (mV)')
plt.show()
# Plot the results
fig, ax = plt.subplots()
ax.plot(t, potentials)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Membrane Potential (mV)')
ax.set_title('Dendritic Processing')
plt.show()