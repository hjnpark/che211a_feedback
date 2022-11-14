import numpy as np
import matplotlib.pyplot as plt

#Part a)
def maxwell_boltzmann(m, kbT, vx):
    return ((m/(2*np.pi*kbT))**(0.5))*(np.exp(-(m*vx**2)/(2*kbT)))/10

kb = 1.380649*10**-29 #m^2 kg s^-2 K^-1
T = 150 #K
kbT = kb * T
m = .084 #kg/mol
m = m/(6.023*10**23)

#print(maxwell_boltzmann(m,kbT, vx))
x_velocities = np.arange(-1, 1, .001)
mb_dist = []
for vx in x_velocities:
    mb_dist.append(maxwell_boltzmann(m,kbT,vx))

plt.plot(x_velocities, mb_dist)
y_max = np.max(mb_dist)
plt.plot(x_velocities, np.ones(2000,)*y_max, color='r')
plt.plot([-1, -1], [0, y_max], color= 'r')
plt.plot([1, 1], [0, y_max], color= 'r')
plt.plot([x_velocities[0], x_velocities[-1]], [0,0], color='r')
plt.title('Velocity distribution of Kr at 150K')
plt.xlabel('x-component of the velocity (m/s)')
plt.ylabel('Probability')
plt.show()

#Part b)

#the following is called rejection sampling
def sampling_distribution():
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(0, 1)*y_max
    if maxwell_boltzmann(m,kbT,x) > y:
        return x
x_sample = []
y = []
for sample in range(1000):
    x_value = sampling_distribution()
    print(x_value)
    if x_value is not None:
        x_sample.append(x_value)
        y.append(maxwell_boltzmann(m, kbT, x_value))


plt.hist(x_sample, bins= [i for i in np.arange(-1, 1, .1)], density= True)
plt.xlim(-1, 1)
plt.show()

#Part C)
#def maxwell_boltzman_generation():
#do velocity distribution in 1D, find SD and times by 3
#this is

def gen_Max_Boltz_velocities(mass, temp):
    """
    Generates sample points from the Maxwell_Boltzmann distribution

    Parameters
    ----------
    mass : float
        Mass of particle (g/mol)

    temp : float
        Temperature (K)

    Returns
    ----------
    velocity : float
        Velocity taken from Maxwell-Boltzmann distribution (nm/ps)
    """
    standard_deviation = np.sqrt(kb*temp/mass)
    velocity = standard_deviation*np.random.randn(1, 2)
    return velocity
kb = 1.380649*10**-29 #m^2 g s^-2 K^-1
T = 150 #K
kbT = kb * T
m = .0840 #kg/mol
m = m/(6.023*10**23)
x = []
y = []
for sample in range(1000):
    sample_point = gen_Max_Boltz_velocities(m, T)
    x.append(sample_point[0,0])
    y.append(sample_point[0,1])
# desnity = True normalizes a histogram, ie. sum of the bar area equals 1
plt.hist(x, density=True, bins= [i for i in np.arange(-1, 1, .1)])
#plt.hist(x,20)
plt.xlim(-1, 1)
plt.show()

#Part d)

def Andersen_thermostat(num_particles, mass, kbT):
    """
    Generates velocities from the Maxwell_Boltzmann distribution

    Parameters
    ----------
    mass : float
        Mass of particle (g/mol)

    temp : float
        Temperature (K)

    num_particles : int
        Number of particles

    Returns
    ----------
    velocities : numpy array (num_particles x 3)
        Velocity of each particle taken from Maxwell-Boltzmann distribution (nm/ps)
    """

    standard_deviation = np.sqrt(kbT / mass)
    velocities = standard_deviation * np.random.randn(num_particles, 3)

    return velocities

print(Andersen_thermostat(2, m, kbT))
