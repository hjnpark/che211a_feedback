import numpy as np
import matplotlib.pyplot as plt

kb = 1.380649 * 10**(-23)
T = 300 #K
m = .084 #kg/mol
m = m/(6.02 * 10**23)
v = 10
kbT = kb*T

mean_speed = np.sqrt(8*kb*T/(np.pi*m))
standard_deviation = np.sqrt(kbT/m)
print(mean_speed)
print(standard_deviation)

def speed_distribution(m, kbT, v):
    return (4*np.pi*(v**2))*((m/(kbT*2*np.pi))**(3/2))*np.exp(-m*(v**2)/(2*kbT))
6

speed = []
for i in np.arange(0, 1000, 0.1):
    speed.append(speed_distribution(m, kbT, i))

plt.plot((np.arange(0, 1000, 0.1)), speed)
plt.scatter(mean_speed, speed_distribution(m, kbT, mean_speed), label = 'Mean')
plt.scatter(mean_speed -standard_deviation, speed_distribution(m, kbT, mean_speed - standard_deviation), label = 'Std Dev.')
plt.scatter(mean_speed + standard_deviation, speed_distribution(m, kbT, mean_speed + standard_deviation), label = 'Std Dev.')
plt.legend()
plt.show()

