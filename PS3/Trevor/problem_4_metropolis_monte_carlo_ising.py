import numpy as np
import matplotlib.pyplot as plt

#Part C & D

def tot_energy_lattice(spins):
    rolly = np.roll(spins, 1, axis = 0)
    rollx = np.roll(spins, 1, axis = 1)
    return -1*((spins*rollx).sum() + (spins*rolly).sum())


def monte_carlo(E_before, spins, kbT = 1):
    #E_before = tot_energy_lattice(spins)
    spin_row = np.random.randint(len(spins))
    spin_column = np.random.randint(len(spins))
    spins[spin_row, spin_column] *= -1
    if spin_column != 7 and spin_row != 7: # You probably don't want to hardcoded the end of column and row as 7.
        rolly = np.roll(spins, 1, axis=0)
        rollx = np.roll(spins, 1, axis=1)
        new_x_interations = (spins[spin_row, spin_column:(spin_column+2)]*rollx[spin_row, spin_column:(spin_column+2)]).sum()
        new_y_interactions = (spins[spin_row:(spin_row+2), spin_column]*rolly[spin_row:(spin_row+2), spin_column]).sum()
        new_interactions = (new_y_interactions + new_x_interations).sum()
    elif spin_column !=7 and spin_row == 7:
        rolly = np.roll(spins, -1, axis=0)
        rollx = np.roll(spins, 1, axis=1)
        new_x_interations = (spins[spin_row, spin_column:(spin_column + 2)] * rollx[spin_row,
                                                                              spin_column:(spin_column + 2)]).sum()
        new_y_interactions = (
                    spins[(spin_row - 1):(spin_row +1), spin_column] * rolly[(spin_row -1):(spin_row+1), spin_column]).sum()
        new_interactions = (new_y_interactions + new_x_interations).sum()
    elif spin_column ==7 and spin_row != 7:
        rolly = np.roll(spins, 1, axis=0)
        rollx = np.roll(spins, -1, axis=1)
        new_x_interations = (spins[spin_row,(spin_column-1):(spin_column+1)]*rollx[spin_row, (spin_column-1):(spin_column+1)]).sum()
        new_y_interactions = (spins[spin_row:(spin_row+2), spin_column]*rolly[spin_row:(spin_row+2), spin_column]).sum()
        new_interactions = (new_y_interactions + new_x_interations).sum()

    else:
        rolly = np.roll(spins, -1, axis=0)
        rollx = np.roll(spins, -1, axis=1)
        new_x_interations = (spins[spin_row,(spin_column-1):(spin_column+1)]*rollx[spin_row, (spin_column-1):(spin_column+1)]).sum()
        new_y_interactions = (
                    spins[(spin_row - 1):(spin_row +1), spin_column] * rolly[(spin_row -1):(spin_row+1), spin_column]).sum()
        new_interactions = (new_y_interactions + new_x_interations).sum()
    E_after = -1*(-1*E_before + 2*new_interactions)

    if E_after < E_before:
        return E_after
    else:
        rand = np.random.random()
        boltz = np.exp( -(E_after - E_before)/kbT)
        if rand < boltz:
            return E_after
        else:
            spins[spin_row, spin_column] *= -1
            return E_before

def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2

spins = np.ones((8,8))
trialE = tot_energy_lattice(spins)
print(trialE)

E_before = tot_energy_lattice(spins)

avg_spins = []
print(spins)
kbT =1 
for i in range(10**6):
    avg_spins.append(spins.mean())
    E_before = monte_carlo(E_before,spins, kbT)
print(spins)
plt.plot(avg_spins)
plt.title(f"Avg. Total Magnetization versus step; kbT = {kbT}")
plt.xlabel("Step number")
plt.ylabel("Avg. Total Magnetization")
plt.show()
print(np.array(avg_spins).mean())

plt.imshow(spins)
plt.colorbar()
plt.title(f"100,000 Monte Carlo steps; kbT = {kbT}")
plt.show()


spins = checkerboard((100,100))
spins = spins*2 -1
E_before = tot_energy_lattice(spins)
kbT = 1
avg_spins = []
for i in range(10**6):
    # print(my_spins)
    avg_spins.append(spins.mean())
    E_before = monte_carlo(E_before,spins, kbT)
plt.imshow(spins)
plt.title(f"100,000 Monte Carlo steps; kbT = {kbT}")
plt.colorbar()
plt.show()

plt.plot(avg_spins)
plt.title(f"Avg. Total Magnetization versus step; kbT = {kbT}")
plt.xlabel("Step number")
plt.ylabel("Avg. Total Magnetization")
plt.show()
