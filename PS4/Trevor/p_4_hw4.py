#!/usr/bin/env python

'''
The following code will conduct molecular dynamics
The units used
E [=] kj/mol
Mass [=] g/mol
Forces [=] kj/(mol*nm)

Note: need to convert to Angstrom when writing the .xyz file

'''

#Part a)
import numpy as np
import matplotlib.pyplot as plt

def LJ(r1, r2, e=1.77, sig=.410):
    """
    Lennard-Jones potential and force calculation

    Parameters
    ----------
    r1, r2 : numpy array
        A coordinate array of one atom. (nm)

    e : float
        The depth of potential curve (kJ/mol)

    sig : float
        The distance that provides zero potential (nm)

    Returns
    ----------
    potential : float
        Calculated potential

    force_vec1, force_vec2 : numpy array
        A numpy array with x, y and z forces of an atom.
    """
    r = r1 - r2
    r_norm = np.linalg.norm(r)
    # want force to be in units of kJ/(mol*nm)
    conversion_factor = 10 # Angstrom/nm This was already applied elsewhere to make units consistent
    potential = 4 * e * ((sig / r_norm) ** 12 - (sig / r_norm) ** 6)
    force = (
        4
        * e
        * (-12 * (sig) ** 12 * (1 / r_norm) ** 13 + 6 * (sig) ** 6 * (1 / r_norm) ** 7)
    )
    r_vec = r/r_norm  # r unit vector
    force_vec1 = -force * r_vec # product of force and r_vec (kJ/(mol*nm))
    force_vec2 = -force_vec1
    return potential, force_vec1, force_vec2

#Part b)

def find_delta_v(force, time_step, mass):
    """"
    Finding the change in velocity over a time step delta t.

    Parameters
    ----------
    force : numpy array
        A numpy array with x, y and z forces of an atom. (kJ/(mol*nm))

    time_step : float
        Time step (ps)

    mass : float
        Mass of a given atom. (g/mol)

    Returns
    ----------
    delta_v : numpy array
        A numpy array with x, y, and z velocity components. (nm/ps)


    """
    delta_v = (force/mass) * time_step
    # kJ/(g*nm)*ps -> 10^3((g*m^2/s^2)/(g*nm))*ps -> 10^3*m^2/(s^2*nm) * ps -> nm / ps
    return delta_v

#Part c)

def find_delta_x(velocity, time_step, force, mass):
    """"
    Finding the change in velocity over a time step delta t.

    Parameters
    ----------
    velocity: numpy array
        A numpy array with x, y, and z velocity components. (nm/ps)

    force : numpy array
        A numpy array with x, y and z forces of an atom. (kJ/(mol*nm))

    time_step : float
        Time step (ps)

    mass : float
        Mass of a given atom. (g/mol)

    Returns
    ----------
    delta_x : numpy array
        A numpy array with x, y, and z position components. (nm)

    """
    delta_x = velocity*time_step + 0.5 * (force/mass) * (time_step**2)
    return delta_x


#Part d)

def get_force_array_and_pot_energy(positions):
    force_array = np.zeros_like(positions)
    E_tot = 0
    for i in range(positions.shape[0]):
        for j in range(i):
            E, force1, force2 = LJ(positions[i], positions[j])
            E_tot += E
            force_array[i] += force1
            force_array[j] += force2
    return force_array, E_tot

def euler_molecular_dynamics(current_positions, current_velocities, time_step, mass):
    force_array, pot_energy = get_force_array_and_pot_energy(current_positions)
    pos_updates = find_delta_x(current_velocities, time_step, force_array, mass)
    vel_updates = find_delta_v(force_array, time_step, mass)
    new_pos = current_positions + pos_updates
    new_vel = current_velocities + vel_updates
    return pot_energy, new_pos, new_vel

def make_cube(n, spacing):
    positions = np.zeros((n ** 3, 3))
    row = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                positions[row] = i * spacing, j * spacing, k * spacing
                row += 1
    return positions


def get_kinetic_energy(velocity, mass):
    KE = (0.5) * mass * (np.sum(velocity**2))
    return KE

#Part e)
time_step = 0.002 #ps
num_steps = range(25,000)
mass = 131.29 #g/mol
min_energy_dist = .46021 #   nm
current_positions = make_cube(3, min_energy_dist)
current_velocities = np.zeros_like(current_positions)
force_array = np.zeros_like(current_positions)
E_tot = []
ke = []
pe = []


for i in range(25000):
    print(i)
    kin_energy = get_kinetic_energy(current_velocities, mass)
    pot_energy, current_positions, current_velocities = euler_molecular_dynamics(current_positions, current_velocities, time_step, mass)
    pe.append(pot_energy)
    ke.append(kin_energy)
    E_tot.append(kin_energy + pot_energy)
    if i==10000:
        break
print(E_tot, "totallll energy")
print(E_tot, 'Etotal')
plt.plot(range(len(E_tot)), E_tot)
plt.title('Total Energy (kJ) vs. Simulation Step (Euler)')
plt.xlabel('Time (ps)')
plt.show()
plt.plot(range(len(ke)), ke)
plt.title('kin Energy (kJ) vs. Simulation Step (Euler)')
plt.xlabel('Time (ps)')
plt.show()
plt.plot(range(len(pe)), pe)
plt.title('pot Energy (kJ) vs. Simulation Step (Euler)')
plt.xlabel('Time (ps)')
plt.show()

#Part f)
def get_kinetic_energy(velocity, mass):
    KE = (0.5) * mass * (np.linalg.norm(velocity))**2
    return KE

#Part g)

def verlet_delta_v(forces, updated_forces, time_step, mass):
    delta_v = (time_step / (2*mass)) * (forces + updated_forces)
    return delta_v

def verlet_molecular_dynamics(force_array, current_positions, current_velocities, time_step, mass):
    pos_updates = find_delta_x(current_velocities, time_step, force_array, mass)
    new_pos = current_positions + pos_updates
    updated_forces, pot_energy = get_force_array_and_pot_energy(new_pos)
    delta_v = verlet_delta_v(force_array, updated_forces, time_step, mass)
    new_vel = current_velocities + delta_v
    return updated_forces, pot_energy, new_pos, new_vel


##### Time step 0.002
time_step = 0.002 #ps
mass = 131.29 #g/mol
min_energy_dist = .46021  # Angstrom
current_positions = make_cube(3, min_energy_dist)
current_velocities = np.zeros_like(current_positions)
force_array, pot_energy = get_force_array_and_pot_energy(current_positions)
E_tot = []
E_tot.append(pot_energy)

for i in range(25000):
    print(i)
    force_array, pot_energy, current_positions, current_velocities = verlet_molecular_dynamics(force_array, current_positions, current_velocities, time_step, mass)
    kin_energy = get_kinetic_energy(current_velocities, mass)
    E_tot.append(pot_energy + kin_energy)
    if i==10000:
        break

print(E_tot, 'Etotal')
plt.plot(range(len(E_tot)), E_tot)
plt.title('Total Energy (kJ) vs. Simulation Step (Verlet)')
plt.xlabel('Time (ps)')
plt.show()

############## larger time step

time_step = 0.005 #ps
mass = 131.29 #g/mol
min_energy_dist = .46021  # Angstrom
current_positions = make_cube(3, min_energy_dist)
current_velocities = np.zeros_like(current_positions)
force_array, pot_energy = get_force_array_and_pot_energy(current_positions)
E_tot = []
E_tot.append(pot_energy)

for i in range(10000):
    print(i)
    force_array, pot_energy, current_positions, current_velocities = verlet_molecular_dynamics(force_array, current_positions, current_velocities, time_step, mass)
    kin_energy = get_kinetic_energy(current_velocities, mass)
    E_tot.append(pot_energy + kin_energy)
    if i==10000:
        break

print(E_tot, 'Etotal')
plt.plot(range(len(E_tot)), E_tot)
plt.title('Total Energy (kJ) vs. Simulation Step (Verlet)')
plt.xlabel('Time (ps)')
plt.show()

############## smaller time step

time_step = 0.001 #ps
mass = 131.29 #g/mol
min_energy_dist = .46021  # Angstrom
current_positions = make_cube(3, min_energy_dist)
current_velocities = np.zeros_like(current_positions)
force_array, pot_energy = get_force_array_and_pot_energy(current_positions)
E_tot = []
E_tot.append(pot_energy)

for i in range(50000):
    print(i)
    force_array, pot_energy, current_positions, current_velocities = verlet_molecular_dynamics(force_array, current_positions, current_velocities, time_step, mass)
    kin_energy = get_kinetic_energy(current_velocities, mass)
    E_tot.append(pot_energy + kin_energy)
    if i==10000:
        break

print(E_tot, 'Etotal')
plt.plot(range(len(E_tot)), E_tot)
plt.title('Total Energy (kJ) vs. Simulation Step (Verlet)')
plt.xlabel('Time (ps)')
plt.show()