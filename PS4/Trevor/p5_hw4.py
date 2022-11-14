import numpy as np
import matplotlib.pyplot as plt


def LJ(r1, r2, atom1_array, atom2_array, ko=138.935456):
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

    ko : float
        Coulomb constant kJ * nm/(mol * e^2)

    Returns
    ----------
    potential : float
        Calculated potential

    force_vec1, force_vec2 : numpy array
        A numpy array with x, y and z forces of an atom.
    """
    #need to include q1 and q2, +1 if Na, -1 if cl
    # need to include
    atom1 = atom1_array[0]
    atom2 = atom2_array[0]
    q1 = atom1_array[1]
    q2 = atom2_array[1]
    if atom1 != atom2:
        #use Lorentz-Berthelot combining rule
        sig = (atom1_array[2] + atom2_array[2]) / 2
        e = np.sqrt(atom1_array[3]*atom2_array[3])
    elif atom1 == atom2:
        sig = atom1_array[2]
        e = atom1_array[3]
    r = r1 - r2
    r_norm = np.linalg.norm(r)
    # want force to be in units of kJ/(mol*nm)
    conversion_factor = 10 # Angstrom/nm
    lj_potential = 4 * e * ((sig / r_norm) ** 12 - (sig / r_norm) ** 6)
    coulomb_potential = ko * q1 * q2 / r_norm
    total_potential = lj_potential + coulomb_potential
    force = (
        4
        * e
        * (-12 * (sig) ** 12 * (1 / r_norm) ** 13 + 6 * (sig) ** 6 * (1 / r_norm) ** 7)) - (ko * q1 * q2) / (r_norm**2)

    r_vec = r / r_norm  # r unit vector
    force_vec1 = -force * r_vec   # product of force and r_vec (kJ/(mol*nm))
    force_vec2 = -force_vec1
    return total_potential, force_vec1, force_vec2


#Part b)


def checkerboard(shape):
    return np.indices(shape).sum(axis=0) % 2
board = checkerboard((4,4,4))
print(board)

def make_cube(n, spacing = 0.282, atoms = [11, 17]):
    positions = np.zeros((n ** 3, 3))
    atom_array = np.zeros(((n ** 3), 5))
    row = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                print(board[i, j, k])
                positions[row] = i * spacing, j * spacing, k * spacing
                if board[i, j, k] % 2 == 0:
                    #atom identity, charge, sigma, e
                    atom_array[row] = atoms[0], 1, 0.332840, 0.011590, 22.98977
                else:
                    atom_array[row] = atoms[1], -1, 0.440104, 0.418400, 35.453
                row += 1
    return positions, atom_array

positions, atom_array  = make_cube(4)

print(positions)
print(atom_array)
with open('nacl.xyz', 'w') as f:
    print("%i" % positions.shape[0], file=f)
    print('NaCL starting traj', file=f)
    for row in range(len(positions)):
        if atom_array[row][0] == 11:
            print("Na % 12.6f % 12.6f % 12.6f" % (positions[row][0], positions[row][1], positions[row][2]), file=f)
        else:
            print("Cl % 12.6f % 12.6f % 12.6f" % (positions[row][0], positions[row][1], positions[row][2]), file=f)


def get_force_array_and_pot_energy(positions):
    force_array = np.zeros_like(positions)
    E_tot = 0
    for i in range(positions.shape[0]):
        for j in range(i):
            E, force1, force2 = LJ(positions[i], positions[j], atom1_array = atom_array[i], atom2_array = atom_array[j])
            E_tot += E
            force_array[i] += force1
            force_array[j] += force2
    return force_array, E_tot

force_array, E_tot = get_force_array_and_pot_energy(positions)
print(f"The total energy of the NaCL cube is {E_tot} kJ/mol")

#Part c)
#Minimize 1000 steps with steepest descent, step size 0.001nm
#INitial velocities with MB dist at 1200 K
#Use velocity verlet at constant energy from 10 ps using 5 fs time step (2000 total steps)
#Verify energy is conserved to within 1 kJ/mol

def write_xyz(positions, atom_array, fname, comment):
    """
    Write an xyz file

    Parameters:
    -----------
    fname : string
        file name

    comment : strting
        comment about the file
    """
    n = positions.shape[0]
    with open(f"{fname}", 'w') as f:
        f.write(f"{n}\n")
        f.write(f"{comment}\n")
        for row in range(len(positions)):
            if atom_array[row][0] == 11:
                print("Na % 12.6f % 12.6f % 12.6f" % (positions[row][0], positions[row][1], positions[row][2]), file=f)
            else:
                print("Cl % 12.6f % 12.6f % 12.6f" % (positions[row][0], positions[row][1], positions[row][2]), file=f)


def optimize(positions, stepsize, stepnumber, fname, comment):

    """
    A function optimizes a cube.

    Parameters:
    -----------
    stepsize : float
        A stepsize of the each iteration of optimization (Å).

    stepnumber : integer
        Iteration number for optimization

    fname : string
        file name

    comment : strting
        comment about the file

    Returns:
    ------------
    coord_opt : numpy array
        optimized coordinates

    energy_list : list
        recorded energies during optimization

    step_list : list
        recorded steps during optimization

    final_energy : float
        optimized energy
    """
    #self.cube_coord = np.copy(self.initial_coord)

    energy_list = []
    step_list = []
    print(f"\nOptimizing {fname} with stepsize of {stepsize}.")
    for i in range(stepnumber):
        force_array, E_tot = get_force_array_and_pot_energy(positions) # Calculating energy and force to take a step
        energy_list.append(E_tot)
        step_list.append(i)
        step = force_array / np.linalg.norm(force_array) * stepsize  # forces is a numpy array
        positions += step  # Takinga step
        if i % (stepnumber * 0.2) == 0:
            status = i / stepnumber * 100
            print(f"{status}% done.")
    print("----------Optimization completed----------")
    force_array, E_tot = get_force_array_and_pot_energy(positions)  # Calculating final energy and force
    write_xyz(positions, atom_array,fname= 'optimized_structure', comment= comment)
    return positions, energy_list, step_list, E_tot


positions, energy_list, step_list, E_tot = optimize(positions, stepsize= 0.001, stepnumber = 1000, fname= "Opt", comment= 'Starting Structure')

plt.plot(step_list, energy_list)
plt.show()

kb = 1.380649*10**-23 #m^2 kg s^-2 K^-1



def Andersen_thermostat(num_particles, atom_array, temp):
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
    velocities = np.zeros((num_particles,3))
    for count, row in enumerate(atom_array):
        mass = row[4]
        mass = mass/ (6.023 * 10 ** 23)
        standard_deviation = np.sqrt(kb * temp / mass)
        velocity = standard_deviation * np.random.randn(1, 3)
        velocities[count,] = velocity
    return velocities/1000

initial_velocities = Andersen_thermostat(64, atom_array, 1200)
print(initial_velocities)

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

    mass : numpy array
        Mass of a given atom. (g/mol)

    Returns
    ----------
    delta_x : numpy array
        A numpy array with x, y, and z position components. (nm)

    """
    delta_x = velocity*time_step + 0.5 *(force)*(1/mass.reshape(64, 1)) * (time_step**2)
    return delta_x

def verlet_delta_v(forces, updated_forces, time_step, mass):
    delta_v = (time_step / (2*mass.reshape(64, 1))) * (forces + updated_forces)
    return delta_v

def verlet_molecular_dynamics(force_array, current_positions, current_velocities, time_step, mass):
    pos_updates = find_delta_x(current_velocities, time_step, force_array, mass)
    new_pos = current_positions + pos_updates
    updated_forces, pot_energy = get_force_array_and_pot_energy(new_pos)
    delta_v = verlet_delta_v(force_array, updated_forces, time_step, mass)
    new_vel = current_velocities + delta_v
    return updated_forces, pot_energy, new_pos, new_vel

def get_kinetic_energy(velocity, mass):
    KE = np.sum((0.5) * mass.reshape(64, 1) * (velocity)**2)

    return KE

time_step = 0.005 #ps
num_steps = range(25,000)
#mass = 131.29 #g/mol
mass_array = atom_array[:, 4]
print(mass_array, 'Mass_array')
current_positions = positions.copy()
current_velocities = initial_velocities
force_array, pot_energy = get_force_array_and_pot_energy(current_positions)
kin_energy = get_kinetic_energy(current_velocities, mass_array)
print(kin_energy, 'hi im ke')
ke = 0
for i in range(len(mass_array)):
    ke += 0.5 * (mass_array[i]) * (np.linalg.norm(current_velocities[i, :]))**2
ke = np.sum(ke)
print(ke, 'is this the same?')
E_tot = []

#E_tot.append(pot_energy + kin_energy)

for i in range(2000):
    print(i)
    force_array, pot_energy, current_positions, current_velocities = verlet_molecular_dynamics(force_array, current_positions, current_velocities, time_step, mass_array)
    kin_energy = get_kinetic_energy(current_velocities, mass_array)
    E_tot.append(pot_energy + kin_energy)


print(E_tot, 'Etotal')
print(E_tot)
plt.plot(range(len(E_tot[50:])), E_tot[50:])
plt.title('Total Energy (kJ) vs.  Step (Verlet)')
plt.xlabel('Time (ps)')
#plt.ylim(np.max(E_tot) - 1000, np.max(E_tot) + 1000)
plt.show()

#Part d)
time_step = 0.005 #ps
num_steps = range(25,000)
#mass = 131.29 #g/mol
mass_array = atom_array[:, 4]
print(mass_array, 'Mass_array')
current_positions = positions.copy()
initial_velocities = Andersen_thermostat(64, atom_array, 1200)
current_velocities = initial_velocities
force_array, pot_energy = get_force_array_and_pot_energy(current_positions)
kin_energy = get_kinetic_energy(current_velocities, mass_array)
print(kin_energy, 'hi im ke')
ke = 0
for i in range(len(mass_array)):
    ke += 0.5 * (mass_array[i]) * (np.linalg.norm(current_velocities[i, :]))**2
ke = np.sum(ke)
print(ke, 'is this the same?')
E_tot = []
Rg = []

def radius_of_gyration(positions):
    return np.sqrt((1 / 64) * np.sum((positions - np.mean(positions))**2))


print()
for i in range(100000):
    current_step = i
    print(current_step)
    force_array, pot_energy, current_positions, current_velocities = verlet_molecular_dynamics(force_array, current_positions, current_velocities, time_step, mass_array)
    kin_energy = get_kinetic_energy(current_velocities, mass_array)
    E_tot.append(pot_energy + kin_energy)
    if (current_step +1) in [i for i in range(250, 100000, 250)]:
        print(current_step)
        current_velocities = Andersen_thermostat(64, atom_array, 1200)
        Rg.append(radius_of_gyration(current_positions))
    if i == 1000:
        break


print(E_tot, 'Etotal')
print(E_tot)
plt.plot(range(len(E_tot)), E_tot)
plt.title('Total Energy (kJ) vs.  Step (Verlet)')
plt.xlabel('Time (ps)')
#plt.ylim(np.max(E_tot) - 1000, np.max(E_tot) + 1000)
plt.show()
