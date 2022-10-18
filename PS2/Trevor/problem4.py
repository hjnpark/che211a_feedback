import numpy as np
import matplotlib.pyplot as plt

min_energy_dist = 4.6021  # Angstrom


def leonard_jones(r1, r2):
    sigma = 4.10  # angstrom
    dispersion_energy = 1.77  # kJ/mol

    r = (np.sum((r2 - r1) ** 2)) ** 0.5  # Distance between atoms in Angstrom
    interaction_energy = 4 * dispersion_energy * ((sigma / r) ** 12 - (sigma / r) ** 6)  # good ol' Leonard Jones

    return interaction_energy

###########Problem 4 b)
def lj_force(r1, r2):
    sigma = 4.10  # angstrom
    dispersion_energy = 1.77  # kJ/mol
    r = (np.sum((r2 - r1) ** 2)) ** 0.5  # Distance between atoms in Angstrom
    force = 4 * dispersion_energy * (-12 * (sigma ** 12) * (r ** -13) - (-6 * (sigma ** 6) * (r ** -7)))
    force_1 = -force * (r1 - r2) / r
    force_2 = -force_1
    return force_1, force_2


def make_cube(n, spacing):
    positions = np.zeros((n ** 3, 3))
    row = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                positions[row] = i * spacing, j * spacing, k * spacing
                row += 1
    return positions

###########Problem 4c)

positions = make_cube(2, min_energy_dist)
total_forces = np.zeros((8, 3))
N = positions.shape[0]
# Atoms are identified by index starting at 0
for atom_index in range(N - 1):
    r1 = positions[atom_index,]
    r2 = positions[(atom_index + 1):N, ]
    for atom_index_update, r in enumerate(r2):
        forces_1, forces_2 = lj_force(r1, r)
        total_forces[atom_index] += forces_1
        total_forces[atom_index+1+atom_index_update] += forces_2
print(total_forces, 'Total forces')
net_force = np.array([np.sum(np.square(row)) for row in total_forces]).reshape((8, 1))
print(f'The net force on each atom should be equivalent due to symmetry, as seen in... \n {net_force}')

# Part d)
positions = make_cube(3, min_energy_dist)
initial_xyz = open('initial_pos.xyz', 'w')
initial_xyz.write(f"{positions.shape[0]} \n \n")
for row in range(positions.shape[0]):
    initial_xyz.write(f"Xe {positions[row][0]} {positions[row][1]} {positions[row][2]} \n")
initial_xyz.close()

N = positions.shape[0]
tot_pot_list_med = []
# Atoms are identified by index starting at 0
for opt_step in range(3000):
    total_forces = np.zeros((27, 3))
    for atom_index in range(N - 1):
        r1 = positions[atom_index,]
        r2 = positions[(atom_index + 1):N, ]
        for atom_index_update, r in enumerate(r2):
            forces_1, forces_2 = lj_force(r1, r)
            total_forces[atom_index] += forces_1
            total_forces[atom_index+1+atom_index_update] += forces_2

    # normalizing force vector
    norm_force_vectors = total_forces/np.linalg.norm(total_forces) #np.round(total_forces,4)/net_force
    step_size = 0.01
    atom_displacements = norm_force_vectors*step_size
    positions += atom_displacements
    tot_pot = 0
    for a in range(N - 1):
        r1 = positions[(a + 1):N, ]
        r2 = positions[a,]
        for r in r1:
            tot_pot += leonard_jones(r, r2)
    tot_pot_list_med.append(tot_pot)

final_xyz = open('final_pos.xyz', 'w')
final_xyz.write(f"{positions.shape[0]} \n \n")
for row in range(positions.shape[0]):
    final_xyz.write(f"Xe {'{:.4f}'.format(positions[row][0])} {'{:.4f}'.format(positions[row][1])} {'{:.4f}'.format(positions[row][2])} \n")
final_xyz.close()

positions = make_cube(3, min_energy_dist)


N = positions.shape[0]
tot_pot_list_small = []
# Atoms are identified by index starting at 0
for opt_step in range(3000):
    total_forces = np.zeros((27, 3))
    for atom_index in range(N - 1):
        row = 0
        r1 = positions[atom_index,]
        r2 = positions[(atom_index + 1):N, ]
        for atom_index_update, r in enumerate(r2):
            forces_1, forces_2 = lj_force(r1, r)
            total_forces[atom_index] += forces_1
            total_forces[atom_index+1+atom_index_update] += forces_2

    # normalizing force vector
    norm_force_vectors = total_forces/np.linalg.norm(total_forces) #np.round(total_forces,4)/net_force
    step_size = 0.001
    atom_displacements = norm_force_vectors*step_size
    positions += atom_displacements
    tot_pot = 0
    for a in range(N - 1):
        r1 = positions[(a + 1):N, ]
        r2 = positions[a,]
        for r in r1:
            tot_pot += leonard_jones(r, r2)
    tot_pot_list_small.append(tot_pot)

positions = make_cube(3, min_energy_dist)


N = positions.shape[0]
tot_pot_list_large = []
# Atoms are identified by index starting at 0
for opt_step in range(3000):
    total_forces = np.zeros((27, 3))
    for atom_index in range(N - 1):
        r1 = positions[atom_index,]
        r2 = positions[(atom_index + 1):N, ]
        for atom_index_update, r in enumerate(r2):
            forces_1, forces_2 = lj_force(r1, r)
            total_forces[atom_index] += forces_1
            total_forces[atom_index+1+atom_index_update] += forces_2

    # normalizing force vector
    norm_force_vectors = total_forces/np.linalg.norm(total_forces) #np.round(total_forces,4)/net_force
    step_size = 0.1
    atom_displacements = norm_force_vectors*step_size
    positions += atom_displacements
    tot_pot = 0
    for a in range(N - 1):
        r1 = positions[(a + 1):N, ]
        r2 = positions[a,]
        for r in r1:
            tot_pot += leonard_jones(r, r2)
    tot_pot_list_large.append(tot_pot)



plt.plot(range(3000), tot_pot_list_small, label='0.001 step')
plt.plot(range(3000), tot_pot_list_med, label='0.01 step')
plt.plot(range(3000), tot_pot_list_large, label='0.1 step')
plt.title('Potential energy versus iteration number')
plt.xlabel('Iteration Number')
plt.ylabel('Potential Energy (kJ/mol)')
plt.legend()
plt.show()
