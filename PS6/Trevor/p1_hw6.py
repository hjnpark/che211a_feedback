from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt

#CONSTANTS
# Boltzmann constant in kJ/mol
kb = 1.38064852e-23

# Planck's constant
h = 6.62607004e-34 # j * s

# Avogadro's number
NA = 6.0221409e23
print(NA)

# Speed of light
c = 299792458
'''
Part a)
Creating table for species in Haber-Bosch reaction
'''

def rel_energies(species, molecule_energy):
    '''
    The following function takes molecular energies and converts them
    to energies relative to individual atoms

    Parameters
    ---------
    species: string
        Chemical Formula
    molecule_energy: Hartree
        Obtained from CCCBDB at CCSD(T)=FULL with aug-cc-pVQZ basis

    Returns
    -------
    Relative energy of molecule compared to the energy of its constituent atomic species
    kJ / molecule
    '''
    atom_energies = {
        'H' : -0.499948,
        'N' : -54.553848,
    }
    atom_ref_energies = 0
    character_old = 0
    for count, character in enumerate(species):
        if count != 0:
            if not character.isdigit() and not character_old.isdigit():
                num = 1
                atom_ref_energies += num * atom_energies[f"{character_old}"]
            if character.isdigit():
                num = int(character)
                atom_ref_energies += num * atom_energies[f"{character_old}"]
                print(num)
        character_old = character
    print(f"mol energy {molecule_energy} atom ref {atom_ref_energies}")
    return molecule_energy - atom_ref_energies


species_data_dict = {}

species = ['H2', 'N2', 'NH3']
mol_mass = [2.016, 28.02, 17.03]
'''
Why are we able to use 6 instead of 3?
'''
sym_number = [2, 2, 6]
rot_const = [60.85300, 1.99824, [9.44430, 9.44430, 6.19600]]
#For NH3 there are actually degenerate vibrational modes 2 @ 1627 and 2 @ 3444
vib_freq = [4401.21, 2358.57, [3337, 950, 3444, 3444, 1627, 1627]]
energy = [-1.173864, -109.467423, -56.527294]
rel_e = []
for count, e in enumerate(energy):
    rel_e.append(rel_energies(species[count], e) * 2625.5) # converting hartree to kJ/mol

'''
It would be nice to make a nested dictionary, so that you could intuitively extract info
ex) species_data_dict['H2']['sym_number']

It requires more steps when extracting info for making table.
Need to practice with nested dictionaries
'''
species_data_dict["species"] = species
species_data_dict['mol_mas \namu'] = mol_mass
species_data_dict['sym_number'] = sym_number
species_data_dict['rot_const \n cm^-1'] = rot_const
species_data_dict['vib_freq \n cm^-1'] = vib_freq
species_data_dict['rel_energy \n kJ/molecule'] = rel_e

print(species_data_dict.keys())

print(tabulate(species_data_dict, headers='keys', tablefmt='fancy_grid'))

'''
Part c)
Creating functions to calculate translation partition functions
'''

def translation_partition_function(mass, temp):
    '''
    Calculates the molecular translation partition function at specified temp. The volume contribution has been removed.

    Parameters:
    --------
    mass: float
        Mass of molecule in amu.

    Returns:
    --------
    z: float
        Molecular partition function in units of volume
    '''
    h = 6.626 * 10**-34
    Na = 6.023 * 10**23 #Avogadros Number
    kb = 1.380649 * 10**-23 #kJ/mol
    m = mass / 1000 / Na
    kbT = kb * temp
    thermal_wavelength = h / (np.sqrt(2 * np.pi * m * kbT))
    z = 1 / (thermal_wavelength**3) # this is in m^-3
    return z

tr = translation_partition_function(species_data_dict['mol_mas \namu'][1], 273.15)
print(f"compare this to nitrogen Ztr {tr*3.720e-26}")

def rotational_partition_highT_approx_linear(sigma, temp, rot_const):
    h = 6.62607004e-34 # j * s
    kb = 1.380649 * 10 ** -23 # J/K
    freq = c * rot_const * 100 #Hz
    E = freq * h
    rot_temp = E/ kb
    T = temp
    z = T / (sigma * rot_temp)
    return z
print(f" test rotation part func {rotational_partition_highT_approx_linear(2, 273.15, 1.99824)}")

def rotational_partition_highT_approx_nonlinear(sigma, temp, rot_constant):
    A = rot_constant[0]
    B = rot_constant[1]
    C = rot_constant[2]

    h = 6.62607004e-34 # j * s
    kb = 1.380649 * 10 ** -23 # kJ/mol
    freq1 = c * A * 100 #Hz
    freq2 = c * B * 100 #Hz
    freq3 = c * C * 100  # Hz
   #E1 = freq1 * h/1000 * NA # kJ/mol
    #E2 = freq2 * h/1000 * NA # kJ/mol
    #E3 = freq3 * h/1000 * NA # kJ/mol
    E1 = freq1 * h/1000  # kJ/mol
    E2 = freq2 * h/1000   # kJ/mol
    E3 = freq3 * h/1000   # kJ/mol
    z = (np.sqrt(np.pi)/sigma) * np.sqrt(((kb*temp)**3) / (E1 * E2 * E3))
    return z#, E1, E2, E3
#print(f" rot nonllinear{rotational_partition_highT_approx_nonlinear(2, 273.15, [27.877, 14.512, 9.285])}")

def vib_partition_function(temp, vib_freq):
    h = 6.62607004e-34  # j * s
    kb = 1.380649 * 10 ** -23  # J/K
    c = 299792458
    #If else statements for handling one or many vib frequencies

    if len(vib_freq) == 1:
        vib_temp = vib_freq[0] * 100 * c * h / kb
        z = np.exp(-vib_temp / (2 * temp)) / (1 - np.exp(-vib_temp / temp))
    else:
        z = 1
        for freq in vib_freq:
            vib_temp = freq * 100 * c * h / kb
            z *= np.exp(-vib_temp / (2 * temp)) / (1 - np.exp(-vib_temp / temp))
    return z

print(f"vib test {vib_partition_function(273.15, [2358])}")

def electronic_partition_function(temp, De):
    '''
    Calculates the molecular electronic partition function

    Parameters
    ----------
    temp: np.array
        Temps in K

    De: float
        Formation energy of molecule from its constituent atoms (kJ/molecule)


    Returns
    -------
    np.array
        Molecular electronic partition function at specified temp

    '''
    kb = 1.380649 * 10 ** -23  # J/K
    De *= -1
    z = np.exp(De/(kb/1000*NA * temp))
    return z


#print(rotational_partition_highT_approx_linear(2, 273.15, 1.99824))
#print(rotational_partition_highT_approx_nonlinear(2, 273.15, 27.877, 14.512, 9.285))

print(electronic_partition_function(800, -29))

def haber_equil_const_empirical_fit(temp):
    log_Keq = 2.1 + (2098/temp) - (2.5088 * np.log10(temp)) - (1.006e-4 * temp) + (1.86e-7 * temp**2) + np.log10(0.0821 * temp)
    return log_Keq

temps = np.arange(800, 1200, 10)
print(10**haber_equil_const_empirical_fit(800))

plt.plot(temps, haber_equil_const_empirical_fit(temps))
plt.show()


conc_std = 6.022e26
volume = 3.720e-26
Keq = []
for temp in temps:
    z_trans_H2 = translation_partition_function(species_data_dict['mol_mas \namu'][0], temps)
    z_vib_H2 = vib_partition_function(temps, [species_data_dict['vib_freq \n cm^-1'][0]])
    z_el_H2 = electronic_partition_function(temps, species_data_dict['rel_energy \n kJ/molecule'][0])
    z_rot_H2 = rotational_partition_highT_approx_linear(2, temps, species_data_dict['rot_const \n cm^-1'][0])
    z_H2 = z_el_H2 * z_rot_H2 * z_vib_H2 * z_trans_H2

    z_trans_N2 = translation_partition_function(species_data_dict['mol_mas \namu'][1], temps)
    z_vib_N2 = vib_partition_function(temps, [species_data_dict['vib_freq \n cm^-1'][1]])
    z_el_N2 = electronic_partition_function(temps, species_data_dict['rel_energy \n kJ/molecule'][1])
    z_rot_N2 = rotational_partition_highT_approx_linear(2, temps, species_data_dict['rot_const \n cm^-1'][1])
    z_N2 = z_vib_N2 * z_rot_N2 * z_el_N2  * z_trans_N2

    z_trans_NH3 = translation_partition_function(species_data_dict['mol_mas \namu'][2], temps)
    z_vib_NH3 = vib_partition_function(temps, species_data_dict['vib_freq \n cm^-1'][2])
    z_el_NH3 = electronic_partition_function(temps, species_data_dict['rel_energy \n kJ/molecule'][2])
    z_rot_NH3 = rotational_partition_highT_approx_nonlinear(6, temps, species_data_dict['rot_const \n cm^-1'][2])
    z_NH3 = z_el_NH3 * z_rot_NH3 * z_vib_NH3 * z_trans_NH3


    Keq.append(volume * conc_std * z_NH3 / (((np.sqrt(z_N2)) * (np.sqrt(z_H2))**3)))

plt.plot(temps, 10**haber_equil_const_empirical_fit(temps))
plt.show()
'''
e)
One possible reason for the difference between calculated equilibrium 
constant and the experiment is having errors of a few kcal/mol in electronic energy,
which can throw off results by a factor of 10 or more. 
'''

'''
Unfortunately, I must have mad a mistake somewhere because my values are off.
'''