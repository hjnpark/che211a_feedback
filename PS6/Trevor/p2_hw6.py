import numpy as np
import matplotlib.pyplot as plt
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import time

"""
Part a) Run simulation at room temp (300K) and 1 atm for 100ps. 
"""

'''#system details
pdb = PDBFile('waterbox.pdb')
forcefield = ForceField('tip3p.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=0.9*nanometer, rigidWater=True)

langevin_integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 2*femtosecond)
monte_carlo_barostat = MonteCarloBarostat(1.0 * atmosphere, 300 * kelvin)

system.addForce(monte_carlo_barostat)

simulation = Simulation(pdb.topology, system, langevin_integrator)

simulation.context.setPositions(pdb.positions)
energy = simulation.context.getState(getEnergy=True)
reporter1 = StateDataReporter(sys.stdout, 10, step=True, potentialEnergy=True,
                              kineticEnergy=True, totalEnergy=True, temperature = True)
reporter2 = DCDReporter('H2O_sim_300K.dcd', 10)

simulation.reporters.append(reporter1)
simulation.reporters.append(reporter2)

simulation.step(50000)'''

'''
b) word doc
'''

'''
c)
'''
'''#system details
pdb = PDBFile('waterbox.pdb')
forcefield = ForceField('tip3p.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=0.9*nanometer, rigidWater=True)

langevin_integrator = LangevinIntegrator(370*kelvin, 1.0/picosecond, 2*femtosecond)
monte_carlo_barostat = MonteCarloBarostat(1.0 * atmosphere, 370 * kelvin)

system.addForce(monte_carlo_barostat)

simulation = Simulation(pdb.topology, system, langevin_integrator)

simulation.context.setPositions(pdb.positions)
energy = simulation.context.getState(getEnergy=True)
reporter1 = StateDataReporter(sys.stdout, 10, step=True, potentialEnergy=True,
                              kineticEnergy=True, totalEnergy=True, temperature = True)
reporter2 = DCDReporter('H2O_sim_370K.dcd', 10)

simulation.reporters.append(reporter1)
simulation.reporters.append(reporter2)

simulation.step(50000)'''

'''
d)
'''
tip3p_data = np.loadtxt('300K_tip3p_gofr.dat')
plt.plot(tip3p_data.T[0], tip3p_data.T[1], label = 'tip3p at 300K')
tip3p_370K_data = np.loadtxt('370K_tip3p.dat')
plt.plot(tip3p_370K_data.T[0], tip3p_370K_data.T[1], label = 'tip3p ad 370K')
tip3pFB_300K = np.loadtxt('300K_FB.dat')
plt.plot(tip3pFB_300K.T[0], tip3pFB_300K.T[1], label = 'tip3pFB 300K')
experimental_data = np.loadtxt('oo.als_bestfit.dat')
plt.plot(experimental_data.T[0], experimental_data.T[1], label = 'Exp at 300K')
plt.legend()
plt.show()

'''#system details
pdb = PDBFile('waterbox.pdb')
forcefield = ForceField('tip3pfb.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=0.9*nanometer, rigidWater=True)

langevin_integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 2*femtosecond)
monte_carlo_barostat = MonteCarloBarostat(1.0 * atmosphere, 300 * kelvin)

system.addForce(monte_carlo_barostat)

simulation = Simulation(pdb.topology, system, langevin_integrator)

simulation.context.setPositions(pdb.positions)
energy = simulation.context.getState(getEnergy=True)
reporter1 = StateDataReporter(sys.stdout, 10, step=True, potentialEnergy=True,
                              kineticEnergy=True, totalEnergy=True, temperature = True)
reporter2 = DCDReporter('H2O_sim_FB_300K.dcd', 10)

simulation.reporters.append(reporter1)
simulation.reporters.append(reporter2)

simulation.step(50000)'''