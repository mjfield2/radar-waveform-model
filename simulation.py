import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import griddata
from model import *


# Load the earth model example from file
earth_model = loadmat('crosshole_model.mat')
ep = earth_model['ep']
mu = earth_model['mu']
sig = earth_model['sig']
x = earth_model['x'].flatten()
z = earth_model['z'].flatten()

# Calculate minimum and maximum relative permittivity and permeability
epmin = np.min(ep)
epmax = np.max(ep)
mumin = np.min(mu)
mumax = np.max(mu)

# Create time (s) and source pulse vectors for use with finddx.m
t = np.arange(0, 100e-9, 1e-10)
srcpulse = blackharriswave(100e6, t)

# Use finddx.m to determine maximum possible spatial field discretization
dx, wlmin, fmax = finddx(epmax, mumax, srcpulse, t, 0.02)
print(f"Maximum frequency contained in source pulse = {fmax / 1e6} MHz")
print(f"Minimum wavelength in simulation grid = {wlmin} m")
print(f"Maximum possible electric/magnetic field discretization (dx,dz) = {dx} m")
print(f"Maximum possible electrical property discretization (dx/2,dz/2) = {dx / 2} m")

# Set dx and dz here (m) using the above results as a guide
dx = 0.025
dz = 0.025
print(f"Using dx = {dx} m, dz = {dz} m")

# Find the maximum possible time step using this dx and dz
dtmax = finddt(epmin, mumin, dx, dz)
print(f"Maximum possible time step with this discretization = {dtmax / 1e-9} ns")

# Set proper dt here (s) using the above results as a guide
dt = 0.2e-9
print(f"Using dt = {dt / 1e-9} ns")

# Create time vector (s) and corresponding source pulse
t = np.arange(0, 250e-9, dt)
srcpulse = blackharriswave(100e6, t)

# Interpolate electrical property grids to proper spatial discretization
x2 = np.arange(min(x), max(x), dx / 2)
z2 = np.arange(min(z), max(z), dz / 2)
ep2 = gridinterp(ep, x, z, x2, z2, 'cubic')
mu2 = gridinterp(mu, x, z, x2, z2, 'cubic')
sig2 = gridinterp(sig, x, z, x2, z2, 'cubic')

# Plot electrical property grids to ensure that interpolation was done properly
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.imshow(ep.T, extent=[min(x), max(x), min(z), max(z)], aspect='auto')
plt.colorbar(label='ep')
plt.title('Original ep matrix')
plt.subplot(3, 1, 2)
plt.imshow(mu.T, extent=[min(x), max(x), min(z), max(z)], aspect='auto')
plt.colorbar(label='mu')
plt.title('Original mu matrix')
plt.subplot(3, 1, 3)
plt.imshow(sig.T, extent=[min(x), max(x), min(z), max(z)], aspect='auto')
plt.colorbar(label='sig')
plt.title('Original sig matrix')
plt.tight_layout()
plt.show()

# Pad electrical property matrices for PML absorbing boundaries
npml = 10
ep3, x3, z3 = padgrid(ep2, x2, z2, npml)
mu3, _, _ = padgrid(mu2, x2, z2, npml)
sig3, _, _ = padgrid(sig2, x2, z2, npml)

# Create source and receiver location matrices (includes type)
srcz = np.arange(0.5, 11, 0.25)
srcx = 0.5 * np.ones_like(srcz)
recx = 5.5 * np.ones_like(srcz)
recz = srcz
srctype = 2 * np.ones_like(srcz)
rectype = 2 * np.ones_like(srcz)
srcloc = np.column_stack((srcx, srcz, srctype))
recloc = np.column_stack((recx, recz, rectype))

# Set some output and plotting parameters
outstep = 1
plotopt = [1, 2, 50, 0.001]

# Run the simulation
gather, tout, srcx, srcz, recx, recz = TE_model2d(ep3, mu3, sig3, x3, z3, srcloc, recloc, srcpulse, t, npml, outstep, plotopt)




# Load the earth model example from file
earth_model = loadmat('reflection_model.mat')
ep = earth_model['ep']
mu = earth_model['mu']
sig = earth_model['sig']
x = earth_model['x'].flatten()
z = earth_model['z'].flatten()

# Calculate minimum and maximum relative permittivity and permeability
epmin = np.min(ep)
epmax = np.max(ep)
mumin = np.min(mu)
mumax = np.max(mu)

# Create time (s) and source pulse vectors for use with finddx.m
t = np.arange(0, 100e-9, 1e-10)
srcpulse = blackharrispulse(100e6, t)

# Use finddx.m to determine maximum possible spatial field discretization
dx, wlmin, fmax = finddx(epmax, mumax, srcpulse, t, 0.02)
print(f"Maximum frequency contained in source pulse = {fmax / 1e6} MHz")
print(f"Minimum wavelength in simulation grid = {wlmin} m")
print(f"Maximum possible electric/magnetic field discretization (dx,dz) = {dx} m")
print(f"Maximum possible electrical property discretization (dx/2,dz/2) = {dx / 2} m")

# Set dx and dz here (m) using the above results as a guide
dx = 0.04
dz = 0.04
print(f"Using dx = {dx} m, dz = {dz} m")

# Find the maximum possible time step using this dx and dz
dtmax = finddt(epmin, mumin, dx, dz)
print(f"Maximum possible time step with this discretization = {dtmax / 1e-9} ns")

# Set proper dt here (s) using the above results as a guide
dt = 8e-11
print(f"Using dt = {dt / 1e-9} ns")

# Create time vector (s) and corresponding source pulse
t = np.arange(0, 250e-9, dt)
srcpulse = blackharriswave(100e6, t)

# Interpolate electrical property grids to proper spatial discretization
x2 = np.arange(min(x), max(x), dx / 2)
z2 = np.arange(min(z), max(z), dz / 2)
ep2 = gridinterp(ep, x, z, x2, z2, 'nearest')
mu2 = gridinterp(mu, x, z, x2, z2, 'nearest')
sig2 = gridinterp(sig, x, z, x2, z2, 'nearest')

# Pad electrical property matrices for PML absorbing boundaries
npml = 10
ep3, x3, z3 = padgrid(ep2, x2, z2, npml)
mu3, _, _ = padgrid(mu2, x2, z2, npml)
sig3, _, _ = padgrid(sig2, x2, z2, npml)

# Create source and receiver location matrices
srcx = np.arange(0, 20, 0.2)
srcz = np.zeros_like(srcx)
recx = srcx + 1
recz = srcz
srcloc = np.column_stack((srcx, srcz))
recloc = np.column_stack((recx, recz))

# Set some output and plotting parameters
outstep = 4
plotopt = [1, 50, 0.002]

# Run the simulation
gather, tout, srcx, srcz, recx, recz = TM_model2d(ep3, mu3, sig3, x3, z3, srcloc, recloc, srcpulse, t, npml, outstep, plotopt)

# Extract common offset reflection GPR data from multi-offset data cube and plot the results
co_data = np.zeros((len(tout), len(srcx)))
for i in range(len(srcx)):
    co_data[:, i] = gather[:, i, i]
pos = (srcx + recx) / 2
plt.figure()
plt.imshow(co_data.T, extent=[0, 20, 0, 250], aspect='auto', cmap='gray', vmin=-5e-4, vmax=5e-4)
plt.colorbar(label='Amplitude')
plt.xlabel('Position (m)')
plt.ylabel('Time (ns)')
plt.title('Common Offset Reflection GPR Data')
plt.show()

