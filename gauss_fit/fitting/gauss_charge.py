""" Module to fit the electronic charge density of closed shell molecules obtained
    by self-consistent siesta calculations with Gaussian functionsi

    HOW TO
    -------
         1) run get_data()
         2) run get_atom_pos()
         3) run add_core_density()
         4) run fit_poly()
"""

import sys
import numpy as np
import scipy.optimize as opt
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
from .gauss_util import *
from .mono_fit import *
import struct
rho = np.zeros(2)
unitcell = np.zeros(2)
grid = np.zeros(2)
rhopath = 'RHO.gauss'


def get_data(file_path):
    """Import data from RHO file

    Structure of RHO file:
    first three lines give the unit cell vectors
    fourth line the grid dimensions
    subsequent lines give density on grid

    Parameters:
    -----------

    file_path: string; path to RHO file from which density is read

    Returns:
    --------
    None

    Other:
    ------
    unitcell: (3,3) np.array; saves the unitcell dimension in euclidean coordinates
    grid: (,3) np.array; number of grid points in each euclidean direction
    rho: (grid[1],grid[2],grid[3]) np.array; density on grid
    """

    global rho
    global unitcell
    global grid
    global rhopath
    rhopath = file_path
    unitcell = np.zeros([3, 3])
    grid = np.zeros([4])
    try:
        with open(file_path, 'r') as rhofile:

            # unit cell (in Bohr)
            for i in range(0, 3):
                unitcell[i, :] = rhofile.readline().split()

            grid[:] = rhofile.readline().split()
            grid = grid.astype(int)
            n_el = grid[0] * grid[1] * grid[2] * grid[3]

            # initiatialize density with right shape
            rho = np.zeros(grid)

            for z in range(grid[2]):
                for y in range(grid[1]):
                    for x in range(grid[0]):
                        rho[x, y, z, 0] = rhofile.readline()

        # closed shell -> we don't care about spin.
        rho = rho[:, :, :, 0]

        sync_global(rho, grid, unitcell,rhopath)
    except UnicodeDecodeError:
        get_data_bin(file_path)

def get_data_bin(file_path):
    """ Same as get_data for binary (unformatted) files
    """
    #Warning: Only works for cubic cells!!!
    #TODO: Implement for arb. cells

    global rho
    global unitcell
    global grid
    global rhopath

    bin_file = open(file_path, mode = 'rb')

    unitcell = '<I9dI'
    grid = '<I4iI'

    unitcell = np.array(struct.unpack(unitcell,
        bin_file.read(struct.calcsize(unitcell))))[1:-1].reshape(3,3)

    grid = np.array(struct.unpack(grid,bin_file.read(struct.calcsize(grid))))[1:-1]
    if (grid[0] == grid[1] == grid[2]) and grid[3] == 1:
        a = grid[0]
    else:
        raise Exception('get_data_bin cannot handle non-cubic unitcells or spin')

    block = '<' + 'I{}fI'.format(a)*a*a
    content = np.array(struct.unpack(block,bin_file.read(struct.calcsize(block))))

    rho = content.reshape(a+2, a, a, order = 'F')[1:-1,:,:]

    sync_global(rho, grid, unitcell,rhopath)

def set_rho(newrho):
    global rho
    gauss_util.rho = newrho
    rho = newrho

def add_core_density(molecule_list):
    """Adds the core density to the valence charge density stored in the global
    variable "rho"

    Parmeters:
    ----------
    molecule_list: list of Molecule objects

    Returns:
    --------
    None
    """
    if not isinstance(molecule_list,list):
        molecule_list = [molecule_list]

    global rho
    if rho == np.zeros(2):
        raise Exception("Valence density not loaded")

    Xm, Ym, Zm = mesh_3d()
    Z = Zm * unitcell[2, 2] / grid[2]
    Y = Ym * unitcell[1, 1] / grid[1]
    X = Xm * unitcell[0, 0] / grid[0]
    box_vol = unitcell[0, 0] / grid[0] * unitcell[1, 1] / grid[1] * unitcell[
        2, 2] / grid[2]

    core_charge = np.zeros_like(X)
    for molecule in molecule_list:
        n_core_el = 0
        for atom in molecule.get_atom_list():
            if atom.is_core_corrected():
                core_charge += atom.core_density(X, Y, Z)
                n_core_el += atom.core_electrons
        # Due to coarse graining around core normalization is not necessarily equal
        # to number of electrons => Introduce ad-hoc correction to fix this
        #correction = n_core_el / (np.sum(core_charge) * box_vol)
        correction = 1.0
        # Add core charge density to global(valence) density
    rho[Xm, Ym, Zm] += core_charge * correction


# ========================= Main Routine - Fit ====================  #


def fit_poly(molecule_list,
             rmax=0,
             use_sym=False,
             plot_out=False,
             out_path=rhopath,
             box_buffer=2.0,
             write_xyz=False,
             U = 0,
             colin_reg = 0,
             mp_reg = []):

    """ Fits n_gauss Gaussians to charge density from any number of  water molecules

       Input parameters:
       -----------------
       molecule_list: list of Molecule objects/ Molecule object
       rmax: 3d array/list; lower and upper box limits; rmax = 0 uses full grid,
             rmax = -1: automatically determine smallest box around molecules
       use_sym: boolean; use the symmetries specified for each molecule
       plot_out: after fitting is done, plot the resulting charge density and
                 compare to exact density
       out_path: path where coordinates are saved
       box_buffer: float, if box size is determined automatically this determines
                   the buffer size in Bohr around the molecules.
       write_xyz: boolean; write .xyz file
       U: float; overlap penalty
       colin_reg: float; regularization parameter that forces Gaussians lie on
            bonds, if colin_reg > 1e5 simply enforce projection
       mp_reg: [float]; regularization parameters for multipoles, the larger the more
        accurate the dipole moment will be reproduced, sacrificing fit quality.

       Returns:
       --------
       final_results, rmse, max_error

       final_results: pandas DataFrame; fitting parameters, positions in Bohr
       rmse: float; RMSE of fit
       max_error: float; Maximum absolute error of fit

       Other:
       ------
       The final_results DataFrame has 4 columns containing the coordinates of the
       Gaussian center and its width w defined as :
       G(r) = 2 * (w * np.pi)**(-3/2) * np.exp(-(r - r0)**2 / w)

    """

    # If single molecule is not passed as a list
    if not isinstance(molecule_list,list):
        molecule_list = [molecule_list]

    n_molecules = len(molecule_list)

    # Total amount of gaussians used to fit density
    n_gauss = 0

    # Number of atoms
    n_atoms = 0

    gauss_separator = []
    atom_separator = []
    for molecule in molecule_list:
        gauss_separator.append(n_gauss)
        atom_separator.append(n_atoms)
        n_gauss += molecule.get_n_gauss()
        n_atoms += molecule.get_n_atoms()

    atom_pos = np.zeros([n_atoms, 3])

    for m, molecule in enumerate(molecule_list):
        n_a = molecule.get_n_atoms()
        atom_pos[atom_separator[m]:atom_separator[m] + n_a, :] = molecule.get_atom_pos()

    # automatically determine smallest box around molecules
    if rmax == -1:
        rmax = smallest_box(atom_pos, box_buffer)

    # --------------- 3D fit -------------------- #

    Xm, Ym, Zm = mesh_3d(rmin=[0, 0, 0], rmax=rmax)
    X, Y, Z = mesh_3d(rmin=[0, 0, 0], rmax=rmax, scaled = True)

    sol = rho[Xm, Ym, Zm]

    V = unitcell[2, 2] / grid[2] * unitcell[1, 1] / grid[1] * unitcell[0, 0] / grid[0]

    mesh_size = len(X.flatten())
    fit_func = n_gauss_3d

    cost_len = len(X.flatten())

    # Molecule centered meshes
    mcm = []
    mcm_scaled = []

    for m in molecule_list:
        mcm.append(molecule_centered_mesh(m, buffer = box_buffer*2))
        mcm_scaled.append(molecule_centered_mesh(m, buffer = box_buffer*2, scaled = True))

    # Multipole moments
    comp_dipole = False
    comp_quadrupole = False
    dipoles =  []

    for i, reg in enumerate(mp_reg):
        if reg != 0:
            if i == 0:
                comp_dipole = True
                for m, mol in enumerate(molecule_list):
                    dp = dipole_moment(*mcm_scaled[m], V, atom_pos, gauss_util.rho_val[mcm[m][0],mcm[m][1],mcm[m][2]])
                    print('Fitting dipole moment {} for molecule {}'.format(dp,m + 1))
                    dipoles.append(dp)
                    cost_len += 3

                dp = dipole_moment(X,Y,Z, V, atom_pos, gauss_util.rho_val[Xm,Ym,Zm])
                print('Fitting dipole moment {} for total charge density'.format(dp))
                dipoles.append(dp)
                mcm_scaled.append([X,Y,Z])
                cost_len += 3

            elif i ==1 :

                comp_quadrupole = True
                quadrupole = quadrupole_moment(X, Y, Z, V, atom_pos, gauss_util.rho_val[Xm,Ym,Zm])
                print('Fitting quadrupole moment \n {} \n for total charge density'.format(quadrupole))
                cost_len += 9

    if U != 0:
        comp_u = True
        cost_len += 1
    else:
        comp_u = False

    if colin_reg != 0:
        comp_colin = True
        cost_len += 1
    else:
        comp_colin = False

    par_select = []

    index = 0
    for m,_ in enumerate(molecule_list):
         par_select += list(range(index,index + 16))
         index += 20


    # Cost function that is minimized by gradient descent
    def cost(par):

        cost_array = np.zeros(cost_len)
        start = 0


        # Penalty on Gaussian position
        if comp_colin:
            if colin_reg > 1e5: # If regularization is large enough simply enforce constraint
                par = restricted_to_euclid(par,molecule_list)
                cost_array[start] = 0
                start += 1
            else:
                cost_array[start] = colin_reg * colinear_cost(par, molecule_list)
                start += 1

        mp_par = np.array(par)[par_select].tolist()

        # Enforce symmetries
        if use_sym:
            for molecule, gs in zip(molecule_list, gauss_separator):
                n_g = molecule.get_n_gauss()
                par[gs * 4:gs * 4 + 4 * n_g] = molecule.use_constraints(
                    par[gs * 4:gs * 4 + 4 * n_g])


        # Density fit cost
        cost_array[start:start + mesh_size] = (fit_func(X, Y, Z, par, n_gauss) - sol).flatten()
        start += mesh_size

        # Overlap cost
        if comp_u:
            cost_array[start] = U/mesh_size * gauss_overlap(par, n_gauss)
            start += 1

        # Multipole cost
        if comp_dipole:
            for mesh, dipole in zip(mcm_scaled,dipoles):
                cost_array[start: start + 3] = mp_reg[0] * \
                    (dipole_moment(*mesh,V,atom_pos, fit_func(mesh[0],mesh[1],mesh[2],mp_par, n_gauss - len(molecule_list))) - dipole)
                start += 3

        if comp_quadrupole:

            cost_array[start:start + 9] = mp_reg[1] * \
                (quadrupole_moment(X,Y,Z,V,atom_pos,fit_func(X,Y,Z,mp_par, n_gauss - len(molecule_list))) - \
                    quadrupole).flatten()
            start += 9

        return cost_array


    # Get initial parameters for fit
    init_par = []
    for molecule in molecule_list:
        init_par += molecule.get_init()


    if colin_reg > 1e5:
        init_par = euclid_to_restricted(init_par, molecule_list)


    # Actual fitting
    fit = opt.least_squares(cost, init_par)


    # Apply constraints used during fitting
    if use_sym:
        if colin_reg > 1e5: # If regularization is large enough simply enforce constraint
            fit.x = restricted_to_euclid(fit.x,molecule_list)

        for molecule, gs in zip(molecule_list, gauss_separator):
            n_g = molecule.get_n_gauss()
            fit.x[gs * 4:gs * 4 + 4 * n_g] = molecule.use_constraints(
                fit.x[gs * 4:gs * 4 + 4 * n_g])


    Xf, Yf, Zf = mesh_3d(rmin=[0, 0, 0], rmax=0, scaled = True)

    print('Dipole_moment [D]: {}'.format(dipole_moment(Xf,Yf,Zf, V, atom_pos, fit_func(Xf,Yf,Zf,np.array(fit.x)[par_select].tolist(), n_gauss- len(molecule_list)))))
    print('Quadrupole moment [a.u.]: \n{}'.format(quadrupole_moment(Xf,Yf,Zf, V, atom_pos, fit_func(Xf,Yf,Zf,np.array(fit.x)[par_select].tolist(), n_gauss - len(molecule_list)))))

    sqrd_errors = ((fit_func(X, Y, Z, fit.x, n_gauss) - sol).flatten())**2
    rmse = np.sqrt(np.mean(sqrd_errors))
    max_error = np.max(np.sqrt(sqrd_errors))

    if plot_out:
        plot_overview(molecule_list, fit_func, fit.x, rmax)

    # Save Gaussian parameters
    final_results = pd.DataFrame(
        [list(fit.x[i * 4:i * 4 + 4]) for i in range(n_gauss)])
    final_results.to_csv(out_path, header=None, index=None)

    # Bad Fit Warning
    if rmse > 0.03 * n_molecules:
        print('WARNING: RMSE/Molecule > 0.03')

    if rmse > 0.05 * n_molecules:
        print('!!!!!!!!!!! WARNING: RMSE/Molecule > 0.05 !!!!!!!!!!!!!!!!')

    if write_xyz != False:
        output_xyz(out_path[:-6] + '.xyz', molecule_list, fit.x, write_xyz)

    # Needed by vary_parallel to determine wether all cores have finished
    print('done')
    return final_results, rmse, max_error
