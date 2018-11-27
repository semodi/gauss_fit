"""Subroutines used by gauss_charge
"""

import numpy as np
import scipy.optimize as opt
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
from .mono_fit import local_axes

rho = np.zeros(2)
rho_val = np.zeros(2)
unitcell = np.zeros(2)
grid = np.zeros(2)
rhopath = 'RHO.gauss'

AtoBohr = 1.889725989
Dtoau = 0.393430307

def sync_global(rho_, grid_, unitcell_, rhopath_):
    global rho
    global unitcell
    global grid
    global rhopath
    global rhoval
    rho = rho_
    grid = grid_
    unitcell = unitcell_
    rhopath = rhopath_
    rho_val = np.array(rho_)

def check_norm():
    """ Check normalization of charge density

        Returns
        -------
        float; integrated charge density
    """
    Xm, Ym, Zm = mesh_3d()
    box_vol = unitcell[0, 0] / grid[0] * unitcell[1, 1] / grid[1] * unitcell[
        2, 2] / grid[2]
    return np.sum(rho[Xm, Ym, Zm]) * box_vol


def get_atom_pos(file_path, units='Bohr'):
    """Get the atomic positions saved in file_path
       order should be:  O H H O H ...
       Input Parameters:
       -----------------
       file_path: string, read geometry from this file
       units: string, units ( 'A' or 'Bohr' ) in which coordinates are saved in file,
                        will convert to Bohr (a.u.).

       Returns:
       --------
       atom_pos: (,3) array; atomic coordinates in Bohr
       """

    global atom_pos
    coord = pd.DataFrame.from_csv(file_path, index_col=None, header=None)
    if units != 'Bohr':
        atom_pos = np.array(coord) * AtoBohr
    else:
        atom_pos = np.array(coord)

    return atom_pos


def smallest_box(atom_pos, box_buffer=0.5):
    """Determine smallest box that includes all molecules.
       Called by fit_poly if rmax = -1

       Parameters
       ----------
       atom_pos: (,3) np.array; atomic coordinates
       box_buffer: float; buffer around smallest box

       Returns
       --------
       rmax: (3) list; the maximum box dimensions in 3 euclid. directions
    """

    rmax = [0, 0, 0]
    for a in atom_pos:
        for i in range(3):
            if abs(a[i]) > rmax[i]:
                rmax[i] = abs(a[i])
    for i in range(3):
        rmax[i] = (int)((rmax[i] + box_buffer) * grid[i] / unitcell[i, i])
        if rmax[i] > grid[i]:
            rmax[i] = grid[i]
    return rmax


# ==================== Mesh Functions ==================== #


def plane_cut(data,
              plane,
              height,
              unitcell,
              grid,
              rmin=[0, 0, 0],
              rmax=0,
              return_mesh=False):
    """return_mesh = False : returns a two dimensional cut through 3d data
                     True : instead of data, 2d mesh is returned

      Parameters:
      ----------
         data
         plane = {0: yz-plane, 1: xz-plane, 2:xy-plane}
         unitcell = 3x3 array size of the unitcell
         grid = 3x1 array size of grid
         rmin,rmax = lets you choose the min and max grid cutoff
                       rmax = 0 means the entire grid is used
         return_mesh = boolean; decides wether mesh or cut through data is returned
    """

    if rmax == 0:
        mid_grid = (grid / 2).astype(int)
        rmax = mid_grid

    # resolve the periodic boundary conditions
    x_pbc = list(range(-rmax[0], -rmin[0])) + list(range(rmin[0], rmax[0]))
    y_pbc = list(range(-rmax[1], -rmin[1])) + list(range(rmin[1], rmax[1]))
    z_pbc = list(range(-rmax[2], -rmin[2])) + list(range(rmin[2], rmax[2]))
    height = (int)(np.round(height * grid[plane] / unitcell[plane, plane]))

    pbc_grids = [x_pbc, y_pbc, z_pbc]
    pbc_grids.pop(plane)

    A, B = np.meshgrid(*pbc_grids)

    indeces = [A, B]
    indeces.insert(plane, height)
    if not return_mesh:
        return data[indeces[0], indeces[1], indeces[2]]
    else:
        return A, B


def mesh_3d(rmin=[0, 0, 0], rmax=0, scaled = False):
    """Returns a 3d mesh taking into account periodic boundary conditions

        Parameters
        ----------
        rmin, rmax: (3) list; lower and upper cutoff
        scaled: boolean; scale the meshes with unitcell size?

        Returns
        -------
        X, Y, Z: np.arrays; meshgrid
    """
    print(grid)
    if rmax == 0:
        mid_grid = (grid / 2).astype(int)
        rmax = mid_grid

    # resolve the periodic boundary conditions
    x_pbc = list(range(-rmax[0], -rmin[0])) + list(range(rmin[0], rmax[0] + 1))
    y_pbc = list(range(-rmax[1], -rmin[1])) + list(range(rmin[1], rmax[1] + 1))
    z_pbc = list(range(-rmax[2], -rmin[2])) + list(range(rmin[2], rmax[2] + 1))

    Xm, Ym, Zm = np.meshgrid(x_pbc, y_pbc, z_pbc)
    if scaled:
        Z = Zm * unitcell[2, 2] / grid[2]
        Y = Ym * unitcell[1, 1] / grid[1]
        X = Xm * unitcell[0, 0] / grid[0]
        return X,Y,Z
    else:
        return Xm,Ym,Zm

def molecule_centered_mesh(molecule, scaled = False, buffer = 2.0):

    coord = molecule.get_atom_pos()

    bounds = np.zeros([3,2])

    for i in range(3):
        bounds[i,0] = min(coord[:,i]) - buffer
        bounds[i,1] = max(coord[:,i]) + buffer


    # Convert to grid points

    bounds_grid = np.zeros([3,2])
    ranges = []
    for i in range(3):
        bounds_grid[i, :] = (bounds[i, :] * grid[i] / unitcell[i,i]).astype(int)
        if scaled:
            scaling = unitcell[i,i]/grid[i]
            ranges.append(np.arange(bounds_grid[i,0],bounds_grid[i,1] + 1)*scaling)
        else:
            scaling = 1
            ranges.append(np.arange(bounds_grid[i,0],bounds_grid[i,1] + 1).astype(int))


    return np.meshgrid(ranges[0],ranges[1],ranges[2])


# ================= Plotting ===================== #


def glimpse(rmin=[0, 0, 0], rmax=0, plane=2, height = 0):
    """Take a quick look at the loaded data in a particular plane

        Parameters
        ----------
        rmin,rmax: (3) list; upper and lower cutoffs
        plane = {0: yz-plane, 1: xz-plane, 2: xy-plane}
    """

    RHO = plane_cut(rho, plane, height, unitcell, grid, rmin=rmin, rmax=rmax)

    plt.figure()
    CS = plt.imshow(
        RHO, cmap=plt.cm.jet, origin='lower')
    plt.colorbar()
    plt.show()


def plot_overview(molecule_list, fit_func, fit, rmax=0):
    """Plot overview of the exact data and the fit.
       Called by fit(plot_out=True)

       Parameters:
       -----------
       molecule_list: list of molecule objects
       fit_func: function with which density was fitted (usually Gaussians)
       fit: fitting parameters
       rmax: upper cutoff
    """
#TODO: for multiple molecules fix plotting

    def plot_sub(RHO, est):
        plt.subplot(2, 2, 1)
        CS = plt.imshow(
            RHO, cmap=plt.cm.jet, interpolation='spline16', origin='lower')
        plt.colorbar()

        plt.subplot(2, 2, 2)
        CS = plt.imshow(
            est, cmap=plt.cm.jet, interpolation='spline16', origin='lower')
        plt.colorbar()

        plt.subplot(2, 2, 3)
        CS = plt.imshow(
            est - RHO,
            cmap=plt.cm.jet,
            interpolation='spline16',
            origin='lower')
        plt.colorbar()
        plt.show()

    for molecule in molecule_list:

        O_coord = molecule.get_atom_pos()[0, :]

        RHO = plane_cut(rho_val, 2, O_coord[2], unitcell, grid, rmax=rmax)
        Xm, Ym = plane_cut(
            rho_val, 2, O_coord[2], unitcell, grid, rmax=rmax, return_mesh=True)

        Y = Ym * unitcell[1, 1] / grid[1]
        X = Xm * unitcell[0, 0] / grid[0]

        n_gauss = molecule.get_n_gauss()
        for atom in molecule.get_atom_list():
            n_gauss -= int(atom.core_electrons/2)

        est = fit_func(X, Y, O_coord[2], fit[:4*n_gauss], n_gauss)
        plot_sub(RHO, est)

        RHO = plane_cut(rho_val, 0, O_coord[0], unitcell, grid, rmax=rmax)
        Ym, Zm = plane_cut(
            rho_val, 0, O_coord[0], unitcell, grid, rmax=rmax, return_mesh=True)

        Y = Ym * unitcell[1, 1] / grid[1]
        Z = Zm * unitcell[2, 2] / grid[2]

        est = fit_func(O_coord[0], Y, Z, fit[:4*n_gauss], n_gauss)

        plot_sub(RHO, est)


def output_xyz(out_path, mol_list, center_pos, title = True):
    """ Write the fitting results into an .xyz file

    Parameters
    ----------
    out_path: string; path to write to
    mol_list: (,3) np.array; atomic coordinates
    center_pos: (,4) np.array; fitting parameters
    n_molecules: int; number of molecules
    """
    if title == True:
        title = 'Title'

    n_atoms = 0
    for mol in mol_list:
        n_atoms += len(mol.get_atom_pos())

    n_gauss = int(len(center_pos)/4)

    out_array = [[n_atoms + n_gauss, '', '', '']]
    out_array.append([title, '', '', ''])
    for mol in mol_list:
        for a in mol.get_atom_list():
            ap = a.get_coord() / AtoBohr
            out_array.append([a.get_label(), ap[0], ap[1], ap[2]])


    for i in range(n_gauss):
        out_array.append([
            'x', center_pos[i * 4] / AtoBohr, center_pos[i * 4 + 1] / AtoBohr,
            center_pos[i * 4 + 2] / AtoBohr
        ])

    out_df = pd.DataFrame(np.array(out_array))
    out_df.to_csv(out_path, sep=' ', index=None, header=None)


def n_gauss_3d(x, y, z, par, n):
    """ Sum of n 3d-Gaussians

    Parameters
    ----------
    x,y,z: floats/np.arrays coordinats at which function is evaluated
    par: list; fitting parameters ordered the following way:
            [mu_x,mu_y,mu_z,Width, repeat for multiple Gaussians]
    n: int; number of Gaussians

    Returns
    -------
    float/np.array of floats; functional values
    """

    result = 0
    for i in range(n):
        result += 2 * ((np.pi * par[i * 4 + 3])**(
            -3 / 2)) * np.exp(-((x - par[i * 4 + 0])**2 +
                                (y - par[i * 4 + 1])**2 +
                                (z - par[i * 4 + 2])**2) / par[i * 4 + 3])
    return result


def euclid_to_restricted(par, mol_list):
    """ Convert euclidean parameters to restricted parameters
    that are used by n_gauss_3d
    """

    gauss = np.array(par).reshape(-1,4)
    par = []
    start_index = 0

    for m, molecule in enumerate(mol_list):
        coord = molecule.get_atom_pos().reshape(1,3,3)

        oh1 = coord[0,1] - coord[0,0]
        oh1 = oh1/np.linalg.norm(oh1)
        oh2 = coord[0,2] - coord[0,0]
        oh2 = oh2/np.linalg.norm(oh2)

        origin = coord[0,0,:]
        loc_axes = local_axes(coord)

        # B1 radius
        par.append(np.dot(gauss[start_index,:3]-origin,oh1))
        # B1 width
        par.append(gauss[start_index,3])

        # B2 radius
        par.append(np.dot(gauss[start_index + 1,:3]-origin,oh2))
        # B2 width
        par.append(gauss[start_index + 1,3])

        # LP1 x,y, width
        par.append(np.dot(gauss[start_index + 2, :3] - origin, loc_axes[0, 6:9]))
        par.append(np.dot(gauss[start_index + 2, :3] - origin, loc_axes[0, 0:3]))
        par.append(gauss[start_index + 2, 3])
        # LP2 x,y, width
        par.append(np.dot(gauss[start_index + 3, :3] - origin, loc_axes[0, 6:9]))
        par.append(np.dot(gauss[start_index + 3, :3] - origin, loc_axes[0, 0:3]))
        par.append(gauss[start_index + 3, 3])

        start_index += molecule.get_n_gauss()

    return par

def restricted_to_euclid(par, mol_list):
    """ Convert restricted parameters
    to euclidean parameters that are used by n_gauss_3d
    Restricted parameters:
    [rb1, wb1, rb2, wb2, xlp1, zlp1, wlp1, xlp2, zlp2, wlp2]
    """

    gauss = np.array(par).reshape(-1,10)
    par_e = []

    for m, molecule in enumerate(mol_list):
        coord = molecule.get_atom_pos().reshape(1,3,3)

        oh1 = coord[0,1] - coord[0,0]
        oh1 = oh1/np.linalg.norm(oh1)
        oh2 = coord[0,2] - coord[0,0]
        oh2 = oh2/np.linalg.norm(oh2)

        origin = coord[0,0,:]
        loc_axes = local_axes(coord)

        par_e += (gauss[m,0] * oh1 + origin).tolist()
        par_e.append(gauss[m,1])
        par_e += (gauss[m,2] * oh2 + origin).tolist()
        par_e.append(gauss[m,3])

        z =  gauss[m, 5]
        x =  gauss[m, 4]

        par_e += (loc_axes[0, 0:3] * x + loc_axes[0, 6:9] * z + origin).tolist()
        par_e.append(gauss[m,6])

        z = gauss[m, 8]
        x = gauss[m, 7]

        par_e += (loc_axes[0, 0:3] * x + loc_axes[0, 6:9] * z + origin).tolist()
        par_e.append(gauss[m,9])

        if molecule.get_atom_list()[0].is_core_corrected():
            par_e += [0,0,0,0] # Lock_cores takes care of this

    return par_e


def colinear_cost(par, mol_list):
    """ Calculate the cost associated with deviations of Gaussian positions
    from OH bonds
    """

    par_ = restricted_to_euclid(euclid_to_restricted(par, mol_list), mol_list)

    cost = np.linalg.norm(np.array(par_) - np.array(par))

    if cost < 1e-3: cost = 0
    return cost


def quadrupole_moment(X, Y, Z, V, coord, rho, diagonal = False, verbose = False):
    """Calculates the quadrupole moment in Debye of a given charge distribution

    Parameters
    ----------
    X, Y, Z: np.array; Mesh arrays
    V: float; Volume of a grid cell
    coord: np.array; atomic coordinates, ordered like [O,H,H,O,H,...]
    par: [float]; Gaussian fitting parameters
    n: int; number of gaussians
    diagonal: boolean; Only compute diagonal elements
    verbose: boolean; print Ionic and Electronic contribution
    """

    elec_quadrupole = np.zeros([3,3])

    meshes = [X,Y,Z]

    ionic_quadrupole = np.zeros([3,3])

    charge = [6,1,1] * int(len(coord)/3)

    for i in range(3):
        for j in range(3):
            for a,c in zip(coord,charge):
                if i == j:
                    ionic_quadrupole[i,j] -= c * np.linalg.norm(a)**2
                ionic_quadrupole[i,j] += 3 * c * a[i]*a[j]

    for i in range(3):
        for j in range(i, 3):
            if i == j:
                if i == 2: continue # Determine last diagonal entry by trace cond.

                rsq = np.zeros_like(meshes[0])
                for k in range(3):
                    rsq += (meshes[k])**2
                elec_quadrupole[i,j] -= np.sum(rsq * rho * V)
            elif diagonal: # Only calculate diagonal elements
                continue
            elec_quadrupole[i,j] += np.sum(3 * meshes[i] * meshes[j]  * rho * V)

    #Fill lower triangle
    if not diagonal:
        for i in range(3):
            for j in range(i):
                elec_quadrupole[i,j] = elec_quadrupole[j,i]

    elec_quadrupole[2,2] =  - elec_quadrupole[0,0] - elec_quadrupole[1,1]

    if diagonal: return (ionic_quadrupole - elec_quadrupole).diagonal()
    else: return  (ionic_quadrupole - elec_quadrupole)

def dipole_moment(X, Y, Z, V, coord, rho, verbose = False):
    """Calculates the dipole moment in Debye of a given charge distribution

    Parameters
    ----------
    X, Y, Z: np.array; Mesh arrays
    V: float; Volume of a grid cell
    coord: np.array; atomic coordinates, ordered like [O,H,H,O,H,...]
    par: [float]; Gaussian fitting parameters
    n: int; number of Gaussians
    verbose: boolean; print Ionic and Electronic contribution
    """

    #TODO: implement possibility to choose which elements are fitted

    charge_com = np.array([ np.sum(mesh * rho * V) for mesh in [X,Y,Z]])

    coord = coord.reshape(-1,3,3)
    ionic_contrib = np.zeros(3)
    for a in coord:
        ionic_contrib += a[1] + a[2] + 6 * a[0]
    if verbose:
        print('Ionic {} [a.u.]'.format(ionic_contrib))
        print('Electronic {} [a.u.]'.format(charge_com))

    return (ionic_contrib - charge_com)/Dtoau

def gauss_overlap(par,n):
    """Calculate the overlap between n spherical Gaussians. Used as fitting
    regularization.

    Parameters
    ----------
    par: [float], gaussian parameters (position, radius)
    n: int, number of gaussians

    Returns
    -------
    overlap: float
    """

    overlap = 0
    gamma = 0
    for n1 in range(n):
        for n2 in range(n1,n):
            gamma = 1/(par[n1 * 4 + 3] + par[n2 * 4 + 3])
            rsq = 0
            for i in range(3):
                rsq += (par[n1 * 4 + i] - par[n2 * 4 + i]) ** 2
            overlap += gamma ** (3/2) * np.exp(- rsq * gamma)
    return overlap
