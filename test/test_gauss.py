import sys
import os
import pytest
import gauss_fit.fitting.gauss_charge as gc
import gauss_fit.fitting.gauss_util as gu
import numpy as np
from gauss_fit.molecule_classes import H2O
from gauss_fit.atom import Atom
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
coord = gu.get_atom_pos('./coord_test.csv','A')

def test_four_center():
    ''' Test gaussian fitting on a water molecule with four Gaussians
    '''
    gc.get_data('0.RHO')

    O = Atom(8, coord[0, :], 0, n_gauss = 2)
    H1 = Atom(1, coord[1, :], 0, n_gauss = 1)
    H2 = Atom(1, coord[2, :], 0, n_gauss = 1)
    h2o = H2O(O,H1,H2)
    mol_list = [h2o]


    assert np.allclose(gc.check_norm(),8)

    fit, rmse, rmax = gc.fit_poly(mol_list, rmax = -1, box_buffer = 2,
        plot_out=False, write_xyz=False)
    assert rmse < 0.03
    assert np.allclose(np.genfromtxt('four_center_ref.gauss', delimiter =','),
           np.genfromtxt('RHO.gauss', delimiter =','))

def test_constraints():
    ''' Test gaussian fitting on a water molecule with four Gaussians and
        symmetry constraints
    '''

    O = Atom(8, coord[0, :], 2, n_gauss = 2 )
    H1 = Atom(1, coord[1, :], 0, n_gauss = 1)
    H2 = Atom(1, coord[2, :], 0, n_gauss = 1 )
    h2o = H2O(O,H1,H2)

    fix_pos = {0:[None,None,0,None], 2:[0,None,None,None]}

    # Check if invalid input for symmetry constraints raises error
    exception_occured = False
    try:
        fix_sym = {(0,1):[-1,1,1], (2,3):[1,1,-1,1]}
        h2o.set_symmetry(fix_sym)
    except Exception as exc:
        exception_occured = True
    assert exception_occured


    fix_sym = {(0,1):[-1,1,1,1], (2,3):[1,1,-1,1]}
    h2o.set_fix_pos(fix_pos)
    h2o.set_symmetry(fix_sym)

    mol_list = [h2o]


    assert np.allclose(gc.check_norm(),8)

    fit, rmse, rmax = gc.fit_poly(mol_list, rmax = -1, use_sym=True, box_buffer = 2,
        plot_out=False, write_xyz=False)
    assert rmse < 0.03

    assert np.allclose(np.genfromtxt('constraints_ref.gauss', delimiter =','),
           np.genfromtxt('RHO.gauss', delimiter =','))

def test_five_center():
    ''' Test gaussian fitting on a water molecule with four Gaussians
    '''
    gc.get_data('0.RHO')

    O = Atom(8, coord[0, :], 2, 'O', './O.core', n_gauss = 3)
    H1 = Atom(1, coord[1, :], 0, 'H', n_gauss = 1)
    H2 = Atom(1, coord[2, :], 0, 'H', n_gauss = 1)
    h2o = H2O(O,H1,H2)
    mol_list = [h2o]
    gc.add_core_density(mol_list)
    h2o.lock_cores()
    fit, rmse, rmax = gc.fit_poly(mol_list, rmax = -1, box_buffer = 2,
        plot_out=False, write_xyz=False,use_sym = True)

    assert rmse < 0.03

    assert np.allclose(np.genfromtxt('five_center_ref.gauss', delimiter =','),
           np.genfromtxt('RHO.gauss', delimiter =','))
