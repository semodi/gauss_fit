import numpy as np
from .molecule import Molecule
import sys
import gauss_fit.fitting.mono_fit as mf

class H2O(Molecule):

    def __init__(self,O,H1,H2):

        atom_list = [O,H1,H2]

        if O.core_corrected:
            n_gauss = 5
        else:
            n_gauss = 4

        super().__init__(atom_list, n_gauss)

    def get_init(self):

        init = mf.get_gauss_euclid(self.atom_pos).flatten().tolist()
        if self.n_gauss == 5:
            o = self.atom_pos[0,:]
            init += [*(o.tolist()),self.atom_list[0].core_width[0]]

        return init


class O2(Molecule):

    def __init__(self,O_1,O_2):

        atom_list = [O_1,O_2]
        if O_1.core_corrected and O_2.core_corrected:
            n_gauss = 8
        else:
            n_gauss = 6

        super().__init__(atom_list, n_gauss)

    def get_init(self):
        o1 = self.atom_pos[0,:]
        o2 = self.atom_pos[1,:]
        do = (o2-o1)/np.linalg.norm(o2-o1)

        loc_z = do + np.array([0,0.12,-0.07])
        loc_z = loc_z - gs_project(loc_z,do)
        loc_z = loc_z / np.linalg.norm(loc_z)

        loc_y = do + np.array([0,0.06,0.011])
        loc_y = loc_z - gs_project(loc_y,do) - gs_project(loc_y,loc_z)
        loc_y = loc_y / np.linalg.norm(loc_y)

        init = []

        #Bonds
        init += [*(((o1+(o2-o1)*.1)/2).tolist()),1.54]
        init += [*(((o2+(o1-o2)*.1)/2).tolist()),1.54]

        #Lonely pairs
        init += [*((o1 + loc_z * 0.05).tolist()),0.75]
        init += [*((o1 - loc_z * 0.05).tolist()),0.75]
        init += [*((o2 + loc_y * 0.05).tolist()),0.75]
        init += [*((o2 - loc_y * 0.05).tolist()),0.75]

        if self.n_gauss == 8:
            init += [*(o1.tolist()),self.atom_list[0].core_width[0]]
            init += [*(o2.tolist()),self.atom_list[0].core_width[0]]

        return init



def gs_project(a,b): # projection of a onto b
    return np.dot(a,b)/np.dot(b,b)*b
