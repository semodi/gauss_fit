import numpy as np
import gauss_fit.atom as atom
AtoBohr = 1.88972599

class Molecule():

    __n_atoms = 0

    def __init__(self, atom_list = [], n_gauss = 0):
        self.atom_pos = np.zeros([len(atom_list),3])
        for a, atom in enumerate(atom_list):
            self.atom_pos[a,:] = atom.get_coord()

        self.n_gauss = n_gauss
        self.__n_atoms = len(atom_list)
        self.atom_list = atom_list
        self.sym_dict = {}
        self.fix_dict = {}

    def __str__(self):
        str = 'Molecule fit with {} Gaussians \nAtoms: \n'.format(self.n_gauss)
        for atom in self.atom_list:
            str += '({},{}) ::: {} ::: {}, {} ::: {} \n'.format(atom.label,atom.Z,
            np.array_str(atom.get_coord(),precision = 3),atom.core_electrons,atom.core_file,atom.n_gauss)
        return str

    def get_atom_pos(self):
        return self.atom_pos

    def set_atom_pos(self, atom_pos):
        self.atom_pos = atom_pos
        self.__n_atoms = len(atom_pos)

    def get_n_atoms(self):
        return self.__n_atoms

    def get_n_gauss(self):
        return self.n_gauss

    def set_n_gauss(self, n_gauss):
        self.n_gauss = n_gauss

    def get_init(self):
        """ Get initial guess for fitting parameters.
        If not implemented by child class place all Gaussians onto
        cores. It is recommended to implement get_init() seperately for
        every molecule for speedup.
        """

        init = []
        n = 0
        for atom in self.atom_list:
            if atom.n_gauss == -1:
                raise Exception("Number of Gaussians on atom {} not defined.".format(atom.label) +
                " Either set number of Gaussians or implement a routine that" +
                " returns the initial fitting parameters for this Molecule.")
            else:
                n +=  atom.n_gauss
        if n != self.n_gauss:
            raise Exception("Sum of Gaussians on atoms does not equal" +
            "number of Gaussians to fit Molecule")
        else:
            for atom in self.atom_list:
                if atom.core_corrected:
                    core_gauss = int(atom.core_electrons/2)
                else:
                    core_gauss = 0

                for i in range(atom.n_gauss - core_gauss):
                    init += [*atom.get_coord().tolist(),1.0]
                for i in range(core_gauss):
                    init += [*atom.get_coord().tolist(),atom.core_width[i]]
        return init

    def get_atom_list(self):
        return self.atom_list

    def set_fix_pos(self,fix_dict):
        """ use a dictionary to fix gaussian positions
            Example: { 2: [0,0,0,None] }, fixes the third gaussian
            to (0,0,0) leaving the width unconstrained
        """
        if not isinstance(fix_dict,dict):
            raise Exception('Not a dictionary')

        for key in fix_dict:
            if len(fix_dict[key]) != 4:
                raise Exception('Constraint type not understood, ' +
                'make sure to specify 4 parameters')
        for index, item in fix_dict.items():
            self.fix_dict[index] = item

    def add_mirror_plane(self, ):
        """
        """
        pass
    def set_symmetry(self,sym_dict):
        """ use a dictionary to fix symmetry relations between gaussians
            Example: { (2,3): [None,None,None,1]}, fixes the third and
            fourth gaussians to have the same width
        """
        if not isinstance(sym_dict,dict):
            raise Exception('Not a dictionary')

        for key in sym_dict:
            if len(sym_dict[key]) != 4:
                raise Exception('Constraint type not understood, ' +
                'make sure to specify 4 parameters')

        for index, item in sym_dict.items():
            self.sym_dict[index] = item

    def use_constraints(self,par):
        """ use the rules on par that are specified in fix_dict and sym_dict
        """
        if len(self.fix_dict) != 0:

            for where, what in self.fix_dict.items():
                for i,w in enumerate(what):
                    if w != None:
                        par[where*4+i] = w
        if len(self.sym_dict) != 0:

            for where, what in self.sym_dict.items():
                for i,w in enumerate(what):
                    if w != None:
                        par[where[1]*4+i] = par[where[0]*4+i]*w
        return par

    def lock_cores(self):
        """ Set the constraints so that the core_electron Gaussians
        are not altered during fit.
        """
        # Count number of gaussians that describe core density
        # to access them from the back of the parameter list
        gauss_pointer = 1

        for atom in self.atom_list:
            if atom.is_core_corrected():
                # Try if multiple Gaussians per core
                # if not, for-loop will raise TypeError
                try:
                    for w in atom.core_width:
                        self.fix_dict[self.n_gauss - gauss_pointer] = \
                            [*(atom.get_coord()).tolist(),w]
                        gauss_pointer +=1
                except TypeError:
                    if not isinstance(atom.core_width,float):
                        print(atom.core_width)
                        raise Exception("atom.core_width has wrong data type!" +
                        " Expected: list of floats or float;" +
                        " Given: {}".format(type(atom.core_width)))
                    else:
                        self.fix_dict[self.n_gauss - gauss_pointer] = \
                            [*(atom.get_coord().tolist()),atom.core_width]
                        gauss_pointer +=1


def from_fdf(file_path):
    """Import molecule geometry from siesta .fdf file
    The file has to contain a ChemicalSpeciesLabel block
    and a AtomicCoordinatesAndAtomicSpecies block where the
    units used are Angstrom
    """

    #Chemical Species Labels
    with open(file_path,'r') as fdf_file:
        content = fdf_file.read()
    start_species = content.find('%block ChemicalSpeciesLabel')
    end_species = content.find('%endblock ChemicalSpeciesLabel')
    species = content[start_species:end_species].splitlines()
    atoms = {}

    #Determine the Molecule geometry
    for s in species[1:]:
        atom = s.split()
        atoms[int(atom[0])]= [int(atom[1]),atom[2]]
    start_coord = content.find("%block AtomicCoordinatesAndAtomicSpecies")
    end_coord = content.find("%endblock AtomicCoordinatesAndAtomicSpecies")
    coord_list = content[start_coord:end_coord].splitlines()
    coords = np.zeros([len(coord_list)-1,4])

    #Build Molecule instance
    for i, c in enumerate(coord_list[1:]):
        coords[i,:] = np.array([float(k) for k in c.split()])
    atom_list = []
    charge = 0
    for c in coords:
        atom_list.append(atom.Atom(atoms[c[3]][0],c[:3]*AtoBohr,0, label = atoms[c[3]][1],n_gauss = int(atoms[c[3]][0]/2)))
        charge += atoms[c[3]][0]

    return Molecule(atom_list,int(charge/2))
