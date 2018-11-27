import numpy as np
import scipy.optimize as opt

class Atom():

    def __init__(self, Z, coord, core_electrons,label = '', core_file = None, n_gauss = -1):
        self.coord = coord
        self.Z = Z
        self.label = label
        self.core_width = 0
        self.core_file = core_file
        self.core_electrons = core_electrons
        self.n_gauss = n_gauss

        # If file with core_density exists, correct charge density
        if core_file != None and core_electrons > 0:
            self.core_corrected = True
            self.fit_core()
            print('Warning: core_file included, will try to fit to core ' +
            'charge density.\nMake sure to run gauss_charge.add_core_density()')
        else:
            self.core_corrected = False

    def set_n_gauss(self, n):
        self.n_gauss = n

    def set_core_electrons(self, n):
        self.core_electrons = n
        if n > 0 and self.core_file != None:
            self.core_corrected = True
            self.fit_core()
            print('Warning: core_file included, will try to fit to core ' +
            'charge density.\nMake sure to run gauss_charge.add_core_density()')

    def set_core_file(self, path):
        self.core_file = path
        if self.core_electrons > 0 and self.core_file != None:
            self.core_corrected = True
            self.fit_core()
            print('Warning: core_file included, will try to fit to core ' +
            'charge density.\nMake sure to run gauss_charge.add_core_density()')

    def get_label(self):
        if self.label == '':
            raise Exception("No label attached")
        else:
            return self.label

    def is_core_corrected(self):
        return self.core_corrected

    def get_coord(self):
        return self.coord

    def fit_core(self, cutoff = 600):
        data = np.genfromtxt(self.core_file)
        r = data[:cutoff, 0]
        core_rho = data[:cutoff, 3]
        # Function to fit s-like core
        def cost(par):
            result = self.core_electrons * (par / np.pi)**(3/2)*np.exp(-par*r**2)
            return (result-core_rho).flatten()

        # Fit function to core density
        fit = opt.least_squares(cost,0.2)
        self.core_width = (1/np.array(fit.x)).tolist() * int(self.core_electrons/2)

    def core_density(self,x,y,z):
        #It is assumed that the Gaussian centers align with the core
        coord = self.coord
        if self.core_corrected:
            result = 0
            for i in range(int(self.core_electrons/2)):
                result += 2*(self.core_width[i]*np.pi)**(-3/2)*np.exp(-((x-coord[0])**2 +
                    (y-coord[1])**2+(z-coord[2])**2)/self.core_width[i])
            return result
        else:
            return np.zeros_like(x)
