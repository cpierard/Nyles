"""
The objective of this module is to average and integrate the variables of
interest, and compute the terms in the kinetic and potential energy balance
equations, specifically for the forced plume
experiments.
The main idea is to perform this operations without merging the subdmains that
are created from a simulation with several cores, in order to save memory.
In development. To do:
- Compute APE.
- Volume, buoyancy and momentum balances.
- ...
"""

from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import pickle

class plume:

    def __init__(self, folder_path, experiment_name):
        self.path = folder_path
        self.name = experiment_name
        file = self.path + 'param.pkl'
        try:
            with open(file, 'rb') as f:
                self.params = pickle.load(f)
        except:
            print(f'There is no {file} file in folder.')

    def read_vars(self, vars, file):
        """
        Read a list of variables from the paramters of the simulation
        'NN' for Brunt-vaisala squared.
        'KE' for Kinetic energy.
        """
        fields = {}
        for var in vars:
            with Dataset(file, 'r') as nc:
                if var in nc.variables:
                    fields[var] = nc[var][:].data
                elif var == 'NN':
                    fields[var] = self.brunt_vaisalla(file)
                elif var == 'KE':
                    fields[var] = self.kinetic_energy(file)
                elif var == 'test':
                    fields[var] = self.test(file)
        return fields

    def brunt_vaisalla(self, file):
        f = self.read_vars(['b','z'], file)
        NN = - np.diff(f['b'], axis=1)/np.diff(f['z'])[0]
        return NN

    def kinetic_energy(self, file):
        f = self.read_vars(['u','v','w'], file)
        KE = (f['u']**2 + f['v']**2 + f['w']**2)/2
        return KE

    def test(self, file):
        f = self.read_vars(['u'], file)
        test_field = np.zeros_like(f['u']) # to verify the averaging
        return test_field


    def disk_average(self, var):
        """
        Computes the average in a horizontal disk, neglecting the sponge layers
        in the borders, and not merging subfiles. Only works for horizontal
        subdomains (so far).
        var a variable in string format. It can be:
            - a var in netCDF file. e.g. 'b', 'u', 'w', etc.
            - 'NN' for squared Brunt-Vaisala freq.
            - 'KE' for kinetic energy.
            - The list will increase as it increases the number of functions.
        """
        npx = self.params['npx']
        npy = self.params['npy']
        npz = self.params['npz']
        number_domains = npx*npy*npz # so far only works for number_domains < 100
        Lx = self.params['Lx']
        Ly = self.params['Ly']
        Lz = self.params['Lz']
        x0 = Lx/2 # center point in the x domain.
        y0 = Ly/2 # center point in the y domain.
        nz = self.params['nz']

        if var == 'NN': # maybe interpolate is field...
            nz = nz - 1

        t = self.read_vars('t', f'{self.path}{self.name}_00_hist.nc')['t']
        n_time = t.shape[0]

        r_max = 1700 # define this radius from nudging.

        means = np.zeros((n_time, nz))

        for i in range(number_domains):
            subfile = f'{self.path}{self.name}_{i:02d}_hist.nc'

            fields = self.read_vars([var, 'x', 'y'], subfile)

            XX, YY = np.meshgrid(fields['x'], fields['y'])
            r = np.sqrt((XX - x0)**2 + (YY - y0)**2)
            mask = ma.masked_outside(r, 0, r_max)

            for t in range(n_time):
                for z_lvl in range(nz):
                    field_new = ma.masked_array(fields[var][t, z_lvl, :, :], mask.mask)
                    means[t, z_lvl] += field_new.mean()

        means = means/number_domains
        return means
