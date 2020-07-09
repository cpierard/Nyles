"""
The objective of this module is to average and integrate the variables of
interest and compute the terms in the kinetic and potential energy balance
equations, specifically for the forced plume experiment.
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
from iosubdomains import Variable

class plume:

    def __init__(self, folder_path, experiment_name):
        self.path = folder_path
        self.name = experiment_name
        self.template = folder_path + experiment_name + '_%02i_hist.nc'
        file = self.path + 'param.pkl'
        try:
            with open(file, 'rb') as f:
                self.params = pickle.load(f)
        except:
            print(f'There is no {file} file in folder.')

    def read_vars(self, vars):
        """
        Read a list of variables from the paramters of the simulation

        'NN' for Brunt-vaisala squared.
        'KE' for Kinetic energy.
        """
        fields = {}
        for var in vars:
            try:
                fields[var] = Variable(self.template, var)[:]
            except:
                if var == 'NN':
                    fields[var] = self.brunt_vaisalla()
                elif var == 'KE':
                    fields[var] = self.kinetic_energy()
                elif var == 'test':
                    fields[var] = self.test()

        # for var in vars:
        #     with Dataset(file, 'r') as nc:
        #         if var in nc.variables:
        #             fields[var] = nc[var][:].data
        #         elif var == 'NN':
        #             fields[var] = self.brunt_vaisalla(file)
        #         elif var == 'KE':
        #             fields[var] = self.kinetic_energy(file)
        #         elif var == 'test':
        #             fields[var] = self.test(file)
        return fields

    def brunt_vaisalla(self):
        f = self.read_vars(['b','z'])
        NN = - np.diff(f['b'], axis=1)/np.diff(f['z'])[0]
        return NN

    def kinetic_energy(self):
        f = self.read_vars(['u','v','w'])
        u = self.velocity_interpolation(f['u'], axis=3)
        v = self.velocity_interpolation(f['v'], axis=2)
        w = self.velocity_interpolation(f['w'], axis=1)

        KE = (f['u']**2 + f['v']**2 + f['w']**2)/2
        return KE

    def test(self):
        f = self.read_vars(['u'], file)
        test_field = np.zeros_like(f['u']) # to verify the averaging
        return test_field

    def velocity_interpolation(self, a, axis=-1):
        """
        velocity_interpolation(a, axis=-1)

        Linear interpolation for velocity in a staggered type C grid.
        Z-convention (nz, ny, nx)

        Parameters
        ----------
        a : array_like
            Input array
        axis : int, optional
            The axis along which the difference is taken, default is the
            last axis.

        Returns
        -------
        U_interp : ndarray
            Array with same dimension as input.
        """
        nd = len(a.shape)

        # adding one extra dimension to field at the lower boundary with
        # zeros.
        a_shape = list(a.shape)
        a_shape[axis] = a.shape[axis] + 1
        a_shape = tuple(a_shape)
        slice0 = [slice(None)] * nd
        slice0[axis] = slice(1, None)
        slice0 = tuple(slice0)
        a_prim = np.zeros(a_shape)
        a_prim[slice0] = a

        # doing the interpolation
        slice1 = [slice(None)] * nd
        slice2 = [slice(None)] * nd
        slice1[axis] = slice(None, -1)
        slice2[axis] = slice(1, None)
        slice1 = tuple(slice1)
        slice2 = tuple(slice2)

        a_interp = (a_prim[slice1] + a_prim[slice2])/2
        return a_interp

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
        number_domains = npx*npy*npz # so far only works for number_domains<100
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
