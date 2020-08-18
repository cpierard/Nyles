"""
memory efficient... but extremely slow.

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
        self.time = Variable(self.template, 't')[:]
        try:
            with open(file, 'rb') as f:
                self.params = pickle.load(f)
        except:
            print(f'There is no {file} file in folder.')

        self.params['dx'] = (self.params['Lx']/self.params['global_nx'])
        self.params['dy'] = (self.params['Ly']/self.params['global_ny'])
        self.params['dz'] = (self.params['Lz']/self.params['global_nz']) # just maintain a grid with the same dx in the three directions

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

           if var in ['u','v','w']:
                # just maintain a grid with the same dx in the three directions
               fields[var] = fields[var]/self.params['dx']

       return fields

    def brunt_vaisalla(self):
        f = self.read_vars(['b','z'])
        NN = - np.diff(f['b'], axis=1)/np.diff(f['z'])[0]
        return NN

    def kinetic_energy(self):
        f = self.read_vars(['u','v','w'])
        u = velocity_interpolation(f['u'], axis=3)
        v = velocity_interpolation(f['v'], axis=2)
        w = velocity_interpolation(f['w'], axis=1)

        KE = (f['u']**2 + f['v']**2 + f['w']**2)/2
        return KE

    def test(self):
        f = self.read_vars(['u'], file)
        test_field = np.zeros_like(f['u']) # to verify the averaging
        return test_field

    def disk_average(self, var, r_lim):
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
        # change the mask for the one in Flux
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

        t = self.read_vars('t')['t']
        n_time = t.shape[0]

        r_max = r_lim #0.45 # as in forced_plume_nudging.py
        z_max = 0.95

        means = np.zeros((n_time, nz))

        fields = self.read_vars([var, 'x', 'y'])

        if var in ['u', 'v', 'w']:
            axis_vel = {'u': 3, 'v': 2, 'w':1}
            fields[var] = velocity_interpolation(fields[var], axis=axis_vel[var])

        XX, YY = np.meshgrid(fields['x']/Lx - 0.5,
                             fields['y']/Ly - 0.5)

        r = np.sqrt(XX**2 + YY**2)
        mask = ma.masked_outside(r, 0, r_max)
        #mask_2 = ma.masked_outside(ZZ, 0, z_max)

        for t in range(n_time):
            for z_lvl in range(nz):
                field_new = ma.masked_array(fields[var][t, z_lvl, :, :], mask.mask)
                means[t, z_lvl] = field_new.mean()

        #means = means/number_domains
        return means


    def Flux(self, flux, r_lim, z_lim):
        """
        Computes the the mass, momentum and buoyancy fluxes in a cildrical
        control volume, defined by the nudging (the sponge layer) limits

        flux - a string idicating the tipe of flux: "mass", "momentum" or
         "buoyancy".
        """
        if flux == 'mass':
            set_integrand = lambda x: x

        elif flux == 'momentum':
            set_integrand = lambda x: x**2

        elif flux == 'buoyancy':
            b = self.read_vars('b')['b']
            set_integrand = lambda x: x*b

        npx = self.params['npx']
        Lx = self.params['Lx']
        Ly = self.params['Ly']
        Lz = self.params['Lz']
        nz = self.params['nz']

        dx = Lx/npx
        t = self.read_vars('t')['t']
        n_time = t.shape[0]

        r_max = r_lim # as in forced_plume_nudging.py
        z_max = z_lim
        new_nz = int(nz*z_lim)

        flux = np.zeros(n_time)

        fields = self.read_vars(['w', 'x', 'y', 'z'])
        w = velocity_interpolation(fields['w'], axis=1)

        XX, YY = np.meshgrid(fields['x']/Lx - 0.5,
                             fields['y']/Ly - 0.5)

        r = np.sqrt(XX**2 + YY**2)
        mask_1 = ma.masked_outside(r, 0, r_max)
        #mask_2 = ma.masked_outside(ZZ, 0, z_max)

        # defining integrand
        integrand = set_integrand(w)

        for t in range(n_time):
            aux = np.zeros(new_nz)
            for z_i in range(new_nz):
                field_new = ma.masked_array(integrand[t, z_i], mask_1.mask)
                aux[z_i] = field_new.sum()

            flux[t] = aux.sum()

        return flux

    def Flux_levels(self, flux, r_lim=0.45):
        """
        Computes the the mass, momentum and buoyancy fluxes in a cildrical
        control volume, defined by the nudging (the sponge layer) limits

        flux - a string idicating the tipe of flux: "mass", "momentum" or
         "buoyancy".
        """
        if flux == 'mass':
            set_integrand = lambda x: x

        elif flux == 'momentum':
            set_integrand = lambda x: x**2

        elif flux == 'buoyancy':
            b = self.read_vars('b')['b']
            set_integrand = lambda x: x*b

        npx = self.params['npx']
        Lx = self.params['Lx']
        Ly = self.params['Ly']
        Lz = self.params['Lz']
        nz = self.params['nz']

        dx = Lx/npx
        t = self.read_vars('t')['t']
        n_time = t.shape[0]

        r_max = r_lim # as in forced_plume_nudging.py
        z_max = 0.95


        flux = np.zeros((n_time, nz))

        fields = self.read_vars(['w', 'x', 'y', 'z'])
        w = velocity_interpolation(fields['w'], axis=1)

        XX, YY = np.meshgrid(fields['x']/Lx - 0.5,
                             fields['y']/Ly - 0.5)

        r = np.sqrt(XX**2 + YY**2)
        mask_1 = ma.masked_outside(r, 0, r_max)
        #mask_2 = ma.masked_outside(ZZ, 0, z_max)

        # defining integrand
        integrand = set_integrand(w)

        for t in range(n_time):
            for z_i in range(nz):
                field_new = ma.masked_array(integrand[t, z_i], mask_1.mask)
                flux[t, z_i] = field_new.sum()

        return flux

    def Budget(self, var, r_lim, z_lim):
        """

        """
        npx = self.params['npx']
        Lx = self.params['Lx']
        Ly = self.params['Ly']
        Lz = self.params['Lz']
        nz = self.params['nz']

        dx = Lx/npx
        t = self.read_vars('t')['t']
        n_time = t.shape[0]

        r_max = r_lim # as in forced_plume_nudging.py
        z_max = z_lim
        new_nz = int(nz*z_lim)

        budget = np.zeros(n_time)
        # ineficient to read all velocities, but no time to be efficient here
        fields = self.read_vars([var, 'w', 'u', 'v', 'x', 'y'])
        w = velocity_interpolation(fields['w'], axis=1)
        v = velocity_interpolation(fields['v'], axis=2)
        u = velocity_interpolation(fields['u'], axis=3)

        X, Y = np.meshgrid(fields['x']/Lx - 0.5,
                             fields['y']/Ly - 0.5)
        r = np.sqrt(X**2 + Y**2)

        #mask_1 = ma.masked_outside(r, r_max - 0.01, r_max)
        mask = ma.masked_outside(r, 0, r_max)
        m = mask.mask*1
        mask_ring = np.roll(m, -1, axis=0) + np.roll(m, 1, axis=0)
        mask_ring += np.roll(m, -1, axis=1) + np.roll(m, 1, axis=1)
        mask_ring -= 4*m

        for t in range(n_time):
            sides = 0
            for z_i in range(new_nz-1):
                f = fields[var][t,z_i]
                rad_proy = (u[t,z_i]*X + v[t,z_i]*Y)/r
                #aux = ma.masked_array(f*rad_proy, mask_1.mask)
                aux = ma.masked_array(f*rad_proy, mask_ring>=0)
                sides += aux.mean()

            lid = ma.masked_array(f*w[t, new_nz], mask.mask)
            budget[t] = sides + lid.mean()


        return budget

    def Budget_2(self, var, r_lim, z_lim):
        """

        """
        npx = self.params['npx']
        Lx = self.params['Lx']
        Ly = self.params['Ly']
        Lz = self.params['Lz']
        nz = self.params['nz']

        dx = Lx/npx

        t = self.read_vars('t')['t']
        n_time = t.shape[0]

        r_max = r_lim # as in forced_plume_nudging.py
        z_max = z_lim
        new_nz = int(nz*z_lim)

        budget = np.zeros(n_time)
        # ineficient to read all velocities, but no time to be efficient here
        fields = self.read_vars(['x', 'y'])

        X, Y = np.meshgrid(fields['x']/Lx - 0.5,
                             fields['y']/Ly - 0.5)
        r = np.sqrt(X**2 + Y**2)

        #mask_1 = ma.masked_outside(r, r_max - 0.01, r_max)
        mask = ma.masked_outside(r, 0, r_max)
        m = mask.mask*1
        mask_ring = np.roll(m, -1, axis=0) + np.roll(m, 1, axis=0)
        mask_ring += np.roll(m, -1, axis=1) + np.roll(m, 1, axis=1)
        mask_ring -= 4*m

        for t_i in range(n_time):
            sides = 0
            for z_i in range(new_nz-1):
                u = Variable(self.template, 'u')[t_i, z_i]/self.params['dx']
                u = velocity_interpolation(u, axis=1)
                v = Variable(self.template, 'v')[t_i, z_i]/self.params['dy']
                v = velocity_interpolation(v, axis=0)
                f = Variable(self.template, var)[t_i, z_i]

                rad_proy = (u*X + v*Y)/r
                aux = ma.masked_array(f*rad_proy, mask_ring>=0)
                sides += aux.mean()

            w = Variable(self.template, 'w')[t_i]/self.params['dz']
            w = velocity_interpolation(w, axis=0)

            lid = ma.masked_array(f*w[new_nz], mask.mask)
            budget[t_i] = sides + lid.mean()

        return budget

    def KE_Budget(self, r_lim, z_lim, t0):
        """

        """
        npx = self.params['npx']
        Lx = self.params['Lx']
        Ly = self.params['Ly']
        Lz = self.params['Lz']
        nz = self.params['nz']

        dx = Lx/npx

        t = Variable(self.template, 't') #self.read_vars('t')['t']
        n_time = t.shape[0]

        r_max = r_lim # as in forced_plume_nudging.py
        z_max = z_lim
        new_nz = int(nz*z_lim)

        budget = np.zeros(n_time)
        # ineficient to read all velocities, but no time to be efficient here
        x = Variable(self.template, 'x')
        y = Variable(self.template, 'y')

        #fields = self.read_vars(['x', 'y'])

        X, Y = np.meshgrid(x[:]/Lx - 0.5,
                             y[:]/Ly - 0.5)
        r = np.sqrt(X**2 + Y**2)

        #mask_1 = ma.masked_outside(r, r_max - 0.01, r_max)
        mask = ma.masked_outside(r, 0, r_max)
        m = mask.mask*1
        mask_ring = np.roll(m, -1, axis=0) + np.roll(m, 1, axis=0)
        mask_ring += np.roll(m, -1, axis=1) + np.roll(m, 1, axis=1)
        mask_ring -= 4*m

        w = Variable(self.template, 'w')
        u = Variable(self.template, 'u')
        v = Variable(self.template, 'v')
        for t_i in range(n_time):
            sides = 0
            w_i = w[t_i]/self.params['dz']
            w_i = velocity_interpolation(w_i, axis=0)
            u_i = u[t_i]/self.params['dx']
            u_i = velocity_interpolation(u_i, axis=2)
            v_i = v[t_i]/self.params['dy']
            v_i = velocity_interpolation(v_i, axis=1)

            for z_i in range(new_nz-1):
                #u_i = u[t_i, z_i]/self.params['dx']
                #u_i = velocity_interpolation(u_i, axis=1)
                #v_i = v[t_i, z_i]/self.params['dy']
                #v_i = velocity_interpolation(v_i, axis=0)

                #KE = (u_i**2 + v_i**2 + w_i[new_nz]**2)/2
                KE = (u_i[new_nz]**2 + v_i[new_nz]**2 + w_i[new_nz]**2)/2

                rad_proy = (u_i*X + v_i*Y)/r
                aux = ma.masked_array(KE*rad_proy, mask_ring>=0)
                sides += aux.mean()

            lid = ma.masked_array(KE*w[new_nz], mask.mask)
            budget[t_i] = sides + lid.mean()

        return budget

    def KE_volume(self, r_lim, z_lim):
        """

        """
        npx = self.params['npx']
        Lx = self.params['Lx']
        Ly = self.params['Ly']
        Lz = self.params['Lz']
        nz = self.params['nz']

        t = self.read_vars('t')['t']
        n_time = t.shape[0]

        r_max = r_lim # as in forced_plume_nudging.py
        z_max = z_lim
        new_nz = int(nz*z_lim)

        budget = np.zeros(n_time)
        fields = self.read_vars(['x', 'y'])

        X, Y = np.meshgrid(fields['x']/Lx - 0.5,
                             fields['y']/Ly - 0.5)
        r = np.sqrt(X**2 + Y**2)

        mask = ma.masked_outside(r, 0, r_max)

        for t_i in range(n_time):
            aux = np.zeros(new_nz)
            w = Variable(self.template, 'w')[t_i]/self.params['dz']
            w = velocity_interpolation(w, axis=0)
            for z_i in range(new_nz):
                u = Variable(self.template, 'u')[t_i, z_i]/self.params['dx']
                u = velocity_interpolation(u, axis=1)
                v = Variable(self.template, 'v')[t_i, z_i]/self.params['dy']
                v = velocity_interpolation(v, axis=0)
                KE = (u**2 + v**2 + w[new_nz]**2)/2
                field_new = ma.masked_array(KE, mask.mask)
                aux[z_i] = field_new.mean()

            budget[t_i] = aux.mean()

        return budget

    def APE_volume(self, r_lim, z_lim):
        """

        """
        npx = self.params['npx']
        Lx = self.params['Lx']
        Ly = self.params['Ly']
        Lz = self.params['Lz']
        nz = self.params['nz']
        dx = Lx/npx
        t = self.read_vars('t')['t']
        n_time = t.shape[0]
        r_max = r_lim # as in forced_plume_nudging.py
        z_max = z_lim
        new_nz = int(nz*z_lim)
        budget = np.zeros(n_time)

        br = Variable(self.template, 'b')[0,:,0,0]
        NN = (np.diff(br)/self.params['dz'])[0]
        fields = self.read_vars(['x', 'y'])

        XX, YY = np.meshgrid(fields['x']/Lx - 0.5,
                             fields['y']/Ly - 0.5)
        r = np.sqrt(XX**2 + YY**2)
        mask = ma.masked_outside(r, 0, r_max)

        for t_i in range(n_time):
            aux = np.zeros(new_nz)
            for z_i in range(new_nz):
                b = Variable(self.template, 'b')[t_i, z_i]
                arg = (b - br[z_i])**2/(2*NN)
                field_new = ma.masked_array(arg, mask.mask)
                aux[z_i] = field_new.mean()

            budget[t_i] = aux.mean()

        return budget

    def Phi_z(self, r_lim):
        """

        """
        npx = self.params['npx']
        Lx = self.params['Lx']
        Ly = self.params['Ly']
        Lz = self.params['Lz']
        nz = self.params['nz']
        dx = Lx/npx
        t = self.read_vars('t')['t']
        n_time = t.shape[0]
        r_max = r_lim # as in forced_plume_nudging.py
        budget = np.zeros(n_time)

        br = Variable(self.template, 'b')[0,:,0,0]
        NN = -(np.diff(br)/self.params['dz'])[0]
        fields = self.read_vars(['x', 'y'])
        XX, YY = np.meshgrid(fields['x']/Lx - 0.5,
                             fields['y']/Ly - 0.5)
        r = np.sqrt(XX**2 + YY**2)

        mask = ma.masked_outside(r, 0, r_max)

        r0 = 0.01
        mask_source = 0.5*(1.-np.tanh(r/r0))
        delta = 1/(self.params["global_nz"])
        Q = 1e-5*mask_source/delta

        for t_i in range(n_time):

            b = Variable(self.template, 'b')[t_i, 0]
            arg = (b - br[0])*Q/(NN)
            budget[t_i]= ma.masked_array(arg, mask.mask).mean()

        return budget

    def potential_energy_flux(self, r_lim, z_lim):
        """

        """
        npx = self.params['npx']
        Lx = self.params['Lx']
        Ly = self.params['Ly']
        Lz = self.params['Lz']
        nz = self.params['nz']
        dx = Lx/npx
        t = self.read_vars('t')['t']
        n_time = t.shape[0]
        r_max = r_lim # as in forced_plume_nudging.py
        z_max = z_lim
        new_nz = int(nz*z_lim)
        budget = np.zeros(n_time)
        # ineficient to read all velocities, but no time to be efficient here
        fields = self.read_vars(['x', 'y','z'])
        z = fields['z']
        X, Y = np.meshgrid(fields['x']/Lx - 0.5,
                             fields['y']/Ly - 0.5)
        r = np.sqrt(X**2 + Y**2)

        #mask_1 = ma.masked_outside(r, r_max - 0.01, r_max)
        mask = ma.masked_outside(r, 0, r_max)
        m = mask.mask*1
        mask_ring = np.roll(m, -1, axis=0) + np.roll(m, 1, axis=0)
        mask_ring += np.roll(m, -1, axis=1) + np.roll(m, 1, axis=1)
        mask_ring -= 4*m
        br = Variable(self.template, 'b')[0,:,0,0]
        for t_i in range(n_time):
            sides = 0
            for z_i in range(new_nz):
                u = Variable(self.template, 'u')[t_i, z_i]/self.params['dx']
                u = velocity_interpolation(u, axis=1)
                v = Variable(self.template, 'v')[t_i, z_i]/self.params['dy']
                v = velocity_interpolation(v, axis=0)

                rad_proy = (u*X + v*Y)/r
                aux = ma.masked_array(br[z_i]*z[z_i]*rad_proy, mask_ring>=0)
                sides += aux.mean()

            # w = Variable(self.template, 'w')[t_i]/self.params['dz']
            # w = velocity_interpolation(w, axis=0)
            # lid = ma.masked_array(f*w[new_nz], mask.mask)

            budget[t_i] = sides #+ lid.mean()

        return budget

def velocity_interpolation(a, axis=-1):
    """
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

def find_z_plume(array, percent):
    """
    Returns the index corresponding to the height of the plume.


    Parameters
    ----------
    array : array_like
        must be the momentum flux computed with Flux_levels.
    percent : float between 0 to 1
        the criteria for the plume heigth. Normally is 10% (i.e. 0.1)
        of the maximum momentum flux.

    Returns
    -------
    idx : int
        index of the top limit.
    """
    array = np.asarray(array[1:])
    maximum = array.max()
    difference = np.abs(array - maximum*percent)
    idx = difference.argmin()
    return idx+1
