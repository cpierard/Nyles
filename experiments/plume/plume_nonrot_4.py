import numpy as np
import nyles as nyles_module
import parameters


nh = 3
nxglo = 32
nyglo = 32
nzglo = 16

npx = 1
npy = 1
npz = 1

nx = nxglo//npx
ny = nyglo//npy
nz = nzglo//npz

Lx = 4e3
Ly = 4e3
Lz = 2e3

# Get the default parameters, then modify them as needed
param = parameters.UserParameters()

param.model["modelname"] = "LES"
param.model["geometry"] = "closed"
param.model["Lx"] = Lx
param.model["Ly"] = Ly
param.model["Lz"] = Lz

param.IO["datadir"] = "~/data/Nyles"
#param.IO["datadir"] = "/home1/scratch/groullet/data/Nyles"
param.IO["expname"] = "plume_nonrot_4"
param.IO["mode"] = "overwrite"
param.IO["variables_in_history"] = ['b', 'u','p']

param.IO["timestep_history"] = 1800.  # 0.0 saves every frame
param.IO["disk_space_warning"] = 0.5  # in GB
param.IO["simplified_grid"] = True

param.time["timestepping"] = "LFAM3"
param.time["tend"] = 3600.*24.*10. #Ten days 
param.time["auto_dt"] = True
# parameter if auto_dt is False
param.time["dt"] = 500.
# parameters if auto_dt is True
param.time["cfl"] = 0.8
param.time["dt_max"] = param.time["dt"]

param.discretization["global_nx"] = nxglo
param.discretization["global_ny"] = nyglo
param.discretization["global_nz"] = nzglo
param.discretization["orderVF"] = 5  # upwind-order for vortex-force term
param.discretization["orderA"] = 5  # upwind-order for advection term

param.MPI["nh"] = nh
param.MPI["npx"] = npx
param.MPI["npy"] = npy
param.MPI["npz"] = npz
param.multigrid["nglue"] = 1
param.multigrid["tol"] = 1e-8

param.physics["forced"] = True
param.physics["rotating"] = False
param.physics["coriolis"] = 0e-4

def stratif(z):
    return 1e-2*(z-0.5)

class Forcing(object):
    def __init__(self, param, grid):
        x = grid.x_b.view('i') / param["Lx"]-0.5
        y = grid.y_b.view('i') / param["Ly"]-0.5
        z = grid.z_b.view('i') / param["Lz"]

        d = np.sqrt(x**2+y**2)
        r0 = 0.02 # <= radius of the heat source (domain horizontal extent is 100 r0)
        msk = 0.5*(1.-np.tanh(d/r0))
        delta = 1/(param["global_nz"])
        self.Q = np.zeros_like(z)
        Q0 = 2e-6/delta
        self.Q[0,:,:] = Q0
        self.Q *= msk

        self.bclim = stratif(z)
        #self.bclim[z<0.5] *= 2

        d0 = 0.45
        horwidth = 0.05
        h0 = 0.95
        verwidth = 0.05

        dampingcoef = 1./50 # <= rationalize this value

        self.damping = 0.5*(1+np.tanh((d-d0)/horwidth))
        cooling = self.damping.copy()
        cooling[z>0.2] = 0.
        
        coef = cooling.mean()
        Qmean = self.Q.mean()
        self.Q -= (Qmean/coef)*cooling
        print(Qmean/coef, self.Q.mean(), Qmean)
        
        self.damping *= 0.5*(1+np.tanh((z-h0)/verwidth))
        self.damping_msk = self.damping.copy()
        self.damping *= dampingcoef

        self.t0 = 0.
        
    def add(self, state, dstate, time):
        if time>0:
            dt= time-self.t0
            damping = (0.5/dt)*self.damping_msk
            self.t0=time
        else:
            damping =self.damping
        db = dstate.b.view("i")
        b = state.b.view("i")
        #db += self.Q - self.damping*(b-self.bclim)
        Bcoef=4e-3
        #bdiff = b-self.bclim
        db += self.Q - damping*(b-self.bclim)#(1.+(b-self.bclim)/Bcoef ))
        #db += self.Q - (damping/Bcoef)*np.sign(bdiff)*bdiff**2
        #db += self.Q - (damping/Bcoef**2)*(bdiff**3)
        # add a damping on w
        dw = dstate.u["k"].view("i")
        w = state.u["k"].view("i")
        dw -= damping*w

nyles = nyles_module.Nyles(param)

# the user must attach the forcing to the model
nyles.model.forcing = Forcing(nyles.param, nyles.grid)

b = nyles.model.state.b.view('i')
u = nyles.model.state.u['i'].view('i')
x = nyles.grid.x_b.view('i') / Lx
y = nyles.grid.y_b.view('i') / Ly
z = nyles.grid.z_b.view('i') / Lz

# linear stratification
b[:] = stratif(z)

# todo: add noise near the bottom to help trigger the turbulence

nyles.model.diagnose_var(nyles.model.state)

nyles.run()
