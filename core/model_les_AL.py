import vortex_force as vortf
import variables as var
import tracer as tracer
import timescheme as ts
import vorticity as vort
import bernoulli as bern
import kinenergy as kinetic
import projection
import topology as topo
from timing import timing
import mg

import numpy as np
import matplotlib.pyplot as plt

"""
LES model

At each step n, this model doesn't suppose that the velocity at step n-1 had 0 divergence.
It follows the procedure from the Ferziger p.180

To follow that, changes have been made to the calculate_p_from_dU subroutine, and should be tested
when mg_idx is implemented

"""


class LES(object):

    def __init__(self, param):
        self.state = var.get_state(param)
        self.timescheme = ts.Timescheme(param, self.state)
        self.timescheme.set(self.rhs)
        self.traclist = ['b']
        self.orderB = 1
        self.mg = mg.Multigrid(param)

    def rhs(self, state, t, dstate):
        U_from_u(state)
        # Diagnostic variables
        vort.vorticity(state)
        kinetic.kinenergy(state)

        # buoyancy
        tracer.rhstrac(state, dstate, self.traclist, self.orderB)

        # vortex force
        vortf.vortex_force(state, dstate, 5)  # get order from param
        # bernoulli
        bern.bernoulli(state, dstate)
        # dU from du when enter
        # U_from_u(dstate)
        # pressure
        #calculate_p_from_dU(self.mg, state, dstate)
        projection.calculate_p_from_dU(self.mg, state, dstate)

    def forward(self, t, dt):
        self.timescheme.forward(self.state, t, dt)
        div = self.state.work
        projection.compute_div(div, self.state)
        print("Divergence :", np.mean(div.view()))


def U_from_u(state):
    # copied from lotsofstuff
    idx2, idy2, idz2 = 1., 1., 1.
    # for now implements only cartesian
    metric = 'cartesian'  # dx, dy and dz are uniform, though not necessarily equal

    if metric == 'cartesian':
        u = state.u['i'].view('k')
        v = state.u['j'].view('k')
        w = state.u['k'].view('k')

        U = state.U['i'].view('k')
        V = state.U['j'].view('k')
        W = state.U['k'].view('k')

        U[:] = u*idx2
        V[:] = v*idy2
        W[:] = w*idz2

    elif metric == 'sigma1D':
        # cf Roullet et al, OM2017
        # slope = dz/dx, at cell center
        u, v, w = state.u.view('j')
        U, V, W = state.U.view('j')

        V[:] = v*idy2
        for j in range(ny):
            U[j][:, :] = u[j][:, :] - sxp(slope[j][:, :]*sym(w[j][:, :]))
            W[j][:, :] = gamma[j][:, :]*w[j][:, :] - \
                syp(slope[j][:, :]*sxm(u[j][:, :]))
    else:
        pass

# never need that one


def u_from_U(state):
    metric = 'cartesian'
    idx2, idy2, idz2 = 1., 1., 1.
    if metric == 'cartesian':
        u = state.u['i'].view('k')
        v = state.u['j'].view('k')
        w = state.u['k'].view('k')

        U = state.u['i'].view('k')
        V = state.u['j'].view('k')
        W = state.u['k'].view('k')

        u[:] = U/idx2
        v[:] = V/idy2
        w[:] = W/idz2

    else:
        raise NotImplementedError(
            "Only Cartesian coordinates are implemented for now")


# Ferziger p.180

#
# def calculate_p_from_dU(multg, state, dstate, pstate):
#    print("mg_idx not implemented yet")
#

@timing
def calculate_p_from_dU(multg, state, dstate):

    # This solves the poisson
    # equation with dU (dstate.U),
    # stores the result in p
    # (state.p) and updates dU

    # mg is the multigrid object (with all data and methods)

    # compute divergence
    # cff is the inverse metric tensor
    # TODO: handle this information more neatly
    cff = {'i': 1., 'j': 1., 'k': 1.}
    for count, i in enumerate('jki'):
        #div = state.work.view(i)
        div = state.work.view(i)
        div_u_p = state.work.view(i)

        dU = dstate.u[i].view(i)*cff[i]
        u_prev = state.u[i].view(i)
        if count == 0:
            div *= 0
            div_u_p *= 0
        div[:, :, 1:] += dU[:, :, 1:]-dU[:, :, :-1]
        div_u_p[:, :, 1:] += u_prev[:, :, 1:]-u_prev[:, :, :-1]

    dt = .01
    RHS = div - div_u_p * 1 / dt

    # at the end of the loop div and dU are in the 'i' convention
    # this is mandatory because MG only works with the 'i' convention

    # copy divergence into the multigrid RHS
    # watch out, halo in MG is nh=1, it's wider for div
    b = multg.grid[0].toarray('b')
    x = multg.grid[0].toarray('x')

    # this is triplet of slices than span the MG domain (inner+MG halo)
    # typically mg_idx = (kidx, jidx, iidx)
    # with kidx = slice(k0, k1) the slice in the k direction
    # todo: implement that into state [easy once MG is back into the master branch]
    mg_idx = state.work.mg_idx
    b[:] = RHS[mg_idx]

    # solve
    multg.solve_directly()

    # copy MG solution to pressure
    p = state.p.view('i')
    p[mg_idx] = x[:]

    # correct du (the covariant component)
    # now we start with the 'i' convention
    for i in 'ijk':
        p = state.p.view(i)
        du = dstate.u[i].view(i)
        du[:, :, :-1] -= p[:, :, 1:]-p[:, :, :-1]


if __name__ == "__main__":
    procs = [1, 1, 1]
    topo.topology = 'closed'
    myrank = 0
    nz = 32
    nh = 2

    loc = topo.rank2loc(myrank, procs)
    neighbours = topo.get_neighbours(loc, procs)

    param = {
        'nx': 48, 'ny': 64, 'nz': nz, 'nh': nh,
        'timestepping': 'LFAM3',
        'neighbours': neighbours,
        'procs': procs, 'topology': topo.topology,
        'npre': 3, 'npost': 3, 'omega': 0.8, 'ndeepest': 20, 'maxite': 20, 'tol': 1e-6
    }

    t = 0.
    dx = 1.
    cfl = 0.01
    dt = 0.1

    model = LES(param)

    u = model.state.u['i'].view('k')
    u[:, :, :] = .1

    for kt in range(10):
        model.forward(t, dt)
        t += dt
        u = model.state.u['i'].view('k')

    plt.figure()
    plt.pcolor(u[0, :, :])
    plt.colorbar()

    plt.show()
