"""Provide the grid on which Nyles is discretized.

TODO: implement handling of the halo (ghost points).
TODO: distinguish between total Lx, nx and sub-domain Lx, nx
TODO: multi-core handling: domain of second core starts at Lx/2, not Lx
"""

import numpy as np

# Local imports
from variables import Scalar, Vector


class Grid(object):
    def __init__(self, param):
        # Copy needed parameters
        self.nx = param["nx"]
        self.ny = param["ny"]
        self.nz = param["nz"]

        self.Lx = param["Lx"]
        self.Ly = param["Ly"]
        self.Lz = param["Lz"]

        # Define useful quantities
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz

        self.dx2 = self.dx**2
        self.dy2 = self.dy**2
        self.dz2 = self.dz**2

        self.idx = 1 / self.dx
        self.idy = 1 / self.dy
        self.idz = 1 / self.dz

        self.vol = self.dx * self.dy * self.dz

        # Define coordinates at buoyancy-point (cell centers)
        self.x_b = Scalar(param, "x at cell centers", "x_b", "m")
        self.y_b = Scalar(param, "y at cell centers", "y_b", "m")
        self.z_b = Scalar(param, "z at cell centers", "z_b", "m")
        # Create coordinate vectors
        self.x_b_1D = np.linspace(self.dx/2, self.Lx - self.dx/2, self.nx)
        self.y_b_1D = np.linspace(self.dy/2, self.Ly - self.dy/2, self.ny)
        self.z_b_1D = np.linspace(self.dz/2, self.Lz - self.dz/2, self.nz)
        # Make the vectors into coordinate matrices and save them
        x_b = self.x_b.view("i")
        y_b = self.y_b.view("i")
        z_b = self.z_b.view("i")
        z_b[:,:,:], y_b[:,:,:], x_b[:,:,:] = np.meshgrid(
            self.z_b_1D, self.y_b_1D, self.x_b_1D, indexing="ij"
        )
        # Note: the argument indexing="ij" to meshgrid is crucial to
        # comply with the [k,j,i]-convention.  Otherwise, if it was
        # omitted, the default value indexing="xy" would be taken, which
        # would result in the order [j,k,i], incompatible with the
        # implementation of the Scalar class.  Further information:
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html

        # Define coordinates at velocity-points (cell faces)
        self.x_vel = Vector(param, "x at cell faces", "x_vel", "m", is_velocity=True)
        self.y_vel = Vector(param, "y at cell faces", "y_vel", "m", is_velocity=True)
        self.z_vel = Vector(param, "z at cell faces", "z_vel", "m", is_velocity=True)
        # Create coordinate vectors
        self.x_u_1D = self.x_b_1D.copy() + self.dx/2
        self.y_u_1D = self.y_b_1D.copy()
        self.z_u_1D = self.z_b_1D.copy()
        self.x_v_1D = self.x_b_1D.copy()
        self.y_v_1D = self.y_b_1D.copy() + self.dy/2
        self.z_v_1D = self.z_b_1D.copy()
        self.x_w_1D = self.x_b_1D.copy()
        self.y_w_1D = self.y_b_1D.copy()
        self.z_w_1D = self.z_b_1D.copy() + self.dz/2
        # Make the vectors into coordinate matrices and save them
        x_u = self.x_vel["i"].view("i")
        y_u = self.y_vel["i"].view("i")
        z_u = self.z_vel["i"].view("i")
        z_u, y_u, x_u = np.meshgrid(
            self.z_u_1D, self.y_u_1D, self.x_u_1D, indexing="ij"
        )
        x_v = self.x_vel["j"].view("i")
        y_v = self.y_vel["j"].view("i")
        z_v = self.z_vel["j"].view("i")
        z_v, y_v, x_v = np.meshgrid(
            self.z_v_1D, self.y_v_1D, self.x_v_1D, indexing="ij"
        )
        x_w = self.x_vel["k"].view("i")
        y_w = self.y_vel["k"].view("i")
        z_w = self.z_vel["k"].view("i")
        z_w, y_w, x_w = np.meshgrid(
            self.z_w_1D, self.y_w_1D, self.x_w_1D, indexing="ij"
        )

        # Define coordinates at vorticity-points (cell edges)
        self.x_vor = Vector(param, "x at cell edges", "x_vor", "m", is_velocity=False)
        self.y_vor = Vector(param, "y at cell edges", "y_vor", "m", is_velocity=False)
        self.z_vor = Vector(param, "z at cell edges", "z_vor", "m", is_velocity=False)
        # Create coordinate vectors
        self.x_vor_i_1D = self.x_b_1D.copy()
        self.y_vor_i_1D = self.y_b_1D.copy() + self.dy/2
        self.z_vor_i_1D = self.z_b_1D.copy() + self.dz/2
        self.x_vor_j_1D = self.x_b_1D.copy() + self.dx/2
        self.y_vor_j_1D = self.y_b_1D.copy()
        self.z_vor_j_1D = self.z_b_1D.copy() + self.dz/2
        self.x_vor_k_1D = self.x_b_1D.copy() + self.dx/2
        self.y_vor_k_1D = self.y_b_1D.copy() + self.dy/2
        self.z_vor_k_1D = self.z_b_1D.copy()
        # Make the vectors into coordinate matrices and save them
        x_vor_i = self.x_vor["i"].view("i")
        y_vor_i = self.y_vor["i"].view("i")
        z_vor_i = self.z_vor["i"].view("i")
        z_vor_i, y_vor_i, x_vor_i = np.meshgrid(
            self.z_vor_i_1D, self.y_vor_i_1D, self.x_vor_i_1D, indexing="ij"
        )
        x_vor_j = self.x_vor["j"].view("i")
        y_vor_j = self.y_vor["j"].view("i")
        z_vor_j = self.z_vor["j"].view("i")
        z_vor_j, y_vor_j, x_vor_j = np.meshgrid(
            self.z_vor_j_1D, self.y_vor_j_1D, self.x_vor_j_1D, indexing="ij"
        )
        x_vor_k = self.x_vor["k"].view("i")
        y_vor_k = self.y_vor["k"].view("i")
        z_vor_k = self.z_vor["k"].view("i")
        z_vor_k, y_vor_k, x_vor_k = np.meshgrid(
            self.z_vor_k_1D, self.y_vor_k_1D, self.x_vor_k_1D, indexing="ij"
        )


if __name__ == "__main__":
    import topology as topo

    procs = [4, 2, 1]
    topo.topology = "closed"
    myrank = 3

    loc = topo.rank2loc(myrank, procs)
    neighbours = topo.get_neighbours(loc, procs)

    param = {
        "Lx": 50,
        "Ly": 60,
        "Lz": 10,
        "nx": 5,
        "ny": 4,
        "nz": 3,
        "nh": 2,
        "neighbours": neighbours,
    }

    grid = Grid(param)