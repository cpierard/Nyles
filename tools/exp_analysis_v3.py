import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


plt.ion()


datadir = "../data/Nyles"
expname = "ekman_0"
ncfile = "%s/%s/%s_00_hist.nc" % (datadir, expname, expname)


def read_param(ncfile):
    """ Retrieve all the parameters stored in the history file

    Parameters
    ----------
    ncfile : str
          the name of the NetCDF file

    Returns
    -------
    param : dict
         a dictionary of the experiment parameters
    """
    integers = "0123456789"
    param = {}
    with Dataset(ncfile, "r") as nc:
        param_list = nc.ncattrs()
        # print(param_list)
        for p in param_list:
            val = nc.getncattr(p)
            if type(val) is str:
                if val in ["False", "True"]:
                    val = (val == "True")
                elif "class 'list" in val:
                    val = val.split('>:')[-1].strip()
                    val = val[1:-1].split(', ')
                    if val[0][0] in integers:
                        val = [int(e) for e in val if e[0] in integers]
                    elif val[0][0] is "'":
                        val = [e.strip("'") for e in val]
            param[p] = val
    return param

param = read_param(ncfile)
dx = param["Lx"]/param["global_nx"]
dy = param["Ly"]/param["global_ny"]
dz = param["Lz"]/param["global_nz"]

nh = param["nh"]
nz, ny, nx = param["global_nz"], param["global_ny"], param["global_nx"]

if "x" in param["geometry"]:
    dom_x = slice(nh, nh+nx)
else:
    dom_x = slice(nx)

if "y" in param["geometry"]:
    dom_y = slice(nh, nh+ny)
else:
    dom_y = slice(ny)

if "z" in param["geometry"]:
    dom_z = slice(nh, nh+nz)
else:
    dom_z = slice(nz)

with Dataset(ncfile) as nc:
    # remember: (u,v,w) in the model are the *covariant*
    # components of the velocity. Their dimension is L^2 T^-1
    # the "real" velocity components are obtained by division
    # with the cell lengths dx, dy and dz
    u4 = nc.variables["u"][:, dom_z, dom_y, dom_x]/dx
    v4 = nc.variables["v"][:, dom_z, dom_y, dom_x]/dy
    w4 = nc.variables["w"][:, dom_z, dom_y, dom_x]/dz
    b4 = nc.variables["b"][:, dom_z, dom_y, dom_x]
    time = nc.variables["t"][...]
    x = nc.variables["x"][dom_x]
    y = nc.variables["y"][dom_y]
    z = nc.variables["z"][dom_z]

nt, nz, ny, nx = np.shape(u4)
print("the shape of 4D variables is: ", u4.shape)
# as you can see nx, ny, nz are not necessarily equal to
print("nx=%i / nx_glo=%i" % (nx, param["global_nx"]))
print("ny=%i / ny_glo=%i" % (ny, param["global_ny"]))
print("nz=%i / nz_glo=%i" % (nz, param["global_nz"]))

# this is because of the "halo".
# If a dimension is periodic then an extra halo is added on the left
# and the right. The halo width is in
nh = param["nh"]

# Watch out, when you do averages over the domain, you don't want to
# double count these points ...


def get_slice(n, nglo, nh, direction):
    """ return the slice that spans the interior elements,
    in one direction, detecting whether there is a halo or not
    """
    if n == nglo:
        idx = slice(0, n)
        print("closed in %s" % direction)
    else:
        idx = slice(nh, n+nh)
        print("periodic in %s" % direction)
    return idx


xidx = get_slice(nx, param["global_nx"], nh, "x")
yidx = get_slice(ny, param["global_ny"], nh, "y")


def horizontal_average(field, xidx, yidx):
    """
    Do a horizontal average

    Parameters
    ----------
    field : array,  2D or 3D
         the field that you want to average

    xidx, yidx : slice
         ranging over the interior points, excluding the halo

    Returns
    -------
    res : float or 1D array
       the horizontal average

    """
    if len(field.shape) == 2:
        res = np.mean(field[yidx, xidx])
    elif len(field.shape) == 3:
        res = np.mean(np.mean(field[:, yidx, xidx], axis=2), axis=1)
    else:
        raise ValueError("field should be a 2D or 3D array")
    return res

# let's do a horizontal average of buoyancy
mean_b = np.zeros((nz, nt))
for kt in range(nt):
    mean_b[:, kt] = horizontal_average(b4[kt], xidx, yidx)

plt.figure()
im = plt.pcolor(time, z, mean_b)
plt.xlabel("time")
plt.ylabel("z")
plt.title("buoyancy")
plt.colorbar(im)

# let us compute the mean kinetic energy for each vertical level


def comp_ke(u3, v3, w3, xidx, yidx):
    """ compute the horizontally averaged kinetic energy """
    ku2 = horizontal_average(u3**2, xidx, yidx)
    kv2 = horizontal_average(v3**2, xidx, yidx)
    kw2 = horizontal_average(w3**2, xidx, yidx)
    ke = 0.5*(ku2+kv2+kw2)
    return ke


mean_ke = np.zeros((nz, nt))
for kt in range(nt):
    mean_ke[:, kt] = comp_ke(u4[kt], v4[kt], w4[kt], xidx, yidx)

plt.figure()
im = plt.pcolor(time, z, mean_ke)
plt.xlabel("time")
plt.ylabel("z")
plt.title("kinetic energy")
plt.colorbar(im)

# let us plot now the domain averaged kinetic energy
plt.figure()
plt.plot(time, np.mean(mean_ke, axis=0), label="KE")
plt.xlabel("time")
plt.ylabel("kinetic energy")

# let us superimpose the domain averaged potential energy
mean_pe = z[:, np.newaxis]*mean_b
plt.plot(time, np.mean(mean_pe, axis=0), label="PE")
plt.legend()
plt.grid()

# let's compute the horizontally averaged velocity
mean_u = np.zeros((nz, nt))
mean_v = np.zeros((nz, nt))
mean_w = np.zeros((nz, nt))
for kt in range(nt):
    mean_u[:, kt] = horizontal_average(u4[kt], xidx, yidx)
    mean_v[:, kt] = horizontal_average(v4[kt], xidx, yidx)
    mean_w[:, kt] = horizontal_average(w4[kt], xidx, yidx)

# plot the time evolution at the top boundary
plt.figure()
plt.plot(time, mean_u[-1], label="u")  # do you understand the -1 ?
plt.plot(time, mean_v[-1], label="v")  # it's a trick to pick the top level
plt.xlabel("time")
plt.ylabel("velocity")
plt.legend()

# and do the "hodograph" at the last snapshot
plt.figure(figsize=(8, 8))
plt.plot(mean_u[:, -1], mean_v[:, -1], '+-', linewidth=3)
# you may also try the quiver
# plt.quiver(0,0, mean_u[:,-1],mean_v[:,-1], z)
plt.axis("equal")
plt.xlabel("U")
plt.ylabel("V")


# to compute the product u*w we need to estimate u and w at the same location. Let's say the cell centers
# u3 and w3 are 3D array (z,y,x)
#
# | x | x | x | x |  schematic of four cells
#   0   1   2   3    indexing for cell centers (x)
#     0   1   2   3  indexing for cell   edges (|)
#

u3 = u4[-1]  # last snapshot
w3 = w4[-1]

# if the domain is periodic in x
ucenter = 0.5*(u3+np.roll(u3, 1, axis=2))

# and for w the domain is bounded, the missing w on the left (bottom) is w=0 (no-flow)
wcenter = 0.5*w3
wcenter[1:, :, :] += 0.5*w3[:-1, :, :]

# let's illustrate ucenter
plt.figure()
plt.plot(u3[-1, :, 0], label="i=0")
plt.plot(u3[-1, :, 1], label="i=1")
plt.plot(ucenter[-1, :, 1], label="mid point") # <= mid point between 0 and 1
plt.legend()
plt.ylabel("u(j)")
plt.xlabel("j")
plt.tight_layout()

# let's illustrate wcenter
plt.figure()
plt.plot(w3[-2, 0, :], label="lev=-2") # <= -2 is the last to one level
plt.plot(w3[-3, 0, :], label="lev=-3")
plt.plot(wcenter[-2, 0, :], label="mid point") # <= mid level between -2 and -3
plt.legend()
plt.ylabel("w(i)")
plt.xlabel("i")
plt.tight_layout()

# ----------------------------------------
# let's compute <uw>(z) for a given kt
kt = -1# last snapshot
u3 = u4[kt]
w3 = w4[kt]

# if the domain is periodic in x
ucenter = 0.5*(u3+np.roll(u3, 1, axis=2))
# and for w the domain is bounded, the missing w on the left (bottom)
# is w=0 (no-flow)
wcenter = 0.5*w3
wcenter[1:, :, :] += 0.5*w3[:-1, :, :]

umean = np.zeros((nz))
wmean = np.zeros((nz))
uw = np.zeros((nz))
for kz in range(nz):
    umean[kz] = horizontal_average(ucenter[kz], xidx, yidx)
    wmean[kz] = horizontal_average(wcenter[kz], xidx, yidx)
    uw[kz] = horizontal_average(ucenter[kz]*wcenter[kz], xidx, yidx)

plt.figure()
plt.plot(z, uw)

# ----------------------------------------
um = np.zeros((nt,nz))
for kt in range(nt):
    for kz in range(nz):
        um[kt,kz]=horizontal_average(u4[kt,kz], xidx, yidx)

H = param["Lz"]
tend = param["tend"]
plt.figure()
plt.imshow(um.T,origin="xy",vmin=-.3,vmax=.3,cmap="RdBu_r")
