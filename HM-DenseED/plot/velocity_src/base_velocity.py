'''
Reference: https://github.com/adsodemelk/PRST
'''

import numpy as np
import plot.velocity_src.gridprocessing as gridprocessing
import scipy.io as io
import six
import plot.velocity_src.gridtools
from plot.velocity_src.gridtools import getCellNoFaces
from numpy_groupies.aggregate_numpy import aggregate
from plot.velocity_src.utils import Struct
from plot.velocity_src.rock import permTensor
from plot.velocity_src.rock import Rock
from  plot.velocity_src.utils import recursive_diff, rldecode
import scipy.sparse.linalg
from plot.velocity_src.plotting import plotCellData,_get_cell_nodes
import sys
import matplotlib.pyplot as plt
from scipy import sparse
def loadMRSTGrid(matfile, variablename="G"):
    """Loads MRST grid as PRST grid.

    The grid is saved in MATLAB using the command

        save('mygrid.mat', 'G', '-v7')

    where `G` is the name of the grid variable. All indices are converted to be
    zero-indexed. E.g., Cell 1 in MATLAB will be renamed Cell 0 in Python, and
    so on.

    Args:
        matfile (str): Path to saved grid .mat-file.

        variablename (Optional[str]): Name of grid variable in .mat-file.
            Defaults to "G". This must be specified if the grid is named
            something else than "G".

    Returns:
        G - PRST grid with zero indexing.

    See the source code of this function to see which variables are modified to
    be zero-indexed. Some important consequences of zero-indexing are:

        * The G.faces.neighbors array will no longer use 0 to indicate that
          there are no neighbors. Instead, -1 is used.

    """
    # Convert data to these types
    INT_DTYPE = np.int64
    FLOAT_DTYPE = np.float64

    data = io.loadmat(matfile, squeeze_me=True, struct_as_record=False)
    M = data[variablename] # MRST grid data, one-indexed

    G = gridprocessing.Grid()
    G.cells.num = M.cells.num
    G.cells.facePos = M.cells.facePos.astype(INT_DTYPE) - 1
    G.cells.facePos.shape = (G.cells.facePos.size,1)
    G.cells.faces = M.cells.faces.astype(INT_DTYPE) - 1
    if G.cells.faces.ndim == 1:
        # Make into column array
        G.cells.faces = G.cells.faces[:,np.newaxis]

    # computeGeometry attributes may not exist
    # try:
    G.cells.indexMap = M.cells.indexMap.astype(INT_DTYPE) - 1
    G.cells.indexMap = G.cells.indexMap[:,np.newaxis] # make into column
    # except AttributeError:
    #     prst.log.info("Loaded grid has no cells.indexMap")

    G.cells.volumes = M.cells.volumes.astype(FLOAT_DTYPE)
    G.cells.volumes = G.cells.volumes[:,np.newaxis]



    G.cells.centroids = M.cells.centroids.astype(FLOAT_DTYPE)


    G.faces.areas = M.faces.areas.astype(FLOAT_DTYPE)
    G.faces.areas = G.faces.areas[:,np.newaxis]


    G.faces.centroids = M.faces.centroids.astype(FLOAT_DTYPE)


    G.faces.normals = M.faces.normals.astype(FLOAT_DTYPE)


    G.faces.num = M.faces.num
    G.faces.nodePos = M.faces.nodePos.astype(INT_DTYPE) - 1
    G.faces.nodePos = G.faces.nodePos[:,np.newaxis] # 2d column

    G.faces.neighbors = M.faces.neighbors.astype(INT_DTYPE) - 1

    G.faces.nodes = M.faces.nodes.astype(INT_DTYPE) - 1
    G.faces.nodes = G.faces.nodes[:,np.newaxis] # 2d column

    G.nodes.num = M.nodes.num
    G.nodes.coords = M.nodes.coords.astype(FLOAT_DTYPE)


    G.cartDims = M.cartDims.astype(INT_DTYPE)

    # Matlab saves the gridType either as string or array of strings, depending
    # on the number of grid types. We use "gridType" since type is a Python
    # keyword.
    if isinstance(M.type, six.string_types):
        G.gridType = [M.type]
    elif isinstance(M.type, np.ndarray):
        # Convert to normal Python list for convenience
        G.gridType = list(M.type)
    else:
        raise ValueError("gridType has unknown type " + M.type.__class__.__name__)
    G.gridDim = M.griddim

    return G

def initResSol(G, p0, s0=0.0):
    """
    Initialize incompressible reservoir solution data structure.

    Synopsis:
        state = initResSol(G, p0)
        state = initResSol(G, p0, s0)

    Arguments:
        G (Grid):
            Grid structure

        p0 (scalar or ndarray):
            Initial reservoir pressure. Scalar or array with shape (G.cells.num,).

    Returns: state (State): Initialized reservoir solution structure with
    attributes:
        - pressure -- One scalar pressure value for each cell in `G`.
        - flux     -- One Darcy flux value for each face in `G`.
        - s        -- Phase saturations for all phases in each cell.

    Remarks:
        In the case of a (G.cells.num, 3)-shaped array of fluid saturations
        `state.s`, the columns are generally interpreted as

            0 <-> Aqua, 1 <-> Liquid, 2 <-> Vapour

        Single pressures (p0) and initial phase saturations (s0) are repeated
        uniformly for all grid cells.

        The initial Darcy flux is zero throughout the reservoir.

    See also:
        initWellSol (MRST only), solveIncompFlow (MRST only)
    """
    p0, s0 = np.atleast_2d(p0, s0)

    nc, nf = G.cells.num, G.faces.num
    if hasattr(G, "nnc") and hasattr(G.nnc, "cells"):
        # Expand the number of interfaces with the number of non-neighboring interfaces
        nf += G.nnc.cells.shape[0]

    if s0.shape[0] == 1:
        s0 = np.tile(s0, (nc,1))
    elif s0.shape[0] != nc:
        raise ValueError(
        "Initial saturation must either be 1-by-np or "+\
        "G.cells.num-by-np")

    if p0.size == 1:
        p0 = np.tile(p0, (nc,1))
    else:
        assert p0.shape[0] == nc

    resSol = Struct(pressure=p0,
                    flux=np.zeros((nf,1)),
                    s=s0)
    return resSol



def _dynamic_quantities(state, kr):

    mob = kr /10
    totmob = np.sum(mob, axis=1, keepdims=True)
    omega = np.sum(mob*1, axis=1, keepdims=True) / totmob
    return mob, omega, 1

def _compute_trans(G, T, cellNo, cellFaces, neighborship, totmob, use_trans):
    niface = neighborship.shape[0]
    if use_trans:
        neighborcount = np.sum(neighborship != -1, axis=1, keepdims=True)
        assert T.shape[0] == niface, \
            "Expected one transmissibility for each interface " + \
            "(={}) but got {}".format(niface, T.shape[0])
        raise NotImplementedError("Function not yet implemented for use_trans=True. See source code.")
        # Matlab code for rest of function, from mrst-2015b\modules\incomp\incompTPFA.m
        #fmob = accumarray(cellFaces, totmob(cellNo), ...
        #                    [niface, 1]);
        #
        #fmob = fmob ./ neighborcount;
        #ft   = T .* fmob;
        #
        #% Synthetic one-sided transmissibilities.
        #th = ft .* neighborcount;
        #T  = th(cellFaces(:,1));
    else:
        # Define face transmissibility as harmonic average of mobility 
        # weighted one-sided transmissibilities.
        assert T.shape[0] == cellNo.shape[0], \
            "Expected one one-sided transmissibility for each " +\
            "half face (={}), but got {}.".format(cellNo.shape[0], T.shape[0])

        T = T * totmob[cellNo[:,0],:]
        from numpy_groupies.aggregate_numpy import aggregate
        ft = 1/aggregate(cellFaces[:,0], 1/T[:,0], size=niface)
    return T, ft

def computeTrans(G, rock, K_system="xyz", cellCenters=None, cellFaceCenters=None,
        verbose=False):
    """
    Compute transmissibilities for a grid.

    Synopsis:
        T = computeTrans(G, rock)
        T = computeTrans(G, rock, **kwargs)

    Arguments:
        G (Grid):
            prst.gridprocessing.Grid instance.

        rock (Rock):
            prst.params.rock.Rock instance with `perm` attribute. The
            permeability is assumed to be in units of metres squared (m^2).
            Use constant `darcy` from prst.utils.units to convert to m^2, e.g.,

                from prst.utils.units import *
                perm = convert(perm, from_=milli*darcy, to=meter**2)

            if the permeability is provided in units of millidarcies.

            The field rock.perm may have ONE column for a scalar permeability in each cell,
            TWO/THREE columns for a diagonal permeability in each cell (in 2/D
            D) and THREE/SIX columns for a symmetric full tensor permeability.
            In the latter case, each cell gets the permability tensor.

                K_i = [ k1  k2 ]      in two space dimensions
                      [ k2  k3 ]

                K_i = [ k1  k2  k3 ]  in three space dimensions
                      [ k2  k4  k5 ]
                      [ k3  k5  k6 ]

        K_system (Optional[str]):
            The system permeability. Valid values are "xyz" and "loc_xyz".

        cellCenters (Optional[ndarray]):
            Compute transmissibilities based on supplied cellCenters rather
            than default G.cells.centroids. Must have shape (n,2) for 2D and
            (n,3) for 3D.

        cellFaceCenters (Optional[ndarray]):
            Compute transmissibilities based on supplied cellFaceCenters rather
            than default `G.faces.centroids[G.cells.faces[:,0], :]`.

    Returns:
        T: Half-transmissibilities for each local face of each grid cell in the
        grid. The number of half-transmissibilities equals the number of rows
        in `G.cells.faces`. 2D column array.

    Comments:
        PLEASE NOTE: Face normals are assumed to have length equal to the
        corresponding face areas. This property is guaranteed by function
        `computeGeometry`.

    See also:
        computeGeometry, computeMimeticIP (MRST), darcy, permTensor, Rock
    """
    if K_system not in ["xyz", "loc_xyz"]:
        raise TypeError(
            "Specified permeability coordinate system must be a 'xyz' or 'loc_xyz'")

    if verbose:
        print("Computing one-sided transmissibilites.")

    # Vectors from cell centroids to face centroids
    assert G.cells.facePos.ndim == 2, "facePos has wrong dimensions"
    cellNo = rldecode(np.arange(G.cells.num), np.diff(G.cells.facePos, axis=0))
    if cellCenters is None:
        C = G.cells.centroids
    else:
        C = cellCenters
    if cellFaceCenters is None:
        C = G.faces.centroids[G.cells.faces[:,0],:] - C[cellNo,:]
    else:
        C = cellFaceCenters - C[cellNo,:]

    # Normal vectors
    sgn = 2*(cellNo == G.faces.neighbors[G.cells.faces[:,0], 0]) - 1
    N = sgn[:,np.newaxis] * G.faces.normals[G.cells.faces[:,0],:]

    if K_system == "xyz":
        K, i, j = permTensor(rock, G.gridDim, rowcols=True)
        assert K.shape[0] == G.cells.num, \
            "Permeability must be defined in active cells only.\n"+\
            "Got {} tensors, expected {} (== num cells)".format(K.shape[0], G.cells.num)

        # Compute T = C'*K*N / C'*C. Loop based to limit memory use.
        T = np.zeros(cellNo.size)
        for k in range(i.size):
            tmp = C[:,i[k]] * K[cellNo,k] * N[:,j[k]]
            # Handle both 1d and 2d array.
            if tmp.ndim == 1:
                T += tmp
            else:
                T += np.sum(tmp, axis=1)
        T = T / np.sum(C*C, axis=1)

    elif K_system == "loc_xyz":
        if rock.perm.shape[1] == 1:
            rock.perm = np.tile(rock.perm, (1, G.gridDim))
        if rock.perm.shape[1] != G.cartDims.size:
            raise ValueError(
                "Permeability coordinate system `loc_xyz` is only "+\
                "valid for diagonal tensor.")
        assert rock.perm.shape[0] == G.cells.num,\
            "Permeability must be defined in active cells only. "+\
            "Got {} tensors, expected {} == (num cells)".format(rock.perm.shape[0], G.cells.num)

        dim = np.ceil(G.cells.faces[:,1] / 2)
        raise NotImplementedError("Function not finished for K_system='loc_xyz'")
        # See MRST, solvers/computeTrans.m

    else:
        raise ValueError("Unknown permeability coordinate system {}.".format(K_system))

    is_neg = T < 0
    if np.any(is_neg):
        if verbose:
            prst.log.warn("Warning: {} negative transmissibilities. ".format(np.sum(is_neg))+
                          "Replaced by absolute values...")
            T[is_neg] = -T[is_neg]

    return np.atleast_2d(T).transpose()




def computePressureRHS(G, omega, bc=None, src=None):
    if hasattr(G, "grav_pressure"):
        gp = G.grav_pressure(G, omega)
    else:
        gp = _grav_pressure(G, omega)

    ff = np.zeros(gp.shape)
    gg = np.zeros((G.cells.num, 1))
    hh = np.zeros((G.faces.num, 1))

    # Source terms
    if not src is None:
        print("computePressureRHS is untested for src != None")
        # Compatability check of cell numbers for source terms
        assert np.max(src.cell) < G.cells.num and np.min(src.cell) >= 0, \
            "Source terms refer to cell not existant in grid."

        # Sum source terms inside each cell and add to rhs
        ss = aggregate(src.cell, src.rate)
        ii = aggregate(src.cell, 1) > 0
        gg[ii] += ss[ii]

    dF = np.zeros((G.faces.num, 1), dtype=bool)
    dC = None

    if not bc is None:
        # Check that bc and G are compatible
        assert np.max(bc.face) < G.faces.num and np.min(bc.face) >= 0, \
            "Boundary condition refers to face not existant in grid."
        assert np.all(aggregate(bc.face, 1)) <= 1, \
            "There are repeated faces in boundary condition."

        # Pressure (Dirichlet) boundary conditions.
        # 1) Extract the faces marked as defining pressure conditions.
        #    Define a local numbering (map) of the face indices to the
        #    pressure condition values.
        is_press = bc.type == "pressure"
        face = bc.face[is_press]
        dC = bc.value[is_press]
        map = scipy.sparse.csc_matrix( (np.arange(face.size),
                                     (face.ravel(), np.zeros(face.size)))  )

        # 2) For purpose of (mimetic) pressure solvers, mark the "face"s as
        #    having pressure boundary conditions. This information will be used
        #    to eliminate known pressures from the resulting system of linear
        #    equations. See e.g. `solveIncompFlow` in MRST.
        dF[face] = True

        # 3) Enter Dirichlet conditions into system right hand side.
        #    Relies implicitly on boundary faces being mentioned exactly once
        #    in G.cells.faces[:,0].
        i = dF[G.cells.faces[:,0],:]
        ff[i] = -dC[map[ G.cells.faces[i[:,0],0],0].toarray().ravel()]

        # 4) Reorder Dirichlet conditions according to sort(face).
        #    This allows the caller to issue statements such as 
        #    `X[dF] = dC` even when dF is boolean.
        dC = dC[map[dF[:,0],0].toarray().ravel()]

        # Flux (Neumann) boundary conditions.
        # Note negative sign due to bc.value representing INJECTION flux.
        is_flux = bc.type == "flux"
        hh[bc.face[is_flux],0] = -bc.value[is_flux]

    if not dC is None:
        assert not np.any(dC < 0)

    return ff, gg, hh, gp, dF, dC

def _grav_pressure(G, omega):
    """Computes innerproduct cf (face_centroid - cell_centroid) * g for each face"""
    g_vec = [0,0,0]
    if np.linalg.norm(g_vec[:G.gridDim]) > 0:
        dim = G.gridDim
        assert 1 <= dim <= 3, "Wrong grid dimension"

        cellno = utils.rldecode(np.arange(G.cells.num), np.diff(G.cells.facePos, axis=0))
        cvec = G.faces.centroids[G.cells.faces[:,0],:] - G.cells.centroids[cellno,:]
        ff = omega[cellno] * np.dot(cvec, g_vec[:dim].reshape([3,1]))
    else:
        ff = np.zeros([G.cells.faces.shape[0], 1])
    return ff