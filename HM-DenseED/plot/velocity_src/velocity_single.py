'''
Reference: https://github.com/adsodemelk/PRST
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from plot.velocity_src.base_velocity import loadMRSTGrid, initResSol
import scipy.io as io
import plot.velocity_src.gridtools as gridtools
from plot.velocity_src.gridtools import getCellNoFaces
from numpy_groupies.aggregate_numpy import aggregate
def velocity_post(p,T,ft):
    G = loadMRSTGrid('./plot/mygrid.mat')
    neighborship, n_isnnc = gridtools.getNeighborship(G, "Topological", True, nargout=2)
    cellNo, cf, cn_isnnc = gridtools.getCellNoFaces(G)
    nif = neighborship.shape[0]
    ncf = cf.shape[0]
    nc = G.cells.num
    i = np.all(neighborship != -1, axis=1)
    state = initResSol(G, 0)

    hh = np.zeros((nif, 1))
    grav = np.zeros((65536,1))
    # Reconstruct face pressures and fluxes
    fpress = (aggregate(cf[:,0], (p[cellNo[:,0],0]+grav[:,0])*T[:,0], size=nif)
              /aggregate(cf[:,0], T[:,0], size=nif))[:,np.newaxis]
    # Neumann faces
    b = np.any(G.faces.neighbors==-1, axis=1)[:,np.newaxis]
    fpress[b[:,0],0] -= hh[b] / ft[b[:,0]]

    dF1 = io.loadmat('./plot/dF.mat')
    dF1 = dF1['dF']
    dF1 = np.array(dF1)
    dF2 = dF1==1


    dC = np.zeros((256,1))
    dC[:128,:] = -1
    dC[128:,:] = 1

    # Dirichlet faces
    fpress[dF2] = dC[:,0]

    # Sign for boundary faces
    noti = np.logical_not(i)
    sgn = 2*(G.faces.neighbors[noti,1]==-1)-1
    ni = neighborship[i]

    # Because of floating point loss of precision due to subtraction of similarly sized numbers,
    # this result can be slightly different from MATLAB for very low flux.
    flux = -aggregate(np.where(i)[0], ft[i] * (p[ni[:,1],0]-p[ni[:,0],0]), size=nif)[:,np.newaxis]
    c = np.max(G.faces.neighbors[noti,:], axis=1)[:,np.newaxis]
    flux[noti,0] = -sgn*ft[noti] * ( fpress[noti,0] - p[c[:,0],0])

    N, n_isnnc = gridtools.getNeighborship(G, "Topological", True, nargout=2)
    [cellNo, cellFaces,isNNC] = getCellNoFaces(G)
    sgn = 2*(N[cellFaces[:,0], 0] == cellNo[:,0]) - 1
    temp1 = flux[cellFaces, :]
    temp1 = temp1.reshape(65536,)
    cellFlux = sgn * temp1
    C  = G.faces.centroids[cellFaces, :] - G.cells.centroids[cellNo,:]
    C = C.reshape(65536,2)
    cf = cellFlux

    AAA  = np.arange(0,65536,1)
    cellNo = np.array(cellNo).ravel()
    cellNo = cellNo.astype(int)
    AAA = AAA.astype(int)
    v = sparse.coo_matrix((cf,(cellNo, AAA)),shape=(16384,65536))
    velocity = v*C
    return velocity


