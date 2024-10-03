from copy import deepcopy

import numpy as np

from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.modeling import models, fitting

import scipy.signal
import scipy.ndimage

from . import gpuEnabled

if not gpuEnabled:
    cp = np
    import scipy.interpolate    
    from scipy.interpolate import RegularGridInterpolator

    def congrid(a, newdims, method='linear', centre=False, minusone=False):
        '''Arbitrary resampling of source array to new dimension sizes.
        Currently only supports maintaining the same number of dimensions.
        To use 1-D arrays, first promote them to shape (x,1).

        Uses the same parameters and creates the same co-ordinate lookup points
        as IDL''s congrid routine, which apparently originally came from a VAX/VMS
        routine of the same name.

        method:
        neighbour - closest value from original data
        nearest and linear - uses n x 1-D interpolations using
                             scipy.interpolate.interp1d
        (see Numerical Recipes for validity of use of n 1-D interpolations)
        spline - uses ndimage.map_coordinates

        centre:
        True - interpolation points are at the centres of the bins
        False - points are at the front edge of the bin

        minusone:
        For example- inarray.shape = (i,j) & new dimensions = (x,y)
        False - inarray is resampled by factors of (i/x) * (j/y)
        True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
        This prevents extrapolation one element beyond bounds of input array.
        '''
        if a.dtype not in [np.float64, np.float32]:
            a = np.asarray(a, np.float64)

        m1 = np.asarray(minusone, np.int64)
        ofs = np.asarray(centre, np.int64) * 0.5
        old = np.array(a.shape)
        ndims = len(a.shape)
        if len(newdims) != ndims:
            print("[congrid] dimensions error. "
                  "This routine currently only support "
                  "rebinning to the same number of dimensions.")
            return None
        newdims = np.asarray(newdims, dtype=float)
        dimlist = []

        if method == 'neighbour':
            for i in range(ndims):
                base = np.indices(newdims)[i]
                dimlist.append((old[i] - m1) / (newdims[i] - m1)
                               * (base + ofs) - ofs)
            cd = np.array(dimlist).round().astype(int)
            newa = a[list(cd)]
            return newa

        elif method in ['nearest', 'linear']:
            # calculate new dims
            for i in range(ndims):
                base = np.arange(newdims[i])
                dimlist.append((old[i] - m1) / (newdims[i] - m1)
                               * (base + ofs) - ofs)
            # specify old dims
            olddims = [np.arange(i, dtype=float) for i in list(a.shape)]

            # first interpolation - for ndims = any
            mint = scipy.interpolate.interp1d(
                olddims[-1], a, kind=method, fill_value="extrapolate")
            newa = mint(dimlist[-1])

            trorder = [ndims - 1] + list(range(ndims - 1))
            for i in range(ndims - 2, -1, -1):
                newa = newa.transpose(trorder)

                mint = scipy.interpolate.interp1d(
                    olddims[i], newa, kind=method, fill_value="extrapolate")
                newa = mint(dimlist[i])

            if ndims > 1:
                # need one more transpose to return to original dimensions
                newa = newa.transpose(trorder)

            return newa
        elif method in ['spline']:
            oslices = [slice(0, j) for j in old]
            oldcoords = np.ogrid[oslices]
            nslices = [slice(0, j) for j in list(newdims)]
            newcoords = np.mgrid[nslices]

            newcoords_dims = range(np.rank(newcoords))
            # make first index last
            newcoords_dims.append(newcoords_dims.pop(0))
            newcoords_tr = newcoords.transpose(newcoords_dims)
            # makes a view that affects newcoords

            newcoords_tr += ofs

            deltas = (np.asarray(old) - m1) / (newdims - m1)
            newcoords_tr *= deltas

            newcoords_tr -= ofs

            newa = scipy.ndimage.map_coordinates(a, newcoords)
            return newa
        else:
            print("Congrid error: Unrecognized interpolation type.\n",
                  "Currently only \'neighbour\', \'nearest\',\'linear\',",
                  "and \'spline\' are supported.")
        return None
else:
    import cupy as cp
    from cupyx.scipy.interpolate import RegularGridInterpolator
    # works for now, to be checked in different cases
    def congrid(a, newdims):

        newdims = np.asarray(newdims, dtype=int)
        r1 = a.shape[0]/newdims[0]
        r2 = a.shape[1]/newdims[1]

        interp = RegularGridInterpolator((cp.linspace(0.5, a.shape[0]-0.5, a.shape[0]), cp.linspace(0.5, a.shape[1]-0.5, a.shape[1])), 
                                         a, bounds_error=False, fill_value=None)

        xx = cp.linspace(0.5, a.shape[0] - (r1 - 0.5), newdims[0])
        yy = cp.linspace(0.5, a.shape[1] - (r2 - 0.5), newdims[1])

        X, Y = cp.meshgrid(xx, yy, indexing='ij')
        return interp((X, Y))

degToRad = np.pi/180.0
radToDeg = 1.0/degToRad
radiansToArcsecs = 206265.0
arcsecsToRadians = 1.0/radiansToArcsecs
radiansToMas = radiansToArcsecs * 1000.0

sigmaToFWHM = 2 * np.sqrt(2.0 * np.log(2.0))


def cartesianToPolar(x):
    return np.asarray( [np.sqrt(x[0]**2+x[1]**2), np.arctan2(x[1],x[0])*radToDeg], dtype=np.float64 )


def cartesianToPolar2(x):
    return np.asarray( [np.sqrt(x[:,0]**2+x[:,1]**2), np.arctan2(x[:,1],x[:,0])*radToDeg], dtype=np.float64 ).transpose()



def polarToCartesian(x):
    return x[0] * np.asarray( [np.cos(x[1]*degToRad), np.sin(x[1]*degToRad)], dtype=np.float64 )


def FWHM_from_sigma(sigma):
    return sigma * sigmaToFWHM


def sigma_from_FWHM(fwhm):
    return fwhm / sigmaToFWHM


def polar_to_cart(polar_data, theta_step, range_step, x, y, order=3):
    from scipy.ndimage.interpolation import map_coordinates as mp
    X, Y = np.meshgrid(x, y)
    Tc = np.degrees(np.arctan2(Y, X)).ravel()
    Rc = (np.sqrt(X**2 + Y**2)).ravel()
    Tc[Tc < 0.0] = 360.0 + Tc[Tc < 0.0]
    Tc = Tc / theta_step
    Rc = Rc / range_step
    coords = np.vstack((Tc, Rc))
    polar_data = np.vstack((polar_data, polar_data[-1, :]))
    cart_data = mp(
        polar_data,
        coords,
        order=order,
        mode='constant',
        cval=np.nan)
    return(cart_data.reshape(len(y), len(x)).T)

def fitGaussian(image):
    N = image.shape[0]
    p_init = models.Gaussian2D(
        amplitude=np.max(image),
        x_mean=N / 2,
        y_mean=N / 2)
    fit_p = fitting.LevMarLSQFitter()
    return fit_p, p_init


def fitAiry(image):
    N = image.shape[0]
    p_init = models.AiryDisk2D(amplitude=np.max(image), x_0=N / 2, y_0=N / 2)
    fit_p = fitting.LevMarLSQFitter()
    return fit_p, p_init


def standardPsfPlot(img2):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    s = img2.shape[0]
    axs[0].imshow(img2, cmap=cm.gist_heat)
    axs[1].plot(img2[s // 2, :])
    axs[2].plot(img2[:, s // 2])
    plt.show()

    
def twoPsfsPlot(result, myResult):
    plt.xscale('linear')
    plt.yscale('log')
    plt.plot(result[:, result.shape[0] // 2])
    plt.plot(myResult[:, myResult.shape[0] // 2])
    plt.show()


def simple2Dgaussian(x, y, x0=0.0, y0=0.0, sg=1.0, xp=np):
    return xp.exp(-((x-x0)**2)/(2*sg**2)-((y-y0)**2)/(2*sg**2) )

def intRebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)

    
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plotEllipses(cartPointingCoords, cov_ellipses, ss):
    nPointings = cov_ellipses.shape[0]
    ells = [Ellipse(xy=cartPointingCoords[i],  width=cov_ellipses[i,1]*ss, height=cov_ellipses[i,2]*ss, angle=cov_ellipses[i,0]*180/np.pi) for i in range(nPointings)]
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'}, figsize=[8,8])
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    plt.show()

def tiledDisplay(results):    
    nn = len(results)
    ncols = int(np.sqrt(nn))
    nrows = ncols
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=[10,10])
    
    if nrows>1 or ncols>1:
        for i in range(nrows):
            for j in range(ncols):
                img = np.log(np.abs(results[i*ncols+j].sampling) + 1e-20)
                ax[nrows-i-1,j].imshow(img, cmap='hot')
                #ax[nrows-i-1,j].axis('off')
    else:
        img = np.log(np.abs(results[0].sampling) + 1e-20)
        ax.imshow(img, cmap='hot')
        
    
#    for i in range(nrows):
#        for j in range(ncols):
#            img = np.log(np.abs(results[i*ncols+j].sampling) + 1e-20)
#            ax[nrows-i-1,j].imshow(img, cmap='hot')
#            #ax[nrows-i-1,j].axis('off')
    fig.tight_layout()
    plt.show()
