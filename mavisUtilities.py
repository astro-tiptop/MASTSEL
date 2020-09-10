from copy import deepcopy
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.modeling import models, fitting

import scipy.signal
import scipy.ndimage
import scipy.interpolate

degToRad = np.pi/180.0
radToDeg = 1.0/degToRad
radiansToArcsecs = 206265.0
arcsecsToRadians = 1.0/radiansToArcsecs
radiansToMas = radiansToArcsecs * 1000.0

sigmaToFWHM = 2 * np.sqrt(2.0 * np.log(2.0))


def cartesianToPolar(x):
    return np.asarray( [np.sqrt(x[0]**2+x[1]**2), np.arctan2(x[1],x[0])*radToDeg], dtype=np.float64 )
    

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
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
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
        olddims = [np.arange(i, dtype=np.float) for i in list(a.shape)]

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

