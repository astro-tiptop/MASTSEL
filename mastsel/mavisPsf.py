import numpy as np

from . import gpuEnabled

if not gpuEnabled:
    cp = np
else:
    import cupy as cp

from mastsel.mavisUtilities import *
from mastsel.mavisFormulas import *

fit_window_max_size = 512
defaultArrayBackend = cp

def hostData(_data):
    if defaultArrayBackend == cp and gpuEnabled:
        return cp.asnumpy(_data)
    else:
        return _data


def ft_ift2(G, xp=defaultArrayBackend):
    g = xp.fft.fftshift(xp.fft.ifft2(xp.fft.fftshift(G)))
    return g


def ft_ft2(G, xp=defaultArrayBackend):
    g = xp.fft.fftshift(xp.fft.fft2(xp.fft.fftshift(G)))
    return g


def FFTConvolve(in1, in2, xp=defaultArrayBackend):
    if in1.ndim == in2.ndim == 0:  # scalar inputs
        return in1 * in2
    elif not in1.ndim == in2.ndim:
        raise ValueError("Dimensions do not match.")
    elif in1.size == 0 or in2.size == 0:  # empty arrays
        return array([])
    rr = ft_ift2(ft_ft2(in1, xp) * ft_ft2(in2, xp), xp)
    return rr


def KernelConvolve(in1, kernel, xp=defaultArrayBackend):
    if in1.ndim == kernel.ndim == 0:  # scalar inputs
        return in1 * kernel
    elif not in1.ndim == kernel.ndim:
        raise ValueError("Dimensions do not match.")
    elif in1.size == 0 or kernel.size == 0:  # empty arrays
        return array([])
    rr = ft_ift2(ft_ft2(in1, xp) * kernel, xp)
    return rr

# pixel size does not change


def zeroPad(input_grid, n, xp=defaultArrayBackend):
    N = input_grid.shape[0]
    output_grid = xp.zeros((N + 2 * n, N + 2 * n), dtype=input_grid.dtype)
    output_grid[n:N + n, n:N + n] = input_grid
    return output_grid

# pixel size does not change


def centralSquare(input_grid, n, xp=defaultArrayBackend):
    N = input_grid.shape[0]
    output_grid = xp.zeros((n, n), dtype=input_grid.dtype)
    output_grid = input_grid[int(N/2-n/2):int(N/2+n/2), int(N/2-n/2):int(N/2+n/2)]
    return output_grid


class Field(object):
    def __init__(
            self,
            wvl,
            N,
            width,
            unit='m',
            xp=defaultArrayBackend,
            _dtype=defaultArrayBackend.float64):
        self.xp = xp
        self.sampling = self.xp.zeros((N, N), dtype=_dtype)
        self.__N = N
        self.unit = unit
        self.width = width
        self.wvl = wvl

    @property
    def sampling(self):
        return self.__sampling

    @sampling.setter
    def sampling(self, value):
        self.__sampling = value
        self.__N = value.shape[0]

    @property
    def wvl(self):
        return self.__wvl

    @wvl.setter
    def wvl(self, value):
        self.__wvl = value
        self.__kk = 2.0 * np.pi / value

    @property
    def kk(self):
        return self.__kk

    @property
    def N(self):
        return self.__N

    @property
    def width(self):
        return self.__width

    @width.setter
    def width(self, value):
        self.__width = value
        self.__pixel_size = self.__width / self.__N
        self.__half_pixel_size = self.__pixel_size / 2.0

    @property
    def pixel_size(self):
        return self.__pixel_size

    @property
    def half_pixel_size(self):
        return self.__half_pixel_size

    def hostData(self, _data):
        if self.xp == cp and gpuEnabled:
            return cp.asnumpy(_data)
        else:
            return _data

    def gModel1(self, x, y, sigma_X, sigma_Y, angle):
        A = 1.0
        x0 = self.N / 2
        y0 = self.N / 2
        x_stddev= sigma_X / self.pixel_size
        y_stddev= sigma_Y / self.pixel_size
        theta = angle
        a = self.xp.cos(theta)*self.xp.cos(theta)/(2*x_stddev*x_stddev) + self.xp.sin(theta)*self.xp.sin(theta)/(2*y_stddev*y_stddev)
        b = self.xp.sin(2*theta)/(2*x_stddev*x_stddev) - self.xp.sin(2*theta)/(2*y_stddev*y_stddev)
        c = self.xp.sin(theta)*self.xp.sin(theta)/(2*x_stddev*x_stddev) + self.xp.cos(theta)*self.xp.cos(theta)/(2*y_stddev*y_stddev)
        return A * self.xp.exp( -a*(x-x0)*(x-x0)-b*(x-x0)*(y-y0)-c*(y-y0)*(y-y0) )

    def loadSamplingFromFile(self, filename):
        hdul = fits.open(filename)
        self.sampling = self.xp.asarray(hdul[0].data, self.sampling.dtype)

    # put the peak at the center of the Field, could change if needed
    # angle in rad
    # sigma_X, sigma_Y in the same unit as the Field
    def setAsGaussianKernel(self, sigma_X, sigma_Y, angle):
        yg, xg = self.xp.mgrid[:self.N, :self.N]
        hostSampling1 = self.gModel1(xg, yg, sigma_X, sigma_Y, angle)
        self.sampling = hostSampling1


    def setAsTelescopeMask(self, telescope_radius, occlusion_radius=0):
        fx = (self.xp.arange(-self.N / 2., self.N / 2., 1.0) + 0.5) * \
            self.pixel_size
        (fx, fy) = self.xp.meshgrid(fx, fx)
        self.sampling = self.xp.where(
            self.xp.logical_or(
                fx**2 +
                fy**2 > telescope_radius**2,
                fx**2 +
                fy**2 < occlusion_radius**2),
            0.0,
            1.0)
        self.telescope_radius = telescope_radius

    def setAsPSD(self, grid_diameter, r0_, L0_, l0_, method = 'VonKarman'):
        grid_pixel_scale = grid_diameter / float(self.N)
        freq_range = 1.0 / grid_pixel_scale
        ss, new_pixel_size = ft_PSD_phi( r0_, self.N, freq_range, L0_, l0_, method)
        self.sampling = self.xp.asarray(ss)
        self.unit = 'm'
        self.width = new_pixel_size * self.N

    def PSDToPhaseScreen(self):
        #        complexField = xp.zeros( (N,N), dtype=xp.complex128 )
        #        rand_pahse = xp.random.random_sample( (N,N), dtype=xp.float64 )
        #        complexField = xp.exp( rand_pahse*1j*xp.pi*2.0 ) * xp.sqrt(xp.abs(self.sampling))
        #        realField = xp.real()
        N = self.N
        freq_range = self.width
        cn = (
            (self.xp.random.normal(
                size=(
                    N,
                    N)) +
                1j *
                self.xp.random.normal(
                size=(
                    N,
                    N))) *
            self.xp.sqrt(
                self.sampling) *
            self.pixel_size)
        self.sampling = ft_ift2(cn, self.xp).real
        self.width = 1.0 / (freq_range / N)

    def pupilToPsf(self):
        self.sampling = self.xp.square(
            self.xp.absolute(
                ft_ift2(
                    self.sampling,
                    self.xp)))
        in_grid_size = self.pixel_size * self.N
        new_pixel_size = self.wvl / in_grid_size
        self.unit = 'rad'
        self.width = new_pixel_size * self.N

    def pupilToOtf(self):
        self.sampling = self.xp.real(
            ft_ft2(
                self.xp.square(
                    self.xp.absolute(
                        ft_ift2(
                            self.sampling,
                            self.xp)))) )
    def FWHM(self):
        if self.N <= fit_window_max_size:
            s1 = self.sampling
        else:
            s1 = centralSquare(self.sampling, fit_window_max_size, self.xp)
        nn = min(fit_window_max_size, self.N)
        z = self.hostData(self.xp.copy(s1))
        y, x = np.mgrid[:nn, :nn]
        fit_p, p_init = fitGaussian(z)
        p = fit_p(p_init, x, y, z)
        sigma_X = p.x_stddev.value
        sigma_Y = p.y_stddev.value
        return (
            FWHM_from_sigma(sigma_X) *
            self.pixel_size,
            FWHM_from_sigma(sigma_Y) *
            self.pixel_size)

    def radial_profile(self, center):
        data = self.sampling
        xp = self.xp
        x, y = xp.indices((data.shape))
        r = xp.sqrt((x - center[0])**2 + (y - center[1])**2)
        r = r.astype(xp.int)
        tbin = xp.bincount(r.ravel(), data.ravel())
        nr = xp.bincount(r.ravel())
        radialprofile = tbin / nr
        return radialprofile

    def n_square_energy(self, center):
        data = self.sampling
        xp = self.xp
        x, y = xp.indices((data.shape))
        r = xp.fmin(xp.abs(x - center[0]), xp.abs(y - center[1]))
        r = r.astype(xp.int)
        tbin = xp.bincount(r.ravel(), data.ravel())
        nr = xp.bincount(r.ravel())
        radialprofile = tbin
        energy = xp.cumsum(radialprofile)
        return energy

    def peak(self):
        if self.N <= fit_window_max_size:
            s1 = self.sampling
        else:
            s1 = centralSquare(self.sampling, fit_window_max_size, self.xp)
        nn = min(fit_window_max_size, self.N)
        z = self.hostData(self.xp.copy(s1))
        y, x = np.mgrid[:nn, :nn]
        fit_p, p_init = fitGaussian(z)
        p = fit_p(p_init, x, y, z)
        value = p.amplitude.value
        return value

    def normalize(self):
        self.sampling = self.sampling / self.xp.sum(self.sampling)

    def downSample(self, factor):
        self.sampling = congrid(
            self.sampling, [
                self.N / factor, self.N / factor])
        self.width *= factor

    def standardPlot(self, log=False, zoom=1):
        img1 = self.xp.copy(self.sampling)
        if log:
            img1 = np.log(np.absolute(self.sampling))
        if zoom > 1:
            img1 = centralSquare(img1, int(self.N / zoom / 2), self.xp)

        img2 = self.hostData(img1)       
        standardPsfPlot(img2)

    def printStatus(self):
        print('Wavelength: ', self.wvl)
        print('Grid side elements: ', self.N)
        print('width: ', self.width, '[', self.unit, ']')
        if self.unit == 'rad':
            print('width: ', self.width * radiansToMas, '[ mas ]')
        print('pixel_size: ', self.pixel_size, '[', self.unit, ']')
        if self.unit == 'rad':
            print('pixel_size: ', self.pixel_size * radiansToMas, '[ mas ]')


def StrehlFromReference(psf, reference):
    pp = psf.peak()
    print("Actual psf peak:", pp)
    return pp / reference


def StrehlFromMask(psf, mask):
    refPsf = mask
    refPsf.pupilToPsf()
    ref = refPsf.peak()
    print("Diffraction limeted psf peak:", ref)
    return StrehlFromReference(psf, ref)


def convolve(psf, kernel, xp=defaultArrayBackend):
    xp = psf.xp
    if (psf.N != kernel.N):
        print('psf and kernel sampling not compatible (grids sizes in pixels are different!)')
        print('psf.N:', psf.N) 
        print('kernel.N:', kernel.N) 
        return
#    if xp.abs((psf.pixel_size - kernel.pixel_size)) > 0.001:
#        print('These values should be the same!!!')
#        print('psf pixel size: ', psf.pixel_size)
#        print('kernel pixel size: ', kernel.pixel_size)
#        return
    result = Field(psf.wvl, psf.N, psf.width, unit='m')

    result.sampling = xp.real( KernelConvolve(psf.sampling/psf.sampling.sum(), kernel.sampling) )

    result.sampling = centralSquare(result.sampling, int(psf.N), xp)
    if xp.__name__=='cupy':
        result.sampling = xp.asnumpy(result.sampling)
    return result


def shortExposurePsf(mask, phaseScreen):
    xp = mask.xp
    if (mask.N != phaseScreen.N):
        print('Mask and phaseScreen sampling not compatible (grids sizes in pixels are different!)')
        print(mask.N, phaseScreen.N)
        return
    if xp.abs((mask.pixel_size - phaseScreen.pixel_size)) > 0.001:
        print('These values should be the same!!!')
        print('mask pixel size: ', mask.pixel_size)
        print('phaseScreen pixel size: ', phaseScreen.pixel_size)
        return
    result = Field(mask.wvl, mask.N, mask.width, unit='m')
    result.sampling = mask.sampling * phaseScreen.sampling
    result.standardPlot()
    result.pupilToPsf()
    return result


def longExposurePsf(mask, psd, otf_tel = None):
    xp = mask.xp
    if (mask.N != psd.N):
        print('Mask and PSD sampling not compatible (grids sizes in pixels are different!)')
        print('mask grid size: ', mask.N)
        print('psf grid size: ', psd.N)
        return
    freq_range = psd.width
    pitch = 1.0 / freq_range
    if xp.abs((mask.pixel_size - pitch) / pitch) > 0.001:
        print('These values should be the same!!!')
        print('otf_tel pixel size: ', mask.pixel_size)
        print('otf_turb pixel size: ', pitch)
        return
    p_final_psf = mask.wvl / (pitch * mask.N)  # rad
    result = Field(mask.wvl, mask.N, psd.N * p_final_psf, unit='rad')
    ################################################
    
    # step 0 : compute telescope otf
    if otf_tel is None:
        maskC = Field(mask.wvl, mask.N, pitch*mask.N)
        maskC.sampling = xp.copy(mask.sampling)
        maskC.pupilToOtf()
        otf_tel = maskC.sampling
    
    psd.sampling = zeroPad(psd.sampling, psd.sampling.shape[0]//2)
    
    # step 1 : compute phase autocorrelation
    B_phi = xp.real(xp.fft.ifft2(xp.fft.ifftshift(psd.sampling))) * (psd.kk * freq_range) ** 2
    b0 = B_phi[0, 0]
    B_phi = xp.fft.fftshift(B_phi)

    # step 2 : compute structure function
    # D_phi = 2.0 * (-B_phi + b0)
    D_phi = 2.0 * b0 - (B_phi + B_phi.conj())
        
    # step 3 : compute turbolence otf
    otf_turb = xp.exp(-0.5 * (D_phi))    
    # p_otft_turb = pitch
    otf_turb = congrid(otf_turb, [otf_turb.shape[0]//2, otf_turb.shape[0]//2])

    # step 4 : combine telescope and turbolence otfs
    otf_system = otf_turb * otf_tel

    # step 5 : system otf to system psf
    result.sampling = xp.real(ft_ft2(otf_system))
    if xp.__name__=='cupy':
        result.sampling =  xp.asnumpy(result.sampling)
        
    return result


def residualToSpectrum(ellp, wvl, N,  pixel_scale_otf):
    convKernelFFT = Field(wvl, int(N), N*pixel_scale_otf, '')
    convKernelFFT.setAsGaussianKernel(1/(2*np.pi*ellp[1]), 1/(2*np.pi*ellp[2]), -ellp[0])

    return convKernelFFT
