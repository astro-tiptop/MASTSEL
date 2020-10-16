# path_pupil = "/home/frossi/dev/fourierPSF/_calib_/vlt_pup_240.fits"

#%% TELESCOPE #%%
TelescopeDiameter   = 8                                                                 # Telescope diameter  in m
zenithAngle         = 30                                                                # Telescope zenith angle in degrees
obscurationRatio    = 0.14                                                              # Central obstruction ratio
resolution          = 200                                                               # Pupil resolution in pixels
path_pupil          = []                                                                # Pupil path. If left empty, the pupil is circular

#%% ATMOSPHERE AT ZENITH#%%
atmosphereWavelength= 500e-9                                                            # Atmosphere wavelength in m
seeing              = 0.6                                                               # Seeing in arcsec - zenith at atmosphereWavelength
L0                  = 25                                                                # Outer scale in m
Cn2Weights          = [0.59, 0.02, 0.04, 0.06, 0.01, 0.05, 0.09, 0.04, 0.05, 0.05] # [0.5, 0.5] # temporary [0.59, 0.02, 0.04, 0.06, 0.01, 0.05, 0.09, 0.04, 0.05, 0.05]      # Fractional weights of layers
Cn2Heights          = [30, 140, 281, 562, 1125, 2250, 4500, 7750, 11000, 14000, 10000] #   [5000,15000, 10000] # temporary       # Layers altitude in m
wSpeed              = [6.6, 5.9, 5.1, 4.5, 5.1, 8.3, 16.3, 30.2, 34.3, 17.5]            # Wind speed in m/s
wDir                = [0., 0., 0., 0., 90., -90., -90., 90., 0., 0.]                    # WInd direction in degrees
nLayersReconstructed= 10                                                                # Number of reconstructed layers

#%% PSF EVALUATION DIRECTIONS #%%
ScienceWavelength   = 640e-9                                                            # Imaging wavelength [m]
ScienceZenith       = [0]                                                               # Distance from on-axis [arcsec]
ScienceAzimuth      = [45]                                                              # Azimuthal angle [degrees]
psInMas             = 7.4                                                               # PSF pixel scale in mas
psf_FoV             = 2.96                                                              # PSF fov [arcsec]
technical_FoV       = 120                                                               # Technical field of view (diameter)  [arcsec]

#%% GUIDE STARS  - HIGH ORDER LOOP 
GuideStarZenith_HO  = [17.5, 17.5 ,17.5 ,17.5, 17.5, 17.5, 17.5, 17.5]                  # Guide stars zenith position [arcsec]                                          
GuideStarAzimuth_HO = [0 , 45 , 90 , 135 , 180 , 225 , 270 , 315]                       # Guide stars azimuth position [degrees]
GuideStarHeight_HO  = 90e3                                                              # Guide stars height in m [(0 if infinite)]

#%% DM #%%
DmPitchs            = [0.22, 0.22, 0.35]                                                # DM actuators pitchs in m             
DmHeights           = [0 , 4000 , 10000] #temporary # [0 , 4000 , 14000]                                                # DM altitude in m
OptimizationZenith  = [0 , 15 , 15 , 15 , 15 , 15 , 15 , 15 , 15, 60 , 60 , 60 , 60 , 60 , 60 , 60 , 60] # Zenith position in arcsec
OptimizationAzimuth = [0 , 0 , 45 , 90 , 135 , 180 , 225 , 270 , 315 , 0 , 45 , 90 , 135 , 180 , 225 , 270 , 315] # Azimuth in degrees
OptimizationWeight  = [4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1]   # Weights
OptimizationConditioning = 1e2                                                          # Matrix Conditioning

#%% HIGH ORDER SENSOR
nLenslet_HO            = 40                                                             # number of WFS lenslets
SensingWavelength_HO   = 589e-9                                                         # Sensing wavelength in [m]
loopGain_HO            = 0.5                                                            # HO Loop gain
SensorFrameRate_HO     = 1e3                                                            # HO loop frequency in [Hz]
loopDelaySteps_HO      = 1                                                              # HO loop frame delay
nph_HO                 = 10                                                             # Flux return in [nph/frame/subaperture]
sigmaRON_HO            = 0.2                                                            # read-out noise std in [e-]
Npix_per_subap_HO      = 6                                                              # Number of pixels per subaperture
pixel_scale_HO         = 7.6                                                            # HO WFS pixel scale in [mas]

#%% LOW ORDER SENSOR
N_sa_tot_LO            = 4
SensingWavelength_LO   = 1650*1e-9                                                      # Sensing wavelenth in [m]
SensorFrameRate_LO     = 500                                                            # (= loop frequency): [500 Hz]  # , 250 Hz or 100 Hz (to be optimized)
loopDelaySteps_LO      = 3                                                              #Corresponding delays (in frames):     [3]       # , 2, 1 
pixel_scale_LO         = 40                                                             # [mas]
Npix_per_subap_LO      = 50                                                             #
WindowRadiusWCoG_LO    = 2                                                              # [pixels] calcolo sigma e mu, dimensione della finestra, cerchio di diametro 4
sigmaRON_LO            = 0.5                                                            # [e-] NB: E'la sigma o la sigma**2: sigma 
ExcessNoiseFactor_LO   = 1.3                                                            #
Dark_LO                = 30                                                             # [e-/s/pix]
skyBackground_LO       = 35                                                             # [e-/s/pix]
ThresholdWCoG_LO       = 0 #
NewValueThrPix_LO      = 0

#nph_LO ?, GuideStarZenith_LO ?