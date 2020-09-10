from mavisUtilities import *

#####TELESCOPE#####
TelescopeDiameter = 8.0 #[m]
DM_height = 10000 # [m]
MCAO_FoV = 120 #  (diameter)  [arcsec]
#####ATMOSPHERE#####
WindSpeed = 9.0 # [m/s]
r0_Value = 0.15 # [m]
L0_Value = 25.0 # [m]
Cn2_Value = [0.5, 0.5] # 
heights_Cn2 = [5000,15000, 10000] # [m]
#####SENSOR#####
#Sensor 2x2 subapertures N_sa,tot=4
SensingWavelength = 1650*1e-9 #[m]

SensorFrameRate = 500 # (= loop frequency): [500 Hz]  # , 250 Hz or 100 Hz (to be optimized)
#Corresponding delays (in frames):     [3]       # , 2, 1 
# dove sono usati? per ogni psf hai la FWHM in mas, con questi posso convertire in pixels
pixel_scale = 40 # [mas]
Npix_per_subap = 50 #
WindowRadiusWCoG = 2 # [pixels] calcolo sigma e mu, dimensione della finestra, cerchio di diametro 4
sigmaRON = 0.5 # [e-] NB: E'la sigma o la sigma**2: sigma ????????????????????????????????????????????????
ExcessNoiseFactor = 1.3 # nei calcoli precedentemente mettevamo 1.5, e'ok ?
Dark = 30 # [e-/s/pix]
skyBackground = 35 # [e-/s/pix]
ThresholdWCoG = 0 #
NewValueThrPix = 0

#####OTHERS#####
#PSD wind/vibration: the one for MAVIS (forest of peaks)
#Gains to be explored: linear vector from 0.01 to 0.99
#Directions for total jitter residual estimation (cartesian in arcsec): sono direzioni di puntamento, mappa 3x3
#[0,0], [5,-5], [5,5], [-5,5], [-5,-5], [15,-15], [15,15], [-15,15], [-15,-15]

cartTestPointingCoords = np.asarray([5,5])

#####NGS#####
#NGS coordinates (polar in [arcsec,degrees]): [30,0], [50,100],[10,240]
NGS_flux = [10000, 30000, 5000] # viene usato nel calcolo del rumore, fattore di normalizzazione della gaussian
NGS_SR_1650 = [0.4, 0.2, 0.6]
NGS_FWHM_mas = [90, 110, 85]

testNGSCoords = np.asarray([[30.0,0.0], [50.0,100.0],[10.0,240.0]])

cartTestNGSCoords0 = polarToCartesian(testNGSCoords[0])  
cartTestNGSCoords1 = polarToCartesian(testNGSCoords[1])  
cartTestNGSCoords2 = polarToCartesian(testNGSCoords[2]) 

cartTestNGSCoords = np.asarray([cartTestNGSCoords0, cartTestNGSCoords1, cartTestNGSCoords2])
