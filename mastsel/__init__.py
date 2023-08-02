import numpy as np
import os

gpuEnabled = False
cp = None

systemDisable = os.environ.get('MASTSEL_DISABLE_GPU', 'FALSE')
if systemDisable=='FALSE':
    try:
        import cupy as cp
        print("Cupy import successfull. Installed version is:", cp.__version__)
        gpuEnabled = True
    except:
        print("Cupy import failed. MASTSEL will fall back to CPU use.")
        cp = np
else:
    print("env variable MASTSEL_DISABLE_GPU prevents using the GPU.")
    cp = np
    
from mastsel.mavisPsf import *
from mastsel.mavisLO import *
from mastsel.mavisFormulas import *
from mastsel._version import __version__
