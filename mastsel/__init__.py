import numpy as np

gpuEnabled = False
cp = None

try:
    import cupy as cp
    print("Cupy import successfull. Installed version is:", cp.__version__)
    gpuEnabled = True
except:
    print("Cupy import failed. TIPTOP will fall back to CPU use.")
    cp = np


from mastsel.mavisPsf import *
from mastsel.mavisLO import *
from mastsel.mavisFormulas import *
