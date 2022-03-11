from setuptools import setup

NAME = 'mastsel'
DESCRIPTION = ''
URL = 'https://github.com/FabioRossiArcetri/MASTSEL'
EMAIL = 'fabio.rossi@inaf.it'
AUTHOR = 'Fabio Rossi'
LICENSE = 'MIT'


setup(name='mastsel',
      description=DESCRIPTION,
      url=URL,
      author_email=EMAIL,
      author=AUTHOR,
      license=LICENSE,
      version='1.0',
      packages=['mastsel', ],
      install_requires=[
          "numpy",
          "scipy",
          "astropy",
          "matplotlib",
          "cupy",
          "symao",
      ]
      )
