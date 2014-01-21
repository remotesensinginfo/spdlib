from __future__ import print_function
import os
import sys
from numpy.distutils.core import setup, Extension

INCLUDE_OPTIONS = ('--gdalinclude=', '--boostinclude=', '--gslinclude=', 
                '--cgalinclude=', '--lasinclude=', '--hdf5include=')

def getFlags():
    """
    Return the include flags required
    """
    extra_includes = []
    new_argv = [sys.argv[0]]
    for arg in sys.argv[1:]:
        handled = False
        for opt in INCLUDE_OPTIONS:
            if arg.startswith(opt):
                inc = arg.split('=')[1]
                extra_includes.append(inc)
                handled = True
        if arg.startswith('--help'):
            print('Header options:')
            for opt in INCLUDE_OPTIONS:
                print(opt, 'Include path')
        if not handled:
            new_argv.append(arg)

    sys.argv = new_argv
    return extra_includes

# get the flags for GDAL etc
extraincludes = getFlags()

# create our extension
spdpy2module = Extension(name="spdpy2._spdpy2", 
                sources=["src/spdpy2module.cpp", "src/pyspdfile.cpp", "src/recarray.cpp", 
                            "src/pulsearray.cpp", "src/pointarray.cpp"],
                library_dirs=[os.path.join('..', 'src')],
                libraries=['spdio'],
                include_dirs=[os.path.join("..","include")] + extraincludes)

setup(name="spdpy2",
        version="0.1",
        ext_modules=[spdpy2module],
        description="Python Bindings for SPDLib using structred numpy arrays",
        packages=['spdpy2'],
        author="Sam Gillingham",
        author_email="gillingham.sam@gmail.com")
