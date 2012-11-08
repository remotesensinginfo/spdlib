import os
import sys
from numpy.distutils.core import setup, Extension


def getGDALFlags():
    """
    Return the flags needed to link in GDAL as a dictionary
    """
    extraargs = {}
    if sys.platform == 'win32':
        # Windows - rely on $GDAL_HOME being set and set 
        # paths appropriately
        gdalhome = os.getenv('GDAL_HOME')
        if gdalhome is None:
            raise SystemExit("need to define $GDAL_HOME")
        extraargs['include_dirs'] = [os.path.join(gdalhome, 'include')]
        #extraargs['library_dirs'] = [os.path.join(gdalhome, 'lib')]
        #extraargs['libraries'] = ['gdal_i']
    else:
        # Unix - can do better with actual flags using gdal-config
        import subprocess
        try:
            cflags = subprocess.check_output(['gdal-config', '--cflags']).strip()
            extraargs['extra_compile_args'] = cflags.split()

            #ldflags = subprocess.check_output(['gdal-config', '--libs']).strip()
            #extraargs['extra_link_args'] = ldflags.split()
        except OSError:
            raise SystemExit("can't find gdal-config")
    return extraargs

# get the flags for GDAL
gdalargs = getGDALFlags()

# create our extension
extkwargs = {'name':"spdpy2", 
                'sources':["spdpy2module.cpp", "pyspdfile.cpp", "recarray.cpp", "pulsearray.cpp", "pointarray.cpp"],
                'library_dirs':[os.path.join('..', 'src')],
                'libraries':['spdio'],
                'include_dirs':[os.path.join("..","include")]}
# add gdalargs
extkwargs.update(gdalargs)

spdpy2module = Extension(**extkwargs)

setup(name="spdpy2",
        version="0.1",
        ext_modules=[spdpy2module],
        description="blah",
        author="Sam Gillingham",
        author_email="gillingham.sam@gmail.com")
