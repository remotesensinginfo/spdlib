
Experimental New Python Bindings
================================

These bindings aim to improve performance by using numpy 'structured' arrays.

Numpy must be installed.

To build:

python setup.py build --gdalinclude=$GDAL_INCLUDE_PATH --boostinclude=$BOOST_INCLUDE_PATH \
        --gslinclude=$GSL_INCLUDE_PATH --cgalinclude=$CGAL_INCLUDE_PATH \
        --lasinclude=$LIBLAS_INCLUDE_PATH --hdf5include=$HDF5_INCLUDE_PATH
python setup.py install

Obviously, $GDAL_INCLUDE_PATH must be set properly for your install. If these packages are
installed in default locations then you may be able to get away with just 'python setup.py build'

I would recommend using the --prefix to the install command and specifying the location
of the SPDLib installed directory and setting your PYTHONPATH environment variable
appropriately.

See test.py for a simple example of how to use these bindings.
