
Experimental New Python Bindings
================================

These bindings aim to improve performance by using numpy 'structured' arrays.

Numpy must be installed.

To build:

python setup.py build
python setup.py install

I would recommend using the --prefix to the install command and specifying the location
of the SPDLib installed directory and setting your PYTHONPATH environment variable
appropriately.

setup.py uses gdal-config to determine the location of the GDAL/OGR include files.
On Windows, the GDAL_HOME environment variable it used to determine this.

See test.py for a simple example of how to use these bindings.
