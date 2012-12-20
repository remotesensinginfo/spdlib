#!/usr/bin/env python

"""
Test script for new generation SPDLib Python bindings
"""
from __future__ import print_function
import sys
import spdpy2
import numpy

def countAboveLevel(a):
    """
    Function that returns the number of points that
    have the 'z' about 250
    """
    abovemask = numpy.where(a['z'] > 250, 1, 0)
    return abovemask.sum()

# open the file
spdfile = spdpy2.SPDFile(sys.argv[1])
# read out a block of points 10*10
# result is a 2-d array of objects, each object
# being a structured array of points
pointArray = spdfile.readPointsIntoBlock2d((0,0,10,10))

# create a vectorized function that will go though the
# array and return number of points above z threshold
vfunc = numpy.vectorize(countAboveLevel)

# do the calculations
result = vfunc(pointArray)
print(result)





