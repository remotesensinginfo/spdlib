#!/usr/bin/env python
#
#  inSPDnoOut.py
#  SPDLIB
#
#  Created by Sam Gillingham on 22/01/2014.
#  Copyright 2013 SPDLib. All rights reserved.
#
#  This file is part of SPDLib.
#
#  SPDLib is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  SPDLib is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with SPDLib.  If not, see <http://www.gnu.org/licenses/>.
#

"""
Test for the Python spdapplier module. 
Mirrors the functionality of the inSPDnoOut program
under examples/BlockProcessor.
"""
from __future__ import print_function
import sys
import numpy
from spdpy2 import spdapplier
try:
    from numba import autojit
except ImportError:
    def autojit(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper

infile = sys.argv[1]

@autojit
def findAverageZ(spdfile, pulses, points, otherin):
    """
    Since this is a bit of a contrived example, the following code works
    well since all the points are now in one array and you don't have to 
    iterate through the pulses:
    otherinputs.ptCount += points.shape[0]
    otherinputs.sum += points['z'].sum()
    
    However, in the spirit of the example, here is some numba code that
    performs the much same thing as the C++
    Note: with numba you can access the fields like pulses[npulse].numberOfReturns
    if you prefer
    """
    for npulse in range(pulses.shape[0]):
        if pulses[npulse]['numberOfReturns'] > 0:
            startPoint = pulses[npulse]['startPtsIdx']
            for npoint in range(pulses[npulse]['numberOfReturns']):
                otherin.arr[0] += points[startPoint+npoint]['z']
                otherin.arr[1] += 1

# inputs to an autojitted need to be arrays...
otherinputs = spdapplier.OtherInputs()
otherinputs.arr = numpy.zeros(2)

spdapplier.apply(findAverageZ, infile, otherinputs=otherinputs)

print('Mean Z', otherinputs.arr[0] / otherinputs.arr[1])
