#!/usr/bin/env python
#
#  inSPDoutSPD.py
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
Deletes a subset of the input points
"""

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
outfile = sys.argv[2]

@autojit
def deletePts(spdfile, pulses, points):
    """
    Delete a random subset of the points
    Note: you don't have to update numberOfReturns
    to take into account the deleted points - done by spdapplier
    """
    for npulse in range(pulses.shape[0]):
        if pulses[npulse]['numberOfReturns'] > 0:
            startPoint = pulses[npulse]['startPtsIdx']
            for npoint in range(pulses[npulse]['numberOfReturns']):
                points[startPoint+npoint]['deleteMe'] = numpy.random.randint(0, 10) < 5

spdapplier.apply(deletePts, infile, outputSPDFile=outfile)
