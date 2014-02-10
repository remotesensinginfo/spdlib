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
Mirrors the functionality of the inSPDinImageoutSPD program
under examples/BlockProcessor.
"""

import sys
import numpy
from numba import autojit
from spdpy2 import spdapplier

infile = sys.argv[1]
inimage = sys.argv[2]
outfile = sys.argv[3]

@autojit
def removeMeanZ(pulses, points, imagedata):
    """
    Removes the image data from the 'z' for each point
    """
    for npulse in range(pulses.shape[0]):
        if pulses[npulse]['numberOfReturns'] > 0:
            x = pulses[npulse]['blockX']
            y = pulses[npulse]['blockY']
            startPoint = pulses[npulse]['startPtsIdx']
            for npoint in range(pulses[npulse]['numberOfReturns']):
                # note: imagedata indexing is: y, x, z
                points[startPoint+npoint]['z'] -= imagedata[y, x, 0]

spdapplier.apply(removeMeanZ, infile, inputImageFile=inimage, outputSPDFile=outfile)
