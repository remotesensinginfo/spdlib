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

Writes out an image that has the max received waveform for each pulse.
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
outimage = sys.argv[2]

@autojit
def maxReceivedImg(spdfile, pulses, points, imagedata, transmitted, received):
    """
    Finds the maximum received waveform for each bin
    """
    for npulse in range(pulses.shape[0]):
        x = pulses[npulse]['blockX']
        y = pulses[npulse]['blockY']
        if pulses[npulse]['numOfReceivedBins'] > 0:
            startPoint = pulses[npulse]['startReceivedIdx']
            nBins = pulses[npulse]['numOfReceivedBins']
            imagedata[y, x, 0] = received[startPoint:startPoint+nBins].max()
        else:
            imagedata[y, x, 0] = 0
                
spdapplier.apply(maxReceivedImg, infile, outputImageFile=outimage, passWaveforms=True)
