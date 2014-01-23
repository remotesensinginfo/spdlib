#!/usr/bin/env python
#
#  test.py
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
Test script for new generation SPDLib Python bindings
"""
from __future__ import print_function
import sys
import spdpy2
import numpy

# open the file
spdfile = spdpy2.SPDFile(sys.argv[1])

pulses, points = spdfile.readBlock((20, 20, 120, 120))

print(pulses.dtype, pulses.size)
print(pulses)
print(points.dtype)
print(points, points.size)
print(pulses[pulses['nPoints'] != 0])





